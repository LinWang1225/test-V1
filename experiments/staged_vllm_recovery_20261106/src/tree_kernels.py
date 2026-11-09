"""Frozen-codebook uint8 reference writer and paged GPU attention for vLLM.

The optional nibble path is a separate validation draft; formal reference
experiments use one uint8 per scalar code and do not implement bit planes.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _word(z, T2, T3, T4, LO, HI, side, h, UNIFORM:tl.constexpr):
    if UNIFORM:
        lo=tl.load(LO+side*8+h);hi=tl.load(HI+side*8+h)
        return tl.minimum(15,tl.maximum(0,libdevice.nearbyint((z-lo)/(hi-lo)*15))).to(tl.int32)
    a=tl.load(T2+(side*8+h)*3)
    b=tl.load(T2+(side*8+h)*3+1)
    c=tl.load(T2+(side*8+h)*3+2)
    index=(z>a).to(tl.int32)+(z>b).to(tl.int32)+(z>c).to(tl.int32)
    index=index*2+(z>tl.load(T3+(side*8+h)*4+index)).to(tl.int32)
    return index*2+(z>tl.load(T4+(side*8+h)*8+index)).to(tl.int32)


@tr.jit
def _store(K,V,SM,CACHE,MU,SIGMA,T2,T3,T4,LO,HI,COUNTS,
           S0:tl.constexpr,S1:tl.constexpr,BLOCK:tl.constexpr,UNIFORM:tl.constexpr,PACKED:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1);side=tl.program_id(2)
    slot=tl.load(SM+t);valid=slot>=0
    d=tl.arange(0,64)
    even=2*d;odd=even+1
    base=side*8*128+h*128
    if side==0:
        x0=tl.load(K+t*S0+h*S1+even);x1=tl.load(K+t*S0+h*S1+odd)
    else:
        x0=tl.load(V+t*S0+h*S1+even);x1=tl.load(V+t*S0+h*S1+odd)
    z0=(x0-tl.load(MU+base+even))/tl.load(SIGMA+base+even)
    z1=(x1-tl.load(MU+base+odd))/tl.load(SIGMA+base+odd)
    q0=_word(z0,T2,T3,T4,LO,HI,side,h,UNIFORM)
    q1=_word(z1,T2,T3,T4,LO,HI,side,h,UNIFORM)
    block=slot//BLOCK;offset=slot%BLOCK
    if PACKED:
        dst=((((block*2+side)*BLOCK+offset)*8+h)*64+d)
        tl.store(CACHE+dst,(q0|(q1<<4)).to(tl.uint8),valid)
    else:
        dst=((((block*2+side)*BLOCK+offset)*8+h)*128+even)
        tl.store(CACHE+dst,q0.to(tl.uint8),valid)
        tl.store(CACHE+dst+1,q1.to(tl.uint8),valid)
    if h==0 and side==0:
        tl.atomic_add(COUNTS+1,valid.to(tl.int64))


@tr.jit
def _decode(Q,CACHE,BT,LENS,MU,SIGMA,C4,PART,LSE,COUNTS,
            QS0:tl.constexpr,QS1:tl.constexpr,BTS0:tl.constexpr,
            BLOCK:tl.constexpr,SPLITS:tl.constexpr,SCALE:tl.constexpr,PACKED:tl.constexpr,
            TILE:tl.constexpr=64):
    request=tl.program_id(0);qh=tl.program_id(1);split=tl.program_id(2)
    h=qh//4;n=tl.load(LENS+request)
    span=tl.cdiv(n,128*SPLITS)*128
    first=split*span;end=tl.minimum(first+span,n)
    d=tl.arange(0,128);r=tl.arange(0,TILE)
    q=tl.load(Q+request*QS0+qh*QS1+d)
    mu_k=tl.load(MU+h*128+d);sig_k=tl.load(SIGMA+h*128+d)
    mu_v=tl.load(MU+8*128+h*128+d);sig_v=tl.load(SIGMA+8*128+h*128+d)
    m=tl.full((),-float('inf'),tl.float32);mass=tl.full((),0,tl.float32)
    acc=tl.full((128,),0,tl.float32)
    for start in range(first,end,TILE):
        t=start+r;valid=t<end
        block=tl.load(BT+request*BTS0+t//BLOCK,valid,other=0)
        width:tl.constexpr=64 if PACKED else 128
        column=d//2 if PACKED else d
        base=(((block[:,None]*2)*BLOCK+t[:,None]%BLOCK)*8+h)*width+column[None,:]
        packed_k=tl.load(CACHE+base,valid[:,None],other=0).to(tl.int32)
        packed_v=tl.load(CACHE+base+BLOCK*8*width,valid[:,None],other=0).to(tl.int32)
        if PACKED:
            k_code=(packed_k>>((d[None,:]%2)*4))&15
            v_code=(packed_v>>((d[None,:]%2)*4))&15
        else:
            k_code=packed_k&15;v_code=packed_v&15
        k=tl.load(C4+h*16+k_code)*sig_k[None,:]+mu_k[None,:]
        v=tl.load(C4+8*16+h*16+v_code)*sig_v[None,:]+mu_v[None,:]
        score=tl.sum(k*q[None,:],axis=1)*SCALE
        score=tl.where(valid,score,-float('inf'))
        new_m=tl.maximum(m,tl.max(score,axis=0))
        alpha=tl.exp(m-new_m);p=tl.exp(score-new_m)
        acc=acc*alpha+tl.sum(p[:,None]*v,axis=0)
        mass=mass*alpha+tl.sum(p,axis=0);m=new_m
    offset=((request*32+qh)*SPLITS+split)
    tl.store(PART+offset*128+d,tl.where(mass>0,acc/mass,0))
    tl.store(LSE+offset,tl.where(mass>0,m+tl.log(mass),-float('inf')))
    if qh==0 and split==0:
        tl.atomic_add(COUNTS,(n>0).to(tl.int64))


def store(k,v,slots,cache,book,counts):
    dummy=book['mu']
    _store[(len(slots),8,2)](k,v,slots,cache,book['mu'],book['sigma'],
        book.get('t2',dummy),book.get('t3',dummy),book.get('t4',dummy),
        book.get('lo',dummy),book.get('hi',dummy),counts,k.stride(0),k.stride(1),
        cache.shape[2], 'lo' in book,cache.shape[-1]==64,num_warps=2)


def decode(query,cache,block_table,lengths,book,counts,scale,splits=8):
    n=query.shape[0]
    parts=torch.empty((n,32,splits,128),device=query.device,dtype=torch.float32)
    lse=torch.empty((n,32,splits),device=query.device,dtype=torch.float32)
    _decode[(n,32,splits)](query,cache,block_table,lengths,book['mu'],book['sigma'],
        book['c4'],parts,lse,counts,query.stride(0),query.stride(1),block_table.stride(0),
        cache.shape[2],splits,scale,cache.shape[-1]==64,num_warps=4)
    weights=(lse-lse.logsumexp(-1,keepdim=True)).exp()
    return (parts*weights[...,None]).sum(2)
