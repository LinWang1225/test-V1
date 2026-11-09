"""Paged reader for packed K=(3-bit MSE + 1-bit QJL), V=4-bit MSE.

QJL residual inner products are evaluated directly; the full key residual
vector is never reconstructed in a global temporary cache.
"""
import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _attention(Q, QJLQ, CACHE, NORMS, BT, LENS, C3, C4, PART, LSE,
               QS0:tl.constexpr, QS1:tl.constexpr, BTS0:tl.constexpr,
               BLOCK:tl.constexpr, SPLITS:tl.constexpr, SCALE:tl.constexpr,
               TILE:tl.constexpr=16):
    request=tl.program_id(0);qh=tl.program_id(1);split=tl.program_id(2)
    h=qh//4;n=tl.load(LENS+request)
    span=tl.cdiv(n,128*SPLITS)*128
    first=split*span;end=tl.minimum(first+span,n)
    d=tl.arange(0,128);r=tl.arange(0,TILE)
    q=tl.load(Q+request*QS0+qh*QS1+d)
    qjlq=tl.load(QJLQ+request*QS0+qh*QS1+d)
    m=tl.full((),-float('inf'),tl.float32)
    mass=tl.full((),0,tl.float32);acc=tl.full((128,),0,tl.float32)
    for start in range(first,end,TILE):
        t=start+r;valid=t<end
        block=tl.load(BT+request*BTS0+t//BLOCK,valid,other=0)
        offset=t%BLOCK
        base=(((block[:,None]*2)*BLOCK+offset[:,None])*8+h)*64+d[None,:]//2
        word=tl.load(CACHE+base,valid[:,None],other=0).to(tl.int32)
        code=(word>>((d[None,:]%2)*4))&15
        normbase=(block*3*BLOCK+offset)*8+h
        kn=tl.load(NORMS+normbase,valid,other=0)
        gamma=tl.load(NORMS+normbase+BLOCK*8,valid,other=0)
        kmse=tl.load(C3+(code&7))
        signs=(code>>3).to(tl.float32)*2-1
        score=kn*(tl.sum(kmse*q[None,:],1)+gamma*(1.2533141373155001/128)*tl.sum(signs*qjlq[None,:],1))*SCALE
        score=tl.where(valid,score,-float('inf'))
        new_m=tl.maximum(m,tl.max(score,0))
        alpha=tl.exp(m-new_m);p=tl.exp(score-new_m)
        vword=tl.load(CACHE+base+BLOCK*8*64,valid[:,None],other=0).to(tl.int32)
        vcode=(vword>>((d[None,:]%2)*4))&15
        vn=tl.load(NORMS+normbase+2*BLOCK*8,valid,other=0)
        v=tl.load(C4+vcode)*vn[:,None]
        acc=acc*alpha+tl.sum(p[:,None]*v,0)
        mass=mass*alpha+tl.sum(p,0);m=new_m
    index=(request*32+qh)*SPLITS+split
    tl.store(PART+index*128+d,tl.where(mass>0,acc/mass,0))
    tl.store(LSE+index,tl.where(mass>0,m+tl.log(mass),-float('inf')))


def attention(query, projected_query, cache, norms, block_table, lengths, centers3, centers4, scale):
    n=query.shape[0];splits=8
    parts=torch.empty((n,32,splits,128),device=query.device,dtype=torch.float32)
    lse=torch.empty((n,32,splits),device=query.device,dtype=torch.float32)
    _attention[(n,32,splits)](query,projected_query,cache,norms,block_table,lengths,
        centers3,centers4,parts,lse,query.stride(0),query.stride(1),block_table.stride(0),
        cache.shape[2],splits,scale,num_warps=4)
    weights=(lse-lse.logsumexp(-1,keepdim=True)).exp()
    return (parts*weights[...,None]).sum(2)
