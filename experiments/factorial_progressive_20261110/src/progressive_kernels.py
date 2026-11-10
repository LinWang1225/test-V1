import torch,triton,triton.language as tl
@triton.jit
def _encode(Z,C,T2,T3,T4,N,L,CAP:tl.constexpr,START,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N;h=i//(L*128);t=(i//128)%L;c=i%128
 z=tl.load(Z+i,mask,other=0)
 p=(z>tl.load(T2+h*3)).to(tl.int32)+(z>tl.load(T2+h*3+1)).to(tl.int32)+(z>tl.load(T2+h*3+2)).to(tl.int32)
 p=p*2+(z>tl.load(T3+h*4+p)).to(tl.int32)
 p=p*2+(z>tl.load(T4+h*8+p)).to(tl.int32)
 tl.store(C+h*CAP*128+(START+t)*128+c,p,mask)
@triton.jit
def _decode(C,LUT,DEPTH,MU,SG,LM,LS,OUT,T,P,CAP:tl.constexpr,GROUP:tl.constexpr,NG:tl.constexpr,R:tl.constexpr,SIDE:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<8*T*128
 h=i//(T*128);t=(i//128)%T;c=i%128
 if SIDE==0: di=h*P*128+(t//128)*128+c
 else: di=h*T+t
 b=tl.load(DEPTH+di,valid,other=0).to(tl.int32)
 word=tl.load(C+h*CAP*128+t*128+c,valid&(b>0),other=0).to(tl.int32)
 idx=(1<<b)-1+(word>>(4-b))
 z=tl.load(LUT+h*31+idx,valid,other=0).to(tl.float32)
 mu=tl.load(MU+h*128+c,valid,other=0).to(tl.float32);sg=tl.load(SG+h*128+c,valid,other=1).to(tl.float32)
 if R==1:
  sg=sg*tl.load(LS+(h*NG+t//GROUP)*128+c,valid,other=1).to(tl.float32)
 elif R>=2:
  mu=tl.load(LM+(h*NG+t//GROUP)*128+c,valid,other=0).to(tl.float32)
  sg=tl.load(LS+(h*NG+t//GROUP)*128+c,valid,other=1).to(tl.float32)
 tl.store(OUT+i,mu+sg*z,valid)
def encode(z,codes,thresholds,start):
 z=z.contiguous();n=z.numel();l=z.shape[1]
 _encode[(triton.cdiv(n,256),)](z,codes,*thresholds,n,l,codes.shape[1],start,256)
def decode(codes,lut,depth,mu,sigma,local_mu,local_scale,t,group,r,side):
 out=torch.empty((8,t,128),device=mu.device,dtype=torch.float32)
 _decode[(triton.cdiv(out.numel(),256),)](codes,lut,depth.contiguous(),mu,sigma,local_mu,local_scale,out,t,triton.cdiv(t,128),codes.shape[1],group,local_scale.shape[1],r,side,256)
 return out
