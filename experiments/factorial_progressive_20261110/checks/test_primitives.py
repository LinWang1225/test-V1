from pathlib import Path
import sys,json,torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from progressive_kernels import encode,decode
from allocation import allocate,histogram
from transforms import normalization_stats
out={};torch.manual_seed(42)
z=torch.randn((8,129,128),device='cuda');cap=256
c=torch.empty((8,cap,128),device='cuda',dtype=torch.uint8)
t2=torch.tensor([-1.,0.,1.],device='cuda',dtype=torch.float16).expand(8,3).contiguous()
t3=torch.tensor([-1.5,-.5,.5,1.5],device='cuda',dtype=torch.float16).expand(8,4).contiguous()
t4=torch.linspace(-1.75,1.75,8,device='cuda',dtype=torch.float16).expand(8,8).contiguous()
encode(z,c,[t2,t3,t4],0)
p2=(z[:,:,:,None]>t2[:,None,None,:]).sum(-1);hh=torch.arange(8,device='cuda')[:,None,None]
p3=2*p2+(z>t3[hh,p2]);ref=2*p3+(z>t4[hh,p3]);assert torch.equal(c[:,:129].long(),ref.long())
lut=torch.zeros((8,31),device='cuda',dtype=torch.float16)
for b in range(1,5):lut[:,(1<<b)-1:(1<<(b+1))-1]=torch.arange(1<<b,device='cuda')
mu=torch.randn((8,128),device='cuda').half();sg=torch.ones_like(mu);dummy=torch.empty((1,1,1),device='cuda',dtype=torch.float16)
for b in range(5):
 for side in [0,1]:
  dep=torch.full((8,2,128) if side==0 else (8,129),b,device='cuda',dtype=torch.uint8)
  codes=torch.empty(0,device='cuda',dtype=torch.uint8).reshape(0,cap,128) if b==0 else c
  actual=decode(codes,lut,dep,mu,sg,dummy,dummy,129,128,0,side)
  expected=mu.float()[:,None,:]+(ref>>(4-b)).float() if b else mu.float()[:,None,:].expand_as(z)
  assert torch.equal(actual,expected),(b,side)
 out[f'prefix_{b}']=True
risk=torch.tensor([1.,.25,.0625,.015625,.00390625],device='cuda').expand(8,2,128,5)
cost=torch.tensor([128,1],device='cuda')[None,:,None]
for budget in [0,1,2,3,4]:
 dep=allocate(risk,cost,budget*8*129*128);hist=histogram(dep,cost)
 bits=(hist*torch.arange(5,device='cuda')).sum().item();assert bits<=budget*8*129*128
 assert dep.min()>=0 and dep.max()<=4
out['all_depths_integer_budget']=True
constant=torch.ones_like(risk);assert allocate(constant,cost,999999).sum()==0
out['unhelpful_budget_unspent']=True
x=torch.arange(135,device='cuda').float()[None,:,None].expand(8,135,128)
for R in ['R0','R1','R2','R3']:
 for side in [0,1]:
  m,s,e,g=normalization_stats(x,mu.float(),sg.float(),R,side)
  assert g==(32 if R=='R3' and side==1 else 128),(R,side,g)
  assert torch.isfinite(m).all() and torch.isfinite(s).all() and (s>0).all()
out['R3_granularity_K128_V32']=True
out['zero_payload_empty_pointer']=True
torch.cuda.synchronize();print(json.dumps(out))
(Path(__file__).parent/'primitives.json').write_text(json.dumps(out))
