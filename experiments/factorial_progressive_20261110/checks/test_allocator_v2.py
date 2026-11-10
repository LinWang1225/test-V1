from pathlib import Path
import sys,json,itertools,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from allocation import allocate,histogram
out={};torch.manual_seed(123)
for device in ['cpu','cuda']:
 for depth in range(5):
  d=torch.tensor([[1.,.95,.8,.4,.01]],device=device);b=allocate(d,torch.tensor(7,device=device),7*depth)
  assert b.item()==depth,(device,depth,b)
 assert allocate(torch.tensor([[1.,1.2,.3,.1,.05]],device=device),torch.tensor(1,device=device),3).item()==3
 assert allocate(torch.tensor([[1.,.25,.0625,.015625,.00390625]],device=device).repeat(2,1)*torch.tensor([128,26],device=device)[:,None],torch.tensor([128,26],device=device),462).tolist()==[3,3]
 assert allocate(torch.ones(3,5,device=device),torch.ones(3,device=device),12).sum()==0
 assert allocate(torch.tensor([[1.,1.2,1.3,1.4,1.5]],device=device),torch.tensor(1,device=device),4).item()==0
 for repeat in range(100):
  d=torch.rand(1,5,device=device);cost=torch.tensor(3,device=device);budget=repeat%5
  b=allocate(d,cost,3*budget).long();assert float(d[0,b])<=float(d[0,:budget+1].min())+1e-6
 for repeat in range(100):
  d=torch.rand(8,2,128,5,device=device);cost=torch.tensor([128,26],device=device)[None,:,None];cap=(repeat%5)*8*154*128
  b=allocate(d,cost,cap).long();assert (b<=4).all() and (b>=0).all()
  assert int((b*cost).sum())<=cap
  selected=d.gather(-1,b[:,:,:,None]).squeeze(-1)
  assert (selected<=d[:,:,:,0]+1e-6).all(), (device,repeat,float((selected-d[:,:,:,0]).max()))
 out[device]={'all_depths_reachable':True,'nonmonotonic_jump':True,'unequal_unit_total_risk':True,'no_unhelpful_spending':True,'single_unit_exact_100':True,'random_integer_budget_100':True}
print(json.dumps(out));(Path(__file__).parent/'allocator_v2_tests.json').write_text(json.dumps(out))
