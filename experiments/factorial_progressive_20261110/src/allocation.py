"""Integer multiple-choice allocation using lower-convex-hull prefix jumps.
Risk is TOTAL loss for each unit and each depth. Costs are elements per bit.
This is a feasible greedy approximation, not an exact knapsack optimizer.
"""
import torch
IDEAL=[1.,.25,.0625,.015625,.00390625]
def allocate(risk,cost,budget):
 if risk.shape[-1]!=5:raise ValueError('five depth risks required')
 shape=risk.shape[:-1];d=risk.reshape(-1,5);n=d.shape[0];device=d.device
 costs=torch.broadcast_to(cost,shape).reshape(-1).to(torch.int64)
 levels=torch.arange(5,device=device);current=torch.zeros(n,device=device,dtype=torch.int64)
 slopes=[];edgecost=[];destinations=[]
 for _ in range(4):
  delta=levels[None,:]-current[:,None]
  gain=d.gather(1,current[:,None])-d
  ratio=gain/(delta.clamp_min(1)*costs[:,None])
  ratio=torch.where(delta>0,ratio,torch.full_like(ratio,-float('inf')))
  target=ratio.argmax(-1);priority=ratio.gather(1,target[:,None]).squeeze(1)
  valid=(priority>0)&(target>current)
  jump=torch.where(valid,target-current,torch.zeros_like(current))
  slopes.append(torch.where(valid,priority,torch.full_like(priority,-float('inf'))))
  edgecost.append(jump*costs);destinations.append(torch.where(valid,target,current))
  current=torch.where(valid,target,current)
 priorities=torch.stack(slopes,dim=-1)
 # Floating-point ties preserve predecessor order within a unit.
 priorities=torch.cummin(priorities,dim=-1).values
 edges=torch.stack(edgecost,dim=-1);ends=torch.stack(destinations,dim=-1)
 order=torch.argsort(priorities.reshape(-1),descending=True,stable=True)
 spent=torch.cumsum(edges.reshape(-1)[order],0)
 chosen=(spent<=max(0,int(budget)))&(priorities.reshape(-1)[order]>0)
 targets=torch.where(chosen,ends.reshape(-1)[order],torch.zeros_like(order))
 depth=torch.zeros(n,device=device,dtype=torch.int64)
 depth.scatter_reduce_(0,order//4,targets,reduce='amax',include_self=True)
 # A hull jump can exceed the remaining integer budget. Consider every direct
 # depth that fits, including intermediate non-hull depths, without reading
 # or changing the empirical curve values.
 remaining=(max(0,int(budget))-(depth*costs).sum()).clamp_min(0)
 delta=levels[None,:]-depth[:,None];extra=delta*costs[:,None]
 gain=d.gather(1,depth[:,None])-d
 valid=(delta>0)&(extra<=remaining)&(gain>0)
 ratio=torch.where(valid,gain/extra.clamp_min(1),torch.full_like(gain,-float('inf')))
 target=torch.where(valid,gain,torch.full_like(gain,-float('inf'))).argmax(-1);priority=ratio.gather(1,target[:,None]).squeeze(1)
 extra=extra.gather(1,target[:,None]).squeeze(1).clamp_min(0)
 order=torch.argsort(priority,descending=True,stable=True)
 selected=(torch.cumsum(extra[order],0)<=remaining)&(priority[order]>0)
 depth[order]=torch.where(selected,target[order],depth[order])
 return depth.reshape(shape).to(torch.uint8)
def histogram(depth,cost):
 costs=torch.broadcast_to(cost,depth.shape).reshape(-1).to(torch.int64)
 out=torch.zeros(5,device=depth.device,dtype=torch.int64)
 out.scatter_add_(0,depth.reshape(-1).long(),costs)
 return out
