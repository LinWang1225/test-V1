"""Exact vectorization of the prior two-refinement prerequisite greedy.

For a nonconcave two-edge chain, its child is taken immediately after its
parent. Giving the child the parent's sort priority reproduces that order.
Stable flattened indices reproduce prior heap (gain, unit, level) ties.
This preserves the greedy policy, not an optimality claim for nonconcave curves.
"""
import torch


def allocate(curves,target=3.):
    shape=curves.shape[:-1];d=curves.reshape(-1,3);n=len(d)
    budget=int(round((target-2)*n))
    if budget==0:return torch.full(shape,2,dtype=torch.long,device=d.device)
    if budget==2*n:return torch.full(shape,4,dtype=torch.long,device=d.device)
    gains=d[:,:2]-d[:,1:]
    priority=torch.stack((gains[:,0],torch.minimum(gains[:,0],gains[:,1])),-1)
    order=torch.argsort(priority.flatten(),descending=True,stable=True)[:budget]
    return (torch.bincount(order//2,minlength=n)+2).reshape(shape)


def keys(query,curves,scale,policy):
    weight=query.double().square().sum(1)*scale**2
    if policy=='query_only3':pred=weight[...,None]*torch.tensor([1.,.25,.0625],dtype=torch.float64,device=query.device)
    elif policy=='query_curve3':pred=weight[...,None]*curves
    else:raise ValueError(policy)
    return allocate(pred)


def values(phat,curves,policy):
    weight=phat.double().square().sum(1)
    if policy=='query_only3':pred=weight[...,None]*torch.tensor([1.,.25,.0625],dtype=torch.float64,device=phat.device)
    elif policy=='query_curve3':pred=weight[...,None]*curves.sum(1)[:,None,:]
    else:raise ValueError(policy)
    return allocate(pred)
