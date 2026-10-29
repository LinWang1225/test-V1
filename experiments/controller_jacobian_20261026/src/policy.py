"""Pure causal policy: Q + frozen K curves; then phat + frozen V curves.

This module has no activation-cache, evaluator, dataset or model access.
Costs are constant *within* a stage, including byte-padded channel planes.
Prerequisite-constrained greedy is audited, not claimed globally optimal.
"""
import heapq
import numpy as np

TARGETS=(2.,2.5,3.,3.5,4.)
POLICIES=('uniform','static','query_only','query_curve')


def allocate(curves, target):
    d=np.asarray(curves,np.float64).reshape(-1,3)
    n=len(d);budget=int(np.floor((target-2)*n+1e-9))
    if not 0<=budget<=2*n:
        raise ValueError('target outside [2,4]')
    if not np.isfinite(d).all():
        raise ValueError('nonfinite calibrated curve')
    if budget==0:return np.full(curves.shape[:-1],2,np.int8)
    if budget==2*n:return np.full(curves.shape[:-1],4,np.int8)
    gains=d[:,:2]-d[:,1:]
    if np.all(gains[:,0]>=gains[:,1]):
        # Stable ties expose parent edge before child; prerequisite is respected.
        order=np.lexsort((np.arange(2*n),-gains.ravel()))[:budget]
        result=np.bincount(order//2,minlength=n)+2
    else:
        heap=[(-float(gains[i,0]),i,0) for i in range(n)];heapq.heapify(heap)
        result=np.full(n,2,np.int8)
        for _ in range(budget):
            _,i,level=heapq.heappop(heap);result[i]+=1
            if level==0:heapq.heappush(heap,(-float(gains[i,1]),i,1))
    assert int((result-2).sum())==budget
    return result.reshape(curves.shape[:-1]).astype(np.int8)


def uniform_depths(shape,target):
    n=int(np.prod(shape));budget=int(np.floor((target-2)*n+1e-9))
    # Spread fractional-bit refinements evenly rather than favoring early tokens.
    base,remainder=divmod(budget,n)
    result=np.full(n,2+base,np.int8)
    if remainder:
        ix=np.floor(np.arange(remainder)*n/remainder).astype(int)
        result[ix]+=1
    return result.reshape(shape)


def key_decision(query, calibrated_D, scale, target, policy):
    """query [head, sibling, channel], D [head, channel, depth]."""
    d=np.asarray(calibrated_D,np.float64)
    weight=np.square(np.asarray(query,np.float64)).sum(1)*scale**2
    if policy=='uniform':return uniform_depths(d.shape[:-1],target)
    if policy=='static':pred=d
    elif policy=='query_only':
        # Same 4^-b shape for every channel, calibrated variance scale removed.
        pred=weight[...,None]*np.array([1.,.25,.0625])
    elif policy=='query_curve':pred=weight[...,None]*d
    else:raise ValueError(policy)
    return allocate(pred,target)


def value_decision(phat, calibrated_D, target, policy):
    """Only reconstructed attention is available here; no true V or P."""
    p=np.asarray(phat,np.float64)
    weight=np.square(p).sum(1)
    d=np.asarray(calibrated_D,np.float64).sum(1) # sum original-domain channels
    if policy=='uniform':return uniform_depths(weight.shape,target)
    if policy=='static':pred=np.broadcast_to(d[:,None,:],(*weight.shape,3))
    elif policy=='query_only':pred=weight[...,None]*np.array([1.,.25,.0625])
    elif policy=='query_curve':pred=weight[...,None]*d[:,None,:]
    else:raise ValueError(policy)
    return allocate(pred,target)


def predicted_error(query,phat,kd,vd,curves,scale):
    k=np.asarray(curves[0],np.float64)
    v=np.asarray(curves[1],np.float64).sum(1)
    k_chosen=np.take_along_axis(k,(kd-2)[...,None],axis=-1)[...,0]
    vk=np.broadcast_to(v[:,None,:],(*vd.shape,3))
    v_chosen=np.take_along_axis(vk,(vd-2)[...,None],axis=-1)[...,0]
    # K proxy: mean over cached tokens and GQA sibling queries.
    pk=(np.square(query).sum(1)*scale**2*k_chosen).sum(1)/query.shape[1]
    pv=(np.square(phat).sum(1)*v_chosen).sum(1)/(query.shape[1]*query.shape[2])
    return pk,pv
