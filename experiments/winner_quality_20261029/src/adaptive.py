import torch
TOL={'adaptive005':.05,'adaptive015':.15,'adaptive030':.30}
def choose(pred,policy):
    # Excess predicted distortion relative to full4; no bit-budget constraint.
    excess=(pred-pred[...,2:]).clamp_min(0)
    allowance=TOL[policy]*excess[...,0].mean()
    return torch.where(excess[...,0]<=allowance,2,torch.where(excess[...,1]<=allowance,3,4)).long()
def keys(query,curves,scale,policy):
    return choose(query.double().square().sum(1)[...,None]*scale**2*curves,policy)
def values(phat,curves,policy):
    return choose(phat.double().square().sum(1)[...,None]*curves.sum(1)[:,None,:],policy)
