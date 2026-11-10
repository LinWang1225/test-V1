import torch

def rotate(values,basis):
    return torch.matmul(values,basis)
def normalization_stats(values,global_mean,global_scale,reconstruction,side):
    # values [head,arrived_token,channel]; padding never contributes to moments.
    h,n,d=values.shape
    group=32 if reconstruction=='R3' and side==1 else 128
    groups=(n+group-1)//group
    padded=torch.nn.functional.pad(values,(0,0,0,groups*group-n)).reshape(h,groups,group,d)
    lengths=(n-torch.arange(groups,device=values.device)*group).clamp(1,group).float()
    mean=padded.sum(2)/lengths[None,:,None]
    second=padded.square().sum(2)/lengths[None,:,None]
    variance=(second-mean.square()).clamp_min(0)
    gmean=global_mean[:,None,:]
    gscale=global_scale[:,None,:]
    if reconstruction=='R0':
        mu=gmean.expand_as(mean);sigma=gscale.expand_as(mean)
    elif reconstruction=='R1':
        rms=(variance+(mean-gmean).square()).clamp_min(0).sqrt()
        alpha=(rms/gscale).clamp_min(1e-4).half().float()
        mu=gmean.expand_as(mean);sigma=gscale*alpha
    elif reconstruction in ('R2','R3'):
        mu=mean.half().float();sigma=variance.sqrt().clamp_min(1e-4).half().float()
    else:raise ValueError(reconstruction)
    if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma<=0).any():raise ValueError('Invalid deployed local normalization')
    energy=variance+(mean-mu).square()
    return mu,sigma,energy,group
