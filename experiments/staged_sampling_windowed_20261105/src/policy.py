"""Causal decisions receive only current Q/phat and saved metadata.

Five-level prerequisite greedy is an algorithm baseline, not a general optimum.
Costs count logical bits. Prefix-minimum priority ensures parent-before-child.
"""
import torch

def from_gains(gains, anchor=None):
    tail = torch.zeros_like(gains[..., :1]) if anchor is None else anchor[..., None]
    return torch.cat((torch.flip(torch.cumsum(torch.flip(gains, [-1]), -1), [-1])+tail, tail), -1)

def allocate(distortion, costs, *, budget_bits=None, tolerance=None):
    if (budget_bits is None) == (tolerance is None):
        raise ValueError('Specify exactly one resource budget or error tolerance')
    shape = distortion.shape[:-1]
    d = distortion.reshape(-1,5).double()
    cost = torch.broadcast_to(costs, shape).reshape(-1).double()
    if not torch.isfinite(d).all() or not torch.isfinite(cost).all() or (cost<=0).any():
        raise ValueError('Invalid distortion or logical cost')
    gains = d[:,:-1]-d[:,1:]
    priority = torch.cummin(gains / cost[:,None], dim=1).values
    order = torch.argsort(priority.flatten(), descending=True, stable=True)
    edge_cost = cost.repeat_interleave(4)[order]
    if budget_bits is not None:
        # Prefix selection respects prerequisites, including a short final block.
        keep = int(((edge_cost.cumsum(0) <= float(budget_bits)+1e-8) &
                    (priority.flatten()[order] > 0)).sum().item())
    else:
        target = float(tolerance)
        if target < 0:
            raise ValueError('Negative tolerance')
        residual = torch.cat(((d[:,0]-d[:,4]).sum()[None],
            (d[:,0]-d[:,4]).sum()-gains.flatten()[order].cumsum(0)))
        feasible = torch.nonzero(residual <= target+1e-12).flatten()
        keep = int(feasible[0].item()) if len(feasible) else len(order)
    selected = order[:keep]
    depths = torch.bincount(selected//4, minlength=len(d)).reshape(shape)
    return depths, dict(logical_bits=int(edge_cost[:keep].sum().item()),
        budget_bits=None if budget_bits is None else int(budget_bits),
        negative_edges=int((gains<0).sum().item()),
        increasing_gain_pairs=int((gains[:,1:] > gains[:,:-1]+1e-12).sum().item()))

def curve_view(curves, mode):
    d = curves['J'] if mode in ('jacobian','scale_jacobian') else curves['D']
    if mode == 'query_only':
        return torch.tensor([1., .25, .0625, .015625, .00390625], device=d.device).expand_as(d)
    if mode in ('shared','head'):
        # Preserve D0 amplitude; only remove channel-specific curve shape.
        shape = d / d[...,:1].clamp_min(1e-30)
        axes = (1,2) if mode=='shared' else (2,)
        return shape.mean(axes, keepdim=True)*d[...,:1]
    if mode in ('delta_fp32','delta_fp16'):
        dtype = torch.float16 if mode.endswith('fp16') else torch.float32
        gain = (d[...,:-1]-d[...,1:]).to(dtype)
        if not torch.isfinite(gain).all():
            raise ValueError('Marginal metadata overflow; use a calibrated scaled format')
        return from_gains(gain.double(), d[...,-1])
    return d
