"""Frozen histogram objectives, clipped uniform and nested multi-depth codebooks."""
import sys,time
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'src'))
from codec import lloyd,fit_tree
NODE_EVENTS=[]
def lloyd(x,w,n,iterations=100):
    # Same arithmetic and stop rule as frozen codec.lloyd, with convergence evidence.
    if w.sum()<=0:
        NODE_EVENTS.append(dict(clusters=n,samples=len(x),iterations=0,converged=False,empty=True))
        return np.linspace(float(x.min()),float(x.max()),n)
    cw=np.cumsum(w)/w.sum();c=np.interp((np.arange(n)+.5)/n,cw,x);old=np.inf;history=[];converged=False
    for it in range(iterations):
        ix=np.searchsorted((c[:-1]+c[1:])/2,x)
        new=np.array([np.sum(w[ix==j]*x[ix==j])/w[ix==j].sum() if w[ix==j].sum()>0 else c[j] for j in range(n)])
        loss=np.sum(w*(x-new[ix])**2);history.append(float(loss));c=np.sort(new)
        if old<np.inf and abs(old-loss)<=1e-6*max(old,1e-30):converged=True;break
        old=loss
    NODE_EVENTS.append(dict(clusters=n,samples=len(x),iterations=it+1,converged=converged,loss_history=history))
    return c

def drain_node_events():
    a=list(NODE_EVENTS);NODE_EVENTS.clear();return a

import codec as _legacy_codec
_legacy_codec.lloyd=lloyd
BITS=(2,3,4)
QUANTILES=[0.,.0001,.001,.005,.01,.02]

def uniform_decode(x,lo,hi,b):
    q=np.rint((x-lo)/(hi-lo)*15).clip(0,15).astype(int)
    return lo+((q>>(4-b))*2**(4-b)+(2**(4-b)-1)/2)/15*(hi-lo)

def clipped(x,w):
    cw=np.cumsum(w);cw=cw/max(cw[-1],1e-30)
    best=None
    for l in QUANTILES:
        for r in QUANTILES:
            lo,hi=np.interp([l,1-r],cw,x)
            lo,hi=float(np.float16(lo)),float(np.float16(hi))
            if hi-lo<1e-5:continue
            ds=[np.average((x-uniform_decode(x,lo,hi,b))**2,weights=w) for b in BITS]
            score=np.mean(ds)
            if best is None or score<best[0]:best=(score,lo,hi,l,r,ds)
    return dict(lo=np.float16(best[1]),hi=np.float16(best[2])),dict(objective=best[0],clip_lower_quantile=best[3],clip_upper_quantile=best[4],D=best[5],candidates=36)

def independent(x,w):
    result={}
    for b in BITS:
        c=lloyd(x,w,2**b);result[f'c{b}']=c.astype(np.float16);result[f'f{b}']=((c[:-1]+c[1:])/2).astype(np.float16)
    return result

def boundaries(book):
    t=np.zeros(15,float);t[[3,7,11]]=book['t2'];t[[1,5,9,13]]=book['t3'];t[::2]=book['t4'];return t

def from_cuts(x,w,cuts,old=None):
    labels=np.searchsorted(cuts,np.arange(len(x)),side='right')
    tables={}
    for b in BITS:
        ix=labels>>(4-b);mass=np.bincount(ix,weights=w,minlength=2**b);total=np.bincount(ix,weights=w*x,minlength=2**b)
        fallback=np.linspace(x.min(),x.max(),2**b) if old is None else old[f'c{b}']
        tables[f'c{b}']=np.divide(total,mass,out=np.array(fallback,dtype=float),where=mass>0)
    return tables,labels

def joint(x,w,seed,only4=False):
    if only4:
        c=lloyd(x,w,16);t=(c[:-1]+c[1:])/2
    else:t=boundaries(seed)
    cuts=np.searchsorted(x,t);old=seed;history=[];alpha=[0.,0.,1.] if only4 else [1/3]*3
    for it in range(1 if only4 else 60):
        tab,labels=from_cuts(x,w,cuts,old);old=tab
        loss=sum(a*np.sum(w*(x-tab[f'c{b}'][labels>>(4-b)])**2)/w.sum() for a,b in zip(alpha,BITS));history.append(float(loss))
        if not only4:
            for j in range(15):
                left=cuts[j-1] if j else 0;right=cuts[j+1] if j<14 else len(x)
                xx=x[left:right];ww=w[left:right];delta=np.zeros(len(xx))
                for a,b in zip(alpha,BITS):
                    delta+=a*((xx-tab[f'c{b}'][j>>(4-b)])**2-(xx-tab[f'c{b}'][(j+1)>>(4-b)])**2)
                cuts[j]=left+int(np.argmin(np.r_[0,np.cumsum(ww*delta)]))
        if len(history)>1 and abs(history[-2]-loss)<1e-6*max(history[-2],1e-30):break
    tab,labels=from_cuts(x,w,cuts,old)
    bounds=np.array([(x[0]-1 if c==0 else x[-1]+1 if c==len(x) else (x[c-1]+x[c])/2) for c in cuts],np.float16)
    result={k:v.astype(np.float16) for k,v in tab.items()}
    result.update(t2=bounds[[3,7,11]],t3=bounds[[1,5,9,13]],t4=bounds[::2])
    return result,dict(iterations=len(history),loss_history=history,alpha=alpha)

def distortion(x,w,book):
    labels=np.searchsorted(boundaries(book),x)
    return np.array([np.average((x-book[f'c{b}'][labels>>(4-b)])**2,weights=w) for b in BITS])

def regularize(x,w,book,lam=10.):
    """Calibrated head-level diminishing-gain regularizer; no test-driven gating."""
    ds=distortion(x,w,book);g=ds[:-1]-ds[1:];violation=max(0.,g[1]-g[0],-g[0],-g[1])/max(ds[0],1e-20)
    if violation<.01:return book,dict(triggered=False,normalized_violation=float(violation))
    labels=np.searchsorted(boundaries(book),x);xx=torch.tensor(x,dtype=torch.float64);ww=torch.tensor(w/w.sum(),dtype=torch.float64)
    cs=[torch.tensor(book[f'c{b}'].astype(float),requires_grad=True) for b in BITS]
    ix=[torch.tensor(labels>>(4-b)) for b in BITS];optimizer=torch.optim.Adam(cs,lr=.01)
    scale=max(ds[0],1e-12)
    for _ in range(100):
        optimizer.zero_grad();d=torch.stack([((xx-c[i])**2*ww).sum() for c,i in zip(cs,ix)])
        g23=d[0]-d[1];g34=d[1]-d[2]
        penalty=(torch.relu(g34-g23)/scale)**2+(torch.relu(-g23)/scale)**2+(torch.relu(-g34)/scale)**2
        loss=d.mean()/scale+lam*penalty;loss.backward();optimizer.step()
    out={**book,**{f'c{b}':c.detach().numpy().astype(np.float16) for b,c in zip(BITS,cs)}}
    return out,dict(triggered=True,normalized_violation=float(violation),lambda_regularizer=lam,steps=100,scope='calibrated head-level curve; not per-test-token constraint')

# k/v book names, enabling genuine cross-encoder experiments.
METHODS={
 'U_MinMax':('uniform:minmax','uniform:minmax'),
 'U_Clipped':('uniform:mse','uniform:mse'),
 'Gaussian_Tree':('tree:gaussian','tree:gaussian'),
 'Data_Tree':('tree:mse','tree:mse'),
 'Jacobian_Tree':('tree:jac','tree:attn'),
 'JacK_MSEV':('tree:jac','tree:mse'),
 'ScoreK_MSEV':('tree:score','tree:mse'),
 'ScoreK_AttnV':('tree:score','tree:attn'),
 'ScoreK_MixV':('tree:score','tree:mix'),
 'JacK_MixV':('tree:jac','tree:mix'),
 'Uniform_ScoreK_MSEV':('uniform:score','uniform:mse'),
 'Uniform_ScoreK_AttnV':('uniform:score','uniform:attn'),
 'Uniform_ScoreK_MixV':('uniform:score','uniform:mix'),
 'Independent_MSE':('independent:mse','independent:mse'),
 'Leaf4_Only':('tree:leaf4','tree:leaf4'),
 'Joint_234':('tree:joint','tree:joint'),
 'Joint_Regularized':('tree:regularized','tree:regularized'),
}


def independent_uniform(x,w):
    cw=np.cumsum(w)/w.sum();out={}
    for b in BITS:
        best=None
        for l in QUANTILES:
            for r in QUANTILES:
                lo,hi=np.interp([l,1-r],cw,x);lo=float(np.float16(lo));hi=float(np.float16(hi))
                if hi-lo<1e-5:continue
                q=np.rint((x-lo)/(hi-lo)*(2**b-1)).clip(0,2**b-1)
                loss=np.average((x-(lo+q/(2**b-1)*(hi-lo)))**2,weights=w)
                if best is None or loss<best[0]:best=(loss,lo,hi)
        out[f'lo{b}']=np.float16(best[1]);out[f'hi{b}']=np.float16(best[2])
    return out
METHODS['Independent_Uniform']=('uniform_independent:mse','uniform_independent:mse')
