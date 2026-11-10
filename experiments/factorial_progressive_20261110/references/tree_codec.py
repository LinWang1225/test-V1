"""Independent scalar prefix codecs; no KVarN/vLLM imports."""
import numpy as np
import torch
from statistics import NormalDist

def hadamard(d=128,seed=20261024):
    h=np.ones((1,1),np.float32)
    while len(h)<d:h=np.block([[h,h],[h,-h]])
    signs=np.random.default_rng(seed).choice([-1,1],d).astype(np.float32)
    return h*signs[:,None]/np.sqrt(d)

def lloyd(x,w,n,iterations=100):
    if w.sum()<=0:return np.linspace(float(x.min()),float(x.max()),n)
    cw=np.cumsum(w)/w.sum();c=np.interp((np.arange(n)+.5)/n,cw,x)
    old=np.inf
    for _ in range(iterations):
        ix=np.searchsorted((c[:-1]+c[1:])/2,x)
        new=np.array([np.sum(w[ix==j]*x[ix==j])/w[ix==j].sum() if w[ix==j].sum()>0 else c[j] for j in range(n)])
        loss=np.sum(w*(x-new[ix])**2)
        c=np.sort(new)
        if old<np.inf and abs(old-loss)<=1e-6*max(old,1e-30):break
        old=loss
    return c

def fit_tree(x,w,gaussian=False):
    """28 centroids plus 12 branch thresholds; initial 4-way seed for C/D."""
    tables={};thresholds={}
    if gaussian:
        # Dense deterministic midpoint normal quadrature, then recursive binary Lloyd.
        nd=NormalDist();x=np.array([nd.inv_cdf((i+.5)/262144) for i in range(262144)])
        w=np.ones_like(x);groups=[np.arange(len(x))]
        for depth in (1,2,3,4):
            nxt=[];cs=[];ts=[]
            for ix in groups:
                c=lloyd(x[ix],w[ix],2);t=np.float16(c.mean()).item()
                a=ix[x[ix]<=t];b=ix[x[ix]>t]
                for j in (a,b):cs.append(np.sum(w[j]*x[j])/max(w[j].sum(),1e-30))
                ts.append(t);nxt.extend((a,b))
            tables[depth]=np.array(cs,np.float16);thresholds[depth]=np.array(ts,np.float16);groups=nxt
    else:
        c=lloyd(x,w,4);bounds=((c[:-1]+c[1:])/2).astype(np.float16)
        labels=np.searchsorted(bounds,x);groups=[np.where(labels==j)[0] for j in range(4)]
        tables[2]=c.astype(np.float16);thresholds[2]=bounds
        for depth in (3,4):
            nxt=[];cs=[];ts=[]
            for gi,ix in enumerate(groups):
                if len(ix)==0:
                    parent=float(tables[depth-1][gi]);cs.extend([parent,parent]);ts.append(parent);nxt.extend([ix,ix]);continue
                c=lloyd(x[ix],w[ix],2);t=np.float16(c.mean()).item();ts.append(t)
                for j in (ix[x[ix]<=t],ix[x[ix]>t]):
                    cs.append(np.sum(w[j]*x[j])/w[j].sum() if w[j].sum()>0 else float(c.mean()));nxt.append(j)
            tables[depth]=np.array(cs,np.float16);thresholds[depth]=np.array(ts,np.float16);groups=nxt
    # Store only seed boundaries and 2->3/3->4 splits for deployment.
    if gaussian:thresholds[2]=np.array([thresholds[2][0],thresholds[1][0],thresholds[2][1]],np.float16)
    return {**{f'c{b}':tables[b] for b in (2,3,4)},**{f't{b}':thresholds[b] for b in (2,3,4)}}

def encode(z,book):
    # z [head,token,channel], tables [head,level].
    ix=(z[...,None]>book['t2'][:,None,None,:]).sum(-1)
    for b in (3,4):
        t=torch.gather(book[f't{b}'],1,ix.flatten(1)).view_as(ix)
        ix=ix*2+(z>t).long()
    return ix

def decode(code,book,b):
    ix=code>>(4-b)
    return torch.gather(book[f'c{b}'],1,ix.flatten(1)).view_as(code)

def pack4(code):
    a=np.asarray(code,np.uint8).ravel();assert len(a)%2==0
    return a[::2]|(a[1::2]<<4)

def unpack4(a):
    return np.stack((a&15,a>>4),-1).ravel()

def allocation_check(d):
    """Equal cost refinements, 2-bit baseline; exact DP vs greedy."""
    gains=np.stack((d[:,0]-d[:,1],d[:,1]-d[:,2]),1)
    records=[]
    for budget in range(2*len(d)+1):
        dp=np.full(budget+1,-np.inf);dp[0]=0
        for g in gains:
            nxt=np.full_like(dp,-np.inf)
            for n,v in enumerate((0,g[0],g.sum())):
                if n<=budget:nxt[n:]=np.maximum(nxt[n:],dp[:budget+1-n]+v)
            dp=nxt
        levels=np.zeros(len(d),int);greedy=0.
        for _ in range(budget):
            choices=np.array([gains[i,l] if l<2 else -np.inf for i,l in enumerate(levels)])
            i=int(choices.argmax());greedy+=choices[i];levels[i]+=1
        optimum=float(np.max(dp)) # at most budget; negative gains need not be taken
        records.append({'budget':budget,'optimal_gain':optimum,'greedy_gain':float(greedy),'gap':float(max(0.,optimum-greedy))})
    return records


def pack_planes(code,omit_last=0):
    """Actual compact 4-plane stream; optional deterministic Bit0 omission."""
    a=np.asarray(code,np.uint8).ravel()
    return [np.packbits(((a if bit else a[omit_last:])>>bit)&1,bitorder='little') for bit in (3,2,1,0)]

def unpack_planes(planes,n,omit_last=0):
    a=np.zeros(n,np.uint8)
    for bit,p in zip((3,2,1,0),planes):
        start=omit_last if bit==0 else 0
        a[start:]|=np.unpackbits(p,bitorder='little')[:n-start].astype(np.uint8)<<bit
    return a
