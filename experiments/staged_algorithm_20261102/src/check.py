"""Meaningful CPU audits before long GPU experiments."""
import heapq
import itertools
import numpy as np
import torch
from common import *
from policy import allocate, from_gains
from prefix_codec import encode, make_lut

def heap_reference(d,cost,budget):
    gain=d[:,:-1]-d[:,1:]
    heap=[(-gain[i,0]/cost[i],i,0) for i in range(len(d))]
    heapq.heapify(heap);depth=np.zeros(len(d),int);used=0
    while heap:
        neg,i,edge=heapq.heappop(heap)
        if neg>=0 or used+cost[i]>budget:
            break
        used+=cost[i];depth[i]+=1
        if edge<3:
            heapq.heappush(heap,(-gain[i,edge+1]/cost[i],i,edge+1))
    return depth

def main():
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    cases=0
    for trial in range(80):
        gains=rng.integers(-2,20,(5,4)).astype(float)
        d=np.c_[np.flip(np.cumsum(np.flip(gains,1),1),1),np.zeros(5)]
        costs=rng.integers(1,9,5)
        for budget in (0,8,30,60,120):
            actual,diag=allocate(torch.tensor(d),torch.tensor(costs),budget_bits=budget)
            expected=heap_reference(d,costs,budget)
            assert np.array_equal(actual.numpy(),expected),(trial,budget,actual,expected)
            assert (actual.numpy()*costs).sum()<=budget
            assert diag['logical_bits']==int((actual.numpy()*costs).sum())
            cases+=1
    # For concave equal-cost objectives, compare with exhaustive global optimum.
    for trial in range(12):
        gains=np.sort(rng.integers(1,20,(3,4)),axis=1)[:,::-1].copy()
        d=np.c_[np.flip(np.cumsum(np.flip(gains,1),1),1),np.zeros(3)]
        for budget in range(13):
            actual,_=allocate(torch.tensor(d),torch.ones(3),budget_bits=budget)
            achieved=sum(d[i,b] for i,b in enumerate(actual.tolist()))
            optimum=min(sum(d[i,b] for i,b in enumerate(levels)) for levels in itertools.product(range(5),repeat=3) if sum(levels)<=budget)
            assert achieved==optimum
            cases+=1
    # Gains reconstruct relative curves exactly, even with nonmonotone steps.
    d=torch.randn(7,5,dtype=torch.float64)
    assert torch.allclose(from_gains(d[:,:-1]-d[:,1:],d[:,-1]),d,atol=1e-14)
    pred=torch.tensor([[10.,7.,4.,2.,1.],[6.,4.,3.,1.,0.]],dtype=torch.float64)
    for tol in (0.,1.,3.,20.):
        levels,_=allocate(pred,torch.ones(2),tolerance=tol)
        residual=sum(float(pred[i,b]-pred[i,4]) for i,b in enumerate(levels.tolist()))
        assert residual<=tol+1e-10
    # Decode is a function only of prefix; lower suffix bits cannot change it.
    for codec in CODECS:
        with np.load(ROOT/'codebooks'/codec/'layer0.npz') as a:
            book={k:torch.tensor(a[k][0].astype(np.float32)) for k in a.files if k not in ('D','J')}
        lut=make_lut(book)
        for b in range(5):
            for word in range(16):
                peer=(word>>(4-b))<<(4-b) if b else 0
                assert torch.equal(lut[:,b,word],lut[:,b,peer])
        assert torch.equal(lut[:,0],torch.zeros_like(lut[:,0]))
        with np.load(ROOT/'codebooks'/codec/'layer0.npz') as a:
            if codec!='U_Clipped':
                for b in (2,3,4):
                    assert np.array_equal(lut[:,b].numpy(),a[f'c{b}'][0].astype(np.float32)[:,np.arange(16)>>(4-b)])
    m=json.loads((ROOT/'inputs/manifest.json').read_text())
    sample=list(rows(ROOT/'inputs/cases.jsonl'))
    assert m['total_cases']==len(sample)==464 and m['calibration_overlap']==0
    assert m['datasets']['humaneval']==164 and m['datasets']['math500']==100
    assert sum(v for k,v in m['datasets'].items() if k not in ('humaneval','math500'))==200
    assert sha(ROOT/'inputs/cases.jsonl')==m['cases_sha256']
    assert all(r['input_ids_sha256']==digest(r['input_ids']) and not r['input_truncated'] for r in sample)
    atomic(ROOT/'checks_cpu.json',dict(passed=True,allocation_cases=cases,prefix_levels=list(range(5)),
        nonconcave_heap_equivalence=True,concave_exhaustive_optimum=True,tolerance_feasible=True,
        frozen_464=True,source_hashes={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}))
    print('CPU audits passed',cases,flush=True)

if __name__=='__main__':
    main()
