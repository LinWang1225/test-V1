import sys
from pathlib import Path
import numpy as np
from policy import *


def exact_optimum(d,budget):
    table=np.full(budget+1,np.inf);table[0]=0
    for row in d:
        nxt=np.full_like(table,np.inf)
        for cost in range(3):
            if cost<=budget:nxt[cost:]=np.minimum(nxt[cost:],table[:len(table)-cost]+row[cost])
        table=nxt
    return float(table[budget])


def main():
    rng=np.random.default_rng(12)
    for _ in range(20):
        gains=np.sort(rng.random((8,2)),axis=1)[:,::-1]
        d=np.c_[gains.sum(1),gains[:,1],np.zeros(8)]
        for b in range(17):
            result=allocate(d,2+b/8)
            assert np.isclose(d[np.arange(8),result-2].sum(),exact_optimum(d,b))
    d=np.array([[100.,100.,0.],[2.,1.,0.]])
    result=allocate(d,3)
    assert d[np.arange(2),result-2].sum()>exact_optimum(d,2)
    q=rng.normal(size=(8,4,128));curves=np.abs(rng.normal(size=(2,8,128,3)))
    p=rng.random((8,4,257));p/=p.sum(-1,keepdims=True)
    for policy in POLICIES:
        for target in TARGETS:
            kd=key_decision(q,curves[0],1/np.sqrt(128),target,policy)
            vd=value_decision(p,curves[1],target,policy)
            assert (kd-2).sum()==int((target-2)*kd.size)
            assert (vd-2).sum()==int((target-2)*vd.size)
            assert np.array_equal(kd,key_decision(q.copy(),curves[0].copy(),1/np.sqrt(128),target,policy))
    # Value decisions depend on the returned K attention; vary it explicitly.
    p2=np.zeros_like(p);p2[:,:,0]=1
    assert not np.array_equal(value_decision(p,curves[1],3,'query_curve'),value_decision(p2,curves[1],3,'query_curve'))
    # Typed call boundaries reject an attempted extra future-query argument.
    try:key_decision(q,curves[0],1.,3.,'query_curve',future_query=q)
    except TypeError:pass
    else:raise AssertionError('future input accepted')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'encoding_extension_20261026/src'))
    from common import atomic
    atomic(Path(__file__).resolve().parents[1]/'correctness_checks.json',dict(passed=True,concave_exact_DP_instances=20*17,nonconcave_counterexample_detected=True,all_budget_checks=True,policy_inputs='Q+frozen curves, then reconstructed attention+frozen curves; no cache or evaluator reference',value_attention_dependency=True,scope='interface and numerical tests, not physical low-bit reads'))
    print('All causal controller gates passed',flush=True)


if __name__=='__main__':main()
