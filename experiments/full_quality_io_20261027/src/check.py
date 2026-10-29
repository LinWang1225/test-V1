"""Meaningful algorithm and natural-stop gates before a full quality run."""
import argparse,heapq,sys
from common import *
import numpy as np
import torch
from policy_gpu import allocate


def heap_reference(d,budget):
    g=d[:,:2]-d[:,1:];result=np.full(len(d),2,np.int64)
    heap=[(-float(g[i,0]),i,0) for i in range(len(d))];heapq.heapify(heap)
    for _ in range(budget):
        _,i,j=heapq.heappop(heap);result[i]+=1
        if not j:heapq.heappush(heap,(-float(g[i,1]),i,1))
    return result


def policy_checks():
    rng=np.random.default_rng(20261027);count=0
    for n in (2,4,8,32,128):
        for _ in range(12):
            # Integer ties, negative gains and nonconcave chains are deliberate.
            d=rng.integers(-3,10,size=(n,3)).astype(np.float64)
            tensor=torch.tensor(d,device='cuda')
            for budget in sorted(set([0,1,n//2,n,2*n-1,2*n])):
                expected=heap_reference(d,budget)
                actual=allocate(tensor,2+budget/n).cpu().numpy()
                assert np.array_equal(actual,expected),(d,budget,actual,expected)
                count+=1
    # Actual frozen calibration curves, not only random curves.
    for layer in range(36):
        with np.load(BOOKS/f'layer{layer}_seed20261024.npz') as z:
            for d in (z['controller_D'][0].reshape(-1,3),z['controller_D'][1].sum(1)):
                weight=rng.random(len(d));weighted=d*weight[:,None]
                assert np.array_equal(allocate(torch.tensor(weighted,device='cuda')).cpu().numpy(),heap_reference(weighted,len(d)))
                count+=1
    atomic(ROOT/'policy_checks.json',dict(passed=True,exact_heap_comparisons=count,negative_and_nonconcave_and_tied_cases=True,actual_calibration_curves=True))
    print('GPU policy matches prerequisite heap',count,flush=True)


def model_checks():
    from generate import QualityModel
    from types import SimpleNamespace
    # Validate incremental code caching and selected-depth decode against the
    # previous full reconstruction formulas on real calibration activations.
    m=QualityModel('query_curve3');rng=np.random.default_rng(20261027)
    boundary_ties=0;checked_codes=0
    for layer in (0,17,35):
        from codec import encode as tree_encode,decode as tree_decode
        raw=torch.tensor(rng.normal(size=(2,8,257,128)).astype(np.float16),device='cuda')
        m.reset(257)
        m.append_codes(layer,raw[0,:,:127][None],raw[1,:,:127][None])
        m.append_codes(layer,raw[0][None],raw[1][None])
        book=m.books[layer]
        for side in (0,1):
            rotated=raw[side].float()@m.U
            standardized=(rotated-book['mu'][side,:,None,:])/book['sigma'][side,:,None,:]
            b={n:book[n][side] for n in ('t2','t3','t4','c2','c3','c4')}
            code=tree_encode(standardized,b)
            cached=m.codes[layer][side,:,:257].long()
            different=code!=cached;changed=int(different.sum());boundary_ties+=changed;checked_codes+=code.numel()
            if changed:
                # cuBLAS may choose a different FP32 reduction kernel for a
                # short chunk vs full cache. Only threshold-adjacent ties may
                # change a code; unrelated differences remain a hard failure.
                thresholds=torch.cat([b[n] for n in ('t2','t3','t4')],-1)
                margin=(standardized[...,None]-thresholds[:,None,None,:]).abs().amin(-1)
                tolerance=1e-5*standardized.abs().clamp_min(1.)
                assert bool((margin[different]<=tolerance[different]).all()),'non-boundary incremental code mismatch'
                assert changed/code.numel()<1e-4,'excessive FP32 boundary sensitivity'
            for bits in (2,3,4):
                expected=tree_decode(cached,b,bits)*book['sigma'][side,:,None,:]+book['mu'][side,:,None,:]
                assert torch.equal(m.decode(layer,257,side,bits),expected),'lookup decode differs'
    # Compare an actual old 64-step online request's first eight tokens under
    # all policies that existed in the previous pilot.
    old_manifest=list(json.loads((ROOT.parent/'encoding_ablation_20261024/inputs/manifest.json').read_text()))
    old=[r for r in old_manifest if r['split']=='test'][0]
    row=dict(old,index=0,dataset=old['dataset'],input_tokens=len(old['input_ids']),original_input_tokens=len(old['input_ids']),input_truncated=False,max_new_tokens=8,input_ids_sha256='numeric_gate')
    comparisons=[]
    for policy in ('fp16','full4','uniform3','query_curve3'):
        m.policy=policy
        result=m.generate(row)
        previous=json.loads((ROOT.parent/'controller_jacobian_20261026/results/online'/(old['case_id'].replace('/','_')+'_'+policy+'.json')).read_text())
        assert result['generated_ids']==previous['generated_ids'][:len(result['generated_ids'])],(policy,result['generated_ids'],previous['generated_ids'][:8])
        comparisons.append(policy)
    # Deterministic stub model proves natural EOS is respected without a
    # minimum-length override and length-limited outputs remain distinguishable.
    real=m.model
    class Stub:
        def __init__(self,token):self.token=token;self.calls=0
        def __call__(self,**kwargs):
            self.calls+=1
            logits=torch.full((1,1,151646),-100.,device='cuda');logits[0,0,self.token]=100.
            return SimpleNamespace(logits=logits,past_key_values=None)
    stop=json.loads((ROOT/'inputs/manifest.json').read_text())['eos_token_ids'][0]
    tiny=dict(row,input_ids=[42],max_new_tokens=8)
    m.policy='fp16';m.model=Stub(stop)
    output=m.generate(tiny);assert output['finish_reason']=='eos' and output['generated_tokens']==1 and m.model.calls==1
    m.model=Stub(17);output=m.generate(tiny,5);assert output['finish_reason']=='length' and output['generated_tokens']==5 and m.model.calls==5
    m.model=real
    atomic(ROOT/'model_checks.json',dict(passed=True,incremental_cache_numerically_equivalent=True,fp32_threshold_boundary_ties=boundary_ties,checked_codes=checked_codes,boundary_tolerance='1e-5 * max(1, abs(standardized_value)); <1e-4 changed fraction',depth_decode_bitwise_equal=True,previous_online_8_token_checks=comparisons,natural_EOS_stops_at_first_token=True,length_limit_separately_recorded=True))
    print('Model/cache/natural-EOS checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    policy_checks()
    if a.model:model_checks()
