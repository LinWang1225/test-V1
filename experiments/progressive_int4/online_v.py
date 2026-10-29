"""Stage 4: causal prefix-softmax V selection on frozen real K4V4 traces."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,traceback,collections
from pathlib import Path
import numpy as np
from run import ROOT,blocks,load_trace,atomic_json,write_csv,sha_file,rel

OUT=ROOT/'results/progressive_int4_online_v_20261023'
PREV=ROOT/'results/progressive_int4_20261023'
FRACTIONS=(0,.25,.5,.75,1)
def status(s,**kw):atomic_json(OUT/'status.json',dict(status=s,pid=os.getpid(),updated=time.time(),**kw))

def prepare(data,h):
    q=data['query'][0,h*4:(h+1)*4].astype(float)
    parts=[]
    for u,k,v in blocks(data,h):
        n=len(v)
        va=v if u is None else ((4*(u['vq'][:n].astype(np.uint8)>>2)+1.5)*u['rv'][:n,None]+u['zv'][:n,None])*u['sv'][None,:]
        for start in range(0,n,32):
            stop=min(start+32,n);score=q@k[:,start:stop]*data['scale']
            mx=score.max(axis=1);e=np.exp(score-mx[:,None]);lse=mx+np.log(e.sum(axis=1))
            risk=0. if u is None else 1.5*float(np.abs(u['rv'][start:stop]).max())*float(np.linalg.norm(u['sv']))
            parts.append(dict(lse=lse,full=(e@v[start:stop])/e.sum(axis=1)[:,None],anchor=(e@va[start:stop])/e.sum(axis=1)[:,None],risk=risk,quant=u is not None,tokens=stop-start))
    return parts

def prefix_scores(parts,order):
    prefix=np.full(4,-np.inf);mass=np.zeros(len(parts));risk=np.zeros(len(parts))
    for i in order:
        prefix=np.logaddexp(prefix,parts[i]['lse'])
        mass[i]=np.exp(parts[i]['lse']-prefix).max()
        risk[i]=mass[i]*parts[i]['risk']
    return mass,risk

def output(parts,mask):
    lse=np.stack([p['lse'] for p in parts]);mx=lse.max(axis=0)
    z=np.exp(lse-mx);values=np.stack([p['full'] if mask[i] or not p['quant'] else p['anchor'] for i,p in enumerate(parts)])
    return (z[:,:,None]*values).sum(axis=0)/z.sum(axis=0)[:,None]

def causal_mask(parts,order,scores,threshold,budget):
    mask=np.zeros(len(parts),bool);remaining=sum(p['tokens'] for p in parts if p['quant']);spent=forced=0
    for i in order:
        p=parts[i]
        if not p['quant']:continue
        cost=p['tokens'];remaining-=cost
        # Decision uses prefix scores, frozen threshold and known remaining page
        # sizes only. Never examines unseen logits or low V residuals.
        must=budget-spent>remaining
        if spent+cost<=budget and (scores[i]>=threshold or must):
            mask[i]=True;spent+=cost;forced+=int(must)
    return mask,spent,forced

def main():
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((PREV/'protocol.json').read_text());manifest=protocol['manifest']
    atomic_json(OUT/'experiment_config.json',dict(stage=4,model='Qwen3-4B native K4V4 G128',source_config=str(PREV/'experiment_config.json'),manifest=manifest,script_sha256=sha_file(__file__),unique_requests=6,calibration_requests=3,heldout_requests=3,heldout_snapshots=36,heldout_GQA_groups=288,chunk_tokens=32,K='always full 4-bit',V='Midpoint high2 anchor plus optional native low2',fractions=FRACTIONS,scan_orders=['forward','reverse','random_seed_20261023'],methods=['fixed','random_3_seeds','offline_global_mass','offline_global_mass_risk','online_prefix_mass','online_prefix_mass_risk'],random_seeds=[20261023,20261024,20261025],calibration='Per cache ordinal/KV head/method quantile(1-fraction) of forward prefix scores, calibration requests only; frozen for heldout and all scan orders',budget='Exact V-low2 byte budget; causal late-fill if required using known remaining tile sizes; report forced fraction and positional distribution',online_info='Current/past full K scores, current Query, original scale metadata, known page sizes; no future logits, global denominator, V residual or actual reconstruction error in decisions',execution='CPU float64 trace replay; no GPU timing or task quality claim',acceptance='Compare error at exact same residual bytes; validate endpoints, mask causality and scan sensitivity; this does not pass online GPU integration or task-quality gates'))
    samples=collections.defaultdict(list);prepared={}
    status('running',phase='calibration')
    for ti,item in enumerate(manifest):
        if item['split']!='calibration':continue
        data,m=load_trace(item['path'])
        for h in range(8):
            parts=prepare(data,h);order=np.arange(len(parts));ms,rs=prefix_scores(parts,order)
            ix=np.array([p['quant'] for p in parts])
            for method,s in (('online_prefix_mass',ms),('online_prefix_mass_risk',rs)):
                samples[(m['cache_ordinal'],h,method)].extend(s[ix].tolist())
        print('calibration',ti,flush=True)
    thresholds={}
    for key,v in samples.items():
        for fraction in FRACTIONS:
            thresholds[(*key,fraction)]=float(np.quantile(v,1-fraction))
    atomic_json(OUT/'frozen_thresholds.json',{'|'.join(map(str,k)):v for k,v in thresholds.items()})
    rows=[];positions=[];validation=[]
    for ti,item in enumerate(manifest):
        if item['split']!='heldout':continue
        data,m=load_trace(item['path']);status('running',phase='heldout',trace=ti,total=72)
        for h in range(8):
            parts=prepare(data,h);n=len(parts);ix=np.array([i for i,p in enumerate(parts) if p['quant']]);total=sum(parts[i]['tokens'] for i in ix)
            # Native quantized pages are whole G128; this makes all selectable
            # tiles 32 tokens. Refuse unequal costs rather than fake equal budgets.
            assert all(parts[i]['tokens']==32 for i in ix)
            full=np.ones(n,bool);ref=output(parts,full);anchor=output(parts,np.zeros(n,bool))
            base=dict(trace=Path(item['path']).stem,case=m['request']['case_id'],dataset=m['request']['dataset'],cache_ordinal=m['cache_ordinal'],decode_call=m['decode_call_index'],kv_head=h)
            lses=np.stack([p['lse'] for p in parts]);mx=lses.max(axis=0);z=np.exp(lses-mx);global_mass=(z/z.sum(axis=0)).max(axis=1);risk=np.array([p['risk'] for p in parts])
            orders=dict(forward=np.arange(n),reverse=np.arange(n)[::-1],random=np.random.default_rng(20261023).permutation(n))
            for scan,order in orders.items():
                ms,rs=prefix_scores(parts,order)
                # Prefix-causality check: changing/removing later tiles must not
                # change the score seen at an earlier tile.
                cut=max(1,n//2);short=[parts[i] for i in order[:cut]]
                sms,srs=prefix_scores(short,np.arange(cut))
                assert np.allclose(ms[order[:cut]],sms,rtol=0,atol=0)
                assert np.allclose(rs[order[:cut]],srs,rtol=0,atol=0)
                for fraction in FRACTIONS:
                    k=int(len(ix)*fraction);budget=k*32
                    candidates=[]
                    for name,ranking in [('fixed',ix),('offline_global_mass',ix[np.argsort(-global_mass[ix],kind='stable')]),('offline_global_mass_risk',ix[np.argsort(-(global_mass*risk)[ix],kind='stable')])]:
                        mask=np.zeros(n,bool);mask[ranking[:k]]=True;candidates.append((name,-1,mask,0))
                    for seed in (20261023,20261024,20261025):
                        mask=np.zeros(n,bool);mask[np.random.default_rng(seed+h).permutation(ix)[:k]]=True;candidates.append(('random',seed,mask,0))
                    for name,scores in [('online_prefix_mass',ms),('online_prefix_mass_risk',rs)]:
                        threshold=thresholds[(m['cache_ordinal'],h,name,fraction)]
                        mask,spent,forced=causal_mask(parts,order,scores,threshold,budget)
                        assert spent==budget
                        candidates.append((name,-1,mask,forced))
                    for name,seed,mask,forced in candidates:
                        assert int(mask.sum())==k
                        out=output(parts,mask)
                        if k==0:assert np.array_equal(out,anchor)
                        if k==len(ix):assert np.array_equal(out,ref)
                        rows.append(dict(base,scan=scan,method=name,seed=seed,requested_fraction=fraction,actual_fraction=k/max(1,len(ix)),V_residual_bytes=budget*128//4,quantized_code_bits=3+k/max(1,len(ix)),attention_rel=rel(out,ref),forced_tiles=forced,selected_tiles=k,eligible_tiles=len(ix)))
                        if name.startswith('online'):
                            for bin_id in range(4):
                                b=ix[(ix*4//n)==bin_id];positions.append(dict(base,scan=scan,method=name,fraction=fraction,position_quartile=bin_id,eligible_tiles=len(b),selected_tiles=int(mask[b].sum())))
            print('heldout',ti,h,flush=True)
        write_csv(OUT/'measurements.csv',rows);write_csv(OUT/'position_distribution.csv',positions)
    write_csv(OUT/'measurements.csv',rows)
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['method'],r['scan'],r['requested_fraction'])].append(r)
    aggregate=[]
    for key,rr in groups.items():
        v=[r['attention_rel'] for r in rr]
        aggregate.append(dict(method=key[0],scan=key[1],fraction=key[2],n=len(rr),mean=float(np.mean(v)),p95=float(np.percentile(v,95)),max=float(np.max(v)),forced_fraction=sum(r['forced_tiles'] for r in rr)/max(1,sum(r['selected_tiles'] for r in rr))))
    write_csv(OUT/'aggregate.csv',aggregate)
    atomic_json(OUT/'validation.json',dict(endpoint_output_exact=True,equal_budget_all_rows=True,prefix_causality_exact=True,heldout_groups=288,rows=len(rows),note='Assertions exercised on real traces, all orders and budgets; CPU replay only'))
    lines=['Stage 4: K4 + causal online V controller','', '6 original requests, 3 calibration / 3 heldout. No task quality or GPU speed conclusion.','Thresholds calibrated only on forward calibration traces. Same thresholds used for forward/reverse/random scans. All methods matched exactly in residual code bytes; forced late-fill reported separately.','Random n includes 3 frozen seeds; aggregate statistics over correlated snapshots/groups are descriptive, not independent-request confidence intervals.','', 'method, scan, V low2 fraction, mean relative L2, p95, forced fraction']
    for r in aggregate:lines.append(f"{r['method']}, {r['scan']}, {r['fraction']}, {r['mean']:.8f}, {r['p95']:.8f}, {r['forced_fraction']:.5f}")
    (OUT/'conclusions.txt').write_text('\n'.join(lines)+'\n')
    status('completed',rows=len(rows),scope='Stage 4 CPU same-input causal selection; quality and fused GPU gates pending')

if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));traceback.print_exc();raise
