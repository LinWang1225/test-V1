"""Frozen real-trace experiments, stages 2--6. No task-quality/speed claims."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys, json, hashlib, traceback, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import numpy as np
from kvbench_residual_diag.common import load_trace, atomic_json, write_csv, sha_file
from kvbench_residual.fixtures import unpack_block

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/progressive_int4_20261023'
SOURCE=Path('/home/wanglin/kvbench-3090-cu126/results/residual_diag/run_20261012_161428_3993496/traces')
ROWS=[]
def state(s,**kw):
    atomic_json(OUT/'status.json',dict(status=s,pid=os.getpid(),updated=time.time(),**kw))

def pack2(a):
    a=np.asarray(a,np.uint8).reshape(-1,4)
    return a[:,0]|(a[:,1]<<2)|(a[:,2]<<4)|(a[:,3]<<6)
def un2(a,shape):
    return ((a[:,None]>>np.arange(0,8,2,dtype=np.uint8))&3).reshape(shape)
def reconstruct(c,kind,lut):
    h=(c.astype(np.uint8)>>2)
    return 4*h+({'zero':0.,'mid':1.5}.get(kind,0.)) if kind!='conditional' else lut[h]
def softmax(s):
    e=np.exp(s-s.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))

def blocks(data,h):
    L=data['case'].seq_lens[0]
    for j in range((L+127)//128):
        bid=int(data['block_table'][0,j]);slot=int(data['block_to_slot'][bid]);n=min(128,L-j*128)
        if slot>=0:
            yield None,data['pool_k'][slot,:n,h].astype(float).T,data['pool_v'][slot,:n,h].astype(float)
        else:
            u=unpack_block(data,bid,h)
            yield u,(u['kq'][:,:n]*u['sk'][:,None]+u['zk'][:,None])*u['rk'][None,:n],(u['vq'][:n]*u['rv'][:n,None]+u['zv'][:n,None])*u['sv'][None,:]

def main():
    OUT.mkdir(exist_ok=True)
    paths=[p for p in sorted(SOURCE.glob('trace_*.json')) if json.loads(p.read_text()).get('schema')]
    # Request-level split, frozen without looking at errors. First request of
    # each task calibrates; second is held out. All layers/decode positions retained.
    cases={}
    for p in paths:
        m=json.loads(p.read_text());r=m['request'];cases.setdefault(r['dataset'],[])
        if r['case_id'] not in cases[r['dataset']]:cases[r['dataset']].append(r['case_id'])
    calibration={v[0] for v in cases.values()}
    manifest=[];local=OUT/'inputs';local.mkdir(exist_ok=True)
    import shutil
    for p in paths:
        m=json.loads(p.read_text());dest=local/p.name;arr=local/m['arrays_file']
        if not arr.exists():shutil.copy2(p.parent/m['arrays_file'],arr)
        shutil.copy2(p,dest)
        manifest.append(dict(path=str(dest),metadata_sha256=sha_file(dest),arrays_sha256=m['arrays_sha256'],case=m['request']['case_id'],split='calibration' if m['request']['case_id'] in calibration else 'heldout'))
    atomic_json(OUT/'protocol.json',dict(source=str(SOURCE),script_sha256=sha_file(__file__),seed=20261023,chunk_tokens=32,fractions=[0,.25,.5,.75,1],manifest=manifest,scope='Same-input counterfactual attention; full INT4 retained; no quality, HBM or speed inference',conditional='Separate K/V code conditional means pooled over calibration requests only',thresholds='No thresholds selected using heldout; report all fixed budgets',later_stages='8--11 require separate GPU integration and task-quality acceptance; not automatically passed'))
    counts=np.zeros((2,4));sums=counts.copy();exact=total=0
    state('running',stage=2,phase='calibration and bit-exact roundtrip')
    for i,item in enumerate(manifest):
        data,meta=load_trace(item['path'])
        for h in range(8):
            for u,_,_ in blocks(data,h):
                if u is None:continue
                for side,key in enumerate(('kq','vq')):
                    c=u[key].astype(np.uint8);hi=pack2(c>>2);lo=pack2(c&3)
                    restored=(un2(hi,c.shape)<<2)|un2(lo,c.shape)
                    exact+=int((restored==c).sum());total+=c.size
                    assert np.array_equal(restored,c)
                    assert hi.nbytes+lo.nbytes==c.size//2
                    if item['split']=='calibration':
                        counts[side]+=np.bincount((c>>2).ravel(),minlength=4)
                        sums[side]+=np.bincount((c>>2).ravel(),weights=c.ravel(),minlength=4)
        print('roundtrip',i+1,len(manifest),flush=True)
    lut=np.divide(sums,counts,out=np.tile(np.arange(4)*4+1.5,(2,1)),where=counts>0)
    atomic_json(OUT/'codec.json',dict(matched_codes=exact,total_codes=total,agreement=exact/total,conditional_means=lut.tolist(),calibration_counts=counts.tolist(),storage='High2+Low2 code bytes exactly equal native nibble; metadata retained unchanged; physical layout padding not evaluated'))
    if exact!=total:raise RuntimeError('Full-read gate failed')
    for ti,item in enumerate(manifest):
        if item['split']!='heldout':continue
        data,m=load_trace(item['path']);state('running',stage='2--6 offline',trace=ti,total=len(manifest))
        for h in range(8):
            q=data['query'][0,h*4:(h+1)*4].astype(float);scale=data['scale']
            K=[];V=[];KA={x:[] for x in ('zero','mid','conditional')};VA={x:[] for x in KA};quant=[];krisk=[];vrisk=[]
            for u,k,v in blocks(data,h):
                n=v.shape[0];K.append(k);V.append(v);quant.extend([u is not None]*n)
                for method in KA:
                    if u is None:ka,va=k,v
                    else:
                        ka=(reconstruct(u['kq'][:,:n],method,lut[0])*u['sk'][:,None]+u['zk'][:,None])*u['rk'][None,:n]
                        va=(reconstruct(u['vq'][:n],method,lut[1])*u['rv'][:n,None]+u['zv'][:n,None])*u['sv'][None,:]
                    KA[method].append(ka);VA[method].append(va)
                krisk.extend([0]*n if u is None else (1.5*np.max(np.sum(np.abs(q)*np.abs(u['sk'])[None,:],axis=1))*abs(scale)*np.abs(u['rk'][:n])).tolist())
                vrisk.extend([0]*n if u is None else (1.5*np.abs(u['rv'][:n])*np.linalg.norm(u['sv'])).tolist())
            K=np.concatenate(K,axis=1);V=np.concatenate(V);quant=np.array(quant);N=len(V)
            S=q@K*scale;P=softmax(S);ref=P@V
            base=dict(trace=Path(item['path']).stem,dataset=m['request']['dataset'],case=m['request']['case_id'],cache_ordinal=m['cache_ordinal'],decode_call=m['decode_call_index'],kv_head=h,context=N)
            live=data['reference_gpu_rot'][0,h*4:(h+1)*4].astype(float)
            ROWS.append(dict(base,stage='2_reference',method='float64_vs_captured_gpu',attention_rel=rel(ref,live),attention_max_abs=float(np.abs(ref-live).max())))
            for method in KA:
                ka=np.concatenate(KA[method],axis=1);va=np.concatenate(VA[method]);sa=q@ka*scale
                for kb,vb in ((2,2),(2,4),(4,2),(4,4)):
                    out=softmax(sa if kb==2 else S)@(va if vb==2 else V)
                    ROWS.append(dict(base,stage='2_3',method=method,k_bits=kb,v_bits=vb,code_bits=(kb+vb)/2,attention_rel=rel(out,ref),k_rel=rel(ka,K),v_rel=rel(va,V),logit_rmse=float(np.sqrt(np.mean((sa-S)**2)))))
                if method!='mid':continue
                # Logical 32-token tiles; all GQA heads share each decision.
                chunks=[np.arange(j,min(j+32,N)) for j in range(0,N,32) if quant[j]]
                costs=np.array([len(ix) for ix in chunks]);mass=np.array([P[:,ix].sum(axis=1).max() for ix in chunks]);risk=np.array([np.max(np.array(vrisk)[ix]) for ix in chunks]);kpred=np.array([np.max(np.array(krisk)[ix]) for ix in chunks]);rng=np.random.default_rng(20261023+h)
                orders=dict(random=rng.permutation(len(chunks)),fixed=np.arange(len(chunks)),attention=np.argsort(-mass,kind='stable'),attention_risk=np.argsort(-mass*risk,kind='stable'),query_scale=np.argsort(-kpred,kind='stable'))
                # Offline full-attention ranking is a reference, not single-pass controller.
                for side in ('v','k','kv'):
                    names=('random','fixed','attention','attention_risk') if side=='v' else ('random','fixed','query_scale') if side=='k' else ('random','fixed','attention_risk')
                    for name in names:
                        for fraction in (0,.25,.5,.75,1):
                            mask=np.zeros(N,bool);budget=int(costs.sum()*fraction);spent=0
                            for ci in orders[name]:
                                if spent+costs[ci]>budget:continue
                                mask[chunks[ci]]=True;spent+=int(costs[ci])
                            key=K if side=='v' else np.where(mask[None,:],K,ka)
                            val=V if side=='k' else np.where(mask[:,None],V,va)
                            out=softmax(q@key*scale)@val
                            sides=2 if side=='kv' else 1
                            bits=4-.5*sides*(1-spent/max(1,costs.sum()))*2
                            ROWS.append(dict(base,stage='4_5_6_offline',method=name,side=side,fraction=fraction,residual_fraction=spent/max(1,costs.sum()),code_bits=bits,attention_rel=rel(out,ref)))
            print('evaluation',ti,h,flush=True)
        write_csv(OUT/'measurements.csv',ROWS)
    report()
    state('completed_offline_stages_2_6',rows=len(ROWS),remaining='single-pass controllers, task quality, metadata/granularity screening, physical traffic, fusion, end-to-end and ablations (stages 4--11 acceptance incomplete)')

def report():
    import collections
    groups=collections.defaultdict(list)
    for r in ROWS:
        key=tuple(r.get(k,'') for k in ('stage','method','k_bits','v_bits','side','fraction','code_bits'))
        groups[key].append(r['attention_rel'])
    ag=[]
    for key,v in groups.items():
        ag.append(dict(zip(('stage','method','k_bits','v_bits','side','fraction','code_bits'),key),n=len(v),mean=float(np.mean(v)),p95=float(np.percentile(v,95)),max=float(np.max(v))))
    write_csv(OUT/'aggregate.csv',ag)
    codec=json.loads((OUT/'codec.json').read_text())
    lines=['# 第2步起：真实缓存离线实验结论','',f"完整读取码字一致率：{codec['agreement']:.0%}；检查 {codec['total_codes']:,} 个码字。",'', '按请求划分3个校准请求与3个留出请求；覆盖3个任务、4个cache ordinal、3个decode位置和8个KV heads。条件均值只用校准请求。层序号不等同已验证的语义层名。', '', '|阶段|方法|K/V或侧|补读比例|码字平均bit|attention相对L2均值|p95|','|---|---|---|---|---|---|']
    for r in ag:
        lines.append(f"|{r['stage']}|{r['method']}|{r['side'] or str(r['k_bits'])+'/'+str(r['v_bits'])}|{r['fraction']}|{r['code_bits']}|{r['mean']:.6f}|{r['p95']:.6f}|")
    lines+=['','结论边界：第2步码字恢复验收可由一致率判断；full INT4反量化输入完全相同。原生GPU与float64输出差异另列，不能把原生舍入误差混入部分读取误差。第3步得到的是attention误差—逻辑读取曲线，尚无logit最终层误差与任务质量证据。第4—6步仅离线候选筛查，attention全局权重使用完整K，不是单遍online softmax控制器；KV使用同一mask和等比例预算，不是完成的联合分配器。Query×Scale是读取上界风险代理。','', '只统计量化码逻辑bit，FP16 sink/tail保持全精度；metadata/padding未包含在bit曲线中。完整INT4缓存始终保留，不能宣称2-bit存储、HBM流量减少或端到端加速。随机补读仅一个冻结seed，不能宣称统计优势。token logit、HumanEval/MATH500/LongBench质量、单遍控制器、阶段7筛选以及阶段8—11尚未完成；不得据此宣布完整路线成功。']
    (OUT/'结论.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        state('failed',error=repr(e));traceback.print_exc();raise
