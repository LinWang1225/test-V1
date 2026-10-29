"""Streaming completed-suite analysis, Parquet datasets, figures and byte ledger."""
import os,sys,json,csv,gzip,collections,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(R/'deps'));os.environ['MPLCONFIGDIR']=str(R/'logs/matplotlib')
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def writecsv(p,rows):
    if not rows:return
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rows(path):
    with gzip.open(path,'rt') as f:yield from csv.DictReader(f)
def bootstrap(a,b):
    rng=np.random.default_rng(20261025);a=np.array(a);b=np.array(b);effects=[]
    for _ in range(2000):
        ix=rng.integers(0,len(a),len(a));effects.append(1-a[ix].mean()/max(b[ix].mean(),1e-20))
    return np.percentile(effects,[2.5,97.5]).tolist()
def main():
    manifest=json.loads((R/'input_manifest.json').read_text());expected={r['case_id'].replace('/','_') for r in manifest if r['split']=='test'};folder=R/'metrics/evaluation'
    actual={p.name.removesuffix('_attention.csv.gz') for p in folder.glob('*_attention.csv.gz')};assert expected==actual
    active=json.loads((R/'codebooks/full/active_methods.json').read_text());methods=active['methods']
    attention_extra=collections.defaultdict(lambda:np.zeros(4));relative=collections.defaultdict(lambda:np.zeros(3))
    request=collections.defaultdict(lambda:np.zeros(2));layer=collections.defaultdict(lambda:np.zeros(2));dist=collections.defaultdict(lambda:np.zeros(7));head=collections.defaultdict(lambda:np.zeros(7));sample_static=collections.defaultdict(lambda:np.zeros(2))
    for p in sorted(folder.glob('*_attention.csv.gz')):
        print('analysis attention',p.name,flush=True)
        for r in rows(p):
            attention_extra[r['method'],int(r['k_bits']),int(r['v_bits'])]+=[float(r['score_rmse'])**2,float(r['prob_l1']),float(r['output_rmse']),1.]
            val=float(r['output_rel']);key=(r['case'],r['method'],int(r['k_bits']),int(r['v_bits']));request[key]+=[val,1];layer[r['method'],int(r['layer']),int(r['k_bits']),int(r['v_bits'])]+=[val,1]
            if int(r['seed'])==20261024 and int(r['layer']) in (0,17,35) and int(r['position']) in (1,64) and r['k_bits']==r['v_bits']:
                sample_static[r['method'],float(r['k_bits'])]+=[val,1]
    for p in sorted(folder.glob('*_distortion.csv.gz')):
        for r in rows(p):
            relative[r['method'],int(r['bits'])]+=[np.sqrt(float(r['D_K'])),np.sqrt(float(r['D_V'])),1.]
            val=np.array([float(r[n]) for n in ('k_mse','v_mse','D_K','D_V','k_score_diag','v_attention_diag')]+[1.]);dist[r['method'],int(r['bits'])]+=val;head[r['method'],int(r['layer']),int(r['head']),int(r['bits'])]+=val
    rq={k:float(v[0]/v[1]) for k,v in request.items()};by=collections.defaultdict(dict)
    for (case,m,k,v),value in rq.items():by[m,k,v][case]=value
    assert all(len(a)==90 for a in by.values())
    writecsv(R/'metrics/request_metrics.csv',[dict(case=k[0],method=k[1],k_bits=k[2],v_bits=k[3],output_rel=v) for k,v in sorted(rq.items())])
    attention=[dict(method=k[0],k_bits=k[1],v_bits=k[2],mean=float(np.mean(list(v.values()))),p95=float(np.percentile(list(v.values()),95)),n_requests=len(v)) for k,v in sorted(by.items())]
    for a in attention:
        extra=attention_extra[a['method'],a['k_bits'],a['v_bits']];a.update(score_mse=float(extra[0]/extra[3]),probability_l1=float(extra[1]/extra[3]),output_rmse=float(extra[2]/extra[3]))
    writecsv(R/'metrics/attention_errors.csv',attention)
    reconstruction=[dict(method=k[0],bits=k[1],**dict(zip(('k_mse','v_mse','D_K','D_V','k_score_diag','v_attention_diag'),(v[:6]/v[6]).tolist()))) for k,v in sorted(dist.items())]
    for a in reconstruction:
        extra=relative[a['method'],a['bits']];a.update(k_relative_l2=float(extra[0]/extra[2]),v_relative_l2=float(extra[1]/extra[2]))
        e=attention_extra[a['method'],a['bits'],a['bits']];a['k_score_exact_mse']=float(e[0]/e[3])
    writecsv(R/'metrics/reconstruction_errors.csv',reconstruction)
    writecsv(R/'metrics/layer_head_metrics.csv',[dict(method=k[0],layer=k[1],head=k[2],bits=k[3],**dict(zip(('k_mse','v_mse','D_K','D_V','k_score_diag','v_attention_diag'),(v[:6]/v[6]).tolist()))) for k,v in sorted(head.items())])
    curve_stats=collections.defaultdict(lambda:np.zeros(5));sample=[];sampler=np.random.default_rng(20261025)
    for name in ('distortion_curves.parquet','marginal_gains.parquet'):(R/'metrics'/name).mkdir(exist_ok=True)
    for p in sorted(folder.glob('*_curves.parquet')):
        for name in ('distortion_curves.parquet','marginal_gains.parquet'):
            link=R/'metrics'/name/p.name
            if not link.exists():link.symlink_to(Path('../evaluation')/p.name)
        for batch in pq.ParquetFile(p).iter_batches(batch_size=65536,columns=['method','side','objective','D2','D3','D4','gain_23','gain_34','diminishing_violation','negative_gain','violation_magnitude']):
            a={n:batch.column(n).to_numpy(zero_copy_only=False) for n in batch.schema.names}
            for m in np.unique(a['method']):
                for side in ('K','V'):
                    for objective in ('raw_mse','functional_proxy'):
                        mask=(a['method']==m)&(a['side']==side)&(a['objective']==objective);n=int(mask.sum())
                        if not n:continue
                        key=(m,side,objective);v=curve_stats[key];v[:4]+=[n,a['diminishing_violation'][mask].sum(),a['negative_gain'][mask].sum(),a['violation_magnitude'][mask].sum()];v[4]=max(v[4],float(a['violation_magnitude'][mask].max()))
            ix=sampler.choice(len(a['D2']),min(256,len(a['D2'])),replace=False)
            for i in ix:sample.append(dict(method=a['method'][i],side=a['side'][i],objective=a['objective'][i],gain23=float(a['gain_23'][i]),gain34=float(a['gain_34'][i])))
    violations=[dict(method=k[0],side=k[1],objective=k[2],n_units=int(v[0]),violation_fraction=float(v[1]/v[0]),negative_gain_fraction=float(v[2]/v[0]),mean_normalized_violation=float(v[3]/v[0]),max_normalized_violation=float(v[4])) for k,v in sorted(curve_stats.items())]
    writecsv(R/'metrics/diminishing_violations.csv',violations)
    dp=[];dynamic=[]
    for p in sorted(folder.glob('*_dp.csv.gz')):dp.extend(rows(p))
    writecsv(R/'metrics/greedy_vs_dp.csv',dp)
    for p in sorted(folder.glob('*_dynamic.csv.gz')):dynamic.extend(rows(p))
    writecsv(R/'metrics/dynamic_reading.csv',dynamic)
    utilization=collections.defaultdict(lambda:[np.zeros(16,dtype=np.int64),0.,0.,0])
    for p in sorted(folder.glob('*_codebook.csv.gz')):
        for r in rows(p):
            key=(r['method'],int(r['layer']),r['side'],int(r['head']));v=utilization[key];v[0]+=np.array(json.loads(r['leaf_counts']),np.int64);v[1]+=float(r['encode_ms']);v[2]+=float(r['decode_3depths_ms']);v[3]+=1
    teststats=[]
    for k,v in sorted(utilization.items()):
        hist=v[0];teststats.append(dict(method=k[0],layer=k[1],side=k[2],head=k[3],leaf_counts=json.dumps(hist.tolist()),prefix2_counts=json.dumps(hist.reshape(4,4).sum(1).tolist()),prefix3_counts=json.dumps(hist.reshape(8,2).sum(1).tolist()),used_leaves=int((hist>0).sum()),mean_reference_encode_ms=v[1]/v[3],mean_reference_decode_3depths_ms=v[2]/v[3],n_observations=v[3]))
    writecsv(R/'metrics/test_codebook_statistics.csv',teststats)
    # Training statistics retain clip ranges, histogram curves, metadata and actual joint iteration histories.
    cb=json.loads((R/'codebooks/full/calibration_statistics.json').read_text());writecsv(R/'metrics/codebook_statistics.csv',cb)
    training=json.loads((R/'codebooks/full/training_history.json').read_text())
    # Per-context actual compact state for one deployed method, no omitted KV bits.
    contexts=sorted({r['context']+p for r in manifest if r['split']=='test' for p in (1,8,32,64)});ledger=[]
    for n in contexts:
        for m,pair in methods.items():
            diagnostic=m.startswith('Independent');payloadbits=9 if diagnostic else 4
            payload=(36*2*8*n*128*payloadbits+7)//8;normal=36*2*8*128*4+128
            unique_gaussian=all(x=='tree:gaussian' for x in pair)
            codebytes=86 if unique_gaussian else 36*8*sum(4 if key.startswith('uniform:') else 12 if key.startswith('uniform_independent:') else 106 if key.startswith('independent:') else 86 for key in pair)
            ledger.append(dict(method=m,context=n,stored_payload_bits=payloadbits,diagnostic_independent=diagnostic,payload_bytes=payload,normalization_bytes=normal,codebook_bytes=codebytes,static_total_bytes=payload+normal+codebytes,dynamic_profile_bytes=36*2*8*3*8,tail_fp16_bytes=0))
    writecsv(R/'metrics/storage_ledger.csv',ledger)
    # Nesting penalty compares the corresponding independently trained family; local Lloyd is not a proven global optimum.
    penalties=[]
    for row in reconstruction:
        method=row['method'];b=row['bits'];ref='Independent_Uniform' if method.startswith(('U_','Uniform_')) else 'Independent_MSE'
        reference=next(x for x in reconstruction if x['method']==ref and x['bits']==b)
        for side in ('K','V'):
            field='D_'+side;penalties.append(dict(method=method,bits=b,side=side,reference=ref,nesting_penalty=(row[field]-reference[field])/max(reference[field],1e-20),interpretation='same-family empirical independently trained reference; not globally optimal or strict upper bound'))
    writecsv(R/'metrics/nesting_penalty.csv',penalties)
    # Four mandatory plot groups (standard plotting, standalone PNG artifacts).
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    focus=['U_MinMax','U_Clipped','Gaussian_Tree','Data_Tree','JacK_MSEV','ScoreK_MixV','Joint_234']
    for m in focus:
        for ax,field in zip(axes[:2],('D_K','D_V')):ax.plot([2,3,4],[next(r[field] for r in reconstruction if r['method']==m and r['bits']==b) for b in (2,3,4)],marker='o',label=m)
        axes[2].plot([2,3,4],[np.mean(list(by[m,b,b].values())) for b in (2,3,4)],marker='o',label=m)
    for ax,title in zip(axes,['K normalized reconstruction','V normalized reconstruction','Attention output relative L2']):ax.set_title(title);ax.set_xlabel('read bits');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[2].legend(fontsize=7);fig.tight_layout();fig.savefig(R/'figures/distortion_vs_bits.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,4))
    for ax,field,title in zip(axes,('k_score_diag','v_attention_diag'),('K diagonal score proxy','V attention-weighted proxy')):
        for m in focus:ax.plot([2,3,4],[next(r[field] for r in reconstruction if r['method']==m and r['bits']==b) for b in (2,3,4)],marker='o',label=m)
        ax.set_title(title);ax.set_xlabel('read bits');ax.set_yscale('log');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(R/'figures/functional_distortion_vs_bits.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    for side,ax in zip(('K','V'),axes):
        values=[];labels=[]
        for m in focus:
            for gain in ('gain23','gain34'):
                a=[x[gain] for x in sample if x['side']==side and x['objective']=='functional_proxy' and x['method']==m]
                values.append(a or [0.]);labels.append(m+' '+gain)
        ax.boxplot(values,tick_labels=labels,showfliers=False);ax.tick_params(axis='x',rotation=85);ax.set_yscale('symlog',linthresh=1e-8);ax.set_title(side+' functional gains, fixed random sample; fliers omitted')
    fig.tight_layout();fig.savefig(R/'figures/marginal_gain_distribution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));selected=[r for r in violations if r['objective']=='functional_proxy' and r['method'] in focus]
    ax.bar([r['method']+' '+r['side'] for r in selected],[r['violation_fraction'] for r in selected]);ax.tick_params(axis='x',rotation=80);ax.set_ylabel('violation fraction');fig.tight_layout();fig.savefig(R/'figures/diminishing_violations.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    dyn=collections.defaultdict(list)
    for r in dynamic:dyn[r['method'],float(r['actual_mean_bits'])].append(float(r['output_rel']))
    for m in focus:
        points=sorted((b,np.mean(v)) for (mm,b),v in dyn.items() if mm==m)
        if points:ax.plot([p[0] for p in points],[p[1] for p in points],marker='o',label=m+' dynamic')
        points=sorted((b,float(v[0]/v[1])) for (mm,b),v in sample_static.items() if mm==m)
        if points:ax.plot([p[0] for p in points],[p[1] for p in points],linestyle='--',alpha=.5,label=m+' fixed')
    ax.set_xlabel('actual mean logical payload read bits');ax.set_ylabel('mean per-head output relative L2');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'figures/output_error_vs_read_bytes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
    for b,ax in zip((2,3,4),axes):
        data=[]
        for m in focus:
            data.append([1-(layer[m,l,b,b][0]/layer[m,l,b,b][1])/(layer['Data_Tree',l,b,b][0]/layer['Data_Tree',l,b,b][1]) for l in range(36)])
        im=ax.imshow(data,aspect='auto',cmap='RdBu',vmin=-.5,vmax=.5);ax.set_title(f'{b}-bit improvement vs Data_Tree');ax.set_xlabel('layer (0-based)');ax.set_yticks(range(len(focus)),focus)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8);fig.savefig(R/'figures/layerwise_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Pair primary hypotheses using the fixed reused test cohort; explicitly exploratory.
    comparisons=[]
    for cand,ref in [('U_Clipped','Data_Tree'),('JacK_MSEV','Data_Tree'),('ScoreK_MSEV','Data_Tree'),('ScoreK_MixV','Data_Tree'),('Joint_234','Data_Tree')]:
        for b in (2,3,4):
            aa=by[cand,b,b];bb=by[ref,b,b];cases=sorted(aa);a=[aa[c] for c in cases];z=[bb[c] for c in cases]
            comparisons.append(dict(candidate=cand,reference=ref,bits=b,improvement=1-float(np.mean(a))/float(np.mean(z)),ci=bootstrap(a,z)))
    (R/'metrics/comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
    lines=['# 补充离线消融：结果分析','', '已完成60校准、90复用测试；按用户指示跳过30验证请求。36层、8 KV heads、decode1/8/32/64，三个校准bootstrap种子先在请求内平均。复用测试集已参与研究方向选择，不能称未接触的最终确认集。', '', '主方法全部有效KV元素完整4-bit存储，不含FP16 KV保护、不删token、不省略Bit0。Independent两组仅为非渐进参考，分别存储2/3/4-bit编码合计9-bit，不属于完整4-bit主比较。', '', '|方法|2-bit output rel L2|3-bit|4-bit|','|---|---:|---:|---:|']
    for m in methods:lines.append('|'+m+'|'+ '|'.join(f'{np.mean(list(by[m,b,b].values())):.7g}' for b in (2,3,4))+'|')
    lines+=['','|候选/对照|bit|输出误差改善|95%CI|','|---|---:|---:|---|']
    for r in comparisons:lines.append(f"|{r['candidate']}/{r['reference']}|{r['bits']}|{r['improvement']:.2%}|[{r['ci'][0]:.2%}, {r['ci'][1]:.2%}]|")
    lines+=['', 'KVarN去FP16保留消融见 metrics/native_nt/results_summary.md，和冻结旧E按请求配对。它保持原生分块粒度，与统一每层/head码本主实验仍属于不同量化流程，不能只归因码本。', '', f"收益递减正则校准门槛触发head数：{active['regularizer_triggered_heads']}。若为0，该阶段按规划跳过；不会依据测试结果重新开启。", '', 'E4动态读取为离线PyTorch两阶段原型：K用当前Q与校准曲线分配，V使用重建K得到的attention分配；不使用测试重建误差作重要性。仅覆盖L0/L17/L35与decode1/64、首seed和5个预注册方法。固定精度覆盖全部层/位置/seed。逻辑读取量不等于真实HBM流量。', '', 'DP使用float64复核小规模8-item问题，不能视为全规模分配最优性的证明。逐单元曲线保留固定4个K channel、4个V token/head的raw与functional两种目标，覆盖全部测试层/head/位置/seed；采样布局固定、非按误差选择。', '', '普通MSE训练沿用旧C的标准化空间；Score/Jacobian/Attention权重包含链式Scale。V混合权重=0.5普通MSE+0.5归一化attention权重。独立Lloyd仅是经验参考，不是数学上已证明的全局最优上限。', '', '编码/解码时间是本机PyTorch参考函数的CUDA event时间，非优化Kernel或在线吞吐测试。E5不自动执行，需离线筛选与实装；本报告不宣称端到端收益。', '', '完整方法、权重、元数据、阈值、采样范围及条件跳过记录见protocol.md、config.json、codebooks/full/training_history.json。所有原始测量在metrics/evaluation分片；Parquet数据集两个入口共享同一物理曲线分片，避免重复占用磁盘。']
    (R/'results_summary.md').write_text('\n'.join(lines)+'\n')
    (R/'metrics/completeness.json').write_text(json.dumps(dict(test_requests=90,active_methods=list(methods),validation_requests=0,all_attention_shards_present=True),indent=2))
    with (B/'结果分析.md').open('a') as f:f.write('\n\n下一轮统一4-bit补充实验已完成，见 [supplementary_experiments/results_summary.md](supplementary_experiments/results_summary.md)。原始A–E结果保持冻结。\n')
    print('all supplementary analysis completed',flush=True)
if __name__=='__main__':main()
