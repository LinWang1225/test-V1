"""E versus frozen A-D, with paired request comparisons and explicit byte costs."""
import csv,gzip,json,collections,re
from pathlib import Path
import numpy as np

def report(out,old):
    out=Path(out);old=Path(old);sums=collections.defaultdict(lambda:np.zeros(5));raw=collections.defaultdict(lambda:np.zeros(2));dists=collections.defaultdict(lambda:np.zeros(7))
    for p in sorted(out.glob('*_attention.csv.gz')):
        with gzip.open(p,'rt') as f:
            for r in csv.DictReader(f):
                if r['split']!='test':continue
                k=(r['case'],r['method'],int(r['k_bits']),int(r['v_bits']));raw[k]+=[float(r['output_rel']),1]
                sums[r['method'],int(r['k_bits']),int(r['v_bits'])]+=[float(r['score_rmse']),float(r['prob_l1']),float(r['output_rel']),float(r['output_rmse']),1]
    for p in sorted(out.glob('*_distortion.csv.gz')):
        with gzip.open(p,'rt') as f:
            for r in csv.DictReader(f):
                if r['split']!='test':continue
                dists[r['method'],int(r['bits'])]+=[float(r['D_K']),float(r['D_V']),float(r['k_violation_fraction']),float(r['v_violation_fraction']),float(r['quantized_span_D_K']),float(r['quantized_span_D_V']),1]
    req={k:float(s[0]/s[1]) for k,s in raw.items()};combined=dict(req)
    with (old/'request_metrics.csv').open() as f:
        for r in csv.DictReader(f):combined[r['case'],r['method'],int(r['k_bits']),int(r['v_bits'])]=float(r['output_rel'])
    by=collections.defaultdict(dict)
    for (case,m,k,v),x in combined.items():by[m,k,v][case]=x
    ledger=json.loads((out/'storage_ledger.json').read_text())
    minimum=min((r['minimum_stored_prefix_bits'] for r in ledger),default=4)
    title='E 补充基线：KVarN原生量化器前缀重建'
    lines=['# '+title,'','E直接调用冻结的KVarN量化源码：无新码本、无校准、无学习重建。E用缺失位区间中点，E_zero用缺失位补零；都是同一个原生INT4码字的高位前缀，完整4-bit两者相同。原生FP16 Scale/零点完整保留且计费，不能将“无新增辅助数据”解释为“无需原生元数据”。',
        '','E仅使用128-token完整tile，未满tile尾部保留FP16（计费）；关闭sink/boundary保护，避免引入更多系统策略。A-D原实验量化全部token，因此尾部策略仍有差异；E另报完整量化tile的重建失真，但这不消除attention指标中的尾部保留影响。',
        '','E_eq/E_zero_eq用固定位置从最低位开始逐平面省略，支付原生元数据与FP16尾部相对B的额外成本，保持同上下文且总字节不超过B基准。仅验证紧凑逻辑位平面格式，不声称原生vLLM缓存已按该可变位宽布局运行。',
        '','|方法|K/V bit|请求均值 Output相对L2|请求P95|','|---|---:|---:|---:|']
    for (m,k,v),a in sorted(by.items()):
        if k==v:
            vals=list(a.values());lines.append(f'|{m}|{k}|{np.mean(vals):.7g}|{np.percentile(vals,95):.7g}|')
    lines+=['','|E变体|bit|D_K|D_V|完整量化tile D_K|完整量化tile D_V|K收益递减违反率|V收益递减违反率|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for (m,b),a in sorted(dists.items()):
        x=a[:6]/a[6];lines.append(f'|{m}|{b}|{x[0]:.7g}|{x[1]:.7g}|{x[4]:.7g}|{x[5]:.7g}|{x[2]:.3%}|{x[3]:.3%}|')
    lines+=['','|E变体|K/V bit|Score RMSE|Probability L1|Output RMSE|','|---|---:|---:|---:|---:|']
    for (m,k,v),a in sorted(sums.items()):
        if k==v:lines.append(f'|{m}|{k}|{a[0]/a[4]:.7g}|{a[1]/a[4]:.7g}|{a[3]/a[4]:.7g}|')
    lines+=['','配对改善=1−E误差均值/对照均值，正数表示E更好；95%CI按请求bootstrap2000次。不将层、head、decode当独立样本。','',
        '|候选|对照|bit|配对请求数|Output误差改善|95%CI|','|---|---|---:|---:|---:|---|']
    comparisons=[]
    for m in ('E','E_zero','E_eq','E_zero_eq'):
        refs=('A_eq','B_eq','C_eq','D_eq') if m.endswith('_eq') else ('A','B','C','D')
        for ref in refs:
            for b in (2,3,4):
                x,y=by[m,b,b],by[ref,b,b];cases=sorted(set(x)&set(y))
                if not cases:continue
                a=np.array([x[c] for c in cases]);z=np.array([y[c] for c in cases]);rng=np.random.default_rng(20261024)
                effects=[]
                for _ in range(2000):
                    ix=rng.integers(0,len(a),len(a));effects.append(1-a[ix].mean()/max(z[ix].mean(),1e-20))
                effect=1-a.mean()/max(z.mean(),1e-20);lo,hi=np.percentile(effects,[2.5,97.5])
                comparisons.append(dict(method=m,reference=ref,bits=b,n=len(cases),improvement=float(effect),ci=[float(lo),float(hi)]))
                lines.append(f'|{m}|{ref}|{b}|{len(cases)}|{effect:.2%}|[{lo:.2%}, {hi:.2%}]|')
    lines+=['','|context|E 4-bit码字字节|原生metadata字节|有效FP16尾部字节|E紧凑总字节|E_eq总字节|B总预算字节|','|---:|---:|---:|---:|---:|---:|---:|']
    # All captured contexts retained in JSON; show typical nearest 2/4/8k rows.
    for target in (2049,4097,8193,8256):
        if not ledger:continue
        r=min(ledger,key=lambda r:abs(r['context']-target))
        lines.append('|'+ '|'.join(str(r[k]) for k in ('context','stored_payload_bytes','native_metadata_bytes','tail_fp16_bytes','total_bytes','equal_total_bytes','B_total_budget_bytes'))+'|')
    lines+=['',f'等预算最低实际保留前缀为{minimum}-bit（跨本轮已测context）；这是支付原生元数据和尾部成本后的实际有效精度，不能将请求4-bit解码误称所有标量均保留4-bit。','存储记账是可序列化的紧凑逻辑数组：全部完成tile的native packed bytes/FP16元数据及有效FP16尾部均计入；不计Python/Torch中间数组，不包含vLLM池化分配保留的闲置slot。另在storage_ledger.json记录原生尾池按128-token槽分配时的容量上界，禁止将紧凑字节等同原生运行时显存峰值。',
        '','指标为单层反事实误差；E参考decode token序列逐请求与A-D轨迹SHA256核对。未重新运行A-D，也未评估任务质量、Kernel性能或误差跨层传播。',
        '','原生对照与文件/函数说明见本扩展的PROTOCOL_E.md、CHANGES_E.md；原A-D结果保留在上级目录。']
    (out/'E对比分析.md').write_text('\n'.join(lines)+'\n');(out/'comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
    with (out/'request_metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','method','k_bits','v_bits','output_rel']);w.writerows([(*k,v) for k,v in sorted(req.items())])
