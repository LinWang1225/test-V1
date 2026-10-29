"""Request-clustered paired analysis, generated only from completed result shards."""
from pathlib import Path
import csv,gzip,json,collections
import numpy as np

def bootstrap(a,b):
    a=np.array(a);b=np.array(b);rng=np.random.default_rng(20261024);effects=[]
    for _ in range(2000):
        ix=rng.integers(0,len(a),len(a));effects.append(1-a[ix].mean()/max(b[ix].mean(),1e-20))
    return [float(x) for x in np.percentile(effects,[2.5,97.5])]

def report(out,manifest):
    strata=collections.defaultdict(lambda:np.zeros(4));overall=collections.defaultdict(lambda:np.zeros(4))
    out=Path(out);per=collections.defaultdict(lambda:[0.,0]);tail=collections.defaultdict(list)
    for p in sorted(out.glob('*_attention.csv.gz')):
        with gzip.open(p,'rt') as f:
            for r in csv.DictReader(f):
                if r['split']!='test':continue
                key=(r['case'],r['method'],int(r['k_bits']),int(r['v_bits']))
                per[key][0]+=float(r['output_rel']);per[key][1]+=1
                vals=np.array([float(r['score_rmse']),float(r['prob_l1']),float(r['output_rel']),1.])
                overall[r['method'],int(r['k_bits']),int(r['v_bits'])]+=vals
                length='le2k' if int(r['context'])<=2048 else '2k_4k' if int(r['context'])<=4096 else '4k_8k_plus_decode'
                for dim,value in [('dataset',r['dataset']),('layer',r['layer']),('length',length)]:strata[r['method'],int(r['k_bits']),int(r['v_bits']),dim,value]+=vals
    req={k:s/n for k,(s,n) in per.items()};by=collections.defaultdict(dict)
    for (case,m,k,v),x in req.items():by[m,k,v][case]=x
    sums=collections.defaultdict(lambda:np.zeros(7));case_d=collections.defaultdict(lambda:[0.,0])
    for p in sorted(out.glob('*_distortion.csv.gz')):
        with gzip.open(p,'rt') as f:
            for r in csv.DictReader(f):
                if r['split']!='test':continue
                key=(r['method'],int(r['bits']));sums[key]+=np.array([float(r['D_K']),float(r['D_V']),float(r['k_violation_fraction']),float(r['v_violation_fraction']),float(r['k_mse']),float(r['v_mse']),1])
                for side in ('K','V'):
                    k=(r['case'],r['method'],int(r['bits']),side);case_d[k][0]+=float(r['D_'+side]);case_d[k][1]+=1
    lines=['# 独立渐进 KV 编码消融：结果分析','',
        '本实验独立于 KVarN 与 vLLM，使用同一 FP16 Qwen3-4B 轨迹的量化前真实 Q/K/V。所有层均独立反事实评估；未评估误差逐层传播、任务质量、真实显存流量或速度。',
        '','配置、偏离与文件/函数说明见 ../../PROTOCOL.md 与 ../../CHANGES.md。完整缓存始终保留 4-bit payload，2/3-bit 表示前缀读取深度。',
        '',f"输入请求：{len(manifest)}；校准 {sum(r['split']=='calibration' for r in manifest)}、验证 {sum(r['split']=='validation' for r in manifest)}、测试 {sum(r['split']=='test' for r in manifest)}。三个校准请求 bootstrap 种子先在请求内平均，不视作独立测试样本。",'',
        '|方法|bit|D_K|D_V|K 通道收益递减违反率|V token 收益递减违反率|','|---|---:|---:|---:|---:|---:|']
    for (m,b),s in sorted(sums.items()):
        a=s[:6]/s[6];lines.append(f'|{m}|{b}|{a[0]:.6g}|{a[1]:.6g}|{a[2]:.2%}|{a[3]:.2%}|')
    lines+=['','|方法|K bit|V bit|请求平均输出相对 L2|请求 P95|','|---|---:|---:|---:|---:|']
    for (m,k,v),cases in sorted(by.items()):
        a=list(cases.values());lines.append(f'|{m}|{k}|{v}|{np.mean(a):.6g}|{np.percentile(a,95):.6g}|')
    lines+=['','|方法|K/V bit|Score RMSE|Probability L1|Output 相对 L2|','|---|---|---:|---:|---:|']
    for (m,k,v),a in sorted(overall.items()):
        if k==v:lines.append(f'|{m}|{k}/{v}|{a[0]/a[3]:.6g}|{a[1]/a[3]:.6g}|{a[2]/a[3]:.6g}|')
    with (out/'stratified_attention.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','k_bits','v_bits','dimension','value','score_rmse','prob_l1','output_rel','n_rows']);w.writerows([(*key,*(a[:3]/a[3]),int(a[3])) for key,a in sorted(strata.items())])
    comparisons=[]
    lines+=['','配对改善 = 1 − 候选误差均值 / 对照误差均值。95% CI 按独立请求重采样 2,000 次；5% 为预先冻结的实用收益阈值。多组比较是探索性结果，不构成多重检验校正后的显著性宣称。','',
        '|指标|候选/对照|精度|相对改善|95% CI|判断|','|---|---|---|---:|---|---|']
    def add(metric,m,ref,b,a,c):
        effect=1-np.mean(a)/max(np.mean(c),1e-20);ci=bootstrap(a,c)
        verdict='支持超过5%收益' if ci[0]>.05 else '排除5%收益' if ci[1]<.05 else '证据不足'
        lines.append(f'|{metric}|{m}/{ref}|{b}|{effect:.2%}|[{ci[0]:.2%}, {ci[1]:.2%}]|{verdict}|')
        comparisons.append(dict(metric=metric,candidate=m,reference=ref,precision=b,effect=float(effect),ci=ci,verdict=verdict))
    for m,ref in [('C','B'),('D','C'),('D','B'),('C_eq','B_eq'),('D_eq','C_eq'),('D_eq','B_eq')]:
        for b in (2,3,4):
            x=by[m,b,b];y=by[ref,b,b];cases=sorted(set(x)&set(y))
            if cases:add('Output',m,ref,str(b),[x[c] for c in cases],[y[c] for c in cases])
            for side in ('K','V'):
                cases=sorted(c for c,mm,bb,ss in case_d if mm==m and bb==b and ss==side)
                if cases:add('D_'+side,m,ref,str(b),[case_d[c,m,b,side][0]/case_d[c,m,b,side][1] for c in cases],[case_d[c,ref,b,side][0]/case_d[c,ref,b,side][1] for c in cases])
    maxgap=0.;nalloc=0
    with gzip.open(out/'allocation.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);maxgap=max(maxgap,r['gap']);nalloc+=1
    ledger=json.loads((out/'storage_ledger.json').read_text())
    lines+=['',f'小规模精确动态规划比较：{nalloc} 个预算实例，贪心最大目标差距 {maxgap:.8g}。这只验证已抽取的 8-item 子问题，不能证明全局实例最优。',
        '','|方法|上下文 token|4-bit payload 字节|部署元数据字节|总字节|相同总预算可容纳 token|','|---|---:|---:|---:|---:|---:|']
    for r in ledger:
        if r['read_bits']==4:lines.append('|'+ '|'.join(str(r[k]) for k in ('method','context','stored_payload_bytes','metadata_bytes','total_bytes','max_tokens_under_equal_budget'))+'|')
    lines+=['','存储账本按实际紧凑数组计费：均值/Scale、所有2/3/4层质心、编码阈值、固定变换符号均计入。Python 实验会将码字展开为 int64 以计算，以上是已验证 nibble 打包格式的部署大小，不是 Python 进程实测峰值。',
        '带 _eq 的结果为同上下文、同总字节上限：相对 B 的额外元数据通过在每个 head/side 的固定前缀标量省去 Bit0 支付；4-bit 解码这些位置使用3-bit质心，2/3-bit读取不受影响。省略位置由布局固定，无逐token mask。逐stream字节向上取整，实际大小可能略低于上限。equal_budget.csv 给出对应实测误差。',
        '','重建失真/边际收益为实测值；没有沿用 Gaussian 理论保证。收益递减只检查 2→3→4；0/1-bit 未评估。局部 Jacobian 对角权重是近似目标，不包含完整误差交叉项。',
        '','ReadKV Gaussian 数值采用正态分位点求积近似；C/D 在 8,192-bin 充分统计近似上聚类。官方 Any-Precision 的逐级分裂方法被适配到激活；不复现其权重梯度或 CUDA kernel。',
        '','验证集仅提供独立诊断，本次固定参数不按验证结果调参。原始逐层/head结果、score/probability指标在压缩 CSV 中。完整测试结束才可据此定论。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n');(out/'comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
    with (out/'equal_budget.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','method','k_bits','v_bits','output_rel']);w.writerows([(*k,v) for k,v in sorted(req.items()) if k[1].endswith('_eq')])
    with (out/'request_metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','method','k_bits','v_bits','output_rel']);w.writerows([(*k,v) for k,v in sorted(req.items())])
if __name__=='__main__':
    import sys
    out=Path(sys.argv[1]);manifest=json.loads((out.parents[1]/'inputs/manifest.json').read_text());report(out,manifest)
