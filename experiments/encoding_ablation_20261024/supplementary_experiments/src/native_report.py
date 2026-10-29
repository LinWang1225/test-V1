import csv,gzip,collections,json
import numpy as np

def report(out,base):
    sums=collections.defaultdict(lambda:np.zeros(2))
    for p in out.glob('*_attention.csv.gz'):
        with gzip.open(p,'rt') as f:
            for r in csv.DictReader(f):
                key=(r['case'],r['method'],int(r['k_bits']),int(r['v_bits']));sums[key]+=[float(r['output_rel']),1]
    rows=[dict(case=k[0],method=k[1],k_bits=k[2],v_bits=k[3],output_rel=float(v[0]/v[1])) for k,v in sorted(sums.items())]
    with (out/'request_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    old={}
    with (base/'E_extension/results/full/request_metrics.csv').open() as f:
        for r in csv.DictReader(f):old[r['case'],r['method'],int(r['k_bits']),int(r['v_bits'])]=float(r['output_rel'])
    lines=['# KVarN 无 FP16 KV 保留消融','', '保持原E的输入/轨迹/完整块量化器不变，仅将未满128-token尾块也量化为4-bit。保留必需FP16元数据；没有FP16 KV数据旁路。单token退化块不做方差均衡，原生范围量化与元数据计费保留。', '', '|方案|bit|原E输出相对L2|无FP16 KV输出相对L2|','|---|---:|---:|---:|']
    for m,ref in [('KVARN_nt','E'),('KVARN_nt_zero','E_zero')]:
        for b in (2,3,4):
            rr=[r for r in rows if r['method']==m and r['k_bits']==r['v_bits']==b]
            a=np.mean([old[r['case'],ref,b,b] for r in rr]);v=np.mean([r['output_rel'] for r in rr]);lines.append(f'|{m}|{b}|{a:.7g}|{v:.7g}|')
    (out/'results_summary.md').write_text('\n'.join(lines)+'\n')
