"""Request-level summaries, paired CIs, baseline audit, plots and byte ledger."""
import argparse,csv,gzip,collections
from common import *


def read(path):
    with gzip.open(path,'rt') as f:
        yield from csv.DictReader(f)


def analyze(smoke=False):
    out=ROOT/'results'/('smoke' if smoke else 'full')
    done=json.loads((out/'evaluation_complete.json').read_text())
    expected=1 if smoke else 90
    assert done['complete'] and done['requests']==expected
    fields=('output_mse','output_rel','score_mse','prob_l1','v_diag_mse','v_exact_mse','v_cross_mse','k_output_mse','kv_interaction_mse')
    requests=[];layer_sums=collections.defaultdict(lambda:np.zeros(len(fields)+1))
    paths=sorted(out.glob('*_attention.csv.gz'))
    assert len(paths)==expected
    baseline_check=[]
    for i,path in enumerate(paths):
        state('encoding_analysis',completed=i,total=expected)
        sums=collections.defaultdict(lambda:np.zeros(len(fields)+1))
        legacy_rows={}
        if i==0:
            legacy_path=SUPP/'metrics'/('smoke' if smoke else 'evaluation')/path.name
            def key(r):return (r['method'],r['seed'],r['layer'],r['position'],r['head'],r['k_bits'],r['v_bits'])
            legacy_rows={key(r):r for r in read(legacy_path) if r['method'] in METHODS}
        for r in read(path):
            a=np.array([float(r[n]) for n in fields]+[1.])
            assert np.isfinite(a).all()
            k=(r['case'],r['dataset'],r['method'],int(r['k_bits']),int(r['v_bits']))
            sums[k]+=a
            layer_sums[r['method'],int(r['layer']),int(r['k_bits']),int(r['v_bits'])]+=a
            if legacy_rows and r['method'] in ('U_Clipped','Data_Tree','Jacobian_Tree'):
                old=legacy_rows[key(r)]
                assert np.isclose(float(r['output_rel']),float(old['output_rel']),rtol=2e-5,atol=1e-7),(key(r),r['output_rel'],old['output_rel'])
                baseline_check.append(abs(float(r['output_rel'])-float(old['output_rel'])))
        for k,a in sums.items():
            requests.append(dict(zip(('case','dataset','method','k_bits','v_bits'),k),**dict(zip(fields,map(float,a[:-1]/a[-1])))))
    assert len(requests)==expected*len(METHODS)*15
    write_csv(out/'request_metrics.csv',requests)
    grouped=collections.defaultdict(list)
    for r in requests:grouped[r['method'],r['k_bits'],r['v_bits']].append(r)
    summary=[]
    for k,rows in sorted(grouped.items()):
        assert len(rows)==expected
        summary.append(dict(method=k[0],k_bits=k[1],v_bits=k[2],requests=len(rows),**{n:float(np.mean([r[n] for r in rows])) for n in fields},output_mse_p95=float(np.percentile([r['output_mse'] for r in rows],95)),output_rel_p95=float(np.percentile([r['output_rel'] for r in rows],95))))
    write_csv(out/'summary.csv',summary)
    write_csv(out/'layer_metrics.csv',[dict(method=k[0],layer=k[1],k_bits=k[2],v_bits=k[3],**dict(zip(fields,map(float,a[:-1]/a[-1])))) for k,a in sorted(layer_sums.items())])
    comparisons=[]
    rng=np.random.default_rng(20261026)
    for m in NEW_METHODS:
        for baseline in ('U_Clipped','Data_Tree','Jacobian_Tree'):
            for b in BITS:
                aa={r['case']:r for r in grouped[m,b,b]};bb={r['case']:r for r in grouped[baseline,b,b]}
                assert aa.keys()==bb.keys()
                for field in ('output_mse','output_rel'):
                    a=np.array([aa[c][field] for c in sorted(aa)]);v=np.array([bb[c][field] for c in sorted(aa)])
                    ix=rng.integers(0,len(a),(2000,len(a)))
                    effects=1-a[ix].mean(1)/np.maximum(v[ix].mean(1),1e-30)
                    comparisons.append(dict(method=m,baseline=baseline,bits=b,metric=field,improvement=float(1-a.mean()/max(v.mean(),1e-30)),ci95=np.percentile(effects,[2.5,97.5]).tolist(),n_requests=len(a)))
    atomic(out/'comparisons.json',comparisons)
    ledger=[]
    layers=(0,17,35) if smoke else range(36)
    books=[load_layer(l,SEEDS[0],smoke,'cpu') for l in layers]
    for n in (2048,4096,8192):
        for m in METHODS:
            # Explicitly charge all decoder tables actually used by this reference.
            metadata=sum(sum(v.numel()*2 for k,v in z.items() if k in ('mu','sigma') or k.startswith(m+'_')) for z in books)+128
            scalars=len(layers)*2*8*n*128
            for b in BITS:
                ledger.append(dict(method=m,context=n,read_bits=b,payload_bytes=scalars//2,metadata_bytes=metadata,total_stored_bytes=scalars//2+metadata,logical_read_bytes=scalars*b//8,stored_bits=4,fp16_KV_tail_bytes=0,metadata_policy='FP16 arrays used by reference decoder; controller profile excluded from encoding-only state'))
    write_csv(out/'storage_ledger.csv',ledger)
    audit=dict(complete=True,requests=expected,methods=list(METHODS),attention_rows_expected=expected*len(layers)*4*(1 if smoke else 3)*len(METHODS)*8*15,baseline_checked_rows=len(baseline_check),baseline_max_absolute_output_rel_difference=max(baseline_check,default=None),calibration_only_training=True,test_used_for_design_before_this_exploratory_extension=True,validation_executed=0,controller_fixed_method='Jacobian_Tree',controller_selection_depends_on_encoding_winner=False)
    atomic(out/'audit.json',audit)
    text=['# 四位渐进编码补充实验结果','',f'已完成 {expected} 条测试请求。全部候选完整存储 4-bit，读取 2/3/4-bit；没有 FP16 KV 尾部。统计先在请求内平均种子、层、head 和位置，再在请求间汇总。','', 'Residual211 与 LogResidual211 是论文思想适配，不是 QuantSpec 或 Lynx 原生复现；统一共享预处理，不引入其局部 Page/Chunk 归一化。所有方法仍为固定轨迹单层反事实。','', '|方法|2-bit 相对 L2|3-bit 相对 L2|4-bit 相对 L2|','|---|---:|---:|---:|']
    for m in METHODS:
        vals=[next(r['output_rel'] for r in summary if r['method']==m and r['k_bits']==b and r['v_bits']==b) for b in BITS]
        text.append('|'+m+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
    text.extend(['','绝对 Attention Output MSE、P95、K-only/V-only、V 对角项/交叉项及 K/V 相互作用见 summary.csv。请求配对 bootstrap 95% CI 见 comparisons.json；总存储与逻辑读取分开计费，见 storage_ledger.csv。','',f'原基线重算核对 {len(baseline_check)} 个单元，最大相对 L2 数值差 {max(baseline_check,default=0):.3g}。','', '旧测试数据已用于方案设计，本轮为探索性复用评估。不同编码的训练目标仍不同，不能将 Jacobian 相对 MSE 编码的差异全部解释为编码结构收益。','', '下一阶段固定 Jacobian_Tree，即 K 使用原 D Jacobian 树、V 使用原 D attention 树；不按本轮赢家替换。'])
    (out/'结果分析.md').write_text('\n'.join(text)+'\n')
    if not smoke:
        sys.path.insert(0,str(SUPP/'deps'))
        os.environ['MPLCONFIGDIR']=str(ROOT/'logs/matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for m in METHODS:
            for ax,field in zip(axes,('output_mse','output_rel')):
                ax.plot(BITS,[next(r[field] for r in summary if r['method']==m and r['k_bits']==b and r['v_bits']==b) for b in BITS],marker='o',label=m)
        for ax,title in zip(axes,('Attention output MSE','Attention output relative L2')):
            ax.set_title(title);ax.set_xlabel('Read bits');ax.set_yscale('log');ax.grid(alpha=.2)
        axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'encoding_comparison.png',dpi=180);plt.close(fig)
    atomic(out/'analysis_complete.json',audit)
    state('encoding_analysis_completed',requests=expected,smoke=smoke)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');analyze(p.parse_args().smoke)
