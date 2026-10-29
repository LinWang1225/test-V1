"""Report matched-budget quality, proxy calibration and trajectory diagnostics."""
import sys,csv,gzip,collections,argparse
from pathlib import Path
CROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CROOT.parent/'encoding_extension_20261026/src'))
from common import *


def ranks(a):
    a=np.asarray(a);order=np.argsort(a,kind='stable');out=np.zeros(len(a));i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out


def analyze(smoke=False,online=False):
    folder=CROOT/'results'/('smoke_online' if smoke else 'online') if online else CROOT/'results'/('smoke_replay' if smoke else 'replay')
    done=json.loads((folder/'complete.json').read_text());assert done['complete']
    if online:
        data=done['results'];groups=collections.defaultdict(list)
        for r in data:groups[r['policy']].append(r)
        text=['# Jacobian 编码在线闭环试验','',f"完成 {done['cases']} 个请求，每策略生成 {done['steps']} 步。所有 decode 层的量化 attention 输出均传入后续层和下一步，prefill 保持 FP16。",'', '|策略|与 FP16 轨迹的位置一致率|采样局部 MSE|','|---|---:|---:|']
        for policy,rows in groups.items():text.append(f"|{policy}|{np.mean([r['token_position_agreement'] for r in rows]):.6f}|{np.mean([r['sampled_local_output_mse'] for r in rows]):.6g}|")
        text.extend(['','位置一致率衡量轨迹分歧，不是任务准确率。64 步结果仅为闭环 pilot，未做官方任务评分。','', '该参考适配器保留 FP16 源缓存、重算编码并模拟前缀选择，包含 CPU 分配与同步。因此墙钟、TPOT、峰值显存只用于检查实现开销，不能当作压缩内核、显存流量或服务加速证据。'])
        (folder/'结果分析.md').write_text('\n'.join(text)+'\n')
        atomic(folder/'analysis_complete.json',dict(complete=True,stage='online_pilot',cases=done['cases'],official_task_scores=False,compressed_kernel=False))
        return
    fields=('output_mse','output_rel','score_mse','k_actual_diag_mse','v_exact_mse','v_actual_diag_mse','v_cross_mse','k_output_mse','k_predicted_score_mse','v_predicted_mse','logical_payload_read_bytes','padded_plane_read_bytes','actual_mean_bits','K_policy_cpu_ms','V_policy_cpu_ms')
    requests=[];rank_records=[];ratios=collections.defaultdict(list);pairwise=collections.defaultdict(lambda:[0,0])
    paths=sorted(folder.glob('*_attention.csv.gz'))
    assert len(paths)==done['requests']
    for path in paths:
        sums=collections.defaultdict(lambda:np.zeros(len(fields)+1));schedules=collections.defaultdict(list)
        with gzip.open(path,'rt') as f:
            for r in csv.DictReader(f):
                target=float(r['target_mean_bits']);key=(r['case'],r['dataset'],r['policy'],target)
                a=np.array([float(r[n]) for n in fields]+[1.]);assert np.isfinite(a).all();sums[key]+=a
                schedules[r['case'],int(r['layer']),int(r['position']),int(r['head']),target].append(r)
                pred=float(r['v_predicted_mse']);actual=float(r['v_exact_mse'])
                if pred>1e-30:ratios[r['policy'],target].append(actual/pred)
        for k,a in sums.items():requests.append(dict(zip(('case','dataset','policy','target_mean_bits'),k),**dict(zip(fields,map(float,a[:-1]/a[-1])))))
        for k,rows in schedules.items():
            pred=np.array([float(r['v_predicted_mse']) for r in rows]);true=np.array([float(r['v_exact_mse']) for r in rows])
            rp,rt=ranks(pred),ranks(true)
            if rp.std()>0 and rt.std()>0:
                rank_records.append(dict(case=k[0],layer=k[1],position=k[2],head=k[3],target_mean_bits=k[4],v_proxy_spearman=float(np.corrcoef(rp,rt)[0,1])))
            for i in range(len(rows)):
                for j in range(i):
                    a=pred[i]-pred[j];b=true[i]-true[j]
                    if abs(a)>1e-15 and abs(b)>1e-15:
                        v=pairwise[k[4]];v[0]+=int(a*b>0);v[1]+=1
    write_csv(folder/'request_metrics.csv',requests)
    grouped=collections.defaultdict(list)
    for r in requests:grouped[r['policy'],r['target_mean_bits']].append(r)
    summary=[];comparisons=[]
    for k,rows in sorted(grouped.items()):
        assert len(rows)==done['requests']
        rv=ratios[k]
        summary.append(dict(policy=k[0],target_mean_bits=k[1],requests=len(rows),**{n:float(np.mean([r[n] for r in rows])) for n in fields},output_mse_p95=float(np.percentile([r['output_mse'] for r in rows],95)),v_exact_to_predicted_median=float(np.median(rv)) if rv else 0.,v_exact_to_predicted_p95=float(np.percentile(rv,95)) if rv else 0.))
    rng=np.random.default_rng(20261026)
    for target in (2.5,3.,3.5):
        candidate={r['case']:r for r in grouped['query_curve',target]}
        for baseline in ('uniform','static','query_only'):
            reference={r['case']:r for r in grouped[baseline,target]}
            cases=sorted(candidate);a=np.array([candidate[c]['output_mse'] for c in cases]);b=np.array([reference[c]['output_mse'] for c in cases])
            ix=rng.integers(0,len(cases),(2000,len(cases)));effects=1-a[ix].mean(1)/np.maximum(b[ix].mean(1),1e-30)
            comparisons.append(dict(policy='query_curve',baseline=baseline,target_mean_bits=target,metric='output_mse',improvement=float(1-a.mean()/max(b.mean(),1e-30)),ci95=np.percentile(effects,[2.5,97.5]).tolist(),n_requests=len(cases)))
            assert all(candidate[c]['padded_plane_read_bytes']==reference[c]['padded_plane_read_bytes'] for c in cases)
    write_csv(folder/'summary.csv',summary);atomic(folder/'comparisons.json',comparisons)
    if rank_records:write_csv(folder/'proxy_rank_correlations.csv',rank_records)
    atomic(folder/'proxy_pairwise_ranking.json',[dict(target_mean_bits=k,correct=v[0],total=v[1],accuracy=v[0]/v[1]) for k,v in pairwise.items() if v[1]])
    text=['# Jacobian 编码因果回放实验结果','',f"完成 {done['requests']} 条固定参考轨迹，在第 0、17、35 层与 decode 1、64 比较四种策略。编码固定原始 D：K Jacobian 树与 V attention 树，使用首校准种子。",'', '决策器只接收当前 Q、冻结物理尺度失真曲线和重建 K 产生的 attention。真实 K/V、原始 attention 和真实误差只供评估器使用。参考轨迹与旧 FP16 采集逐请求 hash 一致。','', '|策略|平均读取位数目标|真实输出 MSE|相对 L2|','|---|---:|---:|---:|']
    for row in summary:text.append(f"|{row['policy']}|{row['target_mean_bits']}|{row['output_mse']:.6g}|{row['output_rel']:.6f}|")
    text.extend(['','相同目标预算的策略已核对逻辑读取与逐行字节填充成本一致。请求配对置信区间见 comparisons.json；代理排序与幅值偏差分开保存。V 交叉项保留符号，不能用代理值作为绝对误差保证。','', '这里只模拟位平面选择，不验证实际 HBM 流量。不同单位的额外位需满足前置依赖；非收益递减曲线的贪心不保证全局最优。local_oracle.json 只枚举单个 head 的四单位局部问题。','', '在线闭环另行在所有层执行，先通过小规模门槛，再完成六任务各两条共 12 条的 64 步 pilot。正式任务评分、融合位平面内核和服务性能属于后续阶段。'])
    (folder/'结果分析.md').write_text('\n'.join(text)+'\n')
    atomic(folder/'analysis_complete.json',dict(complete=True,stage='causal_replay',requests=done['requests'],same_budget_checked=True,source_trajectory_hash_checked=True,proxy_magnitude_not_assumed_exact=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--online',action='store_true');a=p.parse_args();analyze(a.smoke,a.online)
