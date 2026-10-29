"""Post-run A-E index and merged metrics; never alter frozen source results."""
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
R=Path(__file__).resolve().parents[1];E=R/'E_extension';O=R/'results/unified';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'inputs/manifest.json').read_text());test={r['case_id'] for r in m if r['split']=='test'};evaluate={r['case_id'] for r in m if r['split']!='calibration'}
assert json.loads((E/'background_exit.json').read_text())['exit_code']==0
provenance=json.loads((E/'provenance.json').read_text())
assert sha(R/'inputs/manifest.json')==provenance['frozen_input_manifest_sha256']
assert sha(R/'results/full/request_metrics.csv')==provenance['frozen_ABCD_request_metrics_sha256']
for case in evaluate:
 n=case.replace('/','_')
 for kind in ('attention','distortion'):assert (E/f'results/full/{n}_{kind}.csv.gz').exists()
 a=json.loads((E/f'results/full/{n}_trajectory.json').read_text());b=json.loads((R/f'results/full/{n}_trajectory.json').read_text());assert a['sha256']==b['sha256']
for name,digest in provenance['sha256'].items():
 if name.startswith(('src/','references/')):assert sha(E/name)==digest,name
ledger=json.loads((E/'results/full/storage_ledger.json').read_text());assert all(x['equal_total_bytes']<=x['B_total_budget_bytes'] for x in ledger)
rows=[];seen=set();groups=collections.defaultdict(list)
for origin,p in [('ABCD',R/'results/full/request_metrics.csv'),('E',E/'results/full/request_metrics.csv')]:
 with p.open() as f:
  for row in csv.DictReader(f):
   assert row['case'] in test
   key=(row['case'],row['method'],int(row['k_bits']),int(row['v_bits']));assert key not in seen;seen.add(key)
   x=float(row['output_rel']);assert np.isfinite(x)
   result=dict(row,source=origin,storage_condition='equal_total_bytes' if row['method'].endswith('_eq') else 'original_payload',method_base=row['method'].removesuffix('_eq'))
   rows.append(result);groups[key[1:]].append(x)
assert len(rows)==90*12*15,len(rows)
assert all(len(v)==90 for v in groups.values())
with (O/'request_metrics_A_to_E.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[dict(method=k[0],k_bits=k[1],v_bits=k[2],n_requests=len(v),output_rel_mean=float(np.mean(v)),output_rel_p95=float(np.percentile(v,95))) for k,v in sorted(groups.items())]
with (O/'summary_A_to_E.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
comparisons={label:json.loads(p.read_text()) for label,p in [('ABCD',R/'results/full/comparisons.json'),('E',E/'results/full/comparisons.json')]}
(O/'comparisons_A_to_E.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
audit=dict(completed=True,evaluated_E_requests=len(evaluate),paired_test_requests=len(test),trajectory_hashes_match=True,E_sources_unchanged=True,ABCD_metrics_unchanged=True,input_manifest_unchanged=True,equal_E_budgets_verified=True,merged_request_rows=len(rows),summary_cells=len(summary),note='File presence and merged values verified; no rerun of GPU measurements.')
(O/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
config=dict(kind='post_run_unified_index_not_replacement_of_frozen_run_configuration',model='Qwen/Qwen3-4B',revision='1cfa9a7208912126459214e8b04321603b3df60c',requests=dict(calibration=60,validation=30,test=90),bits=[2,3,4],decode_positions=[1,8,32,64],methods=dict(A='independent uniform INT4, shared calibrated ranges',B='Gaussian Lloyd tree',C='KV empirical weighted hierarchical clustering',D='attention-output Jacobian weighted hierarchical clustering',E='native KVarN K4V4/G128, 8 Sinkhorn iterations, original FP16 metadata; midpoint missing-bit reconstruction',E_zero='same native codes; zero-fill missing bits'),conditions=dict(original_payload='native metadata charged; E preserves incomplete FP16 tail, A-D quantize all valid tokens',equal_total_bytes='same B total-byte upper bound; fixed low-plane omission; E may retain fewer than requested prefix bits'),sources=dict(ABCD_protocol='PROTOCOL.md',E_protocol='E_extension/PROTOCOL_E.md',ABCD_provenance='provenance.json',E_provenance='E_extension/provenance.json'),results=dict(main_report='结果分析.md',E_report='E_extension/E对比分析.md',merged_request_metrics='results/unified/request_metrics_A_to_E.csv',merged_summary='results/unified/summary_A_to_E.csv',comparisons='results/unified/comparisons_A_to_E.json',audit='results/unified/audit.json'),limitations=['E versus A-D includes quantizer preprocessing and incomplete-tail-policy differences','equal-byte measurements simulate compact serializable layout, not native serving peak GPU allocation','single-layer counterfactual quality; no cross-layer propagation or task-quality guarantee'])
(R/'config_A_to_E.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
lines=['# A–E 完成后的统一结果入口','', '两次运行均已正常完成。详细条件见顶层config_A_to_E.json；原始运行配置与结果保持冻结。','', '|方案（严格等总字节）|2-bit输出相对L2|3-bit|4-bit|','|---|---:|---:|---:|']
for method in ('A_eq','B_eq','C_eq','D_eq','E_eq','E_zero_eq'):
 a=[next(r['output_rel_mean'] for r in summary if r['method']==method and r['k_bits']==b and r['v_bits']==b) for b in (2,3,4)]
 lines.append('|'+method+'|'+ '|'.join(f'{x:.6f}' for x in a)+'|')
lines+=['','E中点恢复明显优于补零。E_eq在本轮输出误差上优于B/C/D，但E使用原生分块均衡、非共享的原生Scale/零点、不同旋转与FP16尾块；这不是只改变码本的一项严格单变量消融，不能据此单独归因于前缀树结构或直接推断任务质量。', '', '机器可读文件：request_metrics_A_to_E.csv（16200行），summary_A_to_E.csv（180个精度组合单元），comparisons_A_to_E.json，audit.json。']
(O/'统一结果摘要.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(audit,ensure_ascii=False))
