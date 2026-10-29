"""Small numerical code-path gate, not experiment data or a quality measurement."""
import sys,json
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parents[1]
from suite import Suite,loadside,pa,pq,exact_dp_check
from codec import hadamard
from types import SimpleNamespace
x=Suite.__new__(Suite);x.stage='evaluate';x.smoke=True;x.U=torch.tensor(hadamard(),device='cuda',dtype=torch.float32);x.pos=1;x.current=dict(case_id='numeric_gate',dataset='synthetic_gate')
x.rows=[];x.distrows=[];x.curves=[];x.code_stats=[];x.dynamic_rows=[];x.dp_rows=[]
x.active=json.loads((R/'codebooks/smoke/active_methods.json').read_text())['methods'];x.suitebooks={}
with np.load(R/'codebooks/smoke/layer0_seed20261024.npz') as z:
 norm=[torch.tensor(z[n].astype(np.float32),device='cuda') for n in ('mu','sigma')];bs={}
 for pair in x.active.values():
  for side,key in enumerate(pair):
   kind,obj=key.split(':')
   if (side,key) not in bs:bs[side,key]=loadside(z,kind,obj,side)
 profiles={(side,key):z[f'{side}_profile_{key.replace(":","_")}'] for pair in x.active.values() for side,key in enumerate(pair)}
 x.suitebooks[20261024,0]=(norm,bs,profiles)
torch.manual_seed(20261025)
q=torch.randn(1,32,1,128,device='cuda',dtype=torch.float16);k=torch.randn(1,8,257,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
x.observe(SimpleNamespace(layer_idx=0),q,k,v,128**-.5)
assert len(x.rows)==len(x.active)*8*15
assert all(np.isfinite(r['output_rel']) for r in x.rows)
columns={key:np.concatenate([entry[key] for entry in x.curves]) for key in x.curves[0]}
pq.write_table(pa.table(columns),R/'metrics/observation_gate.parquet',compression='zstd')
assert pq.read_table(R/'metrics/observation_gate.parquet').num_rows==len(x.active)*128
assert all(abs(r['target_mean_bits']-r['actual_mean_bits'])<1e-10 for r in x.dynamic_rows)
assert all(r['dtype']=='float64' for r in x.dp_rows)
a=exact_dp_check(np.array([[4,2,1],[3,1,.5]],np.float64));assert all(r['gap']<=r['tolerance'] for r in a)
a=exact_dp_check(np.array([[100,100,0],[2,1,0]],np.float64));assert a[2]['gap']>0
(R/'metrics/observation_checks.json').write_text(json.dumps(dict(passed=True,active_methods=len(x.active),attention_rows=len(x.rows),dynamic_rows=len(x.dynamic_rows),unit_curve_rows=len(x.active)*128,exact_DP=True,note='Synthetic numerical code-path gate only; no result-quality claim; real model smoke remains sequential before suite.'),indent=2))
print('observation, curves, dynamic and exact-DP paths passed',len(x.rows),len(x.dp_rows))
