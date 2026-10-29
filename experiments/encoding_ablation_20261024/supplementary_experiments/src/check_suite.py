import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'deps'))
import numpy as np,torch
from books import *
from codec import encode,decode,pack4,unpack4
from suite import greedy_depths
rng=np.random.default_rng(20261025)
x=np.r_[rng.normal(size=100000),[-40,50]];w=np.ones_like(x)
h,e=np.histogram(x,8192);sx=np.histogram(x,e,weights=x)[0];ok=h>0;x=sx[ok]/h[ok];w=h[ok].astype(float)
u,info=clipped(x,w)
assert info['objective']<np.mean([np.average((x-uniform_decode(x,x.min(),x.max(),b))**2,weights=w) for b in BITS])
b=fit_tree(x,w);j,trace=joint(x,w,b)
assert np.isfinite(distortion(x,w,j)).all()
assert distortion(x,w,j).mean()<=distortion(x,w,b).mean()*1.002
books=[b,j,joint(x,w,b,True)[0],regularize(x,w,j)[0]]
for a in books:
 assert np.all(np.diff(boundaries(a))>=0)
 book={k:torch.tensor(v.astype(np.float32)[None]) for k,v in a.items()}
 z=torch.tensor(np.clip(rng.normal(size=(1,64,128)),-8,8).astype(np.float32));q=encode(z,book)
 for bits in BITS:
  r=decode(q,book,bits);assert torch.isfinite(r).all()
 assert np.array_equal(q.numpy().ravel(),unpack4(pack4(q.numpy())))
from codec import allocation_check
for curves in [np.array([[4.,2.,1.],[3.,1.,.5]]),np.array([[100.,100.,0.],[2.,1.,0.]])]:
 depth=greedy_depths(curves,[2.,2.5,3.,3.5,4.])
 for target,d in depth.items():assert d.sum()==round(target*len(curves))
assert allocation_check(np.array([[100.,100.,0.],[2.,1.,0.]],np.float64))[2]['gap']>0
import pyarrow as pa,pyarrow.parquet as pq
p=R/'metrics/check.parquet';pq.write_table(pa.table({'D2':[1.],'D3':[.5],'D4':[.2]}),p);assert pq.read_table(p).num_rows==1
(R/'metrics/suite_checks.json').write_text(json.dumps(dict(passed=True,uniform_clipping_improves_outlier_baseline=True,nested_joint_valid=True,prefix_roundtrip=True,float64_DP_counterexample=True,parquet_roundtrip=True),indent=2))
print('Suite numerical/encoding/DP/Parquet gates passed')
