"""Meaningful numerical, prefix, packed-layout and DP gates before full run."""
import json,numpy as np,torch
from pathlib import Path
from codec import *
rng=np.random.default_rng(42);x=rng.normal(size=(2,512,128)).astype(np.float32)
b=fit_tree(None,None,True);bt={k:torch.tensor(np.stack([v,v]).astype(np.float32)) for k,v in b.items()}
z=torch.tensor(x);c=encode(z,bt);dist=[]
for n in (2,3,4):
 y=decode(c,bt,n);dist.append(float(((y-z)**2).mean()))
 assert torch.isfinite(y).all()
assert dist[0]>dist[1]>dist[2]>0
assert dist[0]-dist[1]>dist[1]-dist[2]
assert np.allclose(dist,[.1175,.03484,.009796],rtol=.1)
a=rng.integers(0,16,32768,dtype=np.uint8);assert np.array_equal(a,unpack4(pack4(a)))
for omit in (0,32,688):
 ps=pack_planes(a,omit);back=unpack_planes(ps,len(a),omit)
 assert np.array_equal(back[omit:],a[omit:])
 assert np.array_equal(back[:omit]>>1,a[:omit]>>1)
 assert sum(p.nbytes for p in ps)==len(a)//2-omit//8
# Counterexample: a greedy choice can fail when marginal gains increase.
d=np.array([[100,100,0],[2,1,0]],float)
assert allocation_check(d)[2]['gap']>0
# Check output-sensitive K Jacobian in the quantizer's rotated coordinate system.
q=torch.randn(4,8,dtype=torch.float64);k=torch.randn(12,8,dtype=torch.float64,requires_grad=True);v=torch.randn(12,8,dtype=torch.float64)
p=(q@k.T/np.sqrt(8)).softmax(-1);o=p@v
jac=torch.autograd.functional.jacobian(lambda kk:((q@kk.T/np.sqrt(8)).softmax(-1)@v),k)
actual=jac.square().sum((0,1));formula=torch.einsum('gt,gc->tc',p.square()*((v[None]-o[:,None])**2).sum(-1),q.square())/8
assert torch.allclose(actual,formula,rtol=1e-10,atol=1e-10)
U=hadamard();assert np.allclose(U@U.T,np.eye(128),atol=1e-6)
root=Path(__file__).resolve().parents[1]
(root/'results/correctness_checks.json').write_text(json.dumps(dict(passed=True,gaussian_distortion=dist,packing='nibble+bitplane lossless, omitted-Bit0 size verified',jacobian='analytic vs autograd float64',allocation='nonconcave counterexample detected',orthogonality=True),indent=2))
print('All correctness gates passed',dist)
