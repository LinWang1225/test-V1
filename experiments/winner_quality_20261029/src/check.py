from common import *
import torch
from adaptive import choose
p=torch.tensor([[1.,.2,.05],[.01,.005,.001],[0.,0.,0.]],dtype=torch.float64)
last=None
for policy in POLICIES[1:]:
 d=choose(p,policy);assert ((d>=2)&(d<=4)).all();assert d[-1]==2
 if last is not None:assert (d<=last).all()
 last=d
assert (choose(torch.zeros(4,3),'adaptive005')==2).all()
# Strict threshold recovers full4 for positive excess; loose threshold allows coarse reads.
assert choose(torch.tensor([[1.,.5,0.]]),'adaptive005').item()==4
for layer in range(36):
 import numpy as np
 with np.load(BOOKS/f'layer{layer}_seed20261024.npz') as z:
  assert z['controller_D'].shape==(2,8,128,3)
  with np.load(ROOT.parent/f'encoding_ablation_20261024/supplementary_experiments/codebooks/full/layer{layer}_seed20261024.npz') as a:
   for name in ('t2','t3','t4','c2','c3','c4'):
    assert np.array_equal(z['Jacobian_Tree_'+name][0],a['0_tree_jac_'+name])
    assert np.array_equal(z['Jacobian_Tree_'+name][1],a['1_tree_mix_'+name])
print('adaptive and winner book gates passed')
