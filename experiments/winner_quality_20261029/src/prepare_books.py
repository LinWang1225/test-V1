from common import *
import sys,numpy as np,torch
sys.path.insert(0,str(ROOT.parent/'encoding_ablation_20261024/src'))
from codec import encode,decode

def main():
 torch.set_num_threads(4)
 base=ROOT.parent/'encoding_ablation_20261024'
 manifest=json.loads((base/'inputs/manifest.json').read_text())
 files=[base/'calibration/full'/(r['case_id'].replace('/','_')+'.npz') for r in manifest if r['split']=='calibration'];assert len(files)==60
 for layer in range(36):
  target=BOOKS/f'layer{layer}_seed20261024.npz'
  if target.exists():continue
  source=base/'supplementary_experiments/codebooks/full'/target.name
  with np.load(source) as z:
   output={k:z[k] for k in ('mu','sigma')}
   for name in ('t2','t3','t4','c2','c3','c4'):
    output['Jacobian_Tree_'+name]=np.stack([z['0_tree_jac_'+name],z['1_tree_mix_'+name]])
  # Retain loader schema; these arrays contain JacK_MixV, not original D-V.
  mu=torch.tensor(output['mu']);sigma=torch.tensor(output['sigma']);mass=0.;ds=np.zeros((2,8,128,3),np.float64)
  counts=np.bincount(np.random.default_rng(20261024+layer).integers(0,60,60),minlength=60)
  for i,path in enumerate(files):
   if not counts[i]:continue
   with np.load(path) as z:x=torch.tensor(z[f'x{layer}'].astype(np.float32))
   standardized=(x-mu[:,:,None,:])/sigma[:,:,None,:]
   for side in (0,1):
    book={n:torch.tensor(output['Jacobian_Tree_'+n][side].astype(np.float32)) for n in ('t2','t3','t4','c2','c3','c4')}
    codes=encode(standardized[side],book)
    for j,b in enumerate((2,3,4)):
     rec=decode(codes,book,b)*sigma[side,:,None,:]+mu[side,:,None,:]
     ds[side,...,j]+=(rec-x[side]).double().square().mean(1).numpy()*counts[i]
   mass+=counts[i]
  output['controller_D']=ds/mass
  assert np.isfinite(output['controller_D']).all()
  with target.with_suffix('.tmp').open('wb') as f:np.savez(f,**output)
  target.with_suffix('.tmp').replace(target)
  print('prepared',layer,flush=True)
 atomic(ROOT/'books_provenance.json',dict(method='JacK_MixV',seed=20261024,calibration_only=True,source_hashes={str(p):sha(p) for p in sorted((base/'supplementary_experiments/codebooks/full').glob('layer*_seed20261024.npz'))},prepared_hashes={str(p):sha(p) for p in sorted(BOOKS.glob('*.npz'))}))
if __name__=='__main__':main()
