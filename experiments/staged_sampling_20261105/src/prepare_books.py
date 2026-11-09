"""Reuse frozen 2/3/4 trees; fit 0/1 prefix reconstructions on calibration only."""
import argparse
import numpy as np
import torch
from common import *
from prefix_codec import encode, make_lut

METHOD_KEYS = {'Gaussian_Tree':('tree_gaussian','tree_gaussian'),
    'Data_Tree':('tree_mse','tree_mse'), 'Jacobian_Tree':('tree_jac','tree_attn'),
    'JacK_MixV':('tree_jac','tree_mix'), 'U_Clipped':('uniform_mse','uniform_mse')}

def prepare(layers):
    torch.set_num_threads(4)
    manifest=json.loads((BASE/'inputs/manifest.json').read_text())
    files=[BASE/'calibration/full'/(r['case_id'].replace('/','_')+'.npz') for r in manifest if r['split']=='calibration']
    assert len(files)==60
    calibration_hashes={p.name:sha(p) for p in files}
    for layer in layers:
        source=BASE/'supplementary_experiments/codebooks/full'/f'layer{layer}_seed20261024.npz'
        provenance=dict(source_sha256=sha(source),calibration_hashes=calibration_hashes,
            training_code_sha256=sha(__file__),prefix_code_sha256=sha(ROOT/'src/prefix_codec.py'),
            seed=20261024,layer=layer,method='c0=offline mean; c1=conditional standardized means within existing MSB groups')
        fingerprint=digest(provenance)
        if all((ROOT/'codebooks'/m/f'layer{layer}.npz').exists() and
               (ROOT/'codebooks'/m/f'layer{layer}.json').exists() and
               json.loads((ROOT/'codebooks'/m/f'layer{layer}.json').read_text())['fingerprint']==fingerprint for m in CODECS):
            continue
        print('prepare_calibration_layer',layer,flush=True)
        with np.load(source) as a:
            frozen={k:a[k] for k in a.files}
        X=[];W=[]
        for p in files:
            with np.load(p) as a:
                X.append(a[f'x{layer}'].astype(np.float32));W.append(a[f'w{layer}'].astype(np.float32))
        x=np.stack(X);weights=np.stack(W).astype(np.float64)
        counts=np.bincount(np.random.default_rng(20261024+layer).integers(0,60,60),minlength=60)
        mass=float(counts.sum())
        mu=frozen['mu'].astype(np.float32);sigma=frozen['sigma'].astype(np.float32)
        z=(x-mu[None,:,:,None,:])/sigma[None,:,:,None,:]
        # Merge requests and tokens for encoding, keep heads/channels distinct.
        zt=torch.tensor(z.transpose(1,2,0,3,4).reshape(2,8,-1,128))
        for method in CODECS:
            target=ROOT/'codebooks'/method/f'layer{layer}.npz'
            target.parent.mkdir(parents=True,exist_ok=True)
            output=dict(mu=frozen['mu'],sigma=frozen['sigma'])
            code=[]
            for side,key in enumerate(METHOD_KEYS[method]):
                keys=('lo','hi') if method=='U_Clipped' else ('t2','t3','t4','c2','c3','c4')
                sb={k:torch.tensor(frozen[f'{side}_{key}_{k}'].astype(np.float32)) for k in keys}
                code.append(encode(zt[side],sb).numpy().reshape(8,60,128,128).transpose(1,0,2,3))
                for k in keys:
                    output.setdefault(k,[None,None])[side]=frozen[f'{side}_{key}_{k}']
            code=np.stack(code,axis=1) # request, side, head, token, channel
            c1=np.zeros((2,8,2),np.float32)
            for side in (0,1):
                for h in range(8):
                    for branch in (0,1):
                        w=(code[:,side,h]>>3==branch)*counts[:,None,None]
                        c1[side,h,branch]=(z[:,side,h]*w).sum()/max(w.sum(),1)
            output={k:np.asarray(v) for k,v in output.items()}
            output['c0']=np.zeros((2,8,1),np.float16)
            output['c1']=c1.astype(np.float16)
            lut=np.stack([make_lut({k:torch.tensor(v[s].astype(np.float32)) for k,v in output.items()}).numpy() for s in (0,1)])
            D=[];J=[]
            normal_weight=weights*counts[:,None,None,None,None]
            wsum=normal_weight.sum(axis=(0,3))
            for b in range(5):
                rec=np.empty_like(x)
                for side in (0,1):
                    for h in range(8):
                        rec[:,side,h]=lut[side,h,b][code[:,side,h]]*sigma[side,h]+mu[side,h]
                err=(rec.astype(np.float64)-x)**2
                D.append((err.mean(3)*counts[:,None,None,None]).sum(0)/mass)
                # Calibration-only weighted MSE, normalized per channel.
                # It is a risk proxy; multiplying by current q² may overlap sensitivity.
                J.append(np.divide((err*normal_weight).sum(axis=(0,3)),wsum,
                    out=np.zeros_like(wsum),where=wsum>0))
            output['D']=np.stack(D,-1)
            output['J']=np.stack(J,-1)
            assert output['D'].shape==(2,8,128,5)
            assert all(np.isfinite(a).all() for a in output.values())
            tmp=target.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:
                np.savez(f,**output)
            tmp.replace(target)
            atomic(target.with_suffix('.json'),dict(fingerprint=fingerprint,**provenance,
                codebook_sha256=sha(target),stored_array_bytes={k:v.nbytes for k,v in output.items()},
                negative_marginal_count=int((np.diff(output['D'],axis=-1)>0).sum())))
            print('prepared',method,layer,flush=True)
    if list(layers)==list(range(36)):
        atomic(ROOT/'codebooks/ready.json',dict(complete=True,methods=CODECS,layers=36,depths=list(range(5)),calibration_cases=60))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--layers',type=int,nargs='+')
    args=p.parse_args();prepare(range(36) if args.layers is None else args.layers)
