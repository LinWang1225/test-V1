import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import json,pathlib,time
import numpy as np
from common import ROOT,CONFIG,atomic,sha,progress

def wait_ready():
    start=time.monotonic()
    while not (ROOT/'calibration/basis_capture/complete.json').is_file():
        pointer=json.loads((ROOT/'jobs/calibration_basis.json').read_text())
        p=pathlib.Path(pointer['job_dir'])/'state.json'
        if p.is_file() and json.loads(p.read_text())['status']=='failed':raise RuntimeError('Basis capture failed')
        if time.monotonic()-start>7200:raise TimeoutError('Basis capture prerequisite timeout')
        time.sleep(10)
def hadamard():
    h=np.ones((1,1),np.float32)
    while len(h)<128:h=np.block([[h,h],[h,-h]])
    signs=np.random.default_rng(20261024).choice([-1,1],128).astype(np.float32)
    return (h*signs[:,None]/np.sqrt(128)).astype(np.float32)
def main():
    wait_ready()
    source=ROOT/'calibration/basis_capture';folder=ROOT/'calibration/basis';folder.mkdir(exist_ok=True)
    u=hadamard();np.save(folder/'hadamard.npy',u)
    reports=[]
    for layer in range(36):
        target=folder/f'layer{layer:02d}.npz'
        if target.is_file():continue
        progress('basis_fit','fitting',layer=layer)
        arrays=[]
        for i in range(60):
            with np.load(source/f'{i:03d}.npz') as a:arrays.append(a[f'layer{layer}'].astype(np.float64))
        x=np.concatenate(arrays,axis=2)
        mean=x.mean(2);centered=x-mean[:,:,None,:]
        cov=np.einsum('shnc,shnd->shcd',centered,centered,optimize=True)/x.shape[2]
        values,vectors=np.linalg.eigh(cov)
        vectors=vectors[...,::-1].copy()
        for side in range(2):
            for head in range(8):
                b=vectors[side,head]
                piv=np.argmax(np.abs(b),axis=0)
                b*=np.where(b[piv,np.arange(128)]<0,-1.,1.)[None,:]
        b2=vectors.astype(np.float32)
        data={'B2':b2,'raw_mean':mean.astype(np.float32),'raw_covariance':cov.astype(np.float32)}
        for name,b in [('B0',np.eye(128)),('B1',u),('B2',b2)]:
            rotated_mean=np.einsum('shc,...cd->shd',mean,b) if b.ndim==2 else np.einsum('shc,shcd->shd',mean,b)
            if b.ndim==2:variance=np.einsum('ci,shcd,di->shi',b,cov,b,optimize=True)
            else:variance=np.einsum('shci,shcd,shdi->shi',b,cov,b,optimize=True)
            data['mu_'+name]=rotated_mean.astype(np.float16)
            data['sigma_'+name]=np.sqrt(np.maximum(variance,0)).clip(1e-4).astype(np.float16)
        ortho=np.max(np.abs(np.matmul(b2.swapaxes(-1,-2),b2)-np.eye(128)))
        assert ortho<2e-5,ortho
        with target.with_suffix('.npz.tmp').open('wb') as f:np.savez(f,**data)
        target.with_suffix('.npz.tmp').replace(target)
        reports.append({'layer':layer,'samples_per_head':x.shape[2],'klt_max_orthogonality_error':float(ortho),'sha256':sha(target)})
        del arrays,x,centered,cov
    assert len(list(folder.glob('layer*.npz')))==36
    atomic(folder/'complete.json',{'complete':True,'layers':36,'calibration_requests':60,'config_sha256':sha(ROOT/'config.json'),'reports':reports,'hadamard_sha256':sha(folder/'hadamard.npy')})
    progress('basis_fit','completed',layers=36)
if __name__=='__main__':main()
