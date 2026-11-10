import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import hashlib,json,pathlib,subprocess,time
import numpy as np
import torch
from common import ROOT,CONFIG,rows,atomic,sha,progress
from native_runner import NativeRunner
from native_backend import STATE
from transforms import normalization_stats
class FitObserver:
    def __init__(self):
        self.assets={};self.vgrams={}
        self.u=torch.tensor(np.load(ROOT/'calibration/basis/hadamard.npy'),device='cuda',dtype=torch.float32)
        for layer in range(36):
            with np.load(ROOT/'calibration/basis'/f'layer{layer:02d}.npz') as a:
                self.assets[layer]={k:torch.tensor(a[k].astype(np.float32),device='cuda') for k in a.files if k.startswith(('mu_','sigma_')) or k=='B2'}
    def start(self,row):self.row=row;self.data={};self.energy={};self.seen=set()
    def add(self,layer,key,array):
        self.data.setdefault(layer,{}).setdefault(key,[]).append(array.detach().cpu().numpy().astype(np.float32))
    def basis(self,layer,name,side):
        if name=='B0':return None
        if name=='B1':return self.u
        return self.assets[layer]['B2'][side]
    def rot(self,x,b):return x if b is None else torch.matmul(x,b)
    def vgram(self,layer,name,wo):
        key=(layer,name)
        if key not in self.vgrams:
            b=self.basis(layer,name,1)
            transformed=wo if b is None else torch.matmul(wo,b if b.ndim==2 else b[:,None,:,:])
            self.vgrams[key]=torch.einsum('hgoc,hroc->hgrc',transformed,transformed)
        return self.vgrams[key]
    def __call__(self,layer,position,q,k,v,scale):
        key=(layer,position)
        assert key not in self.seen;self.seen.add(key)
        q=q.float();k=k.float();v=v.float()
        score=torch.einsum('hgc,htc->hgt',q,k)*scale
        p=score.softmax(-1);o=p@v
        seed=20261024+layer*1000+position+int(hashlib.sha256(self.row['case_id'].encode()).hexdigest()[:8],16)
        indices=np.random.default_rng(seed).choice(k.shape[1],min(32,k.shape[1]),replace=False)
        ix=torch.tensor(indices,device=k.device)
        ps=p.index_select(2,ix)
        vs=v.index_select(1,ix)
        delta=vs[:,None,:,:]-o[:,:,None,:]
        attention_weight=ps.square().sum(1)
        self.add(layer,'attention_weight',attention_weight)
        # Full output projection across sibling query heads; calibration only.
        weight=STATE.weight_provider(layer).float()
        wo=weight.reshape(weight.shape[0],8,4,128).permute(1,2,0,3).contiguous()
        projected_delta=torch.einsum('hgsd,hgod->hgso',delta,wo)
        gram=torch.einsum('hgso,hrso->hsgr',projected_delta,projected_delta)
        ps_t=ps.transpose(1,2)
        gram=gram*ps_t[:,:,:,None]*ps_t[:,:,None,:]*(scale**2)
        delta_norm=delta.square().sum(-1)
        for name in ['B0','B1','B2']:
            bk=self.basis(layer,name,0);bv=self.basis(layer,name,1)
            qr=self.rot(q,bk)
            old_k=torch.einsum('hgs,hgc->hsc',ps.square()*delta_norm,qr.square())*(scale**2)
            denominator=qr.square().sum(1)*(scale**2)
            projected_k=torch.einsum('hgc,hsgr,hrc->hsc',qr,gram,qr).clamp_min(0)
            jac_k=projected_k/denominator[:,None,:].clamp_min(1e-20)
            jac_v=torch.einsum('hgs,hgrc,hrs->hsc',ps,self.vgram(layer,name,wo),ps).clamp_min(0)
            jac_v=jac_v/attention_weight[:,:,None].clamp_min(1e-20)
            for label,array in [('oldK',old_k),('jacK',jac_k),('jacV',jac_v)]:self.add(layer,label+'_'+name,array)
            rotated=[self.rot(k,bk),self.rot(v,bv)]
            for reconstruction in ['R0','R1','R2','R3']:
                zs=[];scales=[];energies=[]
                for side in range(2):
                    mu,sigma,energy,group=normalization_stats(rotated[side],self.assets[layer]['mu_'+name][side],self.assets[layer]['sigma_'+name][side],reconstruction,side)
                    which=ix//group
                    selected_mu=mu.index_select(1,which)
                    selected_sigma=sigma.index_select(1,which)
                    z=(rotated[side].index_select(1,ix)-selected_mu)/selected_sigma
                    assert torch.isfinite(z).all()
                    zs.append(z);scales.append(selected_sigma);energies.append(energy.index_select(1,which).mean(1))
                prefix=name+'_'+reconstruction
                self.add(layer,prefix+'_z',torch.stack(zs))
                self.add(layer,prefix+'_scale',torch.stack(scales))
                self.energy.setdefault(layer,{}).setdefault(prefix,[]).append(torch.stack(energies).cpu().numpy())
    def finish(self,index):
        assert len(self.seen)==144,len(self.seen)
        receipts=[]
        for layer in range(36):
            folder=ROOT/'calibration/normalized'/f'layer{layer:02d}';folder.mkdir(parents=True,exist_ok=True)
            target=folder/f'{index:03d}.npz'
            arrays={}
            for key,values in self.data[layer].items():
                axis=2 if key.endswith(('_z','_scale')) else 1
                arrays[key]=np.concatenate(values,axis=axis)
            for prefix,values in self.energy[layer].items():arrays[prefix+'_energy_mean']=np.stack(values).mean(0)
            with target.with_suffix('.npz.tmp').open('wb') as f:np.savez(f,**arrays)
            target.with_suffix('.npz.tmp').replace(target)
            receipts.append({'path':str(target.relative_to(ROOT)),'sha256':sha(target),'bytes':target.stat().st_size})
            del arrays
        self.data={};self.energy={}
        return receipts
def wait_ready():
    started=time.monotonic()
    while not (ROOT/'calibration/basis/complete.json').is_file():
        pointer=json.loads((ROOT/'jobs/basis_fit.json').read_text())
        state=pathlib.Path(pointer['job_dir'])/'state.json'
        if state.is_file() and json.loads(state.read_text())['status']=='failed':raise RuntimeError('Basis fitting failed')
        if time.monotonic()-started>7200:raise TimeoutError('Basis fitting prerequisite timeout')
        time.sleep(10)
    for attempt in range(60):
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        if not p.stdout.strip():return
        time.sleep(2)
    raise RuntimeError('GPU remains busy before calibration second pass')
def main():
    wait_ready();folder=ROOT/'calibration/normalized';folder.mkdir(exist_ok=True)
    observer=FitObserver();progress('calibration_fit_capture','loading_native_model')
    runner=NativeRunner(observer)
    try:
        for index,row in enumerate(rows('calibration')):
            receipt=folder/f'case{index:03d}.json'
            if receipt.exists():
                previous=json.loads(receipt.read_text());assert previous['case_id']==row['case_id']
                assert all(sha(ROOT/r['path'])==r['sha256'] for r in previous['files'])
                continue
            observer.start(row);progress('calibration_fit_capture','capturing',request=index+1,total=60,case_id=row['case_id'])
            result=runner.generate(row,calibration=True,job_name='calibration_fit_capture',run_id='fit_capture_fp16')
            original=json.loads((ROOT/'calibration/basis_capture'/f'{index:03d}.json').read_text())
            assert result['generated_ids']==original['generated_ids'],'Calibration trajectories changed between basis and fit capture'
            receipts=observer.finish(index)
            atomic(receipt,{'case_id':row['case_id'],'files':receipts,'generated_ids':result['generated_ids'],'config_sha256':sha(ROOT/'config.json')})
            progress('calibration_fit_capture','captured',completed=index+1,total=60,case_id=row['case_id'])
    finally:runner.close()
    atomic(folder/'complete.json',{'complete':True,'requests':60,'config_sha256':sha(ROOT/'config.json'),'all_B_R_captured':12,'same_teacher_trajectory':True})
    progress('calibration_fit_capture','completed',requests=60)
if __name__=='__main__':main()
