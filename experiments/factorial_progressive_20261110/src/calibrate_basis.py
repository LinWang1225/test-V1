import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import hashlib,json,pathlib,time
import numpy as np
import torch
from common import ROOT,CONFIG,rows,atomic,sha,progress
from native_runner import NativeRunner
class BasisObserver:
    def start(self,row):self.row=row;self.samples={};self.seen=set()
    def __call__(self,layer,position,q,k,v,scale):
        key=(layer,position)
        assert key not in self.seen;self.seen.add(key)
        seed=20261024+layer*1000+position+int(hashlib.sha256(self.row['case_id'].encode()).hexdigest()[:8],16)
        indices=np.random.default_rng(seed).choice(k.shape[1],min(32,k.shape[1]),replace=False)
        ix=torch.tensor(indices,device=k.device)
        sample=torch.stack((k.index_select(1,ix),v.index_select(1,ix))).detach().cpu().numpy().astype(np.float16)
        self.samples.setdefault(layer,[]).append(sample)
    def finish(self,path):
        assert len(self.seen)==36*4,self.seen
        arrays={f'layer{layer}':np.concatenate(self.samples[layer],axis=2) for layer in range(36)}
        with path.with_suffix('.npz.tmp').open('wb') as f:np.savez(f,**arrays)
        path.with_suffix('.npz.tmp').replace(path)
        self.samples={}
def main():
    folder=ROOT/'calibration/basis_capture';folder.mkdir(parents=True,exist_ok=True)
    observer=BasisObserver();progress('calibration_basis','loading_native_model',requests=60)
    runner=NativeRunner(observer)
    try:
        for i,row in enumerate(rows('calibration')):
            dest=folder/f'{i:03d}.npz';meta=folder/f'{i:03d}.json'
            if dest.exists() and meta.exists():
                record=json.loads(meta.read_text());assert record['case_id']==row['case_id'] and record['data_sha256']==sha(dest)
                continue
            observer.start(row)
            progress('calibration_basis','capturing',request=i+1,total=60,case_id=row['case_id'])
            result=runner.generate(row,calibration=True,job_name='calibration_basis',run_id='calibration_fp16')
            assert result['actual_decode_steps']==64,result['actual_decode_steps']
            observer.finish(dest)
            atomic(meta,{'case_id':row['case_id'],'input_sha256':hashlib.sha256(np.asarray(row['input_ids'],np.int32).tobytes()).hexdigest(),'generated_ids':result['generated_ids'],'data_sha256':sha(dest),'observations':144,'sampled_positions':[1,8,32,64],'samples_per_observation':32})
            progress('calibration_basis','captured',completed=i+1,total=60,case_id=row['case_id'])
    finally: runner.close()
    atomic(folder/'complete.json',{'complete':True,'requests':60,'config_sha256':sha(ROOT/'config.json')})
    progress('calibration_basis','completed',requests=60)
if __name__=='__main__':main()
