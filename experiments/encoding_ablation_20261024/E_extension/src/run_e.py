"""Append native KVarN prefix baseline E to the frozen A-D experiment."""
import os,sys,json,time,gzip,argparse,hashlib,traceback
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1];BASE=HERE.parent
sys.path.insert(0,str(BASE/'src'))
import run as base
from codec import allocation_check
from native import NativeCodec

def status(phase,**kw):
    base.atomic(HERE/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(phase,kw,flush=True)
base.status=status

class EExperiment(base.Experiment):
    def __init__(self,smoke=False):
        super().__init__(smoke)
        self.out=HERE/'results'/('smoke' if smoke else 'full');self.out.mkdir(exist_ok=True)
        self.native=NativeCodec();self.ledger={}
    def observe(self,module,q,k,v,scale):
        layer=module.layer_idx
        if self.smoke and layer not in (0,17,35):return
        rawk,rawv=k[0],v[0];K=rawk.float()@self.U;V=rawv.float()@self.U
        Q=(q[0,:,0].float()@self.U).view(8,4,128)
        S=torch.einsum('hgd,htd->hgt',Q,K)*scale;P=S.softmax(-1);O=P@V
        data=self.native.quantize(rawk,rawv);n=K.shape[1]
        led=self.native.storage(n);self.ledger[n]=led
        core=dict(case=self.current['case_id'],dataset=self.current['dataset'],split=self.current['split'],layer=layer,position=self.pos,context=n)
        # Reconstruction of native FP16 metadata; no new auxiliary data or fitting.
        for method,midpoint,equal in [('E',True,False),('E_zero',False,False),('E_eq',True,True),('E_zero_eq',False,True)]:
            rec={b:(self.native.reconstruct(data,b,midpoint,led['equal_removed_bits_per_head_side'] if equal else 0)@self.U) for b in (2,3,4)}
            dk=torch.stack([(rec[b][0]-K).square().mean(1) for b in (2,3,4)],-1)
            dv=torch.stack([(rec[b][1]-V).square().mean(2) for b in (2,3,4)],-1)
            kg=torch.stack((dk[...,0]-dk[...,1],dk[...,1]-dk[...,2]),-1)
            vg=torch.stack((dv[...,0]-dv[...,1],dv[...,1]-dv[...,2]),-1)
            kval=((kg[...,0]+1e-7<kg[...,1])|(kg.min(-1).values< -1e-7)).float().mean(-1)
            vval=((vg[...,0]+1e-7<vg[...,1])|(vg.min(-1).values< -1e-7)).float().mean(-1)
            km=dk.mean(1);vm=dv.mean(1)
            metrics=torch.stack([km,vm,km/K.square().mean((1,2))[:,None].clamp_min(1e-20),vm/V.square().mean((1,2))[:,None].clamp_min(1e-20)],-1).cpu().numpy()
            violation=torch.stack([kval,vval,(kg[...,1]-kg[...,0]).clamp_min(0).max(-1).values,(vg[...,1]-vg[...,0]).clamp_min(0).max(-1).values],-1).cpu().numpy()
            # Separately report reconstruction on actually quantized full tiles to expose tail-policy advantage.
            full=data['full'];span=torch.stack([torch.stack([(rec[b][0,:,:full]-K[:,:full]).square().sum((1,2))/K[:,:full].square().sum((1,2)).clamp_min(1e-20),(rec[b][1,:,:full]-V[:,:full]).square().sum((1,2))/V[:,:full].square().sum((1,2)).clamp_min(1e-20)],-1) for b in (2,3,4)],1).cpu().numpy()
            for h in range(8):
                for j,b in enumerate((2,3,4)):
                    a=metrics[h,j];vl=violation[h]
                    self.distrows.append(dict(core,seed=20261024,method=method,head=h,bits=b,k_mse=float(a[0]),v_mse=float(a[1]),D_K=float(a[2]),D_V=float(a[3]),k_violation_fraction=float(vl[0]),v_violation_fraction=float(vl[1]),k_violation_max=float(vl[2]),v_violation_max=float(vl[3]),quantized_span_D_K=float(span[h,j,0]),quantized_span_D_V=float(span[h,j,1]),native_fp16_tail_tokens=n-full))
            if self.pos==1:
                for side,d,weight in [('K',dk[0,:8],Q[0,:,:8].square().sum(0)),('V',dv[0,:8],P[0,:,:8].square().sum(0))]:
                    for objective,curve in [('mse',d),('attention_weighted',d*weight[:,None])]:
                        for alloc in allocation_check(curve.double().cpu().numpy()):self.allocfile.write(json.dumps(dict(core,method=method,side=side,objective=objective,**alloc))+'\n')
            ss={b:torch.einsum('hgd,htd->hgt',Q,rec[b][0])*scale for b in (2,3,4)}
            pp={b:s.softmax(-1) for b,s in ss.items()}
            cells=[(b,16) for b in (2,3,4)]+[(16,b) for b in (2,3,4)]+[(a,b) for a in (2,3,4) for b in (2,3,4)]
            for kb,vb in cells:
                s=S if kb==16 else ss[kb];p=P if kb==16 else pp[kb];vv=V if vb==16 else rec[vb][1];oo=p@vv
                a=torch.stack([(s-S).square().mean((1,2)).sqrt(),(s-S).flatten(1).norm(dim=1)/S.flatten(1).norm(dim=1).clamp_min(1e-12),(p-P).abs().sum(-1).mean(-1),(oo-O).flatten(1).norm(dim=1)/O.flatten(1).norm(dim=1).clamp_min(1e-12),(oo-O).square().mean((1,2)).sqrt()],-1).cpu().numpy()
                for h in range(8):self.rows.append(dict(core,seed=20261024,method=method,head=h,k_bits=kb,v_bits=vb,score_rmse=float(a[h,0]),score_rel=float(a[h,1]),prob_l1=float(a[h,2]),output_rel=float(a[h,3]),output_rmse=float(a[h,4]),native_fp16_tail_tokens=n-full))
    def run(self):
        selected=[r for r in self.manifest if r['split']!='calibration']
        reference=BASE/'results'/('smoke' if self.smoke else 'full')
        with gzip.open(self.out/'allocation.jsonl.gz','wt') as self.allocfile:
            for i,r in enumerate(selected):
                status('evaluating_E',case=r['case_id'],completed=i,total=len(selected),smoke=self.smoke)
                self.capture(r,'evaluate')
                name=r['case_id'].replace('/','_')+'_trajectory.json'
                a=json.loads((self.out/name).read_text());b=json.loads((reference/name).read_text())
                if a['sha256']!=b['sha256']:raise RuntimeError('Reference trajectory mismatch '+r['case_id'])
        base.atomic(self.out/'storage_ledger.json',list(self.ledger.values()))
        from report_e import report
        report(self.out,BASE/'results/full')
        status('completed_smoke' if self.smoke else 'completed',results=str(self.out))

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    try:EExperiment(a.smoke).run()
    except BaseException as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
