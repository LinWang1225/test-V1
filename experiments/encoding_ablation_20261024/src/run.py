"""Standalone real-activation encoding ablation, staged and resumable."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,gzip,csv,argparse,shutil,zipfile,re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
from codec import hadamard,fit_tree,encode,decode,allocation_check
ROOT=Path(__file__).resolve().parents[1]
MODEL='/home/wanglin/kvbench-3090-cu126/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'
TASKS=['qasper','hotpotqa','gov_report','trec','passage_retrieval_en','lcc']
SEEDS=[20261024,20261025,20261026]
POSITIONS=[1,8,32,64]

def atomic(path,data):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(phase,**kw):
    atomic(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(phase,kw,flush=True)

def prepare(tok,smoke=False):
    dest=ROOT/('inputs/smoke.json' if smoke else 'inputs/manifest.json')
    if dest.exists():return json.loads(dest.read_text())
    rows=[];seen=set()
    for task in TASKS:
        path=ROOT.parents[1]/'data/operator_profile'/f'{task}.jsonl'
        src=[json.loads(s) for s in path.read_text().splitlines()]
        archive='/home/wanglin/kvbench-3090-cu126/data_sources/longbench_5e628be/data.zip'
        with zipfile.ZipFile(archive) as zz:original=[json.loads(x) for x in zz.read('data/'+task+'.jsonl').splitlines()]
        unique=[];fingerprints=[];doc_hashes=set()
        for r in sorted(src,key=lambda r:r['case_id']):
            context=' '.join(original[r['source_index']]['context'].split())
            dh=hashlib.sha256(context.encode()).hexdigest()
            words=context.split();sketch=set(sorted({hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=8).digest() for i in range(max(1,len(words)-4))})[:256])
            if dh in doc_hashes or any(len(sketch&ss)/max(1,len(sketch|ss))>=.8 for ss in fingerprints):continue
            doc_hashes.add(dh);fingerprints.append(sketch);r['document_sha256']=dh;unique.append(r)
        if len(unique)<30:raise ValueError(f'{task}: fewer than 30 distinct document groups: {len(unique)}')
        src=unique
        src.sort(key=lambda r:(r['input_tokens'],r['case_id']))
        # Select evenly across available length distribution, then stratified splits.
        candidates=[src[i] for i in np.linspace(0,len(src)-1,30).round().astype(int)]
        for j,r in enumerate(candidates):
            text=r['body']['messages'];signature=hashlib.sha256(json.dumps(text,sort_keys=True).encode()).hexdigest()
            if signature in seen:raise ValueError('Duplicate prompt across requests; inspect source')
            seen.add(signature)
            ids=tok.apply_chat_template(text,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if len(ids)>8192:ids=ids[:4096]+ids[-4096:]
            split=['calibration','test','validation','test','calibration','test'][j%6]
            row=dict(case_id=r['case_id'],dataset=task,split=split,input_ids=ids,context=len(ids),source_sha256=sha(path),prompt_sha256=signature,document_sha256=r['document_sha256'])
            rows.append(row)
    if smoke:
        rows=[dict(rows[0],split='calibration'),dict(rows[1],split='test')]
        for r in rows:r['input_ids']=r['input_ids'][:256];r['context']=len(r['input_ids'])
    atomic(dest,rows);return rows

class Experiment:
    def __init__(self,smoke=False):
        self.smoke=smoke;self.cal=ROOT/('calibration/smoke' if smoke else 'calibration/full');self.cal.mkdir(exist_ok=True)
        self.out=ROOT/('results/smoke' if smoke else 'results/full');self.out.mkdir(exist_ok=True)
        self.cbdir=ROOT/('codebooks/smoke' if smoke else 'codebooks/full');self.cbdir.mkdir(exist_ok=True)
        self.U=torch.tensor(hadamard(),device='cuda',dtype=torch.float32)
        self.pos=0;self.mode='none';self.buff={};self.books={};self.rows=[];self.distrows=[]
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.manifest=prepare(self.tok,smoke)
        status('loading_model',smoke=smoke)
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').eval().to('cuda')
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa']
        def hooked(module,q,k,v,mask,**kw):
            if self.pos in POSITIONS:self.observe(module,q,k,v,kw['scaling'])
            return self.original(module,q,k,v,mask,**kw)
        ALL_ATTENTION_FUNCTIONS.register('encoding_ablation',hooked)
        self.model.config._attn_implementation='encoding_ablation'
        # Propagate explicitly to shared attention configs.
        for l in self.model.model.layers:l.self_attn.config._attn_implementation='encoding_ablation'
    @torch.inference_mode()
    def capture(self,r,mode):
        self.mode=mode;self.current=r;self.buff={};self.rows=[];self.distrows=[]
        ids=torch.tensor([r['input_ids']],device='cuda');cache=None;self.pos=0
        for start in range(0,ids.shape[1],512):
            out=self.model(input_ids=ids[:,start:start+512],past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values
        nextid=out.logits[:,-1].argmax(-1,keepdim=True);generated=[]
        for pos in range(1,65):
            self.pos=pos;generated.append(int(nextid.item()))
            out=self.model(input_ids=nextid,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values;nextid=out.logits[:,-1].argmax(-1,keepdim=True)
        self.pos=0
        atomic(self.out/(r['case_id'].replace('/','_')+'_trajectory.json'),dict(case_id=r['case_id'],generated_ids=generated,sha256=hashlib.sha256(np.asarray(generated,np.int32).tobytes()).hexdigest()))
        del out,cache
        if mode=='calibration':
            arrays={k:np.concatenate(v,axis=2) for k,v in self.buff.items()}
            np.savez_compressed(self.cal/(r['case_id'].replace('/','_')+'.npz'),**arrays)
        else:
            for kind,rows in [('attention',self.rows),('distortion',self.distrows)]:
                p=self.out/(r['case_id'].replace('/','_')+f'_{kind}.csv.gz')
                with gzip.open(str(p)+'.tmp','wt') as f:
                    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
                Path(str(p)+'.tmp').replace(p)
        del self.buff;self.buff={};torch.cuda.empty_cache()
    def observe(self,module,q,k,v,scale):
        layer=module.layer_idx
        if self.smoke and layer not in (0,17,35):return
        K=k[0].float()@self.U;V=v[0].float()@self.U
        Q=(q[0,:,0].float()@self.U).view(8,4,128)
        S=torch.einsum('hgd,htd->hgt',Q,K)*scale;P=S.softmax(-1);O=P@V
        if self.mode=='calibration':
            # Uniform token sample for every selected decode query; independent of saliency.
            rng=np.random.default_rng(20261024+layer*1000+self.pos+int(hashlib.sha256(self.current['case_id'].encode()).hexdigest()[:8],16))
            ix=torch.tensor(rng.choice(K.shape[1],min(32,K.shape[1]),replace=False),device='cuda')
            ks=K[:,ix];vs=V[:,ix];ps=P[:,:,ix]
            delta=(vs[:,None,:,:]-O[:,:,None,:]).square().sum(-1)
            wk=torch.einsum('hgt,hgd->htd',ps.square()*delta,Q.square())*scale**2
            wv=ps.square().sum(1)[:,:,None].expand_as(vs)
            for key,a in [(f'x{layer}',torch.stack((ks,vs))),(f'w{layer}',torch.stack((wk,wv)))]:
                self.buff.setdefault(key,[]).append(a.cpu().numpy().astype(np.float32 if key[0]=='w' else np.float16))
            return
        base=dict(case=self.current['case_id'],dataset=self.current['dataset'],split=self.current['split'],layer=layer,position=self.pos,context=K.shape[1])
        for seed in SEEDS:
            normal=self.books[(seed,layer,'norm')];mu,sigma=normal
            z=(torch.stack((K,V))-mu[:,:,None,:])/sigma[:,:,None,:]
            for label in ('A','B','C','D','A_eq','B_eq','C_eq','D_eq'):
                method=label.split('_')[0]
                if seed!=SEEDS[0] and method in ('A','B'):continue
                book=self.books[(seed,layer,method)]
                if method=='A':
                    lo,hi=book['lo'],book['hi'];code=torch.round((z-lo[:,:,None,None])/(hi-lo)[:,:,None,None]*15).clamp(0,15).long()
                else:
                    code=torch.stack([encode(z[s],{n:a[s] for n,a in book.items()}) for s in (0,1)])
                rec={};dk=[];dv=[]
                for b in (2,3,4):
                    if method=='A':
                        iz=code>>(4-b);cent=iz*2**(4-b)+(2**(4-b)-1)/2
                        zz=lo[:,:,None,None]+cent/15*(hi-lo)[:,:,None,None]
                    else:zz=torch.stack([decode(code[s],{n:a[s] for n,a in book.items()},b) for s in (0,1)])
                    x=zz*sigma[:,:,None,:]+mu[:,:,None,:];rec[b]=x
                    dk.append((x[0]-K).square().mean(1));dv.append((x[1]-V).square().mean(2))
                if label.endswith('_eq'):
                    # Pay extra metadata by omitting Bit0 on a deterministic prefix of scalars.
                    layers=3 if self.smoke else 36
                    per_stream=max(0.,(4 if method=='A' else 0 if method=='B' else 86)-86/(layers*16))
                    omit=int(np.ceil(per_stream))*8
                    if omit:
                        flat=rec[4].flatten(2);lower=rec[3].flatten(2)
                        flat[:,:,:omit]=lower[:,:,:omit]
                        dk[2]=(rec[4][0]-K).square().mean(1);dv[2]=(rec[4][1]-V).square().mean(2)
                # Distortion per physical head; local diminishing-gain violations per K channel / V token.
                dkc=torch.stack(dk,-1);dvt=torch.stack(dv,-1)
                for h in range(8):
                    kd=dkc[h].mean(0);vd=dvt[h].mean(0)
                    kg=torch.stack((dkc[h,:,0]-dkc[h,:,1],dkc[h,:,1]-dkc[h,:,2]),-1)
                    vg=torch.stack((dvt[h,:,0]-dvt[h,:,1],dvt[h,:,1]-dvt[h,:,2]),-1)
                    tol=1e-7
                    violk=((kg[:,0]+tol<kg[:,1])|(kg.min(-1).values < -tol));violv=((vg[:,0]+tol<vg[:,1])|(vg.min(-1).values < -tol))
                    # Small exact DP windows; save all budget gaps for first 8 K channels / 8 V tokens.
                    if self.pos==1 and h==0:
                        for side,d in [('K',dkc[h,:8]),('V',dvt[h,:8])]:
                            weight=Q[h,:,:8].square().sum(0) if side=='K' else P[h,:,:8].square().sum(0)
                            for objective,curve in [('mse',d),('attention_weighted',d*weight[:,None])]:
                                for alloc in allocation_check(curve.cpu().numpy()):
                                    self.allocfile.write(json.dumps(dict(base,seed=seed,method=label,side=side,objective=objective,**alloc))+'\n')
                    for j,b in enumerate((2,3,4)):
                        self.distrows.append(dict(base,seed=seed,method=label,head=h,bits=b,k_mse=float(kd[j]),v_mse=float(vd[j]),D_K=float(kd[j]/K[h].square().mean().clamp_min(1e-20)),D_V=float(vd[j]/V[h].square().mean().clamp_min(1e-20)),k_violation_fraction=float(violk.float().mean()),v_violation_fraction=float(violv.float().mean()),k_violation_max=float((kg[:,1]-kg[:,0]).clamp_min(0).max()),v_violation_max=float((vg[:,1]-vg[:,0]).clamp_min(0).max())))
                # 3 K-only, 3 V-only, and all 9 joint precision cells.
                kh={b:torch.einsum('hgd,htd->hgt',Q,rec[b][0])*scale for b in (2,3,4)}
                pp={b:s.softmax(-1) for b,s in kh.items()}
                cells=[(b,16) for b in (2,3,4)]+[(16,b) for b in (2,3,4)]+[(a,b) for a in (2,3,4) for b in (2,3,4)]
                for kb,vb in cells:
                    ss=S if kb==16 else kh[kb];pr=P if kb==16 else pp[kb];vv=V if vb==16 else rec[vb][1];oo=pr@vv
                    for h in range(8):
                        self.rows.append(dict(base,seed=seed,method=label,head=h,k_bits=kb,v_bits=vb,score_rmse=float((ss[h]-S[h]).square().mean().sqrt()),score_rel=float((ss[h]-S[h]).norm()/S[h].norm().clamp_min(1e-12)),prob_l1=float((pr[h]-P[h]).abs().sum(-1).mean()),output_rel=float((oo[h]-O[h]).norm()/O[h].norm().clamp_min(1e-12)),output_rmse=float((oo[h]-O[h]).square().mean().sqrt())))
        # One FP64 and rotation-equivalence check per request, actual full-length input.
        if layer==0 and self.pos==1:
            q0=q[0,:,0].float().view(8,4,128);k0=k[0].float();v0=v[0].float()
            raw=(torch.einsum('hgd,htd->hgt',q0,k0)*scale).softmax(-1)@v0
            fp64=(torch.einsum('hgd,htd->hgt',Q.double(),K.double())*scale).softmax(-1)@V.double()
            atomic(self.out/(self.current['case_id'].replace('/','_')+'_numerical.json'),dict(rotation_output_rel=float(((O@self.U.T)-raw).norm()/raw.norm()),fp32_vs_fp64_rel=float((O.double()-fp64).norm()/fp64.norm())))
    def train(self):
        files=sorted(self.cal.glob('*.npz'));layers=(0,17,35) if self.smoke else range(36)
        gaussian=fit_tree(None,None,gaussian=True)
        for layer in layers:
            status('fitting_codebooks',layer=layer,smoke=self.smoke)
            xx=[];ww=[]
            for p in files:
                with np.load(p) as z:xx.append(z[f'x{layer}'].astype(np.float32));ww.append(z[f'w{layer}'])
            X=np.stack(xx);W=np.stack(ww);mu=X.mean((0,3)).astype(np.float16)
            sigma=X.std((0,3)).clip(1e-4).astype(np.float16)
            Z=(X-mu[None,:,:,None,:])/sigma[None,:,:,None,:]
            for seed in SEEDS:
                rng=np.random.default_rng(seed+layer);counts=np.bincount(rng.integers(0,len(X),len(X)),minlength=len(X))
                book={'mu':mu,'sigma':sigma,'A_lo':Z.min((0,3,4)).astype(np.float16),'A_hi':Z.max((0,3,4)).astype(np.float16)}
                for method in ('B','C','D'):
                    records=[]
                    for side in (0,1):
                        hs=[]
                        for h in range(8):
                            if method=='B':b=gaussian
                            else:
                                x=Z[:,side,h].reshape(-1);weights=np.broadcast_to(counts[:,None,None],Z[:,side,h].shape).reshape(-1).astype(float)
                                if method=='D':weights=weights*(W[:,side,h]*sigma[side,h][None,None,:].astype(np.float32)**2).reshape(-1)
                                # Histogram sufficient statistics; weighted mean representative per nonempty bin.
                                hist,edges=np.histogram(x,bins=8192,weights=weights)
                                sums=np.histogram(x,bins=edges,weights=weights*x)[0];ok=hist>0
                                b=fit_tree(sums[ok]/hist[ok],hist[ok])
                            hs.append(b)
                        records.append({n:np.stack([b[n] for b in hs]) for n in hs[0]})
                    for n in records[0]:book[f'{method}_{n}']=np.stack([r[n] for r in records])
                np.savez(self.cbdir/f'layer{layer}_seed{seed}.npz',**book)
        self.storage_ledger()
    def load_books(self):
        layers=(0,17,35) if self.smoke else range(36)
        for layer in layers:
            for seed in SEEDS:
                with np.load(self.cbdir/f'layer{layer}_seed{seed}.npz') as b:
                    self.books[seed,layer,'norm']=[torch.tensor(b[n].astype(np.float32),device='cuda') for n in ('mu','sigma')]
                    self.books[seed,layer,'A']={n:torch.tensor(b['A_'+n].astype(np.float32),device='cuda') for n in ('lo','hi')}
                    for method in ('B','C','D'):
                        self.books[seed,layer,method]={n:torch.tensor(b[method+'_'+n].astype(np.float32),device='cuda') for n in ('c2','c3','c4','t2','t3','t4')}
    def storage_ledger(self):
        layers=3 if self.smoke else 36;heads=8;d=128
        # Actual serialized arrays, compact device representation; no ZIP/container overhead as runtime state.
        common=layers*2*heads*d*2*2+128 # means/scales FP16 + deterministic signs int8
        ledger=[]
        for n in (2048,4096,8192):
            total_scalars=layers*2*heads*n*d;payload=(total_scalars+1)//2
            metas={m:common+layers*2*heads*(4 if m=='A' else 86) for m in ('A','B','C','D')}
            # B uses one global Gaussian table, counted once; A ranges per head, C/D head-specific.
            metas['B']=common+86
            budget=payload+min(metas.values())
            for m,meta in metas.items():
                capacity=max(0,(budget-meta)*2//(layers*2*heads*d))
                omitted_bytes_per_stream=int(np.ceil(max(0.,(meta-metas['B'])/(layers*16))))
                equal_actual=payload+meta-omitted_bytes_per_stream*layers*16
                for b in (2,3,4):ledger.append(dict(method=m,context=n,read_bits=b,stored_payload_bytes=payload,metadata_bytes=meta,total_bytes=payload+meta,effective_stored_bits=8*(payload+meta)/total_scalars,logical_read_bytes=(total_scalars*b+7)//8,equal_total_budget_bytes=budget,max_tokens_under_equal_budget=capacity,equal_omitted_bit0_scalars_per_stream=omitted_bytes_per_stream*8,equal_budget_actual_bytes=equal_actual))
        atomic(self.out/'storage_ledger.json',ledger)
    def run(self):
        for r in self.manifest:
            if r['split']!='calibration':continue
            path=self.cal/(r['case_id'].replace('/','_')+'.npz')
            if not path.exists():status('capturing_calibration',case=r['case_id'],smoke=self.smoke);self.capture(r,'calibration')
        self.train();self.load_books()
        with gzip.open(self.out/'allocation.jsonl.gz','wt') as self.allocfile:
            for r in self.manifest:
                if r['split']=='calibration':continue
                # No reuse of partial request rows: overwrite atomically at request completion.
                status('evaluating',case=r['case_id'],split=r['split'],smoke=self.smoke);self.capture(r,'evaluate')
        from report import report
        report(self.out,self.manifest)
        status('completed_smoke' if self.smoke else 'completed',results=str(self.out))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    try:Experiment(args.smoke).run()
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
