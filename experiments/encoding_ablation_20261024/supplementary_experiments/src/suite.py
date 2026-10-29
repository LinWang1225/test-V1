"""E0-E4 offline suite: no validation requests, complete 4-bit main payload."""
import os,sys,time,json,gzip,csv,hashlib,argparse,traceback,heapq
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(R/'deps'));sys.path.insert(0,str(B/'src'))
import pyarrow as pa
import pyarrow.parquet as pq
import run as base
from codec import encode,decode,allocation_check
from books import METHODS,clipped,independent,independent_uniform,joint,regularize,distortion,drain_node_events
SEEDS=base.SEEDS;BITS=(2,3,4)

def state(stage,**kw):
    base.atomic(R/'suite_status.json',dict(stage=stage,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(stage,kw,flush=True)
base.status=state

def savecsv(path,rows):
    if not rows:return
    with gzip.open(str(path)+'.tmp','wt') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    Path(str(path)+'.tmp').replace(path)

def loadside(z,kind,objective,side):
    prefix=f'{side}_{kind}_{objective}_'
    return {n[len(prefix):]:torch.tensor(z[n].astype(np.float32),device='cuda') for n in z.files if n.startswith(prefix)}

def code_side(x,kind,book):
    if kind=='uniform':return torch.round((x-book['lo'][:,None,None])/(book['hi']-book['lo'])[:,None,None]*15).clamp(0,15).long()
    if kind=='tree':return encode(x,book)
    if kind=='uniform_independent':return {b:torch.round((x-book[f'lo{b}'][:,None,None])/(book[f'hi{b}']-book[f'lo{b}'])[:,None,None]*(2**b-1)).clamp(0,2**b-1).long() for b in BITS}
    return {b:(x[...,None]>book[f'f{b}'][:,None,None,:]).sum(-1) for b in BITS}

def decode_side(code,kind,book,b):
    if kind=='uniform':
        c=(code>>(4-b))*2**(4-b)+(2**(4-b)-1)/2
        return book['lo'][:,None,None]+c/15*(book['hi']-book['lo'])[:,None,None]
    if kind=='tree':return decode(code,book,b)
    if kind=='uniform_independent':return book[f'lo{b}'][:,None,None]+code[b]/(2**b-1)*(book[f'hi{b}']-book[f'lo{b}'])[:,None,None]
    ix=code[b];return torch.gather(book[f'c{b}'],1,ix.flatten(1)).view_as(ix)

def greedy_depths(curves,targets):
    """Actual prerequisite-constrained greedy; equal one-plane cost within each side."""
    shape=curves.shape[:-1];d=np.asarray(curves,np.float64).reshape(-1,3);g=d[:,:-1]-d[:,1:];n=len(d)
    budgets=[int(round((x-2)*n)) for x in targets];order=np.argsort(budgets);result={};levels=np.zeros(n,np.int8)
    if np.all(g[:,0]>=g[:,1]) and np.all(g>=0):
        # Tie order keeps parent edges before child edges.
        idx=np.lexsort((np.arange(n*2),-g.reshape(-1)));edge_unit=idx//2
        for i in order:
            count=np.bincount(edge_unit[:budgets[i]],minlength=n);result[targets[i]]=(count+2).reshape(shape)
    else:
        heap=[(-float(g[i,0]),i,0) for i in range(n)];heapq.heapify(heap);spent=0
        for bi in order:
            while spent<budgets[bi] and heap:
                neg,i,l=heapq.heappop(heap)
                # Fixed budget: all refinements are allowed; negative gains retained and reported.
                levels[i]+=1;spent+=1
                if l==0:heapq.heappush(heap,(-float(g[i,1]),i,1))
            result[targets[bi]]=(levels+2).reshape(shape).copy()
    return result

def exact_dp_check(curves):
    d=np.asarray(curves,np.float64);g=np.stack((d[:,0]-d[:,1],d[:,1]-d[:,2]),-1);n=len(d);table=np.full(2*n+1,-np.inf);table[0]=0.
    for row in g:
        new=np.full_like(table,-np.inf)
        for cost,value in enumerate((0.,float(row[0]),float(row.sum()))):
            new[cost:]=np.maximum(new[cost:],table[:len(table)-cost]+value)
        table=new
    depths=greedy_depths(d,[2+b/n for b in range(2*n+1)]);out=[]
    for budget in range(2*n+1):
        level=depths[2+budget/n].reshape(-1)-2
        greedy=float(sum(0. if l==0 else row[0] if l==1 else row.sum() for row,l in zip(g,level)))
        optimum=float(table[budget]);gap=max(0.,optimum-greedy);tol=1e-12+1e-10*max(abs(optimum),abs(greedy),1e-30)
        out.append(dict(budget=budget,optimal_gain=optimum,greedy_gain=greedy,gap=gap,tolerance=tol,material_gap=gap>tol,budget_constraint='exact_equal_cost_refinements',dtype='float64'))
    return out

def fit_all(smoke=False):
    target=R/'codebooks'/('smoke' if smoke else 'full');target.mkdir(exist_ok=True)
    scores=R/'calibration'/('score_smoke' if smoke else 'score_full')
    source=B/'calibration'/('smoke' if smoke else 'full')
    paths=sorted(source.glob('*.npz'));layers=(0,17,35) if smoke else range(36);seeds=SEEDS[:1] if smoke else SEEDS
    training=[];reg_count=0;stats=[]
    for layer in layers:
        state('training',layer=layer,smoke=smoke)
        xx=[];ww=[];ss=[]
        for p in paths:
            with np.load(p) as a:xx.append(a[f'x{layer}'].astype(np.float32));ww.append(a[f'w{layer}'])
            with np.load(scores/p.name) as a:ss.append(a[f's{layer}'][0])
        X=np.stack(xx);W=np.stack(ww);S=np.stack(ss)
        for seed in seeds:
            sourcebook=B/'codebooks'/('smoke' if smoke else 'full')/f'layer{layer}_seed{seed}.npz'
            with np.load(sourcebook) as a:old={k:a[k] for k in a.files}
            mu,sigma=old['mu'],old['sigma'];Z=(X-mu[None,:,:,None,:])/sigma[None,:,:,None,:]
            counts=np.bincount(np.random.default_rng(seed+layer).integers(0,len(X),len(X)),minlength=len(X))
            output=dict(mu=mu,sigma=sigma);profiles={}
            for side in (0,1):
                allbooks=[];allprofiles=[]
                for h in range(8):
                    x=Z[:,side,h].reshape(-1).astype(float)
                    freq=np.broadcast_to(counts[:,None,None],Z[:,side,h].shape).reshape(-1).astype(float)
                    sens=(W[:,side,h]*sigma[side,h][None,None,:].astype(float)**2).reshape(-1)
                    if side==0:score=(S[:,h]*sigma[side,h][None,None,:].astype(float)**2).reshape(-1)
                    else:score=np.ones_like(sens)
                    denom=max(np.sum(freq),1e-30)
                    sens=sens/max(np.sum(freq*sens)/denom,1e-30);score=score/max(np.sum(freq*score)/denom,1e-30)
                    vectors={'mse':freq,'jac':freq*sens,'attn':freq*sens,'score':freq*score,'mix':freq*(.5+.5*sens)}
                    edges=np.linspace(x.min(),x.max(),8193);hc=np.histogram(x,bins=edges,weights=freq)[0];xc=np.histogram(x,bins=edges,weights=freq*x)[0];valid=hc>0
                    cx=xc[valid]/hc[valid];cw=hc[valid]
                    books={}
                    books['uniform:minmax']=dict(lo=np.float16(old['A_lo'][side,h]),hi=np.float16(old['A_hi'][side,h]))
                    books['tree:gaussian']={n:old['B_'+n][side,h] for n in ('c2','c3','c4','t2','t3','t4')}
                    books['tree:mse']={n:old['C_'+n][side,h] for n in ('c2','c3','c4','t2','t3','t4')}
                    books['tree:jac']={n:old['D_'+n][side,h] for n in ('c2','c3','c4','t2','t3','t4')}
                    books['tree:attn']=books['tree:jac']
                    for objective in ('mse','score','attn','mix'):
                        w=vectors[objective];hw=np.histogram(x,bins=edges,weights=w)[0];hx=np.histogram(x,bins=edges,weights=w*x)[0];ok=hw>0
                        if not np.any(ok):hw=hc;hx=xc;ok=valid
                        vx,weights=hx[ok]/hw[ok],hw[ok]
                        u,info=clipped(vx,weights);books['uniform:'+objective]=u
                        training.append(dict(layer=layer,seed=seed,side=side,head=h,kind='uniform',loss_target=objective,**info))
                        if objective in ('score','mix'):
                            from codec import fit_tree
                            books['tree:'+objective]=fit_tree(vx,weights)
                    books['independent:mse']=independent(cx,cw)
                    books['uniform_independent:mse']=independent_uniform(cx,cw)
                    books['tree:leaf4'],info=joint(cx,cw,books['tree:mse'],only4=True)
                    training.append(dict(layer=layer,seed=seed,side=side,head=h,kind='leaf4',**info))
                    books['tree:joint'],info=joint(cx,cw,books['tree:mse'])
                    training.append(dict(layer=layer,seed=seed,side=side,head=h,kind='joint',**info))
                    books['tree:regularized'],info=regularize(cx,cw,books['tree:joint']);reg_count+=int(info['triggered'])
                    training.append(dict(layer=layer,seed=seed,side=side,head=h,kind='regularized',**info))
                    prof={}
                    for key,b in books.items():
                        kind,objective=key.split(':')
                        if kind=='uniform':
                            from books import uniform_decode
                            ds=np.array([np.average((cx-uniform_decode(cx,float(b['lo']),float(b['hi']),bit))**2,weights=cw) for bit in BITS])
                        elif kind=='tree':ds=distortion(cx,cw,b)
                        elif kind=='uniform_independent':ds=np.array([np.average((cx-(b[f'lo{bit}']+np.rint((cx-b[f'lo{bit}'])/(b[f'hi{bit}']-b[f'lo{bit}'])*(2**bit-1)).clip(0,2**bit-1)/(2**bit-1)*(b[f'hi{bit}']-b[f'lo{bit}'])))**2,weights=cw) for bit in BITS])
                        else:ds=np.array([np.average((cx-b[f'c{bit}'][np.searchsorted(b[f'f{bit}'],cx)])**2,weights=cw) for bit in BITS])
                        prof[key]=ds.astype(np.float64)
                        stats.append(dict(layer=layer,seed=seed,side=side,head=h,book=key,D2=float(ds[0]),D3=float(ds[1]),D4=float(ds[2]),metadata_bytes=sum(a.nbytes for a in b.values()),normalized_gain_violation=float(max(0.,(ds[1]-ds[2])-(ds[0]-ds[1]),ds[1]-ds[0],ds[2]-ds[1])/max(ds[0],1e-30))))
                    training.append(dict(layer=layer,seed=seed,side=side,head=h,kind='node_convergence_trace',calls=drain_node_events(),legacy_reused_C_D='frozen codebooks; original node traces unavailable'))
                    allbooks.append(books);allprofiles.append(prof)
                for key in allbooks[0]:
                    kind,objective=key.split(':')
                    for name in allbooks[0][key]:output[f'{side}_{kind}_{objective}_{name}']=np.stack([bs[key][name] for bs in allbooks])
                    output[f'{side}_profile_{kind}_{objective}']=np.stack([p[key] for p in allprofiles])
            np.savez(target/f'layer{layer}_seed{seed}.npz',**output)
    base.atomic(target/'training_history.json',training);base.atomic(target/'calibration_statistics.json',stats)
    active=dict(METHODS)
    if reg_count==0:active.pop('Joint_Regularized')
    base.atomic(target/'active_methods.json',dict(methods=active,regularizer_triggered_heads=reg_count,regularizer_gate='normalized head-calibration violation >= 0.01; no test/validation selection'))
    state('training_complete',active_methods=list(active),regularizer_triggered_heads=reg_count)

class Suite(base.Experiment):
    def __init__(self,stage,smoke=False):
        self.stage=stage
        super().__init__(smoke)
        self.cal=R/'calibration'/('score_smoke' if smoke else 'score_full');self.cal.mkdir(exist_ok=True)
        self.out=R/'metrics'/('smoke' if smoke else 'evaluation');self.out.mkdir(exist_ok=True)
        self.cbdir=R/'codebooks'/('smoke' if smoke else 'full')
        self.suitebooks={}
        if stage=='evaluate':
            self.active=json.loads((self.cbdir/'active_methods.json').read_text())['methods']
            for layer in ((0,17,35) if smoke else range(36)):
                for seed in (SEEDS[:1] if smoke else SEEDS):
                    with np.load(self.cbdir/f'layer{layer}_seed{seed}.npz') as z:
                        norm=[torch.tensor(z[n].astype(np.float32),device='cuda') for n in ('mu','sigma')]
                        bs={}
                        for pair in self.active.values():
                            for side,key in enumerate(pair):
                                kind,obj=key.split(':')
                                if (side,key) not in bs:bs[side,key]=loadside(z,kind,obj,side)
                        profiles={(side,key):z[f'{side}_profile_{key.replace(":","_")}'] for pair in self.active.values() for side,key in enumerate(pair)}
                        self.suitebooks[seed,layer]=(norm,bs,profiles)
    def observe(self,module,q,k,v,scale):
        layer=module.layer_idx
        if self.smoke and layer not in (0,17,35):return
        Q=(q[0,:,0].float()@self.U).view(8,4,128)
        if self.stage=='calibrate':
            s=(Q.square().sum(1)*scale**2)[:,None,:].expand(8,32,128)[None].cpu().numpy()
            self.buff.setdefault(f's{layer}',[]).append(s);return
        K=k[0].float()@self.U;V=v[0].float()@self.U;N=K.shape[1]
        score=torch.einsum('hgd,htd->hgt',Q,K)*scale;P=score.softmax(-1);O=P@V
        core=dict(case=self.current['case_id'],dataset=self.current['dataset'],split='test',layer=layer,position=self.pos,context=N)
        for seed in (SEEDS[:1] if self.smoke else SEEDS):
            (mu,sigma),books,profiles=self.suitebooks[seed,layer];zz=(torch.stack((K,V))-mu[:,:,None,:])/sigma[:,:,None,:]
            # Decode each unique side/book once per observation; combinations reuse exact tensors.
            used={};codes={};timings={}
            for pair in self.active.values():
                for side,key in enumerate(pair):
                    if (side,key) in used:continue
                    kind=key.split(':')[0];book=books[side,key]
                    start=torch.cuda.Event(enable_timing=True);encoded=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();code=code_side(zz[side],kind,book);encoded.record()
                    xs={b:decode_side(code,kind,book,b)*sigma[side,:,None,:]+mu[side,:,None,:] for b in BITS};end.record();end.synchronize()
                    timings[side,key]=(start.elapsed_time(encoded),encoded.elapsed_time(end));used[side,key]=xs;codes[side,key]=code
            for method,pair in self.active.items():
                rec={b:torch.stack((used[0,pair[0]][b],used[1,pair[1]][b])) for b in BITS}
                dk=torch.stack([(rec[b][0]-K).double().square().mean(1) for b in BITS],-1)
                dv=torch.stack([(rec[b][1]-V).double().square().mean(2) for b in BITS],-1)
                sk=Q.double().square().sum(1)*scale**2
                aw=P.double().square().sum(1)*128
                functional=[dk*sk[:,:,None],dv*aw[:,:,None]]
                raw=[dk,dv]
                proxy_k=functional[0].sum(1);proxy_v=functional[1].sum(1)
                mseK=dk.mean(1);mseV=dv.mean(1)
                values=torch.stack((mseK,mseV,mseK/K.double().square().mean((1,2))[:,None].clamp_min(1e-20),mseV/V.double().square().mean((1,2))[:,None].clamp_min(1e-20),proxy_k,proxy_v),-1).cpu().numpy()
                for h in range(8):
                    for j,b in enumerate(BITS):
                        a=values[h,j];self.distrows.append(dict(core,seed=seed,method=method,head=h,bits=b,k_mse=float(a[0]),v_mse=float(a[1]),D_K=float(a[2]),D_V=float(a[3]),k_score_diag=float(a[4]),v_attention_diag=float(a[5])))
                # Fixed 4 K channels and 4 V tokens/head, all layers/positions/seeds, both raw and functional curves.
                for side,unit in [(0,'channel'),(1,'token')]:
                    idx=np.linspace(0,raw[side].shape[1]-1,4).round().astype(int)
                    for objective,tensor in [('raw_mse',raw[side]),('functional_proxy',functional[side])]:
                        a=tensor[:,idx].cpu().numpy().reshape(-1,3);gain=a[:,:2]-a[:,1:]
                        tol=1e-10+1e-5*np.maximum(np.abs(a).max(1),1e-20)
                        violation=(gain[:,1]>gain[:,0]+tol)|(gain.min(1)<-tol)
                        magnitude=np.maximum.reduce((np.zeros(len(a)),gain[:,1]-gain[:,0],-gain[:,0],-gain[:,1]))/np.maximum(a[:,0],1e-20)
                        n=len(a);entry={key:np.repeat(value,n) for key,value in dict(core,seed=seed,method=method,side='K' if side==0 else 'V',unit_type=unit,objective=objective).items()}
                        entry.update(head=np.repeat(np.arange(8),4),unit=np.tile(idx,8),D2=a[:,0],D3=a[:,1],D4=a[:,2],gain_23=gain[:,0],gain_34=gain[:,1],gain_ratio=np.divide(gain[:,1],gain[:,0],out=np.full(n,np.nan),where=gain[:,0]>0),negative_gain=gain.min(1)<-tol,diminishing_violation=violation,violation_magnitude=magnitude,tolerance=tol)
                        self.curves.append(entry)
                # Full code/prefix utilization, aggregated into per-request codebook statistics.
                for side,key in enumerate(pair):
                    code=codes[side,key];code=code[4] if isinstance(code,dict) else code
                    offset=torch.arange(8,device='cuda')[:,None,None]*16
                    hist=torch.bincount((code+offset).flatten(),minlength=128).view(8,16).cpu().numpy()
                    for h in range(8):
                        self.code_stats.append(dict(core,seed=seed,method=method,side='K' if side==0 else 'V',head=h,leaf_counts=json.dumps(hist[h].tolist()),prefix2_counts=json.dumps(hist[h].reshape(4,4).sum(1).tolist()),prefix3_counts=json.dumps(hist[h].reshape(8,2).sum(1).tolist()),used_leaf_count=int((hist[h]>0).sum()),encode_ms=timings[side,key][0],decode_3depths_ms=timings[side,key][1],timing_scope='PyTorch reference functions, not custom kernel'))
                ss={b:torch.einsum('hgd,htd->hgt',Q,rec[b][0])*scale for b in BITS};pp={b:s.softmax(-1) for b,s in ss.items()}
                for kb,vb in [(b,16) for b in BITS]+[(16,b) for b in BITS]+[(a,b) for a in BITS for b in BITS]:
                    s=score if kb==16 else ss[kb];p=P if kb==16 else pp[kb];vv=V if vb==16 else rec[vb][1];oo=p@vv
                    a=torch.stack(((s-score).square().mean((1,2)).sqrt(),(p-P).abs().sum(-1).mean(-1),(oo-O).flatten(1).norm(dim=1)/O.flatten(1).norm(dim=1).clamp_min(1e-12),(oo-O).square().mean((1,2)).sqrt()),-1).cpu().numpy()
                    for h in range(8):self.rows.append(dict(core,seed=seed,method=method,head=h,k_bits=kb,v_bits=vb,score_rmse=float(a[h,0]),prob_l1=float(a[h,1]),output_rel=float(a[h,2]),output_rmse=float(a[h,3])))
                if seed==SEEDS[0] and layer in (0,17,35) and self.pos in (1,64) and method in ('U_Clipped','Data_Tree','JacK_MSEV','ScoreK_MixV','Joint_234'):
                    predk=sk.cpu().numpy()*sigma[0].cpu().numpy()**2
                    predk=predk[:,:,None]*profiles[0,pair[0]][:,None,:]
                    targets=[2.,2.5,3.,3.5,4.];kdepth=greedy_depths(predk,targets)
                    small=predk.reshape(-1,3)[np.linspace(0,predk.reshape(-1,3).shape[0]-1,8).round().astype(int)]
                    for row in exact_dp_check(small.astype(np.float64)):self.dp_rows.append(dict(core,seed=seed,method=method,side='K',target_mean_bits=-1.,objective='calibrated_functional_proxy',**row))
                    for target in targets:
                        kd=torch.tensor(kdepth[target],device='cuda')
                        kk=torch.where((kd==2)[:,None,:],rec[2][0],torch.where((kd==3)[:,None,:],rec[3][0],rec[4][0]))
                        phat=(torch.einsum('hgd,htd->hgt',Q,kk)*scale).softmax(-1)
                        predv=phat.double().square().sum(1).cpu().numpy()*sigma[1].double().square().sum(-1).cpu().numpy()[:,None]
                        predv=predv[:,:,None]*profiles[1,pair[1]][:,None,:]
                        vdepth=greedy_depths(predv,[target]);vd=torch.tensor(vdepth[target],device='cuda')
                        small=predv.reshape(-1,3)[np.linspace(0,predv.reshape(-1,3).shape[0]-1,8).round().astype(int)]
                        for row in exact_dp_check(small.astype(np.float64)):self.dp_rows.append(dict(core,seed=seed,method=method,side='V',target_mean_bits=target,objective='calibrated_functional_proxy',**row))
                        vv=torch.where((vd==2)[:,:,None],rec[2][1],torch.where((vd==3)[:,:,None],rec[3][1],rec[4][1]))
                        oo=phat@vv
                        err=((oo-O).flatten(1).norm(dim=1)/O.flatten(1).norm(dim=1).clamp_min(1e-12)).mean()
                        readbits=int(kdepth[target].sum()*N+vdepth[target].sum()*128)
                        self.dynamic_rows.append(dict(core,seed=seed,method=method,target_mean_bits=target,actual_mean_bits=readbits/(2*8*N*128),logical_payload_read_bytes=(readbits+7)//8,output_rel=float(err),controller='calibrated per-head distortion, current Q, then reconstructed-K attention for V; offline PyTorch prototype',storage_bits=4))
            del used,codes
    def execute(self):
        if self.stage=='calibrate':
            selected=[r for r in self.manifest if r['split']=='calibration']
            for i,r in enumerate(selected):
                state('capture_score_weights',completed=i,total=len(selected),case=r['case_id'])
                p=self.cal/(r['case_id'].replace('/','_')+'.npz')
                if not p.exists():self.capture(r,'calibration')
        else:
            selected=[r for r in self.manifest if r['split']=='test']
            for i,r in enumerate(selected):
                state('evaluate_test',completed=i,total=len(selected),case=r['case_id'],methods=len(self.active),seeds=len(SEEDS) if not self.smoke else 1)
                self.curves=[];self.code_stats=[];self.dynamic_rows=[];self.dp_rows=[]
                self.capture(r,'evaluate');name=r['case_id'].replace('/','_')
                cols={k:np.concatenate([c[k] for c in self.curves]) for k in self.curves[0]}
                pq.write_table(pa.table(cols),self.out/(name+'_curves.parquet'),compression='zstd',row_group_size=65536)
                savecsv(self.out/(name+'_codebook.csv.gz'),self.code_stats);savecsv(self.out/(name+'_dynamic.csv.gz'),self.dynamic_rows);savecsv(self.out/(name+'_dp.csv.gz'),self.dp_rows)
                del self.curves,self.code_stats,self.dynamic_rows,self.dp_rows
        reference=B/'results'/('smoke' if self.smoke else 'full')
        for r in selected:
            name=r['case_id'].replace('/','_')+'_trajectory.json'
            assert json.loads((self.out/name).read_text())['sha256']==json.loads((reference/name).read_text())['sha256'],r['case_id']
        state('stage_completed',stage_name=self.stage,requests=len(selected))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibrate','train','evaluate'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    try:
        if a.stage=='calibrate' and not a.smoke:
            import subprocess
            for gate in ('calibrate','train','evaluate'):
                state('end_to_end_smoke_gate',phase=gate)
                subprocess.run([sys.executable,'-u',__file__,'--stage',gate,'--smoke'],check=True)
        if a.stage=='train':fit_all(a.smoke)
        else:Suite(a.stage,a.smoke).execute()
    except BaseException as e:state('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
