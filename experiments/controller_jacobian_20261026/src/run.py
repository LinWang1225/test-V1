"""Causal replay and closed-loop reference adapter, fixed original D codebooks.

This is an information-causal numerical experiment. The harness retains FP16
source caches and simulates selected prefixes, so timings are NOT serving speed.
"""
import sys,os,json,time,argparse,traceback,itertools
from pathlib import Path
CROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CROOT.parent/'encoding_extension_20261026/src'))
from common import *
from policy import TARGETS,POLICIES,key_decision,value_decision,predicted_error,allocate


def cstate(stage,**kw):
    atomic(CROOT/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))
    print(stage,kw,flush=True)


def select(rec,depth,side):
    d=torch.tensor(depth,device='cuda')
    mask=d[:,None,:] if side==0 else d[:,:,None]
    return torch.where(mask==2,rec[2][side],torch.where(mask==3,rec[3][side],rec[4][side]))


def decide(Q,rec,profile,scale,target,policy):
    query=Q.double().cpu().numpy()
    start=time.perf_counter()
    kd=key_decision(query,profile[0],scale,target,policy)
    key_ms=(time.perf_counter()-start)*1000
    kk=select(rec,kd,0)
    score=torch.einsum('hgd,htd->hgt',Q,kk)*scale
    phat=score.softmax(-1)
    attention=phat.double().cpu().numpy()
    start=time.perf_counter()
    vd=value_decision(attention,profile[1],target,policy)
    value_ms=(time.perf_counter()-start)*1000
    vv=select(rec,vd,1)
    return phat@vv,phat,kk,vv,kd,vd,query,attention,key_ms,value_ms


class ControllerExperiment(legacy.Experiment):
    def __init__(self,stage,smoke=False):
        self.stage=stage
        legacy.status=cstate
        super().__init__(smoke)
        self.out=CROOT/'results'/('smoke_'+stage if smoke else stage)
        self.out.mkdir(parents=True,exist_ok=True)
        ready=ROOT/'results'/('smoke' if smoke else 'full')/'analysis_complete.json'
        assert json.loads(ready.read_text())['complete'],'Encoding analysis has not completed'
        layers=(0,17,35) if smoke and stage=='replay' else range(36)
        # Online smoke uses all 36 layers, hence formal books are required. A
        # separate gate constructs all-layer smoke books from frozen old D.
        self.cbooks={l:load_layer(l,SEEDS[0],smoke and stage=='replay') for l in layers}
        self.profiles={l:self.cbooks[l]['controller_D'].double().cpu().numpy() for l in layers}
        self.oracle=[]
        if stage=='online':
            def closed(module,q,k,v,mask,**kw):
                if self.pos==0 or self.online_policy=='fp16':
                    return self.original(module,q,k,v,mask,**kw)
                out=self.online_attention(module,q,k,v,kw['scaling'])
                return out,None
            legacy.ALL_ATTENTION_FUNCTIONS.register('controller_closed_loop',closed)
            self.model.config._attn_implementation='controller_closed_loop'
            for layer in self.model.model.layers:
                layer.self_attn.config._attn_implementation='controller_closed_loop'

    def observe(self,module,q,k,v,scale):
        layer=module.layer_idx
        if layer not in (0,17,35) or self.pos not in (1,64):return
        Q=(q[0,:,0].float()@self.U).view(8,4,128)
        K,V=k[0].float()@self.U,v[0].float()@self.U
        score=torch.einsum('hgd,htd->hgt',Q,K)*scale
        P=score.softmax(-1);O=P@V
        rec,_=reconstruct(torch.stack((K,V)),'Jacobian_Tree',self.cbooks[layer])
        curves=self.profiles[layer]
        n=K.shape[1]
        core=dict(case=self.current['case_id'],dataset=self.current['dataset'],layer=layer,position=self.pos,context=n)
        for target in TARGETS:
            for policy in POLICIES:
                result=decide(Q,rec,curves,scale,target,policy)
                out,phat,kk,vv,kd,vd,query,attention,kms,vms=result
                pk,pv=predicted_error(query,attention,kd,vd,curves,scale)
                delta=(vv-V).double();pe=phat.double()
                vexact=(pe@delta).square().mean((1,2))
                vdiag=torch.einsum('hgt,ht->h',pe.square(),delta.square().sum(-1))/(4*128)
                ke=(pe-P.double())@V.double()
                exact_score=(torch.einsum('hgd,htd->hgt',Q,kk-K)*scale).double().square().mean((1,2))
                kdiag=torch.einsum('hgd,htd->h',Q.double().square(),(kk-K).double().square())*scale**2/(4*n)
                actual=torch.stack(((out-O).square().mean((1,2)),(out-O).flatten(1).norm(dim=1)/O.flatten(1).norm(dim=1).clamp_min(1e-12),exact_score,kdiag,vexact,vdiag,vexact-vdiag,ke.square().mean((1,2))),-1).cpu().numpy()
                readbits=int(kd.sum()*n+vd.sum()*128)
                readbytes=int(kd.sum()*((n+7)//8)+vd.sum()*16)
                assert int((kd-2).sum())==int((target-2)*kd.size)
                assert int((vd-2).sum())==int((target-2)*vd.size)
                for h,a in enumerate(actual):
                    self.rows.append(dict(core,policy=policy,target_mean_bits=target,head=h,output_mse=float(a[0]),output_rel=float(a[1]),score_mse=float(a[2]),k_actual_diag_mse=float(a[3]),v_exact_mse=float(a[4]),v_actual_diag_mse=float(a[5]),v_cross_mse=float(a[6]),k_output_mse=float(a[7]),k_predicted_score_mse=float(pk[h]),v_predicted_mse=float(pv[h]),k_depth2=int((kd[h]==2).sum()),k_depth3=int((kd[h]==3).sum()),k_depth4=int((kd[h]==4).sum()),v_depth2=int((vd[h]==2).sum()),v_depth3=int((vd[h]==3).sum()),v_depth4=int((vd[h]==4).sum()),logical_payload_read_bytes=(readbits+7)//8,padded_plane_read_bytes=readbytes,actual_mean_bits=readbits/(2*8*n*128),K_policy_cpu_ms=kms,V_policy_cpu_ms=vms,timing_scope='CPU allocator only; excludes sync, transfers, selection and attention; not serving latency'))
                self.distrows.append(dict(core,policy=policy,target_mean_bits=target,k_depth_sum=int(kd.sum()),v_depth_sum=int(vd.sum()),padded_plane_read_bytes=readbytes))
        if self.current['case_id']==self.first_case and layer==0 and self.pos==1:
            self.small_oracle(Q,K,V,P,O,rec,curves,scale,core)

    def small_oracle(self,Q,K,V,P,O,rec,curves,scale,core):
        # Local finite oracle only: four fixed units/head0, other units at 3-bit.
        schedules=[a for a in itertools.product(BITS,repeat=4) if sum(a)==12]
        for side in (0,1):
            entries=[]
            for levels in schedules:
                kk=rec[3][0].clone();vv=rec[3][1].clone()
                if side==0:
                    for c,b in enumerate(levels):kk[0,:,c]=rec[b][0][0,:,c]
                else:
                    for t,b in enumerate(levels):vv[0,t,:]=rec[b][1][0,t,:]
                p=(torch.einsum('hgd,htd->hgt',Q,kk)*scale).softmax(-1)
                value=float((p[0]@vv[0]-O[0]).square().mean())
                if side==0:
                    weights=Q[0].double().square().sum(0).cpu().numpy()*scale**2
                    proxy=sum(weights[i]*curves[0,0,i,b-2] for i,b in enumerate(levels))
                else:
                    weights=p[0].double().square().sum(0).cpu().numpy()
                    d=curves[1,0].sum(0)
                    proxy=sum(weights[i]*d[b-2] for i,b in enumerate(levels))
                entries.append(dict(levels=levels,output_mse=value,proxy=float(proxy)))
            selected=min(entries,key=lambda r:r['proxy']);best=min(entries,key=lambda r:r['output_mse'])
            self.oracle.append(dict(core,side='K' if side==0 else 'V',head=0,units=[0,1,2,3],fixed_other_bits=3,candidates=len(entries),proxy_choice=selected,exact_local_oracle=best,absolute_regret=selected['output_mse']-best['output_mse'],scope='four-unit, head0 local schedule enumeration, not global oracle'))

    def replay(self):
        selected=[r for r in self.manifest if r['split']=='test'];self.first_case=selected[0]['case_id']
        for i,r in enumerate(selected):
            marker=self.out/(r['case_id'].replace('/','_')+'_complete.json')
            cstate('controller_causal_replay',completed=i,total=len(selected),case=r['case_id'],smoke=self.smoke)
            if marker.exists():trajectory_check(self.out,r,self.smoke);continue
            self.oracle=[]
            self.capture(r,'evaluate');trajectory_check(self.out,r,self.smoke)
            assert len(self.rows)==3*2*len(TARGETS)*len(POLICIES)*8
            if self.oracle:atomic(self.out/'local_oracle.json',self.oracle)
            atomic(marker,dict(complete=True,rows=len(self.rows)))
        atomic(self.out/'complete.json',dict(complete=True,requests=len(selected),encoding='Jacobian_Tree',seed=SEEDS[0],layers=[0,17,35],positions=[1,64],causal=True,physical_low_bit_kernel=False))
        cstate('controller_replay_completed',requests=len(selected))

    def online_attention(self,module,q,k,v,scale):
        layer=module.layer_idx
        Q=(q[0,:,0].float()@self.U).view(8,4,128)
        K,V=k[0].float()@self.U,v[0].float()@self.U
        rec,_=reconstruct(torch.stack((K,V)),'Jacobian_Tree',self.cbooks[layer])
        if self.online_policy=='full4':
            phat=(torch.einsum('hgd,htd->hgt',Q,rec[4][0])*scale).softmax(-1)
            out=phat@rec[4][1]
        else:
            policy='uniform' if self.online_policy=='uniform3' else 'query_curve'
            out,*_=decide(Q,rec,self.profiles[layer],scale,3.,policy)
        if self.pos in (1,8,32,64):
            reference=(torch.einsum('hgd,htd->hgt',Q,K)*scale).softmax(-1)@V
            self.online_local.append(dict(layer=layer,position=self.pos,output_mse=float((out-reference).square().mean())))
        # Transformers SDPA interface expects [batch, sequence, heads, dim].
        return (out@self.U.T).reshape(1,1,32,128).to(q.dtype)

    @torch.inference_mode()
    def online(self):
        assert json.loads((CROOT/'results'/('smoke_replay' if self.smoke else 'replay')/'analysis_complete.json').read_text())['complete']
        selected=[r for r in self.manifest if r['split']=='test']
        if not self.smoke:
            selected=[r for task in legacy.TASKS for r in [next(x for x in selected if x['dataset']==task),next(x for x in reversed(selected) if x['dataset']==task)]]
        else:selected=selected[:1]
        steps=8 if self.smoke else 64
        summaries=[]
        for i,r in enumerate(selected):
            # Smoke manifest has a 256-token prompt; formal pilot retains 8K cap.
            trajectories={}
            for policy in ('fp16','full4','uniform3','query_curve3'):
                name=r['case_id'].replace('/','_')+'_'+policy+'.json';dest=self.out/name
                if dest.exists():
                    record=json.loads(dest.read_text());trajectories[policy]=record['generated_ids'];summaries.append(record);continue
                cstate('controller_online_closed_loop',completed_cases=i,total_cases=len(selected),case=r['case_id'],policy=policy,steps=steps)
                torch.cuda.reset_peak_memory_stats()
                self.online_policy=policy;self.online_local=[];self.pos=0
                ids=torch.tensor([r['input_ids']],device='cuda');cache=None
                for start in range(0,ids.shape[1],512):
                    output=self.model(input_ids=ids[:,start:start+512],past_key_values=cache,use_cache=True,logits_to_keep=1);cache=output.past_key_values
                nextid=output.logits[:,-1].argmax(-1,keepdim=True);generated=[];torch.cuda.synchronize();t0=time.perf_counter()
                for pos in range(1,steps+1):
                    self.pos=pos;generated.append(int(nextid.item()))
                    output=self.model(input_ids=nextid,past_key_values=cache,use_cache=True,logits_to_keep=1);cache=output.past_key_values
                    nextid=output.logits[:,-1].argmax(-1,keepdim=True)
                torch.cuda.synchronize();elapsed=time.perf_counter()-t0;self.pos=0
                reference=generated if policy=='fp16' else trajectories['fp16']
                mismatch=[j for j,(a,b) in enumerate(zip(generated,reference)) if a!=b]
                record=dict(case=r['case_id'],dataset=r['dataset'],policy=policy,context=r['context'],steps=steps,generated_ids=generated,text=self.tok.decode(generated),token_position_agreement=float(np.mean(np.array(generated)==np.array(reference))),first_divergence=(mismatch[0]+1 if mismatch else None),decode_wall_seconds=elapsed,reference_adapter_TPOT_ms=elapsed*1000/steps,peak_allocated_bytes=torch.cuda.max_memory_allocated(),sampled_local_output_mse=float(np.mean([v['output_mse'] for v in self.online_local])) if self.online_local else 0.,quality_scope='greedy trajectory diagnostics; no official task scoring',performance_scope='reference adapter retains FP16 source cache, CPU allocator, re-encodes and simulates prefixes; not bandwidth or serving speed evidence')
                if policy=='fp16':
                    frozen=json.loads((BASE/'results'/('smoke' if self.smoke else 'full')/(r['case_id'].replace('/','_')+'_trajectory.json')).read_text())['generated_ids'][:steps]
                    assert generated==frozen,'FP16 online reference diverged from frozen trajectory'
                atomic(dest,record);summaries.append(record);trajectories[policy]=generated
                del output,cache;torch.cuda.empty_cache()
        atomic(self.out/'complete.json',dict(complete=True,cases=len(selected),policies=['fp16','full4','uniform3','query_curve3'],steps=steps,results=summaries,official_task_scores=False,serving_performance_validated=False))
        cstate('controller_online_pilot_completed',cases=len(selected),steps=steps)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['replay','online'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    try:
        exp=ControllerExperiment(a.stage,a.smoke)
        exp.replay() if a.stage=='replay' else exp.online()
    except BaseException:
        cstate('failed',traceback=traceback.format_exc());raise
