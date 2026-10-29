"""Complete greedy generation with causal progressive-KV attention at decode.

Code cache avoids re-encoding old entries. This remains a quality adapter with
FP16 source KV retained, NOT a deployed compressed-cache performance benchmark.
"""
import argparse,sys,time,traceback
from common import *
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
sys.path.insert(0,str(ROOT.parent/'encoding_ablation_20261024/src'))
from codec import hadamard,encode as encode_tree
from adaptive import keys,values


class QualityModel:
    def __init__(self,policy):
        self.policy=policy;self.prefill=True
        self.tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
        self.U=torch.tensor(hadamard(),device='cuda',dtype=torch.float32)
        self.books={};self.lut={};self.curves={}
        if policy!='fp16':
            for layer in range(36):
                with np.load(BOOKS/f'layer{layer}_seed20261024.npz') as z:
                    book={n:torch.tensor(z[n].astype(np.float32),device='cuda') for n in ('mu','sigma')}
                    for n in ('t2','t3','t4','c2','c3','c4'):
                        book[n]=torch.tensor(z['Jacobian_Tree_'+n].astype(np.float32),device='cuda')
                    self.books[layer]=book
                    code=torch.arange(16,device='cuda')
                    self.lut[layer]=torch.stack([book[f'c{b}'][:,:,code>>(4-b)] for b in (2,3,4)],2)
                    self.curves[layer]=torch.tensor(z['controller_D'],device='cuda',dtype=torch.float64)
        status('loading_quality_model',policy=policy)
        self.model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').eval().to('cuda')
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa']
        def hooked(module,q,k,v,mask,**kw):
            if self.policy=='fp16':return self.original(module,q,k,v,mask,**kw)
            layer=module.layer_idx
            self.append_codes(layer,k,v)
            if self.prefill:return self.original(module,q,k,v,mask,**kw)
            return self.attention(layer,q,k.shape[2],kw['scaling']),None
        ALL_ATTENTION_FUNCTIONS.register('full_quality_progressive',hooked)
        self.model.config._attn_implementation='full_quality_progressive'
        for layer in self.model.model.layers:layer.self_attn.config._attn_implementation='full_quality_progressive'

    def reset(self,capacity):
        self.capacity=capacity;self.codes={};self.seen={};self.prefill=True
        self.depth_units=[dict.fromkeys((2,3,4),0) for _ in range(2)];self.depth_elements=[dict.fromkeys((2,3,4),0) for _ in range(2)]
        self.logical_read_bits=0;self.padded_read_bytes=0;self.attention_calls=0

    def append_codes(self,layer,k,v):
        n=k.shape[2];start=self.seen.get(layer,0)
        assert n<=self.capacity and n>=start
        if layer not in self.codes:
            self.codes[layer]=torch.empty((2,8,self.capacity,128),device='cuda',dtype=torch.uint8)
        if start==n:return
        book=self.books[layer]
        for side,x in enumerate((k[0,:,start:],v[0,:,start:])):
            rotated=x.float()@self.U
            z=(rotated-book['mu'][side,:,None,:])/book['sigma'][side,:,None,:]
            tree={key:book[key][side] for key in ('t2','t3','t4')}
            self.codes[layer][side,:,start:n]=encode_tree(z,tree).to(torch.uint8)
        self.seen[layer]=n

    def decode(self,layer,n,side,depth):
        code=self.codes[layer][side,:,:n].long()
        lut=self.lut[layer][side]
        heads=torch.arange(8,device='cuda')[:,None,None]
        if isinstance(depth,int):centroid=lut[heads,depth-2,code]
        else:
            level=(depth-2)[:,None,:] if side==0 else (depth-2)[:,:,None]
            centroid=lut[heads,level,code]
        b=self.books[layer]
        return centroid*b['sigma'][side,:,None,:]+b['mu'][side,:,None,:]

    def attention(self,layer,q,n,scale):
        query=(q[0,:,0].float()@self.U).reshape(8,4,128)
        if self.policy=='full4':kd=vd=4;meanbits=4
        elif self.policy=='uniform3':kd=vd=3;meanbits=3
        else:
            kd=keys(query,self.curves[layer][0],scale,self.policy);vd=None;meanbits=3
        khat=self.decode(layer,n,0,kd)
        phat=(torch.einsum('hgd,htd->hgt',query,khat)*scale).softmax(-1)
        del khat
        if vd is None:vd=values(phat,self.curves[layer][1],self.policy)
        vhat=self.decode(layer,n,1,vd)
        out=(phat@vhat)@self.U.T
        self.attention_calls+=1
        ks=8*128*kd if isinstance(kd,int) else int(kd.sum().item())
        vs=8*n*vd if isinstance(vd,int) else int(vd.sum().item())
        self.logical_read_bits+=ks*n+vs*128
        self.padded_read_bytes+=ks*((n+7)//8)+vs*16
        for side,depth,count in ((0,kd,8*128),(1,vd,8*n)):
            for bit in (2,3,4):
                units=count if isinstance(depth,int) and depth==bit else (0 if isinstance(depth,int) else int((depth==bit).sum().item()))
                self.depth_units[side][bit]+=units
                self.depth_elements[side][bit]+=units*(n if side==0 else 128)
        return out.reshape(1,1,32,128).to(q.dtype)

    @torch.inference_mode()
    def generate(self,row,limit=None):
        ids=row['input_ids'];maximum=row['max_new_tokens'] if limit is None else min(limit,row['max_new_tokens'])
        self.reset(len(ids)+maximum)
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        tensor=torch.tensor([ids],device='cuda');cache=None
        for start in range(0,len(ids),512):
            output=self.model(input_ids=tensor[:,start:start+512],past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values
        nextid=output.logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize();prefill_s=time.perf_counter()-begin;self.prefill=False
        eos=set(json.loads((ROOT/'inputs/manifest.json').read_text())['eos_token_ids'])
        generated=[];reason='length';decode_begin=time.perf_counter()
        for step in range(maximum):
            token=int(nextid.item());generated.append(token)
            if token in eos:reason='eos';break
            if step+1==maximum:break
            output=self.model(input_ids=nextid,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values;nextid=output.logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize();decode_s=time.perf_counter()-decode_begin
        answer=self.tok.decode(generated,skip_special_tokens=True)
        result=dict(case_id=row['case_id'],dataset=row['dataset'],index=row['index'],policy=self.policy,input_ids_sha256=row['input_ids_sha256'],input_tokens=len(ids),original_input_tokens=row['original_input_tokens'],input_truncated=row['input_truncated'],max_new_tokens=maximum,generated_ids=generated,generated_tokens=len(generated),text=answer,finish_reason=reason,hit_output_limit=reason=='length',prefill_seconds=prefill_s,decode_seconds=decode_s,elapsed_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),logical_KV_read_bits=self.logical_read_bits,padded_plane_read_bytes=self.padded_read_bytes,decode_attention_calls=self.attention_calls,encoding='original Jacobian_Tree, seed 20261024' if self.policy!='fp16' else 'FP16',prefill_attention='FP16',all_decode_layers_replaced=self.policy!='fp16',performance_scope='reference quality adapter with FP16 source cache and code cache; not serving performance',completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        result['depth_units_K_V']=self.depth_units
        result['depth_elements_K_V']=self.depth_elements
        total=sum(sum(x.values()) for x in self.depth_elements)
        result['element_weighted_mean_bits']=self.logical_read_bits/total if total else None
        result['encoding']='JacK_MixV, frozen seed 20261024'
        result['traffic_scope']='logical bit-plane payload; not hardware DRAM measurement'
        result['used_in_codec_calibration']=row.get('used_in_codec_calibration',False)
        result['previous_exploratory_case']=row.get('previous_exploratory_case',False)
        del output,cache,tensor
        self.codes={};self.seen={};torch.cuda.empty_cache()
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=POLICIES,required=True);p.add_argument('--dataset',required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--indices',type=int,nargs='*');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(42)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    assert sha(ROOT/'inputs/frozen'/f'{args.dataset}.jsonl')==manifest['datasets'][args.dataset]['sha256']
    rows=list(jsonlines(ROOT/'inputs/frozen'/f'{args.dataset}.jsonl'))
    if args.indices is not None:rows=[r for r in rows if r['index'] in args.indices]
    elif args.pilot:rows=[rows[0]]
    if not args.pilot:assert args.indices is None and args.limit is None,'Full run cannot silently subset or shorten tasks'
    missing=[]
    for r in rows:
        dest=result_path(args.policy,args.dataset,r['index'],args.pilot)
        if dest.exists():
            old=json.loads(dest.read_text());assert old['input_ids_sha256']==r['input_ids_sha256']
            assert old['policy']==args.policy and old['max_new_tokens']==(r['max_new_tokens'] if args.limit is None else min(args.limit,r['max_new_tokens']))
        else:missing.append(r)
    if not missing:status('generation_already_complete',policy=args.policy,dataset=args.dataset,count=len(rows));return
    model=QualityModel(args.policy)
    for i,r in enumerate(missing):
        status('full_generation' if not args.pilot else 'pilot_generation',policy=args.policy,dataset=args.dataset,remaining=len(missing)-i,case=r['case_id'],input_tokens=r['input_tokens'],max_new_tokens=r['max_new_tokens'])
        try:
            record=model.generate(r,args.limit)
            atomic(result_path(args.policy,args.dataset,r['index'],args.pilot),record)
            print('finished_case',r['case_id'],record['finish_reason'],record['generated_tokens'],round(record['elapsed_seconds'],3),flush=True)
        except BaseException:
            status('failed',policy=args.policy,dataset=args.dataset,case=r['case_id'],traceback=traceback.format_exc());raise
    status('dataset_generation_completed',policy=args.policy,dataset=args.dataset,cases=len(rows),pilot=args.pilot)


if __name__=='__main__':main()
