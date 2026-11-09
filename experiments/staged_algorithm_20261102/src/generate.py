"""Natural-EOS quality adapter; logical traffic only, no hardware claims."""
import argparse
import traceback
import numpy as np
import torch
from common import *
from prefix_codec import load_books, encode, hadamard, metadata
from policy import allocate, curve_view, from_gains
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS

CURVES = ('query_only','global','shared','head','scale','jacobian','scale_jacobian','delta_fp32','delta_fp16','local_gain_fp16')
BLOCK = 128

class Model:
    def __init__(self, codec, policy='fixed', depth=4, curve='global', budget=3, threshold=.05):
        self.codec, self.policy, self.depth = codec, policy, depth
        self.curve, self.budget, self.threshold = curve, budget, threshold
        self.identity = run_id(codec, policy, depth, curve, budget, threshold)
        self.selected_codec = codec
        self.manifest = json.loads((ROOT/'inputs/manifest.json').read_text())
        self.tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        self.U = torch.tensor(hadamard(), device='cuda', dtype=torch.float32)
        if codec != 'fp16':
            self.books, self.luts, self.curves = load_books(codec, 'cuda')
            self.saved_metadata = metadata(codec)
        else:
            self.saved_metadata = {}
        self.model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa').eval().to('cuda')
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        self.prefill = True
        def hooked(module, q, k, v, mask, **kw):
            if self.prefill:
                if self.codec != 'fp16':
                    self.append(module.layer_idx,k,v)
                return original(module,q,k,v,mask,**kw)
            if self.codec == 'fp16':
                self.record(k.shape[2], 16, 16)
                return original(module,q,k,v,mask,**kw)
            self.append(module.layer_idx,k,v)
            return self.attention(module.layer_idx,q,k.shape[2],kw['scaling']),None
        ALL_ATTENTION_FUNCTIONS.register('staged_algorithm', hooked)
        self.model.config._attn_implementation = 'staged_algorithm'
        for layer in self.model.model.layers:
            layer.self_attn.config._attn_implementation = 'staged_algorithm'

    def reset(self):
        self.codes, self.seen, self.local = {}, {}, {}
        self.logical = [0,0]
        self.elements = [0,0]
        self.depth_elements = [dict.fromkeys(range(17),0),dict.fromkeys(range(17),0)]
        self.calls = 0
        self.allocation_diagnostics = dict(negative_edges=0,increasing_gain_pairs=0)
        self.metadata_peak = 0
        self.metadata_written = 0
        self.metadata_read_allowance = 0

    def append(self, layer,k,v):
        n = k.shape[2]
        start = self.seen.get(layer,0)
        if n==start:
            return
        if layer not in self.codes or n>self.codes[layer].shape[2]:
            capacity = min(self.manifest['native_context_length'], max(n, 512 if layer not in self.codes else self.codes[layer].shape[2]*2))
            target = torch.empty((2,8,capacity,128), device='cuda', dtype=torch.uint8)
            if layer in self.codes:
                target[:,:,:start] = self.codes[layer][:,:,:start]
            self.codes[layer] = target
        book = self.books[layer]
        local_mode = self.curve in ('scale','scale_jacobian','local_gain_fp16') and self.policy!='fixed'
        if local_mode:
            # Recompute only changed blocks at write time, including the active tail.
            first = start//BLOCK
            begin = first*BLOCK
            js = (n+BLOCK-1)//BLOCK
            old = self.local.get(layer)
            trailing = (4,) if self.curve=='local_gain_fp16' else ()
            loc = torch.empty((2,8,js,128,*trailing),device='cuda',dtype=torch.float16)
            if old is not None and first:
                loc[:,:,:first] = old[:,:,:first]
        for side,x in enumerate((k[0],v[0])):
            rotated = x[:,start:n].float()@self.U
            z = (rotated-book['mu'][side,:,None,:])/book['sigma'][side,:,None,:]
            sb = {key: value[side] for key,value in book.items()}
            self.codes[layer][side,:,start:n] = encode(z,sb).to(torch.uint8)
            if local_mode:
                for j in range(first,js):
                    end = min(n,(j+1)*BLOCK)
                    raw = x[:,j*BLOCK:end].float()@self.U
                    if self.curve=='local_gain_fp16':
                        code = self.codes[layer][side,:,j*BLOCK:end].long()
                        errors = []
                        h = torch.arange(8,device='cuda')[:,None,None]
                        for b in range(5):
                            rec = self.luts[layer][side][h,b,code]*book['sigma'][side,:,None,:]+book['mu'][side,:,None,:]
                            errors.append((rec-raw).double().square().mean(1))
                        ds = torch.stack(errors,-1)
                        saved = (ds[...,:-1]-ds[...,1:]).half()
                    else:
                        # Relative second moment around the SAME offline mean.
                        # D0 uses the offline mean reconstruction, in physical units.
                        saved = ((raw-book['mu'][side,:,None,:]).double().square().mean(1)/self.curves[layer]['D'][side,...,0].clamp_min(1e-30)).half()
                    if not torch.isfinite(saved).all():
                        raise ValueError('Local FP16 metadata overflow')
                    loc[side,:,j] = saved
                    self.metadata_written += saved.numel()*saved.element_size()
        if local_mode:
            self.local[layer] = loc
            self.metadata_peak = max(self.metadata_peak,sum(a.numel()*a.element_size() for a in self.local.values()))
        self.seen[layer] = n

    def decode(self,layer,n,side,depth):
        code = self.codes[layer][side,:,:n].long()
        h = torch.arange(8,device='cuda')[:,None,None]
        if isinstance(depth,int):
            level = depth
        elif side==0:
            level = depth.repeat_interleave(BLOCK,dim=1)[:,:n,:]
        else:
            level = depth[:,:,None]
        rec = self.luts[layer][side][h,level,code]
        b = self.books[layer]
        return rec*b['sigma'][side,:,None,:]+b['mu'][side,:,None,:]

    def decision(self,pred,costs):
        if self.policy=='budget':
            depth,diag = allocate(pred,costs,budget_bits=int(costs.sum().item())*self.budget)
        else:
            # Relative extra-risk tolerance against the full4 proxy, total per side.
            tolerance = self.threshold*float((pred[...,0]-pred[...,4]).clamp_min(0).sum().item())
            depth,diag = allocate(pred,costs,tolerance=tolerance)
        for k in self.allocation_diagnostics:
            self.allocation_diagnostics[k] += diag[k]
        return depth

    def attention(self,layer,q,n,scale):
        query = (q[0,:,0].float()@self.U).reshape(8,4,128)
        if self.policy=='fixed':
            kd=vd=self.depth
        else:
            ds = curve_view(self.curves[layer],self.curve)
            lens = torch.tensor([min(BLOCK,n-j) for j in range(0,n,BLOCK)],device='cuda')
            kc = lens[None,:,None].expand(8,-1,128)
            kdcurve = ds[0,:,None,:,:].expand(-1,len(lens),-1,-1)
            if self.curve in ('scale','scale_jacobian'):
                kdcurve = kdcurve*self.local[layer][0].double()[...,None]
            elif self.curve=='local_gain_fp16':
                kdcurve = from_gains(self.local[layer][0].double())
            pred = query.double().square().sum(1)[:,None,:,None]*scale**2*kdcurve*kc[...,None]
            kd = self.decision(pred,kc)
            vd = None
        khat = self.decode(layer,n,0,kd)
        phat = (torch.einsum('hgd,htd->hgt',query,khat)*scale).softmax(-1)
        if vd is None:
            vc = torch.full((8,n),128,device='cuda')
            if self.curve in ('scale','scale_jacobian'):
                vcurve = (ds[1,:,None,:,:]*self.local[layer][1].double()[...,None]).sum(2).repeat_interleave(BLOCK,dim=1)[:,:n]
            elif self.curve=='local_gain_fp16':
                vcurve = from_gains(self.local[layer][1].double()).sum(2).repeat_interleave(BLOCK,dim=1)[:,:n]
            else:
                vcurve = ds[1].sum(1)[:,None,:]
            pred = phat.double().square().sum(1)[...,None]*vcurve
            vd = self.decision(pred,vc)
            self.metadata_read_allowance += ds.numel()*ds.element_size()
            if layer in self.local:
                self.metadata_read_allowance += self.local[layer].numel()*self.local[layer].element_size()
        out = (phat@self.decode(layer,n,1,vd))@self.U.T
        self.record(n,kd,vd)
        return out.reshape(1,1,32,128).to(q.dtype)

    def record(self,n,kd,vd):
        self.calls += 1
        for side,depth in enumerate((kd,vd)):
            count = 8*n*128
            self.elements[side] += count
            if isinstance(depth,int):
                self.logical[side] += count*depth
                self.depth_elements[side][depth] += count
            else:
                if side==0:
                    lens = torch.tensor([min(BLOCK,n-j) for j in range(0,n,BLOCK)],device=depth.device)[None,:,None].expand_as(depth)
                    hist = torch.bincount(depth.flatten(),weights=lens.flatten().double(),minlength=5).long()
                else:
                    hist = torch.bincount(depth.flatten(),minlength=5)*128
                counts = hist.tolist()
                for b,num in enumerate(counts):
                    self.depth_elements[side][b] += num
                    self.logical[side] += b*num

    @torch.inference_mode()
    def generate(self,row,checkpoint=None,fingerprint=None):
        self.reset()
        self.prefill = True
        ids = row['input_ids']
        replay=[]
        if checkpoint is not None and checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            if saved.get('case_id')==row['case_id'] and saved.get('fingerprint')==fingerprint and fingerprint is not None:
                assert saved['run']==self.identity and saved['input_ids_sha256']==row['input_ids_sha256']
                replay=saved['generated_ids']
                assert all(isinstance(t,int) and t>=0 for t in replay)
                status('replaying_saved_prefix',run=self.identity,case=row['case_id'],tokens=len(replay))
        maximum = self.manifest['native_context_length']-len(ids)
        assert maximum>0
        assert len(replay)<=maximum, 'Saved prefix exceeds native context boundary'
        begin=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        tensor=torch.tensor([ids],device='cuda')
        cache=None
        for start in range(0,len(ids),512):
            output=self.model(input_ids=tensor[:,start:start+512],past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values
        token=output.logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize()
        prefill_seconds=time.perf_counter()-begin
        self.prefill=False
        generated=[]
        reason='native_context_limit'
        for step in range(maximum):
            t=int(token.item())
            if step<len(replay) and t!=replay[step]:
                raise RuntimeError(f'Resume prefix mismatch at token {step}; saved output not overwritten')
            generated.append(t)
            if t in self.manifest['eos_token_ids']:
                reason='eos'
                break
            if checkpoint is not None and len(generated)%256==0 and len(generated)>=len(replay):
                atomic(checkpoint,dict(case_id=row['case_id'],run=self.identity,fingerprint=fingerprint,
                    input_ids_sha256=row['input_ids_sha256'],generated_ids=generated,complete=False,
                    updated=time.strftime('%Y-%m-%d %H:%M:%S')))
                status('generating',run=self.identity,case=row['case_id'],generated_tokens=len(generated))
            if step+1==maximum:
                break
            output=self.model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values
            token=output.logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        total_elements=sum(self.elements)
        result=dict(case_id=row['case_id'],index=row['index'],dataset=row['dataset'],run=self.identity,
            input_ids_sha256=row['input_ids_sha256'],input_tokens=len(ids),input_truncated=False,
            generated_ids=generated,generated_tokens=len(generated),text=self.tok.decode(generated,skip_special_tokens=True),
            finish_reason=reason,natural_completion=reason=='eos',artificial_output_limit=None,
            resumed_prefix_tokens=len(replay),resume_method='deterministic token-prefix replay to rebuild KV; logical ledger counts the trajectory once',
            native_context_length=self.manifest['native_context_length'],logical_K_V_bits=self.logical,
            logical_KV_read_bits=sum(self.logical),logical_bytes_per_output_token=sum(self.logical)/8/len(generated),
            accessed_K_V_elements=self.elements,element_weighted_mean_bits=sum(self.logical)/total_elements if total_elements else None,
            depth_elements_K_V=self.depth_elements,decode_attention_calls=self.calls,
            saved_book_arrays_bytes=self.saved_metadata,local_metadata_peak_bytes=self.metadata_peak,
            reference_shared_tensor_bytes=(0 if self.codec=='fp16' else
                sum(t.numel()*t.element_size() for b in self.books.values() for t in b.values())+
                sum(t.numel()*t.element_size() for b in self.curves.values() for t in b.values())+
                sum(t.numel()*t.element_size() for t in self.luts.values()))+self.U.numel()*self.U.element_size(),
            local_metadata_write_bytes=self.metadata_written,curve_read_allowance_bytes=self.metadata_read_allowance,
            metadata_access_scope='analytical tensor-use allowance; no DRAM measurement',
            prefill_seconds=prefill_seconds,elapsed_seconds=elapsed,
            reference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocation_diagnostics=self.allocation_diagnostics,
            performance_scope='algorithm reference; uint8 code and FP16 source cache; no hardware saving claim',
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        del output,cache,tensor
        self.codes={};self.local={};self.seen={}
        torch.cuda.empty_cache()
        return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--codec',choices=('fp16',*CODECS),required=True)
    p.add_argument('--policy',choices=('fixed','budget','tolerance'),default='fixed')
    p.add_argument('--depth',type=int,choices=range(5),default=4)
    p.add_argument('--curve',choices=CURVES,default='global')
    p.add_argument('--budget',type=int,choices=range(5),default=3)
    p.add_argument('--threshold',type=float,default=.05)
    p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--exclude-datasets',choices=sorted(json.loads((ROOT/'inputs/manifest.json').read_text())['datasets']),nargs='+')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-tag',default='current')
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    assert sha(ROOT/'inputs/cases.jsonl')==manifest['cases_sha256']
    selected=list(rows(ROOT/'inputs/cases.jsonl'))
    if a.exclude_datasets is not None:
        excluded=set(a.exclude_datasets)
        selected=[r for r in selected if r['dataset'] not in excluded]
    if a.indices is not None:
        if not a.smoke:
            raise ValueError('Subsets require --smoke; production is always 464 cases')
        selected=[r for r in selected if r['index'] in a.indices]
    name=run_id(a.codec,a.policy,a.depth,a.curve,a.budget,a.threshold)
    folder=(ROOT/'smoke'/a.smoke_tag if a.smoke else ROOT/'results')/name
    provenance=dict(codec=a.codec,policy=a.policy,depth=a.depth,curve=a.curve,budget=a.budget,threshold=a.threshold,
        samples_sha256=manifest['cases_sha256'],selected_case_ids_sha256=digest([r['case_id'] for r in selected]),
        source_hashes={p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))},
        model_revision=MODEL.name,book_hashes={} if a.codec=='fp16' else {p.name:sha(p) for p in sorted((ROOT/'codebooks'/a.codec).glob('*.npz'))})
    fingerprint=digest(provenance)
    prov=folder/'provenance.json'
    if prov.exists():
        assert json.loads(prov.read_text())['fingerprint']==fingerprint,'Source/config changed; use a new experiment version'
    else:
        atomic(prov,dict(fingerprint=fingerprint,**provenance))
    missing=[]
    for r in selected:
        dest=folder/f"{r['index']:05d}.json"
        if dest.exists():
            prior=json.loads(dest.read_text())
            assert prior['fingerprint']==fingerprint and prior['input_ids_sha256']==r['input_ids_sha256']
        else:
            missing.append(r)
    if not missing:
        return
    status('loading_model',run=name,remaining=len(missing))
    model=Model(a.codec,a.policy,a.depth,a.curve,a.budget,a.threshold)
    for i,r in enumerate(missing):
        dest=folder/f"{r['index']:05d}.json"
        status('generating',run=name,case=r['case_id'],remaining=len(missing)-i,input_tokens=r['input_tokens'])
        try:
            record=model.generate(r,folder/'active_checkpoint.json',fingerprint)
            record['fingerprint']=fingerprint
            atomic(dest,record)
            status('case_completed',run=name,case=r['case_id'],generated_tokens=record['generated_tokens'],finish_reason=record['finish_reason'])
        except BaseException:
            atomic(folder/'failure.json',dict(case_id=r['case_id'],error=traceback.format_exc(),fingerprint=fingerprint))
            raise
    atomic(folder/'generation_complete.json',dict(cases=len(selected),fingerprint=fingerprint,smoke=a.smoke))

if __name__=='__main__':
    main()
