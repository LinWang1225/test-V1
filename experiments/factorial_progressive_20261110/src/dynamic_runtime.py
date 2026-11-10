import re,time,math,inspect
from pathlib import Path
import numpy as np,torch
from common import ROOT
from transforms import normalization_stats
from progressive_kernels import encode,decode
from allocation import allocate,histogram,IDEAL
ACTIVE=None
class Runtime:
 def __init__(self,identity,controller=None,budget=3,uniform=None,metrics=True,cost_mode='payload'):
  self.e,self.b,self.r=identity.split('_');self.ri=int(self.r[1:]);self.controller=controller;self.budget=budget;self.uniform=uniform;self.metrics=metrics;self.cost_mode=cost_mode
  self.assets={};self.states={};self.hist=torch.zeros((2,5),device='cuda',dtype=torch.int64)
  self.counters={'payload_codeword_write_bytes':0,'local_metadata_write_bytes':0,'decision_metadata_access_bytes':0,'decision_index_write_bytes':0,'reencoding_elements':0,'raw_open_buffer_peak_bytes':0,'local_metadata_peak_bytes':0,'allocation_host_seconds':0.,'steps':0,'initial_encoding_elements':0,'new_decode_encoding_elements':0,'rewritten_existing_elements':0,'encoding_buffer_write_bytes':0,'encoding_buffer_read_bytes':0,'version_update_write_bytes':0}
  self.errors=[];self.events=[];self.global_bytes=0;self.index_peak=0;self.traffic=[0,0];self.total_caps=[0,0]
  shared_u=torch.from_numpy(np.load(ROOT/'calibration/basis/hadamard.npy')).cuda()
  for li in range(36):
   with np.load(ROOT/'calibration/basis'/f'layer{li:02}.npz') as a:
    matrices=[None,None] if self.b=='B0' else ([shared_u,shared_u] if self.b=='B1' else [torch.from_numpy(a['B2'][s]).cuda() for s in range(2)])
    mu=torch.from_numpy(a[f'mu_{self.b}']).cuda();sg=torch.from_numpy(a[f'sigma_{self.b}']).cuda()
   with np.load(ROOT/'codebooks'/identity/f'layer{li:02}.npz') as a:
    trees=[([torch.from_numpy(a[f't{depth}_{s}']).cuda() for depth in [2,3,4]],torch.from_numpy(a[f'lut_{s}']).cuda()) for s in range(2)]
    curve=None if controller in [None,'C0'] else torch.from_numpy(a['jacobian' if controller=='C3' else 'mse']).cuda()
    energy=torch.from_numpy(a['energy']).cuda() if controller=='C2' else None
   asset={'basis':matrices,'mu':mu,'sg':sg,'trees':trees,'curve':curve,'energy':energy}
   self.assets[li]=asset
   tensors=[mu,sg]+[v for tr,lut in trees for v in tr+[lut]]
   if curve is not None:tensors.append(curve)
   if energy is not None:tensors.append(energy)
   self.global_bytes+=sum(v.numel()*v.element_size() for v in tensors)
   if self.b=='B2':self.global_bytes+=sum(v.numel()*v.element_size() for v in matrices)
  if self.b=='B1':self.global_bytes+=shared_u.numel()*shared_u.element_size()
 def rotate(self,x,matrix,inverse=False):
  if matrix is None:return x.float()
  m=matrix.transpose(-1,-2) if inverse else matrix
  return torch.matmul(x.float(),m)
 def _newstate(self,li,t):
  a=self.assets[li];state={'T':0,'side':[],'versions':torch.zeros((2,321),device='cuda',dtype=torch.int32)}
  for s in range(2):
   group=32 if self.ri==3 and s==1 else 128;ng=math.ceil(40960/group)
   state['side'].append({'code':torch.empty((8,40960,128),device='cuda',dtype=torch.uint8),
      'mu':torch.empty((8,ng,128),device='cuda',dtype=torch.float16) if self.ri>=2 else torch.empty((1,1,1),device='cuda',dtype=torch.float16),
      'sg':torch.empty((8,ng,128),device='cuda',dtype=torch.float16) if self.ri>=1 else torch.empty((1,1,1),device='cuda',dtype=torch.float16),
      'r':torch.empty((8,320,128),device='cuda',dtype=torch.float16) if self.controller=='C2' else None,'group':group,'raw_open':None})
  self.states[li]=state;return state
 def _encode_range(self,li,s,rot,start,initial=False):
  a=self.assets[li];state=self.states[li]['side'][s];group=state['group'];n=rot.shape[1]
  mu,sg,en,g=normalization_stats(rot,a['mu'][s].float(),a['sg'][s].float(),self.r,s)
  assert g==group,(self.r,s,g,group)
  # Actual normalization parameters are committed before decoding in this serial CUDA stream.
  groups=mu.shape[1];offset=start//group
  if self.ri>=2:state['mu'][:,offset:offset+groups]=mu.half()
  if self.ri>=1:
   params=sg/a['sg'][s].float()[:,None,:] if self.ri==1 else sg
   state['sg'][:,offset:offset+groups]=params.half()
   self.counters['local_metadata_write_bytes']+=8*groups*128*2*(1 if self.ri==1 else 2)
  mu=mu.repeat_interleave(group,dim=1)[:,:n];sg=sg.repeat_interleave(group,dim=1)[:,:n]
  encode(((rot-mu)/sg).contiguous(),state['code'],a['trees'][s][0],start)
  self.counters['payload_codeword_write_bytes']+=rot.numel()
  self.counters['encoding_buffer_read_bytes']+=rot.numel()*4
  if not initial:
   self.counters['new_decode_encoding_elements']+=8*128
   self.counters['rewritten_existing_elements']+=max(0,rot.numel()-8*128)
  self.counters['initial_encoding_elements' if initial else 'reencoding_elements']+=rot.numel()
 def _update_r(self,li,s,rot,start):
  if self.controller!='C2':return
  a=self.assets[li];state=self.states[li]['side'][s];n=rot.shape[1]
  # Local risk statistics are separate from encoding parameters, including R0 and R1.
  mu,sg,en,g=normalization_stats(rot,a['mu'][s].float(),a['sg'][s].float(),self.r,s)
  if g==32:
   expanded=en.repeat_interleave(g,dim=1)[:,:n]
   pad=(-n)%128;expanded=torch.nn.functional.pad(expanded,(0,0,0,pad))
   lengths=torch.arange(math.ceil(n/128),device='cuda')*128;lengths=(n-lengths).clamp(1,128)
   en=expanded.reshape(8,-1,128,128).sum(2)/lengths[None,:,None]
  ratio=en/a['energy'][s,:,None,:].clamp_min(1e-20)
  ratio=ratio.clamp(0,65504).half()
  state['r'][:,start//128:start//128+ratio.shape[1]]=ratio
  self.counters['local_metadata_write_bytes']+=ratio.numel()*2
 def update(self,li,cache,meta):
  from native_backend import STATE
  if li==0 or not hasattr(self,'_step_t'):self._step_t=int(meta.seq_lens[0])
  t=self._step_t;a=self.assets[li];first=li not in self.states
  state=self._newstate(li,t) if first else self.states[li]
  if not first:assert t==state['T']+1,(li,t,state['T'])
  table=meta.block_table[0]
  if first:
   page_ids=table[:math.ceil(t/128)].long();pages=cache.index_select(0,page_ids)
   for s in range(2):
    raw=pages[:,s].reshape(-1,8,128)[:t].transpose(0,1).float();rot=self.rotate(raw,a['basis'][s])
    self._encode_range(li,s,rot,0,initial=True);self._update_r(li,s,rot,0)
    state['side'][s]['raw_open']=rot[:,((t-1)//128)*128:].clone()
  else:
   block=(t-1)//128;offset=(t-1)%128;page=cache.index_select(0,table[block:block+1].long())[0]
   for s in range(2):
    side=state['side'][s];new=self.rotate(page[s,offset].float()[:,None,:],a['basis'][s])
    side['raw_open']=new if offset==0 else torch.cat([side['raw_open'],new],dim=1)
    group=side['group'];length=((t-1)%group)+1
    if self.ri==0:self._encode_range(li,s,new,t-1)
    else:self._encode_range(li,s,side['raw_open'][:,-length:],t-length)
    self._update_r(li,s,side['raw_open'],block*128)
  state['T']=t;state['versions'][:,(t-1)//128]+=1
  self.counters['encoding_buffer_write_bytes']+=sum(side['raw_open'].numel()*side['raw_open'].element_size() for side in state['side'])
  self.counters['version_update_write_bytes']+=8
  local=sum(sum(x.numel()*x.element_size() for x in [s['mu'],s['sg']] if x.numel()>1)+(0 if s['r'] is None else s['r'].numel()*2) for st in self.states.values() for s in st['side'])
  raw=sum(s['raw_open'].numel()*s['raw_open'].element_size() for st in self.states.values() for s in st['side'])
  self.counters['local_metadata_peak_bytes']=max(local,self.counters['local_metadata_peak_bytes']);self.counters['raw_open_buffer_peak_bytes']=max(raw,self.counters['raw_open_buffer_peak_bytes'])
  return t
 def _read(self,li,s,depth,t):
  a=self.assets[li];ss=self.states[li]['side'][s]
  return decode(ss['code'],a['trees'][s][1],depth,a['mu'][s],a['sg'][s],ss['mu'],ss['sg'],t,ss['group'],self.ri,s)
 def _risk(self,li,s,q,phat,t):
  a=self.assets[li];p=math.ceil(t/128)
  imp=(q.square().sum(1)*self.scale**2) if s==0 else phat.square().sum(1)
  if self.controller=='C0':
   ideal=torch.tensor(IDEAL,device='cuda')
   if s==0:return (imp*a['sg'][0].float().square())[:,None,:,None].expand(-1,p,-1,5)*ideal
   return imp[:,:,None]*a['sg'][1].float().square().sum(-1)[:,None,None]*ideal
  curve=a['curve'][s]
  if s==0:
   risk=imp[:,None,:,None]*curve[:,None,:,:]
   risk=risk.expand(-1,p,-1,-1)
   if self.controller=='C2':risk=risk*self.states[li]['side'][0]['r'][:,:p].float()[:,:,:,None]
  else:
   if self.controller=='C2':
    rr=self.states[li]['side'][1]['r'][:,:p].float()
    d=torch.einsum('hpc,hcb->hpb',rr,curve).repeat_interleave(128,dim=1)[:,:t]
   else:d=curve.sum(1)[:,None,:]
   risk=imp[:,:,None]*d
  self.counters['decision_metadata_access_bytes']+=curve.numel()*curve.element_size()
  if self.controller=='C2':self.counters['decision_metadata_access_bytes']+=8*p*128*2
  return risk
 def reset_case(self):
  self.states.clear();self.hist.zero_();self.errors=[];self.events=[];self.index_peak=0;self.traffic=[0,0];self.total_caps=[0,0]
  self.counters={k:(0. if isinstance(v,float) else 0) for k,v in self.counters.items()}
  torch.cuda.reset_peak_memory_stats()
 def metadata_read_cost(self,s,t,worst=False):
  p=math.ceil(t/128);group=32 if s==1 and (worst or self.ri==3) else 128
  basis=2*8*128*128*4 if worst or self.b=='B2' else (2*128*128*4 if self.b=='B1' else 0)
  local=8*math.ceil(t/group)*128*(4 if worst or self.ri>=2 else (2 if self.ri==1 else 0))
  curve=8*128*5*4 if worst or self.controller not in [None,'C0'] else 0
  ratio=8*p*128*2 if worst or self.controller=='C2' else 0
  indices=8*p*128 if s==0 else 8*t
  return basis+local+curve+ratio+indices+8*128*4+8*31*2
 def payload_cap(self,s,t):
  meta=self.metadata_read_cost(s,t);cap=math.floor(self.budget*8*t*128)
  if self.cost_mode=='total':cap+=8*(self.metadata_read_cost(s,t,True)-meta)
  self.traffic[s]+=meta
  self.total_caps[s]+=math.floor(self.budget*8*t*128)+8*self.metadata_read_cost(s,t,True)
  return cap
 def attention(self,li,query,cache,meta,scale):
  self.scale=scale;t=self.update(li,cache,meta);a=self.assets[li]
  q=self.rotate(query[0].reshape(8,4,128),a['basis'][0]);p=math.ceil(t/128)
  nk=(t-torch.arange(p,device='cuda')*128).clamp(1,128)[None,:,None]
  vk=torch.tensor(128,device='cuda',dtype=torch.int64)
  depths=[];start=time.perf_counter();ev=None
  from native_backend import STATE
  if STATE.decode_position in [1,8,32,64]:
   ev=[torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)];ev[0].record()
  cap_k=self.payload_cap(0,t);cap_v=self.payload_cap(1,t)
  if self.uniform is not None:bk=torch.full((8,p,128),self.uniform,device='cuda',dtype=torch.uint8)
  else:bk=allocate(self._risk(li,0,q,None,t)*nk[:,:,:,None],nk,cap_k)
  k=self._read(li,0,bk,t)
  score=torch.einsum('hgd,htd->hgt',q,k)*scale;phat=torch.softmax(score,dim=-1)
  if self.uniform is not None:bv=torch.full((8,t),self.uniform,device='cuda',dtype=torch.uint8)
  else:bv=allocate(self._risk(li,1,q,phat,t),vk,cap_v)
  if ev:
   ev[1].record();self.events.append(ev)
  self.counters['allocation_host_seconds']+=time.perf_counter()-start
  v=self._read(li,1,bv,t);out=self.rotate(torch.einsum('hgt,htd->hgd',phat,v),a['basis'][1],inverse=True)
  self.hist[0]+=histogram(bk,nk);self.hist[1]+=histogram(bv,vk)
  indexbytes=bk.numel()+bv.numel();self.index_peak=max(self.index_peak,indexbytes)
  self.counters['decision_index_write_bytes']+=indexbytes
  self.counters['steps']+=1
  if self.metrics and STATE.decode_position in [1,8,32,64]:self.measure(li,q,cache,meta,k,v,phat,out,t)
  return out.reshape_as(query[0]).to(query.dtype)
 def measure(self,li,q,cache,meta,k,v,prob,out,t):
  # Observer runs after decisions. It never returns data to the controller.
  a=self.assets[li];pages=cache.index_select(0,meta.block_table[0,:math.ceil(t/128)].long())
  truth=[self.rotate(pages[:,s].reshape(-1,8,128)[:t].transpose(0,1).float(),a['basis'][s]) for s in range(2)]
  true_score=torch.einsum('hgd,htd->hgt',q,truth[0])*self.scale;true_p=torch.softmax(true_score,dim=-1)
  true_o=self.rotate(torch.einsum('hgt,htd->hgd',true_p,truth[1]),a['basis'][1],inverse=True)
  values=[]
  for b in range(5):
   dk=self._read(li,0,torch.full((8,math.ceil(t/128),128),b,device='cuda',dtype=torch.uint8),t)
   dv=self._read(li,1,torch.full((8,t),b,device='cuda',dtype=torch.uint8),t)
   values.extend([(dk-truth[0]).square().sum(),(dv-truth[1]).square().sum()])
  values.extend([(k-truth[0]).square().sum(),(v-truth[1]).square().sum(),(torch.einsum('hgd,htd->hgt',q,k)*self.scale-true_score).square().sum(),(prob-true_p).square().sum(),(out-true_o).square().sum()])
  self.errors.append({'layer':li,'position':__import__('native_backend').STATE.decode_position,'elements_kv':8*t*128,'elements_score':32*t,'elements_output':32*128,'sse':torch.stack(values)})
 def finish(self):
  torch.cuda.synchronize();hist=self.hist.cpu().tolist();bits=[sum(b*n for b,n in enumerate(side)) for side in hist]
  count=[sum(side) for side in hist]
  if self.uniform is None and self.cost_mode=='payload':assert all(n<=self.budget*d+1e-6 for n,d in zip(bits,count))
  if self.cost_mode=='total':assert all(bits[s]+8*self.traffic[s]<=self.total_caps[s] for s in range(2))
  errors=[]
  for e in self.errors:errors.append({**{k:v for k,v in e.items() if k!='sse'},'sse':e['sse'].cpu().tolist()})
  return {'stage':'logical_algorithm','physical_payload_storage':'uint8 codeword; true DRAM bit-plane measurement pending','actual_element_weighted_bits':dict(zip(['K','V'],[n/d if d else 0 for n,d in zip(bits,count)])),
   'depth_element_counts':dict(zip(['K','V'],hist)),'logical_payload_read_bits':dict(zip(['K','V'],bits)),
   'cost_mode':self.cost_mode,'logical_metadata_read_bytes':dict(zip(['K','V'],self.traffic)),'logical_payload_plus_metadata_bits':dict(zip(['K','V'],[bits[s]+8*self.traffic[s] for s in range(2)])),'matched_total_cost_ceiling_bits':dict(zip(['K','V'],self.total_caps)),'matched_total_reference':'same T, b-bit payload plus max B2/R3/C2 metadata per side; integer unused budget allowed; logical access model, not measured DRAM','global_resident_metadata_bytes':self.global_bytes,'decision_index_peak_bytes':self.index_peak,'allocation_gpu_sample_milliseconds':[a.elapsed_time(b) for a,b in self.events],
   'allocation_time_scope':'host launch includes K reconstruction and attention estimate; CUDA samples likewise, positions 1/8/32/64',
   'counters':self.counters,'errors_after_decision':errors,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
   'metadata_representation':'FP16 local parameters and risk ratios, FP32 bases and curves, uint8 depth indices; int32 versions','live_raw_encoding_buffer':'FP32 open 128-token block per side/layer; counted separately',
   'version_management_bytes':36*2*321*4,'all_zero_semantics':'no payload load, prescribed mean, token retained in softmax'}

def attach(runtime):
 global ACTIVE
 ACTIVE=runtime
 from native_backend import CaptureBackend,CaptureImpl,STATE
 signature=inspect.signature(CaptureImpl.forward)
 class DynamicImpl(CaptureImpl):
  def forward(self,*args,**kwargs):
   if ACTIVE is None or not STATE.active or STATE.phase!='decode':return super().forward(*args,**kwargs)
   bound=signature.bind(self,*args,**kwargs);bound.apply_defaults();v=bound.arguments
   layer=v['layer'];name=getattr(layer,'layer_name',getattr(layer,'prefix',''))
   match=re.search(r'layers\.(\d+)\.',name);assert match,name;li=int(match.group(1))
   STATE.layer_calls[li]=STATE.layer_calls.get(li,0)+1
   result=ACTIVE.attention(li,v['query'],v['kv_cache'],v['attn_metadata'],self.scale)
   output=v['output'];output[0].copy_(result);return output
 CaptureBackend.get_impl_cls=staticmethod(lambda *args,**kwargs:DynamicImpl)
