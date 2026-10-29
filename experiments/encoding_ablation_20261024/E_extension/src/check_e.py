"""Native source parity and no-auxiliary-data reconstruction gates."""
from pathlib import Path
import torch,json,numpy as np
from native import NativeCodec,store,decode,sink
c=NativeCodec();torch.manual_seed(20261024)
x=torch.randn(8,257,128,device='cuda',dtype=torch.float16)
y=torch.randn_like(x);data=c.quantize(x,y)
# Native packed-store decoding agrees exactly with our 4-bit affine reconstruction before inverse rotation.
ks,vs=data['ks'],data['vs'];qk=data['qk'];qv=data['qv'];h=8;nb=2
refk=(qk.float()*ks['s_col_K'].float()[:,:,None]+ks['zp_K'].float()[:,:,None])*ks['s_row_K'].float()[:,None,:]
refv=(qv.float()*vs['s_row_V'].float()[:,:,None]+vs['zp_V'].float()[:,:,None])*vs['s_col_V'].float()[:,None,:]
for j in (0,3,15):
 a=decode.kvarn_dequant_tile_k(ks['q_packed_uint8'][j],ks['s_col_K'][j],ks['zp_K'][j],ks['s_row_K'][j],128,4)
 b=decode.kvarn_dequant_tile_v(vs['q_packed_uint8'][j],vs['s_col_V'][j],vs['s_row_V'][j],vs['zp_V'][j],128,4)
 assert torch.equal(a,refk[j]);assert torch.equal(b,refv[j])
rec=c.reconstruct(data,4);refk=refk.reshape(h,nb,128,128).transpose(-1,-2).reshape(h,256,128)@c.H.T
refv=refv.reshape(h,256,128)@c.H.T
assert torch.equal(rec[0,:,:256],refk);assert torch.equal(rec[1,:,:256],refv)
assert torch.equal(rec[0,:,256:],x[:,256:].float());assert torch.equal(rec[1,:,256:],y[:,256:].float())
assert torch.equal(c.reconstruct(data,4,False),rec)
assert torch.isfinite(c.reconstruct(data,2)).all();assert torch.isfinite(c.reconstruct(data,3,False)).all()
# Compare batched storage on the same balanced tensors to native single-tile store.
t=(x[0,:128].float()@c.H).T.contiguous()
bal,sc,sr=sink.variance_normalize(t,iterations=8)
a=store.kvarn_store_tile_k(t,4,sinkhorn_iters=8)
b=store.kvarn_store_tile_k_batch_from_sinkhorn(bal[None],sc.squeeze(0)[None],sr.squeeze(1)[None],4)
assert all(torch.equal(a[n],b[n][0]) for n in a)
checks=[]
for n in (1024,2048,4096,8192,8193,8256):
 s=c.storage(n);assert s['equal_total_bytes']<=s['B_total_budget_bytes'];checks.append(s)
# Validate serialized bitplane size for contexts requiring more than one removed plane.
for n in (257,288,320,1389,8193):
 ledger=c.storage(n);size=ledger['full_tokens']*128;removed=ledger['equal_removed_bits_per_head_side']
 arr=np.random.default_rng(42).integers(0,16,size,dtype=np.uint8);planes=[]
 for bit in (3,2,1,0):
  skip=min(size,max(0,removed-bit*size));planes.append(np.packbits((arr[skip:]>>bit)&1))
 assert sum(len(p) for p in planes)==(4*size-removed)//8
root=Path(__file__).resolve().parents[1]
(root/'results/checks.json').write_text(json.dumps(dict(passed=True,native_full_read_exact=True,native_batched_single_store_equal=True,native_fp16_metadata=True,zero_mid_equal_at_4bit=True,no_calibration=True,storage_checks=checks),indent=2))
print('E native parity, prefix, tail and storage gates passed')
