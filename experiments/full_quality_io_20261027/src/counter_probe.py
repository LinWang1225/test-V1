"""Small actual counter workload; not a substitute for request measurements."""
import torch
torch.set_num_threads(2)
x=torch.ones(1048576,device='cuda');torch.cuda.synchronize()
torch.cuda.nvtx.range_push('readkv_attention')
y=x+1
torch.cuda.nvtx.range_pop();torch.cuda.synchronize()
assert y.shape==x.shape
