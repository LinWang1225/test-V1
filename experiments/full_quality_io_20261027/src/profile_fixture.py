"""ReadKV Appendix G.2 boundary: existing cache, complete single eager call.

No model weights or cache construction inside NVTX range. One application run
can be replayed by ncu; it never skips measured work because output exists.
"""
import argparse
from types import SimpleNamespace
from common import *
import torch
from generate import QualityModel
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--condition',choices=['warm','evicted'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(a.fixture,map_location='cpu',weights_only=True)
    policy=data['policy'];q=data['q'].cuda();layer=data['layer'];n=data['context'];scale=data['scale']
    if policy=='fp16':
        k=data['k'].cuda();v=data['v'].cuda();module=SimpleNamespace(num_key_value_groups=4,is_causal=True)
        def call():return sdpa_attention_forward(module,q,k,v,None,scaling=scale)[0]
    else:
        m=object.__new__(QualityModel);m.policy=policy;m.U=data['U'].cuda();m.codes={layer:data['codes'].cuda()}
        m.books={layer:{k:v.cuda() for k,v in data['book'].items()}};m.lut={layer:data['lut'].cuda()};m.curves={layer:data['curves'].cuda()}
        m.logical_read_bits=0;m.padded_read_bytes=0;m.attention_calls=0
        def call():return m.attention(layer,q,n,scale)
    with torch.inference_mode():
        out=call()
        torch.testing.assert_close(out.cpu(),data['expected_output'],rtol=2e-3,atol=2e-3)
        del out
        eviction=torch.empty(128<<20,dtype=torch.uint8,device='cuda') if a.condition=='evicted' else None
        for _ in range(5):call()
        if eviction is not None:eviction.zero_();eviction.add_(1)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push('readkv_attention')
        out=call()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        # Outside the measured region, no use of profiler-reported duration.
        assert torch.isfinite(out).all()
    print('fixture_checked',data['case_id'],policy,a.condition,n,flush=True)


if __name__=='__main__':main()
