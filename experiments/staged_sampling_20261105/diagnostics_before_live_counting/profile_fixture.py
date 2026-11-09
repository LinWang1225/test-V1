"""Replay exactly one captured decode attention with hardware DRAM counters."""
import argparse
from types import SimpleNamespace
import torch
from common import *
from generate import Model
from transformers.integrations.sdpa_attention import sdpa_attention_forward


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixture',required=True)
    parser.add_argument('--condition',choices=['warm','evicted'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(args.fixture,map_location='cpu',weights_only=True)
    q=data['q'].cuda();layer=data['layer'];n=data['context'];scale=data['scale']
    if data['codec']=='fp16':
        k=data['k'].cuda();v=data['v'].cuda()
        module=SimpleNamespace(num_key_value_groups=4,is_causal=True)
        def call():
            return sdpa_attention_forward(module,q,k,v,None,scaling=scale)[0]
    else:
        m=object.__new__(Model)
        for name in ('codec','policy','depth','curve','budget','threshold'):
            setattr(m,name,data[name])
        m.reset()
        m.U=data['U'].cuda();m.codes={layer:data['codes'].cuda()}
        m.books={layer:{a:b.cuda() for a,b in data['book'].items()}}
        m.luts={layer:data['lut'].cuda()}
        m.curves={layer:{a:b.cuda() for a,b in data['curves'].items()}}
        if data['local'] is not None:m.local={layer:data['local'].cuda()}
        def call():
            return m.attention(layer,q,n,scale)
    out=call()
    torch.testing.assert_close(out.cpu(),data['expected_output'],rtol=0,atol=0)
    del out
    eviction=torch.empty(128<<20,dtype=torch.uint8,device='cuda') if args.condition=='evicted' else None
    for _ in range(5):call()
    if eviction is not None:eviction.zero_();eviction.add_(1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('readkv_attention')
    out=call()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(),data['expected_output'],rtol=0,atol=0)
    print('fixture_checked',data['case_id'],data['codec'],args.condition,n,flush=True)


if __name__=='__main__':main()
