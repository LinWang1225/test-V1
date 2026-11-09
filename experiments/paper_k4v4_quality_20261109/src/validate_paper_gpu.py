"""Exclusive-GPU conformance check against an independent NumPy oracle."""
import math
import numpy as np
import torch
from common import ROOT, atomic, sha
from turboquant_math import quantize_numpy, dequantize_numpy


@torch.inference_mode()
def validate():
    from matched_prefill import STATE
    from turboquant_backend import PaperTurboQuantImpl
    from turboquant_kernel import attention
    from types import SimpleNamespace
    torch.manual_seed(20261109)
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(ROOT/'references/turboquant_parameters.npz') as data:
        params={k:data[k] for k in data.files}
    n=257
    raw=(torch.randn(2,n,8,128,device='cuda')*2).half()
    cache=torch.empty((4,2,128,8,128),device='cuda',dtype=torch.float16)
    impl=PaperTurboQuantImpl.__new__(PaperTurboQuantImpl)
    table=torch.tensor([[3,1,0,2]],device='cuda',dtype=torch.int32)
    positions=torch.arange(n,device='cuda')
    slots=table[0,positions//128].long()*128+positions%128
    STATE.phase='decode'
    impl.do_kv_cache_update(SimpleNamespace(layer_name='model.layers.0.self_attn'),raw[0],raw[1],cache,slots)
    kcode,kmeta=quantize_numpy(raw[0].cpu().numpy(),params,key=True)
    vcode,vmeta=quantize_numpy(raw[1].cpu().numpy(),params,key=False)
    actual_k=impl.packed[slots//128,0,slots%128].cpu().numpy()
    actual_v=impl.packed[slots//128,1,slots%128].cpu().numpy()
    # Both references have independent arithmetic (FP64 vs FP32). A code
    # mismatch fails visibly instead of silently weakening the comparison.
    np.testing.assert_array_equal(actual_k,kcode)
    np.testing.assert_array_equal(actual_v,vcode)
    np.testing.assert_allclose(impl.norms[slots//128,0,slots%128].cpu().numpy(),kmeta[...,0],rtol=2e-6)
    np.testing.assert_allclose(impl.norms[slots//128,1,slots%128].cpu().numpy(),kmeta[...,1],rtol=2e-5,atol=1e-6)
    query=torch.randn(1,32,128,device='cuda')
    p=impl.parameters
    qr=query@p['RK'].T
    result=attention(qr,qr@p['S'].T,impl.packed,impl.norms,table,
                     torch.tensor([n],device='cuda',dtype=torch.int32),p['C3'],p['C4'],128**-.5)@p['RV']
    k=dequantize_numpy(kcode,kmeta,params,key=True).transpose(1,0,2)
    v=dequantize_numpy(vcode,vmeta,params,key=False).transpose(1,0,2)
    q=query.cpu().numpy().reshape(8,4,128).astype(np.float64)
    logits=q@k.transpose(0,2,1)/math.sqrt(128)
    probs=np.exp(logits-logits.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
    expected=(probs@v).reshape(1,32,128)
    np.testing.assert_allclose(result.cpu().numpy(),expected,rtol=5e-4,atol=5e-4)
    atomic(ROOT/'validation/turboquant_paper_kernel.json',dict(
        passed=True,context_tokens=n,physical_page_order=[3,1,0,2],
        key_3plus1_codes_exact=True,value_4bit_codes_exact=True,
        qjl_attention_matches_full_paper_reconstruction=True,
        max_abs_attention_error=float(np.max(np.abs(result.cpu().numpy()-expected))),
        parameters_sha256=sha(ROOT/'references/turboquant_parameters.npz')))
    STATE.impls.clear()
    del impl,raw,cache,query
    torch.cuda.empty_cache()


if __name__=='__main__':validate()
