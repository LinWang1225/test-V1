"""GPU prefix, causal metadata and logical-ledger audits without test answers."""
import numpy as np
import torch
from common import *
from generate import Model

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261102)
    torch.backends.cuda.matmul.allow_tf32=False
    m=Model('Jacobian_Tree')
    n=257
    samples=[]
    for side in (0,1):
        raw=m.books[0]['mu'][side,:,None,:]+m.books[0]['sigma'][side,:,None,:]*torch.randn(8,n,128,device='cuda')
        samples.append((raw@m.U.T).half()[None])
    k,v=samples
    q=torch.randn(1,32,1,128,device='cuda',dtype=torch.float16)
    scale=128**-.5
    results={}
    for mode in ('global','scale','jacobian','scale_jacobian','delta_fp32','delta_fp16','local_gain_fp16'):
        m.curve=mode;m.policy='budget';m.budget=3;m.reset()
        # Encode 127 then extend across full and partial blocks, as in causal decode.
        m.append(0,k[:,:,:127],v[:,:,:127])
        m.append(0,k,v)
        assert m.seen[0]==n
        out=m.attention(0,q,n,scale)
        assert out.shape==(1,1,32,128) and torch.isfinite(out).all()
        assert sum(m.elements)==2*8*n*128
        assert sum(m.logical)<=3*sum(m.elements)
        assert sum(sum(x.values()) for x in m.depth_elements)==sum(m.elements)
        assert sum(b*count for x in m.depth_elements for b,count in x.items())==sum(m.logical)
        results[mode]=dict(logical_bits=m.logical,metadata_bytes=m.metadata_peak)
    m.policy='fixed';m.reset();m.append(0,k,v)
    for b in range(5):
        m.depth=b
        actual=m.attention(0,q,n,scale)
        khat=m.decode(0,n,0,b);vhat=m.decode(0,n,1,b)
        query=(q[0,:,0].float()@m.U).reshape(8,4,128)
        expected=(((torch.einsum('hgd,htd->hgt',query,khat)*scale).softmax(-1)@vhat)@m.U.T).reshape(1,1,32,128).half()
        assert torch.equal(actual,expected)
    # Integer varying depths must choose exactly the corresponding fixed-prefix values.
    kd=torch.randint(0,5,(8,3,128),device='cuda')
    vd=torch.randint(0,5,(8,n),device='cuda')
    for side,depth in ((0,kd),(1,vd)):
        mixed=m.decode(0,n,side,depth)
        token_depth=depth.repeat_interleave(128,1)[:,:n] if side==0 else depth[:,:,None].expand_as(mixed)
        for b in range(5):
            fixed=m.decode(0,n,side,b)
            assert torch.equal(mixed[token_depth==b],fixed[token_depth==b])
    atomic(ROOT/'checks_gpu.json',dict(passed=True,context=n,tail_block_test=True,depths=list(range(5)),
        integer_mixed_decode_exact=True,curve_modes=results,source_hashes={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}))
    print('GPU prefix/metadata/ledger audits passed',flush=True)

if __name__=='__main__':
    main()
