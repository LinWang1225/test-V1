"""Validate actual uint8 paged GPU writers/readers for all five codecs."""
import numpy as np
import torch
from common import *
from prefix_codec import encode,make_lut
from tree_kernels import store,decode

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(51)
    torch.backends.cuda.matmul.allow_tf32=False
    results={}
    for method in CODECS:
        with np.load(ROOT/'codebooks'/method/'layer0.npz') as a:
            book={k:torch.tensor(a[k].astype(np.float32),device='cuda') for k in a.files if k not in ('D','J')}
        if 'lo' in book:
            book['c4']=book['lo'][...,None]+torch.arange(16,device='cuda',dtype=torch.float32)/15*(book['hi']-book['lo'])[...,None]
        n=257
        x=torch.randn(2,n,8,128,device='cuda')*book['sigma'][:,None]+book['mu'][:,None]
        cache=torch.zeros(4,2,128,8,128,device='cuda',dtype=torch.uint8)
        table=torch.tensor([[3,1,0,2]],device='cuda',dtype=torch.int32)
        t=torch.arange(n,device='cuda');slots=(table[0,t//128].long()*128+t%128)
        count=torch.zeros(2,device='cuda',dtype=torch.int64)
        store(x[0],x[1],slots,cache,book,count)
        reconstructed=[]
        for side in (0,1):
            z=((x[side]-book['mu'][side])/book['sigma'][side]).transpose(0,1)
            sb={k:v[side] for k,v in book.items()}
            word=encode(z,sb).transpose(0,1).to(torch.uint8)
            packed=word[...,::2]|(word[...,1::2]<<4)
            assert torch.equal(cache[slots//128,side,slots%128],word),method+' uint8 writer mismatch'
            lut=make_lut(sb)
            h=torch.arange(8,device='cuda')[None,:,None]
            reconstructed.append(lut[h,4,word.long()]*book['sigma'][side]+book['mu'][side])
        q=torch.randn(1,32,128,device='cuda')
        actual=decode(q,cache,table,torch.tensor([n],device='cuda',dtype=torch.int32),book,count,128**-.5)
        k,v=[r.transpose(0,1) for r in reconstructed]
        query=q.reshape(8,4,128)
        expected=((query@k.transpose(-1,-2)*128**-.5).softmax(-1)@v).reshape(1,32,128)
        torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-4)
        assert count.tolist()==[1,n]
        results[method]=dict(uint8_codes_exact=True,paged_attention_close=True,max_abs_error=(actual-expected).abs().max().item())
    atomic(ROOT/'checks_gpu.json',dict(passed=True,results=results,paged_tail_context=257,physical_page_order=[3,1,0,2]))
    print('All five vLLM uint8 GPU writer/reader checks passed',flush=True)

if __name__=='__main__':main()
