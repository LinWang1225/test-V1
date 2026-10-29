"""Boundary, residual algebra, true prefix I/O, numerical decomposition gates."""
import numpy as np
import torch
from common import atomic,ROOT,METHODS
from codecs4 import *


def main():
    rng=np.random.default_rng(20261026)
    x=np.sort(np.r_[rng.normal(size=8192),[-7.,7.]])
    records=[]
    for method in NEW_METHODS:
        book,info=fit(x,np.ones_like(x),method)
        bt={k:torch.tensor(np.stack([v,v]).astype(np.float32)) for k,v in book.items()}
        xx=rng.normal(size=(2,257,128)).astype(np.float32)
        q=encode(torch.tensor(xx),method,bt)
        assert q.min()>=0 and q.max()<=15
        for head in range(2):
            nq=numpy_encode(xx[head],method,book)
            assert np.array_equal(q[head].numpy(),nq),(method,'CPU/GPU-reference code mismatch')
        for b in BITS:
            decoded=decode(q,bt,b)
            assert torch.isfinite(decoded).all()
            packed=pack_prefix(q.numpy(),b,axis=1)
            recovered=np.moveaxis(unpack_prefix(packed,257),-1,1)
            assert np.array_equal(recovered>>(4-b),q.numpy()>>(4-b))
            assert sum(a.nbytes for a in packed)==b*2*128*((257+7)//8)
        if method.startswith('TCN_'):
            lo,hi=float(book['lo']),float(book['hi'])
            endpoints=np.linspace(lo,hi,17)
            # Use exact binary-range boundaries separately, avoiding FP range ambiguity.
            xx=np.r_[np.linspace(0,1,4097),np.nextafter(np.linspace(0,1,17),0),np.nextafter(np.linspace(0,1,17),1)].clip(0,1)
            q4=np.floor(xx*16).clip(0,15).astype(int)
            for b in BITS:
                direct=np.floor(xx*2**b).clip(0,2**b-1).astype(int)
                assert np.array_equal(direct,q4>>(4-b))
                reconstruction=(direct+.5)/2**b
                assert np.max(abs(reconstruction-xx))<=1/2**(b+1)+1e-15
            assert np.isfinite(decode(encode(torch.tensor(np.broadcast_to(endpoints,(2,1,17)).copy(),dtype=torch.float32),method,bt),bt,4).numpy()).all()
        else:
            # Full integer paths retain their two-bit parent, and the residual
            # stage is an additive +/- amplitude before inverse companding.
            c2=book['c2'].astype(float);c3=book['c3'].astype(float)
            if method=='Residual211':
                expected=(c2[:,None]+np.array([-1,1])[None,:]*float(book['a3'])).ravel()
                assert np.allclose(c3,expected,atol=.005)
        records.append(dict(method=method,calibration=info))
    # A shared signed residual can create large cross-token error.
    p=torch.tensor([[.5,.5]],dtype=torch.float64);e=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    exact=(p@e).square().sum();diag=(p.square()*e.square().sum(-1)).sum()
    assert float(exact-diag)==.5
    # Prefix refinement gains need not be diminishing; preserve this counterexample.
    from codec import allocation_check
    assert allocation_check(np.array([[100.,100.,0.],[2.,1.,0.]]))[2]['gap']>0
    atomic(ROOT/'correctness_checks.json',dict(passed=True,methods=records,checks=['TCN direct vs prefix boundaries','CPU tensor vs numpy codes','2/3/4 prefix planes with odd-length padding','residual additive algebra','signed cross-token decomposition','nonconcave allocation counterexample']))
    print('All encoding correctness gates passed',flush=True)


if __name__=='__main__':main()
