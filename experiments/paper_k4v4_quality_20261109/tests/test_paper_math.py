"""CPU checks for bit accounting and both original TurboQuant algorithms."""
from pathlib import Path
import sys
import math
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from turboquant_math import parameters,sphere_lloyd,quantize_numpy,dequantize_numpy,unpack_nibbles


class PaperMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params=parameters(d=16,seed=19)

    def test_haar_and_analytic_codebook_symmetry(self):
        for key in ['RK','RV']:
            np.testing.assert_allclose(self.params[key]@self.params[key].T,np.eye(16),atol=1e-12)
        for key in ['C3','C4']:
            c=self.params[key]
            self.assertTrue(np.all(np.diff(c)>0))
            np.testing.assert_allclose(c,-c[::-1],atol=1e-12)

    def test_one_bit_sphere_centroid(self):
        d=128
        # E[|X_1|] on S^(d-1), including the exact finite-dimensional density.
        expected=math.exp(math.lgamma(d/2)-math.lgamma((d-1)/2))/math.sqrt(math.pi)*2/(d-1)
        np.testing.assert_allclose(sphere_lloyd(d,1),[-expected,expected],atol=1e-11)

    def test_key_uses_three_plus_one_bits_and_exact_qjl_factor(self):
        x=np.random.default_rng(12).standard_normal((7,16))
        packed,meta=quantize_numpy(x,self.params,True)
        self.assertEqual(packed.nbytes,7*16*4//8)
        codes=unpack_nibbles(packed)
        mse=self.params['C3'][codes&7]
        signs=2*(codes>>3).astype(float)-1
        expected=meta[:,:1]*(mse+math.sqrt(math.pi/2)/16*meta[:,1:2]*(signs@self.params['S']))@self.params['RK']
        np.testing.assert_allclose(dequantize_numpy(packed,meta,self.params,True),expected,atol=1e-12)
        self.assertGreater(np.max(np.abs(expected-meta[:,:1]*mse@self.params['RK'])),.01)

    def test_values_use_lloyd_codebook_without_norm_correction(self):
        x=np.random.default_rng(14).standard_normal((7,16))
        packed,meta=quantize_numpy(x,self.params,False)
        expected=meta*self.params['C4'][unpack_nibbles(packed)]@self.params['RV']
        actual=dequantize_numpy(packed,meta,self.params,False)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        self.assertFalse(np.allclose(np.linalg.norm(actual,axis=-1),np.linalg.norm(x,axis=-1)))

    def test_zero_vectors_and_full_prefix_pack_roundtrip(self):
        for key in [True,False]:
            code,meta=quantize_numpy(np.zeros((3,16)),self.params,key)
            np.testing.assert_array_equal(dequantize_numpy(code,meta,self.params,key),np.zeros((3,16)))

    def test_qjl_unbiased_in_expectation_over_gaussian_projection(self):
        rng=np.random.default_rng(123)
        residual=rng.standard_normal(16);q=rng.standard_normal(16)
        s=rng.standard_normal((4096,16,16))
        signs=np.sign(s@residual)
        estimates=math.sqrt(math.pi/2)/16*np.linalg.norm(residual)*np.sum((s@q)*signs,axis=-1)
        self.assertLess(abs(estimates.mean()-q@residual),4*estimates.std()/math.sqrt(len(estimates)))


if __name__=='__main__':unittest.main()
