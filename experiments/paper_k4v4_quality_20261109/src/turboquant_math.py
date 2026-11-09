"""Paper Algorithms 1/2, with no norm correction or uniform value quantizer.

Rows represent vectors. R maps a vector to x @ R.T. QJL may be applied in
that rotated frame: an independent Gaussian matrix remains Gaussian under
an independent orthogonal change of basis. The query is transformed equally.
"""
import math
from statistics import NormalDist
import numpy as np


def sphere_lloyd(d, bits, tolerance=1e-12):
    """Solve continuous scalar k-means for exact sphere-coordinate density."""
    assert d >= 3 and bits >= 1
    count = 2 ** bits
    points, weights = np.polynomial.legendre.leggauss(64)
    initial = [(j + .5) / count for j in range(count)]
    centers = np.array([NormalDist().inv_cdf(p) for p in initial]) / math.sqrt(d)
    centers = centers.clip(-1, 1)
    for iteration in range(4000):
        cuts = np.concatenate(([-1.], (centers[:-1] + centers[1:]) / 2, [1.]))
        half = (cuts[1:] - cuts[:-1]) / 2
        middle = (cuts[1:] + cuts[:-1]) / 2
        x = middle[:, None] + half[:, None] * points
        density = np.maximum(0, 1 - x*x) ** ((d - 3) / 2)
        mass = (density * weights).sum(1)
        new = (x * density * weights).sum(1) / mass
        new = (new - new[::-1]) / 2  # enforce analytic symmetry
        difference = np.max(np.abs(new - centers))
        centers = new
        if difference < tolerance:
            return centers
    raise RuntimeError('Sphere-density Lloyd solver did not converge')


def haar_rotation(rng, d):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.where(np.diag(r) >= 0, 1., -1.)[None, :]


def parameters(d=128, seed=20261109):
    rng = np.random.default_rng(seed)
    return dict(RK=haar_rotation(rng, d), RV=haar_rotation(rng, d),
                S=rng.standard_normal((d, d)), C3=sphere_lloyd(d, 3),
                C4=sphere_lloyd(d, 4))


def pack_nibbles(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    assert codes.shape[-1] % 2 == 0 and np.all(codes < 16)
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def unpack_nibbles(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    return np.stack((packed & 15, packed >> 4), axis=-1).reshape(*packed.shape[:-1], -1)


def quantize_numpy(x, params, key=True):
    """Independent CPU oracle, including arbitrary-vector L2 scaling."""
    x = np.asarray(x, dtype=np.float64)
    rotation = params['RK' if key else 'RV']
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    rotated = (x @ rotation.T) / np.maximum(norm, 1e-30)
    centers = params['C3' if key else 'C4']
    indices = np.searchsorted((centers[:-1] + centers[1:]) / 2, rotated).astype(np.uint8)
    if key:
        residual = rotated - centers[indices]
        gamma = np.linalg.norm(residual, axis=-1, keepdims=True)
        signs = (residual @ params['S'].T >= 0).astype(np.uint8)
        return pack_nibbles(indices | (signs << 3)), np.concatenate((norm, gamma), axis=-1)
    return pack_nibbles(indices), norm


def dequantize_numpy(packed, metadata, params, key=True):
    code = unpack_nibbles(packed)
    if key:
        mse = params['C3'][code & 7]
        signs = 2 * (code >> 3).astype(np.float64) - 1
        correction = math.sqrt(math.pi / 2) / code.shape[-1] * metadata[..., 1:2] * (signs @ params['S'])
        return metadata[..., :1] * (mse + correction) @ params['RK']
    return metadata * params['C4'][code] @ params['RV']


def main():
    from common import ROOT, atomic, sha
    path = ROOT / 'references/turboquant_parameters.npz'
    if path.exists():
        raise FileExistsError('Parameters already frozen; refusing overwrite')
    p = parameters()
    np.savez(path, **p)
    atomic(ROOT / 'references/turboquant_parameters.json', dict(
        dimension=128, seed=20261109, sha256=sha(path),
        source='Paper Algorithms 1/2, equation for exact Beta coordinate density',
        calibration_data_used=False, key_payload_bits=4, value_payload_bits=4,
        key_norm_and_gamma_bits=64, value_norm_bits=32,
        projected_query_formula='q_rot @ S.T; correction=sqrt(pi/2)/128*gamma*dot(projected_query,signs)',
        norm_correction=False, value_uniform_quantization=False))
    print('Prepared data-oblivious Haar rotations, Gaussian QJL and exact Beta-density codebooks')


if __name__ == '__main__':
    main()
