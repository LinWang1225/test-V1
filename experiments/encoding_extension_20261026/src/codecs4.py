"""Four stored bits; deterministic 2/3/4-bit prefix decoders.

Residual211 and LogResidual211 adapt paper ideas, not native QuantSpec/Lynx.
No test-time parameter fitting; all ranges and stages come from calibration.
"""
import numpy as np
import torch

BITS = (2, 3, 4)
QUANTILES = (0., .0001, .001, .005, .01, .02)
NEW_METHODS = ('TCN_MinMax', 'TCN_Clipped', 'Residual211', 'LogResidual211')


def compand(x, alpha):
    y = np.sign(x) * np.log1p(alpha * np.abs(x)) / np.log1p(alpha)
    return np.where(np.abs(x) == 1, np.sign(x), y)


def expand(x, alpha):
    return np.sign(x) * np.expm1(np.abs(x) * np.log1p(alpha)) / alpha


def _residual_fit(x, w, lo, hi, alpha=0.):
    if alpha:
        # Symmetric bound, then log companding of signed values.
        radius = max(abs(lo), abs(hi), 1e-4)
        u = compand(np.clip(x / radius, -1., 1.), alpha)
        low, high = -1., 1.
    else:
        radius = 1.
        u = np.clip(x, lo, hi)
        low, high = lo, hi
    step = (high - low) / 3
    q = np.rint((u - low) / step).clip(0, 3).astype(np.int64)
    rec = low + q * step
    scales = []
    tables = {2: low + np.arange(4) * step}
    # One symmetric residual bit per stage, shared scale per head/side.
    # The weighted optimum amplitude for a fixed residual sign is E|error|.
    for b in (3, 4):
        residual = u - rec
        amplitude = float(np.float16(np.average(np.abs(residual), weights=w)))
        scales.append(amplitude)
        bit = (residual >= 0).astype(np.int64)
        rec = rec + (2 * bit - 1) * amplitude
        parent = tables[b - 1]
        tables[b] = (parent[:, None] + np.array([-amplitude, amplitude])[None, :]).ravel()
    if alpha:
        tables = {b: radius * expand(v, alpha) for b, v in tables.items()}
    return tables, dict(lo=low, hi=high, radius=radius, alpha=alpha, a3=scales[0], a4=scales[1])


def numpy_encode(x, method, p):
    lo, hi = float(p['lo']), float(p['hi'])
    if method.startswith('TCN_'):
        return np.floor((x - lo) / (hi - lo) * 16).clip(0, 15).astype(np.int64)
    alpha = float(p['alpha'])
    u = compand(np.clip(x / float(p['radius']), -1, 1), alpha) if alpha else x
    step = (hi - lo) / 3
    q = np.rint((u - lo) / step).clip(0, 3).astype(np.int64)
    rec = lo + q * step
    for b in (3, 4):
        bit = (u - rec >= 0).astype(np.int64)
        rec += (2 * bit - 1) * float(p[f'a{b}'])
        q = q * 2 + bit
    return q


def fit(x, w, method, range_bounds=None):
    """Use the same frozen histogram and 36 range candidates as old uniform.

    TCN_MinMax is exact non-clipping TCN on calibration extrema. Others select
    ranges for mean *original standardized-domain* D2,D3,D4. Log alpha is
    selected on calibration only from a preregistered finite grid.
    """
    cw = np.cumsum(w) / w.sum()
    ranges = [(0., 0.)] if method == 'TCN_MinMax' else [(a, b) for a in QUANTILES for b in QUANTILES]
    best = None
    for left, right in ranges:
        lo, hi = np.interp([left, 1 - right], cw, x).astype(np.float16).astype(float)
        if range_bounds is not None:
            if left == 0:
                lo = float(np.float16(range_bounds[0]))
            if right == 0:
                hi = float(np.float16(range_bounds[1]))
        if hi - lo < 1e-5:
            continue
        for alpha in ((1., 4., 16., 64.) if method == 'LogResidual211' else (0.,)):
            if method.startswith('TCN_'):
                tables = {b: lo + (np.arange(2**b) + .5) * (hi - lo) / 2**b for b in BITS}
                p = dict(lo=lo, hi=hi, radius=1., alpha=0., a3=0., a4=0.)
            else:
                tables, p = _residual_fit(x, w, lo, hi, alpha)
            # Account for deployed FP16 precision while selecting parameters.
            p = {k: np.float16(v) for k, v in p.items()}
            if method.startswith('TCN_'):
                tables = {b: float(p['lo']) + (np.arange(2**b) + .5) * (float(p['hi']) - float(p['lo'])) / 2**b for b in BITS}
            else:
                values = float(p['lo']) + np.arange(4) * (float(p['hi']) - float(p['lo'])) / 3
                tables = {2: values}
                for b in (3, 4):
                    values = (values[:, None] + np.array([-float(p[f'a{b}']), float(p[f'a{b}'])])[None, :]).ravel()
                    tables[b] = values
                if alpha:
                    tables = {b: float(p['radius']) * expand(v, float(p['alpha'])) for b, v in tables.items()}
            tables = {b: v.astype(np.float16) for b, v in tables.items()}
            q = numpy_encode(x, method, p)
            ds = [float(np.average((x - tables[b][q >> (4 - b)].astype(float))**2, weights=w)) for b in BITS]
            objective = float(np.mean(ds))
            if best is None or objective < best[0]:
                best = (objective, p, tables, left, right, ds)
    if best is None:
        raise ValueError('Calibration range collapsed')
    objective, p, tables, left, right, ds = best
    out = {**p, **{f'c{b}': tables[b] for b in BITS}}
    return out, dict(objective=objective, D=ds, clip_left=left, clip_right=right, alpha=float(p['alpha']), range_candidates=len(ranges))


def encode(x, method, book):
    def param(n):
        return book[n][:, None, None]
    lo, hi = param('lo'), param('hi')
    if method.startswith('TCN_'):
        return torch.floor((x - lo) / (hi - lo) * 16).clamp(0, 15).long()
    alpha = param('alpha')
    if method == 'LogResidual211':
        z = (x / param('radius')).clamp(-1, 1)
        u = z.sign() * torch.log1p(alpha * z.abs()) / torch.log1p(alpha)
        u = torch.where(z.abs() == 1, z, u)
    else:
        u = x
    step = (hi - lo) / 3
    code = torch.round((u - lo) / step).clamp(0, 3).long()
    rec = lo + code * step
    for b in (3, 4):
        bit = (u - rec >= 0).long()
        rec = rec + (2 * bit - 1) * param(f'a{b}')
        code = code * 2 + bit
    return code


def decode(code, book, bits):
    return torch.gather(book[f'c{bits}'], 1, (code >> (4 - bits)).flatten(1)).view_as(code)


def pack_prefix(code, bits, axis=-1):
    """Independent planes padded per physical row, MSB first."""
    a = np.moveaxis(np.asarray(code, np.uint8), axis, -1)
    return [np.packbits((a >> bit) & 1, axis=-1, bitorder='little') for bit in range(3, 3 - bits, -1)]


def unpack_prefix(planes, length):
    out = np.zeros((*planes[0].shape[:-1], length), np.uint8)
    for bit, plane in zip(range(3, -1, -1), planes):
        out |= np.unpackbits(plane, axis=-1, bitorder='little')[..., :length] << bit
    return out
