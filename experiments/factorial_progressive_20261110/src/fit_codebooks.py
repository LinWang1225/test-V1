import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import json, time, hashlib, concurrent.futures
import numpy as np
from common import ROOT, atomic, progress, sha
N = 60

def lloyd(x, w, n):
    if w.sum() <= 0:
        w = np.ones_like(w)
    q = (np.arange(n) + 0.5) / n
    c = np.interp(q, np.cumsum(w) / w.sum(), x)
    for _ in range(100):
        labels = np.searchsorted((c[:-1] + c[1:]) / 2, x)
        mass = np.bincount(labels, weights=w, minlength=n)
        sums = np.bincount(labels, weights=w * x, minlength=n)
        nc = np.where(mass > 0, sums / np.maximum(mass, 1e-300), c)
        nc = np.sort(nc)
        change = np.max(np.abs(nc - c)) / (1 + np.max(np.abs(c)))
        c = nc
        if change < 1e-06:
            break
    return c

def code_numpy(z, t2, t3, t4):
    labels = np.sum(z[..., None] > t2[:, None, None, :], axis=-1).astype(np.int64)
    h = np.arange(z.shape[0])[:, None, None]
    labels = labels * 2 + (z > t3[h, labels])
    labels = labels * 2 + (z > t4[h, labels])
    return labels

def fit_tree(z, w):
    thresholds2 = []
    thresholds3 = []
    thresholds4 = []
    luts = []
    for h in range(8):
        x = z[h].ravel().astype(np.float64)
        weight = w[h].ravel().astype(np.float64)
        assert np.isfinite(x).all() and np.isfinite(weight).all() and (weight >= 0).all()
        if not weight.sum():
            weight = np.ones_like(weight)
        lo, hi = (float(x.min()), float(x.max()))
        if hi - lo < 1e-08:
            lo -= 1
            hi += 1
        hist, edges = np.histogram(x, bins=8192, range=(lo, hi), weights=weight)
        xx = (edges[:-1] + edges[1:]) / 2
        c4 = lloyd(xx, hist, 4)
        t2 = ((c4[:-1] + c4[1:]) / 2).astype(np.float16)
        p2 = np.searchsorted(t2.astype(float), xx, side='left')
        t3 = np.empty(4, dtype=np.float16)
        for p in range(4):
            take = p2 == p
            pair = lloyd(xx[take], hist[take], 2) if take.any() else np.array([c4[p], c4[p]])
            t3[p] = (pair[0] + pair[1]) / 2
        p3 = 2 * p2 + (xx > t3[p2])
        t4 = np.empty(8, dtype=np.float16)
        for p in range(8):
            take = p3 == p
            pair = lloyd(xx[take], hist[take], 2) if take.any() else np.array([0.0, 0.0])
            t4[p] = (pair[0] + pair[1]) / 2
        p4 = 2 * p3 + (xx > t4[p3])
        lut = np.zeros(31, dtype=np.float16)
        for b in range(1, 5):
            labels = p4 >> 4 - b
            offset = (1 << b) - 1
            for p in range(1 << b):
                mask = labels == p
                mass = hist[mask].sum()
                lut[offset + p] = np.dot(hist[mask], xx[mask]) / mass if mass > 0 else 0.0
        assert all((np.isfinite(a).all() for a in [t2, t3, t4, lut]))
        thresholds2.append(t2)
        thresholds3.append(t3)
        thresholds4.append(t4)
        luts.append(lut)
    return [np.stack(x) for x in [thresholds2, thresholds3, thresholds4, luts]]

def fit_layer(li):
    out = ROOT / 'codebooks'
    receipts = []
    paths = [ROOT / 'calibration/normalized' / f'layer{li:02}' / f'{i:03}.npz' for i in range(N)]
    for bi in range(3):
        bn = f'B{bi}'
        att = []
        old = []
        jk = []
        jv = []
        for p in paths:
            with np.load(p) as a:
                att.append(a['attention_weight'])
                old.append(a[f'oldK_{bn}'])
                jk.append(a[f'jacK_{bn}'])
                jv.append(a[f'jacV_{bn}'])
        att = np.concatenate(att, axis=1)
        old = np.concatenate(old, axis=1)
        jac = np.stack([np.concatenate(jk, axis=1), np.concatenate(jv, axis=1)])
        for ri in range(4):
            prefix = f'{bn}_R{ri}'
            zs = []
            ss = []
            ee = []
            for p in paths:
                with np.load(p) as a:
                    zs.append(a[prefix + '_z'])
                    ss.append(a[prefix + '_scale'])
                    ee.append(a[prefix + '_energy_mean'])
            z = np.concatenate(zs, axis=2)
            scale = np.concatenate(ss, axis=2)
            del zs, ss
            energy = np.mean(ee, axis=0).astype(np.float32)
            for ei in range(2):
                ident = f'E{ei}_{prefix}'
                dest = out / ident / f'layer{li:02}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.with_suffix('.json').exists():
                    receipts.append({'identity': ident, 'sha256': sha(dest)})
                    continue
                assets = {}
                mse = np.empty((2, 8, 128, 5), np.float32)
                jcurve = np.empty_like(mse)
                for side in range(2):
                    if side == 0:
                        weight = old.copy()
                    else:
                        aw = att[:, :, None] * np.ones((1, 1, 128), np.float32)
                        weight = aw if ei == 0 else 0.5 * np.ones_like(aw) + 0.5 * aw / np.maximum(aw.mean(axis=(1, 2), keepdims=True), 1e-30)
                    weight = weight * scale[side] ** 2
                    weight = weight / np.maximum(weight.mean(axis=(1, 2), keepdims=True), 1e-30)
                    t2, t3, t4, lut = fit_tree(z[side], weight)
                    for name, arr in [('t2', t2), ('t3', t3), ('t4', t4), ('lut', lut)]:
                        assets[f'{name}_{side}'] = arr
                    code = code_numpy(z[side], t2, t3, t4)
                    hh = np.arange(8)[:, None, None]
                    for bits in range(5):
                        decoded = np.zeros_like(z[side]) if bits == 0 else lut[hh, (1 << bits) - 1 + (code >> 4 - bits)].astype(np.float32)
                        error = (scale[side] * (z[side] - decoded)) ** 2
                        mse[side, :, :, bits] = error.mean(axis=1, dtype=np.float64)
                        jcurve[side, :, :, bits] = (error * jac[side]).mean(axis=1, dtype=np.float64)
                assets.update(mse=mse, jacobian=jcurve, energy=energy)
                assert all((np.isfinite(x).all() for x in assets.values()))
                np.savez(dest, **assets)
                atomic(dest.with_suffix('.json'), {'identity': ident, 'layer': li, 'calibration_cases': N, 'samples_per_channel': z.shape[2], 'tree': 'weighted 4-way Lloyd then binary 3/4; 1-bit high-prefix conditional centroid; 0-bit prescribed mean', 'histogram_bins': 8192, 'lloyd_max_iterations': 100, 'V_mix_lambda': 0.5, 'K_weight': 'attention output-Jacobian diagonal times reconstruction scale squared', 'V_weight': 'attention squared; E1 half normalized MSE plus half normalized attention squared, times scale squared', 'curve': 'physical-space empirical error, own E/B/R deployed FP16 codebook', 'sha256': sha(dest)})
                receipts.append({'identity': ident, 'sha256': sha(dest)})
    return {'layer': li, 'assets': receipts}

def main():
    marker = ROOT / 'calibration/normalized/complete.json'
    start = time.time()
    while not marker.exists():
        pointer = ROOT / 'jobs/calibration_fit.json'
        if pointer.exists():
            p = Path(json.loads(pointer.read_text())['job_dir']) / 'state.json'
            if p.exists() and json.loads(p.read_text()).get('status') == 'failed':
                raise RuntimeError('normalized capture failed')
        if time.time() - start > 7200:
            raise TimeoutError('normalized capture')
        progress('codebook_fit', 'waiting_for_normalized_capture')
        time.sleep(15)
    receipts = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fit_layer, li): li for li in range(36)}
        for f in concurrent.futures.as_completed(futures):
            receipts.append(f.result())
            progress('codebook_fit', 'fitting', **{'completed_layers': len(receipts), 'layers': 36})
    atomic(ROOT / 'codebooks/complete.json', {'layers': sorted(receipts, key=lambda x: x['layer']), 'encodings': 24, 'curves_per_identity': 36, 'source_calibration': sha(marker)})
    progress('codebook_fit', 'completed', **{'identities': 24, 'layers': 36})
if __name__ == '__main__':
    main()
