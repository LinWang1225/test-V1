"""Fit new codecs using frozen calibration activations, preserve old books."""
import argparse
from common import *
from codecs4 import fit


def train(smoke=False):
    folder = ROOT / 'codebooks' / ('smoke' if smoke else 'full')
    folder.mkdir(parents=True, exist_ok=True)
    source = BASE / 'calibration' / ('smoke' if smoke else 'full')
    manifest = json.loads((BASE / 'inputs' / ('smoke.json' if smoke else 'manifest.json')).read_text())
    files = [source / (r['case_id'].replace('/', '_') + '.npz') for r in manifest if r['split'] == 'calibration']
    assert len(files) == (1 if smoke else 60)
    statistics = []
    for layer in ((0, 17, 35) if smoke else range(36)):
        state('encoding_fit_layer', layer=layer, smoke=smoke)
        X = np.stack([np.load(p)[f'x{layer}'].astype(np.float32) for p in files])
        for seed in (SEEDS[:1] if smoke else SEEDS):
            target = folder / f'layer{layer}_seed{seed}.npz'
            if target.exists():
                # Resume only if the whole file has expected methods and profiles.
                with np.load(target) as a:
                    assert all(f'{m}_c4' in a for m in NEW_METHODS)
                    assert a['controller_D'].shape == (2, 8, 128, 3)
                continue
            old_path = SUPP / 'codebooks' / ('smoke' if smoke else 'full') / target.name
            with np.load(old_path) as a:
                old = {k: a[k] for k in a.files}
            mu, sigma = old['mu'], old['sigma']
            Z = (X - mu[None, :, :, None, :]) / sigma[None, :, :, None, :]
            counts = np.bincount(np.random.default_rng(seed + layer).integers(0, len(X), len(X)), minlength=len(X))
            output = dict(mu=mu, sigma=sigma)
            for m, key in [('U_Clipped', 'uniform_mse'), ('Data_Tree', 'tree_mse'), ('Jacobian_Tree', 'tree_jac')]:
                for side in (0, 1):
                    source_key = 'tree_attn' if m == 'Jacobian_Tree' and side == 1 else key
                    for name in (('lo', 'hi') if m == 'U_Clipped' else ('c2', 'c3', 'c4', 't2', 't3', 't4')):
                        output.setdefault(m + '_' + name, [None, None])[side] = old[f'{side}_{source_key}_{name}']
            for m in NEW_METHODS:
                sides = []
                for side in (0, 1):
                    heads = []
                    for head in range(8):
                        x = Z[:, side, head].reshape(-1).astype(float)
                        w = np.broadcast_to(counts[:, None, None], Z[:, side, head].shape).reshape(-1).astype(float)
                        edges = np.linspace(x.min(), x.max(), 8193)
                        hw = np.histogram(x, bins=edges, weights=w)[0]
                        hx = np.histogram(x, bins=edges, weights=w*x)[0]
                        ok = hw > 0
                        book, info = fit(hx[ok] / hw[ok], hw[ok], m, (x.min(), x.max()))
                        heads.append(book)
                        statistics.append(dict(layer=layer, seed=seed, side=side, head=head, method=m, **info))
                    sides.append({k: np.stack([b[k] for b in heads]) for k in heads[0]})
                for k in sides[0]:
                    output[m + '_' + k] = np.stack([a[k] for a in sides])
            output = {k: np.asarray(v) for k, v in output.items()}
            # Physical-scale, per-channel frozen Jacobian distortion curves.
            # Test-time controller needs no true current K or V to use them.
            tensorbook = {k: torch.tensor(v.astype(np.float32)) for k, v in output.items()}
            dsum = np.zeros((2, 8, 128, 3), np.float64)
            mass = 0.
            for i, activation in enumerate(X):
                if not counts[i]:
                    continue
                xt = torch.tensor(activation)
                decoded, _ = reconstruct(xt, 'Jacobian_Tree', tensorbook)
                for j, b in enumerate(BITS):
                    dsum[..., j] += (decoded[b] - xt).double().square().mean(2).numpy() * counts[i]
                mass += counts[i]
            output['controller_D'] = dsum / mass
            with open(str(target) + '.tmp', 'wb') as f:
                np.savez(f, **output)
            Path(str(target) + '.tmp').replace(target)
            atomic(folder / f'layer{layer}_seed{seed}_training.json', [s for s in statistics if s['layer'] == layer and s['seed'] == seed])
    atomic(folder / 'ready.json', dict(complete=True, layers=[0,17,35] if smoke else list(range(36)), seeds=SEEDS[:1] if smoke else SEEDS, methods=METHODS))
    state('encoding_fit_completed', smoke=smoke)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true')
    torch.set_num_threads(4)
    train(p.parse_args().smoke)
