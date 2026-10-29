import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'encoding_ablation_20261024'
SUPP = BASE / 'supplementary_experiments'
sys.path.insert(0, str(BASE / 'src'))
import run as legacy
from codec import encode as tree_encode, decode as tree_decode
from codecs4 import NEW_METHODS, BITS
METHODS = ('U_Clipped', 'Data_Tree', 'Jacobian_Tree', *NEW_METHODS)
SEEDS = legacy.SEEDS


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def state(stage, **kw):
    atomic(ROOT / 'status.json', dict(stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'), **kw))
    print(stage, kw, flush=True)


# Legacy initializer reads frozen manifest and model; redirect all status writes.
legacy.status = state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_layer(layer, seed, smoke=False, device='cuda'):
    folder = ROOT / 'codebooks' / ('smoke' if smoke else 'full')
    with np.load(folder / f'layer{layer}_seed{seed}.npz') as z:
        return {k: torch.tensor(z[k].astype(np.float64 if k == 'controller_D' else np.float32), device=device) for k in z.files}


def reconstruct(x, method, book):
    import codecs4
    mu, sigma = book['mu'], book['sigma']
    z = (x - mu[:, :, None, :]) / sigma[:, :, None, :]
    rec, codes = {}, []
    for side in (0, 1):
        p = {k.removeprefix(method + '_'): v[side] for k, v in book.items() if k.startswith(method + '_')}
        if method == 'U_Clipped':
            code = torch.round((z[side] - p['lo'][:, None, None]) / (p['hi'] - p['lo'])[:, None, None] * 15).clamp(0, 15).long()
            decoded = {b: p['lo'][:, None, None] + ((code >> (4 - b)) * 2**(4 - b) + (2**(4 - b) - 1) / 2) / 15 * (p['hi'] - p['lo'])[:, None, None] for b in BITS}
        elif method in ('Data_Tree', 'Jacobian_Tree'):
            code = tree_encode(z[side], p)
            decoded = {b: tree_decode(code, p, b) for b in BITS}
        else:
            code = codecs4.encode(z[side], method, p)
            decoded = {b: codecs4.decode(code, p, b) for b in BITS}
        codes.append(code)
        for b in BITS:
            rec.setdefault(b, []).append(decoded[b] * sigma[side, :, None, :] + mu[side, :, None, :])
    return {b: torch.stack(v) for b, v in rec.items()}, torch.stack(codes)


def trajectory_check(folder, row, smoke=False):
    name = row['case_id'].replace('/', '_') + '_trajectory.json'
    reference = BASE / 'results' / ('smoke' if smoke else 'full') / name
    actual = json.loads((folder / name).read_text())
    assert actual['sha256'] == json.loads(reference.read_text())['sha256'], row['case_id']


def write_csv(path, rows):
    import csv, gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows for {path}')
    temp = path.with_name(path.name + '.tmp')
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(temp, 'wt') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
