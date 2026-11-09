"""Frozen 4-bit words with five prefix reconstruction levels."""
import sys
import numpy as np
import torch
from common import *
sys.path.insert(0, str(BASE / 'src'))
from codec import encode as tree_encode, hadamard

def encode(z, book):
    if 'lo' in book:
        return torch.round((z-book['lo'][:, None, None]) / (book['hi']-book['lo'])[:, None, None]*15).clamp(0,15).long()
    return tree_encode(z, book)

def make_lut(book):
    q = torch.arange(16, device=book['mu'].device)
    levels = []
    for b in range(5):
        if b < 2 or 'lo' not in book:
            levels.append(book[f'c{b}'][:, q >> (4-b)])
        else:
            center = (q >> (4-b))*2**(4-b)+(2**(4-b)-1)/2
            levels.append(book['lo'][:, None]+center/15*(book['hi']-book['lo'])[:, None])
    return torch.stack(levels, 1)  # head, depth, word

def load_books(codec, device):
    books, luts, curves = {}, {}, {}
    for layer in range(36):
        path = ROOT / 'codebooks' / codec / f'layer{layer}.npz'
        with np.load(path) as z:
            book = {k: torch.tensor(z[k].astype(np.float32), device=device) for k in z.files if k not in ('D', 'J')}
            curves[layer] = {k: torch.tensor(z[k], device=device, dtype=torch.float64) for k in ('D','J')}
        books[layer] = book
        luts[layer] = torch.stack([make_lut({k: v[s] for k,v in book.items()}) for s in (0,1)])
    return books, luts, curves

def metadata(codec):
    stored = {}
    for p in sorted((ROOT/'codebooks'/codec).glob('layer*.npz')):
        with np.load(p) as z:
            for k in z.files:
                stored[k] = stored.get(k, 0) + z[k].nbytes
    return stored
