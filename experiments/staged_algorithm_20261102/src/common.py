"""Isolated algorithm experiment; never modifies historical runs."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
LEGACY = EXPERIMENTS / 'full_quality_20261027'
BASE = EXPERIMENTS / 'encoding_ablation_20261024'
MODEL = Path('/home/wanglin/kvbench-3090-cu126/cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c')
SEED = 20261102
CODECS = ('Gaussian_Tree', 'Data_Tree', 'Jacobian_Tree', 'JacK_MixV', 'U_Clipped')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)

def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def status(stage, **kw):
    value = dict(stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'), **kw)
    atomic(ROOT / 'worker_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)

def run_id(codec, policy, depth=4, curve='global', budget=3, threshold=0.05):
    if codec == 'fp16':
        return 'fp16'
    if policy == 'fixed':
        return f'{codec}__fixed{depth}'
    return f'{codec}__{policy}__{curve}__' + (f'budget{budget}' if policy == 'budget' else f'threshold{threshold}')
