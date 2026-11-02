"""Small end-to-end test; run through run_with_dram.py -- COMMAND."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'experiments/full_quality_io_20261027'
sys.path.insert(0, str(ROOT / 'src'))
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from traffic import collect

assert os.geteuid() != 0, 'The test client should remain unprivileged'
torch.manual_seed(42)
q = torch.randn(1, 32, 1, 128, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 32, 128, device='cuda', dtype=torch.float16)
v = torch.randn_like(k)
module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
with torch.inference_mode():
    expected = sdpa_attention_forward(module, q, k, v, None, scaling=128**-.5)[0]
name = f'collector_smoke_{time.time_ns()}'
fixture = ROOT / 'fixtures' / f'{name}.pt'
torch.save(dict(q=q.cpu(), k=k.cpu(), v=v.cpu(), expected_output=expected.cpu(),
                layer=17, context=32, scale=128**-.5, policy='fp16', case_id=name), fixture)
try:
    for condition in ('warm', 'evicted'):
        receipt = collect(fixture, condition, ROOT / 'profiles' / name / condition)
        print(json.dumps(dict(receipt, client_uid=os.geteuid())), flush=True)
finally:
    fixture.unlink(missing_ok=True)
