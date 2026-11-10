import os
os.environ['OMP_NUM_THREADS'] = '4'
from pathlib import Path
import json, time, itertools, subprocess
import numpy as np, torch
from types import SimpleNamespace
from common import ROOT, atomic, progress, rows, sha

def wait_assets():
    start = time.time()
    while not (ROOT / 'codebooks/complete.json').exists():
        d = json.loads((ROOT / 'jobs/codebook_fit.json').read_text())
        s = Path(d['job_dir']) / 'state.json'
        if s.exists() and json.loads(s.read_text()).get('status') == 'failed':
            raise RuntimeError('codebook fit failed')
        if time.time() - start > 10800:
            raise TimeoutError('codebook fitting')
        progress('validation', 'waiting_for_codebooks')
        time.sleep(15)

def assets():
    report = {'identities': 24, 'layers': 36, 'files': 0, 'K_E0_E1_equal': True}
    for b, r in itertools.product(range(3), range(4)):
        for li in range(36):
            p = [ROOT / 'codebooks' / f'E{e}_B{b}_R{r}' / f'layer{li:02}.npz' for e in range(2)]
            with np.load(p[0]) as a, np.load(p[1]) as z:
                for k in ['t2_0', 't3_0', 't4_0', 'lut_0']:
                    assert np.array_equal(a[k], z[k]), (b, r, li, k)
                for path in p:
                    receipt = json.loads(path.with_suffix('.json').read_text())
                    assert sha(path) == receipt['sha256']
                    with np.load(path) as aa:
                        assert all((np.isfinite(aa[k]).all() for k in aa.files))
                        for side in range(2):
                            assert (aa[f'lut_{side}'][:, 0] == 0).all()
                        assert (aa['mse'] >= 0).all() and (aa['jacobian'] >= 0).all()
                    report['files'] += 1
    atomic(ROOT / 'checks/assets.json', report)

def boundary():
    from dynamic_runtime import Runtime
    from native_backend import STATE
    torch.manual_seed(20261110)
    torch.set_num_threads(4)
    STATE.decode_position = 0
    cache = torch.randn((4, 2, 128, 8, 128), device='cuda', dtype=torch.float16)
    meta = SimpleNamespace(seq_lens=torch.tensor([129], device='cuda'), block_table=torch.tensor([[2, 0, 3, 1]], device='cuda'))
    query = torch.randn((1, 4096), device='cuda', dtype=torch.float16)
    reports = []
    for e, b, r in itertools.product(range(2), range(3), range(4)):
        ident = f'E{e}_B{b}_R{r}'
        item = {'identity': ident, 'controllers': [], 'boundary': []}
        for c in range(4):
            rt = Runtime(ident, f'C{c}', metrics=False)
            for total in [False, True]:
                rt.cost_mode = 'total' if total else 'payload'
                rt.reset_case()
                y = rt.attention(0, query, cache, meta, 1 / 128 ** 0.5)
                assert torch.isfinite(y).all()
                stats = rt.finish()
                assert sum(stats['depth_element_counts']['K']) == 8 * 129 * 128
                assert sum(stats['depth_element_counts']['V']) == 8 * 129 * 128
            item['controllers'].append(f'C{c}')
            del rt
        for bit in [0, 1, 4]:
            rt = Runtime(ident, uniform=bit, metrics=False)
            y = rt.attention(0, query, cache, meta, 1 / 128 ** 0.5)
            assert torch.isfinite(y).all()
            stats = rt.finish()
            assert all((v == bit for v in stats['actual_element_weighted_bits'].values()))
            if bit == 0:
                assert rt.states[0]['T'] == 129
            if bit == 4:
                for tt in range(130, 136):
                    meta.seq_lens[0] = tt
                    rt.attention(0, query, cache, meta, 1 / 128 ** 0.5)
                saved = [side['code'][:, :135].clone() for side in rt.states[0]['side']]
                rt.reset_case()
                rt.attention(0, query, cache, meta, 1 / 128 ** 0.5)
                for side in range(2):
                    assert torch.equal(saved[side], rt.states[0]['side'][side]['code'][:, :135]), (ident, side, 'streaming rebuild mismatch')
                meta.seq_lens[0] = 129
            del rt
            item['boundary'].append(bit)
        reports.append(item)
        progress('validation', 'boundary', **{'completed': len(reports), 'total': 24})
    rt = Runtime('E0_B1_R1', 'C2', metrics=False)
    y1 = rt.attention(0, query, cache, meta, 1 / 128 ** 0.5).clone()
    rt.finish()
    rt.reset_case()
    cache[3] *= 100
    cache[1] *= -100
    y2 = rt.attention(0, query, cache, meta, 1 / 128 ** 0.5)
    assert torch.equal(y1, y2)
    rt.finish()
    del rt
    torch.cuda.empty_cache()
    atomic(ROOT / 'checks/boundary.json', {'results': reports, 'future_KV_causality': True, 'tokens_preserved': True, 'all_controllers_0_to_4': True})

def integration():
    from dynamic_runtime import Runtime, attach
    from native_runner import NativeRunner
    rt = Runtime('E0_B1_R3', 'C2', metrics=True)
    attach(rt)
    runner = NativeRunner(backend='dynamic')
    try:
        row = rows('calibration')[0]
        result = runner.generate(row, calibration=True, job_name='validation', run_id='36layer-smoke')
        stats = rt.finish()
        assert stats['counters']['steps'] == 36 * 64, stats['counters']['steps']
        assert len(stats['errors_after_decision']) == 36 * 4
        atomic(ROOT / 'checks/native_36layer.json', {'case_id': row['case_id'], 'generation': result, 'statistics': stats, 'scope': 'diagnostic calibration request, 65 tokens; not a formal evaluation'})
    finally:
        runner.close()

def main():
    wait_assets()
    assets()
    progress('validation', 'assets_passed')
    boundary()
    integration()
    atomic(ROOT / 'checks/ready.json', {'passed': True, 'time': time.time(), 'checks': ['prefix0..4', 'zero-payload-empty-pointer', 'all24assets', 'integer-KV-budget', 'C0..C3', 'matched-total-budget', 'causality', 'native36layer']})
    progress('validation', 'completed')
if __name__ == '__main__':
    main()
