import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
import json, time, hashlib, gc, traceback
from pathlib import Path
from common import ROOT, atomic, progress, rows, sha
from formal_queue import runs

def source_digest():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src').glob('*.py'))}

def wait_gate():
    start = time.time()
    while not (ROOT / 'checks/ready.json').exists():
        pointer = ROOT / 'jobs/validation.json'
        if pointer.exists():
            p = Path(json.loads(pointer.read_text())['job_dir']) / 'state.json'
            if p.exists() and json.loads(p.read_text()).get('status') == 'failed':
                raise RuntimeError('validation failed: formal generation remains blocked')
        if time.time() - start > 14400:
            raise TimeoutError('validation')
        progress('formal', 'waiting_for_validation', **{'formal_generation_started': False})
        time.sleep(15)

def lock_protocol():
    lock = ROOT / 'protocol_lock.json'
    payload = {'allocator_revision': 'v2-convex-hull-jumps-total-unit-risk', 'sources': source_digest(), 'generator_cases': sha(ROOT / 'inputs/generator_cases.jsonl'), 'scoring_cases': sha(ROOT / 'inputs/scoring_cases.jsonl'), 'calibration_cases': sha(ROOT / 'inputs/calibration_cases.jsonl'), 'configuration': sha(ROOT / 'config.json'), 'selection': sha(ROOT / 'inputs/selection.json'), 'codebook_manifest': sha(ROOT / 'codebooks/complete.json'), 'validation': sha(ROOT / 'checks/ready.json'), 'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'algorithm_phase': 'logical accesses, uint8 prefix emulation, native FP16 shadow for observer only', 'queue_order': 'phase, frozen case, configuration; case-major paired comparisons', 'primary_run_identities': 145, 'supplementary_total_cost_3bit': 96, 'later_2bit_extension': 216, 'formal_cases': 90, 'thinking': 'native Qwen3 thinking prompt, same original token IDs', 'output_limit': '40960 total native tokens only', 'matched_total_cost': 'per KV side and decode step: b*arrived_elements plus max B2/R3/C2 logical metadata and basis/index accesses; same ceiling for every configuration; unused beneficial budget allowed', 'metadata_access_definition': 'logical unique tensor accesses per read unit; matrices 2 passes/side/step, FP16 local params per group, curve/ratio and uint8 decision arrays, global mean/scale and LUT; updates and resident bytes reported separately', 'hardware_claims': False}
    if lock.exists():
        assert json.loads(lock.read_text()) == payload, 'protocol changed after freeze'
    else:
        atomic(lock, payload)
    return sha(lock)

def main():
    wait_gate()
    protocol = lock_protocol()
    phases = runs()
    cases = rows()
    assert len(cases) == 90
    for row in cases:
        assert set(row) <= set(['case_id', 'dataset', 'index', 'input_ids', 'input_tokens', 'input_ids_sha256', 'sampling_seed'])
    atomic(ROOT / 'execution_queue.json', {'protocol': protocol, 'phases': phases, 'cases': 90, 'main3': 145, 'all_including_supplements': 457})
    import torch
    from dynamic_runtime import Runtime, attach
    from native_runner import NativeRunner
    attach(None)
    runner = NativeRunner(backend='dynamic')
    completed = 0
    try:
        for phase in phases:
            stage = phase[0]['stage']
            for ci, row in enumerate(cases):
                filename = f'{ci:03}_' + hashlib.sha256(row['case_id'].encode()).hexdigest()[:12] + '.json'
                for run in phase:
                    path = ROOT / 'results' / run['run_id'] / filename
                    if path.exists():
                        previous = json.loads(path.read_text())
                        assert previous['protocol'] == protocol and previous['input_ids_sha256'] == row['input_ids_sha256'] and (previous['configuration'] == run)
                        completed += 1
                        continue
                    progress('formal', 'generating', **{'formal_generation_started': True, 'stage_name': stage, 'run_id': run['run_id'], 'case_id': row['case_id'], 'case_position': ci + 1, 'cases': 90, 'completed_requests': completed})
                    atomic(ROOT / 'status.json', {'stage': 'formal_generation', 'formal_generation_started': True, 'run_id': run['run_id'], 'case_id': row['case_id'], 'case_position': ci + 1, 'phase': stage, 'completed_requests': completed, 'protocol': protocol, 'time': time.time()})
                    rt = None
                    if run['kind'] != 'fp16':
                        rt = Runtime(run['identity'], controller=run.get('controller'), budget=run.get('budget', 3), uniform=run.get('uniform'), metrics=True, cost_mode=run['cost_mode'])
                    attach(rt)
                    torch.cuda.reset_peak_memory_stats()
                    generated = runner.generate(row, job_name='formal', run_id=run['run_id'])
                    assert generated['artificial_output_limit'] is None
                    assert not generated['input_truncated']
                    assert generated['native_context_length'] == 40960
                    assert generated['layers_exercised'] == list(range(36))
                    statistics = rt.finish() if rt else {'kind': 'native_fp16', 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                    record = {'protocol': protocol, 'configuration': run, 'case_id': row['case_id'], 'dataset': row['dataset'], 'index': row['index'], 'case_position': ci, 'input_ids_sha256': row['input_ids_sha256'], 'sampling_seed': row['sampling_seed'], 'input_tokens': row['input_tokens'], 'generation': generated, 'statistics': statistics, 'finished_time': time.time()}
                    atomic(path, record)
                    completed += 1
                    attach(None)
                    del rt
                    gc.collect()
                    torch.cuda.empty_cache()
                    progress('formal', 'request_completed', **{'run_id': run['run_id'], 'case_id': row['case_id'], 'completed_requests': completed, 'total_requests': 457 * 90})
            atomic(ROOT / 'jobs' / f'{stage}_complete.json', {'requests': len(phase) * 90, 'time': time.time(), 'protocol': protocol})
        atomic(ROOT / 'jobs/generation_complete.json', {'requests': completed, 'protocol': protocol, 'time': time.time()})
        atomic(ROOT / 'status.json', {'stage': 'generation_completed', 'formal_generation_started': True, 'requests': completed, 'protocol': protocol, 'time': time.time()})
    finally:
        runner.close()
if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        atomic(ROOT / 'jobs/formal_failure.json', {'error': str(e), 'traceback': traceback.format_exc(), 'time': time.time()})
        raise
