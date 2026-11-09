"""Run complete live vLLM requests with mandatory same-execution counters."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import CODECS, atomic, rows, sha
from source_guard import verify_original


def main():
    if os.geteuid() != 0:
        raise PermissionError('Physical counters require root')
    owner = (ROOT / 'config.json').stat()
    status = ROOT / 'validation/live_checks.json'
    atomic(ROOT / 'validation/active_process.json', dict(pid=os.getpid(),
           start_time_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    cases = list(rows(ROOT / 'inputs/cases.jsonl'))
    manifest = json.loads((ROOT / 'inputs/manifest.json').read_text())
    assert len(cases) == 464 and sha(ROOT / 'inputs/cases.jsonl') == manifest['cases_sha256']
    assert manifest['generation']['enable_thinking'] and manifest['generation']['do_sample']
    assert not manifest['input_truncation'] and manifest['artificial_output_limit'] is None
    assert all(0 < c['input_tokens'] < 40960 for c in cases)
    verify_original()
    completed = []
    # Include a native page boundary crossing in the separate long-prompt check.
    long_case = next(c['index'] for c in cases if c['dataset'] not in ('humaneval', 'math500') and c['input_tokens'] > 1024)
    checks = [('fp16', 0), ('Jacobian_Tree', 0),
              ('JacK_MixV', 0), ('Data_Tree', 0), ('Gaussian_Tree', 0), ('U_Clipped', 0),
              ('Jacobian_Tree', long_case)]
    stamp = str(time.time_ns())
    try:
        for codec, index in checks:
            tag = f'counter_{codec}_case{index}_{stamp}'
            atomic(status, dict(state='running', completed=completed, total_checks=len(checks),
                                active_codec=codec, active_case=index, active_log=str(ROOT / 'validation' / (tag + '.log')),
                                updated=time.strftime('%Y-%m-%d %H:%M:%S')))
            with (ROOT / 'validation' / (tag + '.log')).open('x') as log:
                subprocess.run([str(ROOT.parents[1] / '.venv/bin/python'), '-u',
                                str(ROOT / 'validation/probe_vllm.py'), codec,
                                '--case', str(index), '--tag', tag], stdout=log, stderr=subprocess.STDOUT, check=True)
            path = ROOT / 'validation' / (tag + '.json')
            record = json.loads(path.read_text())
            physical = record['physical_traffic']
            assert record['backend'] == 'vllm' and physical['status'] == 'measured'
            assert physical['execution_id'] == record['execution_id'] and physical['hardware_passes'] == 1
            assert physical['stages']['prefill']['units'] == record['input_tokens']
            assert physical['stages']['decode']['units'] == record['decode_attention_calls'] // 36
            assert record['decode_graph_modes'].get('FULL', 0) == physical['stages']['decode']['units']
            assert record['natural_completion'] and record['thinking_complete'], 'Live check did not complete naturally'
            if codec != 'fp16':
                assert len(record['vllm_tree_dispatch']) == 36
            # Compare only against the matching privileged launch environment.
            previous = ROOT / 'validation' / ('root_plain_' + codec + '.json')
            equal = None
            if index == 0 and previous.exists():
                equal = record['generated_ids'] == json.loads(previous.read_text())['generated_ids']
                assert equal, 'Enabling hardware collection changed the sampled token sequence'
            completed.append(dict(codec=codec, index=index, result=str(path), generated_tokens=record['generated_tokens'],
                                  elapsed_seconds=record['elapsed_seconds'], dram_total_bytes=physical['dram_total_bytes'],
                                  exact_same_tokens_without_counters=equal))
        atomic(status, dict(state='passed', completed=completed, total_checks=len(checks),
                            original_source_audit=verify_original(), completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    except BaseException:
        atomic(status, dict(state='failed', completed=completed, total_checks=len(checks),
                            error=traceback.format_exc(), updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
    finally:
        for path in (ROOT / 'validation').glob('*'):
            if path.is_file():os.chown(path, owner.st_uid, owner.st_gid)


if __name__ == '__main__':main()
