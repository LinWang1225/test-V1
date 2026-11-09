"""Independent root hardware queue; all answer scoring runs as the normal user."""
import fcntl
import json
import os
import pwd
import subprocess
import time
import traceback
from pathlib import Path
from common import ROOT, atomic, run_id, sha
from provenance import QUEUE, freeze


def main():
    if os.geteuid() != 0:raise PermissionError('The live CUPTI worker requires root')
    owner = (ROOT / 'config.json').stat()
    account = pwd.getpwuid(owner.st_uid).pw_name
    project = ROOT.parents[1]
    python = project / '.venv/bin/python'
    evaluator = project / '.venv-eval/bin/python'
    lock = (ROOT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = (project / 'locks/staged_algorithm_gpu.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def update(**value):
        atomic(ROOT / 'stage_status.json', dict(**value, pid=os.getpid(),
               updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        os.chown(ROOT / 'stage_status.json', owner.st_uid, owner.st_gid)

    # The validation matrix is separate; failed or missing validation stops launch.
    while True:
        checks = json.loads((ROOT / 'validation/live_checks.json').read_text())
        if checks['state'] == 'passed':break
        if checks['state'] != 'running':raise RuntimeError('Live vLLM validation failed: ' + checks.get('error', 'unknown'))
        process_info = json.loads((ROOT / 'validation/active_process.json').read_text())
        process_stat = Path('/proc') / str(process_info['pid']) / 'stat'
        assert process_stat.exists(), 'Validation process stopped before reporting a result'
        assert process_stat.read_text().split()[21] == process_info['start_time_ticks'], 'Validation process identity changed'
        update(stage='waiting_for_live_vllm_validation', checks_completed=len(checks['completed']),
               checks_total=checks['total_checks'])
        time.sleep(10)
    last_check = json.loads(Path(checks['completed'][-1]['result']).read_text())
    assert last_check['persistent_engine_reset_validated'], 'Persistent engine reuse must pass before the formal queue'
    assert last_check['quantized_counter_control_validated'], 'Quantized counter equivalence must pass before formal work'
    frozen = freeze()
    assert json.loads((ROOT / 'checks_gpu.json').read_text())['passed']
    assert json.loads((ROOT / 'validation/counter_equivalence.json').read_text())['passed']
    assert json.loads((ROOT / 'validation/scorer_checks.json').read_text())['passed']
    for directory in (ROOT / 'analysis', ROOT / 'results'):
        directory.mkdir(exist_ok=True)
        os.chown(directory, owner.st_uid, owner.st_gid)

    def score_and_report(identity):
        for executable, script, arguments in ((evaluator, 'score.py', ['--run', identity]),
                                               (python, 'report.py', [])):
            subprocess.run(['runuser', '-u', account, '--', str(executable),
                            str(ROOT / 'src' / script), *arguments], cwd=project, check=True)

    for codec in QUEUE:
        assert freeze() == frozen
        identity = run_id(codec, 'fixed', 4)
        update(stage='stage1_vllm_encoding_baselines', active_run=identity,
               queue=[run_id(c, 'fixed', 4) for c in QUEUE], cases_per_run=464,
               backend='vllm', physical_measurement_required=True, run_fingerprint=frozen['fingerprint'])
        with (ROOT / 'logs' / (identity + '.log')).open('a') as log:
            worker = subprocess.Popen([str(python), '-u', str(ROOT / 'src/codec_worker.py'),
                                       '--codec', codec], cwd=project, stdout=log, stderr=subprocess.STDOUT)
            last_scoring = time.monotonic()
            try:
                while True:
                    try:
                        code = worker.wait(timeout=20)
                        if code:raise RuntimeError(f'vLLM {identity} worker failed ({code}); see {log.name}')
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - last_scoring >= 300:
                            score_and_report(identity)
                            last_scoring = time.monotonic()
            except BaseException:
                worker.terminate()
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill();worker.wait()
                raise
        score_and_report(identity)
        folder = ROOT / 'results' / identity
        assert len(list(folder.glob('[0-9]*.json'))) == 464
        assert len(list((folder / 'traffic_totals').glob('[0-9]*.json'))) == 464
        assert json.loads((folder / 'scoring_status.json').read_text())['complete']
    update(stage='stage1_vllm_encoding_baselines_complete',
           next='Review all six complete quality/traffic results before subsequent research stages')


if __name__ == '__main__':
    try:main()
    except BaseException:
        atomic(ROOT / 'queue_failure.json', dict(error=traceback.format_exc(),
               updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
