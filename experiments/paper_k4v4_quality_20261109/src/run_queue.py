"""Durable user service; wait for the existing queue before using its GPU."""
import fcntl
import json
import subprocess
import time
import traceback

from common import ROOT, PROJECT, PARENT, METHODS, atomic, cases, freeze, status


def grade(method):
    for python, script, arguments in [
        (PROJECT / '.venv-eval/bin/python', 'score.py', ['--method', method]),
        (PROJECT / '.venv/bin/python', 'report.py', [])]:
        result = subprocess.run([str(python), str(ROOT / 'src' / script), *arguments], cwd=PROJECT)
        if result.returncode:
            atomic(ROOT / 'analysis' / (method + '.grading_error.json'),
                   dict(script=script, exit_code=result.returncode))
            return False
    return True


def main():
    for name in ('logs', 'analysis', 'validation', 'failures', 'results'):
        (ROOT / name).mkdir(exist_ok=True)
    with (ROOT / 'queue.lock').open('a') as own_lock:
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        freeze()
        rows = cases()
        if all(len(list((ROOT / 'results' / m / 'scores').glob('[0-9]*.json'))) == 464 for m in METHODS):
            status('quality_complete', methods=list(METHODS), scored_per_method=464)
            return
        with (PROJECT / 'locks/staged_algorithm_gpu.lock').open('a') as gpu_lock:
            status('waiting_for_existing_queue', parent=str(PARENT), methods=list(METHODS),
                   expected_per_method=464, gpu_started=False,cancelled_methods=['U_Clipped'],
                   handoff='After Gaussian_Tree, using explicit administrator-installed boundary handoff')
            while True:
                parent_status = json.loads((PARENT / 'stage_status.json').read_text())
                handoff = ROOT / 'handoff_status.json'
                approved_handoff = handoff.exists() and json.loads(handoff.read_text()).get('stage') == 'ready_for_paper_methods'
                terminal = approved_handoff
                if terminal:
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        pass
                time.sleep(15)
            run_errors = []
            for method in METHODS:
                freeze()
                while True:
                    checkpoint = ROOT / 'checkpoint.json'
                    status('running_native_quality', method=method, gpu_started=True)
                    with (ROOT / 'logs' / (method + '.log')).open('a') as log:
                        worker = subprocess.Popen(
                            [str(PROJECT / '.venv/bin/python'), '-u', str(ROOT / 'src/generate.py'),
                             '--method', method], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
                        start = time.monotonic()
                        while worker.poll() is None:
                            time.sleep(10)
                            if time.monotonic() - start >= 300:
                                grade(method)
                                start = time.monotonic()
                    if worker.returncode == 0:
                        break
                    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
                    if state.get('method') == method and state.get('stage') == 'request_active':
                        row = state['row']
                        name = f"{row['index']:06d}.json"
                        failure = ROOT / 'failures' / method / name
                        if not failure.exists() and not (ROOT / 'results' / method / name).exists():
                            atomic(failure, dict(row=row, status='failed', exit_code=worker.returncode,
                                                action='Retain failure and continue in a new native engine'))
                            grade(method)
                            continue
                    run_errors.append(dict(method=method, exit_code=worker.returncode, checkpoint=state))
                    atomic(ROOT / 'failures' / (method + '.run_error.json'), run_errors[-1])
                    break
                grade(method)
            counts = {m:dict(generated=len(list((ROOT / 'results' / m).glob('[0-9]*.json'))),
                             scored=len(list((ROOT / 'results' / m / 'scores').glob('[0-9]*.json'))),
                             failed=len(list((ROOT / 'failures' / m).glob('[0-9]*.json')))) for m in METHODS}
            complete = all(c['generated'] == c['scored'] == 464 and not c['failed'] for c in counts.values())
            status('quality_complete' if complete else 'quality_finished_with_failures',
                   methods=counts, run_errors=run_errors, gpu_started=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        status('queue_error', error=traceback.format_exc())
        raise
