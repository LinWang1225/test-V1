"""Wait for the active KVarN run, then execute TurboQuant and FP8 serially."""
import json
from pathlib import Path
import subprocess
import time

from run_operator_experiment import ROOT, atomic


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2] != 'Z'


def main():
    queue_path = ROOT / 'results/method_queue.json'
    queue = json.loads(queue_path.read_text())
    def update(**kw):
        queue.update(kw, updated_unix=time.time())
        atomic(queue_path, queue)
        print(json.dumps(kw), flush=True)
    try:
        for item in queue['runs']:
            run = Path(item['run_dir'])
            if item['method'] == 'kvarn':
                update(status='waiting_for_kvarn', active_method='kvarn')
                while True:
                    state = json.loads((run / 'status.json').read_text())
                    if state['status'] == 'failed':
                        raise RuntimeError('KVarN failed; preserving queue for inspection')
                    if not alive(state['pid']):
                        assert state['status'] == 'complete', state
                        break
                    time.sleep(15)
            else:
                if (run / 'status.json').exists():
                    state = json.loads((run / 'status.json').read_text())
                    if state['status'] == 'complete' and not alive(state['pid']):
                        continue
                    assert not alive(state['pid']), 'Method already running'
                run.mkdir(parents=True, exist_ok=True)
                update(status='running', active_method=item['method'])
                with (run / 'background.nohup.log').open('ab') as log:
                    proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u',
                        str(ROOT / 'scripts/run_matched_method_firstpass.py'),
                        '--source', queue['baseline_source'], '--run-dir', str(run),
                        '--method', item['method']], cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    (ROOT / 'results/operator_profile.latest.txt').write_text(str(run) + '\n')
                    update(active_pid=proc.pid)
                    code = proc.wait()
                assert code == 0, f"{item['method']} exited {code}; see {run}"
                assert json.loads((run / 'status.json').read_text())['status'] == 'complete'
            report = subprocess.run([str(ROOT / '.venv/bin/python'),
                str(ROOT / 'scripts/operator_final_report.py'), str(run)], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (run / 'final-report.log').write_text(report.stdout)
            assert report.returncode == 0, f'Report failed: {run}'
        update(status='complete', active_method=None, completed_unix=time.time())
    except BaseException as exc:
        update(status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
