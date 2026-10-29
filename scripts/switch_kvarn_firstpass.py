"""Preserve the current batch, then switch to the first-pass-only runner."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
old = Path((ROOT / 'results/operator_profile.latest.txt').read_text().strip())
initial = json.loads((old / 'status.json').read_text())
pid = initial['pid']
current = old / 'points/kvarn' / initial['point'] / 'state.json'
assert 'run_matched_kvarn.py' in Path(f'/proc/{pid}/cmdline').read_text()
print('Waiting for current batch to finish:', initial['point'], flush=True)
while True:
    if json.loads(current.read_text()).get('status') == 'complete':
        break
    if not Path(f'/proc/{pid}').exists():
        break
    if json.loads((old / 'status.json').read_text())['status'] == 'failed':
        break
    time.sleep(0.25)
if Path(f'/proc/{pid}').exists():
    os.kill(pid, signal.SIGTERM)
    while Path(f'/proc/{pid}').exists():
        stat = Path(f'/proc/{pid}/stat').read_text().split()[2]
        if stat == 'Z':
            break
        time.sleep(0.5)
run = ROOT / 'results' / ('operator_kvarn_firstpass_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
run.mkdir()
for statepath in (old / 'points/kvarn').glob('*/state.json'):
    state = json.loads(statepath.read_text())
    if state['status'] == 'complete' and state['point']['repeat'] == 1:
        dest = run / 'points/kvarn' / statepath.parent.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(statepath.parent, target_is_directory=True)
source = json.loads((old / 'identity.json').read_text())['source']
with (run / 'background.nohup.log').open('ab') as log:
    proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u',
        str(ROOT / 'scripts/run_matched_kvarn_firstpass.py'), '--source', source, '--run-dir', str(run)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / 'results/operator_profile.latest.txt').write_text(str(run) + '\n')
record = {'new_run': str(run), 'pid': proc.pid, 'previous_run': str(old),
          'unique_samples': 672, 'formal_requests_per_method': 672,
          'points_per_method': 89, 'extra_repeat_requests_cancelled': 336}
(old / 'firstpass_switch.json').write_text(json.dumps(record, indent=2) + '\n')
with (ROOT / 'STATUS.txt').open('a') as f:
    f.write('\n用户要求赶时间取消额外复测：KVarN仅运行首轮672条/89批；已完成批次复用，FP16代表复测保留但不计入首轮。新目录：' + str(run) + '\n')
print(json.dumps(record), flush=True)
