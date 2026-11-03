"""systemd foreground guardian; watches an existing queue or resumes it.

Kept outside experiment/src so deployment changes do not invalidate numerical
provenance. All children stay in the service cgroup; no detached worker here.
"""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'experiments/staged_algorithm_20261102'
sys.path.insert(0,str(ROOT/'src'))
from common import atomic

def queue_active():
    with (ROOT/'queue.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False

def complete():
    path=ROOT/'stage_status.json'
    if not path.exists():
        return False
    return json.loads(path.read_text()).get('stage')=='stage1_encoding_baselines_complete'

def state(phase,**extra):
    atomic(ROOT/'autostart_status.json',dict(phase=phase,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**extra))
    print(phase,extra,flush=True)

def wait_for_gpu():
    for attempt in range(40):
        try:
            probe=subprocess.run(['/usr/bin/nvidia-smi','-L'],capture_output=True,text=True,timeout=15)
            if probe.returncode==0 and 'GPU ' in probe.stdout:
                return
        except (OSError,subprocess.TimeoutExpired):
            pass
        if attempt==0:
            state('waiting_for_gpu_driver')
        time.sleep(15)
    raise RuntimeError('GPU unavailable after boot; systemd will apply its restart policy')

def main():
    watching=False
    while queue_active():
        if not watching:
            state('watching_existing_queue')
            watching=True
        time.sleep(15)
    if complete():
        state('queue_complete')
        return 0
    wait_for_gpu()
    state('resuming_queue')
    # --foreground is essential: systemd must supervise the real queue process.
    cmd=[str(PROJECT/'.venv/bin/python'),str(PROJECT/'scripts/resume_staged_experiment.py'),'--foreground']
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    with (ROOT/'logs/stage1_queue.log').open('ab',buffering=0) as log:
        code=subprocess.call(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
    if code:
        state('queue_failed',exit_code=code)
        return code
    if complete():
        state('queue_complete')
        return 0
    # A manual launcher may have won the startup race; watch it on next retry.
    state('queue_handoff_or_unexpected_exit')
    return 1

if __name__=='__main__':
    raise SystemExit(main())
