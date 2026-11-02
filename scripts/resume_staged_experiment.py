"""Idempotent queue launcher after reboot; no stale PID is trusted."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'experiments/staged_algorithm_20261102'
sys.path.insert(0,str(ROOT/'src'))
from common import atomic

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--foreground',action='store_true')
    args=parser.parse_args()
    # flock is released by the kernel on power loss, unlike a stale PID file.
    guard=(ROOT/'launch.lock').open('a+')
    fcntl.flock(guard,fcntl.LOCK_EX)
    queue_lock=(ROOT/'queue.lock').open('a+')
    try:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Experiment queue is already running; no second worker started.')
        return
    fcntl.flock(queue_lock,fcntl.LOCK_UN)
    cmd=[str(PROJECT/'.venv/bin/python'),str(ROOT/'src/supervise.py')]
    if args.wait_pid is not None:
        cmd+=['--wait-pid',str(args.wait_pid)]
    if args.foreground:
        os.execv(cmd[0],cmd)
    (ROOT/'logs').mkdir(exist_ok=True)
    log_path=ROOT/'logs/stage1_queue.log'
    with log_path.open('ab',buffering=0) as log:
        child=subprocess.Popen(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    # Wait until the new process actually owns the queue lock.
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError(f'Queue exited at launch: {child.returncode}; see {log_path}')
        try:
            fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(queue_lock,fcntl.LOCK_UN)
        except BlockingIOError:
            atomic(ROOT/'background.json',dict(pid=child.pid,command=cmd,log=str(log_path),
                launched=time.strftime('%Y-%m-%d %H:%M:%S'),restart='python scripts/resume_staged_experiment.py'))
            print('Queue started:',child.pid)
            return
        time.sleep(.1)
    raise RuntimeError('Queue did not acquire its lock; inspect the log before retrying')

if __name__=='__main__':
    main()
