"""Start a root DRAM collector and run a command as the invoking user.

Usage: .venv/bin/python scripts/run_with_dram.py --probe-only
       .venv/bin/python scripts/run_with_dram.py -- COMMAND [ARG ...]
Passwords are handled interactively by sudo, never by this script.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'experiments/full_quality_io_20261027'
PYTHON = PROJECT / '.venv/bin/python'


def privileged(probe_only):
    if os.geteuid() != 0:
        raise PermissionError('Collector requires sudo')
    uid, gid = int(os.environ['SUDO_UID']), int(os.environ['SUDO_GID'])
    folder = ROOT / 'profiles/probes'
    before = set(folder.glob('*'))
    try:
        subprocess.run([str(PYTHON), str(ROOT / 'src/check_hardware.py')], check=True)
    finally:
        for path in (set(folder.glob('*')) - before) | {ROOT / 'hardware_capability.json'}:
            if path.exists():
                os.chown(path, uid, gid)
        for path in (ROOT / 'profiles', folder):
            if path.exists() and path.stat().st_uid == 0:
                os.chown(path, uid, gid)
    capability = json.loads((ROOT / 'hardware_capability.json').read_text())
    if not capability.get('available'):
        raise RuntimeError(capability.get('reason', 'Root DRAM probe failed'))
    print(json.dumps(capability['probe_metrics'], ensure_ascii=False), flush=True)
    if probe_only:
        return
    sys.path.insert(0, str(ROOT / 'src'))
    from collector_service import main
    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    finally:
        status = ROOT / 'collector_status.json'
        if status.exists() and json.loads(status.read_text()).get('pid') == os.getpid():
            status.unlink()


def main():
    args = sys.argv[1:]
    if args in (['--root-probe'], ['--root-collector']):
        privileged(args == ['--root-probe'])
        return
    if not args or args == ['--help']:
        print(__doc__)
        return
    probe_only = args == ['--probe-only']
    if not probe_only and (args[0] != '--' or len(args) < 2):
        raise SystemExit('Use --probe-only or -- COMMAND [ARG ...]')
    if os.geteuid() == 0:
        raise SystemExit('Run this launcher as your normal user; it invokes sudo itself')
    status = ROOT / 'collector_status.json'
    if status.exists() and time.time() - json.loads(status.read_text()).get('heartbeat_epoch', 0) < 30:
        raise SystemExit('A collector is already running; use it or stop it before launching another')
    subprocess.run(['sudo', '-v'], check=True)
    command = ['sudo', '-n', str(PYTHON), str(Path(__file__).resolve()),
               '--root-probe' if probe_only else '--root-collector']
    if probe_only:
        subprocess.run(command, check=True)
        return
    with (ROOT / 'logs/dram_collector.log').open('a') as log:
        collector = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 150
            while True:
                if collector.poll() is not None:
                    raise RuntimeError('Collector exited; see logs/dram_collector.log')
                if status.exists() and time.time() - json.loads(status.read_text()).get('heartbeat_epoch', 0) < 10:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Collector startup timed out')
                time.sleep(.5)
            print('Root DRAM collector ready; starting command as UID', os.getuid(), flush=True)
            result = subprocess.run(args[1:], cwd=PROJECT)
        finally:
            if collector.poll() is None:
                collector.terminate()
                collector.wait(timeout=135)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
