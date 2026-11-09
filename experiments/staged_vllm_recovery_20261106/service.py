"""Install this experiment's own unit without changing existing launchers."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
UNIT = 'kvarn-vllm-recovery-20261106.service'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true', required=True)
    parser.parse_args()
    assert os.geteuid() == 0
    text = f'''[Unit]
Description=Isolated vLLM encoding experiment with live hardware counters
After=nvidia-persistenced.service

[Service]
Type=simple
User=root
WorkingDirectory={PROJECT}
ExecStart={PROJECT}/.venv/bin/python -u {ROOT}/src/run_queue.py
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64
Environment=PATH={PROJECT}/.venv/bin:/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=MAX_JOBS=4
Restart=no
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
'''
    path = Path('/etc/systemd/system') / UNIT
    if path.exists():
        assert path.read_text() == text, 'Existing independent unit differs; refusing to overwrite'
    else:
        path.write_text(text)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'enable', '--now', UNIT], check=True)
    print('Started ' + UNIT, flush=True)


if __name__ == '__main__':main()
