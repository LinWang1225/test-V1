"""Install/serve only the privileged DRAM collector for the sampling experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'experiments/staged_sampling_windowed_20261105'
PYTHON=PROJECT/'.venv/bin/python'
UNIT='kvarn-sampling-dram.service'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--serve',action='store_true')
    args=parser.parse_args()
    if os.geteuid()!=0:
        raise PermissionError('Only the hardware collector requires administrator privileges')
    if args.install:
        unit=f'''[Unit]
Description=Hardware DRAM collector for Qwen sampling experiment
After=nvidia-persistenced.service
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
Type=simple
User=root
WorkingDirectory={PROJECT}
ExecStart={PYTHON} {Path(__file__).resolve()} --serve
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Restart=on-failure
RestartSec=30
TimeoutStopSec=30
KillMode=control-group

[Install]
WantedBy=multi-user.target
'''
        (Path('/etc/systemd/system')/UNIT).write_text(unit)
        subprocess.run(['systemctl','daemon-reload'],check=True)
        subprocess.run(['systemctl','enable','--now',UNIT],check=True)
        print('Installed and started privileged hardware collector',flush=True)
        return
    if not args.serve:
        parser.error('Use --install or --serve')
    owner=(ROOT/'config.json').stat()
    temporary=Path('/var/tmp/kvarn-sampling-ncu')
    temporary.mkdir(mode=0o700,exist_ok=True)
    os.environ['TMPDIR']=str(temporary)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['LD_LIBRARY_PATH']='/usr/local/cuda-12.6/extras/CUPTI/lib64'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    sys.path.insert(0,str(ROOT/'src'))
    try:
        subprocess.run([str(PYTHON),str(ROOT/'src/check_hardware.py')],check=True)
    finally:
        for p in [ROOT/'hardware_capability.json',*(ROOT/'profiles/probes').glob('*')]:
            if p.exists():os.chown(p,owner.st_uid,owner.st_gid)
        for p in (ROOT/'profiles').rglob('*'):
            if p.is_dir() and p.stat().st_uid==0:os.chown(p,owner.st_uid,owner.st_gid)
    capability=json.loads((ROOT/'hardware_capability.json').read_text())
    if not capability.get('available'):
        raise RuntimeError('Root physical counter probe failed: '+capability.get('reason','unknown'))
    from collector_service import main as serve
    serve()


if __name__=='__main__':main()
