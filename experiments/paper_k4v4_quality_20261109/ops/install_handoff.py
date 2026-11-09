"""Install reviewed root scheduling changes; never alter numerical sources."""
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
PYTHON=PROJECT/'.venv/bin/python'
OLD='kvarn-vllm-recovery-20261106.service'
UNIT='kvarn-paper-handoff-20261109.service'


def write_exact(path,text):
    if path.exists():
        assert path.read_text()==text, f'Existing configuration differs: {path}'
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)


def main():
    assert os.geteuid()==0, 'System service scheduling requires administrator privileges'
    override=f'''[Service]
ExecStart=
ExecStart={PYTHON} -u {ROOT}/ops/scoped_parent.py
'''
    # A reboot before handoff resumes the five selected methods, never U_Clipped.
    write_exact(Path('/etc/systemd/system')/(OLD+'.d')/'90-selected-methods.conf',override)
    service=f'''[Unit]
Description=User-requested handoff from Gaussian to paper TurboQuant and KVarN
After=nvidia-persistenced.service
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
Type=simple
User=root
WorkingDirectory={PROJECT}
ExecStart={PYTHON} -u {ROOT}/ops/handoff.py
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Restart=on-failure
RestartSec=60
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
'''
    write_exact(Path('/etc/systemd/system')/UNIT,service)
    subprocess.run(['systemctl','daemon-reload'],check=True)
    subprocess.run(['systemctl','enable','--now',UNIT],check=True)
    print('Installed boundary handoff. Current Gaussian generation is left running.')


if __name__=='__main__':main()
