"""Validate the changed boundary path before resuming the independent queue."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import atomic

try:
    for codec in ('fp16','Jacobian_Tree'):
        atomic(ROOT/'recovery_status.json',dict(stage='validating_native_context_boundary',codec=codec,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        with (ROOT/'validation'/('capacity_'+codec+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'validation/check_capacity.py'),codec],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['systemctl','disable','kvarn-vllm-encoding-20261105.service'],check=True)
    subprocess.run([sys.executable,str(ROOT/'service.py'),'--install'],check=True)
    atomic(ROOT/'recovery_status.json',dict(stage='resumed',updated=time.strftime('%Y-%m-%d %H:%M:%S')))
except BaseException:
    atomic(ROOT/'recovery_status.json',dict(stage='recovery_validation_failed',error=traceback.format_exc(),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
    raise
