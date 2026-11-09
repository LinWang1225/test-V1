"""Privileged, persistent method-boundary handoff; never interrupt Gaussian."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
PARENT=ROOT.parent/'staged_vllm_recovery_20261106'
UNIT='kvarn-vllm-recovery-20261106.service'
OWNER=(ROOT/'experiment.json').stat()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path);os.chown(path,OWNER.st_uid,OWNER.st_gid)


def update(stage,**values):
    value=dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S'),
               cancelled_methods=['U_Clipped'],**values)
    atomic(ROOT/'handoff_status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def full_gaussian():
    folder=PARENT/'results/Gaussian_Tree__fixed4'
    return len(list(folder.glob('[0-9]*.json')))==464 and len(list((folder/'scores').glob('[0-9]*.json')))==464


def main():
    assert os.geteuid()==0
    current=ROOT/'handoff_status.json'
    if current.exists() and json.loads(current.read_text()).get('stage')=='ready_for_paper_methods':return
    update('waiting_for_gaussian_completion',original_unit=UNIT)
    previous=None
    while not full_gaussian():
        count=len(list((PARENT/'results/Gaussian_Tree__fixed4').glob('[0-9]*.json')))
        if count!=previous:
            update('waiting_for_gaussian_completion',gaussian_generated=count,expected=464)
            previous=count
        if subprocess.run(['systemctl','is-active','--quiet',UNIT]).returncode:
            subprocess.run(['systemctl','start',UNIT],check=True)
        time.sleep(1)
    update('stopping_old_queue_at_method_boundary',gaussian_generated=464)
    subprocess.run(['systemctl','stop',UNIT],check=True)
    subprocess.run(['systemctl','disable',UNIT],check=True)
    account=pwd.getpwuid(OWNER.st_uid).pw_name
    for executable,script,args in [
        (PROJECT/'.venv-eval/bin/python','score.py',['--run','Gaussian_Tree__fixed4']),
        (PROJECT/'.venv/bin/python','report.py',[])]:
        subprocess.run(['runuser','-u',account,'--',str(executable),str(PARENT/'src'/script),*args],
                       cwd=PROJECT,check=True)
    assert full_gaussian()
    # Observe lock release before enabling the successor. The successor also
    # acquires this lock, so both sides enforce exclusive GPU execution.
    with (PROJECT/'locks/staged_algorithm_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=json.loads((PARENT/'stage_status.json').read_text())
        atomic(ROOT/'ops/parent_status_at_handoff.json',original)
        atomic(PARENT/'queue_control.json',dict(user_requested=True,
            cancelled_methods=['U_Clipped'],cancelled_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            gaussian_complete=True,next_experiment=str(ROOT)))
        atomic(PARENT/'stage_status.json',dict(stage='stage1_finished_with_user_cancellation',
            completed_methods=['fp16','Jacobian_Tree__fixed4','JacK_MixV__fixed4','Data_Tree__fixed4','Gaussian_Tree__fixed4'],
            skipped_methods=['U_Clipped'],reason='Explicit user cancellation on 2026-11-09',
            next_experiment=str(ROOT),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        update('ready_for_paper_methods',gaussian_complete=True,
               u_clipped_generated=len(list((PARENT/'results/U_Clipped__fixed4').glob('[0-9]*.json'))),
               next_methods=['turboquant_paper_k4v4','kvarn_k4v4_g128'])


if __name__=='__main__':
    try:main()
    except BaseException:
        update('handoff_error',error=traceback.format_exc());raise
