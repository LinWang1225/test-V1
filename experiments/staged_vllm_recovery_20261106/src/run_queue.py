"""Durable recovery queue: failed requests/runs are recorded and skipped."""
import fcntl
import json
import os
import pwd
import subprocess
import time
import traceback
from pathlib import Path
from common import ROOT, atomic, run_id, rows
from provenance import QUEUE, freeze
from failure_policy import failures, record_failure, give, CASE_FAILURE_EXIT


def main():
    if os.geteuid()!=0:raise PermissionError('Live physical counters require root')
    owner=(ROOT/'config.json').stat();account=pwd.getpwuid(owner.st_uid).pw_name
    project=ROOT.parents[1];python=project/'.venv/bin/python';evaluator=project/'.venv-eval/bin/python'
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=(project/'locks/staged_algorithm_gpu.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('failure_policy_check.json','capacity_fp16.json','capacity_Jacobian_Tree.json'):
        assert json.loads((ROOT/'validation'/name).read_text())['passed'], 'Recovery validation must pass before starting'
    frozen=freeze()
    assert json.loads((ROOT/'validation/live_checks.json').read_text())['state']=='passed'
    for directory in (ROOT/'analysis',ROOT/'results',ROOT/'failures',ROOT/'checkpoints'):
        directory.mkdir(exist_ok=True);give(directory)
    run_errors=[]
    def update(**value):
        path=ROOT/'stage_status.json'
        atomic(path,dict(**value,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S')));give(path)
    def score_and_report(identity):
        for executable,script,arguments in ((evaluator,'score.py',['--run',identity]),(python,'report.py',[])):
            result=subprocess.run(['runuser','-u',account,'--',str(executable),str(ROOT/'src'/script),*arguments],cwd=project)
            if result.returncode:
                error=ROOT/'analysis'/(identity+'_'+script+'.error.json')
                atomic(error,dict(script=script,run=identity,exit_code=result.returncode,
                       error='Scoring/report failed; generation continues and grading remains incomplete',
                       updated=time.strftime('%Y-%m-%d %H:%M:%S')));give(error)
    for codec in QUEUE:
        assert freeze()==frozen
        identity=run_id(codec,'fixed',4);folder=ROOT/'results'/identity
        folder.mkdir(exist_ok=True);give(folder)
        while True:
            previous_failures=len(failures(identity))
            update(stage='stage1_vllm_encoding_baselines',active_run=identity,
                   queue=[run_id(c,'fixed',4) for c in QUEUE],cases_per_run=464,
                   backend='vllm',physical_measurement_required=True,
                   failed_cases=previous_failures,run_fingerprint=frozen['fingerprint'])
            with (ROOT/'logs'/(identity+'.log')).open('a') as log:
                worker=subprocess.Popen([str(python),'-u',str(ROOT/'src/codec_worker.py'),'--codec',codec],
                    cwd=project,stdout=log,stderr=subprocess.STDOUT)
                last_scoring=time.monotonic()
                try:
                    while True:
                        try:
                            code=worker.wait(timeout=20);break
                        except subprocess.TimeoutExpired:
                            progress_paths=[ROOT/'worker_status.json',ROOT/'checkpoints'/(identity+'.json')]
                            last_activity=max((p.stat().st_mtime for p in progress_paths if p.exists()),default=time.time())
                            if time.time()-last_activity>600:
                                print('No request progress for 600 seconds; recording failure and replacing worker',flush=True)
                                worker.kill();code=worker.wait();break
                            if time.monotonic()-last_scoring>=300:
                                score_and_report(identity);last_scoring=time.monotonic()
                except BaseException:
                    worker.terminate()
                    try:worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:worker.kill();worker.wait()
                    raise
            if code==0:break
            check=ROOT/'checkpoints'/(identity+'.json')
            checkpoint=json.loads(check.read_text()) if check.exists() else {}
            if checkpoint.get('stage')=='request_active':
                row=checkpoint['row'];output=folder/f"{row['index']:06d}.json"
                if not output.exists():
                    tail='\n'.join((ROOT/'logs'/(identity+'.log')).read_text(errors='replace').splitlines()[-60:])
                    record_failure(identity,row,error=tail,exit_code=code,
                        execution_id=checkpoint.get('execution_id'),profile_base=checkpoint.get('profile_base'),
                        source_log=str(ROOT/'logs'/(identity+'.log')))
            current_failures=len(failures(identity))
            if current_failures>previous_failures:
                update(stage='restarting_after_recorded_case_failure',active_run=identity,
                       failed_cases=current_failures,exit_code=code)
                score_and_report(identity)
                continue
            # A model initialization or non-case error has no safe request to
            # blame. Record a run error and continue to the next codec.
            error=dict(run=identity,exit_code=code,checkpoint=checkpoint,
                       message='Worker failed without new case progress; remaining cases unattempted',
                       recorded_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            path=ROOT/'failures'/(identity+'.run_error.json');atomic(path,error);give(path)
            run_errors.append(error);break
        score_and_report(identity)
        success=len(list(folder.glob('[0-9]*.json')));failed=len(failures(identity))
        graded=len(list((folder/'scores').glob('[0-9]*.json')))
        update(stage='codec_finished_with_failures' if failed or success!=464 or graded!=success else 'codec_complete',
               active_run=identity,successful_cases=success,failed_cases=failed,
               scored_cases=graded,unattempted_cases=464-success-failed)
    summary={}
    for codec in QUEUE:
        identity=run_id(codec,'fixed',4);success=len(list((ROOT/'results'/identity).glob('[0-9]*.json')))
        failed=len(failures(identity));graded=len(list((ROOT/'results'/identity/'scores').glob('[0-9]*.json')))
        summary[identity]=dict(successful=success,failed=failed,unattempted=464-success-failed,scored=graded,unscored=success-graded)
    has_errors=bool(run_errors) or any(x['failed'] or x['unattempted'] or x['unscored'] for x in summary.values())
    update(stage='stage1_finished_with_failures' if has_errors else 'stage1_vllm_encoding_baselines_complete',
           runs=summary,run_errors=run_errors,
           next='Review valid results and recorded failures; failures are not successful benchmark completions')


if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'queue_failure.json',dict(error=traceback.format_exc(),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
