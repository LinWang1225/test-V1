"""Durable sequential first-stage queue. Failures stop; no skipped samples."""
import fcntl
import argparse
import subprocess
import sys
from common import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    lock=(ROOT/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Shared advisory lock prevents two of our new experiment queues using GPU.
    gpu_lock=(ROOT.parents[1]/'locks/staged_algorithm_gpu.lock').open('w')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.wait_pid is not None:
        watched=Path('/proc')/str(args.wait_pid)/'stat'
        before=watched.read_text().split()[21] if watched.exists() else None
        atomic(ROOT/'stage_status.json',dict(stage='waiting_for_natural_generation_check',pid=os.getpid(),
            wait_pid=args.wait_pid,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        while watched.exists():
            try:
                info=watched.read_text().split()
            except FileNotFoundError:
                break
            if info[21]!=before or info[2]=='Z':
                break
            time.sleep(10)
    project=ROOT.parents[1]
    python=project/'.venv/bin/python';evaluator=project/'.venv-eval/bin/python'
    source_hashes={p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))}
    def execute(script,args=(),evaluation=False):
        assert {p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))}==source_hashes,'Source changed while queue running'
        cmd=[str(evaluator if evaluation else python),str(ROOT/'src'/script),*args]
        process=subprocess.Popen(cmd,cwd=project)
        last_score=time.monotonic()
        while True:
            try:
                code=process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                if script=='generate.py' and time.monotonic()-last_score>=300:
                    current=json.loads((ROOT/'stage_status.json').read_text())['active_run']
                    score_code=subprocess.call([str(evaluator),str(ROOT/'src/score.py'),'--run',current],cwd=project)
                    if score_code:
                        process.terminate();process.wait()
                        raise RuntimeError('Incremental scoring failed; generation stopped')
                    subprocess.check_call([str(python),str(ROOT/'src/report.py')],cwd=project)
                    last_score=time.monotonic()
        if code:
            atomic(ROOT/'stage_status.json',dict(stage='stage1_failed',command=cmd,exit_code=code,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
            raise RuntimeError(f'Worker failed: {cmd}')
    execute('prepare.py')
    if not (ROOT/'codebooks/ready.json').exists():
        execute('prepare_books.py')
    execute('check.py')
    execute('check_model.py')
    execute('metadata_report.py')
    execute('check_sampling.py')
    queue=[(m,('--codec',m,'--depth','4'))
        for m in ('fp16','Jacobian_Tree','JacK_MixV','Data_Tree','Gaussian_Tree','U_Clipped')]
    for codec,args in queue:
        identity=run_id(codec,'fixed',4)
        atomic(ROOT/'stage_status.json',dict(stage='stage1_encoding_baselines',active_run=identity,pid=os.getpid(),
            queue=[run_id(c,'fixed',4) for c,_ in queue],
            datasets=list(json.loads((ROOT/'inputs/manifest.json').read_text())['datasets']),
            cases_per_run=464,generation=GENERATION,physical_measurement_required=True,
            source_hashes=source_hashes,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        execute('generate.py',args)
        execute('score.py',('--run',identity),evaluation=True)
        execute('report.py')
    atomic(ROOT/'stage_status.json',dict(stage='stage1_encoding_baselines_complete',
        next='Review paired quality and prefix/controller checks before freezing codec; basis/controller/engineering stages not yet executed.',
        updated=time.strftime('%Y-%m-%d %H:%M:%S')))

if __name__=='__main__':
    try:
        main()
    except BaseException:
        import traceback
        atomic(ROOT/'queue_failure.json',dict(error=traceback.format_exc(),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
