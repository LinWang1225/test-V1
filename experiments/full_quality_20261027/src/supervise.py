"""Detached full benchmark with per-case resume and fail-closed official scoring."""
import sys,subprocess,traceback,fcntl
from common import *


def main():
    lock=(ROOT/'logs/pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model_python=PROJECT/'.venv/bin/python';score_python=PROJECT/'.venv-eval/bin/python'
    inputs=json.loads((ROOT/'inputs/manifest.json').read_text());assert inputs['total_cases_per_policy']==5414
    files=sorted([*ROOT.glob('src/*.py'),*ROOT.glob('vendor/**/*.py'),*ROOT.glob('vendor/longbench/config/*.json'),ROOT/'config.json',ROOT/'PROTOCOL.md',ROOT/'inputs/manifest.json',*sorted(BOOKS.glob('layer*_seed20261024.npz')),ROOT.parent/'encoding_ablation_20261024/src/codec.py'])
    signature={str(p):sha(p) for p in files}
    frozen=ROOT/'provenance.json'
    if frozen.exists():assert json.loads(frozen.read_text())['source_input_hashes']==signature,'Frozen benchmark code or inputs changed'
    else:atomic(frozen,dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),source_input_hashes=signature,frozen_dataset_hashes={n:v['sha256'] for n,v in inputs['datasets'].items()},official_sources=inputs['official_sources'],model_python=str(model_python),score_python=str(score_python)))
    for n,info in inputs['datasets'].items():assert sha(ROOT/info['path'])==info['sha256'],n
    def phase(stage,**kw):atomic(ROOT/'pipeline_status.json',dict(stage=stage,supervisor_pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))
    def run(stage,python,script,*args):
        assert signature=={str(p):sha(p) for p in files},'Frozen source/input modification during run'
        print('\nSTART',stage,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True);t0=time.time()
        child=subprocess.Popen([str(python),'-u',str(script),*args],cwd=PROJECT)
        while True:
            phase(stage,worker_pid=child.pid,elapsed_seconds=time.time()-t0)
            try:rc=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:pass
        atomic(ROOT/'logs'/(stage+'_exit.json'),dict(exit_code=rc,elapsed_seconds=time.time()-t0,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
        if rc:raise RuntimeError(stage+f' exited {rc}')
    try:
        run('model_and_natural_EOS_gates',model_python,ROOT/'src/check.py','--model')
        run('official_scoring_canonical_gates',score_python,ROOT/'src/check_scoring.py')
        # Actual task-length generation on short/code, math, long summary and
        # Chinese prompts gives a meaningful pilot before the full queue.
        pilots=['humaneval','math500','qasper','gov_report','multifieldqa_zh']
        for dataset in pilots:
            for policy in POLICIES:
                run(f'pilot_generate_{dataset}_{policy}',model_python,ROOT/'src/generate.py','--dataset',dataset,'--policy',policy,'--pilot')
                run(f'pilot_score_{dataset}_{policy}',score_python,ROOT/'src/score.py','--dataset',dataset,'--policy',policy,'--pilot')
        atomic(ROOT/'pilot_complete.json',dict(complete=True,generations=len(pilots)*len(POLICIES),datasets=pilots,full_dataset_counts_unchanged=True))
        for dataset in inputs['datasets']:
            for policy in POLICIES:
                run(f'generate_{dataset}_{policy}',model_python,ROOT/'src/generate.py','--dataset',dataset,'--policy',policy)
                run(f'score_{dataset}_{policy}',score_python,ROOT/'src/score.py','--dataset',dataset,'--policy',policy)
                run(f'report_{dataset}_{policy}',score_python,ROOT/'src/report.py')
        assert json.loads((ROOT/'results/complete.json').read_text())['official_task_scores_complete']
        phase('completed',official_quality_complete=True,total_generations=inputs['total_generations'])
        atomic(ROOT/'background_exit.json',dict(exit_code=0,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
    except BaseException:
        error=traceback.format_exc();phase('failed',traceback=error)
        atomic(ROOT/'background_exit.json',dict(exit_code=1,traceback=error,finished=time.strftime('%Y-%m-%d %H:%M:%S')));print(error,flush=True);raise


if __name__=='__main__':main()
