"""Detached, fail-closed sequential encoding -> causal replay -> online pilot."""
import sys,os,json,time,subprocess,traceback,fcntl,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
C=R.parent/'controller_jacobian_20261026'


def atomic(path,data):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    lock=(R/'logs/pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tracked=sorted([*R.glob('src/*.py'),*C.glob('src/*.py'),R/'config.json',R/'PROTOCOL.md',C/'PROTOCOL.md'])
    sources={str(p):digest(p) for p in tracked}
    freeze=R/'provenance.json'
    if freeze.exists():
        previous=json.loads(freeze.read_text())
        assert sources==previous['source_hashes'],'Sources changed: use a new run directory rather than mixing results'
    else:
        base=R.parent/'encoding_ablation_20261024'
        assets=[base/'inputs/manifest.json',base/'inputs/smoke.json',*sorted((base/'calibration/full').glob('*.npz')),*sorted((base/'supplementary_experiments/codebooks/full').glob('layer*.npz')),*sorted((base/'supplementary_experiments/codebooks/smoke').glob('layer*.npz'))]
        atomic(freeze,dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),source_hashes=sources,input_asset_hashes={str(p):digest(p) for p in assets},python=sys.executable,encoding_then_controller=True))
    def status(stage,**kw):
        atomic(R/'pipeline_status.json',dict(stage=stage,supervisor_pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))
    def run(stage,script,*args):
        assert sources=={str(p):digest(p) for p in tracked},'Frozen source changed during run'
        status(stage)
        print('\nSTART',stage,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        start=time.time()
        process=subprocess.Popen([sys.executable,'-u',str(script),*args],cwd=R.parents[1])
        while True:
            status(stage,worker_pid=process.pid,elapsed_seconds=time.time()-start)
            try:code=process.wait(timeout=30);break
            except subprocess.TimeoutExpired:pass
        atomic(R/'logs'/f'{stage}_exit.json',dict(exit_code=code,elapsed_seconds=time.time()-start,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
        if code:raise RuntimeError(f'{stage}: worker exit {code}')
    try:
        run('encoding_numeric_gates',R/'src/check.py')
        run('encoding_smoke_fit',R/'src/train.py','--smoke')
        run('encoding_smoke_evaluate',R/'src/evaluate.py','--smoke')
        run('encoding_smoke_analysis',R/'src/analyze.py','--smoke')
        run('encoding_fit_60_calibration',R/'src/train.py')
        run('encoding_evaluate_90_test',R/'src/evaluate.py')
        run('encoding_analysis_and_baseline_audit',R/'src/analyze.py')
        assert json.loads((R/'results/full/analysis_complete.json').read_text())['complete']
        run('controller_numeric_causality_gates',C/'src/check.py')
        run('controller_causal_smoke',C/'src/run.py','--stage','replay','--smoke')
        run('controller_causal_smoke_analysis',C/'src/analyze.py','--smoke')
        run('controller_causal_90_test',C/'src/run.py','--stage','replay')
        run('controller_causal_analysis',C/'src/analyze.py')
        run('controller_online_smoke',C/'src/run.py','--stage','online','--smoke')
        run('controller_online_smoke_analysis',C/'src/analyze.py','--online','--smoke')
        run('controller_online_12_case_pilot',C/'src/run.py','--stage','online')
        run('controller_online_pilot_analysis',C/'src/analyze.py','--online')
        status('completed',encoding_completed=True,causal_replay_completed=True,online_pilot_completed=True,official_task_scoring_completed=False,compressed_kernel_completed=False)
        atomic(R/'background_exit.json',dict(exit_code=0,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
    except BaseException:
        error=traceback.format_exc();status('failed',traceback=error)
        atomic(R/'background_exit.json',dict(exit_code=1,traceback=error,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
        print(error,flush=True);raise


if __name__=='__main__':main()
