"""Continue full LongBench quality after the frozen HumanEval/MATH phase."""
import sys,subprocess,traceback,fcntl
from common import *


def main():
    lock=(ROOT/'logs/pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    for policy in POLICIES:
        for dataset,count in [('humaneval',164),('math500',500)]:
            info=json.loads((ROOT/'results/full'/policy/dataset/'score_summary.json').read_text())
            assert info['complete'] and info['n_cases']==count,'Legacy dataset not complete at version boundary'
    paths=sorted([*ROOT.glob('src/*.py'),ROOT/'config.json',ROOT/'PROTOCOL.md',ROOT/'inputs/manifest.json',*BOOKS.glob('layer*_seed20261024.npz')])
    signature={str(p):sha(p) for p in paths};provenance=ROOT/'provenance.json'
    if provenance.exists():assert json.loads(provenance.read_text())['source_hashes']==signature
    else:atomic(provenance,dict(source_hashes=signature,version_boundary='HumanEval and MATH-500 frozen v1; all 21 LongBench datasets v2',legacy_run=str(ROOT.parent/'full_quality_20261027'),hardware_failure_does_not_stop_quality=True))
    model_python=PROJECT/'.venv/bin/python';score_python=PROJECT/'.venv-eval/bin/python'
    def phase(name,**kw):atomic(ROOT/'pipeline_status.json',dict(stage=name,supervisor_pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))
    def run(name,python,script,*args,soft=False):
        assert signature=={str(p):sha(p) for p in paths},'New source modified after freeze'
        print('\nSTART',name,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True);start=time.time()
        child=subprocess.Popen([str(python),'-u',str(script),*args],cwd=PROJECT)
        while True:
            phase(name,worker_pid=child.pid,elapsed_seconds=time.time()-start)
            try:rc=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:pass
        atomic(ROOT/'logs'/(name+'_exit.json'),dict(exit_code=rc,elapsed_seconds=time.time()-start))
        if rc and not soft:raise RuntimeError(name+f' exit {rc}')
    try:
        run('IO_CPU_gates',model_python,ROOT/'src/check_io.py')
        run('IO_generation_parity_gate',model_python,ROOT/'src/check_io_model.py')
        for dataset in manifest['datasets']:
            if dataset in ('humaneval','math500'):continue
            # Actual counter probe, not a configuration-only assumption. Failure
            # is recorded and never blocks task generation or official grading.
            for policy in POLICIES:
                run(f'probe_DRAM_{dataset}_{policy}',model_python,ROOT/'src/check_hardware.py',soft=True)
                run(f'generate_{dataset}_{policy}',model_python,ROOT/'src/generate_io.py','--dataset',dataset,'--policy',policy)
                run(f'score_{dataset}_{policy}',score_python,ROOT/'src/score.py','--dataset',dataset,'--policy',policy)
                run(f'quality_report_{dataset}_{policy}',score_python,ROOT/'src/report.py')
                run(f'traffic_report_{dataset}_{policy}',model_python,ROOT/'src/report_io.py')
        assert json.loads((ROOT/'results/complete.json').read_text())['official_task_scores_complete']
        phase('completed',official_quality_complete=True,traffic_attempts_complete=True,physical_measurement_coverage_report=str(ROOT/'results/traffic_summary.json'))
        atomic(ROOT/'background_exit.json',dict(exit_code=0,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
    except BaseException:
        error=traceback.format_exc();phase('failed',traceback=error)
        atomic(ROOT/'background_exit.json',dict(exit_code=1,traceback=error));print(error,flush=True);raise


if __name__=='__main__':main()
