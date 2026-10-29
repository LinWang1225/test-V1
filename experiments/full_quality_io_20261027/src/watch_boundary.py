"""Adopt the current worker; finish MATH unchanged, then switch at dataset edge."""
import sys,subprocess,traceback,signal
from common import *
LEGACY=ROOT.parent/'full_quality_20261027'


def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.rsplit(')',1)[1].strip().split()[0]!='Z'
    except FileNotFoundError:return False


def main():
    adopted=json.loads((ROOT/'adoption.json').read_text());old_sources=json.loads((LEGACY/'provenance.json').read_text())['source_input_hashes']
    def verify():assert all(sha(p)==h for p,h in old_sources.items()),'Legacy frozen source changed'
    verify()
    # Ack only after all critical paths and the exact legacy source are checked.
    atomic(ROOT/'adoption_ack.json',dict(pid=os.getpid(),accepted=True,adopted_worker=adopted['worker_pid']))
    def phase(name,**kw):
        data=dict(stage=name,supervisor_pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
        atomic(ROOT/'handoff_status.json',data)
        atomic(LEGACY/'pipeline_status.json',dict(data,new_version_root=str(ROOT),version='legacy current dataset, boundary handoff armed'))
    try:
        phase('finish_existing_math_worker',worker_pid=adopted['worker_pid'],adopted_stage=adopted['stage'])
        while alive(adopted['worker_pid']):
            phase('finish_existing_math_worker',worker_pid=adopted['worker_pid'],adopted_stage=adopted['stage']);time.sleep(10)
        # Idempotent legacy scripts skip all already complete requests and scores.
        # No generation parameters, model, prompt or scorer are changed for MATH.
        def run(name,python,script,*args):
            verify();print('\nSTART',name,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
            child=subprocess.Popen([str(python),'-u',str(script),*args],cwd=PROJECT)
            while True:
                phase(name,worker_pid=child.pid)
                try:rc=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:pass
            if rc:raise RuntimeError(name+f' exit {rc}')
        for policy in POLICIES:
            run(f'legacy_generate_math500_{policy}',PROJECT/'.venv/bin/python',LEGACY/'src/generate.py','--dataset','math500','--policy',policy)
            run(f'legacy_score_math500_{policy}',PROJECT/'.venv-eval/bin/python',LEGACY/'src/score.py','--dataset','math500','--policy',policy)
            run(f'legacy_report_math500_{policy}',PROJECT/'.venv-eval/bin/python',LEGACY/'src/report.py')
        atomic(LEGACY/'dataset_boundary.json',dict(complete_through=['humaneval','math500'],next_dataset='narrativeqa',new_run=str(ROOT),old_worker_completed_without_interruption=True))
        old_report=LEGACY/'results/结果分析.md'
        old_report.write_text(old_report.read_text()+'\n\n## 后续 LongBench 结果入口\n\nHumanEval 与 MATH-500 五策略已按旧版本完成。后续 LongBench 使用带读取量账本的新版本，见[新版本完整质量报告](../../full_quality_io_20261027/results/结果分析.md)及[逻辑与物理流量对比](../../full_quality_io_20261027/results/读取量对比.md)。硬件采集失败不阻断质量评测。\n')
        atomic(LEGACY/'background_exit.json',dict(exit_code=0,full_run_completed=False,completion_scope='HumanEval and MATH-500 complete, remainder handed off',next_run=str(ROOT)))
        phase('switch_to_IO_LongBench',old_datasets_completed=['humaneval','math500'])
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'src/supervise_io.py')],cwd=PROJECT)
        atomic(ROOT/'io_background.json',dict(pid=child.pid,started=time.strftime('%Y-%m-%d %H:%M:%S')))
        rc=child.wait()
        if rc:raise RuntimeError(f'New LongBench supervisor exit {rc}')
        phase('completed_all_datasets',new_run=str(ROOT))
    except BaseException:
        error=traceback.format_exc();phase('failed',traceback=error);print(error,flush=True);raise


if __name__=='__main__':main()
