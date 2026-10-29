import subprocess,sys,json,time,os,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def state(stage,**kw):
 p=R/'experiment_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,supervisor_pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw),ensure_ascii=False,indent=2));tmp.replace(p)
def run(stage,args):
 state(stage);p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=R.parents[2]);state(stage,worker_pid=p.pid)
 rc=p.wait()
 if rc:raise RuntimeError(f'{stage} failed with exit {rc}')
try:
 run('KVarN_no_FP16_90_test',[R/'src/run_native_nt.py'])
 state('waiting_for_suite_freeze')
 deadline=time.time()+7200
 while not (R/'suite.ready').exists():
  if time.time()>deadline:raise RuntimeError('Suite source readiness deadline expired')
  time.sleep(10)
 run('capture_calibration_score_weights',[R/'src/suite.py','--stage','calibrate'])
 run('train_all_frozen_candidates',[R/'src/suite.py','--stage','train'])
 run('evaluate_90_test_no_validation',[R/'src/suite.py','--stage','evaluate'])
 run('final_analysis_and_figures',[R/'src/analyze.py'])
 state('completed');(R/'background_exit.json').write_text(json.dumps(dict(exit_code=0,finished=time.strftime('%Y-%m-%d %H:%M:%S'))))
except BaseException:
 state('failed',traceback=traceback.format_exc());(R/'results_summary.md').write_text('# 补充实验未完成\n\n```\n'+traceback.format_exc()+'\n```\n');raise
