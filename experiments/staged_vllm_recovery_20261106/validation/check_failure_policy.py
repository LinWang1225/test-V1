"""Fault-injection integration test; fake children never run a model."""
import os,sys,json,tempfile,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_queue,failure_policy
from common import atomic
assert os.geteuid()==0
base=pathlib.Path(tempfile.mkdtemp(prefix='kvarn-failure-policy-'))
project=base/'project';fixture=project/'experiments'/'fixture'
for relative in ('src','results','analysis','logs','failures','checkpoints','validation'):(fixture/relative).mkdir(parents=True,exist_ok=True)
(project/'locks').mkdir(parents=True)
for folder in ('.venv','.venv-eval'):
 (project/folder/'bin').mkdir(parents=True);(project/folder/'bin/python').symlink_to(sys.executable)
atomic(fixture/'config.json',{})
atomic(fixture/'validation/live_checks.json',{'state':'passed'})
for name in ('failure_policy_check.json','capacity_fp16.json','capacity_Jacobian_Tree.json'):
 atomic(fixture/'validation'/name,{'passed':True})
(fixture/'src/codec_worker.py').write_text(r"""
import json,os,sys,pathlib
r=pathlib.Path(__file__).resolve().parents[1];codec=sys.argv[-1];identity='fp16' if codec=='fp16' else codec+'__fixed4'
checkpoint=r/'checkpoints'/(identity+'.json');counter=r/(codec+'.calls');calls=int(counter.read_text())+1 if counter.exists() else 1;counter.write_text(str(calls))
if codec=='Jacobian_Tree':
 checkpoint.write_text(json.dumps({'stage':'model_initializing'}));raise SystemExit(2)
if codec=='fp16' and calls==1:
 row=dict(index=0,case_id='fixture/0',dataset='fixture',input_ids_sha256='fixed',sampling_seed=123)
 base=r/'logs'/'partial';base.with_suffix('.csv').write_text('partial counters')
 checkpoint.write_text(json.dumps(dict(stage='request_active',row=row,profile_base=str(base),execution_id='fixture-execution')))
 print('Injected process crash at first request',flush=True);os._exit(139)
if codec=='fp16':assert (r/'failures/fp16/000000.json').exists()
out=r/'results'/identity;out.mkdir(exist_ok=True)
(out/('000001.json' if codec=='fp16' else '000000.json')).write_text('{}')
checkpoint.write_text(json.dumps({'stage':'codec_finished'}))
""")
for script in ('score.py','report.py'):(fixture/'src'/script).write_text('print("fixture grading callback")\n')
run_queue.ROOT=fixture;failure_policy.ROOT=fixture
run_queue.QUEUE=('fp16','Jacobian_Tree','Data_Tree')
run_queue.freeze=lambda:{'fingerprint':'fixture'}
run_queue.main()
assert (fixture/'fp16.calls').read_text()=='2'
assert len(list((fixture/'failures/fp16').glob('[0-9]*.json')))==1
assert not (fixture/'results/fp16/000000.json').exists()
assert (fixture/'results/fp16/000001.json').exists()
assert (fixture/'results/Data_Tree__fixed4/000000.json').exists()
assert (fixture/'failures/Jacobian_Tree__fixed4.run_error.json').exists()
state=json.loads((fixture/'stage_status.json').read_text());assert state['stage']=='stage1_finished_with_failures'
atomic(ROOT/'validation/failure_policy_check.json',dict(passed=True,failed_case_recorded=True,failed_case_not_fabricated_as_success=True,worker_restarted=True,next_case_completed=True,next_codec_after_startup_error_completed=True,fixture=str(fixture)))
print('Fault injection passed: crash recorded, skipped, engine child restarted; later case and codec continued')
