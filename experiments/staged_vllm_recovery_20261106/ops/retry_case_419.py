"""Operator-requested retry using the unchanged frozen worker and sampler."""
import json,os,pwd,subprocess,sys,time,uuid,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import atomic,sha
from provenance import freeze

UNIT='kvarn-vllm-recovery-20261106.service'
INDEX=419
NAME=f'{INDEX:06d}.json'
owner=(ROOT/'config.json').stat()
request=ROOT/'retry_requests'/('fp16_'+str(INDEX)+'_'+uuid.uuid4().hex+'.json')
state=dict(run='fp16',index=INDEX,user_requested_retry=True,operator_script_sha256=sha(__file__),
           run_fingerprint=freeze()['fingerprint'],requested_at=time.strftime('%Y-%m-%d %H:%M:%S'))

def update(**value):
    state.update(value);state['updated']=time.strftime('%Y-%m-%d %H:%M:%S')
    atomic(request,state);os.chown(request,owner.st_uid,owner.st_gid)

try:
    assert os.geteuid()==0
    output=ROOT/'results/fp16'/NAME
    failure=ROOT/'failures/fp16'/NAME
    assert not output.exists() and failure.exists()
    checkpoint=ROOT/'checkpoints/fp16.json'
    current=json.loads(checkpoint.read_text())
    wait_index=current.get('row',{}).get('index') if current.get('stage')=='request_active' else None
    update(status='waiting_for_current_case',waiting_for_index=wait_index)
    if wait_index is not None:
        while not (ROOT/'results/fp16'/f'{wait_index:06d}.json').exists() and not (ROOT/'failures/fp16'/f'{wait_index:06d}.json').exists():
            if subprocess.run(['systemctl','is-active','--quiet',UNIT]).returncode:break
            time.sleep(.2)
    subprocess.run(['systemctl','stop',UNIT],check=True)
    # Archive the failed attempt, retaining its exact bytes and all its evidence.
    archive=ROOT/'failure_history/fp16'/('000419_'+str(time.time_ns())+'.json')
    archive.parent.mkdir(parents=True,exist_ok=True)
    original_hash=sha(failure)
    failure.rename(archive)
    assert sha(archive)==original_hash
    os.chown(archive,owner.st_uid,owner.st_gid)
    update(status='retry_enqueued',failed_attempt_archive=str(archive),failed_attempt_sha256=original_hash)
    # The existing worker now sees this input as pending and generates freely
    # from the original prompt/seed, collecting counters during that execution.
    subprocess.run(['systemctl','start',UNIT],check=True)
    update(status='retry_running')
    while not output.exists() and not failure.exists():
        if subprocess.run(['systemctl','is-active','--quiet',UNIT]).returncode:
            raise RuntimeError('Main queue stopped before the retry produced a result')
        time.sleep(1)
    if failure.exists():
        update(status='retry_failed_and_queue_continues',new_failure=str(failure))
    else:
        result=json.loads(output.read_text())
        assert result['run_fingerprint']==state['run_fingerprint']
        assert result['execution_id']==result['physical_traffic']['execution_id']
        update(status='retry_succeeded',result=str(output),result_sha256=sha(output),
               execution_id=result['execution_id'],generated_tokens=result['generated_tokens'],
               finish_reason=result['finish_reason'],elapsed_seconds=result['elapsed_seconds'])
        account=pwd.getpwuid(owner.st_uid).pw_name
        project=ROOT.parents[1]
        for executable,script,args in ((project/'.venv-eval/bin/python','score.py',['--run','fp16']),
                                        (project/'.venv/bin/python','report.py',[])):
            subprocess.run(['runuser','-u',account,'--',str(executable),str(ROOT/'src'/script),*args],check=True)
except BaseException:
    update(status='retry_controller_error',error=traceback.format_exc())
    raise
