"""Administrator-launched hardware collector; never executes job-supplied code.

Jobs can only name data fixtures inside this run and warm/evicted conditions.
The trusted profiler command and sources are fixed at service startup. Inference
and official task scoring remain under the original unprivileged quality worker.
"""
import argparse,subprocess,traceback,os,time,threading
from common import *
from traffic import execute_fixture


def main():
    if os.geteuid()!=0:raise PermissionError('Start this single collector using an administrator terminal; no password is stored')
    owner=(ROOT/'config.json').stat();uid,gid=owner.st_uid,owner.st_gid
    jobs=ROOT/'collector_jobs';jobs.mkdir(exist_ok=True);os.chown(jobs,uid,gid)
    sources={p:sha(p) for p in (ROOT/'src').glob('*.py')}
    def give(path):
        if path.exists():os.chown(path,uid,gid)
    def heartbeat():
        while True:
            atomic(ROOT/'collector_status.json',dict(pid=os.getpid(),heartbeat_epoch=time.time(),administrator_collector=True));give(ROOT/'collector_status.json');time.sleep(2)
    threading.Thread(target=heartbeat,daemon=True).start()
    while True:
        for job in sorted(jobs.glob('*.json')):
            if job.name.endswith(('.done.json','.error.json')):continue
            done=job.with_suffix('.done.json');error=job.with_suffix('.error.json')
            if done.exists() or error.exists():continue
            data=json.loads(job.read_text());fixture=Path(data['fixture']).resolve();base=Path(data['base']).resolve()
            try:
                assert sources=={p:sha(p) for p in sources},'Trusted collector source changed'
                assert fixture.is_relative_to((ROOT/'fixtures').resolve()) and fixture.is_file() and fixture.stat().st_size<600<<20
                assert base.is_relative_to((ROOT/'profiles').resolve())
                assert data['condition'] in ('warm','evicted')
                result=execute_fixture(fixture,data['condition'],base)
                atomic(done,dict(complete=True,receipt=str(base.with_suffix('.json'))));give(done)
            except BaseException:
                atomic(error,dict(error=traceback.format_exc()));give(error)
            for p in base.parent.glob(base.name+'*'):give(p)
            give(base.parent)
        time.sleep(1)


if __name__=='__main__':main()
