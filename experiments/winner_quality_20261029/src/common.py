"""Independent full-generation quality experiment utilities."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import json,hashlib,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
OLD=Path('/home/wanglin/kvbench-3090-cu126')
MODEL=OLD/'cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'
BOOKS=ROOT/'codebooks'
POLICIES=('full4','adaptive005','adaptive015','adaptive030')


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for a in iter(lambda:f.read(4<<20),b''):h.update(a)
    return h.hexdigest()


def status(stage,**kw):
    atomic(ROOT/'worker_status.json',dict(stage=stage,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))
    print(stage,kw,flush=True)


def jsonlines(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def write_jsonlines(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    tmp.replace(path)


def result_path(policy,dataset,index,pilot=False):
    return ROOT/'results'/('pilot' if pilot else 'full')/policy/dataset/f'{index:05d}.json'
