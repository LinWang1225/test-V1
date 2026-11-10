import collections,hashlib,json,os,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
CONFIG=json.loads((ROOT/'config.json').read_text())
MODEL=pathlib.Path(CONFIG['model']['path'])
GENERATION=CONFIG['generation']
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
def rows(name='generator'):
    return [json.loads(line) for line in (ROOT/'inputs'/(name+'_cases.jsonl')).read_text().splitlines() if line]
def progress(name,stage,**details):
    record=dict(stage=stage,pid=os.getpid(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**details)
    atomic(ROOT/'jobs'/(name+'_progress.json'),record)
    print(json.dumps(record,ensure_ascii=False),flush=True)
def split_response(tokenizer,ids):
    close=tokenizer.convert_tokens_to_ids('</think>')
    locations=[i for i,x in enumerate(ids) if x==close]
    raw=tokenizer.decode(ids,skip_special_tokens=False)
    if not locations:
        return dict(text='',thinking_text=tokenizer.decode(ids,skip_special_tokens=True),raw_text=raw,thinking_complete=False,thinking_tokens=len(ids),final_tokens=0)
    boundary=locations[-1]
    return dict(text=tokenizer.decode(ids[boundary+1:],skip_special_tokens=True),thinking_text=tokenizer.decode(ids[:boundary],skip_special_tokens=True),raw_text=raw,thinking_complete=True,thinking_tokens=boundary+1,final_tokens=len(ids)-boundary-1)
