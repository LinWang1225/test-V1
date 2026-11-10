import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
from pathlib import Path
import sys,json,time,hashlib,importlib.util,collections,math
ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT.parents[1]/'experiments/full_quality_20261027'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,data):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def progress(stage,**detail):
 data={'stage':stage,'time':time.time(),**detail};atomic(ROOT/'jobs/scoring_progress.json',data);print(json.dumps(data),flush=True)
def load_scorer():
 pin=LEGACY/'vendor/official_sources.json';expected=json.loads(pin.read_text())
 for relative,value in expected.items():
  digest=value if isinstance(value,str) else value['sha256']
  assert sha(LEGACY/relative)==digest,relative
 atomic(ROOT/'checks/scorer_pins.json',{'manifest_sha256':sha(pin),'files':len(expected),'all_verified':True})
 sys.path.insert(0,str(LEGACY/'vendor'));sys.path.insert(0,str(LEGACY/'src'))
 spec=importlib.util.spec_from_file_location('official_frozen_score',LEGACY/'src/score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=LEGACY
 return module

def final_text(g):
 if 'text' in g:return g['text']
 if 'final_text' in g:return g['final_text']
 for key in ['raw_text','full_text','decoded_text']:
  if key in g:
   full=g[key];return full.rsplit('</think>',1)[1].strip() if '</think>' in full else ''
 raise KeyError('generation result lacks final answer field: '+str(list(g)))
def score_one(path,row,m):
 rec=json.loads(path.read_text());g=rec['generation'];text=final_text(g);dataset=row['dataset']
 if dataset=='humaneval':ret=m.human_score(text,row['source_row'])
 elif dataset=='math500':ret=m.math_score(text,row['source_row'])
 else:ret=m.longbench_score(text,row['source_row'],dataset)
 assert ret['grading_status']=='graded',ret
 return {'run_id':rec['configuration']['run_id'],'case_id':row['case_id'],'dataset':dataset,'result':ret,'source':str(path.relative_to(ROOT)),
  'source_sha256':sha(path),'protocol':rec['protocol'],'output_tokens':g.get('output_tokens',g.get('generated_tokens',g.get('generated_token_count'))),
  'termination':g.get('finish_reason'),'eos':bool(g.get('generated_ids') and g['generated_ids'][-1] in json.loads((ROOT/'config.json').read_text())['eos_token_ids']),'context_limit_reached':not g.get('natural_completion',False) and g['input_tokens']+g['generated_tokens']>=g['native_context_length'],'thinking_complete':g.get('thinking_complete'),'thinking_tokens':g.get('thinking_tokens'),'final_tokens':g.get('final_tokens'),'missing_thinking_close':not g.get('thinking_complete',False),'finished_time':time.time()}
def aggregate(scores,counts):
 byrun=collections.defaultdict(list)
 for x in scores:byrun[x['run_id']].append(x)
 out={}
 for run,items in byrun.items():
  by=collections.defaultdict(list)
  for x in items:by[x['dataset']].append(x)
  ds={}
  for dataset,values in by.items():
   ss=[float(x['result']['score']) for x in values];ds[dataset]={'mean_score':sum(ss)/len(ss),'graded':len(ss),'expected':counts[dataset],'complete':len(ss)==counts[dataset],'mean_output_tokens':sum(x['output_tokens'] for x in values)/len(values),'eos_rate':sum(x['eos'] for x in values)/len(values),'native_context_limit_rate':sum(x['context_limit_reached'] for x in values)/len(values),'missing_thinking_close_rate':sum(x['missing_thinking_close'] for x in values)/len(values)}
  lb=[x for k,x in ds.items() if k not in ['humaneval','math500']]
  out[run]={'graded':len(items),'expected':90,'complete':len(items)==90,'datasets_and_LongBench_tasks':ds,
    'LongBench_macro_over_observed_tasks':sum(x['mean_score'] for x in lb)/len(lb) if lb else None,'LongBench_observed_tasks':len(lb),'LongBench_expected_tasks':21,
    'quality_scope':'exploratory fixed paired sample; interim means show only completed requests'}
 return out

def main():
 m=load_scorer();rows=[json.loads(x) for x in (ROOT/'inputs/scoring_cases.jsonl').read_text().splitlines()];lookup={x['case_id']:x for x in rows};counts=collections.Counter(x['dataset'] for x in rows)
 scores=[];seen=set()
 for p in (ROOT/'scores').glob('*/*.json'):
  value=json.loads(p.read_text());source=ROOT/value['source'];assert source.exists() and sha(source)==value['source_sha256'];scores.append(value);seen.add(str(source))
 while True:
  changed=False
  for p in sorted((ROOT/'results').glob('*/*.json')):
   if str(p) in seen:continue
   rec=json.loads(p.read_text());row=lookup[rec['case_id']];assert rec['input_ids_sha256']==row['input_ids_sha256']
   value=score_one(p,row,m);atomic(ROOT/'scores'/p.parent.name/p.name,value);scores.append(value);seen.add(str(p));changed=True
   progress('scored',run_id=value['run_id'],case_id=value['case_id'],scored_requests=len(scores))
  if changed:atomic(ROOT/'analysis/task_scores.json',{'updated':time.time(),'runs':aggregate(scores,counts),'official_pins':sha(LEGACY/'vendor/official_sources.json')})
  if (ROOT/'jobs/generation_complete.json').exists():
   assert len(seen)==457*90;progress('completed',scored_requests=len(scores));return
  pointer=ROOT/'jobs/formal.json'
  if pointer.exists():
   s=Path(json.loads(pointer.read_text())['job_dir'])/'state.json'
   if s.exists() and json.loads(s.read_text()).get('status')=='failed':raise RuntimeError('formal generator failed; scored completed outputs only')
  progress('watching',scored_requests=len(scores));time.sleep(15)
if __name__=='__main__':main()
