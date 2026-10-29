"""Freeze and report the completed four-method, first-pass comparison."""
import csv,json,sys,math
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parent))
from operator_final_report import family
ROOT=Path(__file__).resolve().parents[1]
RUNS={'fp16':'operator_profile_20261019_210036','kvarn':'operator_kvarn_firstpass_20261020_153745','turboquant':'operator_turboquant_firstpass_20261020_queue','fp8':'operator_fp8_firstpass_20261020_queue'}
def main():
 states={}; totals={}; raw=defaultdict(float); points=[]
 for m,r in RUNS.items():
  states[m]={}
  for p in sorted((ROOT/'results'/r/'points'/m).glob('*/state.json')):
   s=json.loads(p.read_text())
   if s['status']=='complete' and s['point']['repeat']==1:states[m][s['point']['id']]=s
 ids=set(states['fp16'])
 assert len(ids)==89
 for m in RUNS:
  assert set(states[m])==ids,(m,len(states[m]))
  for i in ids:assert states[m][i]['point']['case_ids']==states['fp16'][i]['point']['case_ids']
  a={'total':0.,'busy':0.,'tokens':0,'samples':0,'phase_min':1.,'datasets':defaultdict(float),'cats':defaultdict(float),'calls':defaultdict(int),'wall':0.,'latencies':[],'ttft':[],'tpot':[]}
  for i in sorted(ids):
   s=states[m][i]; trace=Path(s['attempt'])/'traces';o=json.loads((trace/'kernel_summary.json').read_text());q=json.loads((Path(s['attempt'])/'measured/request_metrics.json').read_text());rows=q['rows']
   assert len(rows)==len(s['point']['case_ids'])
   assert all(x['success']==1 and x['completion_tokens']==1024 for x in rows)
   assert s['generated_tokens']==len(rows)*1024
   assert math.isclose(sum(c['total_gpu_ms'] for c in o['categories']),o['cumulative_gpu_event_ms'],rel_tol=1e-9)
   a['total']+=o['cumulative_gpu_event_ms']/1000;a['busy']+=o['gpu_busy_union_ms']/1000;a['tokens']+=s['generated_tokens'];a['samples']+=len(rows);a['phase_min']=min(a['phase_min'],o['phase_matched_fraction']);a['datasets'][s['point']['dataset']]+=o['cumulative_gpu_event_ms']/1000;a['wall']+=q['wall_s']
   for k,col in [('latencies','latency'),('ttft','first_chunk_latency'),('tpot','time_per_output_token')]:a[k].extend(x[col] for x in rows)
   for c in o['categories']:
    key=c['phase']+' / '+family(c['category']);a['cats'][key]+=c['total_gpu_ms']/1000;a['calls'][key]+=c['calls']
   with (trace/'kernels.csv').open() as f:
    for row in csv.DictReader(f):
     raw[(m,row['category'],row['kernel'])]+=float(row['total_gpu_ms'])/1000
     if row['category']=='other_unclassified' and 'flashinfer::' in row['kernel'] and any(n in row['kernel'] for n in ('BatchPrefillWithPagedKVCacheKernel','PersistentVariableLengthMergeStatesKernel')):
      old=row['phase']+' / other_unclassified';new=row['phase']+' / attention_read_dequant_compute_combined';v=float(row['total_gpu_ms'])/1000
      a['cats'][old]-=v;a['cats'][new]+=v
      a['calls'][old]-=int(row['calls']);a['calls'][new]+=int(row['calls'])
   points.append({'method':m,'point':i,'samples':len(rows),'gpu_s':o['cumulative_gpu_event_ms']/1000,'attempt':s['attempt']})
  assert a['samples']==672 and a['tokens']==688128
  totals[m]=a
 out=ROOT/'results/experiment_analysis_20261022';out.mkdir(exist_ok=True)
 serial={m:{k:v for k,v in a.items() if k not in ('latencies','ttft','tpot')} for m,a in totals.items()}
 (out/'validated_results.json').write_text(json.dumps({'runs':RUNS,'totals':serial,'points':points},ensure_ascii=False,indent=2))
 with (out/'raw_kernel_totals.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['method','category','kernel','gpu_s']);w.writerows((*k,v) for k,v in sorted(raw.items(),key=lambda x:-x[1]))
 print(json.dumps(serial,ensure_ascii=False,indent=2))
 return out,totals,raw
if __name__=='__main__':
 out,totals,raw=main()
 from render_completed_report import render
 render(out,totals,raw)
