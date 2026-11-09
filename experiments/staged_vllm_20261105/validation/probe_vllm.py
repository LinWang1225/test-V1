import os,sys,uuid,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
from vllm_runner import VllmRunner
from common import rows,atomic

p=argparse.ArgumentParser()
p.add_argument('codec');p.add_argument('--eager',action='store_true')
p.add_argument('--no-counters',action='store_true');p.add_argument('--case',type=int,default=0)
p.add_argument('--tag')
a=p.parse_args()
runner=VllmRunner(a.codec,eager=a.eager)
row=next(r for r in rows(ROOT/'inputs/cases.jsonl') if r['index']==a.case)
name=a.tag or ('live_'+a.codec)
execution_id=uuid.uuid4().hex
try:
    if a.case != 0:
        # Check that a second request reuses the engine with fresh per-request
        # RNG, cache state and CUPTI session. Both requests generate freely.
        first=next(rows(ROOT/'inputs/cases.jsonl'))
        prior=runner.generate(first,ROOT/'validation'/(name+'_prior'),uuid.uuid4().hex,with_counters=not a.no_counters)
        atomic(ROOT/'validation'/(name+'_prior.json'),prior)
        if not a.no_counters:
            plain=runner.generate(first,ROOT/'validation'/(name+'_counter_control'),uuid.uuid4().hex,with_counters=False)
            atomic(ROOT/'validation'/(name+'_counter_control.json'),plain)
            assert prior['generated_ids']==plain['generated_ids'], 'Quantized same-engine counter control changed tokens'
    result=runner.generate(row,ROOT/'validation'/name,execution_id,with_counters=not a.no_counters)
    result['persistent_engine_reset_validated']=a.case!=0
    result['quantized_counter_control_validated']=a.case!=0 and not a.no_counters
    atomic(ROOT/'validation'/(name+'.json'),result)
    print('VLLM_RESULT',result['backend'],result['generated_tokens'],result['elapsed_seconds'],result['finish_reason'],result['decode_graph_modes'],flush=True)
finally:
    runner.close()
