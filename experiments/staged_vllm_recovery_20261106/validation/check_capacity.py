"""Native context boundary smoke test; no benchmark result is synthesized."""
import json,sys,uuid,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import rows,atomic
from vllm_runner import VllmRunner
codec=sys.argv[1];row=dict(next(rows(ROOT/'inputs/cases.jsonl')))
runner=VllmRunner(codec)
try:
 original=row['input_ids'];filler=runner.tokenizer.encode(' a',add_special_tokens=False)[0]
 ids=original[:6]+[filler]*(40958-len(original))+original[6:]
 row.update(index=-1,case_id='validation/native_context_boundary',dataset='validation',input_ids=ids,input_tokens=len(ids),input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
 name='capacity_'+codec+'_'+uuid.uuid4().hex[:8]
 result=runner.generate(row,ROOT/'validation'/name,uuid.uuid4().hex,with_counters=True)
 assert result['execution_environment']['usable_kv_tokens']>=40960
 assert result['input_tokens']+result['generated_tokens']==40960
 assert result['physical_traffic']['stages']['prefill']['units']==40958
 assert result['physical_traffic']['stages']['decode']['units']==1
 assert result['finish_reason']=='native_context_limit'
 atomic(ROOT/'validation'/('capacity_'+codec+'.json'),dict(passed=True,request_native_boundary=40960,usable_cache_tokens=runner.usable_kv_tokens,result=result))
 print('Native full-context capacity and hardware coverage passed',codec,flush=True)
finally:runner.close()
