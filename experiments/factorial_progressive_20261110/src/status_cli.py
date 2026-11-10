from pathlib import Path
import json,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
out={'experiment':str(ROOT),'time':time.time(),'formal_generation_started':(ROOT/'protocol_lock.json').exists(),'jobs':{}}
for name in ['calibration_basis','basis_fit','calibration_fit','codebook_fit','validation','formal','scoring']:
 pointer=ROOT/'jobs'/f'{name}.json'
 if not pointer.exists():continue
 d=Path(json.loads(pointer.read_text())['job_dir']);s=d/'state.json'
 if not s.exists():continue
 v=json.loads(s.read_text());out['jobs'][name]={k:v.get(k) for k in ['status','exit_code','elapsed_seconds','child_pid','updated_at']};out['jobs'][name]['job_dir']=str(d)
 if v['status']=='failed':out['jobs'][name]['error_tail']=(d/'output.log').read_text(errors='replace').splitlines()[-8:]
for name in ['status.json','jobs/formal_progress.json','jobs/scoring_progress.json','jobs/codebook_fit_progress.json','jobs/validation_progress.json']:
 p=ROOT/name
 if p.exists():out[name]=json.loads(p.read_text())
out['finished_generation_requests']=sum(1 for _ in (ROOT/'results').glob('*/*.json'))
out['finished_scoring_requests']=sum(1 for _ in (ROOT/'scores').glob('*/*.json'))
out['gate_passed']=(ROOT/'checks/ready.json').exists()
out['codebook_files']=sum(1 for _ in (ROOT/'codebooks').glob('*/layer*.npz'))
a=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True);out['gpu']=a.stdout.strip()
print(json.dumps(out,ensure_ascii=False,indent=2))
