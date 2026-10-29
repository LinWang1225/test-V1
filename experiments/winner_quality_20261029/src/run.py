from common import *
import subprocess,sys,traceback,fcntl

def main():
 lock=(ROOT/'logs/run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 paths=sorted((ROOT/'src').glob('*.py'))+[ROOT/'PROTOCOL.md']
 signature={str(p):sha(p) for p in paths}
 provenance=ROOT/'provenance.json'
 if provenance.exists():assert json.loads(provenance.read_text())['source_hashes']==signature
 else:atomic(provenance,dict(source_hashes=signature))
 def run(stage,python,script,*args):
  assert all(sha(p)==h for p,h in signature.items())
  child=subprocess.Popen([str(PROJECT/python),'-u',str(ROOT/'src'/script),*args],cwd=PROJECT)
  atomic(ROOT/'pipeline_status.json',dict(stage=stage,supervisor_pid=os.getpid(),worker_pid=child.pid,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
  rc=child.wait()
  if rc:raise RuntimeError(f'{stage}: exit {rc}')
 try:
  run('prepare_winner_books','.venv/bin/python','prepare_books.py')
  run('adaptive_CPU_gates','.venv/bin/python','check.py')
  for policy in POLICIES:
   run('generate_'+policy,'.venv/bin/python','generate.py','--dataset','humaneval','--policy',policy)
   run('score_'+policy,'.venv-eval/bin/python','score.py','--dataset','humaneval','--policy',policy)
   run('report_'+policy,'.venv/bin/python','report.py')
  atomic(ROOT/'pipeline_status.json',dict(stage='completed',policies=POLICIES))
 except BaseException:
  atomic(ROOT/'pipeline_status.json',dict(stage='failed',error=traceback.format_exc()));raise
if __name__=='__main__':main()
