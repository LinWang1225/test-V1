import subprocess,sys,json,time,shutil,traceback,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
try:
    p=subprocess.Popen([sys.executable,'-u',str(root/'src/run_e.py')],cwd=root.parents[2])
    (root/'background.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),worker_pid=p.pid,started=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))
    rc=p.wait()
    if rc==0:
        shutil.copy2(root/'results/full/E对比分析.md',root/'E对比分析.md')
        with (root.parent/'结果分析.md').open('a') as f:
            f.write('\n\n---\n\n'+(root/'E对比分析.md').read_text()+'\n\n扩展源码与逐请求结果位于 E_extension/；原A-D部分未重跑。\n')
    else:
        state=json.loads((root/'status.json').read_text())
        (root/'E对比分析.md').write_text('# E补充实验失败\n\n未完成，不能据不完整结果得出结论。\n\n```json\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n```\n')
    (root/'background_exit.json').write_text(json.dumps(dict(exit_code=rc,finished=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))
except BaseException:
    (root/'supervisor_failure.txt').write_text(traceback.format_exc());raise
