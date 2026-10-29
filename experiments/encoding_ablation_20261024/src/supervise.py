"""Detached supervisor: final analysis is explicit even on failed experiments."""
import subprocess,sys,json,time,shutil,traceback,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
started=time.strftime('%Y-%m-%d %H:%M:%S')
try:
    child=subprocess.Popen([sys.executable,'-u',str(root/'src/run.py')],cwd=root.parents[1])
    (root/'background.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),worker_pid=child.pid,started=started),indent=2))
    code=child.wait()
    if code==0:
        shutil.copy2(root/'results/full/结果分析.md',root/'结果分析.md')
        with (root/'结果分析.md').open('a') as f:f.write('\n完整源码、协议、原始指标均位于本文件同级目录；详细数据见 results/full/。\n')
    else:
        state=json.loads((root/'status.json').read_text())
        (root/'结果分析.md').write_text('# 独立渐进编码消融：运行失败\n\n本轮未完成，禁止依据不完整样本给出方法优劣结论。\n\n```json\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n```\n\n完整日志：logs/background.log；协议：PROTOCOL.md；修改说明：CHANGES.md。\n')
    (root/'background_exit.json').write_text(json.dumps(dict(exit_code=code,finished=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))
except BaseException:
    (root/'supervisor_failure.txt').write_text(traceback.format_exc());raise
