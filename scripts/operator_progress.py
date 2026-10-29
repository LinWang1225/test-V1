"""Read progress without loading model or touching the experiment process."""
import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path((ROOT / 'results/operator_profile.latest.txt').read_text().strip())
state = json.loads((run / 'status.json').read_text())
heartbeat = json.loads((run / 'heartbeat.json').read_text()) if (run / 'heartbeat.json').exists() else {}
plan = json.loads((run / 'plan.json').read_text())
done = []
failed = []
for p in (run / 'points').glob('*/*/state.json'):
    s = json.loads(p.read_text())
    if s['point']['repeat'] == 0:
        continue
    if s['status'] == 'complete':
        done.append(s)
    elif s['status'] == 'failed':
        failed.append(s)
total = plan['points_per_method'] * len(plan['methods'])
try:
    os.kill(state['pid'], 0)
    alive = True
except (ProcessLookupError, PermissionError):
    alive = False
print('目录:', run)
print('状态:', state['status'], '| 控制器 PID:', state['pid'], '| 进程存在:', alive)
print('方法:', state.get('method'), '| 数据集:', state.get('dataset'), '| 批次:', state.get('point'))
print(f'正式采集: {len(done)}/{total} ({len(done) / total:.1%})')
print('已验证的正式生成请求:', sum(len(s['point']['case_ids']) for s in done), '/', plan['total_profiled_requests'])
if heartbeat:
    print('心跳距今:', round(time.time() - heartbeat['heartbeat_unix'], 1), '秒')
    print('磁盘剩余 GiB:', round(heartbeat['disk_free_gib'], 1), '| GPU:', heartbeat.get('gpu_csv'))
if state.get('error'):
    print('错误:', state['error'])
if state.get('attempt'):
    print('当前批次日志:', state['attempt'])
print('诊断:', run / 'failure.log')
