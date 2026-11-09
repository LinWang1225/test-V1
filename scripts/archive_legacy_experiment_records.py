"""Archive historical experiment conditions and metric tables without GPU use."""
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LEGACY = Path('/home/wanglin/kvbench-3090-cu126')
OUTPUT = PROJECT / 'results/experiment_record_20261109_legacy'
SKIP = {'cache', '.git', '__pycache__', 'raw_responses', 'captures', 'capture',
        'fixtures', 'runtime', 'runtimes', 'runtime_snapshot', 'private_runtime',
        'servers', 'server', 'logs', 'metrics.samples', 'warmup', 'warmup_export',
        'points', 'source', 'src', 'traces', 'data', 'arrays', 'fixture_arrays',
        'residual_stats', 'capture_arrays', 'node_modules', 'cases', 'inputs',
        'dispatch', 'profiles', 'timelines'}


def table(columns, rows):
    def c(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['|' + '|'.join(map(c, columns)) + '|',
                      '|' + '|'.join(['---'] * len(columns)) + '|'] +
                     ['|' + '|'.join(map(c, r)) + '|' for r in rows]) + '\n\n'


def dump(p, o):
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2) + '\n')


def main():
    OUTPUT.mkdir(exist_ok=True)
    sources, artifact_rows, fields, issues = [], [], [], []
    roots = [LEGACY / 'results' / name for name in
             ['asym_equal_bytes', 'longcontext_v3', 'progressive_online',
              'real', 'residual_diag', 'residual_v1']]
    roots += [LEGACY / name for name in ['kvbench_asym', 'kvbench_residual',
              'kvbench_residual_diag', 'kvbench_longcontext', 'kvbench_progressive_online',
              'kvbench_real_v5', 'kvbench_real_v6', 'kvbench_precision_v8', 'kvbench_v7_patch']]
    for root in roots:
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(('server_', 'runtime_', 'attempt_')))
            for name in sorted(names):
                p = Path(current) / name
                if p.is_symlink() or p.suffix not in {'.json', '.csv', '.md', '.txt'}:
                    continue
                if name.endswith(('.trace.json', '.torch_trace.json')) or name.startswith('trace_'):
                    continue
                if name.endswith(('.log', 'pip-freeze.txt')) or name in {'autonomous_next.source.py'}:
                    continue
                rel = p.relative_to(LEGACY)
                info = dict(source=str(p), relative_source=str(rel), bytes=p.stat().st_size)
                if p.suffix == '.csv':
                    with p.open(encoding='utf-8-sig') as f:
                        header = next(csv.reader(f), [])
                    info['columns'] = header
                    fields.extend(dict(source=str(p), field=k) for k in header)
                if info['bytes'] <= 10_000_000:
                    raw = p.read_bytes()
                    dest = OUTPUT / 'sources' / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
                    info['snapshot'] = str(dest.relative_to(OUTPUT))
                    info['sha256'] = hashlib.sha256(raw).hexdigest()
                    if p.suffix == '.json':
                        try:
                            json.loads(raw)
                        except Exception as e:
                            issues.append(dict(source=str(p), error=str(e)))
                    sources.append(info)
                else:
                    info['snapshot'] = None
                    info['note'] = '超过10MB，仅索引原文件；配置与摘要另行冻结'
                artifact_rows.append(info)
    runs = []
    for family in roots[:6]:
        for run in sorted(family.iterdir()):
            if not run.is_dir():
                continue
            item = dict(family=family.name, run=run.name, source=str(run))
            for name in ['plan.json', 'suite.json', 'settings.json', 'status.json',
                         'continuation.status.json', 'online_plan.json', 'online_status.json',
                         'sweep_plan.json', 'validation_plan.json', 'stop_reason.json']:
                p = run / name
                if p.is_file():
                    try:
                        item[name] = json.loads(p.read_text())
                    except Exception as e:
                        issues.append(dict(source=str(p), error=str(e)))
            item['evidence'] = [s['snapshot'] for s in sources
                                if s['relative_source'].startswith(str(run.relative_to(LEGACY)) + '/')]
            runs.append(item)
    dump(OUTPUT / 'runs.json', runs)
    dump(OUTPUT / 'source_manifest.json', dict(recorded_at=dt.datetime.now().astimezone().isoformat(),
         source_root=str(LEGACY), sources=sources, artifacts=artifact_rows, issues=issues,
         note='Read-only historical archive. No inference, scoring, services or original files changed.'))
    for filename, rows in [('metric_fields.csv', fields), ('metric_artifacts.csv', [
        {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
         for k, v in r.items()} for r in artifact_rows])]:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (OUTPUT / filename).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=columns);w.writeheader();w.writerows(rows)
    body = '# 前置项目历史实验来源索引\n\n'
    body += '本目录补齐当前工作区之前的实验；正式、部分完成、失败启动按各原始状态分别解释。\n\n'
    body += table(['实验族', '运行目录', '实际状态', '冻结条件文件'], [
        [r['family'], r['run'], r.get('status.json', {}).get('status',
          r.get('suite.json', {}).get('phase', r.get('online_status.json', {}).get('status', '见原始状态'))),
         '、'.join('[{}](sources/{}/{}/{})'.format(n, 'results', r['family'] + '/' + r['run'], n)
          for n in ['plan.json','suite.json','status.json','settings.json'] if n in r)] for r in runs])
    body += '## 全部配置和指标文件\n\n'
    body += table(['来源', '冻结副本', '字节数'], [[s['relative_source'],
                  '[打开](' + s['snapshot'] + ')', s['bytes']] for s in sources])
    (OUTPUT / '历史来源索引.md').write_text(body)

    # Correct comparisons within each dataset instead of copying legacy display
    # deltas, some of which used the first dataset as a shared denominator.
    corrected = []
    for p in (OUTPUT / 'sources/results/progressive_online').glob('longbench20_each_*/comparison.csv'):
        with p.open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        baseline = {r['dataset']: r for r in rows if r['arm'] == 'parent'}
        for row in rows:
            if row['dataset'] not in baseline:
                continue
            base = baseline[row['dataset']]
            rr = dict(row)
            rr['run'] = p.parent.name
            rr['recomputed_tpot_improvement_pct_same_dataset'] = 100 * (1 - float(row['tpot_ms_mean']) / float(base['tpot_ms_mean']))
            rr['recomputed_score_difference_pp_same_dataset'] = float(row['task_score_0_100']) - float(base['task_score_0_100'])
            rr['source'] = str(LEGACY / p.relative_to(OUTPUT / 'sources'))
            corrected.append(rr)
    with (OUTPUT / 'progressive_online_same_dataset.csv').open('w', encoding='utf-8-sig', newline='') as f:
        columns = list(dict.fromkeys(k for r in corrected for k in r))
        w = csv.DictWriter(f, fieldnames=columns);w.writeheader();w.writerows(corrected)
    print(json.dumps(dict(sources=len(sources), runs=len(runs),
                          metric_files=sum(s['relative_source'].endswith('.csv') for s in artifact_rows),
                          copied_megabytes=sum(s['bytes'] for s in sources)/1e6,
                          issues=issues, output=str(OUTPUT)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
