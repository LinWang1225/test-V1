"""Read-only CPU snapshot of experiment evidence; never starts an experiment.

Run with the project Python. Writes only a NEW output directory, supplied by
--output. Existing evidence and the running queues are never modified.
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
EXCLUDED = {'deps', 'vendor', 'references', 'source_baseline', '__pycache__',
            'calibration', 'codebooks', 'profiles', 'collector_jobs', 'checkpoints',
            'logs', 'smoke', 'smoke_online', 'smoke_replay', 'smoke_development',
            'smoke_replay_development', 'diagnostics_before_live_counting',
            'diagnostics_profiler_stall_20261105', 'inputs', 'native', 'points',
            'servers', 'auto_report_snapshots', 'matplotlib_cache', 'figures',
            'evaluation', 'distortion_dataset', 'diminishing_dataset'}
LARGE_RECORD_FIELDS = {'generated_ids', 'token_ids', 'text', 'raw_text',
                       'thinking_text', 'completion', 'final_response',
                       'fp16_final_response', 'original_source_audit'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def table(headers, rows):
    def cell(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['|' + '|'.join(map(cell, headers)) + '|',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['|' + '|'.join(map(cell, row)) + '|' for row in rows]) + '\n\n'


def fields(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in LARGE_RECORD_FIELDS:
                continue
            label = '*' if str(k).isdigit() else str(k)
            yield from fields(v, prefix + ('.' if prefix else '') + label)
    elif isinstance(obj, list):
        if not obj:
            yield prefix + '[]', 'empty_list'
        for value in obj:
            yield from fields(value, prefix + '[]')
    else:
        yield prefix, type(obj).__name__


def build_raw_metric_catalog(out):
    """Index raw metric files and their exact headers without copying big data."""
    artifacts, field_rows, examples = [], set(), []
    roots = [p for p in (PROJECT / 'experiments').iterdir() if p.is_dir()]
    roots += [PROJECT / 'results/progressive_int4_20261023',
              PROJECT / 'results/progressive_int4_online_v_20261023']
    skip = EXCLUDED - {'evaluation', 'distortion_dataset', 'diminishing_dataset'}
    for root in sorted(roots):
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith('diagnostics_')
                             and not d.startswith('smoke'))
            for name in sorted(names):
                p = Path(current) / name
                if not name.endswith(('.csv', '.csv.gz', '.parquet', '.jsonl.gz')):
                    continue
                rel = str(p.relative_to(PROJECT))
                header = []
                if name.endswith(('.csv', '.csv.gz')):
                    opener = gzip.open if name.endswith('.gz') else open
                    with opener(p, 'rt', encoding='utf-8-sig') as f:
                        header = next(csv.reader(f), [])
                    field_rows.update((rel, column) for column in header)
                artifacts.append(dict(source=rel, bytes=p.stat().st_size,
                    format='csv_gzip' if name.endswith('.csv.gz') else p.suffix[1:],
                    columns=json.dumps(header, ensure_ascii=False),
                    note='原始文件引用；CSV读取全部表头，未扫描全部行；Parquet/JSONL模式由对应汇总与源码解释'))
    validated = json.loads((PROJECT / 'results/experiment_analysis_20261022/validated_results.json').read_text())
    for method in validated['runs']:
        point = next(p for p in validated['points'] if p['method'] == method)
        for suffix in ['measured/request_metrics.json', 'traces/kernel_summary.json']:
            p = Path(point['attempt']) / suffix
            raw = p.read_bytes()
            o = json.loads(raw)
            rel = str(p.relative_to(PROJECT))
            field_rows.update((rel, key) for key, _ in fields(o))
            examples.append(dict(method=method, source=rel, sha256=sha(raw), evidence=o))
    with (out / 'raw_metric_artifacts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'bytes', 'format', 'columns', 'note'])
        w.writeheader(); w.writerows(artifacts)
    with (out / 'raw_metric_fields.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['source', 'field'])
        w.writerows(sorted(field_rows))
    dump(out / 'operator_metric_examples.json', examples)
    dump(out / 'raw_metric_catalog_manifest.json', dict(
        recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        artifacts=len(artifacts), source_field_pairs=len(field_rows),
        operator_examples=len(examples),
        scope='Metric CSV/CSV.gz headers plus references to Parquet/JSONL; smoke and profiles directories excluded; validation/gate CSVs retained as validation evidence, not formal task results; original generation and hardware paths are in per-request snapshots'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = dt.datetime.now(dt.timezone.utc)
    cutoff = start.timestamp()
    sources, errors, schema = [], [], {}
    source_rows = {}

    def capture(p):
        p = Path(p)
        rel = str(p.relative_to(PROJECT))
        if rel in source_rows or not p.is_file():
            return
        try:
            raw = p.read_bytes()
            destination = out / 'sources' / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            item = dict(source=rel, snapshot=str(destination.relative_to(out)),
                        bytes=len(raw), sha256=sha(raw),
                        mtime=dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat())
            if p.suffix == '.csv':
                reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
                columns = reader.fieldnames or []
                item['columns'] = columns
                item['rows'] = sum(1 for _ in reader)
                schema[rel] = {(f, 'csv_column') for f in columns}
            elif p.suffix == '.json':
                data = json.loads(raw)
                schema[rel] = set(fields(data))
            sources.append(item)
            source_rows[rel] = item
        except Exception as exc:
            errors.append(dict(source=rel, error=str(exc)))

    # Aggregate evidence, reports and run conditions. Request files are handled
    # separately, so incomplete generations and inherited files are explicit.
    roots = [p for p in (PROJECT / 'experiments').iterdir() if p.is_dir()]
    roots += [p for p in (PROJECT / 'results').iterdir()
              if p.is_dir() and p.name.startswith(('operator_', 'progressive_', 'experiment_analysis_20261022'))]
    for root in sorted(roots):
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith('diagnostics_')
                             and not d.startswith('smoke') and d != 'full')
            # Formal full/ aggregates are needed, but not per-method generations.
            if Path(current).name == 'results' and (Path(current) / 'full').is_dir():
                dirs.append('full')
            if Path(current).name == 'full':
                dirs[:] = []
            if 'results' in Path(current).relative_to(root).parts:
                dirs[:] = [d for d in dirs if d not in {'fp16', 'full4', 'uniform3', 'query_only3',
                         'query_curve3', 'adaptive005', 'adaptive015', 'adaptive030'}
                          and '__fixed' not in d]
            for name in sorted(names):
                p = Path(current) / name
                if p.suffix not in {'.json', '.csv', '.md', '.txt'}:
                    continue
                if name.startswith('longbench_') or name.endswith(('_trajectory.json', '_numerical.json')):
                    continue
                if name in {'controller.pid', 'background.json', 'launch.json', 'firstpass_switch.json'}:
                    continue
                if p.stat().st_size <= 20_000_000:
                    capture(p)
        for p in root.glob('inputs/manifest.json'):
            capture(p)
        for p in root.glob('validation/*.json'):
            if not p.name.startswith(('counter_', 'root_plain_', 'capacity_')):
                capture(p)
    for p in (PROJECT / 'configs').glob('*.json'):
        capture(p)
    for rel in ['results/method_queue.json', 'results/environment-verification.txt',
                'logs/check-cuda.json', 'experiments/active_vllm_experiment.json']:
        capture(PROJECT / rel)

    # Capture representative raw schemas, without reading multi-GB measurement
    # tables. Exact tables remain in their immutable original locations.
    representatives = [
        'experiments/encoding_ablation_20261024/results/full/*_attention.csv.gz',
        'experiments/encoding_ablation_20261024/results/full/*_distortion.csv.gz',
        'experiments/encoding_ablation_20261024/E_extension/results/full/*_attention.csv.gz',
        'experiments/encoding_extension_20261026/results/full/evaluation/*',
        'experiments/staged_vllm_recovery_20261106/profiles/fp16/*.windows.csv',
    ]
    for pattern in representatives:
        paths = sorted(PROJECT.glob(pattern))
        if not paths:
            continue
        p = paths[0]
        if p.name.endswith('.csv.gz'):
            with gzip.open(p, 'rt') as f:
                columns = next(csv.reader(f))
            schema[str(p.relative_to(PROJECT))] = {(k, 'csv_gz_column') for k in columns}
        elif p.suffix == '.csv':
            capture(p)

    generation_rows = []
    scalar_rows = []
    snapshot_checks = {'score_hash_mismatches': [], 'counter_execution_mismatches': [],
                       'dram_addition_mismatches': [], 'duplicate_run_case': []}
    seen = set()
    paths = []
    for directory in ['staged_algorithm_20261102', 'staged_sampling_20261105',
                      'staged_sampling_windowed_20261105', 'staged_vllm_20261105',
                      'staged_vllm_recovery_20261106', 'native_k4v4_quality_20261107']:
        root = PROJECT / 'experiments' / directory
        paths.extend((directory, p) for p in sorted(root.glob('results/*/[0-9]*.json')))
    for directory in ['full_quality_20261027', 'winner_quality_20261029']:
        root = PROJECT / 'experiments' / directory
        paths.extend((directory, p) for p in sorted(root.glob('results/full/*/*/*.json'))
                     if p.stem not in {'scoring_status', 'active_checkpoint', 'complete'})
    for directory, p in paths:
        if p.stat().st_mtime > cutoff:
            continue
        raw = p.read_bytes()
        o = json.loads(raw)
        if not ('generated_tokens' in o and ('case_id' in o or 'id' in o)):
            continue
        rel = str(p.relative_to(PROJECT))
        score_path = p.parent / 'scores' / p.name
        score = json.loads(score_path.read_text()) if score_path.exists() else None
        if score and score.get('generation_sha256') and score['generation_sha256'] != sha(raw):
            snapshot_checks['score_hash_mismatches'].append(rel)
        physical = o.get('physical_traffic', {})
        if physical.get('status') == 'measured':
            if physical.get('execution_id') and physical['execution_id'] != o.get('execution_id'):
                snapshot_checks['counter_execution_mismatches'].append(rel)
            if physical.get('dram_total_bytes') != physical.get('dram_read_bytes', 0) + physical.get('dram_write_bytes', 0):
                snapshot_checks['dram_addition_mismatches'].append(rel)
        run = o.get('run', o.get('policy', p.parent.parent.name))
        identity = (directory, run, o.get('case_id', o.get('id')))
        if identity in seen:
            snapshot_checks['duplicate_run_case'].append(identity)
        seen.add(identity)
        cleaned = {k: v for k, v in o.items() if k not in LARGE_RECORD_FIELDS}
        clean_score = {k: v for k, v in (score or {}).items() if k not in LARGE_RECORD_FIELDS}
        generation_rows.append(dict(experiment=directory, run=run, source=rel,
                                    source_sha256=sha(raw), metrics=cleaned, score=clean_score))
        schema.setdefault(directory + '/generation_json', set()).update(fields(o))
        schema.setdefault(directory + '/score_json', set()).update(fields(score or {}))
        scalar_rows.append(dict(experiment=directory, run=run, dataset=o.get('dataset'),
            case_id=o.get('case_id', o.get('id')), index=o.get('index'),
            input_tokens=o.get('input_tokens'), generated_tokens=o.get('generated_tokens'),
            thinking_tokens=o.get('thinking_tokens'), final_tokens=o.get('final_tokens'),
            thinking_complete=o.get('thinking_complete'), finish_reason=o.get('finish_reason'),
            natural_completion=o.get('natural_completion'),
            elapsed_seconds=o.get('elapsed_seconds', o.get('generation_seconds')),
            logical_KV_read_bits=o.get('logical_KV_read_bits'),
            element_weighted_mean_bits=o.get('element_weighted_mean_bits'),
            physical_status=physical.get('status'),
            dram_read_bytes=physical.get('dram_read_bytes'),
            dram_write_bytes=physical.get('dram_write_bytes'),
            dram_total_bytes=physical.get('dram_total_bytes'),
            score=score.get('score') if score else None,
            completed_at=o.get('completed_at'), source=rel, source_sha256=sha(raw)))
    with gzip.open(out / 'generation_metrics.jsonl.gz', 'wt', encoding='utf-8') as f:
        for row in generation_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    if scalar_rows:
        with (out / 'generation_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(scalar_rows[0])); w.writeheader(); w.writerows(scalar_rows)
    with (out / 'metric_fields.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['source_or_schema', 'field', 'type'])
        for name in sorted(schema):
            for field, typ in sorted(schema[name]):
                w.writerow([name, field, typ])

    packages = {}
    for name in ['torch', 'triton', 'vllm', 'transformers', 'numpy', 'evalscope']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def command(args):
        return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()
    dump(out / 'environment_at_snapshot.json', dict(
        recorded_at=start.isoformat(), scope='Current machine only; historical versions use per-run evidence',
        python=platform.python_version(), platform=platform.platform(), packages=packages,
        gpu=command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv']),
        cpu=command(['lscpu']), memory=command(['free', '-b']),
        warning='No GPU inference or profiling launched by this script'))

    current = [r for r in generation_rows if r['experiment'] == 'staged_vllm_recovery_20261106']
    runs = {}
    tasks = {}
    for row in current:
        runs.setdefault(row['run'], []).append(row)
        tasks.setdefault((row['run'], row['metrics']['dataset']), []).append(row)
    expected = {}
    manifest_cases = PROJECT / 'experiments/staged_vllm_recovery_20261106/inputs/cases.jsonl'
    for line in manifest_cases.read_text().splitlines():
        case = json.loads(line)
        expected[case['dataset']] = expected.get(case['dataset'], 0) + 1
    latest_rows = []
    for (run, dataset), rr in sorted(tasks.items()):
        ss = [r['score']['score'] for r in rr if 'score' in r['score']]
        latest_rows.append(dict(run=run, dataset=dataset, expected=expected[dataset], generated=len(rr),
            scored=len(ss), complete=len(rr) == len(ss) == expected[dataset],
            score_percent=sum(ss) / len(ss) * 100 if ss else None,
            eos=sum(r['metrics']['finish_reason'] == 'eos' for r in rr),
            native_context_limit=sum(r['metrics']['finish_reason'] == 'native_context_limit' for r in rr),
            thinking_complete=sum(bool(r['metrics'].get('thinking_complete')) for r in rr),
            mean_output_tokens=sum(r['metrics']['generated_tokens'] for r in rr) / len(rr),
            mean_thinking_tokens=sum(r['metrics'].get('thinking_tokens', 0) for r in rr) / len(rr)))
    with (out / 'current_task_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(latest_rows[0]));w.writeheader();w.writerows(latest_rows)
    physical_rows = []
    for run, rr in sorted(runs.items()):
        measured = [r for r in rr if r['metrics'].get('physical_traffic', {}).get('status') == 'measured']
        p = dict(run=run, generated=len(rr), scored=sum('score' in r['score'] for r in rr),
                 measured=len(measured), complete=len(rr) == len(measured) == 464,
                 generated_tokens=sum(r['metrics']['generated_tokens'] for r in rr),
                 elapsed_seconds=sum(r['metrics'].get('elapsed_seconds', 0) for r in rr))
        for stage in ['prefill', 'decode']:
            for kind in ['read', 'write', 'total']:
                key = 'dram_' + kind + '_bytes'
                p[stage + '_' + key] = sum(r['metrics']['physical_traffic'].get('stages', {}).get(stage, {}).get(key, 0) for r in measured)
        for kind in ['read', 'write', 'total']:
            key = 'dram_' + kind + '_bytes'
            p[key] = sum(r['metrics']['physical_traffic'].get(key, 0) for r in measured)
        p['decode_bytes_per_output_token'] = p['decode_dram_total_bytes'] / p['generated_tokens']
        physical_rows.append(p)
    dump(out / 'current_physical_totals.json', physical_rows)
    with (out / 'current_run_totals.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(physical_rows[0]));w.writeheader();w.writerows(physical_rows)
    build_raw_metric_catalog(out)

    appendix = ['# 总实验记录数据附录\n\n',
                f'冻结读取起点：{start.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()}。'
                '逐题快照仅纳入此时已经完整落盘的结果；评分可能尚未跟上生成。'
                '旧报告的更新时间不同，以本附录逐题核对结果为准。\n\n',
                '## 当前 vLLM 完整方案的逐任务质量\n\n']
    complete_runs = [p['run'] for p in physical_rows if p['complete'] and p['scored'] == 464]
    ordering = [x for x in ['fp16','Jacobian_Tree__fixed4','JacK_MixV__fixed4','Data_Tree__fixed4'] if x in complete_runs]
    lookup = {(r['run'], r['dataset']): r for r in latest_rows}
    appendix.append(table(['任务', '题数'] + ordering,
        [[dataset, n] + [f"{lookup[(run,dataset)]['score_percent']:.4f}" for run in ordering]
         for dataset, n in expected.items()]))
    appendix += ['以上各任务分别使用官方指标，分数为百分制；不将不同任务分数汇成全部464题的准确率。\n\n',
                 '## 当前全部轮次覆盖\n\n', table(
                     ['方案','已生成','已评分','已实测','输出token','累计生成小时','完整464题'],
                     [[p['run'],p['generated'],p['scored'],p['measured'],p['generated_tokens'],
                       f"{p['elapsed_seconds']/3600:.3f}",p['complete']] for p in physical_rows]),
                 '## 当前物理流量\n\n', table(
                     ['方案','Prefill读写TB','Decode读写PB','总读PB','总写PB','总读写PB','Decode GB/输出token'],
                     [[p['run'],f"{p['prefill_dram_total_bytes']/1e12:.6f}",
                       f"{p['decode_dram_total_bytes']/1e15:.6f}",f"{p['dram_read_bytes']/1e15:.6f}",
                       f"{p['dram_write_bytes']/1e15:.6f}",f"{p['dram_total_bytes']/1e15:.6f}",
                       f"{p['decode_bytes_per_output_token']/1e9:.6f}"] for p in physical_rows]),
                 'TB=10^12字节，PB=10^15字节，GB=10^9字节。未满464题的行只有阶段累计值。'
                 '单token归一化仍受上下文、输出长度及工作路径影响，不是等工作量对照。\n\n',
                 '## 当前自然结束与 thinking 完整性\n\n', table(
                     ['方案','任务','已生成/应有','EOS','上下文触顶','thinking完成','平均输出token','平均thinking token'],
                     [[r['run'],r['dataset'],f"{r['generated']}/{r['expected']}",r['eos'],r['native_context_limit'],
                       r['thinking_complete'],f"{r['mean_output_tokens']:.2f}",f"{r['mean_thinking_tokens']:.2f}"]
                      for r in latest_rows]),
                 '## 全部已归档的汇总与条件文件\n\n',
                 '每个链接是本次复制的冻结证据。文件清单还保存原始路径、字节数和SHA256。'
                 '源码、逐层原始张量和大规模硬件原始数据保留在原目录。\n\n',
                 table(['原始文件','冻结副本','行数或字节数'],
                       [[s['source'],f"[打开]({s['snapshot']})",s.get('rows',str(s['bytes'])+' bytes')]
                        for s in sources])]
    (out / '数据附录.md').write_text(''.join(appendix))
    dump(out / 'source_manifest.json', dict(snapshot_start=start.isoformat(),
         snapshot_end=dt.datetime.now(dt.timezone.utc).isoformat(), sources=sources,
         generation_rows=len(generation_rows), current_generation_rows=len(current),
         checks=snapshot_checks, errors=errors,
         raw_data_policy='Summary/config/report copies plus all generation metric records excluding text/token arrays; original raw data paths retained'))
    print(json.dumps(dict(output=str(out), sources=len(sources), generation_rows=len(generation_rows),
                          current_runs=physical_rows, errors=errors, checks=snapshot_checks),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
