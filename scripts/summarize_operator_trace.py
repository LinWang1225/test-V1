"""Stream worker traces; associate GPU kernels through launch correlations."""
import argparse
import bisect
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import ijson


def events(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as f:
        yield from ijson.items(f, 'traceEvents.item', use_float=True)


def classify(name):
    n = name.lower()
    if any(x in n for x in ['sinkhorn', 'hadamard', 'variance_normal', 'rms_norm_variance']):
        return 'rotation_normalization_sinkhorn'
    if any(x in n for x in ['_tq_fused_store', 'kvarn_flush', 'kvarn_store_tile', '_kvarn_scatter_store', 'quant_and_cache', 'reshape_and_cache']):
        return 'kv_store_quant_pack_fused'
    if any(x in n for x in ['_kvarn_dequant_blocks', '_tq_full_dequant']):
        return 'standalone_dequant'
    if any(x in n for x in ['_kvarn_build_packed', '_kvarn_pool_gather']):
        return 'kv_read_pack_dequant_fused'
    if any(x in n for x in ['_kvarn_fused', '_tq_decode', 'turboquant_decode', 'kvarn_decode']):
        return 'kv_read_dequant_attention_fused'
    if any(x in n for x in ['flash_fwd', 'flashattn', 'flash_attn', 'fmha', 'paged_attention', 'unified_attention', '_fwd_kernel', 'attn']):
        return 'attention_other_or_fp8_read_dequant_fused'
    if any(x in n for x in ['gemm', 'gemv', 'cublas', 'cutlass', 'matmul', 'ampere_s', 'ampere_h', 'sgemm', 'hgemm']):
        return 'gemm_unattributed'
    if any(x in n for x in ['sampling', 'top_k', 'topk', 'top_p', 'topp', 'softmax', 'argmax', 'multinomial']):
        return 'sampling_or_unattributed_softmax'
    if any(x in n for x in ['rotary', 'rope', 'layer_norm', 'layernorm', 'rms_norm', 'rmsnorm', 'silu', 'gelu']):
        return 'normalization_rope_activation'
    if any(x in n for x in ['copy', 'memcpy', 'memset', 'scatter', 'gather']):
        return 'copy_scatter_gather_other'
    return 'other_unclassified'


def describe(values):
    v = sorted(values)
    return {'calls': len(v), 'total_gpu_ms': sum(v) / 1000,
            'mean_gpu_us': sum(v) / len(v) if v else None,
            'p50_gpu_us': v[max(0, math.ceil(len(v) * .5) - 1)] if v else None,
            'p95_gpu_us': v[max(0, math.ceil(len(v) * .95) - 1)] if v else None,
            'max_gpu_us': v[-1] if v else None}


def union_us(intervals):
    if not intervals:
        return 0.0
    intervals.sort()
    start, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def summarize(folder):
    folder = Path(folder)
    paths = sorted(folder.rglob('*.pt.trace.json.gz')) + sorted(folder.rglob('*.pt.trace.json'))
    if not paths:
        raise RuntimeError(f'No worker trace under {folder}')
    kernels = defaultdict(list)
    runtime = defaultdict(list)
    cpu = defaultdict(list)
    category_intervals = defaultdict(list)
    all_intervals = defaultdict(list)
    scopes_out = []
    trace_info = []
    total = matched = 0
    for path in paths:
        scopes = defaultdict(list)
        components = defaultdict(list)
        launches = {}
        is_worker = False
        for e in events(path):
            if e.get('ph') != 'X' or 'dur' not in e:
                continue
            name = str(e.get('name', ''))
            cat = str(e.get('cat', '')).lower()
            ts, dur = float(e.get('ts', 0)), float(e['dur'])
            if name.startswith('op_phase|'):
                fields = dict(s.split('=', 1) for s in name.split('|')[1:] if '=' in s)
                scopes[(e.get('pid'), e.get('tid'))].append((ts, ts + dur, fields))
                scopes_out.append({'trace': str(path), 'cpu_start_us': ts, 'cpu_duration_us': dur, **fields})
                is_worker = True
            if name.startswith('op_component|'):
                components[(e.get('pid'), e.get('tid'))].append((ts, ts + dur, name.split('|', 1)[1]))
            if 'cuda_runtime' in cat or 'cuda_driver' in cat:
                args = e.get('args', {})
                corr = args.get('correlation')
                if corr is not None:
                    launches[(e.get('pid'), corr)] = (ts, e.get('tid'))
                    launches[(None, corr)] = (ts, e.get('tid'))
                runtime[name].append(dur)
            if cat == 'cpu_op':
                cpu[name].append(dur)
        if not is_worker:
            continue
        indexes = {}
        for key, vals in scopes.items():
            vals.sort(key=lambda x: x[0])
            indexes[key] = ([x[0] for x in vals], vals)
        component_indexes = {}
        for key, vals in components.items():
            vals.sort(key=lambda x: x[0])
            component_indexes[key] = ([x[0] for x in vals], vals)
        trace_kernels = 0
        for e in events(path):
            if e.get('ph') != 'X' or 'dur' not in e:
                continue
            cat = str(e.get('cat', '')).lower()
            if 'kernel' not in cat and cat not in ('gpu_memcpy', 'gpu_memset'):
                continue
            args = e.get('args', {})
            ts, dur = float(e.get('ts', 0)), float(e['dur'])
            corr = args.get('correlation')
            launch = launches.get((e.get('pid'), corr), launches.get((None, corr)))
            phase = 'unknown'
            seqs = 'unknown'
            component = 'unattributed'
            if launch:
                launch_ts, tid = launch
                # Runtime and kernel PIDs can differ; use the exact launching thread.
                for (pid0, tid0), (starts, vals) in indexes.items():
                    if tid0 != tid:
                        continue
                    j = bisect.bisect_right(starts, launch_ts) - 1
                    if j >= 0 and vals[j][0] <= launch_ts <= vals[j][1]:
                        phase = vals[j][2].get('phase', 'unknown')
                        seqs = vals[j][2].get('seqs', 'unknown')
                        break
                for (pid0, tid0), (starts, vals) in component_indexes.items():
                    if tid0 != tid:
                        continue
                    j = bisect.bisect_right(starts, launch_ts) - 1
                    for k in range(j, max(-1, j - 16), -1):
                        if vals[k][0] <= launch_ts <= vals[k][1]:
                            component = vals[k][2]
                            break
                    if component != 'unattributed':
                        break
            name = str(e.get('name', 'unknown'))
            category = classify(name) if 'kernel' in cat else cat
            if category == 'gemm_unattributed' and component != 'unattributed':
                category = component + '_gemm'
            kernels[(phase, seqs, component, category, name)].append(dur)
            device = args.get('device', e.get('pid'))
            all_intervals[(str(path), str(device))].append((ts, ts + dur))
            category_intervals[(phase, category, str(path), str(device))].append((ts, ts + dur))
            total += 1
            trace_kernels += 1
            matched += phase != 'unknown'
        trace_info.append({'path': str(path), 'compressed_bytes': path.stat().st_size, 'gpu_events': trace_kernels})
    if not total:
        raise RuntimeError('No annotated worker GPU events; refusing empty summary')
    rows = [{'phase': p, 'active_sequences': s, 'component': component, 'category': c, 'kernel': n, **describe(v)}
            for (p, s, component, c, n), v in kernels.items()]
    rows.sort(key=lambda r: r['total_gpu_ms'], reverse=True)
    aggregate = defaultdict(lambda: {'calls': 0, 'total_gpu_ms': 0.0})
    for r in rows:
        a = aggregate[(r['phase'], r['category'])]
        a['calls'] += r['calls']
        a['total_gpu_ms'] += r['total_gpu_ms']
    categories = [{'phase': p, 'category': c, **a,
                   'busy_union_ms': sum(union_us(v) / 1000 for (pp, cc, _, _), v in category_intervals.items() if pp == p and cc == c)}
                  for (p, c), a in aggregate.items()]
    result = {'trace_files': trace_info, 'gpu_event_count': total,
        'phase_matched_fraction': matched / total,
        'cumulative_gpu_event_ms': sum(r['total_gpu_ms'] for r in rows),
        'gpu_busy_union_ms': sum(union_us(v) / 1000 for v in all_intervals.values()),
        'categories': categories, 'top_kernels': rows[:60],
        'cuda_runtime_inclusive_us': {n: {'calls': len(v), 'total': sum(v), 'max': max(v)} for n, v in runtime.items()},
        'top_cpu_ops_inclusive_us': sorted([{'name': n, 'calls': len(v), 'total': sum(v)} for n, v in cpu.items()], key=lambda r: r['total'], reverse=True)[:60],
        'interpretation': 'Diagnostic eager/profiler timings. GPU busy union differs from cumulative durations. Fused KV reads/dequant/attention not split. CPU op totals are inclusive and non-additive. Phase labels use host launch correlation; unknown is retained.'}
    (folder / 'kernel_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (folder / 'kernels.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (folder / 'scheduler_scopes.jsonl').open('w') as f:
        for row in scopes_out:
            f.write(json.dumps(row) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    a = parser.parse_args()
    result = summarize(a.folder)
    print(json.dumps({k: result[k] for k in ['gpu_event_count', 'phase_matched_fraction', 'cumulative_gpu_event_ms', 'gpu_busy_union_ms']}))
