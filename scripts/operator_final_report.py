"""Comparable semantic families and repeat variation; never infer critical-path savings."""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def family(category):
    if category in ('standalone_dequant', 'kv_read_pack_dequant_fused', 'kv_read_dequant_attention_fused',
                    'attention_other_or_fp8_read_dequant_fused', 'attention_backend_gemm'):
        return 'attention_read_dequant_compute_combined'
    if category in ('kv_store_quant_pack_fused', 'kv_cache_update_gemm', 'rotation_normalization_sinkhorn'):
        return 'kv_write_quant_pack_rotation_combined'
    return category


def main(run):
    run = Path(run)
    summary = run / 'summary'
    summary.mkdir(exist_ok=True)
    plan = json.loads((run / 'plan.json').read_text())
    methods = plan['methods']
    states = defaultdict(dict)
    values = defaultdict(float)
    perpoint = defaultdict(float)
    allkernels = []
    for p in sorted((run / 'points').glob('*/*/state.json')):
        state = json.loads(p.read_text())
        point = state['point']
        if state['status'] != 'complete' or point['repeat'] == 0:
            continue
        method = p.parent.parent.name
        states[method][point['id']] = state
        trace = Path(state['attempt']) / 'traces'
        with (trace / 'kernels.csv').open() as f:
            for r in csv.DictReader(f):
                allkernels.append({'method': method, 'dataset': point['dataset'], 'batch': point['batch'],
                                  'repeat': point['repeat'], 'point': point['id'], **r})
        obj = json.loads((trace / 'kernel_summary.json').read_text())
        for c in obj['categories']:
            key = (method, point['dataset'], point['batch'], point['repeat'], c['phase'], family(c['category']))
            perpoint[key] += c['total_gpu_ms']
            if point['repeat'] == 1:
                values[(method, point['dataset'], c['phase'], family(c['category']))] += c['total_gpu_ms']
    if allkernels:
        with (summary / 'all_point_kernels.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(allkernels[0]))
            w.writeheader()
            w.writerows(allkernels)
    comparisons = []
    for method in methods:
        if method == 'fp16':
            continue
        for dataset in sorted({p['dataset'] for p in plan['points']}):
            wanted = {p['id'] for p in plan['points'] if p['dataset'] == dataset and p['repeat'] == 1}
            complete = wanted <= states[method].keys() and wanted <= states['fp16'].keys()
            if not complete:
                continue  # No partial coverage is promoted to a full comparison.
            keys = {(phase, group) for m, d, phase, group in values if d == dataset and m in (method, 'fp16')}
            total = sum(v for (m, d, _, _), v in values.items() if m == method and d == dataset)
            for phase, group in keys:
                q = values.get((method, dataset, phase, group), 0)
                b = values.get(('fp16', dataset, phase, group), 0)
                comparisons.append({'method': method, 'dataset': dataset, 'phase': phase, 'family': group,
                    'gpu_ms': q, 'fp16_gpu_ms': b, 'delta_gpu_ms': q - b,
                    'ratio_to_fp16': q / b if b else None,
                    'fraction_of_method_dataset_cumulative_gpu_time': q / total if total else None})
    comparisons.sort(key=lambda r: r['delta_gpu_ms'], reverse=True)
    repeats = []
    groups = {(m, d, batch, phase, group) for m, d, batch, _, phase, group in perpoint}
    for m, d, batch, phase, group in sorted(groups):
        v = [perpoint.get((m, d, batch, rep, phase, group), 0) for rep in (1, 2, 3)]
        ids = {f'{d}_b{batch:03d}_r{rep}' for rep in (1, 2, 3)}
        if ids <= states[m].keys():
            repeats.append({'method': m, 'dataset': d, 'batch': batch, 'phase': phase, 'family': group,
                            'repeat_gpu_ms': v, 'mean_gpu_ms': statistics.mean(v),
                            'cv': statistics.stdev(v) / statistics.mean(v) if statistics.mean(v) else None})
    coverage = {m: {'completed_points': len(states[m]), 'expected_points': plan['points_per_method']} for m in methods}
    report = {'coverage': coverage, 'comparisons': comparisons, 'representative_repeat_variation': repeats,
              'limitations': ['Eager profiler measurements; not production throughput or critical-path savings.',
                'FP8 uses FLASHINFER on SM86; E4M3 Triton store is unsupported. Backend differences are part of the result.',
                'Fused KV read/dequant/attention timings cannot be separated with this instrumentation.',
                'Family comparisons combine standalone and fused implementations; raw kernels retained.',
                'Per-phase request mixtures and active sequence counts can differ; inspect scheduler_scopes.jsonl.',
                'KVarN K4V2 and TurboQuant K4V4 use different storage precision; not an equal-byte comparison.',
                'Hardware DRAM counters unavailable with current driver permissions.',
                'Kernel accumulated duration is non-additive with concurrent kernels; busy-time unions reported separately.']}
    (summary / 'operator_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['GPU 算子分析报告', '', '覆盖情况: ' + json.dumps(coverage, ensure_ascii=False), '',
             '优化排查优先级（按相同数据集完整首轮的 GPU 累计时间增量排序；不是端到端可节省时间）:', '']
    for i, r in enumerate([r for r in comparisons if r['delta_gpu_ms'] > 0][:30], 1):
        lines.append(f"{i}. {r['method']} / {r['dataset']} / {r['phase']} / {r['family']}: "
                     f"{r['gpu_ms']:.3f} ms vs FP16 {r['fp16_gpu_ms']:.3f} ms, delta={r['delta_gpu_ms']:.3f} ms, "
                     f"占该方法该数据集累计GPU时间 {r['fraction_of_method_dataset_cumulative_gpu_time']:.2%}")
    lines += ['', '解释边界:', *report['limitations']]
    (summary / 'operator_report.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(coverage, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1])
