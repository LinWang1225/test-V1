"""Separate quality results from the parent's hardware-counter reports."""
from collections import defaultdict
import csv
import json
import numpy as np

from common import ROOT, PARENT, METHODS, atomic, cases, sha


def main():
    expected = defaultdict(int)
    for row in cases():
        expected[row['dataset']] += 1
    summaries = []
    paired = []
    scores = {}
    for name, folder in [('fp16', PARENT / 'results/fp16'),
                         ('Jacobian_Tree__fixed4', PARENT / 'results/Jacobian_Tree__fixed4'),
                         ('JacK_MixV__fixed4', PARENT / 'results/JacK_MixV__fixed4'),
                         ('Data_Tree__fixed4', PARENT / 'results/Data_Tree__fixed4'),
                         ('Gaussian_Tree__fixed4', PARENT / 'results/Gaussian_Tree__fixed4'),
                         *((m, ROOT / 'results' / m) for m in METHODS)]:
        by_dataset = defaultdict(list)
        scores[name] = {}
        for sp in sorted((folder / 'scores').glob('[0-9]*.json')):
            gp = folder / sp.name
            score = json.loads(sp.read_text())
            assert score['generation_sha256'] == sha(gp)
            generation = json.loads(gp.read_text())
            by_dataset[score['dataset']].append((generation, score))
            scores[name][score['case_id']] = (generation, score)
        for dataset, count in expected.items():
            values = by_dataset[dataset]
            failures = list((ROOT / 'failures' / name).glob('[0-9]*.json')) if name in METHODS else []
            failures = [json.loads(p.read_text()) for p in failures]
            summaries.append(dict(method=name, dataset=dataset, expected=count, scored=len(values),
                complete=len(values) == count,
                failed=sum(f['row']['dataset'] == dataset for f in failures),
                score_percent=100 * np.mean([s['score'] for g,s in values]) if values else None,
                unfinished_thinking=sum(not g['thinking_complete'] for g,s in values),
                native_context_limit=sum(g['finish_reason'] == 'native_context_limit' for g,s in values),
                mean_output_tokens=np.mean([g['generated_tokens'] for g,s in values]) if values else None))
    for name, records in scores.items():
        if name == 'fp16':
            continue
        for dataset, count in expected.items():
            ids = sorted(k for k in records.keys() & scores['fp16'].keys()
                         if records[k][1]['dataset'] == dataset)
            if not ids:
                continue
            delta = np.array([records[k][1]['score'] - scores['fp16'][k][1]['score'] for k in ids])
            rng = np.random.default_rng(20261107)
            samples = rng.integers(0, len(ids), (4000, len(ids)))
            losses = [k for k,d in zip(ids,delta) if d < 0]
            paired.append(dict(method=name, baseline='fp16', dataset=dataset, n_pairs=len(ids),
                complete=len(ids) == count, mean_difference_pp=float(delta.mean() * 100),
                ci95_pp=(np.percentile(delta[samples].mean(1), [2.5,97.5]) * 100).tolist(),
                losses=losses, gains=[k for k,d in zip(ids,delta) if d > 0],
                losses_unfinished_thinking=sum(not records[k][0]['thinking_complete'] for k in losses),
                losses_finished_thinking=sum(records[k][0]['thinking_complete'] for k in losses)))
    dest = ROOT / 'analysis'
    dest.mkdir(exist_ok=True)
    with (dest / 'quality_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    atomic(dest / 'paired_comparisons.json', paired)
    atomic(dest / 'progress.json', dict(methods={name:len(v) for name,v in scores.items()},
                                       expected_per_method=464, quality_only=True))


if __name__ == '__main__':
    main()
