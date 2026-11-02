"""Incremental task quality, natural completion and logical-traffic reports."""
from collections import defaultdict
import csv
import numpy as np
from common import *

def main():
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    table=[];paired=[];generation_count=0;scored_count=0
    all_scores={}
    for folder in sorted((ROOT/'results').glob('*')):
        if not folder.is_dir():
            continue
        by_task=defaultdict(list)
        scored={}
        for p in sorted(folder.glob('[0-9]*.json')):
            g=json.loads(p.read_text());s=folder/'scores'/p.name
            score=json.loads(s.read_text()) if s.exists() else None
            by_task[g['dataset']].append((g,score));generation_count+=1
            if score is not None:
                scored_count+=1;scored[g['case_id']]=score
        all_scores[folder.name]=scored
        for dataset,expected in manifest['datasets'].items():
            values=by_task[dataset];graded=[(g,s) for g,s in values if s is not None]
            table.append(dict(run=folder.name,dataset=dataset,expected=expected,generated=len(values),scored=len(graded),
                complete=len(graded)==expected,score_percent=100*float(np.mean([s['score'] for g,s in graded])) if graded else None,
                natural_eos=sum(g['natural_completion'] for g,s in values),native_context_limit=sum(g['finish_reason']=='native_context_limit' for g,s in values),
                mean_output_tokens=float(np.mean([g['generated_tokens'] for g,s in values])) if values else None,
                logical_KV_bytes=sum(g['logical_KV_read_bits'] for g,s in values)/8,
                logical_bytes_per_output_token=sum(g['logical_KV_read_bits'] for g,s in values)/8/sum(g['generated_tokens'] for g,s in values) if values else None,
                local_metadata_peak_bytes=max((g['local_metadata_peak_bytes'] for g,s in values),default=0),
                local_metadata_write_bytes=sum(g['local_metadata_write_bytes'] for g,s in values)))
    rng=np.random.default_rng(SEED)
    baseline=all_scores.get('fp16',{})
    for run,scores in all_scores.items():
        if run=='fp16':
            continue
        for dataset,expected in manifest['datasets'].items():
            ids=sorted(k for k in scores.keys() & baseline.keys() if scores[k]['dataset']==dataset)
            if not ids:
                continue
            delta=np.array([scores[k]['score']-baseline[k]['score'] for k in ids])
            ix=rng.integers(0,len(delta),(2000,len(delta)))
            paired.append(dict(run=run,baseline='fp16',dataset=dataset,n_pairs=len(ids),expected=expected,
                complete=len(ids)==expected,mean_difference_pp=float(delta.mean()*100),
                ci95_pp=(np.percentile(delta[ix].mean(1),[2.5,97.5])*100).tolist(),
                gains=[k for k,d in zip(ids,delta) if d>0],losses=[k for k,d in zip(ids,delta) if d<0]))
    dest=ROOT/'analysis';dest.mkdir(parents=True,exist_ok=True)
    if table:
        with (dest/'task_metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    atomic(dest/'paired_comparisons.json',paired)
    atomic(dest/'progress.json',dict(generated=generation_count,scored=scored_count,cases_per_run=464,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
    text=['# 算法阶段结果进度','',f'已生成{generation_count}条，已评分{scored_count}条；每方案464条。未完整的分数仅为进度，不作为最终排名。',
        '', '逻辑读取量不代表DRAM实测。输出自然结束；上下文触顶单列。HumanEval、MATH与各LongBench子任务分别评分。',
        '', '|方案|任务|已评分/应有|分数%|EOS|原生上下文触顶|平均输出token|', '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        if r['generated']:
            score='待评分' if r['score_percent'] is None else f"{r['score_percent']:.2f}"
            text.append(f"|{r['run']}|{r['dataset']}|{r['scored']}/{r['expected']}|{score}|{r['natural_eos']}|{r['native_context_limit']}|{r['mean_output_tokens']:.1f}|")
    (dest/'结果进度.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    main()
