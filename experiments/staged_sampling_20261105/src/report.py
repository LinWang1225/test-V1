"""Incremental task quality, natural completion and logical-traffic reports."""
from collections import defaultdict
import csv
import numpy as np
from common import *

def main():
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    table=[];paired=[];generation_count=0;scored_count=0;physical_summary=[]
    all_scores={}
    for folder in sorted((ROOT/'results').glob('*')):
        if not folder.is_dir():
            continue
        by_task=defaultdict(list)
        full_receipts={}
        scored={}
        for p in sorted(folder.glob('[0-9]*.json')):
            g=json.loads(p.read_text());s=folder/'scores'/p.name
            total=folder/'traffic_totals'/p.name
            if total.exists():
                receipt=json.loads(total.read_text())
                assert receipt['generation_sha256']==sha(p)
                full_receipts[g['index']]=receipt
            score=json.loads(s.read_text()) if s.exists() else None
            by_task[g['dataset']].append((g,score));generation_count+=1
            if score is not None:
                scored_count+=1;scored[g['case_id']]=score
        all_scores[folder.name]=scored
        total_output=sum(r['generated_tokens'] for r in full_receipts.values())
        summary=dict(run=folder.name,measured_requests=len(full_receipts),expected_requests=464,
            complete=len(full_receipts)==464,measured_output_tokens=total_output,
            scope='same actual end-to-end execution as quality; prefill and all decode including thinking; not KV-only')
        for phase in ('prefill','decode','total'):
            for metric in ('dram_read_bytes','dram_write_bytes','dram_total_bytes'):
                value=sum(r[metric] if phase=='total' else r['stages'][phase][metric] for r in full_receipts.values())
                summary[phase+'_'+metric]=value if full_receipts else None
        summary['decode_dram_bytes_per_output_token']=summary['decode_dram_total_bytes']/total_output if total_output else None
        physical_summary.append(summary)
        for dataset,expected in manifest['datasets'].items():
            values=by_task[dataset];graded=[(g,s) for g,s in values if s is not None]
            receipts=[full_receipts[g['index']] for g,s in values if g['index'] in full_receipts]
            table.append(dict(run=folder.name,dataset=dataset,expected=expected,generated=len(values),scored=len(graded),
                complete=len(graded)==expected,score_percent=100*float(np.mean([s['score'] for g,s in graded])) if graded else None,
                natural_eos=sum(g['natural_completion'] for g,s in values),native_context_limit=sum(g['finish_reason']=='native_context_limit' for g,s in values),
                mean_output_tokens=float(np.mean([g['generated_tokens'] for g,s in values])) if values else None,
                thinking_complete=sum(g['thinking_complete'] for g,s in values),
                mean_thinking_tokens=float(np.mean([g['thinking_tokens'] for g,s in values])) if values else None,
                physical_measured_requests=sum(g['physical_traffic']['status']=='measured' for g,s in values),
                whole_generation_physical_measured_requests=len(receipts),
                cumulative_dram_read_bytes=sum(r['dram_read_bytes'] for r in receipts) if receipts else None,
                cumulative_dram_write_bytes=sum(r['dram_write_bytes'] for r in receipts) if receipts else None,
                cumulative_prefill_dram_bytes=sum(r['stages']['prefill']['dram_total_bytes'] for r in receipts) if receipts else None,
                cumulative_decode_dram_bytes=sum(r['stages']['decode']['dram_total_bytes'] for r in receipts) if receipts else None,
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
    atomic(dest/'physical_totals.json',physical_summary)
    atomic(dest/'progress.json',dict(generated=generation_count,scored=scored_count,cases_per_run=464,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
    text=['# 算法阶段结果进度','',f'已生成{generation_count}条，已评分{scored_count}条；每方案464条。未完整的分数仅为进度，不作为最终排名。',
        '', 'Thinking + 固定请求种子采样；同一批464题，包含全部21个LongBench子任务。输入不截断、无分任务输出上限，只有模型统一40960总上下文边界。',
        '', '物理DRAM在本次真实端到端生成现场采集：prefill与全部decode累计硬件计数和最终质量对应同一次执行，不使用参考token或第二次生成。范围包含模型权重、KV写入、重建和采样，不是纯KV流量。仅最终回答评分，thinking计入全部成本。',
        '', '|方案|任务|已评分/应有|分数%|EOS|原生上下文触顶|平均输出token|', '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        if r['generated']:
            score='待评分' if r['score_percent'] is None else f"{r['score_percent']:.2f}"
            text.append(f"|{r['run']}|{r['dataset']}|{r['scored']}/{r['expected']}|{score}|{r['natural_eos']}|{r['native_context_limit']}|{r['mean_output_tokens']:.1f}|")
    text+=['','## 完整生成累计物理流量','','每方案统计仅累加已实际测量请求；全部464题采集后才是最终总量。GiB=2^30字节，所有原始字节数见physical_totals.json及task_metrics.csv。',
        '', '|方案|已实测/464|Prefill读写GiB|Decode读写GiB|总读取GiB|总写入GiB|', '|---|---:|---:|---:|---:|---:|']
    for r in physical_summary:
        def gib(key):return '待采集' if r[key] is None else f"{r[key]/2**30:.3f}"
        text.append(f"|{r['run']}|{r['measured_requests']}/464|{gib('prefill_dram_total_bytes')}|{gib('decode_dram_total_bytes')}|{gib('total_dram_read_bytes')}|{gib('total_dram_write_bytes')}|")
    (dest/'结果进度.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    main()
