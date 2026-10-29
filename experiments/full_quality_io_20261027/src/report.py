"""Incremental and final official-task quality reports; no generation reruns."""
import argparse
from common import *
import numpy as np


def main():
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    rows=[];summaries={}
    for policy in POLICIES:
        for dataset in manifest['datasets']:
            p=ROOT/'results/full'/policy/dataset/'score_summary.json'
            if p.exists():
                r=json.loads(p.read_text());assert r['complete'] and r['n_cases']==manifest['datasets'][dataset]['count']
                rows.append(r);summaries[policy,dataset]=r
    complete=len(rows)==len(POLICIES)*len(manifest['datasets'])
    totals={p:sum(r['n_cases'] for r in rows if r['policy']==p) for p in POLICIES}
    atomic(ROOT/'quality_progress.json',dict(complete=complete,scored_generations=sum(totals.values()),expected_generations=manifest['total_generations'],scored_by_policy=totals,completed_policy_dataset_pairs=len(rows),expected_policy_dataset_pairs=len(POLICIES)*len(manifest['datasets']),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
    if rows:
        with (ROOT/'results/task_scores.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:
        with (ROOT/'results/task_scores.csv').open('w') as f:
            csv.writer(f).writerow(['policy','dataset','n_cases','score_percent','complete'])
    title='完整生成官方任务质量结果' if complete else '完整生成官方任务质量评测进度'
    text=[f'# {title}','',f"已完成评分 {sum(totals.values())}/{manifest['total_generations']} 次生成。每种策略包含 HumanEval 164、MATH-500 500、LongBench 原版 21 子任务 4750，共 5414 个样本；未完成的单元以空缺显示，不视为零分。",'', '|任务|样本数|'+'|'.join(POLICIES)+'|','|---|---|'+'|'.join(['---']*len(POLICIES))+'|']
    for dataset,info in manifest['datasets'].items():
        vals=[f"{summaries[p,dataset]['score_percent']:.2f}" if (p,dataset) in summaries else '待完成' for p in POLICIES]
        text.append('|'+dataset+'|'+str(info['count'])+'|'+'|'.join(vals)+'|')
    text.extend(['','所有数值以百分制展示。HumanEval 为每题一次贪心生成的官方测试 pass@1，MATH-500 为官方 PRM800K 数学等价判分准确率，LongBench 为原版官方逐任务指标。LongBench 各任务的 F1、ROUGE、分类、检索和代码相似度不可简单当成同一种准确率。'])
    lb=[d for d in manifest['datasets'] if d not in ('humaneval','math500')]
    aggregates=[]
    for p in POLICIES:
        if all((p,d) in summaries for d in lb):
            aggregates.append(dict(policy=p,longbench_task_macro_percent=float(np.mean([summaries[p,d]['score_percent'] for d in lb])),non_calibration_task_macro_percent=float(np.mean([summaries[p,d]['non_calibration_score_percent'] for d in lb]))))
    if aggregates:
        atomic(ROOT/'results/longbench_macro.json',aggregates)
        text.extend(['','LongBench 的 21 任务等权宏平均是本实验声明的辅助汇总，不是官方脚本额外提供的一种统一指标：',''])
        text.extend([f"- {r['policy']}：全量宏平均 {r['longbench_task_macro_percent']:.2f}，排除编码校准请求的辅助宏平均 {r['non_calibration_task_macro_percent']:.2f}。" for r in aggregates])
    if rows:
        text.extend(['','## 自然结束与长度限制','','|策略与任务|平均输出 token|自然 EOS|触顶截断|输入截断|','|---|---|---|---|---|'])
        for r in rows:text.append(f"|{r['policy']} / {r['dataset']}|{r['mean_generated_tokens']:.1f}|{r['natural_eos_stops']}|{r['hit_output_limit']}|{r['truncated_prompts']}|")
    text.extend(['','LongBench 使用官方逐任务输出上限，HumanEval 上限 2048、MATH-500 上限 8192；均不强制最小长度，不忽略 EOS。输入窗口为模型原生 40960 token，LongBench 超长输入按首尾保留并记录，HumanEval 和数学不进行静默截断。','', '全量 LongBench 包含此前用于编码校准的 60 个请求以及曾探索的请求；不删除这些样本以保持全量，但记录重叠标记并提供排除校准请求的辅助指标。该结果不应被描述成全新独立测试集验证。','', '代码固定原始 Jacobian_Tree 的 D-K 与 D-V、种子 20261024。所有策略 prefill 为 FP16，量化策略的 decode 全部 36 层实际使用重建 KV。适配器保留 FP16 源缓存与整数码缓存，性能数据不是实际压缩服务的速度或容量证据。'])
    if complete:
        comparisons=[];rng=np.random.default_rng(20261027)
        for dataset in manifest['datasets']:
            scores={p:[json.loads(x.read_text()) for x in sorted((ROOT/'results/full'/p/dataset/'scores').glob('[0-9]*.json'))] for p in POLICIES}
            ids=[r['case_id'] for r in scores['fp16']]
            for p in POLICIES:assert [r['case_id'] for r in scores[p]]==ids
            for candidate in POLICIES[1:]:
                for baseline in ('fp16','uniform3'):
                    if candidate==baseline:continue
                    delta=np.array([a['score']-b['score'] for a,b in zip(scores[candidate],scores[baseline])]);ix=rng.integers(0,len(delta),(2000,len(delta)))
                    comparisons.append(dict(dataset=dataset,candidate=candidate,baseline=baseline,mean_score_difference_percentage_points=float(delta.mean()*100),ci95_percentage_points=(np.percentile(delta[ix].mean(1),[2.5,97.5])*100).tolist(),n_cases=len(delta)))
        atomic(ROOT/'results/paired_comparisons.json',comparisons)
        atomic(ROOT/'results/complete.json',dict(complete=True,expected_generations=manifest['total_generations'],scored_generations=sum(totals.values()),policy_dataset_pairs=len(rows),official_task_scores_complete=True,serving_performance_validated=False))
        text.extend(['','已保存请求配对 bootstrap 2000 次的分数差和 95% CI，见 paired_comparisons.json。该区间是探索性多比较，不宣称经过多重检验校正。'])
    text.extend(['','评分源码、prompt、原始输出和逐题评分可追溯；见 [协议](../PROTOCOL.md)、[数据清单](../inputs/manifest.json)、[官方源码来源](../vendor/official_sources.json)与 [任务汇总 CSV](task_scores.csv)。'])
    (ROOT/'results/结果分析.md').write_text('\n'.join(text)+'\n')
    print('quality report',sum(totals.values()),'/',manifest['total_generations'],'complete',complete,flush=True)


if __name__=='__main__':main()
