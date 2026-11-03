"""Analyze frozen HumanEval generations and official scores; no GPU or execution."""
from pathlib import Path
import json,csv,ast,math,hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
POLICIES=['fp16','full4','uniform3','query_only3','query_curve3']


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',*['|'+'|'.join(map(str,r))+'|' for r in rows]])


def main():
    scores={};generations={};summary={}
    problems=[json.loads(s) for s in (ROOT/'inputs/frozen/humaneval.jsonl').read_text().splitlines() if s.strip()]
    gold={r['case_id']:r['source_row'] for r in problems}
    for policy in POLICIES:
        folder=ROOT/'results/full'/policy/'humaneval'
        summary[policy]=json.loads((folder/'score_summary.json').read_text())
        scores[policy]={r['case_id']:r for r in [json.loads(p.read_text()) for p in (folder/'scores').glob('[0-9]*.json')]}
        generations[policy]={r['case_id']:r for r in [json.loads(p.read_text()) for p in folder.glob('[0-9]*.json')]}
        assert len(scores[policy])==len(generations[policy])==164 and summary[policy]['complete']
        assert all(r['grading_status']=='graded' for r in scores[policy].values())
        assert sum(r['score'] for r in scores[policy].values())/164==summary[policy]['score']
        for case in scores[policy]:
            p=folder/f"{generations[policy][case]['index']:05d}.json"
            assert hashlib.sha256(p.read_bytes()).hexdigest()==scores[policy][case]['generation_sha256']
    ids=sorted(gold,key=lambda c:int(c.split('/')[-1]))
    comparisons=[];rng=np.random.default_rng(20261027)
    pairs=[('full4','fp16'),('uniform3','fp16'),('query_only3','uniform3'),('query_curve3','uniform3'),('query_curve3','query_only3'),('query_only3','full4'),('query_curve3','full4')]
    for candidate,baseline in pairs:
        a=np.array([scores[candidate][c]['score'] for c in ids]);b=np.array([scores[baseline][c]['score'] for c in ids]);d=a-b
        ci=np.percentile(d[rng.integers(0,164,(10000,164))].mean(1),[2.5,97.5])*100
        wins=int((d>0).sum());losses=int((d<0).sum());n=wins+losses
        p=min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.
        comparisons.append(dict(candidate=candidate,baseline=baseline,difference_percentage_points=float(d.mean()*100),ci95_percentage_points=ci.tolist(),wins=wins,losses=losses,ties=int((d==0).sum()),exact_McNemar_p_unadjusted=p))
    diagnostic={};request=[]
    for policy in POLICIES:
        counts=dict(passed=0,syntax_or_indentation=0,test_or_runtime_failure=0,timeout=0)
        for case in ids:
            score=scores[policy][case];generation=generations[policy][case];problem=gold[case]
            category='passed'
            if not score['score']:
                program=problem['prompt']+score['completion']+'\n'+problem['test']+'\ncheck('+problem['entry_point']+')'
                try:ast.parse(program)
                except SyntaxError:category='syntax_or_indentation'
                else:
                    result=score['official_execution_result']
                    category='timeout' if result.get('timeout') or 'timed out' in str(result.get('error','')) else 'test_or_runtime_failure'
            counts[category]+=1
            request.append(dict(case_id=case,policy=policy,passed=int(score['score']),diagnostic_category=category,generated_tokens=generation['generated_tokens'],finish_reason=generation['finish_reason'],hit_output_limit=generation['hit_output_limit'],extraction_rule=score['extraction_rule']))
        diagnostic[policy]=counts
    with (OUT/'request_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(request[0]));w.writeheader();w.writerows(request)
    (OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n')
    (OUT/'failure_categories.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2)+'\n')
    def pct(policy):return summary[policy]['score_percent']
    results=table(['策略','每次调用平均 K/V payload 位数','通过题数','pass@1','相对 FP16 分数差'],[[p,16 if p=='fp16' else 4 if p=='full4' else 3,f"{diagnostic[p]['passed']}/164",f'{pct(p):.2f}%',f'{pct(p)-pct("fp16"):+.2f} 个百分点'] for p in POLICIES])
    paired=table(['候选相对基线','通过率差','配对 bootstrap 95% CI','候选修复／退化题数'],[[r['candidate']+' vs '+r['baseline'],f"{r['difference_percentage_points']:+.2f} 个百分点",f"[{r['ci95_percentage_points'][0]:+.2f}, {r['ci95_percentage_points'][1]:+.2f}]",f"{r['wins']} / {r['losses']}"] for r in comparisons])
    failures=table(['策略','语法或缩进失败','其余测试／运行失败','超时失败','输出触顶'],[[p,diagnostic[p]['syntax_or_indentation'],diagnostic[p]['test_or_runtime_failure'],diagnostic[p]['timeout'],summary[p]['hit_output_limit']] for p in POLICIES])
    text=f"""# HumanEval 全量生成结果分析

五种策略的 HumanEval 全部 164 题均已完成生成和官方测试，共 820 次生成，无评分缺失。结果支持动态读取分配在相同三位预算下提升代码任务质量，但本轮实测失真曲线没有提供额外收益。Query-only 是当前 HumanEval 上表现更好的三位策略；这不是跨数据集最终选择。

## 一 实验条件与完整性

模型为固定 revision 的 Qwen3-4B、FP16 权重、enable_thinking=False、每题一次贪心生成，上限 2048 token，允许模型 EOS 自然结束。输入 prompt 与官方 HumanEval 问题一致，聊天指令要求实现完整函数；全部策略使用相同输入和冻结输出提取规则。

原始 Jacobian 编码为 D-K output-Jacobian 树与 D-V attention 树、种子 20261024。所有策略 prefill 均为 FP16；量化策略在 decode 的全部 36 层使用重建 KV，并将误差传入后续层和 token。不是固定轨迹单层反事实。

采用 OpenAI 官方测试及 execution.py，在 namespace/chroot 中执行。聊天完整函数输出通过固定格式处理变成 completion；保留原始文本和实际被测试代码，不按测试结果修复。此前 164 个 canonical solution 均通过同一评分链路。本次所有 820 个评分文件的生成 SHA256、覆盖数和分数汇总已复核。

没有输入截断。除 query_curve3 的 HumanEval/32 达到 2048 上限外，其余 819 个输出均自然 EOS。结果不是旧固定 64 步实验的重用。

## 二 整体任务质量

{results}

fp16 到 full4 净少通过两题，说明本轮完整四位编码的点估计损失相对较小。但逐题是 9 题修复、11 题退化，不是只有两题受到影响；也不能由均值接近就认定与 FP16 等价。

uniform3 从 full4 的 126 题降至 86 题，是明显的少读精度损失。query_only3 恢复到 117 题，query_curve3 恢复到 112 题，支持 controller 分配比每处统一三位更有价值。

本轮平均 K、V 读取预算各固定三位，同一次 attention、相同上下文下，相对完整读取四位的逻辑 payload 少 25%；相对 uniform3 没有总预算优势，只改变位数分配位置。不同策略生成长度不同，不能把全请求累计字节的差直接归因于 controller；元数据与实际 DRAM 流量也不包含在这句 25% 中。HumanEval 在新流量版本切换前完成，没有物理流量实测，不能据此宣称 GPU 少传输了 25%。

## 三 请求配对比较

{paired}

区间以题目为配对单位、固定 seed 20261027、bootstrap 10000 次，单位为百分点。它们描述这批任务上的重抽样差异，不代表多模型、多随机生成种子的稳定性。额外保存了精确 McNemar 检验的未校正 p 值，多比较属于探索性分析。

Query-only 相对均匀三位净增加 31 题，差值 +18.90 个百分点，区间为 [+11.59, +26.22]；Query＋实测曲线净增加 26 题，差值 +15.85，区间为 [+9.15, +22.56]。这两项区间均高于零，是本测试集中较明确的动态分配质量收益。

query_curve3 相对 query_only3 修复 10 题、退化 15 题，净少五题；差值 −3.05 个百分点，区间 [−9.15, +3.05] 跨过零。当前没有证据说明实测曲线提供增量收益，也不能据此断言它普遍更差。

query_only3 相对 full4 的点估计仍低 5.49 个百分点，区间 [−11.59, +0.61]。虽然区间跨零，但这不构成非劣或等价证明；query_curve3 相对 full4 低 8.54 个百分点，区间 [−14.02, −2.44]。因此目前不能声称“少读 25% 而没有质量损失”。full4 相对 FP16 的区间同样较宽，不能认定为无损。

## 四 失败类型和生成长度

{failures}

语法/缩进分类通过对实际提交官方测试的程序进行 AST 静态解析获得，没有重新执行模型代码。其余失败归为测试/运行失败，包含断言、边界条件、类型或名称错误等，不能全部称作算法逻辑错误；输出触顶是独立标志，可能与上述失败重叠。

uniform3 出现 24 题语法或缩进失败，动态两组均为 8 题。动态分配不仅改变最终测试均值，也减少了明显的代码结构损坏。它支持有限精度更合理地分配后能保护代码生成，但还不能由此分解 K 与 V 各自贡献。

平均输出 token 分别为 FP16 150.8、full4 158.6、uniform3 118.0、query_only3 146.4、query_curve3 165.6。均匀三位更短并不意味着生成更高效，可能包含代码缺失或提前结束。只有一题触顶，无法解释动态策略与均匀三位之间的大部分质量差距，也不能解释曲线策略相对 Query-only 净少五题的全部差距。

## 五 可核对的具体例子

HumanEval/0 要求判断列表中是否有两数足够接近。uniform3 输出错误函数名并出现括号缺失，官方测试失败；两种动态策略均通过。该例展示相同三位预算下保护生成结构的实际区别，不是单纯 token 轨迹一致率变化。

HumanEval/5 要求在列表相邻元素间插入分隔符。uniform3 的循环括号缺失；query_only3 使用 `if numbers` 保护空列表并通过；query_curve3 无条件访问 `numbers[-1]`，在空列表时触发 list index out of range。这个反例说明更复杂的代理不能保证对最终边界条件更可靠。

HumanEval/32 的 query_curve3 输出在重复说明中耗尽 2048 token，最终测试失败；同题 query_only3 与 uniform3 也未通过。不能把曲线组触顶这一例解释为两动态策略净差五题的来源。

## 六 对 controller 的判断与下一步

目前可以支持的结论是：在 HumanEval 这一完整代码任务上，当前 Query 引导的两阶段 controller 比均匀三位读取更好地保持了任务质量。不能据此证明部署加速、实际 DRAM 节省、与完整四位质量等价，或在数学和长上下文任务上同样有效。

实测曲线在此前局部回放中曾有小幅 MSE 增益，但在当前任务通过率上未体现。校准曲线来自 LongBench，HumanEval 的激活和误差分布可能不同；V 残差交叉项和对角代理失配也可能影响分配。这些是待验证解释，不是已由本次成绩证明的因果结论。

本轮应继续保持五策略、编码和判分冻结，把 MATH-500 与全部 LongBench 跑完，不依据 HumanEval 临时改参。之后优先做 K-only/V-only 动态分配消融，并比较校准预测与真实误差是否在代码任务上失配。若其他任务同样没有稳定曲线增益，Query-only 将是更简单的工程实现候选；是否采用仍需综合全部任务的质量与开销。

## 数据入口

[逐题诊断](request_diagnostics.csv)、[请求配对比较](paired_comparisons.json)、[失败分类](failure_categories.json)；原始生成与官方评分位于上级 full/策略/humaneval。所有源测量未修改，本分析不使用 GPU，不影响正在运行的数学实验。
"""
    (OUT/'结果分析.md').write_text(text)
    (OUT/'audit.json').write_text(json.dumps(dict(complete=True,policies=5,cases_per_policy=164,graded_generations=820,generation_hashes_verified=True,bootstrap_samples=10000,bootstrap_seed=20261027,official_scores_unmodified=True,GPU_used=False),ensure_ascii=False,indent=2)+'\n')
    print(results);print(paired)


if __name__=='__main__':main()
