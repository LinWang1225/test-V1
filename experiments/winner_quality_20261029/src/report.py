from common import *
rows=[]
for policy in POLICIES:
 folder=ROOT/'results/full'/policy/'humaneval'
 records=[json.loads(p.read_text()) for p in sorted(folder.glob('[0-9]*.json'))]
 summary=folder/'score_summary.json'
 score=json.loads(summary.read_text()) if summary.exists() else None
 elements=sum(sum(sum(side.values()) for side in r['depth_elements_K_V']) for r in records)
 bits=sum(r['logical_KV_read_bits'] for r in records)
 rows.append(dict(policy=policy,completed=len(records),score=score,element_weighted_mean_bits=bits/elements if elements else None,logical_payload_bits=bits,full4_payload_ratio=bits/(4*elements) if elements else None,total_generated_tokens=sum(r['generated_tokens'] for r in records)))
atomic(ROOT/'results/summary.json',rows)
lines=['# HumanEval 新赢家质量与读取量','', '|策略|已完成|pass@1 (%)|元素加权平均位宽|相对本轨迹完整4-bit payload|','|---|---:|---:|---:|---:|']
for r in rows:
 lines.append(f"|{r['policy']}|{r['completed']}/164|{r['score']['score_percent'] if r['score'] else '待完成'}|{r['element_weighted_mean_bits']}|{r['full4_payload_ratio']}|")
lines+=['','读取量为 decode KV 逻辑 payload，不是硬件 DRAM 实测。各策略生成轨迹和长度可能不同，比值使用各自轨迹的完整4-bit分母。平均位宽按实际访问KV元素加权。所有阈值均报告，不按测试分数选择。']
(ROOT/'results/结果分析.md').write_text('\n'.join(lines)+'\n')
print(rows,flush=True)
