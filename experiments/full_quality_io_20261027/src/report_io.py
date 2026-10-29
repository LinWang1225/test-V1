"""Attach traffic coverage and matched-call comparisons to full quality reports."""
from common import *
import collections


def main():
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    data=[];summary=collections.defaultdict(lambda:dict(requests=0,measured=0,partial=0,unavailable=0,no_decode_call=0,K_payload_bits=0,V_payload_bits=0,K_packet_bytes=0,V_packet_bytes=0,warm_measured=0,evicted_measured=0,warm_dram_read_bytes=0,evicted_dram_read_bytes=0))
    for policy in POLICIES:
        for dataset in manifest['datasets']:
            if dataset in ('humaneval','math500'):continue
            for path in sorted((ROOT/'results/full'/policy/dataset).glob('[0-9]*.json')):
                a=json.loads(path.read_text());l=a['logical_traffic'];p=a['physical_traffic'];sample=p.get('matching_sample_logical_traffic') or {}
                row=dict(case_id=a['case_id'],policy=policy,dataset=dataset,index=a['index'],generated_tokens=a['generated_tokens'],K_payload_bits=l['K']['payload_bits'],V_payload_bits=l['V']['payload_bits'],K_packet_read_bytes=l['K']['packet_read_bytes'],V_packet_read_bytes=l['V']['packet_read_bytes'],shared_decoder_allowance_bytes=l['decoder_metadata_read_allowance_bytes'],controller_curve_allowance_bytes=l['controller_curve_read_allowance_bytes'],rotation_allowance_bytes=l['reference_dense_rotation_read_allowance_bytes'],physical_status=p['status'],physical_reason=p.get('reason',''),sample_layer=sample.get('layer'),sample_position=sample.get('decode_position'),sample_context=sample.get('context'),sample_logical_total_bytes=sample.get('logical_total_bytes'))
                s=summary[policy,dataset];s['requests']+=1;s[p['status']]+=1
                s['K_payload_bits']+=l['K']['payload_bits'];s['V_payload_bits']+=l['V']['payload_bits'];s['K_packet_bytes']+=l['K']['packet_read_bytes'];s['V_packet_bytes']+=l['V']['packet_read_bytes']
                receipts=p.get('conditions',p.get('partial_conditions',{}))
                for condition in ('warm','evicted'):
                    m=receipts.get(condition,{})
                    for field in ('dram_read_bytes','dram_write_bytes','dram_total_bytes'):row[condition+'_'+field]=m.get(field)
                    row[condition+'_dram_reads_to_logical_sample']=m.get('dram_read_bytes',0)/sample['logical_total_bytes'] if m and sample.get('logical_total_bytes') else None
                    if m:s[condition+'_measured']+=1;s[condition+'_dram_read_bytes']+=m['dram_read_bytes']
                data.append(row)
    if data:
        with (ROOT/'results/traffic_request_metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    else:
        with (ROOT/'results/traffic_request_metrics.csv').open('w') as f:csv.writer(f).writerow(['case_id','policy','dataset','physical_status'])
    results=[dict(policy=p,dataset=d,**s) for (p,d),s in sorted(summary.items())]
    atomic(ROOT/'results/traffic_summary.json',dict(requests_with_ledger=len(data),new_LongBench_generations_expected=4750*5,groups=results,physical_scope='one matched complete layer17 attention per request, warm/evicted; not whole generation or KV-address-only traffic',missing_values_are_null_not_zero=True,legacy_human_math_traffic_not_retroactively_measured=True))
    text=['# 全量质量评测读取量对比','',f'新代码已记录 {len(data)}/23750 条 LongBench 生成的逐请求逻辑账本。HumanEval 与 MATH-500 沿用切换前的冻结版本，不补造硬件测量。','', '|策略与任务|已记录请求|warm 与 evicted 均成功|部分成功|不可用|无 decode 调用|','|---|---|---|---|---|---|']
    for r in results:text.append(f"|{r['policy']} / {r['dataset']}|{r['requests']}|{r['measured']}|{r['partial']}|{r['unavailable']}|{r['no_decode_call']}|")
    text.extend(['','逻辑读取覆盖每条生成的全部实际 decode attention 调用，K/V 分开记录，包含所选位宽、位平面尾部填充和保守共享状态重载账本。FP16 的读取也按 16-bit 正确计入。不同停止长度不能直接用总请求字节判断策略节省。','', '物理测量依照 ReadKV 附录 G.2：每请求第 17 层第一次实际 decode 调用，已有 KV 缓存上的完整 eager attention；排除模型其余层及缓存构造，计入分配、反量化、attention 中间与输出访问。warm 和 128MiB 驱逐条件分别采集 DRAM 读/写/总量。逻辑对比使用同一采集调用的账本，不把单层实测外推成全程流量。','', '当前参考实现的整数码以 uint8 保存，未实现真实按位平面少读。物理流量反映这一实现及其 workspace，不能预设比均匀或 full4 少读。元数据、显式旋转矩阵和 controller 曲线的解析重载额外列出，既不是 DRAM 计数器值，也不混作模型权重。','', '采集失败只记录原因、原始错误和空值，生成与官方评分继续。不可用不是零字节，不使用 NVML 利用率或读取位数推算的字节替代硬件计数器。'])
    reasons=collections.Counter(str(r['physical_reason'])[:160] for r in data if r['physical_status'] in ('unavailable','partial'))
    if reasons:text.extend(['','## 实测失败原因','',*[f'- {count} 条：{reason}' for reason,count in reasons.most_common()]])
    text.extend(['','[逐请求对比 CSV](traffic_request_metrics.csv) · [覆盖汇总](traffic_summary.json) · [官方任务质量](结果分析.md)'])
    (ROOT/'results/读取量对比.md').write_text('\n'.join(text)+'\n')
    print('traffic report',len(data),'requests',flush=True)


if __name__=='__main__':main()
