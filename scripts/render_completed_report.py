"""Render the frozen experiment analysis in Chinese PDF and Markdown."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

def render(out,t,raw):
 pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
 pdfmetrics.registerFont(TTFont('Chinese','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
 styles={k:ParagraphStyle(k,fontName='Chinese',fontSize=size,leading=lead,wordWrap='CJK',spaceAfter=space,textColor=colors.HexColor('#172e45')) for k,size,lead,space in [('title',21,30,18),('h',14,21,12),('p',9,15,9),('small',7,11,5)]}
 story=[];md=[]
 def markup(s):
  return ''.join('<font name="Latin">'+escape(x)+'</font>' if x.isascii() else escape(x) for x in re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',str(s)))
 def p(s,kind='p'):
  story.append(Paragraph(markup(s),styles[kind]));md.append(('## ' if kind=='h' else '# ' if kind=='title' else '')+s+'\n')
 def table(headers,rows,widths):
  data=[[Paragraph(markup(x),styles['small']) for x in r] for r in [headers]+rows]
  tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.6,colors.HexColor('#8aa6bd')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(tab);story.append(Spacer(1,12))
  md.extend(['| '+' | '.join(map(str,headers))+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,r))+' |' for r in rows]+[''])
 def page():story.append(PageBreak())
 def pct(x,y):return f'{100*(x/y-1):+.2f}%'
 methods=list(t);base=t['fp16']['total']
 p('KV 量化算子实验：最终结果','title')
 p('2026-10-22｜四方法首轮对齐结果｜RTX 3090 / Qwen3-4B')
 p('1. 核心结论与完成情况','h')
 p('后台队列于 2026-10-22 01:23（北京时间）完成。逐批核验四方法各 89 批、672 条相同样本，全部请求成功，每种输出 688,128 token。累计 GPU 时间从低到高为 FP8、KVarN、FP16、TurboQuant。')
 table(['方法 / KV 格式','累计 GPU-s','相对 FP16','分批 busy union / s'],[[{'fp16':'FP16','kvarn':'KVarN K4V2 G128','turboquant':'TurboQuant K4V4','fp8':'FP8 E4M3'}[m],f'{t[m]["total"]:.2f}',pct(t[m]['total'],base),f'{t[m]["busy"]:.2f}'] for m in methods],[165,95,95,160])
 p(f'FP8 累计 GPU 时间减少 {100*(1-t["fp8"]["total"]/base):.2f}%，在八个数据集上均低于 FP16；TurboQuant 增加 {100*(t["turboquant"]["total"]/base-1):.2f}%，八项均高于 FP16。KVarN 总体接近 FP16（-1.24%），但任务差异明显，不能宣称普遍加速。')
 p('计量口径：GPU-s 为各批次 CUDA kernel、GPU memcpy/memset 事件累计时长，不是请求延迟、运行时长或生产吞吐。Busy union 去除同批事件重叠；本实验二者四舍五入后接近。以下结论均限于 eager + profiler 诊断环境。')
 p('2. 实验设置与实际覆盖','h')
 p('RTX 3090（SM86），Qwen3-4B，FP16 权重与计算；并发 8，eager，关闭 prefix cache，上下文上限 32768，block size 128；固定生成 1024 token，忽略 EOS，temperature=0，seed=42。FP8 使用 FLASHINFER，后端差异包含在结果中。')
 p('覆盖 LongBench 六项各 50 条（300）、HumanEval 164 条、MATH-500 子集 208 条。FP16 按此前要求提前截止；其首轮对齐子集完整。额外 r2/r3、预检、预热、未完成 attempt 全部排除。量化运行状态的 178/178 包含链接的 FP16 89 批和自身 89 批。')
 page();p('3. 八个数据集的完整配对比较','h')
 rows=[]
 for d in sorted(t['fp16']['datasets']):
  b=t['fp16']['datasets'][d];rows.append([d,164 if d=='humaneval' else 208 if d=='math500' else 50,f'{b:.2f}']+[f'{t[m]["datasets"][d]:.2f} ({pct(t[m]["datasets"][d],b)})' for m in methods[1:]])
 table(['数据集','样本','FP16 / s','KVarN / s (变化)','TurboQuant / s (变化)','FP8 / s (变化)'],rows,[103,32,57,108,108,107])
 p('变化以对应数据集 FP16 为分母，负值表示累计 GPU 时间减少。总量变化由各数据集绝对时间加权，不能用八项百分比的简单平均替代。')
 p('KVarN：passage_retrieval_en 为 -14.99%，hotpotqa 为 -10.56%；qasper 为 +12.29%，lcc 为 +10.73%。它的收益依赖任务与输入形状，单看总体 -1.24% 会掩盖分化。')
 p('FP8：从表中可见，检索及长文任务的下降幅度较大，HumanEval/MATH 的下降较小。这是当前任务样本的观察，尚不能把差异归因于输入长度；需按实际 token 长度及调度条件分层验证。')
 p('4. 统计核验','h')
 p('直接读取 complete 且 repeat=1 的 state.json 指向的成功 attempt。四种方法 point ID 集合完全相同，逐批 case_ids 顺序一致；每批请求数与 case_ids 数相等，success=1、每条 completion_tokens=1024，总输出与 state 记录一致。')
 p('逐批核对 categories 累计值等于 cumulative_gpu_event_ms。四方法最低 phase_matched_fraction 均为 100%；此值仅代表阶段关联成功，不能证明算子分类完全正确。完整机器可读统计及原始 kernel 汇总随报告保存。')
 page();p('5. 算子拆解与优化优先级','h')
 p('读侧族合并 KV 读取、反量化、attention 及其后端 GEMM；写侧族合并旋转、归一化、KV 更新、量化、打包及 store。融合 kernel 无法继续拆成独立反量化耗时。单位为累计 GPU-s。')
 keys=['decode / attention_read_dequant_compute_combined','decode / kv_write_quant_pack_rotation_combined','decode / mlp_gemm','decode / attention_module_gemm','mixed / attention_read_dequant_compute_combined','mixed / kv_write_quant_pack_rotation_combined','prefill / attention_read_dequant_compute_combined']
 labels=['Decode 读侧/attention','Decode KV 写侧','Decode MLP GEMM','Decode attention 投影 GEMM','Mixed 读侧/attention','Mixed KV 写侧','Prefill 读侧/attention']
 table(['阶段 / 语义族']+[m for m in methods],[[label]+[f'{t[m]["cats"].get(k,0):.2f}' for m in methods] for label,k in zip(labels,keys)],[195,80,80,80,80])
 for m in methods[1:]:
  read=keys[0];write=keys[1];rv=t[m]['cats'].get(read,0);wv=t[m]['cats'].get(write,0)
  p(f'{m}：Decode 读侧 {rv:.2f} s（相对 FP16 {pct(rv,t["fp16"]["cats"][read])}，占自身总量 {rv/t[m]["total"]:.2%}）；写侧 {wv:.2f} s（{pct(wv,t["fp16"]["cats"][write])}）。')
 p('优先级一：TurboQuant decode 读侧。其大幅增加是本次最明确的额外成本线索，先检查融合 decode kernel、序列长度与活跃序列数，再做定点优化或微基准。')
 p('优先级二：KVarN 读侧与写侧。Decode 读侧增加 67.48 s，写侧增加 42.12 s；同时 MLP GEMM 减少 101.59 s、attention 投影 GEMM 减少 40.81 s。总量接近来自相互抵消，后两项不能直接当作 KV 量化算法收益。')
 p('优先级三：FP8 后端路径与公平性。FP8 的结果最好，但同时切换 FLASHINFER；要区分 dtype 与后端贡献，需补做相同 attention 后端的 FP16/FP8 配对。')
 page();p('6. FP8 分类修正与原始证据','h')
 evidence=[(k,v) for k,v in raw.items() if k[0]=='fp8' and k[1]=='other_unclassified' and 'flashinfer::' in k[2] and any(n in k[2] for n in ('BatchPrefillWithPagedKVCacheKernel','PersistentVariableLengthMergeStatesKernel'))]
 p(f'发现旧分类器没有识别 FlashInfer 的 paged-KV attention 与 merge-state kernel，合计 {sum(v for k,v in evidence):.2f} GPU-s 原归为 other_unclassified。本报告按 kernels.csv 将这部分重新归入读侧/attention，保持原始 trace、summary 和总时间不变。')
 for k,v in evidence:p(f'{v:.2f} s：'+k[2].split('<')[0], 'small')
 p('复现规则：仅对原类别 other_unclassified，名称含 flashinfer::，且含 BatchPrefillWithPagedKVCacheKernel 或 PersistentVariableLengthMergeStatesKernel 的行重新归类；沿用原 phase。未把整个 other_unclassified 归给 attention。')
 p('7. 结论的适用边界与下一步','h')
 p('本次未做任务评分，不能判断质量保持；固定长度、忽略 EOS 的输出不等同于自然生成评测。未测实际 KV 字节、尾块与元数据开销、最大并发，不能给出实测压缩率或容量收益。KVarN K4V2 与 TurboQuant K4V4 也不是等位宽比较。')
 p('四方法顺序运行且仅首轮对齐；没有量化方法的三次复测，不报告置信区间或统计显著性。调度、kernel 次数/形状、温度及频率漂移可能影响非 KV 计算。保留 mixed 阶段，不按 kernel 名称强行拆分 prefill/decode。')
 p('没有硬件 DRAM 流量、带宽或 stall counter，不能确认带宽瓶颈，也不能把算子族增量当作可消除的端到端时间。带 profiler 的请求延迟仅用于诊断，不作为生产性能排名。')
 p('下一步：先以相同后端、关闭 profiler 的独立测试验证延迟及吞吐；对 TurboQuant/KVarN 的 decode 增量做长度与调度分层分析；再配套测质量、实际 KV 占用和可承载并发。')
 p('8. 结果来源与复现','h')
 from build_completed_report import RUNS
 for m,r in RUNS.items():p(m+'：results/'+r+'/', 'small')
 p('每批来源：points/{method}/*/state.json -> attempt/traces/kernel_summary.json、kernels.csv，以及 measured/request_metrics.json。队列完成证据：results/method_queue.json。固定配置：configs/operator_profile.json；实际启动参数与身份信息位于各 run。','small')
 p('复现：.venv/bin/python scripts/build_completed_report.py。附件：validated_results.json（逐批路径与汇总）、raw_kernel_totals.csv（保留原类别）、本报告同名 Markdown。昨天的中途快照保留用于追溯，本报告替代其中队列未完成、TurboQuant 部分结果和 FP8 待启动的结论。','small')
 path=out/'KV量化实验结果_20261022.pdf'
 def footer(canvas,doc):
  canvas.setFont('Latin',8);canvas.setFillColor(colors.HexColor('#60758a'));canvas.drawString(40,24,'KV Quantization Results | 2026-10-22');canvas.drawRightString(A4[0]-40,24,str(doc.page))
 SimpleDocTemplate(str(path),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=38,bottomMargin=40,title='KV量化算子实验最终结果',author='实验结果分析').build(story,onFirstPage=footer,onLaterPages=footer)
 (out/'experiment_analysis_20261022.md').write_text('\n'.join(md))
 print(path)
