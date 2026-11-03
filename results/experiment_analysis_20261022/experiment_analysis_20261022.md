# KV 量化算子实验：最终结果

2026-10-22｜四方法首轮对齐结果｜RTX 3090 / Qwen3-4B

## 1. 核心结论与完成情况

后台队列于 2026-10-22 01:23（北京时间）完成。逐批核验四方法各 89 批、672 条相同样本，全部请求成功，每种输出 688,128 token。累计 GPU 时间从低到高为 FP8、KVarN、FP16、TurboQuant。

| 方法 / KV 格式 | 累计 GPU-s | 相对 FP16 | 分批 busy union / s |
|---|---|---|---|
| FP16 | 2379.36 | +0.00% | 2379.36 |
| KVarN K4V2 G128 | 2349.90 | -1.24% | 2349.90 |
| TurboQuant K4V4 | 2989.43 | +25.64% | 2989.43 |
| FP8 E4M3 | 1887.66 | -20.67% | 1887.66 |

FP8 累计 GPU 时间减少 20.67%，在八个数据集上均低于 FP16；TurboQuant 增加 25.64%，八项均高于 FP16。KVarN 总体接近 FP16（-1.24%），但任务差异明显，不能宣称普遍加速。

计量口径：GPU-s 为各批次 CUDA kernel、GPU memcpy/memset 事件累计时长，不是请求延迟、运行时长或生产吞吐。Busy union 去除同批事件重叠；本实验二者四舍五入后接近。以下结论均限于 eager + profiler 诊断环境。

## 2. 实验设置与实际覆盖

RTX 3090（SM86），Qwen3-4B，FP16 权重与计算；并发 8，eager，关闭 prefix cache，上下文上限 32768，block size 128；固定生成 1024 token，忽略 EOS，temperature=0，seed=42。FP8 使用 FLASHINFER，后端差异包含在结果中。

覆盖 LongBench 六项各 50 条（300）、HumanEval 164 条、MATH-500 子集 208 条。FP16 按此前要求提前截止；其首轮对齐子集完整。额外 r2/r3、预检、预热、未完成 attempt 全部排除。量化运行状态的 178/178 包含链接的 FP16 89 批和自身 89 批。

## 3. 八个数据集的完整配对比较

| 数据集 | 样本 | FP16 / s | KVarN / s (变化) | TurboQuant / s (变化) | FP8 / s (变化) |
|---|---|---|---|---|---|
| gov_report | 50 | 312.43 | 311.78 (-0.21%) | 421.67 (+34.97%) | 232.24 (-25.67%) |
| hotpotqa | 50 | 413.48 | 369.81 (-10.56%) | 523.08 (+26.50%) | 276.78 (-33.06%) |
| humaneval | 164 | 307.36 | 321.12 (+4.48%) | 354.04 (+15.19%) | 293.11 (-4.64%) |
| lcc | 50 | 154.11 | 170.65 (+10.73%) | 210.84 (+36.81%) | 133.75 (-13.21%) |
| math500 | 208 | 372.96 | 385.44 (+3.35%) | 425.55 (+14.10%) | 357.82 (-4.06%) |
| passage_retrieval_en | 50 | 412.94 | 351.05 (-14.99%) | 495.51 (+20.00%) | 262.90 (-36.34%) |
| qasper | 50 | 180.83 | 203.06 (+12.29%) | 255.51 (+41.30%) | 154.07 (-14.79%) |
| trec | 50 | 225.25 | 236.99 (+5.21%) | 303.23 (+34.62%) | 176.99 (-21.43%) |

变化以对应数据集 FP16 为分母，负值表示累计 GPU 时间减少。总量变化由各数据集绝对时间加权，不能用八项百分比的简单平均替代。

KVarN：passage_retrieval_en 为 -14.99%，hotpotqa 为 -10.56%；qasper 为 +12.29%，lcc 为 +10.73%。它的收益依赖任务与输入形状，单看总体 -1.24% 会掩盖分化。

FP8：从表中可见，检索及长文任务的下降幅度较大，HumanEval/MATH 的下降较小。这是当前任务样本的观察，尚不能把差异归因于输入长度；需按实际 token 长度及调度条件分层验证。

## 4. 统计核验

直接读取 complete 且 repeat=1 的 state.json 指向的成功 attempt。四种方法 point ID 集合完全相同，逐批 case_ids 顺序一致；每批请求数与 case_ids 数相等，success=1、每条 completion_tokens=1024，总输出与 state 记录一致。

逐批核对 categories 累计值等于 cumulative_gpu_event_ms。四方法最低 phase_matched_fraction 均为 100%；此值仅代表阶段关联成功，不能证明算子分类完全正确。完整机器可读统计及原始 kernel 汇总随报告保存。

## 5. 算子拆解与优化优先级

读侧族合并 KV 读取、反量化、attention 及其后端 GEMM；写侧族合并旋转、归一化、KV 更新、量化、打包及 store。融合 kernel 无法继续拆成独立反量化耗时。单位为累计 GPU-s。

| 阶段 / 语义族 | fp16 | kvarn | turboquant | fp8 |
|---|---|---|---|---|
| Decode 读侧/attention | 630.61 | 698.09 | 1384.99 | 312.15 |
| Decode KV 写侧 | 10.53 | 52.65 | 26.30 | 34.21 |
| Decode MLP GEMM | 758.19 | 656.60 | 652.66 | 659.04 |
| Decode attention 投影 GEMM | 312.95 | 272.14 | 270.44 | 273.83 |
| Mixed 读侧/attention | 141.13 | 118.55 | 120.78 | 128.90 |
| Mixed KV 写侧 | 1.12 | 31.37 | 2.75 | 2.07 |
| Prefill 读侧/attention | 25.59 | 23.11 | 23.15 | 26.02 |

kvarn：Decode 读侧 698.09 s（相对 FP16 +10.70%，占自身总量 29.71%）；写侧 52.65 s（+399.85%）。

turboquant：Decode 读侧 1384.99 s（相对 FP16 +119.63%，占自身总量 46.33%）；写侧 26.30 s（+149.72%）。

fp8：Decode 读侧 312.15 s（相对 FP16 -50.50%，占自身总量 16.54%）；写侧 34.21 s（+224.79%）。

优先级一：TurboQuant decode 读侧。其大幅增加是本次最明确的额外成本线索，先检查融合 decode kernel、序列长度与活跃序列数，再做定点优化或微基准。

优先级二：KVarN 读侧与写侧。Decode 读侧增加 67.48 s，写侧增加 42.12 s；同时 MLP GEMM 减少 101.59 s、attention 投影 GEMM 减少 40.81 s。总量接近来自相互抵消，后两项不能直接当作 KV 量化算法收益。

优先级三：FP8 后端路径与公平性。FP8 的结果最好，但同时切换 FLASHINFER；要区分 dtype 与后端贡献，需补做相同 attention 后端的 FP16/FP8 配对。

## 6. FP8 分类修正与原始证据

发现旧分类器没有识别 FlashInfer 的 paged-KV attention 与 merge-state kernel，合计 467.07 GPU-s 原归为 other_unclassified。本报告按 kernels.csv 将这部分重新归入读侧/attention，保持原始 trace、summary 和总时间不变。

301.77 s：void flashinfer::BatchPrefillWithPagedKVCacheKernel

153.69 s：void flashinfer::BatchPrefillWithPagedKVCacheKernel

11.60 s：void flashinfer::PersistentVariableLengthMergeStatesKernel

复现规则：仅对原类别 other_unclassified，名称含 flashinfer::，且含 BatchPrefillWithPagedKVCacheKernel 或 PersistentVariableLengthMergeStatesKernel 的行重新归类；沿用原 phase。未把整个 other_unclassified 归给 attention。

## 7. 结论的适用边界与下一步

本次未做任务评分，不能判断质量保持；固定长度、忽略 EOS 的输出不等同于自然生成评测。未测实际 KV 字节、尾块与元数据开销、最大并发，不能给出实测压缩率或容量收益。KVarN K4V2 与 TurboQuant K4V4 也不是等位宽比较。

四方法顺序运行且仅首轮对齐；没有量化方法的三次复测，不报告置信区间或统计显著性。调度、kernel 次数/形状、温度及频率漂移可能影响非 KV 计算。保留 mixed 阶段，不按 kernel 名称强行拆分 prefill/decode。

没有硬件 DRAM 流量、带宽或 stall counter，不能确认带宽瓶颈，也不能把算子族增量当作可消除的端到端时间。带 profiler 的请求延迟仅用于诊断，不作为生产性能排名。

下一步：先以相同后端、关闭 profiler 的独立测试验证延迟及吞吐；对 TurboQuant/KVarN 的 decode 增量做长度与调度分层分析；再配套测质量、实际 KV 占用和可承载并发。

## 8. 结果来源与复现

fp16：results/operator_profile_20261019_210036/

kvarn：results/operator_kvarn_firstpass_20261020_153745/

turboquant：results/operator_turboquant_firstpass_20261020_queue/

fp8：results/operator_fp8_firstpass_20261020_queue/

每批来源：points/{method}/*/state.json -> attempt/traces/kernel_summary.json、kernels.csv，以及 measured/request_metrics.json。队列完成证据：results/method_queue.json。固定配置：configs/operator_profile.json；实际启动参数与身份信息位于各 run。

复现：.venv/bin/python scripts/build_completed_report.py。附件：validated_results.json（逐批路径与汇总）、raw_kernel_totals.csv（保留原类别）、本报告同名 Markdown。昨天的中途快照保留用于追溯，本报告替代其中队列未完成、TurboQuant 部分结果和 FP8 待启动的结论。
