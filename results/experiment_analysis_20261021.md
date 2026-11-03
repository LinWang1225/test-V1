# KV 量化算子实验结果分析

统计快照：2026-10-21 10:07:38 CST。本报告直接读取已完成首轮批次的 state.json 与 kernel_summary.json；不使用可能滞后的阶段汇总作为计量来源。

## 1. 结论

- **FP16 与 KVarN 已完成严格对齐：89 批、每种 672 条样本。KVarN 的累计 GPU kernel 时间相对 FP16 为 -1.24%。**
- 整个实验队列尚未结束：快照时 TurboQuant 控制器及 vLLM 进程仍在运行，FP8 运行目录尚未生成。KVarN 的完成不能代表四种方法全部完成。
- 这是 eager + profiler 下的算子诊断；累计 kernel 时间比例不能解读为生产吞吐或端到端加速比。未执行准确率评分。

## 2. 设置与覆盖范围

RTX 3090（SM86），Qwen3-4B，FP16 权重与计算；客户端并发 8，eager，关闭 prefix cache，最大上下文 32768，block size 128，固定生成 1024 token、忽略 EOS、temperature=0、seed=42。方法顺序运行，预热及预检不纳入正式统计。

| 方法 | 已完成首轮批次 | 已完成样本 | 状态 |
|---|---:|---:|---|
| fp16 | 89/89 | 672/672 | 按此前要求提前截止；当前对齐子集完整 |
| kvarn | 89/89 | 672/672 | complete |
| turboquant | 43/89 | 308/672 | 运行中，部分结果 |
| fp8 | 0/89 | 0/672 | 队列待启动 |

FP16 原始运行保留额外 r2/r3 代表复测；本报告仅使用 r1。目标对齐集为 LongBench 六项各 50 条（共 300）、HumanEval 164 条、MATH-500 的 208 条；不是完整 MATH-500。status.json 中 KVarN 的 178/178 包含链接的 FP16 89 批与自身 89 批。

## 3. FP16 与 KVarN：完整对齐结果

下表单位为秒，均为各批次 **累计 GPU kernel 时间之和**；不是请求耗时或实验运行时长。

| 数据集 | 样本 | FP16 GPU-s | KVarN GPU-s | 变化 |
|---|---:|---:|---:|---:|
| gov_report | 50 | 312.43 | 311.78 | -0.21% |
| hotpotqa | 50 | 413.48 | 369.81 | -10.56% |
| humaneval | 164 | 307.36 | 321.12 | +4.48% |
| lcc | 50 | 154.11 | 170.65 | +10.73% |
| math500 | 208 | 372.96 | 385.44 | +3.35% |
| passage_retrieval_en | 50 | 412.94 | 351.05 | -14.99% |
| qasper | 50 | 180.83 | 203.06 | +12.29% |
| trec | 50 | 225.25 | 236.99 | +5.21% |
| **合计** | **672** | **2379.36** | **2349.90** | **-1.24%** |

全 GPU busy union 分批相加：FP16 2379.36 s，KVarN 2349.90 s。此值排除同一批次 kernel 重叠的重复计时，但仍不是端到端墙钟时间。两种方法正式输出分别为 688,128 与 688,128 token。

### 3.1 语义算子族分解

融合与独立实现按相同语义合并，避免把 FP16 attention 的类别缺失当成量化新增开销。单位为 GPU-s。

| 阶段 / 算子族 | FP16 | KVarN | 增量 | KVarN占总量 |
|---|---:|---:|---:|---:|
| decode / attention_read_dequant_compute_combined | 630.61 | 698.09 | +67.48 | 29.71% |
| decode / kv_write_quant_pack_rotation_combined | 10.53 | 52.65 | +42.12 | 2.24% |
| mixed / kv_write_quant_pack_rotation_combined | 1.12 | 31.37 | +30.25 | 1.33% |
| mixed / other_unclassified | 0.03 | 10.92 | +10.89 | 0.46% |
| decode / other_unclassified | 6.14 | 16.67 | +10.53 | 0.71% |
| mixed / copy_scatter_gather_other | 0.03 | 9.03 | +9.00 | 0.38% |
| decode / gpu_memcpy | 0.53 | 5.26 | +4.73 | 0.22% |
| mixed / gpu_memcpy | 0.00 | 3.18 | +3.17 | 0.14% |
| prefill / kv_write_quant_pack_rotation_combined | 0.24 | 3.30 | +3.06 | 0.14% |
| prefill / mlp_gemm | 35.92 | 36.40 | +0.48 | 1.55% |
| prefill / attention_module_gemm | 13.09 | 13.24 | +0.15 | 0.56% |
| prefill / normalization_rope_activation | 3.23 | 3.24 | +0.01 | 0.14% |
| decode / gpu_memset | 1.87 | 1.56 | -0.31 | 0.07% |
| mixed / normalization_rope_activation | 15.38 | 14.75 | -0.63 | 0.63% |
| mixed / attention_module_gemm | 62.24 | 59.87 | -2.37 | 2.55% |
| prefill / attention_read_dequant_compute_combined | 25.59 | 23.11 | -2.48 | 0.98% |
| mixed / mlp_gemm | 170.91 | 164.58 | -6.33 | 7.00% |
| decode / normalization_rope_activation | 71.00 | 61.81 | -9.19 | 2.63% |
| decode / gemm_unattributed | 95.14 | 82.30 | -12.84 | 3.50% |
| decode / copy_scatter_gather_other | 22.45 | 8.24 | -14.22 | 0.35% |
| mixed / attention_read_dequant_compute_combined | 141.13 | 118.55 | -22.58 | 5.04% |
| decode / attention_module_gemm | 312.95 | 272.14 | -40.81 | 11.58% |
| decode / mlp_gemm | 758.19 | 656.60 | -101.59 | 27.94% |

读侧族 attention_read_dequant_compute_combined 包括融合 KV 读取/反量化/attention、独立反量化及 attention 后端 GEMM；写侧族 kv_write_quant_pack_rotation_combined 包括旋转/归一化/Sinkhorn、KV 更新 GEMM、量化/打包/store。attention_module_gemm 为模块投影等单列计算。

### 3.2 优化判断

总量接近并不表示 KV 路径没有额外成本：KVarN 的 decode 读侧增加 67.48 GPU-s、decode 写侧增加 42.12 GPU-s，同时 decode MLP/GEMM 减少 101.59 GPU-s、attention 模块 GEMM 减少 40.81 GPU-s，多项变化相互抵消。后两项属于非 KV 核心计算，需进一步用调度、kernel 次数/形状及频率记录解释，不能直接归因于量化算法提速。

数据集表现也明显分化：passage_retrieval_en 与 hotpotqa 分别减少 14.99% 和 10.56%，qasper 与 lcc 则增加 12.29% 和 10.73%。因此合计的 -1.24% 不适合作为跨任务通用收益结论。

- Decode `attention_read_dequant_compute_combined`：630.61 → 698.09 GPU-s，增量 +67.48 s（+10.70%），占 KVarN 总累计时间 29.71%。
- Decode `kv_write_quant_pack_rotation_combined`：10.53 → 52.65 GPU-s，增量 +42.12 s（+399.85%），占 KVarN 总累计时间 2.24%。
- 优先沿 decode 读侧融合 kernel 和写侧旋转/更新/量化路径排查。以上是定位线索，不能把整个算子族增量归因于反量化，也不能承诺等量端到端收益。
- MLP/GEMM 的绝对占比仍应纳入优化上限判断；KV 压缩不会自动缩短模型全部计算。

## 4. TurboQuant：已完成配对批次的阶段结果

只比较快照时已完成的相同 r1 批次；不把部分数据扩推到 672 条全集。

| 数据集 | 配对批次 | 样本 | FP16 GPU-s | TurboQuant GPU-s | 变化 |
|---|---:|---:|---:|---:|---:|
| gov_report | 7 | 50 | 312.43 | 421.67 | +34.97% |
| hotpotqa | 7 | 50 | 413.48 | 523.08 | +26.50% |
| humaneval | 1 | 8 | 14.42 | 16.50 | +14.41% |
| lcc | 7 | 50 | 154.11 | 210.84 | +36.81% |
| passage_retrieval_en | 7 | 50 | 412.94 | 495.51 | +20.00% |
| qasper | 7 | 50 | 180.83 | 255.51 | +41.30% |
| trec | 7 | 50 | 225.25 | 303.23 | +34.62% |
| **当前配对子集合计** | **43** | **308** | **1713.47** | **2226.35** | **+29.93%** |

相同 43 批上 KVarN 累计为 1658.38 GPU-s，TurboQuant 相对 KVarN 为 +34.25%。此结论仅适用于该子集。
- TurboQuant decode `attention_read_dequant_compute_combined`：FP16 548.38 → TurboQuant 1234.23 GPU-s（+125.07%）。
- TurboQuant decode `kv_write_quant_pack_rotation_combined`：FP16 5.73 → TurboQuant 12.13 GPU-s（+111.62%）。

## 5. 可靠性与解释边界

- 所有纳入结果均须 state=complete、repeat=1，忽略未完成批次及历史失败 attempt；相同 point ID 还逐一核对 case_ids。
- 不同方法动态调度、实际驻留序列数、混合 prefill/decode 比例可能不同；阶段结果保留 mixed/unknown，不强行拆分。
- 纳入批次的最低 phase_matched_fraction：FP16 100.0000%、KVarN 100.0000%、TurboQuant 100.0000%。阶段匹配率不能证明类别归因或性能完全无偏。
- 没有三次量化方法复测，故不提供置信区间或统计显著性结论。顺序运行也可能受温度、频率漂移影响。
- KVarN 为 K4V2 G128，TurboQuant 为 K4V4，不是等存储位宽比较；本次未测量真实 KV 字节、元数据/尾块开销或容量上限，不能据此给出实测压缩率。
- 没有硬件 DRAM counter、带宽或 stall 测量，无法确认瓶颈属于显存带宽还是算力。
- FP8 预检可用不等于正式完成；计划在 SM86 使用 FLASHINFER，其后端差异需在正式结果中说明。
- 固定长度并忽略 EOS 用于算子对齐；没有 LongBench 官方评分、HumanEval 执行评分或 MATH 等价判分，不能判断质量保持。

## 6. 后续工作

1. 待 TurboQuant 与 FP8 队列实际完成后，重新冻结 89 批共同集合，补齐四方法表。
2. 针对增量最大的 decode 算子族检查 raw kernel、调度轨迹及后端；若进一步优化，再用独立、关闭 profiler 的测试验证端到端收益。
3. 如需判断压缩价值，另测实际 KV 占用、最大可承载并发及任务准确率。

## 7. 结果来源

路径均相对此项目根目录：

- `fp16`：`results/operator_profile_20261019_210036/`（`points/fp16/*/state.json` → `attempt/traces/kernel_summary.json`，原始 trace 与 kernels.csv 同目录）。
- `kvarn`：`results/operator_kvarn_firstpass_20261020_153745/`（`points/kvarn/*/state.json` → `attempt/traces/kernel_summary.json`，原始 trace 与 kernels.csv 同目录）。
- `turboquant`：`results/operator_turboquant_firstpass_20261020_queue/`（`points/turboquant/*/state.json` → `attempt/traces/kernel_summary.json`，原始 trace 与 kernels.csv 同目录）。
- 队列：`results/method_queue.json`、`results/method_queue.log`。
- 固定配置：`configs/operator_profile.json`；各运行另存 `config.json`、`plan.json`、`identity.json` 及服务器启动参数。
- 既有语义族对照：`results/operator_kvarn_firstpass_20261020_153745/summary/operator_comparison.json`。
