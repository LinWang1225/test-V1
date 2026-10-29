# 后续全量质量测评与读取量对比

从 LongBench 原版全部 21 子任务开始使用新版本，覆盖全部请求。HumanEval 与当前 MATH-500 保持冻结版本及全部五策略，完成后自动按数据集边界切换。

读取量按 [ReadKV 测量协议](PROTOCOL.md)区分整段生成的 K/V 逻辑账本与每请求匹配单层调用的 warm/evicted DRAM 实测。实际采集失败会记录原因并继续生成、官方评分，缺失不写成零。

- [handoff_status.json](handoff_status.json)：当前数据集完成与切换状态。
- [pipeline_status.json](pipeline_status.json)：新版本启动后的 LongBench 队列。
- [results/结果分析.md](results/结果分析.md)：官方任务质量。
- [results/读取量对比.md](results/读取量对比.md)：逻辑与实测对比、覆盖率及失败原因。
- [hardware_capability.json](hardware_capability.json)：最近一次实际硬件探针结果。

`src/check_io.py` 为 CPU-only 账本和硬件 CSV 证据校验；数据集边界运行 `src/check_io_model.py` 验证插桩不改变已有参考生成。新版本不修改旧实验源码，不重跑已完成的 HumanEval/MATH 样本；旧评分链接在此处汇总。
