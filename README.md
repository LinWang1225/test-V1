# Qwen3 KV 编码实验

多服务器运行：[代码同步、模型路径配置、样本分片及结果合并](configs/sync/README.md)。
当前论文对照源码位于`experiments/paper_k4v4_quality_20261109`。

实验归档：[总实验记录](总实验记录.md)按54项实验逐项记录目的、环境、数据、配置、对比方法、指标和结论，包含前置项目的33项历史离线与机制实验；[综合分析](实验综合分析_20261109.md)保留跨阶段结果表。已完成、暂停与待运行阶段分别标明。

当前实验：[thinking＋固定种子采样的端到端编码比较](experiments/staged_sampling_windowed_20261105/PROTOCOL.md)。沿用464个样本，包含全部21个LongBench子任务，不按数据集截断输入或设置输出上限。

进度：[质量与累计物理流量](experiments/staged_sampling_windowed_20261105/analysis/结果进度.md)。完整生成物理计数在真实端到端生成现场执行；同次回答用于质量评分，逐题与汇总分别保存，不以单步采样外推。

后台与断电恢复：[恢复说明](experiments/staged_sampling_windowed_20261105/RECOVERY.md)、[systemd配置](configs/systemd/README.md)。普通用户调度和评分，root执行现场计数的端到端生成。

旧greedy/non-thinking实验已暂停，结果、源码及断点保留在`experiments/staged_algorithm_20261102`，不与新实验混合排名。
