# 渐进式 KV 编码与 Controller 结果入口

先阅读[编码实验综合分析](编码实验综合分析.md)：包含 19 个主渐进候选、2 个独立精度参考、4 个 KVarN 条件和 6 个历史等预算包装的实现、对比与分析。它区分相对 L2、绝对 MSE、有效载荷与总存储预算。

三份阶段报告已补充文字分析：

- [新增编码实验](../encoding_extension_20261026/results/full/结果分析.md)：实现、相对误差、绝对 MSE、配对置信区间和误差来源。
- [Controller 因果回放](../controller_jacobian_20261026/results/replay/结果分析.md)：因果接口、同预算收益、代理排序与幅值偏差、交叉项和局部 oracle。
- [Controller 在线闭环](../controller_jacobian_20261026/results/online/结果分析.md)：整体与逐请求结果、探索性置信区间，以及任务质量和性能结论的边界。

[all_shared_methods.csv](all_shared_methods.csv)包含去重后的 21 方法、15 个精度组合；[all_request_metrics.csv](all_request_metrics.csv)保存 28350 个请求级单元。原始汇总缺少的绝对 MSE 保持空值，不通过平方平均 RMSE 推造。

使用项目 `.venv/bin/python experiments/encoding_summary_20261027/build_reports.py` 可从冻结结果重新生成报告和图，无需 GPU。脚本修改的是此处汇总及上述三份文字报告，不修改原实验源码或原始测量；修改前的三份自动报告保留于 `auto_report_snapshots/`。合并校验与源数据 SHA256 见 [merge_audit.json](merge_audit.json)。
