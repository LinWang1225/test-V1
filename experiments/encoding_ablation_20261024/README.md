# 独立渐进 KV 编码实验

推荐先阅读最新的[编码实验综合分析](../encoding_summary_20261027/编码实验综合分析.md)，合并历史方法与新增四位编码，包含每种实现、完整对比表和文字分析。原[实验结果汇总.md](实验结果汇总.md)保留为此前阶段记录。

本目录包含协议、独立量化器源码、论文/官方代码快照、输入清单、校准样本、码本、日志及结果。A–D 为独立量化器；E 使用 KVarN 原生量化器作为补充基线，两次运行均已完成。

- PROTOCOL.md：冻结目的、步骤、样本量、参数、存储公平性、统计与收益递减验证。
- CHANGES.md：新增文件、函数、目的与口头方案的明确差异。
- status.json：工作进程阶段、当前请求/层、更新时间及失败信息。
- background.json：后台监督器/工作进程PID。
- logs/background.log：正式运行日志；smoke日志保留调试过程。
- results/correctness_checks.json：数值/打包门槛结果。
- results/smoke/：小规模管线验证，禁止当成正式方法收益证据。
- 结果分析.md：运行时标明未完成；结束后自动替换为完整分析或明确失败报告。
- results/full/：正式逐请求指标与详细最终分析。

正确性：项目 .venv/bin/python 本目录/src/check.py
小规模：项目 .venv/bin/python -u 本目录/src/run.py --smoke
正式：项目 .venv/bin/python -u 本目录/src/run.py
后台：由src/supervise.py托管，工作进程输出重定向logs/background.log；不依赖当前终端连接。

只处理本目录新输出。若需恢复校准，可复用已经完成的NPZ；正式评估重跑会重新覆盖逐请求指标，避免误合并半条请求。最终源码与输入hash见provenance.json。

## A–E 统一入口（已完成）

- [config_A_to_E.json](config_A_to_E.json)：当前完整方法、精度、输入、存储条件和结果路径索引；保留原运行协议与provenance，不冒充重新运行的配置。
- [结果分析.md](结果分析.md)：主报告，已追加E完整对比。
- [统一结果摘要](results/unified/统一结果摘要.md)：A–E等预算关键指标。
- [统一请求指标](results/unified/request_metrics_A_to_E.csv)：90测试请求×12方法变体×15精度组合，共16200行。
- [统一汇总](results/unified/summary_A_to_E.csv)：180组合单元，含均值和请求P95。
- [统一对比统计](results/unified/comparisons_A_to_E.json)：原A–D与新增E配对CI，按来源保留。
- [完整性审计](results/unified/audit.json)：120条E轨迹、源hash、输入及预算核对。
- E配置/修改/日志/逐层原始结果：E_extension/。E于2026-10-24 20:47:23正常结束，退出码0。

新增src/build_unified.py可从冻结结果重新生成上述索引与汇总，不运行GPU、不覆盖原始实验结果。上级status.json属于A–D，E状态见E_extension/status.json。

## 2026-10-25 补充离线实验

新的完整4-bit编码/目标/多精度消融与KVarN无FP16 KV保留组已启动，见 [supplementary_experiments/README.md](supplementary_experiments/README.md)。按用户最后指示，跳过30验证请求，使用60校准＋90原测试。新队列与原A–E运行状态分开保存，旧结果保持冻结。
