# 补充离线实验（2026-10-25）

执行原实验规划，并按用户最后指令**不跑30验证请求**：60校准拟合，90原测试请求正式评估，完整36层/head/decode覆盖。

- [protocol.md](protocol.md)：所有方法/目标、4-bit完整存储、条件阶段、逐单元范围与局限。
- [config.json](config.json)：冻结机器可读配置；原划分保留在input_manifest.json，validation只作来源记录、不执行。
- [CHANGES.md](CHANGES.md)：文件、函数、目的与具体适配。
- [experiment_status.json](experiment_status.json)：总后台队列与worker PID。
- [native_status.json](native_status.json)：KVarN无FP16 KV组逐请求进度。
- [suite_status.json](suite_status.json)：校准/拟合/主比较逐阶段进度。
- [logs/background.log](logs/background.log)：总日志；开发门槛日志单独保留。
- [metrics/native_nt/results_summary.md](metrics/native_nt/results_summary.md)：取消FP16保留，与旧E配对的完整结果。
- [results_summary.md](results_summary.md)：全部补充实验完成后自动生成的最终分析。

先运行KVarN无FP16 KV组，再运行端到端门槛、校准Score权重、主候选拟合、90测试、报告和图。旧A–E结果不重写。E5内核/在线阶段按规划留待离线筛选之后，不在本次冒充完成。

metrics/evaluation/每请求保存静态误差CSV.gz、码字统计、逐单元Parquet、动态和DP。最终distortion_curves.parquet/与marginal_gains.parquet/共享同一批分片，不是重复计算两次数据。

运行环境沿用项目.venv；额外pyarrow/Matplotlib依赖仅安装在本目录deps，不修改原环境包。
