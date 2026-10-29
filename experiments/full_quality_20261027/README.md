# 全量任务质量评测

本目录执行五策略完整生成与官方评分：HumanEval 164、MATH-500 500、LongBench 原版全部 21 子任务 4750，每策略 5414，共 27070 次生成。

生成允许 EOS 自然结束，无最低输出长度；HumanEval 上限 2048，MATH-500 上限 8192，LongBench 使用官方逐任务上限。策略与判分细节见 [PROTOCOL.md](PROTOCOL.md)，机器配置见 [config.json](config.json)。

- [pipeline_status.json](pipeline_status.json)：总队列阶段。
- [worker_status.json](worker_status.json)：当前请求与策略。
- [quality_progress.json](quality_progress.json)：已完成官方评分的覆盖范围。
- [logs/pipeline.log](logs/pipeline.log)：完整后台日志。
- [inputs/manifest.json](inputs/manifest.json)：完整数据与输入截断清单。
- [results/结果分析.md](results/结果分析.md)：逐任务得分、自然结束与截断情况，部分阶段明确标记未完成。

使用项目 `.venv/bin/python src/supervise.py` 启动或恢复，推理用项目 .venv，官方判分用独立 .venv-eval。源码和数据 hash 冻结后不允许修改混跑；本目录与此前编码/回放/在线 pilot 完全分开。
