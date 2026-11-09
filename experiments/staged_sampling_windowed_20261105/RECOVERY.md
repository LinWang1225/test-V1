# 后台与恢复

调度/评分：用户服务`kvarn-staged-experiment.service`。实际端到端生成与现场硬件计数：root系统服务`kvarn-sampling-dram.service`。两服务均开机恢复；用户调度器等待计数器真实探针和心跳后启动。

恢复入口：项目根目录`.venv/bin/python scripts/resume_staged_experiment.py`，默认指向本目录，不再恢复旧greedy实验。日志`logs/stage1_queue.log`，状态`stage_status.json`/`worker_status.json`，结果`analysis/结果进度.md`，累计物理流量`analysis/physical_totals.json`。

暂停所有GPU工作需同时停止用户调度和root现场计数服务；仅停止用户服务不会自动停止已交给root的当前请求。用户服务：`systemctl --user stop kvarn-staged-experiment.service`；root服务：`sudo systemctl stop kvarn-sampling-dram.service`。恢复时先启动root服务，再启动用户服务。认证信息不保存。

每题实际执行一次，物理计数与用于评分的回答共享execution_id。完成题校验指纹后跳过。中断题不能跨进程接续硬件区间，保留未完成报告并从原始输入重新运行该题；不使用已有token前缀，不将失败尝试加入有效累计流量。每次尝试有独立invocation guard；CUPTI配置只允许单pass，不重跑生成。新入口默认恢复本目录。
