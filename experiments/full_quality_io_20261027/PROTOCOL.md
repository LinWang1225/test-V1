# 全量质量评测中的逻辑读取与物理流量

本版本在 HumanEval 与 MATH-500 按原冻结代码完成后，接续 LongBench 原版全部 21 子任务、4750 请求、五策略，共 23750 次生成。模型、原始 Jacobian_Tree、输入 token、贪心与自然 EOS、输出上限和官方评分不变，只新增读取账本与硬件流量采集。旧两数据集结果明确标为旧版本，既不补造硬件数据，也不混称全数据均已物理测量。

## ReadKV 原文的测量口径

依据本地 `/home/wanglin/kvbench-3090-cu126/ReadKV.pdf` 的附录 A.1、G.2 与表 12，物理流量在已有 KV 缓存上的完整单层 eager attention 调用采集，覆盖中间与输出访问，排除缓存构造和模型其余部分。分别报告 DRAM reads、DRAM writes、总流量，区分 warm 与 cache-evicted。论文的 CUDA Graph 延迟另测，不能从 profiler 重放耗时推断服务延迟。

原文明确使用硬件计数器，但未在这几页给出 profiler 命令。本版本使用已安装的 NVIDIA Nsight Compute 2024.3.2，实现相同测量边界，不声称复现论文的具体命令或其优化内核。指标为 `dram__bytes_read.sum` 与 `dram__bytes_write.sum`，保留原始 CSV 和 ncu-rep，禁止从位宽、耗时、GPU 利用率或显存占用推算“实测字节”。

## 覆盖全部请求的实现

全部 LongBench 请求保存整个实际生成轨迹上的 K/V 逻辑读取量，以及硬件测量状态。物理流量采样固定为每请求第 17 层的第一次实际 decode attention 调用，避免按测量结果挑选样本。采样覆盖全部请求，不是抽取少量请求；但它仍是每请求单层调用实测，不能外推成全部层、全部步的总流量。

若首 token 即 EOS，实际没有 decode 调用，状态为 `no_decode_call`，不强制多生成一步。其余请求若可读取硬件计数器，则捕获实际 Q 和已有缓存，生成结束后在独立采集进程重放这个 attention 调用；先核对输出与原调用一致，再分别测 warm 与驱逐场景。重放不参与原任务生成，不改变后续 token。

Warm 先运行五次同形调用，随后仅以 NVTX `readkv_attention` 包围一次完整 attention。Evicted 在相同预热后用 128MiB 缓冲驱逐缓存，在 NVTX 范围外同步；随后采集相同完整调用。Nsight 使用 application replay、严格匹配、`cache-control none`，由应用控制缓存状态。缓存构造、驱逐、预热、测试输出和日志均不在计数范围内。

当前参考实现用 uint8 存四位码字，并通过查表模拟前缀，未实现真正按位平面读取的融合内核。物理实测因此包含当前反量化、分配、显式旋转和 workspace 的实际成本；不能预先声称 DRAM 流量会与逻辑位数同比下降。

## 逻辑读取账本

逐层、逐请求分别记录 K/V 的实际 2/3/4/16-bit 深度单位数、payload bits 与 packet bytes。K 单位为跨当前全部 token 的 head/channel，V 单位为一个 head/token 的 128 通道。K 位平面行按 ceil(T/8) 字节，V 为每位 16 字节。FP16 是连续 16-bit 数据，不额外加位平面尾部填充；旧版本 FP16 的零占位计数在新版本中纠正。

同时单列保守共享状态重载账本：当前参考解码表及 mean/scale、query_curve 使用的冻结曲线、显式旋转矩阵。它们是解析 allowance，不是 DRAM 实测；模型权重、缓存写入、分配日程和 workspace 不混入 KV 逻辑读取。物理计数则包含 attention 范围内实际跨 DRAM 边界的相关访问。

物理比较使用被采集那一次调用的对应逻辑账本，不能把单层 DRAM 字节与整段生成的逻辑总字节直接相除。不同策略若自然生成长度不同，全请求累计字节也不能直接作为等工作量节省。

## 获取失败时继续质量测评

每个 LongBench 子任务与策略边界实际运行一次小型 DRAM 计数器探针，不只读取驱动配置。若失败，保存原始日志、命令、错误码与原因，并继续全部请求的生成和官方评分；每请求的物理值为 null、状态为 unavailable。不是零流量，也不以逻辑估算替代。

硬件可用时才为每请求保存临时 Q/K/V 或整数码 fixture，warm/evicted 采集后删除该受控临时文件，原始硬件报告保留。单次 profiler 上限 120 秒；采集或输出校验失败记录原因并降级继续质量，后续边界重新探测。部分场景成功时保留 partial_conditions，报告按条件分别统计采集覆盖率。

本机已实际尝试普通与工具授权执行 ncu，收到 `ERR_NVGPUCTRPERM`；驱动为 `RmProfilingAdminOnly=1`，当前进程无 capability。`sudo -n ncu --version` 同样因需要密码失败。记录的是实际探针失败，不是未尝试就放弃。不修改驱动、不重启当前任务、不因管理员权限缺失阻断质量评测。

可选的 collector_service.py 可由有权限的管理员单独启动，仅运行固定源码、固定 profiler 命令和本目录数据 fixture；质量推理与官方评分仍由原用户运行。它不是全量评测的前提，当前流程不等待其启动，也不要求提供密码。

## 数据集边界切换

只替换调度器，保留当前 MATH-500 worker 继续执行，不杀生成进程。新调度器核验原冻结源码，在该 worker 完成后按原脚本补齐 MATH-500 全部五策略及评分。随后启动新版本 LongBench 队列，运行 CPU 账本和 CSV 解析门槛，再运行生成一致性门槛。

旧源码、数据与已完成样本不修改。新目录以链接只读引用旧冻结输入与评分源码；HumanEval/MATH-500 已完成结果在汇总中链接引用。每条 LongBench 结果使用 traffic_schema_version=2。版本边界和接管 PID 均归档，任何真实生成或评分错误仍按原协议保留现场，不能用缺失样本伪装全量完成；仅流量采集失败允许降级。

## 输出

每请求 JSON：`logical_traffic`、`physical_traffic`、`traffic_postprocess_seconds`、`request_total_seconds`。其中 physical_traffic 包含 measured/unavailable/partial/no_decode_call 状态、匹配单层逻辑量、warm/evicted DRAM 读写及原始报告路径，或失败原因。

汇总：`results/traffic_request_metrics.csv`、`traffic_summary.json`、`读取量对比.md`，与官方 `结果分析.md` 并列。报告必须给出所有请求的账本覆盖率、各条件硬件成功覆盖率和失败原因；未成功测量时只报告 unavailable，不输出虚构硬件收益。
