# 文件、函数与目的

所有本次新增文件均在本目录；未修改项目其他实验、KVarN、vLLM、Transformers或模型权重。references/any-precision-llm 是只读上游代码快照（未调用其CUDA引擎）。本目录在非Git工作区，使用provenance.json记录源码hash；不存在可用的项目git diff。

|文件|函数/入口|目的|
|---|---|---|
|src/codec.py|hadamard|生成固定正交变换，独立于KVarN|
|src/codec.py|lloyd|一维加权Lloyd更新与停止条件|
|src/codec.py|fit_tree|Gaussian递归树、经验/敏感性加权种子与逐簇二分；保存各前缀表|
|src/codec.py|encode/decode|生成唯一4-bit路径；只用高位前缀解码|
|src/codec.py|pack4/unpack4|nibble存储逐位正确性|
|src/codec.py|pack_planes/unpack_planes|实际紧凑位平面与固定Bit0省略布局；验证等预算大小|
|src/codec.py|allocation_check|等成本前缀预算精确DP与贪心比较|
|src/run.py|prepare|读取已有原始请求及LongBench context；去重、长度分层、冻结180条输入和请求划分|
|src/run.py|Experiment.__init__|加载FP16 Hugging Face模型并注册只读采集接口；不修改任何环境库|
|src/run.py|Experiment.capture|分块prefill、参考decode轨迹、保存校准样本或逐请求指标|
|src/run.py|Experiment.observe|获取量化前Q/K/V，计算Jacobian权重、所有精度组合、误差/收益递减/DP与数值诊断；等预算省位反事实|
|src/run.py|Experiment.train|只读校准样本；固定mean/Scale；按请求bootstrap拟合C/D；保存全部部署数组|
|src/run.py|Experiment.load_books|将冻结码本加载GPU，不利用验证/测试更新|
|src/run.py|Experiment.storage_ledger|计算payload、真实元数据、等预算实际总字节及容量|
|src/run.py|Experiment.run/status|校准→拟合→验证/测试→报告；保存进度/失败栈|
|src/check.py|脚本入口|打包/前缀/Gaussian/正交/Jacobian/贪心反例正确性门槛|
|src/report.py|bootstrap/report|按请求配对CI、D/attention汇总、分层结果、等预算结果、DP及最终分析Markdown|
|src/supervise.py|脚本入口|持久后台子进程、记录退出码；完成/失败均更新根目录结果分析.md|
|PROTOCOL.md|全文|冻结实验目的、方法、样本/参数、统计与存储、理论条件和边界|
|references/|论文、官方源码、环境|依据与复现快照；上游提交hash、各文件hash见provenance.json|
|inputs/|manifest.json|全部输入token IDs、来源hash、文档hash、校准/验证/测试划分|
|calibration/|full/*.npz|真实量化前样本及敏感性；完整KV不长期保存|
|codebooks/|full/*.npz|各层/head/side/seed的mean/Scale、范围、质心与阈值|
|results/|full/*|逐层/head原始测量、轨迹、数值门槛、统计和分析|
|logs/|smoke*.log/background.log|小规模调试与正式完整日志|

## 与上轮口头方案的落实与明确差异

1. 遵循用户后续要求，A改为独立均匀INT4；不导入已有原生缓存解析器，不以KVarN缓存反量化作输入。
2. 不保留额外补零重建；固定中点基线，减少非核心变体。
3. ReadKV代码未找到，B为论文附录重实现；Any-Precision代码固定ee5288962ad5934c3988ae6c9a73e3aa6cd76058，核对quantize.py::_increment_group与四簇seed调用，使用自写加权Lloyd适配KV；不运行原权重敏感性/模型量化代码。
4. 为可行训练成本，经验分布采用8192-bin充分统计近似。三个seed对应请求bootstrap，A/B确定性计算一次，不能把三个seed当三倍独立请求。
5. 不长期落盘完整180条全部KV；逐请求在线采集后立即离线反事实评估。保存确定性输入/参考输出token序列、校准样本和所有指标，可重新采集复算；约百GB完整激活不保存。
6. 等总字节条件不仅算容量，额外执行固定位置最低位省略，保证同上下文和同总存储上限。此条件在结果中以_eq标识。
7. 第一轮已实现所有核心指标与同预算比较；不执行任务生成质量评分、Kernel修改、真实内存流量和端到端速度测试。
8. 数值门槛调试时发现DP返回numpy标量不能JSON序列化，已在allocation_check显式转换float并完整重跑smoke；调试日志保留，不把早期失败作实验数据。

## 完成后结果核查

新增 results/audit.json（覆盖、数值、源码哈希、预算及逐层/任务一致性）、results/full/allocation_audit.json（DP分组原始gap统计，含浮点误差解释）。在两份结果分析.md开头补入核查结论与层间退化、DP数值局限。未修改已运行的训练/测量源码，未重新启动GPU实验。

## E完成后的统一配置与结果更新

新增config_A_to_E.json（完整A–E条件/路径索引），新增src/build_unified.py（检查原结果/源码/轨迹hash并合并请求指标与摘要），新增results/unified/下统一CSV、JSON、摘要Markdown及audit.json。更新README.md、主结果分析.md以及原K/V和逐层报告中的E状态说明。E原生来源、代码函数说明在E_extension/CHANGES_E.md。

原PROTOCOL.md、原训练/测量src/*.py（新增整合脚本除外）、原results/full/的A–D CSV、码本、inputs/manifest.json及原provenance.json保持不变。原配置和原始数据不被改写为“包含E”的历史运行；使用新的统一索引准确表示两次已完成运行。A–D原始request_metrics SHA256与E启动前记录一致。
