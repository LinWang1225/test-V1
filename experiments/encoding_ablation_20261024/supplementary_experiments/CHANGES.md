# 新增文件、函数和目的

只新增supplementary_experiments/内容，不改旧A–E测量源码和原始结果。复用函数的冻结副本与SHA256保存在references/及provenance.json；实际依赖原目录时，资产清单记录路径和hash。

|文件|函数/入口|目的|
|---|---|---|
|src/supervise.py|run/state/主入口|后台顺序执行无FP16 KVarN、校准权重、拟合、90测试、分析；保存worker PID/失败栈|
|src/native_nofp16.py|NativeCodec.quantize|完整块原生KVarN不变，尾块按有效token均衡和量化，不保留FP16 KV|
|src/native_nofp16.py|NativeCodec.reconstruct|全部有效token从原4-bit码字高位重建，中点/补零两组；没有原始tail旁路|
|src/native_nofp16.py|NativeCodec.storage|完整4-bit有效payload、奇数nibble padding、原生FP16元数据精确计费|
|src/run_native_nt.py|EExperiment.observe/run|复用旧E采集/评估口径，仅跑90测试、两恢复组、同15精度组合、逐请求轨迹校对|
|src/native_report.py|report|与旧E原始请求指标配对，生成去FP16保留消融报告|
|src/books.py|clipped/independent_uniform|校准集36种截断范围选择，统一渐进/独立均匀参考|
|src/books.py|independent|独立2/3/4-bit非均匀Lloyd参考，不冒充渐进码|
|src/books.py|joint/from_cuts/boundaries|有序叶分界点和各级重建均值交替优化，保持同4-bit叶码的前缀|
|src/books.py|regularize|按预注册校准门槛启用head级收益递减正则，保存触发与训练记录|
|src/books.py|METHODS|E1/E2/E3全部预注册K/V编码和目标组合|
|src/suite.py|fit_all|复用原校准X/W和旧C/D码本，拟合新截断/Score/Mix/联合/独立参考，保存训练和曲线信息|
|src/suite.py|code_side/decode_side|统一编码/各精度解码；独立参考单独标识，主方法必须高位前缀|
|src/suite.py|Suite.observe|采集Score校准权重；正式误差、码字频率、功能代理、逐单元曲线、动态两阶段分配|
|src/suite.py|greedy_depths|相同实际追加位预算、先父后子贪心，凹收益快路径与一般heap路径|
|src/suite.py|exact_dp_check|float64、exact固定预算、同约束DP/贪心对照及绝对相对容差|
|src/suite.py|Suite.execute/main|跳过全部30验证请求；60校准与90测试；自动端到端门槛；trajectory hash校验|
|src/check_suite.py|主入口|截断范围、联合树、前缀打包、DP反例、Parquet正确性门槛|
|src/analyze.py|main/bootstrap|流式汇总所有分片、请求配对CI、嵌套代价、码字利用率、元数据账本、核心图和最终分析|
|config.json/protocol.md|冻结配置|明确用户跳过验证、完整4-bit存储、动态诊断范围、正则条件和后续E5边界|
|input_manifest.json/inputs/|样本及资产来源|完整原划分清单，执行时过滤validation；原60校准X/W和码本文件hash|
|codebooks/|NPZ/JSON|各seed各层码本、校准profiles、剪裁分位数、联合优化历史、正则触发情况|
|metrics/|CSV/Parquet分片|重建/功能/输出误差、逐单元边际收益、DP、动态预算、存储、正确性证据|
|figures/|标准PNG|失真、功能代理、边际分布、违反率、读取预算、逐层热力图|

## 明确的计划细化与边界

- 用户最后指示覆盖原规划的30验证筛选，验证执行数0；不将数值门槛的短请求混称为验证集实验。
- 全部主KV完整4-bit，旧_eq不进入主排名；独立参考三份编码9-bit仅诊断。
- 一般MSE沿用旧C的标准化空间，使D-K+C-V真正对应旧码本；Score/Jacobian/Attn包含链式Scale。
- KVarN可变短尾使用原生PyTorch有效tile参考，singleton显式退化规则；完整块的原生编码/解码保持一致。不是声称原生Triton已支持任意短尾形状。
- 动态读取只在预注册代表层/位置/首seed测量，静态完整36层/全部位置/3seed。逐单元曲线每head固定4个K channel与4个V token，避免全量token数据不可控。
- 正则只由校准曲线触发；不在90复用测试上选超参数。联合regularizer head约束不等价于所有测试单元约束。
- 当前没有E5内核集成、HBM计数、端到端在线质量/性能测量。

## 开发门槛记录

首次suite训练smoke发现训练记录中objective字段重复，已修复字段命名为loss_target并保留失败日志；未影响正在运行的无FP16 KVarN组或任何正式suite结果。只有端到端门槛通过才允许运行正式suite。

并行加载真实模型的suite smoke曾因显存不足退出；无FP16正式组未受影响。队列已设计为其结束后顺序运行真实模型端到端门槛，再进入suite；另有小张量数值路径门槛，独立验证全部候选/动态预算/Parquet/float64 DP，不用数值fixture作实验样本。新训练器记录逐节点Lloyd实际迭代与loss历史；旧C/D复用码本的原节点历史无法追补，明确标识。

## 2026-10-25 报告补充

按用户要求，将“补充实验主要结论”和“均匀/非均匀直接及交叉对照”两项回答追加到results_summary.md与上级结果分析.md，包含数值、改善分母、独立参考和适用边界。未修改配置、训练/测量源码或原始指标，未重跑实验。
