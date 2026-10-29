# 四位渐进编码补充实验协议

本实验比较统一完整存储 4-bit 时的等宽嵌套编码、逐级残差编码和对数域残差编码。所有方法读取 2/3/4-bit，不保留 FP16 KV 尾部，不删 token。沿用现有旋转、逐通道校准 mean/scale 和共享参数粒度，隔离编码结构的影响。局部 Page/Chunk 归一化不在本轮引入。

## 数据与环境

沿用 Qwen3-4B 固定 revision、项目 .venv、RTX 3090 24GiB、CUDA 12.6。FP16 模型生成固定参考轨迹，FP32 attention 参考，关闭 TF32。六项 LongBench 的 60 校准和 90 测试请求使用冻结清单；按用户此前要求不执行原 30 验证请求。每请求 36 层、8 KV heads、32 Query heads、decode 1/8/32/64；三校准 bootstrap 种子在请求内平均。

旧测试结果已参与方案设计，因此本轮仍是探索性复用评估。结果不作为新数据或新模型上的独立确认。

## 编码结构

TCN_MinMax 借鉴本地 BitLadder.pdf 第 3.2 节：16 个等宽区间、floor 编码、高位截断、区间中心重建。TCN_Clipped 在同结构上使用与旧均匀基线相同的 36 组校准截断范围，最小化三精度平均 MSE。共享校准范围不同于论文每 token/head 的动态局部范围；本轮不运行 BitLadder controller 或 PlaneWeave 内核。

Residual211 借鉴 QuantSpec 的“粗量化后再量化误差”：2-bit 非对称 round-to-nearest 基础码，随后两级各 1-bit 的对称残差修正。每级幅度为裁剪后校准残差的加权绝对值均值，量化为 FP16。码字结构为 base2、residual1、residual1，逐级保留前缀；不会用独立 4-bit 残差码冒充可截断残差。

LogResidual211 借鉴 Lynx 的有符号 alpha-law 对数变换，再在变换域做相同的 2＋1＋1 逐级残差编码，各深度逆变换重建。alpha 从 1/4/16/64 在校准上选择。它不复现 Lynx 原生 Anchor 的 biased extraction、有符号修正映射、Page/Chunk 元数据或网络推测执行。对数映射在正负截断端点显式恢复到正负 1，避免浮点 log 除法改变残差符号。

新方法全部使用相同标准化重建 MSE 目标；旧 Jacobian_Tree 保留原 D 的功能加权目标。与 Data_Tree 的比较更接近 MSE 编码家族对比；不能把与 Jacobian 的全部差异只归因于编码结构。

## 评价与门槛

重算 U_Clipped、Data_Tree、Jacobian_Tree，并在首测试请求的全部单元与旧管线逐项核对 output relative L2，容差 rtol 2e-5、atol 1e-7。每请求参考生成 token 的 SHA256 必须与旧采集一致。FP32 与 FP64、旋转前后的输出需通过数值门槛。

每个方法保存 3 个 K-only、3 个 V-only、9 个联合精度组合。主指标为真实 Attention Output MSE 和相对 L2，请求 P95；同时保存 score MSE、probability L1、原始尺度 K/V MSE、收益递减违反比例。

V 误差分解使用该方案重建 K 算出的 attention：精确加权误差减去对角项得到有符号跨 token 交叉项。K 输出项为 `(phat-P) V`，V 项为 `phat (Vhat-V)`，并保存两者的内积交互项。完整联合输出误差独立计算，不能用代理相加代替。

统计单位为请求，先在请求内平均层/head/位置/种子，再报告 90 请求均值、P95 和 2000 次请求配对 bootstrap 95% CI。本轮保留所有候选，不按测试结果更改编码参数。

## 存储与读取

所有有效元素完整存储 4-bit。分别列出 payload、实际使用的 FP16 mean/scale、参数、2/3/4-bit 重建表及总字节。不会通过省略低位来支付元数据。主比较为相同 payload，非严格相同总字节。

位平面检查真实 pack/unpack，在 token 数不整除 8 时按物理行补齐。离线评估仍在 PyTorch 中模拟前缀；逻辑读取减少不代表已经测出 HBM 流量或速度收益。

## 执行顺序与结果

数值门槛 → 编码小规模拟合/评估/基线审计 → 60 校准正式拟合 → 90 测试正式评估 → 结果分析与完整性审计。只有 `results/full/analysis_complete.json` 成功，后台队列才执行 controller 的正式实验。

配置见 config.json；数值门槛见 correctness_checks.json；训练参数见 codebooks/full；逐请求数据、summary.csv、comparisons.json、storage_ledger.csv、audit.json、结果分析.md 和 encoding_comparison.png 见 results/full。源码与输入 hash 见 provenance.json。pipeline_status.json 是总队列，status.json 是编码工作进程，logs/pipeline.log 保存完整输出。任何门槛失败均停止队列并保留失败现场。

实验不修改 KVarN/vLLM 源码或旧 A–E 结果。中断可用同一源码重启 supervisor，完整请求原子落盘后跳过；源码变更不允许混合恢复。
