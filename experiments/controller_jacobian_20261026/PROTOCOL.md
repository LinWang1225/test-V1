# Jacobian 编码 Controller 实验协议

编码固定为旧 Jacobian_Tree：K 使用原 D 的 output Jacobian 树，V 使用原 D 的 attention 树，校准种子 20261024；不是 JacK_MixV，也不按新编码实验的赢家替换。完整存储 4-bit，仅控制 2/3/4-bit 读取。正式执行排在编码实验完整分析之后。

## 因果接口

K 决策只接收当前 Query、冻结的每通道原始尺度 MSE 曲线和预算。K 单位是一个 head/channel 在全部当前 token 上的前缀。随后使用所选 K 算出 phat；V 决策只接收 phat、冻结 V 曲线和预算。V 单位是一个 head/token 的全部 128 通道。

失真曲线用原 60 校准激活和冻结 Jacobian 码本计算，按原始尺度存储，不再重复乘标准化 scale。K 权重为共享 Query heads 的 q²/d 之和；V 权重为共享 heads 的 phat² 之和，并将逐通道 V 失真相加。当前完整 KV、未读取低位、FP16 attention、真实输出及未来 Query 不进入 policy.py。

每阶段使用独立预算；组内每增加一位成本相同，K 为 ceil(T/8) 字节，V 为 16 字节。以代理误差下降/新增字节排序，细化必须满足前置依赖。非收益递减时使用满足依赖的堆贪心，不承诺最优；单元测试包含与精确 DP 的一致性检查和失败反例。

## 因果回放

90 测试请求，参考轨迹与旧 FP16 token hash 一致；层 0/17/35，位置 1/64。比较 uniform、static、query_only、query_curve，分别在每侧平均 2/2.5/3/3.5/4-bit 预算下读取。

uniform 使用尽量均匀分散的固定混合深度；static 仅使用校准原始失真曲线；query_only 使用当前 q² 或 phat² 和统一 4 倍递减的理想失真形状；query_curve 使用当前敏感性和真实校准曲线。query_only 是去掉经验失真形状的消融，不宣称复现 Fathom 全系统。所有策略按相同分侧预算核对实际 bit 数与逐行字节填充成本。

评价真实 output MSE、relative L2、请求 P95、K score 代理与真实误差、V 代理/实际对角项/完整加权误差/交叉项。分别报告跨同预算策略的代理秩相关、成对排序正确率及真实 V 误差/预测误差的中位数与 P95。请求配对 bootstrap 2000 次，95% CI。

首请求第 0 层位置 1 增加 head0 四通道或四 token 的全部同预算局部枚举，其他单位固定 3-bit，记录代理最优与真实输出局部最优的 regret。这个 oracle 不代表整个缓存的全局最优。

回放阶段不把量化输出传给后续层；不使用后续 token 计算当前策略。它用于隔离决策效果，不能替代在线质量验证。

## 在线闭环试验

因果回放分析完成后，先以 256-token smoke prompt 验证 36 层、8 个 decode 步的在线替换接口。通过后，从冻结测试清单每任务取第一条和最后一条，共 12 条，保留原上下文，生成 64 步。四策略为 FP16、完整读取 Jacobian K4V4、uniform K3V3、query_curve 平均 K3V3。prefill 为 FP16，decode 的全部 36 层实际使用所选 KV 的 attention 输出，并传入后续层与后续 token。各策略独立生成，FP16 分支必须复现旧参考 token。

记录生成文本与 token IDs、首次分歧位置、与 FP16 的逐位置 token 一致率，以及位置 1/8/32/64 的局部 attention MSE。轨迹一致率不是任务准确率，64 步试验不作官方 LongBench 任务得分结论。

本适配器保留 FP16 源缓存并重新计算冻结码字、模拟前缀选择；它验证在线因果与误差传播，不是实际压缩缓存内核。墙钟、TPOT、显存只标为参考适配器开销，不能当作吞吐加速或容量收益。真实位平面融合内核、HBM 计量和官方任务评分在后续系统阶段执行。

## 产出与失败处理

status.json 保存当前阶段；results/replay 和 results/online 分别保存正式回放与在线试验；results/smoke_replay、results/smoke_online 只用于门槛。每阶段生成 complete.json 与 analysis_complete.json；未完成的阶段不报告完成。总队列与源码 hash 在相邻 encoding_extension_20261026 目录。任何数值、预算、轨迹或完整性门槛失败均停止后续阶段。
