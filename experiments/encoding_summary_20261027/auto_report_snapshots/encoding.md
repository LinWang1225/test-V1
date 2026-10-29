# 四位渐进编码补充实验结果

已完成 90 条测试请求。全部候选完整存储 4-bit，读取 2/3/4-bit；没有 FP16 KV 尾部。统计先在请求内平均种子、层、head 和位置，再在请求间汇总。

Residual211 与 LogResidual211 是论文思想适配，不是 QuantSpec 或 Lynx 原生复现；统一共享预处理，不引入其局部 Page/Chunk 归一化。所有方法仍为固定轨迹单层反事实。

|方法|2-bit 相对 L2|3-bit 相对 L2|4-bit 相对 L2|
|---|---:|---:|---:|
|U_Clipped|0.568008|0.382779|0.307286|
|Data_Tree|0.566012|0.304408|0.150318|
|Jacobian_Tree|0.465620|0.249693|0.136101|
|TCN_MinMax|2.053959|0.696785|0.288118|
|TCN_Clipped|0.565901|0.357856|0.273968|
|Residual211|0.576968|0.309098|0.204305|
|LogResidual211|0.533893|0.374680|0.315570|

绝对 Attention Output MSE、P95、K-only/V-only、V 对角项/交叉项及 K/V 相互作用见 summary.csv。请求配对 bootstrap 95% CI 见 comparisons.json；总存储与逻辑读取分开计费，见 storage_ledger.csv。

原基线重算核对 155520 个单元，最大相对 L2 数值差 0。

旧测试数据已用于方案设计，本轮为探索性复用评估。不同编码的训练目标仍不同，不能将 Jacobian 相对 MSE 编码的差异全部解释为编码结构收益。

下一阶段固定 Jacobian_Tree，即 K 使用原 D Jacobian 树、V 使用原 D attention 树；不按本轮赢家替换。
