# E：KVarN 原生量化器，无新增辅助数据的前缀重建

2026-10-24用户明确新增E基线。本扩展保持A-D训练、输入清单、结果与源码不变；只增加使用KVarN原生量化器的离线对照，不改KVarN/vLLM Kernel。

## E定义

原生K4V4/G128：Sylvester Hadamard（无随机sign，原生FP16 GEMM旋转）→ K[D,128]/V[128,D] tile → 原生Triton方差均衡8迭代（与当前KVarNConfig默认一致）→ 原生batched asymmetric RTN4 → 原生nibble packing与FP16 absorbed scale/zero point。

源文件未修改，复制到references/并冻结hash；直接调用源函数，不用另写近似均衡算法冒充KVarN。Hadamard helper从原生backend AST取出原函数执行，避免加载整个服务backend。源文件AGENTS.md已阅读；本次只读使用，不提交PR或修改其源码。

同一个q4码字，取q_b=q4>>(4-b)。两种无需额外数据的确定性恢复：

- **E（主）**：qhat4=q_b*2^(4-b)+(2^(4-b)-1)/2，缺失位区间中点。
- **E_zero（诊断）**：qhat4=q_b*2^(4-b)，缺失位补零。

后续直接代入原生反量化公式，不重新拟合scale或阈值，不重新quantize到低bit：
Khat=(qhat4*s_col_K+zp_K)*s_row_K；Vhat=(qhat4*s_row_V+zp_V)*s_col_V。
4-bit时E和E_zero逐位相同于原生解码（旋转前），然后按原生H逆变换。中点常数不需要存储码本，两个方案均无新增辅助数据。

“无新增辅助数据”保留原生FP16 scale/zero point/均衡因子，不意味着删掉原生必需元数据。不开启KVARN_RTN_QUANTILE，不做额外统计训练。3-bit是离线高位前缀重建，不声称原生vLLM有3-bit打包/读取Kernel。

## 输入、参考和尾块

复用上级inputs/manifest.json，E不训练，因此只重新采集30验证+90测试请求。36层、8 KV heads、decode1/8/32/64；input IDs与参考64步生成完全相同。每个请求重新生成的token序列必须与原A-D已保存trajectory SHA256一致，否则失败；不在E量化结果上生成下一token。

主参考在A-D同一个随机sign Hadamard的canonical空间用FP32计算；E原生使用自己的H旋转/量化，逆变换后再转入canonical空间测误差。这样不把不同参考空间当精度收益；数值误差远低于实际量化误差。

不启用first128 sink保护或首末层FP16保护。**唯一保留是未填满128-token的尾块**，按原生block flush语义保留有效FP16 token，不填充未来token、不用未来query。A-D原实验全部token量化，尾块策略有差异，必须明确披露；同时额外报告只在已量化完整tile上的D_K/D_V。完整attention误差仍含FP16尾块影响，不能说所有预处理/保护政策严格相同。

## 存储和等预算

每完整block/head：原生K/V 4-bit payload合计16384字节；原生FP16 scale/zero point/均衡因子合计1536字节（K侧2D+G，V侧D+2G，乘2字节）。尾部有效token存两个FP16向量，计4*tail*D字节/head。无新的码本、校准表、阈值、条件均值或选择mask。

主比较报告真实可序列化紧凑逻辑字节，不称原生运行时显存。原生vLLM尾池按128-token槽分配的容量上界另列，不把空闲池槽或Python张量算作数据码率。E硬件峰值未测。

**E_eq/E_zero_eq**：同请求同上下文，以B完整4-bit+全部部署元数据为总字节上限。把E原生metadata+有效FP16尾部超出的成本，通过固定head/side布局，从Bit0开始逐平面省略来支付（整字节取整）；读取时按min(请求精度,实际保留前缀精度)重建。省略数量完全由context和固定格式推导，不需要存mask。通常只影响4-bit；若原生metadata+FP16尾部成本较高，需要继续省Bit1，则3-bit读取也受影响，账本记录每个context的实际最低保留精度。Bit0省略和元数据全部计费，不宣称原生Kernel已实现该布局。

格式按实际K[D,G]和V[G,D]nibble顺序确定固定省略位置。先省Bit0，若不足再省Bit1，以此类推；若省掉全部量化payload仍不能满足预算，则明确失败，不偷偷扩大预算。原有A-D_eq对照保持冻结。

## 指标与统计

沿用D_K/D_V、raw MSE、Attention score RMSE/relative L2、probability L1、Attention Output relative L2/RMSE、全部15个K/V精度组合、K-channel/V-token边际收益递减违反率。增加quantized_span_D_K/D_V及native_fp16_tail_tokens。

每请求先平均层/head/decode，90个测试请求用于配对统计，2000次request bootstrap CI；验证仅诊断。E不需要随机训练seed。与A-D的对比以原始request_metrics.csv冻结值为准，层/head不当独立样本。

边际收益和前8-item贪心/DP诊断也保留；本扩展DP将曲线升float64避免上一轮float32累计伪gap。仍不能证明全部单元满足收益递减。

## 检查与输出

check_e.py检查原生batched/single store逐字节一致、原生4-bit解码逐值一致、midpoint/zero在4-bit一致、FP16尾部原样保留、2/3-bit有限输出与等预算不超限。smoke用同两条既有短请求，在0/17/35层完整跑采集/评估/报告并核对轨迹。

所有扩展源码/来源快照/参数/日志/指标放本目录。主实验原文件保留；扩展完成后生成E对比分析.md，并在上级结果分析.md追加E部分。失败写明原因，不用不完整结果宣称成功。

## 等预算预检查补充

对完整120请求×4位置预检查发现11个短context场景，仅省Bit0不足以支付原生元数据和FP16尾部。因此等预算扩展改为按固定顺序逐最低位平面省略；不是把E假装成相同4-bit有效payload。E/E_zero主条件不受影响，E_eq另记minimum_stored_prefix_bits。这是由存储约束强制产生的精度损失，不能隐去。256-token smoke极短，会触发更低前缀，仅用于正确性、不作正式精度结论。
