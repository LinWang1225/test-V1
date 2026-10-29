# 统一4-bit、多精度重建补充实验：冻结协议

执行依据：references/实验规划_执行依据.md。2026-10-25用户确认开展，随后明确覆盖“先用30条验证筛选”的原顺序：**不运行验证请求，60条校准拟合后直接跑90条冻结测试请求**。不得再把测试结果用于调参后声称独立确认。

## 已授权范围与队列

1. KVarN无FP16 KV保留消融（单独组，先运行）：90条原测试请求，与已完成E配对。
2. E0端到端正确性门槛：短请求/0、17、35层；用于实现检查，不属于30条验证请求，不用于选超参数或收益结论。
3. 补充采集60条原校准请求的Q² Score权重；复用原X/W校准样本和归一化/旋转/码本。
4. E1、E2、E3预注册候选拟合，然后全部有效候选直接跑90测试请求；E3收益递减正则仅由校准门槛触发。
5. E4动态读取/DP与四类核心图、逐单元数据和分析报告。
6. E5在线GPU/vLLM集成按原规划为后续条件阶段，本次不改Kernel、不宣称在线收益。

后台supervise.py按子进程顺序执行，避免两组正式运行争抢GPU。开发期间的小规模门槛可能并行于不计时的KVarN误差测量；小规模结果不用于性能结论。所有旧A–E源码、输入、结果冻结，不覆盖。

## 模型与主实验控制变量

Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c，FP16模型，FP32 attention参考，TF32关闭。六LongBench任务及input_manifest.json沿用原60/30/90划分，其中30验证不执行。完整36层、8 KV heads、32 Q heads、head_dim128、decode1/8/32/64，原8192 prompt截断和参考64步生成保持不变。每条新轨迹必须与原结果SHA256一致。

主候选统一使用原A–D固定Hadamard-sign、逐channel校准mu/sigma（FP16保存），每层/head/K或V一组编码表。Gaussian为同一全局表。**所有有效K/V元素完整存4-bit**，不删token、不保留FP16尾部、不为元数据省略低位。2/3-bit必须来自同一个4-bit路径。完整4-bit payload相同，真实部署元数据分别计费；不使用给其他方法填空槽来虚构容量优势。

均值/Scale的FP16属于原生必要元数据，不属于FP16 KV旁路。源码以PyTorch展开整型码字便于误差计算，实际可序列化nibble/位平面格式独立验证；总容量是紧凑格式账本，不是Python/Torch显存峰值。

## E1编码结构与独立参考

- U_MinMax：原A，保留原全校准范围。
- U_Clipped：校准histogram上遍历上下截断分位数[0,.0001,.001,.005,.01,.02]的36组合；以同一路径2/3/4-bit均匀前缀重建损失的均值选范围。FP16 lo/hi后再算误差，不在测试上找阈值。
- Gaussian_Tree：原B全局Gaussian递归树。
- Data_Tree：原C经验分布递归树，复用冻结各seed码本。
- Independent_MSE与Independent_Uniform：每个精度独立拟合的参考；不属于渐进主基线，三份编码合计9-bit并单独记账。Lloyd局部解/有限截断网格不是严格数学全局最优上限。P_b与同编码家族的独立参考比较。

统一训练样本：原60请求，每层/head/side约983040标量，8192-bin加权充分统计。种子20261024/25/26为请求bootstrap；不把seed当独立测试请求。旧C/D码本及mu/sigma复用以保持与旧结果直接对照。新的Score权重在相同请求、同decode位置捕获；每位置32个样本的Score权重对token相同，与原样本序列对齐。

## E2 K/V目标与交叉实验

普通MSE沿用C在标准化空间的频率权重。Score-aware K使用sum_GQA(q'_c sigma_c)^2/d；Jacobian K使用sum_h p_ht²||V'_t−O'_h||²(q'_hc sigma_c)^2/d；V attention权重使用sum_h p_ht² sigma_c²。敏感性目标均只来自校准请求，权重按head归一化均值为1。V混合为0.5普通MSE+0.5归一化attention权重。

候选：Data_Tree(MSE/MSE)、Jacobian_Tree(Jac/Attn)、JacK_MSEV、ScoreK_MSEV、ScoreK_AttnV、ScoreK_MixV、JacK_MixV；均匀家族交叉包含Uniform_ScoreK_MSEV、Uniform_ScoreK_AttnV、Uniform_ScoreK_MixV。改变目标时均保持对应家族编码形式/元数据粒度一致。K/V各自学习独立表；粒度一致不表示强迫两侧共享同一质心值。

## E3多精度优化

- Data_Tree/Jacobian_Tree为原逐级分裂对照。
- Leaf4_Only：独立16簇4-bit拟合，按有序叶索引构成固定前缀；粗级重建取对应叶组条件均值。
- Joint_234：alpha2=alpha3=alpha4=1/3，交替更新逐级条件均值和15个有序叶分界点，最多60轮、相对损失改善1e-6停止。始终维持嵌套连续分区，不能用独立三份编码冒充联合优化。
- Joint_Regularized：只在校准head的归一化收益违反幅度>=0.01时触发；其余head沿用joint。lambda10、100次Adam、lr0.01，在固定树分区上优化逐级重建值。约束是校准head级曲线，不保证测试channel/token级满足；全部实际后果独立评估。若所有head均未触发，跳过该候选并记录原因。不会依据测试违反率回头启用或调正则。

## E4功能失真与动态读取

保存raw MSE、归一化平方失真D_K/D_V、相对L2，以及实际Score MSE/RMSE、Probability L1、Output relative L2/RMSE。另保存K对角Score代理和V对角attention-output代理，目标不同不横向比绝对值。15组合=3个K-only+3个V-only+9个联合；16表示参考FP16的单侧隔离测量，不代表该方案部署存储保留FP16。

逐单元Parquet保留D2/D3/D4、gain23/gain34、ratio、负增益、收益递减违反、归一化幅度、绝对/相对容差、层/head/request/step/seed/方法与channel/token编号。每head固定均匀抽4个K channel、4个V token，所有测试请求/层/位置/seed都保存，raw与functional两个目标均保留；不按误差选样本。完整单元的整体误差按全部token/channel计算，原始样本数据不冒充全量单元违反率。

违反容差=1e-10+1e-5*max(|D2|,|D3|,|D4|)，归一化幅度max(0,g34−g23,−g23,−g34)/(D2+epsilon)，额外计入负g23避免漏记非单调。gain ratio仅在g23>0时解释。

动态读取是离线两阶段原型：K以当前Q和校准head曲线分配channel精度，重建K后算attention，再用该attention与校准V曲线分配token精度。**不使用真实测试重建误差当重要性，也不用参考全精度attention分配V。**每个stage内部追加bit成本一致，强制相同目标预算；K/V stage分别预算，平均读取2/2.5/3/3.5/4 bit。物理KV head共享GQA读取费用计一次。

动态/DP预注册范围：L0/L17/L35、decode1/64、首seed、U_Clipped/Data_Tree/JacK_MSEV/ScoreK_MixV/Joint_234。它是代表性点筛查，不声称完成全36层动态控制器验收。固定精度仍完整36层/全部位置/3seed。

DP针对实际分配用的校准加权代理曲线，在固定8-item子问题遍历全部追加位预算，float64计算，**DP和贪心均采用exact固定预算**，并保存1e-12+1e-10相对容差。不是原来float32、at-most预算的混合比较。大规模贪心不因有限小实例通过而被宣称全局最优。

## KVarN无FP16 KV保留：与旧E配对的单独消融

完整128-token块直接复用原E的原生Hadamard FP16旋转、Triton Sinkhorn8、原生RTN4、FP16 absorbed metadata。未满块只对实际有效token计算PyTorch原生均衡参考和RTN，禁止用未来token或padding值参与统计；对只有一个token导致std无定义的退化块，不做方差均衡，保留原生范围量化公式。K最后维奇数时只在打包码字后加空nibble，不把padding数值带入范围拟合。

所有有效KV都有码字，不存在FP16 KV旁路；必需FP16元数据按实际尺寸计费。单token K块范围退化，码字可全零而零点携带该channel值，这是原生affine格式的退化情况，相关元数据仍完整计费，不能宣称只用4bit就无损表示浮点值。

保留中点与补零恢复两组，不训练额外表。full-tile native4解码已逐值核对；新组同90测试、同轨迹、同层/head/step，与旧E配对；不是把KVarN改造成主实验统一粒度编码器。原生Triton不能直接表达任意短尾形状，因此尾块参考分支属于消除FP16保留所需的明确适配，完整块算法不变。

## 输出、统计与图

目录按原规划组织。Parquet的distortion_curves.parquet和marginal_gains.parquet为同一组完整分片的两个数据集入口（符号链接，均包含曲线和边际字段），避免重复写盘。metrics/evaluation保留逐请求压缩CSV/Parquet、码字利用率与reference CUDA event计时。

最终按请求先聚合seed/layer/head/step，报告90请求均值、P95、配对bootstrap2000次95%CI。预设实用收益阈值5%，探索性多比较不作未经校正的显著性宣称。报告最差层/head、负收益、元数据账本、独立参考的嵌套代价。

必交付：失真曲线（含功能代理）、边际收益分布、收益递减违反图、读取预算—输出误差图、逐层热力图，及results_summary.md。编码/解码计时只指PyTorch参考函数的CUDA events，非优化Kernel/HBM/TPOT测量。
