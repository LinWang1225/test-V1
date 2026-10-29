# 独立渐进 KV 编码消融协议（冻结版）

用户于 2026-10-24 明确授权：脱离 KVarN，写新量化器，后台运行，全部协议/代码/修改说明/结果归档，并生成分析 Markdown。本目录是独立实验；不导入、不修改 KVarN 或 vLLM 源码/Kernel。

## 目的与假设

比较同一真实 FP16 激活输入下 A/B/C/D 的2/3/4-bit前缀重建。C 对 B 检验真实分布适配价值；D 对 C/B 检验 attention 敏感性价值。检查实测收益递减，不能继承 Gaussian 码本的分配保证。仅离线单层反事实，不推断任务质量、端到端速度、真实 HBM 流量或跨层误差传播。

## 方法

- A：独立均匀 affine INT4 基线。统一旋转及校准标准化后，每层/head/K或V共享校准最小最大范围。4-bit用最近整数编码；2/3-bit用原始16级码字区间中点解码。它不复现 KVarN 原生 INT4；用户已明确要求脱离 KVarN。为保持单一明确基线，不增加补零变体。
- B：论文 ReadKV 附录 D 的递归 Gaussian Lloyd Tree，最大深度4。用262144个标准正态中点分位数确定性求积，每个父区间二分并 Lloyd 更新；保存FP16质心和阈值。不是独立训练2/3/4-bit Lloyd–Max量化器。官方源码尚未找到，明确属于论文重实现。
- C：Any-Precision式2-bit种子四簇 Weighted K-means，再逐父簇二分至3/4-bit。样本频率作为权重，不引入 attention 权重。
- D：与C相同训练过程；样本权重乘局部 attention output Jacobian 平方范数。K 权重为 sum_h p_ht² ||V_t−O_h||² q_hc²/d，V权重为 sum_h p_ht²；标准化空间再乘 sigma_c²。固定正交变换后在变换坐标计算，链式求导与输出正交不变性已校验。GQA共享query heads贡献求和。只用校准query，冻结码本；测试时不重新编码或使用未来query。
- 原码字为同一4-bit叶路径；2/3-bit是其右移2/1的前缀。低精度绝不重新选最近码本。
- C/D为控制训练成本，在每个head/side的8192-bin充分统计上聚类，每个bin用加权样本均值与总权重表示；这是明确的近似，不宣称逐标量精确复现原论文实现。最多100次Lloyd迭代，相对损失变化1e-6停止，空簇继承父质心。

## 模型、数据、采集

- Qwen/Qwen3-4B，revision 1cfa9a7208912126459214e8b04321603b3df60c；本地权重，FP16，Transformers 4.57.6。完整36层，8 KV heads，32 query heads，head_dim128。
- 6任务：qasper/hotpotqa/gov_report/trec/passage_retrieval_en/lcc。每任务从现有50条候选中按长度取30条，共180。
- 原始LongBench zip中的context用于文档去重：归一化空白后精确SHA256；5-word shingle bottom-256哈希集合Jaccard>=0.8视为近重复，只选一个代表。此为近重复启发式，不保证识别全部语义重复。候选不足30时明确失败，不混入重复填充。
- 每任务长度排序后的6条循环分配 calibration/test/validation/test/calibration/test；每任务10/5/15，共校准60、验证30、测试90。输入清单保存全部实际token IDs、来源hash、文档hash、截断后长度，冻结后不修改。
- 最长8192 prompt tokens，过长首尾各4096；短样本不填充。长度由真实输入分布决定，不强造每个区间相同样本数。报告实际长度覆盖。
- 所有请求单条FP16参考生成，忽略EOS继续64个decode输入；在decode输入位置1/8/32/64截获实际attention接口中的RoPE后Q/K及完整V。相同参考轨迹用于所有方案；新量化器不参与生成，因此隔离编码误差。
- prefill按512 token分块，以降低峰值显存；不改变因果可见范围。SDPA调用前只读采集，原attention仍执行；注册新的独立Transformers attention接口，不改环境库文件。
- 固定Hadamard×随机符号正交变换，seed20261024，K/Q同变换，V使用同一固定变换；输出误差在正交等价空间计算。保存int8符号可重建变换。
- 完整KV在显存即时评估，不长期保存约百GB完整激活。校准每层每decode位置均匀不放回抽32个token，4位置合计128 token，保留128通道；每层/head/side约983040个标量。样本和敏感性保存NPZ。
- 均值/Scale按校准样本逐层/head/channel计算，FP16存储；最小Scale1e-4。三种子20261024/25/26，对校准请求bootstrap重加权，C/D共享对应bootstrap权重。变换、均值/Scale固定。A/B确定性仅跑一次；C/D的三个种子先在请求内部平均。
- validation仅独立诊断，本轮不调参；测试不参与训练、选阈值或方法选择。

## 指标

原始坐标下D_K(b)=||K−Khat||²/||K||²、D_V相同；旋转正交，测量等价。每层/head逐条记录，另报原始MSE。
Attention score S=QK^T/sqrt(d)，报告logit RMSE/相对L2；probability报告L1；output O=softmax(S)V，报告相对L2与绝对RMSE。
3个K-only(K2/3/4,V16)、3个V-only(K16,V2/3/4)、9个联合组合。无sink/tail特殊保护，全部有效KV量化。FP32参考，TF32关闭；每请求首层首decode额外计算FP64参考及原始/旋转坐标等价性。

统计单位是请求。报告请求均值、请求P95、逐层/任务/长度结果。相对改善=1−候选均值/基线均值；请求配对bootstrap2000次，95%CI。冻结实用阈值5%；CI下限>5%支持收益，CI上限<5%排除该幅度收益，其余证据不足。属于探索性多比较，不作未经校正的多重显著性宣称。首轮一个模型，外推其他模型需后续复验。

## 存储、前缀读取与等总字节

主条件：各方法相同4-bit payload，但部署元数据按真实紧凑数组记账。所有2/3/4-bit质心、编码阈值、mean/Scale、符号及字节对齐计入。每棵树28个质心+15个阈值=86字节；C/D逐层/head/side，B所有流共享同一个86字节表；A逐流min/max4字节。共同channel mean/Scale每流512字节。无逐token索引、码本选择或敏感性mask；dense布局由配置推导。Python NPZ容器头、CPU分析数组和bootstrap中间文件不属于部署格式；另记录磁盘使用。

_eq严格同总字节上限：保持同一个请求、同上下文和同mean/Scale。以B完整4-bit存储为总预算，其他方法在每个head/side的固定首批标量省略Bit0以支付额外元数据；每流按整字节取整。该位置4-bit读取时用3-bit质心重建，2/3-bit不受影响；省略布局固定无需额外mask。A每流省32个标量Bit0；C/D每流省688个（正式36层配置）；B不省。余下bitplane真实打包格式在check.py校验，实际总大小不得超过预算。正式过程只模拟该合法紧凑布局，不宣称已实现GPU压缩访问。

同时报告2K/4K/8K的真实容量账本、effective stored bits、payload logical read bytes，以及若不省位时同预算最大token容量。2-bit读取不意味着2-bit存储，未计GPU内存事务或cacheline流量。

## 边际收益与动态规划

Delta23=D2−D3；Delta34=D3−D4。检查Delta23>=Delta34>=0，raw MSE按K head/channel及V head/token诊断，数值绝对容差1e-7；报告违反比例和最大幅度，层/head记录可复算。不同层不要求D相等，而要求各自精度方向的收益条件成立。

每请求首decode、各层首KV head，取前8个K channels和前8个V tokens：分别对原MSE和query/attention加权的可分离目标，遍历0..16追加位预算，用精确DP对比贪心，报告gap。每stage item增加1bit的成本相同；不把K和V不同成本混入同一DP。不使用实际非可分离Output Error宣称贪心最优；不验证0/1-bit范围，也不将有限测试等同理论证明。

## 正确性门槛与产出

check.py检查nibble/位平面逐位恢复、等预算省Bit0字节大小、树前缀、Gaussian失真近似、正交性、Jacobians与FP64 autograd一致、收益不递减反例能被DP识别。
smoke覆盖独立两条请求和0/17/35层，验证端到端采集/训练/评估/报告。正式180请求仅在门槛通过后后台启动。

产出：status.json、inputs/manifest.json、calibration/full/*.npz、codebooks/full/*.npz、results/full逐请求压缩CSV、轨迹、数值检查、allocation DP、storage_ledger、equal_budget.csv、分层汇总、comparisons.json、结果分析.md。日志logs/background.log。失败写异常及栈，监督器生成失败报告，禁止把不完整运行标为完成。
