# 完整生成与官方任务质量评测协议

本实验固定 Qwen3-4B 与原始 Jacobian 编码，以 HumanEval、MATH-500、LongBench 原版全部样本比较五种策略。目标是检验完整生成任务质量，而不是局部输出误差、token 轨迹一致率或参考适配器速度。

## 数据范围

|数据集|完整范围|主要指标|
|---|---|---|
|HumanEval|OpenAI 官方 164 题，每题一个贪心输出|官方测试 pass@1|
|MATH-500|HuggingFaceH4 官方发布的 500 题|PRM800K 官方数学等价判分准确率|
|LongBench 原版|全部 21 子任务，共 4750 题|官方逐任务 F1、ROUGE、分类、检索、计数或代码相似度|

每策略 5414 次生成，五策略共 27070 次；小规模门槛不抵扣正式样本数。LongBench 不再限制此前六任务，每子任务不抽样，两个代码子任务各 500 题全部执行。数据冻结到 inputs/frozen，包含源记录、最终 prompt token IDs、原始与有效长度、截断标志及 SHA256。

HumanEval 从官方仓库下载固定 commit 的原始数据；MATH-500 从 HuggingFaceH4 的固定 revision 下载。两者与此前本地副本逐项核对：HumanEval prompt/test/entry point，MATH problem/answer/unique ID。LongBench 使用已固定 hash 的完整官方 data.zip，并固定其官方 prompt 和评分源码版本。来源和 hash 见 vendor/official_sources.json。

全量 LongBench 包含此前编码校准的 60 个请求，另有此前探索过的请求。为执行全量评测不删除它们，但逐题记录 used_in_codec_calibration 和 previous_exploratory_case；除全量成绩外提供排除校准请求的辅助成绩，不能把全量结果称为全新独立数据验证。

## 模型与五种策略

模型 Qwen3-4B，revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`，FP16 权重，项目 .venv，RTX 3090 24GiB、CUDA 12.6。关闭 TF32，固定随机种子 42，逐请求 batch=1，贪心解码，无采样。

1. fp16：FP16 KV attention。
2. full4：冻结 Jacobian 码字，完整读取 K4V4。
3. uniform3：同一份四位码字固定读取 K3V3。
4. query_only3：平均每侧三位预算，当前 q² 和重建 attention 平方敏感性，加统一理想失真形状。
5. query_curve3：相同预算，加入实测冻结 K/V 失真曲线。

编码严格使用原 D-K output-Jacobian 树与原 D-V attention 树，校准种子 20261024，不更换为 JacK_MixV，不按本轮任务得分调参。所有策略 prefill attention 均为 FP16；量化策略在全部 36 层 decode 时实际使用所选重建 KV，并将结果传给后续层和生成步骤。

K 决策只接受当前 Q 和冻结 K 曲线，随后用读取的 K 算 phat，再依据 phat 与冻结 V 曲线分配 V。完整源缓存只用于构建固定码字和参考适配器，不能作为决策输入，未来 token 和官方答案不进入策略。

## 自然结束与任务长度

|任务|输出上限|结束条件|
|---|---|---|
|HumanEval|2048 token|模型 EOS 或上限|
|MATH-500|8192 token|模型 EOS 或上限|
|LongBench|官方逐任务 dataset2maxlen，32 至 512 token|模型 EOS 或该任务上限|

所有生成允许提前 EOS，最低输出长度为零，不设置 ignore_eos，不强制达到预算。输出上限仅为资源边界，不把触顶截断写成自然结束。每题保存 finish_reason、generated_tokens、hit_output_limit 和原始输出。

使用模型原生最大序列长度 40960，为每任务输出预留空间。LongBench 超长 prompt 只保留 token 首尾，所有策略共享同一冻结截断结果；保存原始/有效 token 数和截断率。HumanEval 与数学若出现异常超长输入则失败，不静默截短问题。

LongBench 按官方模板构造任务文本，遵循 pred.py 中不加聊天封装的 trec/triviaqa/samsum/lsht/lcc/repobench-p 六任务列表；其他任务用 Qwen 聊天模板。HumanEval 要求完整实现函数，MATH 要求逐步推导并以 boxed 给出最终答案。聊天模板统一 enable_thinking=False。自然结束按模型 EOS 执行，不沿用旧的固定 64 步或 ignore_eos 设置。

## 官方评分与输出提取

HumanEval 使用固定的 OpenAI 官方 execution.py，其中 prompt、completion、官方 test 和 entry point 按官方规则组合，3 秒测试限制。代码在 Linux user/net/mount/PID/IPC/UTS namespace 和临时 chroot 内运行，只读 Python 运行库、私有临时目录、丢弃 capability、无外网，并设置资源和外层墙钟限制。无基础设施隔离时停止，不退回宿主直接执行。

HumanEval 对聊天输出采用冻结的格式提取：取第一个 Python 代码块；完整函数输出放在原 prompt 的空函数 stub 后重定义，或将纯函数体作为 completion。保留原始文本、提取规则与实际 completion，不根据测试结果修复代码或挑选候选。一题一次生成，使用通过比例报告 greedy pass@1。

MATH-500 采用 OpenAI PRM800K 的 grader.py 与 math_normalize.py，支持规范化与 SymPy 数学等价，不用简单字符串相等。提取最后一个平衡花括号 boxed/fbox；没有盒子时使用去除 Answer 前缀的最后非空行，并标记提取路径。提取只依赖模型输出，不看 gold；评分超时或基础设施错误单独保存并停止队列，不能伪装为已正确判分的零分。

LongBench 使用未修改的官方 eval.py 与 metrics.py。逐题对全部参考答案取最大分，对 trec/triviaqa/samsum/lsht 按官方规则只评分首行，再按官方公式平均。逐任务指标分别报告；21 任务等权宏平均仅作为本实验明确声明的辅助汇总，不替代官方逐任务成绩，也不与 HumanEval/MATH 的准确率混为一个总分。

## 正式运行前门槛

官方评分门槛包括：164 个 HumanEval canonical solution 全部通过、错误代码失败；500 个数学 canonical solution 与 500 个 boxed gold 全部通过、分数小数等价与错误答案测试；21 个 LongBench 子任务的逐题适配与官方 scorer 汇总一致。隔离环境执行 `/home` 不可见、根目录不可写、外网不可达检查。

模型门槛包括：GPU 分配器对随机、负收益、非收益递减、相等优先级以及全部实际校准曲线的分配结果与旧前置依赖堆算法一致；缓存与解码检查；四种已有策略复现上一轮同请求前八 token；受控模型在首 token 给 EOS 时立即结束，上限输出单独记录。

为减少完整生成开销，原始码字按新进入缓存的 KV 增量构建；不同矩阵形状的 FP32 旋转在极少量阈值边界可能出现末位差异。检查允许距分界 1e-5×max(1, 标准化幅值) 内、比例低于 1e-4 的边界敏感性，其余差异失败。实际门槛记录差异数量，并检查选定码字的各深度重建逐位一致以及已有生成轨迹。它不是新的校准或任务调优。

GPU 分配器仅向量化原两级细化堆贪心顺序：收益增加的链一旦选父位，子位下一次立即可选；稳定排序使用等效优先级和原单位顺序。此优化不赋予非收益递减曲线全局最优保证。

## 队列与可恢复性

先完成上述门槛，再在 HumanEval、数学、英文 QA、长摘要与中文 QA 各一题上，以正式任务上限测试五种策略，并评分。随后按 HumanEval、MATH-500、LongBench 各子任务运行每策略全部样本，每个策略/任务完成后立即官方评分和更新报告。

每题 JSON 原子保存，重启核对 prompt hash 与输出预算后跳过完整请求。运行前固定源码、配置、输入清单、码本与官方评分源码 hash；运行中不混入改动。任一生成、评分或完整性检查失败都停止并保留失败现场，不能把部分进度标为全量结果。

pipeline_status.json 为总队列，worker_status.json 为当前 worker，quality_progress.json 为已完成官方评分的覆盖范围；logs/pipeline.log 是完整日志。逐题原始输出在 results/full/策略/任务，逐题评分在其 scores 子目录。task_scores.csv、结果分析.md、paired_comparisons.json 与 complete.json 是最终输出。

## 统计与结论范围

报告官方任务分数、逐任务和逐请求差异、2000 次请求配对 bootstrap 95% CI、自然结束率、输出触顶率、prompt 截断率、生成长度及评分错误。多比较属于探索性统计，不宣称多重检验校正。参考适配器保留 FP16 源缓存与整数码缓存，所以墙钟、TPOT 和峰值显存不是部署压缩内核的性能或容量证据。

官方来源：[HumanEval](https://github.com/openai/human-eval)、[MATH-500 数据](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[PRM800K 数学评分](https://github.com/openai/prm800k/tree/main/prm800k/grading)、[LongBench 原版](https://github.com/THUDM/LongBench/tree/main/LongBench)。固定版本和文件 hash 见 vendor/official_sources.json。
