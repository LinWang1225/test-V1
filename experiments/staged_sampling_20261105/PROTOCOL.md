# Thinking 与采样的端到端编码比较（2026-11-05）

旧 greedy/non-thinking 实验已暂停，目录 `../staged_algorithm_20261102` 的结果与断点保留。新实验结果与旧结果不合并排名。

沿用原464个样本ID、原始任务文本和抽样：HumanEval 164、MATH-500分层抽样100、LongBench原版21子任务合计200。不删除LongBench、不补抽样、不使用测试答案拟合编码。历史60请求码本冻结复用；其校准并非本轮thinking轨迹，校准失配仍是需要观察的限制。

全部任务使用Qwen聊天模板、`enable_thinking=True`，包括此前采用raw completion/few-shot的六个LongBench任务；原始任务文本不变，封装变化单独记录。模型与revision仍为Qwen3-4B、`1cfa9a7208912126459214e8b04321603b3df60c`。

显式采样：temperature=0.6、top_p=0.95、top_k=20、min_p=0；不另加重复惩罚。主种子20261105，按case ID稳定派生每请求种子，各编码使用相同请求种子。单种子是可复现探索，不代表跨随机种子的稳定结论。所有方案prefill为FP16，decode全36层闭环使用对应重建。

输入完整重新分词、不截断。不按数据集设上下文长度或输出token上限，不按FP16生成长度停止。只使用本地模型配置的统一40960总上下文边界（模型卡把32768描述为原生上下文，40960为默认配置分配；本轮不启用YaRN或自行扩窗）。达到边界记native_context_limit，完整保留评分与完成状态。若完整prompt无法容纳则停止并记录，不静默截断或换题。

thinking与最终回答分开保存；仅`</think>`后的最终回答进入官方评分。没有闭合thinking则最终回答为空，仍进入分母。全部thinking token计入生成长度、耗时、逻辑KV访问；物理采样是否发生在thinking阶段由首次decode位置决定。

第一阶段依次运行FP16、Jacobian_Tree、JacK_MixV、Data_Tree、Gaussian_Tree、U_Clipped，各464条。量化均固定Hadamard-sign、完整4-bit读取。full4仅筛查基础质量，不能单独确定最佳动态读取编码。基底、曲线Controller及真实位平面工程仍按原分阶段依赖另行执行。

## 同次真实生成的物理流量

硬件计数为正式运行的必要条件。root系统服务接收固定的原始输入数据、请求种子和模型配置，启动Nsight包裹下的真实端到端生成；评分由普通用户执行。生成器不接收参考token、预期输出、历史生成结果或测试答案。各编码从自身当前重建KV和Query产生logits，采样的token进入自己的下一步，直到EOS或统一模型总上下文边界。

每请求在本次实际生成现场，分别对完整prefill与全部decode设置NVTX区间，读取`dram__bytes_read.sum`和`dram__bytes_write.sum`。两阶段均包含模型权重及其范围内的KV写入/编码/重建、工作区、采样；thinking包含在decode中。读、写、合计按题、任务和方案汇总；这不是纯KV流量，不以逻辑位数或单步采样外推。

Nsight使用app-range聚合模式收集两项DRAM指标。该工具模式支持多pass，但本实验强制只执行一次应用：生成入口用排他创建的invocation guard记录唯一execution_id，若工具尝试额外启动，立即失败，不生成第二条轨迹。实际回答文件、计数receipt及执行标记共享execution_id，评分使用该回答；若计数失败则停止，不能另跑一次生成来补计数。

不在正式队列执行固定参考轨迹、teacher forcing、逐层反事实、单步fixture重放或事后完整重跑。此前接入过程的诊断代码和结果保存在`diagnostics_before_live_counting/`，不参与正式质量或流量统计。

完整请求成功后原始计数CSV、命令、文件哈希、真实回答及执行标记均保留。正式实验不批量导出体积巨大的二进制ncu-rep，Nsight临时文件位于根分区`/var/tmp/kvarn-sampling-ncu`，工具完成后自动清理。诊断时已有的二进制报告保留。计数器需要管理员权限，因此现场计数的模型目标进程由root服务运行；生成的HumanEval代码只在普通用户的官方隔离评分器中执行。认证信息不写入文件。

断电或中断后跳过已完整保存并校验指纹的请求；当前未完成请求的硬件区间无法跨进程续接，保留失败尝试后从原始输入重新执行该题，不把旧生成前缀喂给模型，也不把失败尝试累计到有效流量。源代码、输入或码本变动导致指纹不匹配时停止。未完整测量的请求留空，不记零；只有464题都完成才标为方案最终总量。

HumanEval、MATH及LongBench各子任务分开报告，配对比较仅使用新协议FP16。同一小面板反复选方案属于探索性评估。记录的生成耗时包含Nsight开销，不作为无profiler的服务延迟或吞吐证据。
