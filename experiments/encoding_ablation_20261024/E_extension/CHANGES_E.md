# E扩展：文件、函数和目的

|文件|函数|目的|
|---|---|---|
|src/native.py|load|加载未修改的KVarN原生源码快照；不导入服务backend|
|src/native.py|原生_hadamard_cached AST提取|执行原生H生成逻辑，保留无sign/FP16旋转行为|
|src/native.py|NativeCodec.quantize|调用原生Triton Sinkhorn8、原生batched K/V RTN4与nibble打包|
|src/native.py|NativeCodec.reconstruct|唯一新增的前缀补零/中点算术；原生Scale/零点保持不变；4-bit对齐原生解码|
|src/native.py|NativeCodec.storage|原生payload/metadata/有效尾部及池槽容量账本；计算等预算逐最低位平面省略数量|
|src/run_e.py|EExperiment.observe|仅新增E/E_zero及等预算变体的15精度组合、误差/边际收益/DP|
|src/run_e.py|EExperiment.run|复用冻结输入，只重采集验证/测试；逐请求trajectory SHA256与A-D核对|
|src/run_e.py|EExperiment.__init__/status|复用原采集器，输出重定向本扩展；不覆盖A-D状态/结果|
|src/check_e.py|入口|原生4-bit编码/解码一致性、低bit前缀与尾部/存储检查|
|src/report_e.py|report|与原A-D冻结请求指标配对比较、CI、存储和完整量化span失真报告|
|src/supervise_e.py|入口|后台运行、退出码/失败报告，正常结束后将E分析追加主报告|
|references/*.py|原始函数不修改|固定原生量化器/解码器/H/默认参数来源与hash|
|PROTOCOL_E.md|全文|E的定义、无新增辅助数据含义、尾块差异、存储与统计协议|
|provenance.json|hash记录|E源码、原生快照、输入及A-D结果hash，便于追溯|

未修改src/KVarN或上级src/*.py。A-D训练与测量无需重跑。E只使用其原生量化器；不将整个实验改为KVarN推理或vLLM Kernel实验。

初版smoke在短context上发现仅省Bit0无法支付原生FP16尾部成本，按失败门槛终止，未启动正式实验。修正等预算条件为固定顺序逐最低位平面省略并重跑全部门槛；主E/E_zero定义保持不变。
