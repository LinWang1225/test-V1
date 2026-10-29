运行位置：/home/wanglin/kvarn-profile-20261019
启动：.venv/bin/python -u experiments/progressive_int4/run.py
输出：results/progressive_int4_20261023/

本批是路线第2步开始的真实缓存离线验证，并非整个12周路线完成。
后台启动后不依赖终端保持连接。status.json 记录阶段/失败原因；
background.log 记录进度；正常结束自动生成 measurements.csv、aggregate.csv、结论.md。

inputs/ 为旧项目72份原生INT4真实trace的本项目独立副本。
vendor/仅复制旧项目fixtures.py和common.py进行原始布局解析与哈希验证，
不修改旧项目源码、环境、运行结果。量化器不变。protocol.json保存来源、
脚本SHA256、请求级校准/留出划分和每份输入SHA256。

本批覆盖：2+2恢复、三种重建、K2/K4×V2/V4、32-token共享GQA mask、
冻结预算的随机/固定/全局attention/attention×风险/Query×Scale候选。
留出样本不用于条件均值拟合或阈值调优。全局attention仅离线参考。
没有任务生成质量、在线单遍控制器、真实DRAM指标或端到端加速证据。
第4--6步验收以及第7--11步仍需后续实现和实验，不能用本批替代。
