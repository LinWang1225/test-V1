# 新赢家 HumanEval 验证

固定 Qwen3-4B、原 HumanEval 164题输入、贪心生成、自然EOS、2048输出上限与原隔离官方评分器。prefill FP16，decode 全36层使用 JacK_MixV（K复用原Jacobian，V使用混合目标树），种子20261024。旧实验结果只作参考，不修改。

先完整4-bit生成并评分，再依次运行adaptive005、adaptive015、adaptive030。新曲线仅使用原60条校准激活与原bootstrap权重重算各通道物理尺度MSE，不使用HumanEval答案或测试分数。保留源码、码本hash，逐题原子保存，重启跳过完整题目。

动态决策：K预测误差为当前旋转Q平方和乘scale平方乘冻结逐通道曲线；K选定位宽重建后计算phat，V预测误差为phat平方和乘冻结曲线通道和。每侧每次调用计算各单位相对于4-bit的非负额外误差。容忍量为该侧所有单位2-bit额外误差均值乘固定系数0.05/0.15/0.30。每单位选择满足容忍量的最小2/3/4-bit，不固定总位宽；误差代理不保证真实输出误差上界。三个阈值预先固定并全部报告，属于探索性质量/读取量曲线。

逐题保存K/V各深度单位数、实际KV元素数、逻辑payload bits、位平面填充bytes与元素加权平均位宽。读取量包含decode attention KV payload，不包含prefill、元数据、决策开销，不能称硬件DRAM节省或服务速度收益。保留FP16源缓存的参考适配器限制同旧实验。

运行：项目根目录 `.venv/bin/python experiments/winner_quality_20261029/src/run.py`。输出见results/summary.json及results/结果分析.md。
