"""Build expanded reports from frozen CSV/JSON outputs; no GPU or reruns.

The original experiment sources/provenance remain frozen. This separate report
builder owns the narrative additions and the cross-run, deduplicated tables.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np

R = Path(__file__).resolve().parent
E = R.parent / 'encoding_extension_20261026'
B = R.parent / 'encoding_ablation_20261024'
S = B / 'supplementary_experiments'
C = R.parent / 'controller_jacobian_20261026'
N = E / 'results/full'
CR = C / 'results/replay'
CO = C / 'results/online'


def rows(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def load(p):
    return json.loads(Path(p).read_text())


def savecsv(p, data):
    with Path(p).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)


def table(headers, body):
    def cell(value):
        return str(value).replace('|', r'\|')
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|', *['|'+'|'.join(cell(v) for v in r)+'|' for r in body]])


def pct(v):
    return f'{100*float(v):.2f}%'


def interval(a):
    return f'[{100*a[0]:.2f}%, {100*a[1]:.2f}%]'


def write(p, text):
    Path(p).write_text(text.strip()+'\n')


# Every measured shared-preprocessing method, including two independent references.
# Name, family, implementation. Details common to a family are explained below.
METHODS = [
 ('U_MinMax','均匀范围','原 A。校准标准化样本的极值确定 lo/hi；round-to-nearest 编成 16 个等间距点，再对截断前缀取细码点区间中点。'),
 ('U_Clipped','均匀范围','同 A 的编码规则；上下尾各从 6 个分位数选择，共搜索 36 个范围，最小化同一码字 2/3/4-bit 平均校准重建损失。'),
 ('Gaussian_Tree','非均匀树','原 B，参考 ReadKV 的 Gaussian 递归 Lloyd 树；262144 个标准正态分位点作求积，每个父区间内二簇更新直到 4-bit，不用经验 KV 重新拟合分区。'),
 ('Data_Tree','非均匀树','原 C，参考 Any-Precision 的增量聚类；真实 KV 先训练 4 个种子簇，再逐父簇二分至 8 和 16 簇；旧父级分区和重建表保留，目标为标准化 MSE。'),
 ('Jacobian_Tree','非均匀树','原 D，沿用 C 的逐级二分；K 用 output-Jacobian 权重，V 用 attention 平方权重。controller 本轮固定的就是这一组合。'),
 ('JacK_MSEV','非均匀树组合','K 直接复用原 D-K，V 直接复用原 C-V；重新计算联合 attention 输出，不将两个单侧误差相加。'),
 ('ScoreK_MSEV','非均匀树组合','K 以 Query 通道系数平方和训练同结构树，V 用 C 的普通 MSE 树；检查只关注 score 对角误差是否足够。'),
 ('ScoreK_AttnV','非均匀树组合','K 用 Score-aware 树，V 用原 D-V attention-aware 树。'),
 ('ScoreK_MixV','非均匀树组合','K 用 Score-aware 树，V 用普通权重与均值归一化 attention 权重各 50% 的混合目标树。'),
 ('JacK_MixV','非均匀树组合','K 复用 D-K，V 用相同 50%/50% 混合目标树；这是全部主候选中平均输出相对 L2 最低的组合。'),
 ('Uniform_ScoreK_MSEV','均匀目标交叉','保留均匀前缀格式，用 Score 权重选择 K 的 36 组截断范围；V 用普通 MSE 选择范围。'),
 ('Uniform_ScoreK_AttnV','均匀目标交叉','同上，但 V 用 attention 权重选择范围；与 ScoreK_AttnV 形成目标类型对应的家族对照。'),
 ('Uniform_ScoreK_MixV','均匀目标交叉','同上，但 V 用普通/attention 各半权重选择范围；与 ScoreK_MixV 对应。'),
 ('Leaf4_Only','非均匀树优化','先拟合 16 个有序叶簇；相邻 4 个叶合成 2-bit 父组、相邻 2 个合成 3-bit 父组，粗级重建用组内加权均值。优先优化 4-bit，再构造粗前缀。'),
 ('Joint_234','非均匀树优化','从 C 初始化，交替调整有序叶分界和各深度重建值，最小化 (D2+D3+D4)/3；最多 60 轮，相对损失阈值 1e-6。'),
 ('TCN_MinMax','新增等宽嵌套','借鉴 BitLadder TCNQuant；校准极值范围分成 16 个等宽区间，floor 编码、前缀截断、区间中心重建。没有采用论文每 token/head 的局部在线范围或其 controller。'),
 ('TCN_Clipped','新增等宽嵌套','与 TCN_MinMax 相同；改为校准搜索 36 组截断范围，最小化 2/3/4-bit 平均 MSE。'),
 ('Residual211','新增残差','借鉴 QuantSpec 的粗量化后量化误差；unsigned 2-bit RTN 基础码，再追加两个 symmetric 1-bit 残差符号。每级共享幅度用裁剪后校准残差的加权 E|e| 拟合。'),
 ('LogResidual211','新增对数残差','借鉴 Lynx 对数压扩思想；按对称半径归一化，经 signed alpha-law 映射后做同样的 2＋1＋1 残差编码，各深度逆变换重建。alpha 在校准上从 1/4/16/64 选择。'),
 ('Independent_Uniform','独立精度参考','2/3/4-bit 各自搜索截断范围、各自独立量化为 4/8/16 个等间距点，编码之间不要求前缀一致。'),
 ('Independent_MSE','独立精度参考','2/3/4-bit 分别拟合 4/8/16 个非均匀 Lloyd 簇，分别编码；不要求共享父分区或前缀。'),
]
MAIN = [m for m, family, _ in METHODS if family != '独立精度参考']
NEW = ['TCN_MinMax','TCN_Clipped','Residual211','LogResidual211']


def collect():
    old = rows(S/'metrics/attention_errors.csv')
    new = rows(N/'summary.csv')
    index = {}
    for r in old:
        k=(r['method'],int(r['k_bits']),int(r['v_bits']))
        index[k]=dict(method=k[0],k_bits=k[1],v_bits=k[2],n_requests=int(r['n_requests']),output_rel=float(r['mean']),output_rel_p95=float(r['p95']),output_mse='',output_mse_p95='',source=str((S/'metrics/attention_errors.csv').relative_to(R.parent)))
    for r in new:
        k=(r['method'],int(r['k_bits']),int(r['v_bits']))
        if k in index:
            assert np.isclose(index[k]['output_rel'],float(r['output_rel']),rtol=0,atol=1e-12)
        index[k]=dict(method=k[0],k_bits=k[1],v_bits=k[2],n_requests=int(r['requests']),output_rel=float(r['output_rel']),output_rel_p95=float(r['output_rel_p95']),output_mse=float(r['output_mse']),output_mse_p95=float(r['output_mse_p95']),source=str((N/'summary.csv').relative_to(R.parent)))
    assert len(index)==21*15
    savecsv(R/'all_shared_methods.csv',[index[k] for k in sorted(index)])
    # Verify reused baselines for all requests/cells, rather than only the first
    # request's original raw-unit audit. Each old method enters the merge once.
    req={}
    for r in rows(S/'metrics/request_metrics.csv'):
        k=(r['case'],r['method'],int(r['k_bits']),int(r['v_bits']))
        req[k]=dict(case=k[0],method=k[1],k_bits=k[2],v_bits=k[3],output_rel=float(r['output_rel']),output_mse='',source='previous_supplement')
    matched=[]
    for r in rows(N/'request_metrics.csv'):
        k=(r['case'],r['method'],int(r['k_bits']),int(r['v_bits']))
        if k in req:
            matched.append(abs(req[k]['output_rel']-float(r['output_rel'])))
            assert matched[-1]<1e-12
        req[k]=dict(case=k[0],method=k[1],k_bits=k[2],v_bits=k[3],output_rel=float(r['output_rel']),output_mse=float(r['output_mse']),source='encoding_extension')
    assert len(req)==21*90*15 and len(matched)==3*90*15
    savecsv(R/'all_request_metrics.csv',[req[k] for k in sorted(req)])
    savecsv(R/'method_implementations.csv',[dict(method=m,family=f,implementation=d,main_progressive=m in MAIN) for m,f,d in METHODS])
    return index,new,dict(deduplicated_methods=21,main_progressive_methods=19,independent_reference_methods=2,summary_cells=len(index),request_cells=len(req),shared_baseline_request_cells_checked=len(matched),max_shared_baseline_difference=max(matched),old_absolute_mse_policy='blank when not available in prior aggregate; mean RMSE is not squared to fabricate MSE')


def reltable(index,methods):
    return table(['方法','2-bit 相对 L2','3-bit 相对 L2','4-bit 相对 L2'],[[m,*[f"{index[m,b,b]['output_rel']:.6f}" for b in (2,3,4)]] for m in methods])


def encoding_report(index,new):
    comp=load(N/'comparisons.json')
    selected=['U_Clipped','Data_Tree','Jacobian_Tree',*NEW]
    cif=[]
    for m in ['TCN_Clipped','Residual211','LogResidual211']:
        vals=[]
        for b in (2,3,4):
            r=next(r for r in comp if r['method']==m and r['baseline']=='U_Clipped' and r['bits']==b and r['metric']=='output_rel')
            vals.append(f"{pct(r['improvement'])}，{interval(r['ci95'])}")
        cif.append([m,*vals])
    mses=table(['方法','2-bit MSE','3-bit MSE','4-bit MSE','4-bit 相对 L2 请求 P95'],[[m,*[f"{index[m,b,b]['output_mse']:.6g}" for b in (2,3,4)],f"{index[m,4,4]['output_rel_p95']:.6f}"] for m in selected])
    write(N/'结果分析.md',f"""
# 四位渐进编码补充实验结果分析

本轮 90 条测试请求已全部完成。新增等宽嵌套、线性逐级残差和对数域逐级残差均未在平均 Attention Output 相对 L2 上超过原 Jacobian Tree。逐级残差在 3/4-bit 显著改善原均匀前缀基线，但仍落后于经验分布树；等宽截断一致性解决了编码结构问题，并不自动带来最佳 attention 精度。

## 实验配置与评价口径

Qwen3-4B 固定 revision，60 条校准、90 条测试，原 30 条验证请求不执行。使用项目 .venv、RTX 3090、FP16 模型、FP32 attention 参考并关闭 TF32。六项 LongBench、36 层、8 KV heads、32 Query heads、decode 1/8/32/64；三个校准 bootstrap 种子在请求内汇总。

全部候选有效 KV 完整存储 4-bit，只从同一码字读取 2/3/4-bit，没有 FP16 KV 尾部、不删 token。统一旋转、逐通道 mean/scale 和每层/head/侧共享编码状态。主比较为相同 4-bit payload，元数据另外计费，并非严格相同总字节。

本轮是固定 FP16 参考轨迹上的单层反事实：在某一步、某层上替换 K/V 重算 attention，随后模型仍用原 FP16 输出继续生成。它隔离编码误差，不测跨层传播。统计先在每请求内部平均种子、层、head、位置，再汇总请求均值与 P95；置信区间为 2000 次请求配对 bootstrap。

相对 L2 是 `‖Ohat−O‖/‖O‖`，不是任务准确率下降百分比；绝对 MSE 是输出元素的平均平方差。所有方法还保存 K-only、V-only 与 9 个联合精度组合。

## 新增方法如何实现

TCN_MinMax 借鉴 BitLadder TCNQuant：`q4=floor(16×(z−lo)/(hi−lo))` 后饱和到 0–15，`qb=q4>>(4−b)`，重建为 `lo+(qb+0.5)×(hi−lo)/2^b`。TCN_Clipped 保持该结构，用校准上的 36 组上下尾分位范围优化三精度平均重建 MSE。这里没有复现论文每 token/head 局部范围、动态分配器或 PlaneWeave 内核。

Residual211 借鉴 QuantSpec 的逐级残差：先用 2-bit round-to-nearest 得到粗值，再分别用两个符号位编码当前残差。第三级和第四级各有一个每 head/侧共享幅度，幅度取裁剪后校准残差的加权绝对值均值；码字为 2＋1＋1，因此 3-bit 有独立且明确的重建规则。它不是原论文 4＋4-bit 配置。

LogResidual211 借鉴 Lynx 对数压扩，先按对称半径归一化，用 signed alpha-law 将幅度压入对数域，再做同样的 2＋1＋1 编码，各深度逆变换回原空间。alpha 仅在校准上从 1/4/16/64 选择；不包含原论文的 Page/Chunk 局部归一化、Anchor biased extraction、原生有符号修正映射或网络推测执行。

三个重算基线为：U_Clipped 的均匀 RTN 前缀；Data_Tree 的经验分布逐簇二分；Jacobian_Tree 的同结构功能加权树。新编码采用普通校准重建 MSE，而 Jacobian 保留功能敏感性目标，不能把它们的全部差异归因于编码格式。

## 读取精度与输出相对误差

{reltable(index,selected)}

优化后的 TCN 相对 U_Clipped 在 2-bit 几乎持平，在 3/4-bit 分别降低约 6.51% 和 10.84%。因此，等宽分箱与中点恢复有实际收益，但还不足以缩小到经验非均匀树的水平。未优化范围的 TCN_MinMax 在 2/3-bit 误差很大，说明共享校准极值会明显放大量程；不能把这一结果外推成原生局部 TCNQuant 的失败。

Residual211 相对 U_Clipped 的 2-bit 误差增加约 1.58%，但 3/4-bit 分别降低约 19.25% 和 33.51%。逐级修正确实可以用新增位弥补粗编码误差。不过，每级只有一个共享修正幅度，无法像树一样为不同父簇学习独立子质心，表达能力更受限制。这是与结果一致的机制解释，本轮没有单独分解“共享幅度约束”贡献了多少差距。

LogResidual211 在 2-bit 相对 U_Clipped 改善约 6.01%，3-bit 改善约 2.12%，4-bit 反而退化约 2.70%。它表现为粗精度受益、细精度收益不足。对数压扩改变了码点密度，也引入对称范围和有限 alpha 搜索的约束；本轮结果不支持继续把它当作默认最优编码，也不能证明包含局部归一化的完整 Lynx 路线无效。

下表每项为相对 U_Clipped 的误差下降及其 95% CI；负值表示退化。

{table(['方法','2-bit 改善与 CI','3-bit 改善与 CI','4-bit 改善与 CI'],cif)}

## 绝对 MSE 与相对 L2 为什么要同时看

{mses}

4-bit 时，Jacobian Tree 的平均相对 L2 为 0.136101，低于 Data Tree 的 0.150318；但其绝对 MSE 为 0.006844，高于 Data Tree 的 0.006128。两者并不矛盾：每个层/head 的相对误差先按该单元参考输出范数归一化，和直接平均原始平方差的隐含权重不同。因此“Jacobian 优于 Data Tree”的判断必须限定为平均相对 L2，不能宣称所有指标都占优。

K-only/V-only 也能定位问题：TCN_MinMax 的 2-bit K-only 相对 L2 为 0.576907，V-only 达 1.867144，低精度严重退化主要在该单侧诊断中表现为 V 重建问题；TCN_Clipped 对应两侧降至 0.396308 与 0.388864。联合误差不能简单等于两侧相加，以上只作来源诊断。

## 完整性与结论边界

原三基线首请求的 155520 个单元重算结果与旧管线完全一致；逐请求 token 轨迹 hash 也一致。综合报告另外核对了全部 90 请求、15 精度组合的重复基线，见[合并审计](../../../encoding_summary_20261027/merge_audit.json)。

本次七方法内，Jacobian Tree 的平均相对 L2 最低；把历史候选一起比较时，Jacobian K＋混合 V 的相对 L2 还更低。controller 按事先约定固定原始 Jacobian_Tree，不在本轮后改选赢家。[全部编码综合报告](../../../encoding_summary_20261027/编码实验综合分析.md)统一说明实现和对比口径。

旧测试结果已用于方案设计，本轮为探索性复用评估；不能视为独立数据确认。当前逻辑位平面读取没有实际压缩 attention 内核，因此不推出 HBM、吞吐或显存收益。

## 数据入口

[逐请求指标](request_metrics.csv)、[完整汇总](summary.csv)、[配对置信区间](comparisons.json)、[存储账本](storage_ledger.csv)、[完整性审计](audit.json)、[比较图](encoding_comparison.png)。实现见[四位编码器](../../src/codecs4.py)、[拟合](../../src/train.py)和[评估](../../src/evaluate.py)。
""")


def replay_report():
    data=rows(CR/'summary.csv');comparisons=load(CR/'comparisons.json')
    idx={(r['policy'],float(r['target_mean_bits'])):r for r in data}
    fulltable=table(['策略','平均读取位数目标','真实输出 MSE','相对 L2'],[[p,f'{b:g}',f"{float(idx[p,b]['output_mse']):.7f}",f"{float(idx[p,b]['output_rel']):.6f}"] for b in (2.,2.5,3.,3.5,4.) for p in ('uniform','static','query_only','query_curve')])
    comparison=table(['相对基线','3-bit MSE 下降','请求配对 95% CI'],[[r['baseline'],pct(r['improvement']),interval(r['ci95'])] for r in comparisons if r['target_mean_bits']==3.])
    proxy=table(['策略','真实 V 误差／预测值 中位数','真实 V 误差／预测值 P95'],[[p,f"{float(idx[p,3.]['v_exact_to_predicted_median']):.3f}",f"{float(idx[p,3.]['v_exact_to_predicted_p95']):.3f}"] for p in ('uniform','static','query_only','query_curve')])
    q=idx['query_curve',3.]
    fraction=float(q['v_cross_mse'])/float(q['v_exact_mse'])
    write(CR/'结果分析.md',f"""
# Jacobian 编码因果回放结果分析

90 条请求已完成。在平均每侧读取 3-bit、编码和读取预算相同的条件下，两阶段 Query＋校准失真曲线 controller 将真实 Attention Output MSE 从均匀读取的 0.071487 降至 0.041036，下降 42.60%。但相对仅 Query 敏感性策略只额外降低 2.98%，本轮主要收益来自当前查询引导的动态分配，经验失真曲线的增量较小。

## 实验范围与信息约束

编码固定原始 Jacobian_Tree，即 K 为原 D 的 output-Jacobian 树、V 为原 D 的 attention 树，首校准种子 20261024。不是历史相对 L2 最佳的 JacK_MixV，避免在比较 controller 时同时换编码。完整存储 4-bit，读取深度为 2/3/4-bit。

使用六任务 90 条冻结 FP16 轨迹，覆盖第 0/17/35 层、decode 1/64。与编码主实验的 36 层、四位置、三种子覆盖不同，两份报告的绝对数值不能直接横向相减。回放中量化输出不进入下一层，参考 token hash 与旧采集一致。

K 决策只接收当前 Q 和冻结的校准 K 曲线；返回所选 K 后计算 phat，再由 phat 和冻结 V 曲线决定 V 读取。policy.py 不接收完整当前 KV、原始 FP16 attention、真实输出或未来 Query。完整数据只由独立评估器计算真实误差，因而“离线回放”不等于“策略可以看未来”。

## 四种策略及预算实现

uniform 尽可能均匀地安排深度；分数位预算使用固定、分散的混合深度。static 仅根据校准失真分配，不用当前 Query。query_only 用 q² 或 phat² 的敏感性，深度失真统一设成 [1, 1/4, 1/16]。query_curve 同时使用当前敏感性与实测校准曲线。

K 的读取单位为一个 head/channel 的全部 token，V 为一个 head/token 的 128 通道。两侧预算独立固定，每个额外位的成本分别为 ceil(T/8) 与 16 字节，按代理误差下降除以新增字节分配。失真已在原始数值尺度测量，不重复乘标准化 scale。同阶段增加一位等成本，可直接按边际收益排序；跨阶段则不能忽略成本差异或 K 对 V 权重的影响。

分配满足先读 2→3 再读 3→4 的前置关系。收益递减时排序分配通过小规模精确 DP 检查；非收益递减时用堆贪心，不保证全局最优。相同目标下各策略的实际位数及逐行字节填充成本均已核对一致。

## 同预算质量比较

{fulltable}

2-bit 和 4-bit 端点各策略结果完全相同，因为没有自由分配空间；差异出现在 2.5/3/3.5-bit 的中间预算。3-bit 时，query_curve 明显优于 uniform 和 static，说明有用的不只是“总体上读更多位”，而是在相同总读取量下把位分给当前更敏感的位置。

{comparison}

静态分配在 3-bit 甚至比均匀读取更差：只优化校准重建失真可能把位数投向幅值大、但当前输出不敏感的 head/channel/token。这是与实验一致的解释，不能由平均指标单独证明具体误差来自哪些单位。

实测曲线并非所有预算都优于理想曲线：3.5-bit 时 query_curve 的 MSE 为 0.030299，query_only 为 0.029912，前者反而高约 1.29%，而相对 L2 基本相同。不能将“加入真实曲线”表述为无条件改善；应继续检验校准代表性、代理失配和曲线粒度。

## 代理排序可靠性与绝对误差偏差

同预算、同一 query/head 的跨策略 V 误差成对排序正确率为：2.5-bit 85.84%、3-bit 87.84%、3.5-bit 82.20%。这些比例只对应四策略之间非平局的比较，不是任意分配方案的全局排序保证，也不是完整联合输出误差的排序正确率。

{proxy}

3-bit query_curve 的真实 V 加权误差／预测值中位数为 1.549，P95 为 8.822。因此代理可以帮助选择，但尚不能据其数值给出绝对输出误差阈值承诺。

固定该策略自身的重建 attention，V 的精确误差包含对角项和跨 token 交叉项：`‖Σ_t phat_t δv_t‖² = Σ_t phat_t²‖δv_t‖² + Σ_(t≠u) phat_t phat_u〈δv_t,δv_u〉`。本轮 query_curve 的平均 V 精确 MSE 为 0.018096，实际对角项为 0.003336，交叉项为 0.014761。后者约占平均精确 V 误差的 {fraction:.2%}，这是两个汇总均值之比，不是逐样本占比的均值或中位数。

这也区分了两种失配：校准曲线可能预测不准真实对角残差；即使对角项预测准确，误差相关性仍可使完整加权输出远大于对角和。后续若做质量阈值 controller，需要单独处理这两者。

## 局部 oracle 与后续验证

首请求第 0 层位置 1 的 head0，对四个 K 通道或四个 V token 枚举 19 个同预算方案，其他单位固定 3-bit。K 侧代理选择的真实 MSE 为 7.9715e-5，局部真实最优为 6.4794e-5，绝对 regret 为 1.4921e-5；V 侧这个小问题的代理选择恰好达到真实最优。样本只涉及一个 query/head，不能据此推断全局最优性。

因果回放之后已完成[在线闭环 pilot](../online/结果分析.md)。下一步应扩大任务质量和独立数据验证，再推进真实位平面读取内核。当前输出改善是数值决策证据，尚非 HBM 流量或端到端速度证据。

## 数据入口

[请求指标](request_metrics.csv)、[完整汇总](summary.csv)、[配对比较](comparisons.json)、[代理成对排序](proxy_pairwise_ranking.json)、[秩相关](proxy_rank_correlations.csv)、[局部 oracle](local_oracle.json)。实现见[纯策略接口](../../src/policy.py)与[回放及在线适配器](../../src/run.py)。
""")


def online_report():
    done=load(CO/'complete.json');data=done['results']
    idx={(r['case'],r['policy']):r for r in data}
    cases=sorted({r['case'] for r in data})
    means={p:dict(agreement=float(np.mean([r['token_position_agreement'] for r in data if r['policy']==p])),mse=float(np.mean([r['sampled_local_output_mse'] for r in data if r['policy']==p]))) for p in ('fp16','full4','uniform3','query_curve3')}
    summary=table(['策略','与 FP16 的 token 位置一致率','采样局部 MSE','64 步完全一致的请求'],[[p,pct(v['agreement']),f"{v['mse']:.8f}",f"{sum(idx[c,p]['token_position_agreement']==1 for c in cases)}/12"] for p,v in means.items()])
    detail=table(['请求','完整 4-bit 一致率','均匀 3-bit 一致率','动态平均 3-bit 一致率'],[[c,*[pct(idx[c,p]['token_position_agreement']) for p in ('full4','uniform3','query_curve3')]] for c in cases])
    delta=np.array([idx[c,'query_curve3']['token_position_agreement']-idx[c,'uniform3']['token_position_agreement'] for c in cases])
    rng=np.random.default_rng(20261027);ci=np.percentile(delta[rng.integers(0,len(cases),(2000,len(cases)))].mean(1),[2.5,97.5])
    savecsv(R/'online_request_comparison.csv',[dict(case=c,dataset=idx[c,'fp16']['dataset'],full4_agreement=idx[c,'full4']['token_position_agreement'],uniform3_agreement=idx[c,'uniform3']['token_position_agreement'],query_curve3_agreement=idx[c,'query_curve3']['token_position_agreement'],difference_query_curve_minus_uniform=delta[i]) for i,c in enumerate(cases)])
    write(CO/'结果分析.md',f"""
# Jacobian 编码在线闭环试验分析

12 条请求、每策略 64 步的闭环 pilot 已完成。动态平均 3-bit 相比均匀 3-bit，平均 token 位置一致率从 {pct(means['uniform3']['agreement'])} 提高到 {pct(means['query_curve3']['agreement'])}，增加 {100*delta.mean():.2f} 个百分点；采样局部 MSE 均值从 {means['uniform3']['mse']:.8f} 降至 {means['query_curve3']['mse']:.8f}，低约 {pct(1-means['query_curve3']['mse']/means['uniform3']['mse'])}。但逐请求为 3 条改善、3 条退化、6 条持平，不能据此宣称稳定的任务质量收益。

## 在线与回放有什么区别

同样固定原 Jacobian_Tree、首校准种子。六任务各取冻结测试清单第一条和最后一条，共 12 条，使用原上下文；prefill 为 FP16。decode 的全部 36 层都用当前策略的量化 attention 输出，结果实际传给下一层和后续 token。每种策略独立生成，因而会形成不同的 Query 和缓存轨迹。

这与单层回放不同：回放将当前局部输出与同一 FP16 输入下的结果配对，但不让其改变生成；在线试验则观察误差如何影响整个后续计算。FP16 分支必须逐 token 复现旧冻结轨迹，该检查已通过。controller 仍只看当前 Q、校准曲线和重建 attention，不看未来 token 或原始完整 attention。

## 策略与整体结果

full4 完整读取 Jacobian K4V4；uniform3 每个单位读取 3-bit；query_curve3 在 K/V 各自平均 3-bit 预算内动态选择 2/3/4-bit。实际码字完整存储 4-bit。采样局部 MSE 在 decode 1/8/32/64 测量，同样先在请求内汇总。

{summary}

完整 4-bit 的平均一致率更高，是本轮有用的精度参考。动态 3-bit 在相同名义 payload 读取量下改善均匀 3-bit 的平均结果，但仍没有达到 full4 的平均一致率；这一观察支持继续研究分配，并不说明 3-bit 能普遍替代完整读取。

## 请求差异与统计不确定性

{detail}

动态方案既有明显改善，也有退化。例如 qasper/16 从 1.56% 一致率提高到 100%，而 gov_report/46 从 100% 降到 35.94%。在这 12 条请求上，平均一致率差的探索性请求配对 bootstrap 95% CI 为 [{100*ci[0]:.2f}, {100*ci[1]:.2f}] 个百分点，区间跨过零。这个补充统计只用于说明小样本不确定性，不是预注册的任务显著性检验。

位置一致率比较相同生成位置的 token。一次早期分歧会改变后续上下文，即使两个分支生成同义但措辞不同的答案，也会拉低一致率；相反，一致率高也不能证明答案正确。它适合诊断轨迹稳定性，不是官方任务准确率、语义相似度或答案质量评分。

采样局部 MSE 也需限定解释：每个策略的局部 FP16 shadow attention 使用该策略当前分支的 Q/K/V 输入，而不是从头全 FP16 生成的同一隐藏状态。因此跨策略的局部均值并非同输入配对测量，更不能作为完整模型输出误差下降 44% 的证据。

## 为什么现在还不能报告速度或显存收益

参考适配器保留 FP16 源缓存，每步重算冻结码字并模拟前缀选择，还包含 CPU 分配、数据同步和 PyTorch 中间张量。它验证因果决策及跨层传播，不是部署用压缩缓存内核。保存的墙钟、TPOT 与峰值显存只反映该适配器开销，不能推出实际 HBM 流量、容量或服务加速。

本轮只生成 64 步，尚未做官方任务评分。下一阶段应保留 uniform3、query_curve3、full4 和 FP16 四个对照，增加自然结束的完整生成与官方任务指标，并在独立数据上检查均值、尾部与逐任务退化；质量证据稳定后再评估真实位平面内核的总延迟。

## 数据入口

每请求各策略的 JSON 包含生成文本、token IDs、首次分歧、参考适配器耗时和局部 MSE；[complete.json](complete.json)汇总完整记录，[逐请求对照表](../../../encoding_summary_20261027/online_request_comparison.csv)用于复核上述表格。另见[因果回放报告](../replay/结果分析.md)与[在线适配器实现](../../src/run.py)。
""")
    return dict(agreement_delta_mean=float(delta.mean()),agreement_delta_ci95=ci.tolist(),wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),bootstrap_samples=2000,bootstrap_seed=20261027)


def combined_report(index):
    impl=table(['方法','结构或目标','具体实现'],METHODS)
    main=reltable(index,MAIN)
    independent=reltable(index,['Independent_Uniform','Independent_MSE'])
    unified=rows(B/'results/unified/summary_A_to_E.csv')
    ui={(r['method'],int(r['k_bits']),int(r['v_bits'])):float(r['output_rel_mean']) for r in unified}
    nr=rows(S/'metrics/native_nt/request_metrics.csv')
    ni={}
    for m in ('KVARN_nt','KVARN_nt_zero'):
        for b in (2,3,4):
            values=[float(r['output_rel']) for r in nr if r['method']==m and int(r['k_bits'])==int(r['v_bits'])==b]
            assert len(values)==90
            ni[m,b]=float(np.mean(values))
    native=table(['KVarN 条件','2-bit 相对 L2','3-bit 相对 L2','4-bit 相对 L2'],[[name,*[f'{value(b):.6f}' for b in (2,3,4)]] for name,value in [('E，FP16 尾部，中点',lambda b:ui['E',b,b]),('E_zero，FP16 尾部，补零',lambda b:ui['E_zero',b,b]),('KVARN_nt，全部 KV 量化，中点',lambda b:ni['KVARN_nt',b]),('KVARN_nt_zero，全部 KV 量化，补零',lambda b:ni['KVARN_nt_zero',b])]])
    eqs=['A_eq','B_eq','C_eq','D_eq','E_eq','E_zero_eq']
    eqtable=table(['历史等总预算包装','2-bit 相对 L2','3-bit 相对 L2','4-bit 相对 L2'],[[m,*[f'{ui[m,b,b]:.6f}' for b in (2,3,4)]] for m in eqs])
    ledger=rows(N/'storage_ledger.csv')
    storage=[]
    # One deployed codebook, not all bootstrap models at once.
    for m in ['U_Clipped','Data_Tree','Jacobian_Tree',*NEW]:
        r=next(r for r in ledger if r['method']==m and r['context']=='8192' and r['read_bits']=='4')
        storage.append([m,f"{int(r['payload_bytes'])/2**20:.3f}",f"{int(r['metadata_bytes'])/1024:.3f}"])
    mses=table(['方法','2-bit 绝对 MSE','3-bit 绝对 MSE','4-bit 绝对 MSE'],[[m,*[f"{index[m,b,b]['output_mse']:.6g}" for b in (2,3,4)]] for m in ['U_Clipped','Data_Tree','Jacobian_Tree',*NEW]])
    write(R/'编码实验综合分析.md',f"""
# 渐进式 KV 编码实验综合分析

本报告合并原 A–E、均匀与非均匀目标消融、KVarN 尾部消融，以及新增 BitLadder 式等宽嵌套和两种逐级残差编码。相同预处理下，共有 **19 个完整 4-bit 渐进候选和 2 个独立精度参考**；KVarN 的 4 个条件与历史 6 个等总预算包装另表列出，避免把不同前处理、尾部保护和预算口径混成一个排名。

按全部 19 个主候选的平均 Attention Output 相对 L2，**Jacobian K＋混合 V（JacK_MixV）在 2/3/4-bit 均最低**；本次新增编码没有改变这一结论。原 Jacobian_Tree 也优于四个新增方案的相对 L2，但不是历史全部候选的最优组合。当前 controller 按事先约定固定原始 Jacobian_Tree，不能把 controller 结果归给 JacK_MixV。

## 一 实验条件与可以比较的范围

模型为 Qwen3-4B，固定 revision `1cfa9a7208912126459214e8b04321603b3df60c`。六项 LongBench 为 qasper、hotpotqa、gov_report、trec、passage_retrieval_en、lcc；60 校准、90 测试，原 30 验证请求按此前要求跳过。prompt 最多 8192 token，参考生成 64 步，在 decode 1/8/32/64 采集；覆盖 36 层、8 KV heads、32 Query heads，head_dim 为 128。项目 .venv、RTX 3090 24GiB、CUDA 12.6，FP16 模型与 FP32 attention 参考，关闭 TF32。

主候选采用共同的固定 Hadamard-sign 旋转、每层/head/channel 校准 mean/scale，标准化为 `z=(x_rot−mean)/scale`；K、V 分别拟合编码参数。参数在测试时冻结。完整存储每标量 4-bit，2/3-bit 必须来自相同 4-bit 码字的前缀，无 FP16 KV 尾部、不删 token、不省略最低位。学习型方法按三个校准 bootstrap 种子在请求内汇总；确定性历史方法按其原协议运行。

单层反事实是“同一步、同一层、相同 Q/K/V 下，假如替换成量化 KV，attention 输出改变多少”。量化结果不进入下一层和下一 token，因此各方法共享参考轨迹；它衡量局部编码质量，不代表完整生成任务质量。统计先在请求内平均层、head、位置和种子，再计算 90 请求的均值、P95 与配对置信区间。

主比较统一的是 4-bit **有效载荷**，元数据单独计费；不是所有方法相同总容量。独立三精度参考若同时保留三份编码共需 9-bit，KVarN 则采用本地块均衡与元数据，历史 `_eq` 为支付元数据省略低位。这三类都不能作为主表的相同条件赢家。

旧测试结果已经参与后续方法设计，全部比较属于探索性复用评估；当前证据仍局限于一个模型和这批请求。

## 二 指标与单份四位码字的含义

参考为 `S=QKᵀ/√d`、`P=softmax(S)`、`O=PV`，重建后得到 Ohat。平均相对 L2 先在每个局部单元计算 `‖Ohat−O‖/‖O‖` 再汇总，数值 0.1 表示该局部输出差值范数约为参考的 10%，不是任务准确率下降 10%。绝对 MSE 为输出元素的平均平方差；两者的归一化和隐含权重不同，必须分别解释。

4-bit 码字如 `1011`，读 2-bit 得到 `10`，读 3-bit 得到 `101`，读 4-bit 才取完整路径。均匀、等宽 TCN、非均匀树与残差编码的重建规则不同，但都不能在读低精度时重新拟合码本或独立重编码。各方法均评估 3 个 K-only、3 个 V-only 与 9 个联合精度组合；下表只展示联合 K2V2/K3V3/K4V4，完整 15 组合见[机器可读汇总](all_shared_methods.csv)。

K/V 重建 MSE、score 误差和 attention 概率变化用于定位原因，不能替代真实输出误差。新实验额外记录 V 对角项、跨 token 交叉项和 K/V 输出交互项，避免把可分离代理当成精确输出预测。

## 三 每种方法的具体实现

{impl}

### 均匀前缀与 TCN 的编码边界不同

旧均匀编码为 `q4=clip(round(15×(z−lo)/(hi−lo)),0,15)`。读 b-bit 时取 `qb=q4>>(4−b)`，在原 16 个细码点中恢复对应索引区间的中点，再按 `lo+(hi−lo)×qhat4/15` 反量化。它支持前缀重建，但直接独立 b-bit affine 编码不一定等于截断 q4。

TCN 使用 `q4=clip(floor(16×(z−lo)/(hi−lo)),0,15)`，b-bit 重建为 `lo+(qb+0.5)×(hi−lo)/2^b`。粗区间是细区间的并，直接低精度分箱与高位截断一致。这保证编码分区嵌套，不保证同一分布下的输出误差最小。

U_Clipped、TCN_Clipped 与均匀目标交叉组均只在校准集选择范围。上下尾候选分位数为 0、0.0001、0.001、0.005、0.01、0.02，共 36 组，目标兼顾三精度。TCN_MinMax 使用原始校准 extrema；新增编码范围参数和重建表采用实际 FP16 精度参与候选评价。

### 非均匀树的结构与训练权重

Data Tree 从 4 个种子簇逐级扩展到 8、16 簇，保留祖先关系；每父簇独立二分，细级码字自然包含父索引。经验拟合使用 8192-bin 加权充分统计近似，最多 100 次 Lloyd 更新、相对损失阈值 1e-6；不是逐标量的全局最优证明。

Gaussian Tree 用固定标准正态先验；Data Tree 用真实 KV 的频率权重。D、Score、Mix 沿用同类树结构，主要区别为校准目标，不是三种新的位打包格式。设 h 遍历共享同一 KV head 的 Query heads，t 为 token，c 为通道，σ 为标准化 scale：

|目标|标准化重建损失使用的敏感性权重|
|---|---|
|普通 MSE|1|
|K Score|Σ_h q[h,c]² σ_K[c]² / d|
|K Output Jacobian|Σ_h p[h,t]² ‖v[t]−o[h]‖² q[h,c]² σ_K[c]² / d|
|V Attention|Σ_h p[h,t]² σ_V[c]²|
|V Mix|0.5＋0.5×均值归一化 attention 权重|

频率权重和请求 bootstrap 权重另行乘入。上述权重从校准数据获得，测试时不依据当前 Query 重训编码；平方导数代理忽略误差交叉项，不能称为直接最小化完整 attention 或任务 loss。

Data Tree 是 Any-Precision 增量聚类结构的 KV 适配；Any-Precision 原本主要量化模型权重，敏感性来源与本实验不同。Gaussian Tree 参考 ReadKV 的递归 Gaussian 编码。所有这些名称都不表示复现了原论文完整推理系统。

### 两种残差编码为什么能读三种精度

Residual211 先用 2-bit RTN 得到粗重建 r2，随后追加 `sign(z−r2)` 得到 r3，再追加 `sign(z−r3)` 得到 r4；每级重建为前一级加上正负共享幅度。训练幅度为裁剪后残差的加权 E|e|，实际未裁剪测试输入的残差符号按当前值计算，基础码饱和。码字 `[base2, sign3, sign4]` 的 2/3/4-bit 均有明确重建表，属于逐级残差细化，不是把一个独立 2-bit 残差码任意丢弃一位。

LogResidual211 先构造 `u=sign(z/r)×log(1+alpha×abs(z/r))/log(1+alpha)`，其中 z/r 饱和到正负 1；在 u 域作同样的三级编码，再用指数逆变换恢复 z。alpha 只在校准上选择；正负截断端点显式固定到正负 1，避免数值舍入改变残差符号。

两种残差方法的修正幅度在每个 head/侧共享；不同于每个父节点都能独立选择两个子质心的树，表达能力受到额外约束。本轮分别借鉴 QuantSpec 的残差量化和 Lynx 的对数压扩，并未复现原生 4＋4-bit 配置、Lynx 的局部 Page/Chunk 统计与原生 Anchor/Residual 映射。这一差异是解释结果所必需的。

### 条件正则为什么不进入结果排名

Joint_Regularized 在校准 head 的归一化收益违反达到 0.01 时，固定分区调整质心，惩罚非单调和收益不递减，lambda=10、Adam 100 步、学习率 0.01。本轮触发 head 数为 0，因此跳过正式评估。它是预注册的条件候选，没有独立实测行，不能将跳过写成已经验证正则有效。

## 四 全部主渐进方法对比

下表为相同共享预处理、完整保存 4-bit 的 19 个候选；数值越低越好。原 A/B/C/D 分别对应 U_MinMax、Gaussian_Tree、Data_Tree、Jacobian_Tree，重复名称不重复计入。

{main}

![全部主候选的前缀误差](all_prefix_methods.png)

图中橙色为本次新增方案，绿色标出原始 Jacobian_Tree 与 JacK_MixV；横轴为对数尺度，数值越小越好。

JacK_MixV 三种精度均为主表最低，约为 0.4644/0.2393/0.1245。它相对 Data Tree 的相对 L2 改善约 18.0%/21.4%/17.2%，优于仅在 K 上使用简单 Score 目标的组合。3/4-bit 下，JacK_MSEV 也优于原 D，而 2-bit 略差；V 的目标选择确实会改变整体折中，不能简单认为 attention 权重越集中越好。

范围优化是均匀编码的关键。原 U_MinMax 的 2-bit 误差 2.2919，U_Clipped 降至 0.5680；TCN_MinMax 到 TCN_Clipped 也从 2.0540 降至 0.5659。仅将极值范围的均匀编码与数据树比较，会夸大由码点形状本身带来的优势。

对优化范围的均匀结构而言，TCN_Clipped 相对 U_Clipped 的 2-bit 基本持平，3/4-bit 改善约 6.5%/10.8%；Residual211 改善约 −1.6%/19.2%/33.5%。残差比普通均匀前缀更能利用细化位，但 4-bit 的 0.2043 仍明显高于 Data Tree 的 0.1503。LogResidual211 的 2-bit 改善不能延伸到 4-bit，支持将其视作粗精度折中的候选，而不是默认替代。

Leaf4_Only 的 3/4-bit 相对 L2 优于从粗到细的 Data Tree；Joint_234 仅获得较小改善。这个结果提示，共享多精度目标、低精度主导的损失量级和后续父分区冻结都会影响结果。当前联合优化的是标准化重建 MSE，并没有直接联合优化完整 attention 输出，因此不能据此否定多精度训练本身。

## 五 绝对误差与独立精度参考

新增实验能够直接提供以下七方法的绝对 MSE；历史其余方法的旧汇总只保留平均 RMSE，不能把“平均 RMSE 的平方”冒充“平均 MSE”，合并 CSV 对这些字段明确留空。

{mses}

4-bit 时，Jacobian 的平均相对 L2 更低，而 Data Tree 的绝对 MSE 更低。相对指标对参考输出较小的单元给予更高相对权重，绝对指标更受大幅度误差影响；必须先明确控制目标，再讨论哪种方法最优。当前“JacK_MixV 全主候选最好”也仅限定于已统一汇总的平均相对 L2，尚不是全部方法的绝对 MSE 排名。

独立精度参考如下，它们不属于同一份 4-bit 前缀编码：

{independent}

独立均匀 4-bit 的误差为 0.1901，显著低于其渐进 U_Clipped 的 0.3073；说明共享范围兼顾粗精度本身有代价。独立非均匀为 0.1382，也低于渐进 Data Tree 的 0.1503。两组只是在各自经验训练下估计嵌套约束代价，不是数学最优上限；三精度同时保留共 9-bit，不能列为主表同预算赢家。

## 六 KVarN 与历史等总字节条件

KVarN 原生路径为 FP16 Hadamard 旋转、128-token 块内行列方差均衡 8 次，再对 K 按通道、V 按 token 作非对称 INT4，并保留原生 absorbed scale、零点及另一轴因子。它在当前块计算局部统计，和主表共享校准状态不同。

E/E_zero 的未满块尾部保留 FP16 KV；KVARN_nt/KVARN_nt_zero 则对有效尾块也量化，单 token 退化块跳过无定义方差均衡，必要的 FP16 元数据保留。中点恢复取缺失细码区间中心，补零取下沿；2/3-bit 前缀是本实验的离线重建逻辑，不是 KVarN 原生 3-bit 内核。

{native}

去掉 FP16 尾部后误差增加，说明旧 E 的部分优势来自尾部保护。即使全部 KV 量化，KVarN 的低误差仍体现整套局部均衡与元数据方案的效果；不能由这张表宣称“均匀码点单独击败非均匀树”。例如完整 8192-token 配置的原生块元数据约 27 MiB，而共享树附加状态约 0.329 MiB，付出的状态与计算成本不同。

历史严格等总预算包装另列：

{eqtable}

A_eq/B_eq/C_eq/D_eq 根据共享 Gaussian 基准的总字节上限，确定性省略部分 Bit0 来支付额外元数据；E_eq/E_zero_eq 也按预算限制部分低位，4-bit 标称读取不保证所有元素都实际可得四位。B_eq 与 B 的部署状态相同，数值相同。它们保留为存储公平性诊断，不满足本轮每标量完整保存 4-bit 的主条件，也不是新的编码训练算法。

## 七 元数据和跨运行合并审计

以完整 8192-token、36 层 K/V、8 KV heads 为例，一份 4-bit payload 为 288 MiB。下表仅计部署一个校准种子的状态，不同时保存三个 bootstrap 模型，也不含 controller 失真曲线。

{table(['方法','完整 payload MiB','编码附加状态 KiB'],storage)}

新增参考实现实际保存各深度重建表和统一参数字段，因此按这些数组计费；TCN 的解析解码理论上可省去表，但本轮不把尚未实现的省表方案写入当前账本。Gaussian Tree 的历史码表是全局共享的一个 86-byte 表；逐 head 的经验树表则按实际数量计费。

合并从旧 17 方法去掉两类独立参考后得 15 个主候选，再加入四个新编码，共 19 个；三种重算基线只保留一份，并优先使用新增实验可用的更完整误差字段。全部 4050 个重复基线请求精度单元已核对，相对 L2 最大差为零；原始首请求另有 155520 个局部单元核对。完整结果保存在[合并汇总](all_shared_methods.csv)、[合并请求数据](all_request_metrics.csv)、[方法实现索引](method_implementations.csv)与[合并审计](merge_audit.json)。

## 八 研究判断与后续实验

本轮证据更支持非均匀经验树和功能敏感性目标：2-bit 时优化范围后的均匀编码仍是有竞争力的低成本对照，3/4-bit 则应保留 Data Tree 与 Jacobian 系列作为主要候选。新增残差编码可以作为结构更简单的备选，但它的理论或实现复杂度优势仍需真实内核验证，不能由误差表推断速度。

controller 与编码需要分开评估。原始 Jacobian_Tree 已固定用于[90 请求因果回放](../controller_jacobian_20261026/results/replay/结果分析.md)：平均 3-bit 下，相对均匀读取的真实输出 MSE 降低 42.6%，但相对仅 Query 敏感性只额外改善约 3%。[12 请求在线 pilot](../controller_jacobian_20261026/results/online/结果分析.md)出现平均改善和逐请求退化并存，尚不能替代官方任务质量评测。

优先继续验证真实输出误差和任务结果，而不是只追求更低的分离代理。V 跨 token 交叉项已经表现出明显影响；未来若按误差阈值控制读取，需验证校准失配和残差相关性。扩大独立任务、模型和自然生成长度，再将确定有价值的策略接入实际位平面布局，测量包含分配、两阶段读取及 attention 的总延迟。

## 九 实现与原始资料

原树与前缀实现：[codec.py](../encoding_ablation_20261024/src/codec.py)；目标、联合优化与组合：[books.py](../encoding_ablation_20261024/supplementary_experiments/src/books.py)和[suite.py](../encoding_ablation_20261024/supplementary_experiments/src/suite.py)；新增编码：[codecs4.py](../encoding_extension_20261026/src/codecs4.py)；原生 KVarN 包装：[native.py](../encoding_ablation_20261024/E_extension/src/native.py)；去尾部保护：[native_nofp16.py](../encoding_ablation_20261024/supplementary_experiments/src/native_nofp16.py)。

原结果索引：[历史汇总](../encoding_ablation_20261024/实验结果汇总.md)、[历史补充指标](../encoding_ablation_20261024/supplementary_experiments/metrics/attention_errors.csv)、[本次编码报告](../encoding_extension_20261026/results/full/结果分析.md)。本报告只整理冻结结果，无新增 GPU 实验，保留原始实验源码与运行 provenance。[重建脚本](build_reports.py)生成表格、图和扩充报告，文档修改前的自动报告保存在 auto_report_snapshots。
""")


def plot(index):
    sys.path.insert(0,str(S/'deps'))
    os.environ['MPLCONFIGDIR']=str(R/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,9),sharey=True)
    yy=np.arange(len(MAIN))
    for ax,b in zip(axes,(2,3,4)):
        values=[index[m,b,b]['output_rel'] for m in MAIN]
        colors=['#dc8e35' if m in NEW else '#249582' if m in ('Jacobian_Tree','JacK_MixV') else '#6886ad' for m in MAIN]
        ax.scatter(values,yy,c=colors,s=42,zorder=3)
        for value,y in zip(values,yy):
            ax.annotate(f'{value:.3f}',(value,y),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
        ax.set_yticks(yy,MAIN);ax.set_xscale('log');ax.set_xlim(min(values)*.9,max(values)*1.3);ax.set_title(f'{b}-bit read; all store 4 bits');ax.set_xlabel('Mean attention output relative L2');ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(R/'all_prefix_methods.png',dpi=180);plt.close(fig)


def main():
    snapshots=R/'auto_report_snapshots';snapshots.mkdir(exist_ok=True)
    for name,path in [('encoding',N/'结果分析.md'),('replay',CR/'结果分析.md'),('online',CO/'结果分析.md')]:
        dest=snapshots/(name+'.md')
        if not dest.exists():shutil.copyfile(path,dest)
    index,new,audit=collect()
    # All source metrics have been loaded and validated before writing reports.
    encoding_report(index,new);replay_report();audit['online_paired_exploratory']=online_report();combined_report(index);plot(index)
    sources=[S/'metrics/attention_errors.csv',S/'metrics/request_metrics.csv',N/'summary.csv',N/'request_metrics.csv',N/'comparisons.json',CR/'summary.csv',CR/'comparisons.json',CR/'proxy_pairwise_ranking.json',CO/'complete.json',B/'results/unified/summary_A_to_E.csv',S/'metrics/native_nt/request_metrics.csv']
    audit['source_hashes']={str(p.relative_to(R.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    audit['reports']=[str(p.relative_to(R.parent)) for p in [R/'编码实验综合分析.md',N/'结果分析.md',CR/'结果分析.md',CO/'结果分析.md']]
    (R/'merge_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='source_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
