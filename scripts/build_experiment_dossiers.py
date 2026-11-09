"""Build per-experiment dossiers from archived local evidence. CPU only."""
import collections
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
LEG = ROOT / 'results/experiment_record_20261109_legacy'
OLD = LEG / 'sources'
SNAP = ROOT / 'results/experiment_record_20261109'
MODEL = 'Qwen/Qwen3-4B；revision 1cfa9a7208912126459214e8b04321603b3df60c；36层、32 Q heads、8 KV heads、head_dim128。'
GPU = 'RTX 3090 24GiB（SM86）；服务conda kvarn-serve；PyTorch 2.11.0+cu126、CUDA12.6；客户端evalscope-client。各批次实际启动参数和源码hash以所附plan/suite为准。'
CPU = 'CPU NumPy float64离线重放，BLAS默认4线程；输入由RTX3090上的conda kvarn-serve原生KVarN服务采集；重放阶段不重新做模型生成。'
MODERN = 'RTX3090 24GiB；项目.venv，Python3.12.13、PyTorch2.11.0+cu126、Transformers4.57.6、CUDA12.6；FP16模型、FP32 attention参考、TF32关闭。'
OFFDATA = 'LongBench六任务qasper/hotpotqa/gov_report/trec/passage_retrieval_en/lcc；原始180请求，60校准/30验证/90测试；本项执行范围如下单列。'
OFFCFG = '完整36层×8 KV heads；decode1/8/32/64；FP16参考固定64步；prompt最多8192，长输入首尾截断；prefill512分块；固定Hadamard-sign旋转与校准逐channel mean/scale；学习码本种子20261024/25/26。'
OFFMET = 'K/V相对平方失真与raw MSE；score RMSE/MSE/relative L2；probability L1；output relative L2/RMSE及可用absolute MSE；请求mean/P95、配对bootstrap95%CI；层/head/任务/长度分层；存储与逻辑读取账本。精确列名以随附CSV为准。'
records = []


def read(rel, legacy=False):
    return json.loads(((OLD if legacy else ROOT) / rel).read_text())


def csvrows(rel, legacy=False):
    with ((OLD if legacy else ROOT) / rel).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def link(path):
    p = Path(path)
    if p.is_absolute() and p.is_relative_to(ROOT):
        p = p.relative_to(ROOT)
    return '[' + p.name + '](' + str(p) + ')'


def legacy(rel):
    return str((OLD / rel).relative_to(ROOT))


def table(headers, rows):
    def cell(x):
        return str(x).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['|' + '|'.join(map(cell, headers)) + '|',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['|' + '|'.join(map(cell, row)) + '|' for row in rows]) + '\n\n'


def add(id, group, name, purpose, env, data, config, methods, metrics, status,
        conclusion, sources, result='', **extra):
    records.append(dict(id=id, group=group, name=name, purpose=purpose,
        model=MODEL, environment=env, data=data, configuration=config,
        compared_methods=methods, recorded_metrics=metrics, completion=status,
        results=result, conclusion=conclusion, evidence=sources, **extra))


def off(id, name, purpose, config, methods, status, conclusion, sources, result='', metrics=OFFMET):
    add(id,'三 10月23日至26日离线编码与Controller',name,purpose,MODERN,OFFDATA,
        OFFCFG+' '+config,methods,metrics,status,conclusion,sources,result)


def build():
    # Historical offline experiments: each is a separate question and evidence unit.
    diag='results/residual_diag/run_20261012_161428_3993496'
    add('H01','一 前置项目历史离线与机制实验','10月12日真实decode跳读诊断',
        '解释Anchor–Residual误差界为何没有转化为有效跳读，区分K/V阈值、GQA并集阻断及split保守性。',
        GPU+' GPU原AR Triton重放与CPU float64策略参考对照；采集C1 eager、APC关闭。',
        'qasper/hotpotqa/passage_retrieval_en各2条，共6请求；cache ordinal0/12/24/35，decode调用0/15/63；72真实快照。',
        '原生K4V4/G128；诊断生成cap96、自然EOS；沿用上游thinking及temperature0.6/top_p0.95/top_k20/min_p0；epsilon0.02；8 KV heads；保留FP16 sink/tail；原BLOCK_N/splits。',
        'bound_k、bound_v、bound_kv；完整量化attention参考；CPU/GPU分支对照。',
        '全部CPU/GPU skipped chunks、logical code/scale/FP16 pool bytes、阈值比例P50/P95、GQA blocked chunks、CPU/GPU计数一致性、绝对/相对L2及max-head/max-element误差、数值门槛。',
        '完成72 traces×3策略=216行。',
        'K完全没有跳过；V也只减少很小的逻辑量，支持旧误差界过于保守的诊断。不是各策略自然生成全程跳读率，不提供DRAM或TPOT结论。',
        [legacy(diag+'/plan.json'),legacy(diag+'/experiment1/summary.csv'),legacy(diag+'/experiment1/readout.md')],
        table(['策略','GPU跳过K chunks','GPU跳过V chunks','平均逻辑码减少'],
              [['bound_k',0,0,'0%'],['bound_v',0,4658,'0.2655%'],['bound_kv',0,3249,'0.1831%']]))
    add('H02','一 前置项目历史离线与机制实验','10月12日Oracle读取量与误差曲线',
        '检验真实激活上是否存在低读取、低误差的选择空间，判断保守bound失败是控制器限制还是该读取预算本身不可行。',
        CPU,'H01中均匀抽取12个真实trace；KV heads0/3/7，共36个GQA组。',
        '原生K4V4、高2位anchor始终读取；FP16 sink/tail全读；残差预算0/10/25/50/75/100%；完整量化attention为参考。',
        'V-only、KV；current_bound_cpu、offline_greedy、best_feasible_envelope；小规模exhaustive_exact；V certified lower bound。',
        '可控/全部残差读取比例、逻辑code减少、绝对/相对输出误差、max-head误差、可行包络、V有效下界、小问题精确枚举及NA字段。',
        '完成12 traces、2664结果行。',
        'V在约10%可控残差读取时可行包络max-head误差中位数约0.00490，而当前bound读约99.14%；说明存在选择空间但未得到廉价在线算法。可行包络是最优误差上界，不能叫全局最优或信息论下界。',
        [legacy(diag+'/experiment2/readout.md'),legacy(diag+'/experiment2/aggregate.csv')],
        table(['模式','残差预算','实际可控读取','逻辑code减少','最大头relative误差中位数'],
              [['V',.1,.09921,.22520,.00490],['V',.5,.49835,.12541,.00130],['KV',.1,.09905,.45047,.01364],['KV',.5,.49874,.25063,.00282]]))
    asym='results/asym_equal_bytes/longbench6x10_20261014'
    add('H03','一 前置项目历史离线与机制实验','10月14日K V同逻辑字节非对称分配',
        '在完全相同逻辑code读取字节下，判断残差预算应该偏K、对称还是偏V，并区分分配方式与总流量的影响。',
        CPU,'narrativeqa/qasper/gov_report/trec/passage_retrieval_en/lcc各10题，共60题、240traces；全部8KV heads。',
        'C1 eager采集；cache层0/35、decode调用0/15；诊断生成cap32；overflow=head-tail；tile32；总残差率1.0/1.1；K priority0/.25/.5/.75/1；mask seeds42/43/44；请求bootstrap1000次。',
        'anchor_mass、staged_mass、random；同总字节的K/V priority；对称priority0.5为配对基线。',
        '逐题/逐任务mean error、P95、informative groups、payload reduction、paired difference/CI、采集覆盖/缺失点、截断标志；全部值在aggregate/case_level/paired CSV。',
        '完成60/60题、240traces。',
        'anchor_mass在总残差率1.0时，六任务等权平均误差以对称0.5最低；两端极偏置明显变差。仅为此面板/策略的反事实结果，不提供非对称策略任务分数或硬件流量。',
        [legacy(asym+'/plan.json'),legacy(asym+'/comparison/aggregate.csv'),legacy(asym+'/comparison/readout.md')],
        table(['K priority','六任务等权mean case error（本次从aggregate重算）'],
              [[0,.09083040],[.25,.01701347],[.5,.01165820],[.75,.01801553],[1,.06772013]]))
    add('H04','一 前置项目历史离线与机制实验','10月13至14日Query-aware阈值控制与误停率',
        '检验Anchor+metadata误差代理能否在指定容忍度下少读，记录真实误差超阈值的underread而不是只看预测值。',
        CPU,'10月13日calibration批次105题（narrativeqa）；10月14日六任务各10题。两批分别汇总，不合并成独立验证。',
        'tolerance=.01/.02/.05，margin1.25；同Parent轨迹；Query-aware rho(q)；C1 eager采集；具体层/步、截断与输入见各plan。',
        '固定分配/对称参照与Query-aware控制；三个容忍度全部保留。',
        'controller_estimated_error、true error mean/P95/max、rho、controller_steps、stop_reason、residual bytes、underread_rate及case级数据。',
        '105题校准批与60题面板完成；schemafix3题/采集fix1题仅为门槛。',
        '更松容忍度降低读取但真实误差不受严格保证。例如六任务qasper tolerance.01仍有2.5%局部误停，不能把该代理写成真实输出误差认证。',
        [legacy('results/asym_equal_bytes/controller_calibration_20261013/comparison/controller_aggregate.csv'),legacy(asym+'/comparison/controller_aggregate.csv')],
        table(['narrativeqa校准105题 tolerance','真实误差均值','rho均值','误停率'],
              [[.01,.00143747,.616370,.00595238],[.02,.00266085,.566919,.000892857],[.05,.00610411,.490809,.000297619]]))
    add('H05','一 前置项目历史离线与机制实验','10月12至15日非对称全量尝试与margin校准',
        '扩大H03/H04任务覆盖，并检查统一margin的跨任务校准；记录运行失败与报告滞后对覆盖结论的影响。',
        CPU,'LongBench21原4750题计划；另有全量controller与六任务margin校准。',
        '原生Parent采集、离线float64；与各plan的层/步和短生成cap匹配，不是全长任务评测。',
        'anchor/staged/random与同字节对称；Query-aware tolerance/margin。',
        'planned/completed/committed cases、trace count、case coverage、paired errors/CI、underread、故障与源码版本。',
        '全21任务批status完成1235后失败，但冻结comparison report只有437题/5244traces；全量controller86题后中断；margin校准25题后server退出。',
        '必须区分采集进度与已进入对比表的覆盖；没有4750题最终结论。端口/GPU占用、schema错误等失败重试均见历史运行索引，不当作独立成功实验。',
        [legacy('results/asym_equal_bytes/longbench21_20261012_210330_1427845/status.json'),legacy('results/asym_equal_bytes/longbench21_20261012_210330_1427845/comparison/report.json'),legacy('results/asym_equal_bytes/longbench6x10_margin_calibration_20261015/status.json')])
    prog='results/progressive_online/'
    add('H06','一 前置项目历史离线与机制实验','10月16日布局开销与时间复用界检查',
        '分离位平面布局和Controller本身的额外开销，并检验上一token误差能否无条件复用为当前Query保证。',
        GPU+' 数学反例和已有同任务在线测量分析；不是新独立全量质量测评。',
        '既有六任务面板；reuse_bound的受控Query反例。',
        '相同任务对齐Parent/full/control；比较unscaled metadata key error与当前query-scaled score bound；未实现正式reuse。',
        'Parent、layout full、control full；不加Query权重/加入Query权重的界。',
        '各任务TPOT/layout slowdown/control slowdown；metadata error、actual score error、query bound、bound_holds及reuse_implemented。',
        '检查记录已完成；reuse_accuracy后续冗余sweep主动停止并保留结果。',
        '仅缓存误差不是当前Q下的score误差界。示例metadata error1.5，对应score error169.7056，只有乘Query敏感性后的界成立；不能据此宣布时间复用已经有效。',
        [legacy(prog+'layout_control_20261016_144217_176289/matched_overhead_check.json'),legacy(prog+'layout_control_20261016_144217_176289/reuse_bound_validation.json'),legacy(prog+'reuse_accuracy_20261016_154258_188087/stop_reason.json')])
    add('H07','一 前置项目历史离线与机制实验','10月16日真实输入组件profiling与元数据反例',
        '定位布局、控制判断、打包和元数据成本，并独立验证writer生成的码格式是否与metadata风险定义一致。',
        GPU+' 固定输入/固定split的Triton重放；CPU/CUDA时间线；ncu计数器探针。',
        'qasper/hotpotqa/passage_retrieval_en真实trace；同Q/KV和geometry。',
        '固定split配置；layout_full、control_full、adaptive_s1与Parent；真实writer输出单独验metadata。',
        'native/layout/control/adaptive；格式界与独立真实误差oracle。',
        'kernel us、layout/control slowdown、packing/metadata kernel数、寄存器等线索、counter_access状态、metadata bound与actual error。',
        '组件诊断完成；DRAM计数器无权限。',
        '发现K2 metadata界0.5小于实际误差1.5的格式反例，旧Controller风险承诺不成立。固定输入layout慢约20.6%—25.0%，额外control又慢约62.7%—80.2%；应先修格式。不能把寄存器增多直接判作spill根因。',
        [legacy(prog+'component_profile_20261016_155242_075020/analysis_decision.json'),legacy(prog+'component_profile_20261016_155242_075020/metadata_format_counterexample.json'),legacy(prog+'component_profile_20261016_155242_075020/counter_access.json')])
    add('H08','一 前置项目历史离线与机制实验','10月16日writer与metadata契约修复验证',
        '确认真实writer位布局、metadata误差界与reader精度决策一致，避免在有格式错误时解释质量或速度收益。',
        GPU+' 私有runtime；实际backend flush和CUDA Graph小规模验证。',
        '3次真实flush，每次16tiles；受控2/3/4位输入与物理页改写/清零。',
        'K4V4原始量化码不变；修复metadata读取码位关系；单/分裂GPU选择、Graph replay。',
        '真实writer vs native oracle；metadata推导 vs独立实数误差；2/3/4位。',
        '20单元测试pass/fail、writes/tiles、max_metadata_difference、实际GPU precision selected-score error、rewrite/zero检查与源码hash。',
        '最终162114批完成，20测试通过，max metadata difference0；前两个尝试状态另存。',
        '仅通过小规模实现契约门槛，未重跑旧全量在线质量；因此不能追溯性地为修复前的自适应结果背书。',
        [legacy(prog+'metadata_contract_fix_20261016_162114_196952/verification_receipt.json')])
    add('H09','一 前置项目历史离线与机制实验','10月16日固定深度与K V控制算子消融',
        '在同输入同geometry下拆分位宽节省和运行控制逻辑的代价，确定K-only/V-only两侧是否都值得动态化。',
        GPU+' Triton fixed-input replay；计数与计时分开；无DRAM权限。',
        '三条真实快照；Qasper/HotpotQA/检索；原始采集K4V4。',
        'fixed K2V2/K3V3/K4V4/K2V4/K3V4/K4V2/K4V3、K/V adaptive/control强制全读；匹配split原生对照。',
        '13固定/控制变体与native_parent；每侧独立消融。',
        'round median us、logical depth counts、relative slowdown、correctness、kernel resources、profile coverage。',
        '163034批完成；更早两次源码匹配失败不计结果。',
        'K几乎全选4位、控制本身约增15%；V多数选2位仍慢于full平面；Qasper固定K2V2仅比native快约3.14%，不足一般在线结论。',
        [legacy(prog+'operator_ablation_20261016_163034_287851/variants.json'),legacy(prog+'operator_ablation_20261016_163034_287851/timings.csv'),legacy(prog+'operator_ablation_20261016_163034_287851/decision.json')])
    add('H10','一 前置项目历史离线与机制实验','10月16日融合packer与固定reader筛选',
        '先验证写端融合能否减少launch与分配，再检查固定低精度reader是否有足够性能余量支撑动态控制。',
        GPU+' CUDA events的eager/Graph独立计时，真实backend CPU/CUDA时间线。',
        'uint8/int32；1/8/64/256tiles；三真实快照、B1/复制B8；两个物理页flush。',
        '融合位平面pack与K metadata；不融合Sinkhorn/RTN/scatter；reader split32、BLOCK_N16/32/64；预设speedup1.10门槛。',
        'torch_pack vs fused_pack，metadata交叉；native、K4V4、K4V2、K2V2固定reader。',
        '字节一致/max error；CPU enqueue、eager/Graph median/min/max/sample us；kernel count/归因；reader speedup与gate。',
        'operator gates完成；不代表整个online continuation完成。',
        '真实flush排除JIT后GPU kernel由196降至44；仅固定K2V2过全部筛选形状10%工程门槛。复制B8共享同KV有缓存局部性偏差，不能当在线C8。',
        [legacy(prog+'staged_optimization_20261016_174407_183860/plan.json'),legacy(prog+'staged_optimization_20261016_174407_183860/writer_checks.json'),legacy(prog+'staged_optimization_20261016_174407_183860/writer_timings.json'),legacy(prog+'staged_optimization_20261016_174407_183860/reader_decision.json')])
    add('H11','一 前置项目历史离线与机制实验','10月16至17日Query选择器同预算性能门槛',
        '判断固定低位读取的kernel空间能否覆盖GPU selector开销，并与相同预算静态分配公平比较。',
        GPU+' 同输入算子probe和独立归因；后续quality阶段另列。',
        '三个真实快照；K为8个KV heads中2个全4位、其余2位；V固定2位。',
        '一个快照校准、两个验证的设计；当前Query及校准敏感性驱动selector；selector计入总时间；10%速度门槛。',
        'native、fixed4、fixed2、static_same_budget、query_adaptive_including_selector。',
        '逐形状speedup_vs_best_native、passes_10pct、capture/operator/attribution/quality_fixed阶段returncode。',
        'capture/operator/attribution完成；quality_fixed失败；总状态completed_with_failures。',
        'fixedK2V2约1.206/1.215/1.224倍过门槛；包括selector的动态组仅0.910/0.891/0.895倍，未兑现速度。quality_fixed中的p211_adaptive标签实际代表固定K2V2，不能按名字误归为动态质量实验。',
        [legacy(prog+'autonomous_next_20261016_235221/final_analysis.json')])
    add('H12','一 前置项目历史离线与机制实验','10月10日Anchor Residual固定形状GPU微基准',
        '隔离重排布局、全部anchor、误差界条件分支、split合并和打包的真实kernel成本，为完整服务结果寻找原因。',
        GPU+' Triton GPU microbench；正式timing kernel不带统计；逻辑字节来自另一次执行。',
        '合成KV固定形状B1/8，L1024/4096/8192/16384/32768；dominant为显式分支压力测试。',
        '原生K4V4；single/split；epsilon.02，joint两侧各.01；不含Q/output旋转、store、prefill与HTTP；同缓存反复访问可能命中L2。',
        'parent、repacked_full、anchor、bound_k、bound_v、bound_kv；native/tile_high2_low2 packer。',
        'median kernel us、相对Parent/full速度、GPU分支logical code bytes/reduction；packer us/输出字节；correctness。',
        'GPU执行完成，慢策略保留。',
        '合成默认随机形状bound组没有逻辑跳读且更慢；anchor可少读但无质量保证。B1L8192 split：native96.730us、planar127.840us、anchor85.062us、boundKV167.424us。64tile pack由16.486增至35.686us，输出字节相同524288。',
        [legacy('results/residual_v1/full_20261010_160834_3993496/microbench/readout.md'),legacy('results/residual_v1/full_20261010_160834_3993496/microbench/benchmark.json')])

    budget=prog+'staged_optimization_20261016_174407_183860/budget_recovery_20261016/'
    br=read(budget+'timings.json',True)
    add('H13','一 前置项目历史离线与机制实验','10月16日独立校准与留出Query预算恢复试验',
        '在排除校准快照的两个留出真实快照上，测试同预算Query选择能否优于静态选择，并把selector成本算入kernel。',
        GPU+' B1固定输入CUDA Graph算子测量，不进行模型闭环生成。',
        'Qasper一快照校准；HotpotQA和passage_retrieval_en两个留出快照。',
        'K恰好2/8heads读4位，其余K2，V固定2；split32、block64；Query平方范数×校准K2残差包络；不时间复用；3轮median。',
        'native、fixed_k2v2、static_same_budget、query_adaptive_including_selector。',
        'round_medians_us、relative_L2_vs_native、query/static plan、校准head_weights、amplified_head响应正确性；payload预算与限制。',
        '独立recovery完成；与H11后续三快照probe是不同批次。',
        '留出HotpotQA动态191.522us快于静态198.449us但误差.010139高于.009781；检索动态159.130us略慢于静态157.622us且误差相同。仅两个留出快照，没有稳定的质量速度双优结论。',
        [legacy(budget+'calibration.json'),legacy(budget+'timings.json'),legacy(budget+'correctness.json'),legacy(budget+'status.json')],
        table(['数据集','变体','3轮median us','relative L2 vs native'],
              [[x['dataset'],x['variant'],f"{statistics.median(x['round_medians_us']):.3f}",f"{x['relative_l2_vs_native']:.8f}"] for x in br]))

    # Earlier online, accuracy and performance experiments retained separately.
    real='results/real/'
    for id,run,count,config in [
        ('P00a','pipeline01/pilot',18,'C2/C4/C8；HumanEval上限8192、数学16384'),
        ('P00b','pipeline02_unbounded/pilot',6,'C8；max_tokens=null、取消数据集特定输出上限')]:
        add(id,'二 前置项目原生质量与系统实验','旧原生32题'+('固定输出预算pilot' if id=='P00a' else '无额外输出cap pilot'),
            '验证原生服务端到客户端协议、并发和资源配置；第二批专门核实取消输出cap后的预算行为。',GPU,
            'HumanEval32、MATH32的pilot子集；'+str(count)+'point，每point32请求。',
            config+'；32768窗口，max_seqs16、batch_tokens1024、APC off；thinking、temperature.6/top_p.95/top_k20、seed_base20260918、自然EOS。',
            'FP16、KVarN K4V2/G128、TurboQuant K4V4；第一批比较并发，第二批检查remaining-context。',
            'requests_expected/recorded/succeeded/failed、response完整性、latency/TTFT/TPOT分布、rps/output_tok_s、长度分布、EOS/length、preemptions、KV容量、GPU显存/温度/功耗/频率采样、recommendations。',
            str(count)+'point全部status=ok；summary未包含完整官方质量评分，不补造正确率。',
            '完成链路/资源pilot，不能代替全HumanEval/MATH质量；有输出cap与null两个协议单独存档，不能合并计算速度变化。',
            [legacy(real+run+'/suite.json'),legacy(real+run+'/summary.json'),legacy(real+run+'/recommendations.json')])
    realrows=read(real+'full_v7_C8/summary.json',True)+read(real+'precision_v8_C8/summary.json',True)
    rt=table(['方法','任务','成功/应有','评分完整','正确题','任务分%','输出tok/s'],
        [[x['method'],x['dataset'],str(x.get('succeeded'))+'/'+str(x.get('requests_expected')),
          x.get('quality_scoring_complete'),x.get('quality_correct_tasks'),
          round(x['accuracy_all_tasks']*100,4) if x.get('accuracy_all_tasks') is not None else '缺失',
          round(x['output_tok_s'],3) if x.get('output_tok_s') is not None else '缺失'] for x in realrows])
    add('P01','二 前置项目原生质量与系统实验','旧原生HumanEval MATH精度与group sweep',
        '比较原生FP16/FP8/TurboQuant/KVarN的任务质量、在线延迟吞吐和KV容量，并单独观察group64与128的差异。',
        GPU+' Qwen原生thinking；服务端和客户端两套conda环境，源码commit7586257f1。',
        'HumanEval164、MATH500；主三方法各两任务完整；扩展7别名共14point，13成功1失败；protocol预检另有FP16数学8题C2。',
        'C8、max_model_len32768、max_seqs16、batch_tokens1024、GPU-util.85、APC关闭；max_tokens传JSON null、仅上下文边界；temperature.6/top_p.95/top_k20，seed_base20260918；自然EOS。',
        'FP16；FP8 e4m3动态scales；TQ4/TQ3V4/TQ3/TQ8V4；KVarN K4V2/K4V4×G64/G128。',
        '每题成功/完整性/原生usage、HumanEval pass@1与数学评分、correct tasks/s、在线latency/TTFT/TPOT分布、rps/output tok/s、输入/输出长度分布、preemptions、KV容量与prefix指标、GPU显存/利用率/温度/功耗/时钟采样。',
        '主6point完成；扩展13完成、tq84数学超时失败。',
        'TQ8V4 HumanEval为0/164、数学失败，不能丢掉异常组再给全方法健康结论；group变化不仅是位宽变化。各策略单次自然生成、后端可能不同，不据此宣称无损或纯dtype速度因果。',
        [legacy(real+'full_v7_C8/suite.json'),legacy(real+'full_v7_C8/summary.json'),legacy(real+'precision_v8_C8/suite.json'),legacy(real+'precision_v8_C8/summary.json')],rt)
    lc='results/longcontext_v3/full_longbench_C8_20261009_201156_2217646'
    lcr=csvrows(lc+'/comparison/overview.csv',True)
    add('P02','二 前置项目原生质量与系统实验','10月9至10日九方法六任务长上下文自然生成',
        '验证多种原生KV格式在真实长输入、自然thinking输出下的质量与服务性能；区别固定输出微基准。',
        GPU,'qasper200、multifieldqa_en150、hotpotqa200、2wikimqa200、gov_report197、passage_retrieval_en200；总1147/方法；gov_report3题超窗口排除。',
        'C8、repeat1、32768总窗口、max_seqs16、batch_tokens1024、GPU-util.85、APC off；thinking采样.6/.95/20/min_p0；data_seed42、request_seed20261009、server_seed20260917；max_tokens=null、request_timeout0。',
        'fp16/fp8/tq4/tq34/tq3/kvarn/kvarn44/kvarn64/kvarn4464。',
        '官方逐任务score、response/evaluation完整性、输入输出长度、TTFT/TPOT/E2E均值与尾部、rps/tok/s、length cap/empty final/预算一致性、preemptions、GPU-wide采样与KV容量。',
        '9方法各6point、1147评分全部完成；合计10323生成。',
        '这是1147条六任务有效子集，不是LongBench21全4750题。六任务TPOT最低分别为FP8/FP8/FP16/FP8/FP8/FP16；质量最高方法随任务变化，没有统一赢家。不同后端、自然输出长度和单次运行不支持纯位宽因果或等价结论。',
        [legacy(lc+'/suite.json'),legacy(lc+'/suite_status.md'),legacy(lc+'/comparison/overview.csv')],
        table(['方法','任务','评分题数','官方分/100','TTFT ms','TPOT ms','输出tok/s','平均输出token','触顶数'],
              [[x['method'],x['dataset'],x['scored_tasks']]+[f"{float(x[k]):.3f}" for k in ['task_score_0_100','ttft_ms_mean','tpot_ms_mean','output_tokens_per_s','output_tokens_mean']]+[x['length_capped']] for x in lcr]))
    ar='results/residual_v1/full_20261010_160834_3993496'
    add('P03','二 前置项目原生质量与系统实验','10月10至11日Anchor Residual六策略闭环',
        '检验High2+Low2重排和按误差界补读在完整生成中的净效果，分开表示开销、激进anchor和有界K/V策略。',
        GPU,'复用P02完全相同六任务1147题/策略；六策略共6882评分，先前原生结果作为冻结参照。',
        '与P02相同C8、自然thinking采样、32768窗口；原生K4V4/G128；epsilon.02，joint K/V各.01；保留FP16 sink/tail、原生scale；新的私有runtime。',
        'AR parent、repacked_full、anchor、bound_k、bound_v、bound_kv；另对照P02原生方法。',
        '任务score及Δpp、TPOT/吞吐/E2E/输出长度变化、完整性/截断/空答案、KV容量、实现验证、packing costs与microbench逻辑量。',
        '六策略各1147条全部完成。',
        'boundKV相对Parent六任务TPOT改善均为负，约−21.77%至−46.82%；全读重排也退化约15.63%—19.68%。anchor速度较好但检索分数下降3.83pp；不能用少读码字直接保证服务收益。',
        [legacy(ar+'/suite.json'),legacy(ar+'/comparison/overview.csv'),legacy(ar+'/comparison/residual_effects.csv'),legacy(ar+'/comparison/residual_effects.md')])
    corrected=csvrows('results/experiment_record_20261109_legacy/progressive_online_same_dataset.csv')
    ct=[x for x in corrected if x['run']=='longbench20_each_20261016_111657_1148280']
    add('P04','二 前置项目原生质量与系统实验','10月15至16日Progressive KV三策略六任务',
        '判断2+1+1平面表示及在线自适应读取能否在原生服务中兼顾质量与性能，full组用于隔离布局成本。',
        GPU,'六LongBench任务每任务20题；parent/full/adaptive，每批360条；10月15、16两次完整批次分别报告，另一次启动失败。',
        'C8、同原生K4V4容量、自然生成；各settings/plan保存完整sampling与输出条件；正式SSE请求，GPU实算；不是离线Oracle。',
        'parent、p211_full、p211_adaptive。',
        'task score、mean/P95 TPOT、TTFT、E2E、output tok/s、output length、length cap、preemptions、native KV容量、GPU采样峰值及逐题quality配对。',
        '两个360请求批次完成；修复前metadata风险定义有已知问题，不能当正确风险认证。',
        '同任务重算后，full/adaptive六任务均比parent慢。旧comparison的部分Δ列错误使用首任务Parent作基线；本表保留正确原始均值，只使用新增same_dataset列计算差值。修复前结果仅描述当时实现。',
        [legacy(prog+'longbench20_each_20261015_145645_215162/settings.json'),legacy(prog+'longbench20_each_20261016_111657_1148280/settings.json'),'results/experiment_record_20261109_legacy/progressive_online_same_dataset.csv'],
        table(['10月16批方法','任务','score','TPOT ms','同任务TPOT改善%','同任务score Δpp'],
              [[x['arm'],x['dataset'],f"{float(x['task_score_0_100']):.3f}",f"{float(x['tpot_ms_mean']):.3f}",f"{float(x['recomputed_tpot_improvement_pct_same_dataset']):.3f}",f"{float(x['recomputed_score_difference_pp_same_dataset']):.3f}"] for x in ct]))
    syscards=[
      ('P05','reader_online_20261016_164703_433695','固定reader调参在线诊断','验证固定输入reader的局部优化是否在真实服务中仍有效。','三arms×Qasper/HotpotQA×C1/C8×两轮×8题=192请求；固定128输出。','planar_default、planar_tuned、native_tuned；全部输出128、preemptions0。','调整planar reader未显示统一收益，例如Qasper C1 TPOT14.562→14.983ms，原生14.300ms；不是任务质量实验。',['online_plan.json','online_comparison.csv','online_findings_20261016_172916.json']),
      ('P06','alternating_writer_20261016_192606_193596','三writer交错顺序对照','降低顺序漂移的干扰，测试原生/平面/融合writer完整实现的在线成本。','3顺序轮转、9服务会话；Qasper/HotpotQA各8题、C8、输出64；144正式请求。','native、planar、fused；writer byte gate先过，保留默认reader。','各轮改善方向不一致，未形成稳定通用加速；64-token诊断不能作为长回答任务得分。',['plan.json','writer_byte_gate.json','comparison.json','readout.json','generation_agreement.json']),
      ('P07','fixed_recovery_20261017_113242','固定低位质量与profile恢复','补齐前一quality_fixed失败后的记录，并确认巨大trace是否完整和可解析。','同plan中的Qasper/HotpotQA面板；profile与quality分开，读取实际恢复状态。','固定K2V2融合writer候选、原生和full参照；不是名称所暗示的动态组。','profile有效性与生成成功只是证据完整性；不能自动归为已验证动态Controller质量收益。',['plan.json','status.json','profile_audit.json']),
      ('P08','decisive_20261017_125900','1024步决定性在线性能门槛','检验固定低位reader在完整服务decode重叠区间是否有至少10%稳定收益。','Qasper/HotpotQA各8题，C1/C8，三顺序，固定1024、greedy、ignoreEOS。','native、fused_full、fused_k2v2；相同full writer，不带adaptive metadata。','总decision=inconclusive；C1低位speedup约0.982/0.931，均未达1.10；Qasper C8重叠无效为null。不能写成在线加速成功。',['plan.json','decision.json','measurements.json','fixed_source_gate.json']),
      ('P09','global_split_20261017_232616','全局位区域split布局诊断','隔离全局平面布局/scatter成本与降低读取深度的收益。','五arms×两任务×C1/C8×3轮=60point；每point8题、固定512步、greedy忽略EOS。','native、tile_full、split_full、tile_k2、split_k2；相同精度对比布局，无adaptive控制；8个不重叠全局区域、无shadow。','60point完成；split_k2虽优于split_full，但相对native四单元TPOT仍增加约2.24%/4.17%/8.82%/7.75%；少读未覆盖系统开销。',['plan.json','byte_gate.json','comparison.json','attribution.json','measurements.json'])]
    for id,run,name,purpose,data,config,conclusion,files in syscards:
        add(id,'二 前置项目原生质量与系统实验','10月16至18日'+name,purpose,GPU, data,config,
            config.split('；')[0], 'TPOT/TTFT/E2E、common overlap decode、吞吐、输出长度、preemptions、有效请求/覆盖、各轮变化；writer/byte/correctness gate、源码hash；GPU利用率/时钟和profile在存在时保留。',
            read(prog+run+'/status.json',True).get('status','见online_status') if (OLD/prog/run/'status.json').exists() else '在线分析完成',
            conclusion,[legacy(prog+run+'/'+f) for f in files])

    warm=prog+'staged_optimization_20261016_174407_183860/online_recovery_20261016_184734_162656/'
    add('P10','二 前置项目原生质量与系统实验','10月16日充分预热writer恢复对照',
        '在每点完整长prompt预热后，测试只融合writer打包的在线效果，并与独立profiling阶段分离。',
        GPU,'Qasper/HotpotQA各8题，两个arms×2轮，共64正式请求；部分原baseline按hash引用先前已完成点。',
        'C8、measure cap128；每个测量点先将全部8条真实prompt预热32输出；fullK4V4，reader默认相同，仅writer pack不同；之后另做16token profiler。',
        'planar_original、planar_fused_writer。',
        '每轮/每请求TPOT与配对变化、TTFT、rps/tok/s、E2E、输出长度、原生日志queue/prefill/decode histogram delta、preemptions、GPU采样；独立profile完整性与停止超时。',
        '64正式测量完成，后续profile停止超时；总pipeline曾502失败，恢复会话成功，分别记录。',
        'Qasper聚合TPOT约79.788→65.978ms（−17.31%），但基线与候选分开时间窗，仍需交错顺序P06检查漂移；不能从单次恢复结果宣称普遍收益或任务无损。',
        [legacy(warm+'plan.json'),legacy(warm+'comparison.json'),legacy(warm+'findings.json'),legacy(warm+'reused_original.json')])

    # Current-workspace offline work split into distinct experimental questions.
    p='results/progressive_int4_20261023/'
    add('O01','三 10月23日至26日离线编码与Controller','10月23日原生INT4拆分恢复与anchor选择',
        '确认High2+Low2无损恢复原码，并比较补零、中点和仅校准条件均值重建的低位损失。',
        CPU+' 本工作区Python3.12.13、NumPy2.3.5。',
        'H01原生6请求72快照独立副本；3校准/3留出；36留出快照×8heads=288GQA组。',
        'float64；G128，tile32；原scale/zero point/FP16 sink-tail保持；完整INT4attention参考；校准条件均值不使用留出请求。',
        'zero/mid/conditional×K2V2/K2V4/K4V2/K4V4。',
        '码字matched/total/agreement；attention relative/max abs、K/V relative、score RMSE；均值/P95/max、逻辑code bits。',
        '恢复与离线stages2—6完成；18144测量行中本项为固定精度子集。',
        '1553989632码字100%恢复；K2V2误差zero0.379298、mid0.101748、conditional0.105011，中点优于补零且未被当前条件均值超越。',
        [p+'experiment_config.json',p+'codec.json',p+'aggregate.csv',p+'measurements.csv'])
    add('O02','三 10月23日至26日离线编码与Controller','10月23日离线K V残差补读候选',
        '固定残差字节预算，判断随机/位置/attention/风险重要性是否有改善空间，并比较K、V与共享mask。',
        CPU,'与O01相同3校准/3留出，288相关GQA组；不是288独立请求。',
        'tile32、GQA共享mask；补读比例0/.25/.5/.75/1；random冻结seed20261023；完整INT4参考；V-only固定K4、K-only固定V4。',
        'random、fixed、global attention mass、mass×risk、Query×Scale、KV shared mask。',
        'attention mean/P95/max、K/V/score重建误差、fraction/residual fraction、logical bits、side、context、cache ordinal。',
        '完成；单seed随机结果，不构成统计优势。',
        '50%V残差下random0.027378、fixed0.043655、attention0.001099、risk0.001095，提示全局attention选择空间；全局权重为离线参考，未实现在线单遍。',
        [p+'experiment_config.json',p+'aggregate.csv',p+'measurements.csv'])
    p='results/progressive_int4_online_v_20261023/'
    add('O03','三 10月23日至26日离线编码与Controller','10月23日因果单遍V控制与扫描顺序',
        '检验不看未来的prefix mass能否保留离线选择收益，并测精确预算、尾部强制补足与扫描顺序的影响。',
        CPU,'同O01六请求；3校准/3留出、288GQA组；随机三seed20261023/24/25。',
        'K4、V中点anchor；tile32；五补读预算；forward/reverse/random扫描；仅forward校准阈值，所有扫描复用。',
        'fixed/random/offline mass/offline mass-risk/online prefix mass/online prefix mass-risk。',
        'mean/P95/max相对L2、exact residual bytes、forced fraction、位置分布、endpoint/causality gates。',
        '完成34560行。','50%预算risk正序0.025404，随机0.029928但P95 risk0.143104高于随机0.139801；逆序约0.005737。顺序与尾部补足敏感，未通过稳健在线收益门槛。',
        [p+'experiment_config.json',p+'aggregate.csv',p+'position_distribution.csv',p+'validation.json',p+'实验结论.txt'])
    ab='experiments/encoding_ablation_20261024/'
    off('O04','10月24日A B C D独立编码主实验',
        '依次检验固定Gaussian先验、真实KV分布适配、attention输出敏感目标各自的价值。',
        '本轮60校准+30验证+90测试；15精度组合；同一4位叶码截断；C/D8192-bin统计、Lloyd最多100次/rtol1e-6；B262144正态分位点求积。',
        'A/U_MinMax、B/Gaussian、C/Data、D/Jacobian；原payload与_eq分别。','正式完成，验证仅诊断。',
        '联合K4V4 relative L2 A.310456、B.186849、C.150318、D.136101；局部输出目标有效，但不代表任务得分。',
        [ab+'PROTOCOL.md',ab+'provenance.json',ab+'results/full/request_metrics.csv',ab+'results/full/stratified_attention.csv'])
    off('O05','10月24日严格同总字节与容量账本',
        '检验编码表与归一化元数据成本是否会抵消相同payload下的误差收益，避免只比较标称4位。',
        'B完整payload+实际metadata为总预算；固定省略其他方法低位支付额外状态；序列化打包校验；2K/4K/8K容量。',
        'A/B/C/D original vs _eq；E扩展同规则但可继续省Bit1。','账本与等预算比较完成。',
        '4位C.150318→C_eq.151905，D.136101→D_eq.140447；E.017738→E_eq.032587。等字节组不是每标量均保存完整4位。',
        [ab+'results/full/storage_ledger.json',ab+'results/full/equal_budget.csv',ab+'results/unified/summary_A_to_E.csv'])
    off('O06','10月24日K-only V-only与不对称精度',
        '确定Jacobian目标的收益来自K还是V，并检验同逻辑读取预算应给哪一侧更多精度。',
        '原_eq条件；另一侧保持FP16；3单K+3单V+9联合；联合误差独立求解不能单侧相加。',
        'B_eq/C_eq/D_eq，K2V4 vsK4V2、K3V4 vsK4V3。','完成90测试请求。',
        'D对C的K-only改善28.04/30.71/21.24%；V-only3.18/−4.16/−13.55%。原D收益主要在K，V高位有退化；推动后续混合目标实测。',
        [ab+'K与V对比分析.md',ab+'results/full/request_metrics.csv'])
    off('O07','10月24日逐层任务与长度异质性',
        '检验全层平均改善是否在所有层、任务与输入长度成立，找出被均值掩盖的退化位置。',
        '90请求、全部36层，主分析_eq；按任务/长度分层；层号0-based；逐层点估计未做多重校正。',
        'C_eq对B_eq，D_eq对C_eq，三位宽。','分层分析完成。',
        'D相对C在2/3/4位分别34/30/18层更好；4位L6改善44.15%、L18退化54.42%；不宜按平均值给每层结论。',
        [ab+'逐层差异分析.md',ab+'results/layer_comparison.json',ab+'results/full/stratified_attention.csv'])
    off('O08','10月24日原生KVarN E中点与补零',
        '比较原生量化流水线的前缀恢复与独立码本，隔离同一码字缺失位恢复规则的作用。',
        'E不训练；30验证+90测试重采；原生FP16无sign Hadamard、Sinkhorn8、K4V4/G128；保留未满块FP16尾部，无first128 sink；先逆变换到共同参考空间。',
        'E中点、E_zero补零及A—D参照；Eq另列。','完成。',
        '中点2/3/4位.134653/.048567/.017738；补零.703191/.226891/.017738。更好不只来自均匀码点，还包含均衡、局部统计与FP16尾部。',
        [ab+'E_extension/PROTOCOL_E.md',ab+'E_extension/E对比分析.md',ab+'results/unified/summary_A_to_E.csv'])
    supp=ab+'supplementary_experiments/'
    off('O09','10月25日优化截断均匀与非均匀直接对照',
        '检验原MinMax基线是否因离群范围而失真，避免把范围优化收益误当非均匀结构优势。',
        '60校准+90测试，跳过验证；36尾分位数组合，按D2/D3/D4等权校准损失选FP16范围；无FP16 KV保护。',
        'U_MinMax、U_Clipped、Data_Tree；相同Score/Attn/Mix目标的均匀交叉组。','完成。',
        'U_MinMax2位2.291858→U_Clipped.568008，接近Data.566012；但Data3/4位相对U_Clipped改善20.47%/51.08%。范围和结构必须分别解释。',
        [supp+'config.json',supp+'metrics/attention_errors.csv',supp+'metrics/comparisons.json'])
    off('O10','10月25日Score Jacobian与V混合目标交叉',
        '检验只优化score是否足够，以及K用Jacobian时V应使用attention/MSE/混合的哪种训练目标。',
        '60校准90测试、无验证；相同树/均匀家族粒度；K score包含q²scale²/d，K Jac包含p²及V−O；V Mix=.5普通+.5归一attention权重。',
        'Data、Jacobian、JacK_MSEV、JacK_MixV、ScoreK_MSEV/AttnV/MixV、三个Uniform交叉。','全部有效候选完成。',
        'JacK_MixV2/3/4位.464366/.239342/.124537，较Data改善17.96/21.37/17.15%；Score-only改善不足1%。目标必须按输出后果评估。',
        [supp+'protocol.md',supp+'metrics/attention_errors.csv',supp+'metrics/JacK_MixV_comparison.json'])
    off('O11','10月25日独立精度与嵌套代价',
        '估计同一码字前缀约束及共享范围带来的代价，与每个精度单独优化作家族内参照。',
        '独立2/3/4位分别拟合，三份编码9位；主渐进组只有一份4位；完整前缀在测试不重编码。',
        'Independent_Uniform vsU_Clipped；Independent_MSE vsData_Tree。','完成。',
        '4位独立均匀relative L2.190074低于U_Clipped.307286，独立MSE.138166低于Data.150318；只是经验嵌套代价，不是数学最优界或等存储赢家。',
        [supp+'metrics/nesting_penalty.csv',supp+'metrics/attention_errors.csv',supp+'metrics/storage_ledger.csv'])
    off('O12','10月25日Leaf4与Joint234多精度目标',
        '检验从粗到细冻结父分区是否限制细精度，比较先拟合叶再合并与联合三精度优化。',
        'Leaf4先16簇有序叶，再分组父表；Joint234三深度权重1/3、最多60轮、rtol1e-6；完整4位存储。',
        'Data_Tree、Leaf4_Only、Joint_234。','完成。',
        '4位relative L2为.150318/.137945/.147185；Joint对Data各位宽约.47/1.64/2.08%改善，未达5%门槛。优化重建代理不等于优化真实输出。',
        [supp+'config.json',supp+'metrics/attention_errors.csv',supp+'metrics/comparisons.json'])
    off('O13','10月25日收益递减正则条件试验',
        '只在校准曲线确有足够违反时，尝试用正则改善可贪心分配的边际形状。',
        '校准head归一违反>=.01才触发；lambda10，Adam100步，lr.01；固定分区调重建值，不根据测试反例开启。',
        'Joint_Regularized vsJoint_234，候选有条件执行。','校准触发head=0，按预注册跳过正式候选。',
        '没有独立正式结果，不能把跳过记录写成正则有效或无效。',
        [supp+'metrics/regularizer_checks.json',supp+'config.json'])
    off('O14','10月25日收益曲线与精确DP复核',
        '检验收益递减/单调假设在raw与功能代理下是否满足，以及贪心在小同预算问题与DP的差距。',
        '全测试层/head/step/seed固定抽4K channel和4V token；违反tol=1e-10+1e-5 maxD；DP8-item、float64、exact预算及1e-12+1e-10相对容差。',
        '17候选raw/functional曲线；greedy vs exact DP；初轮float32记录另存不混判。','完成275400个DP预算实例。',
        '抽样DP gap均0，但仍有局部收益递减反例：Data V约.531%，MixV V/K约.278/.034%。有限子问题不能证明全缓存最优。',
        [supp+'metrics/diminishing_violations.csv',supp+'metrics/greedy_vs_dp.csv',ab+'results/full/allocation_audit.json'])
    dyn=csvrows(supp+'metrics/dynamic_reading.csv')
    d=collections.defaultdict(list)
    for x in dyn:
        if float(x['target_mean_bits'])==3:d[x['method']].append(float(x['output_rel']))
    off('O15','10月25日五种编码两阶段动态读取原型',
        '在相同平均位预算下比较编码对当前Query驱动分配的支持程度，验证比固定读取更细的选择空间。',
        'L0/17/35，decode1/64，首seed20261024；K按Q×校准曲线，V按重建K的attention×曲线；预算2/2.5/3/3.5/4，两侧分别等预算。',
        'U_Clipped、Data、JacK_MSEV、ScoreK_MixV、Joint234；同编码固定读取作参照。','离线筛查完成，不是全部36层闭环Controller验收。',
        '3位下本覆盖JacK_MSEV输出误差较低；该结果不能归给未在此五方法里运行的JacK_MixV，也不能推导真实HBM或任务质量。',
        [supp+'metrics/dynamic_reading.csv',supp+'config.json'],
        table(['平均读取3位编码','该覆盖output relative L2均值'],[[k,f'{statistics.mean(v):.8f}'] for k,v in d.items()]))
    off('O16','10月25日原生KVarN去FP16尾部消融',
        '定量判断旧E低误差有多少来自不满块FP16保护，避免把保护收益归因于码本。',
        '同90题/轨迹；完整块原生不变，短尾valid-only参考均衡+RTN；单token跳过无定义方差；无FP16 KV旁路，必要metadata保留。',
        'E vs KVARN_nt；E_zero vs KVARN_nt_zero，三位宽。','完成。',
        '中点4位误差.017738→.041072；2/3位.134653/.048567→.223515/.093901。尾部有贡献；仍有本地均衡/粒度差异，非码点单变量结论。',
        [supp+'metrics/native_nt/results_summary.md',supp+'metrics/native_nt/request_metrics.csv',supp+'config.json'])
    ext='experiments/encoding_extension_20261026/'
    off('O17','10月26日TCN与残差编码结构',
        '在统一完整4位、同预处理下比较等宽嵌套、2+1+1残差与对数残差是否改善现有编码。',
        '60校准90测试，验证0；TCN floor分箱、36范围；Residual211共享E|e|幅度；Log alpha从1/4/16/64选；实际FP16参数。',
        'TCN_MinMax/Clipped、Residual211、LogResidual211、U_Clipped/Data/Jacobian。','7方法×15精度组合完成。',
        '四个新方法K4V4 relative L2分别.288118/.273968/.204305/.315570，未优于Data.150318或Jacobian.136101；方法只借鉴结构，不复现论文全系统。',
        [ext+'config.json',ext+'results/full/summary.csv',ext+'results/full/comparisons.json'])
    off('O18','10月26日K V输出误差精确分解',
        '检验对角误差代理为何不能准确预测真实attention output，并观察K/V误差相互作用。',
        '使用每方案重建K得到phat，再算V精确项/对角/跨token项；K项=(phat−P)V；交互独立保存；不得简单代理相加。',
        'O17七种编码，K/V单侧与联合15格。','分解与absolute MSE统计完成。',
        '4位Data absolute MSE.00612783优于Jacobian.00684394，但relative L2顺序相反；指标目标需明确。V相关残差交叉项不能忽略。',
        [ext+'results/full/summary.csv',ext+'results/full/request_metrics.csv'],
        metrics=OFFMET+' output_mse、v_diag_mse、v_exact_mse、v_cross_mse、k_output_mse、kv_interaction_mse。')
    ctrl='experiments/controller_jacobian_20261026/'
    off('O19','10月26日Jacobian编码四策略因果回放',
        '固定编码和总位预算，分离校准失真、当前Query和经验曲线对分配收益的贡献。',
        '90测试请求；仅L0/17/35、decode1/64，首seed；K单位head/channel跨token，V单位head/token跨128通道；2/2.5/3/3.5/4位；policy不接收真实FP16输出或未来Q。',
        'uniform/static/query_only/query_curve；固定原Jacobian，非JacK_MixV。','90题×4策略×5预算完成。',
        '3位query_curve真实MSE.0410364对uniform.0714867降42.60%，对query_only.0422947仅降2.98%；3.5位反而略差。经验曲线非无条件增益。',
        [ctrl+'PROTOCOL.md',ctrl+'results/replay/summary.csv',ctrl+'results/replay/comparisons.json'],
        metrics=OFFMET+' K/V policy CPUms、actual_mean_bits、padded bytes、K predicted/actual diag、V predicted/exact/diag/cross。')
    off('O20','10月26日代理排序幅值与局部Oracle',
        '检验Controller代理是否能排序，以及其绝对值能否作为真实输出误差阈值。',
        '与O19同输入覆盖；同预算跨策略V成对排序；首请求L0 pos1 head0四K channel/四V token局部枚举19方案，其余固定3位。',
        '四策略代理 vs真实误差；proxy optimum vs local true optimum。','完成排序/相关/误差比与局部枚举。',
        '3位V排序正确率87.84%；query_curve真实/预测中位数1.549、P95 8.822；V跨token项占均值精确项81.57%；局部K regret1.4921e-5。代理可辅助排序但不能认证绝对阈值或全局最优。',
        [ctrl+'results/replay/proxy_pairwise_ranking.json',ctrl+'results/replay/proxy_rank_correlations.csv',ctrl+'results/replay/local_oracle.json'],
        metrics='成对排序正确率、秩相关、真实/预测中位/P95、V对角/交叉/精确MSE、局部枚举方案/代理与真实最优/regret。')
    # Non-offline work in the current project, also with explicit per-card purpose/config.
    add('N01','四 本工作区在线任务与算子实验','10月19至22日四方法算子profiling',
        '在固定输入和输出工作量下定位GPU开销，而不是以KV位宽推断加速。',MODERN+' 本项为vLLM服务+.venv-eval客户端；eager+profiler诊断，非Transformers参考生成。',
        'LongBench六任务各50、HumanEval164、MATH子集208；总672题/方法，各89批。',
        '并发8，eager+profiler，window32768，block128，prefix cache off；greedy seed42；固定1024输出、ignoreEOS；FP8为FlashInfer后端。',
        'FP16、FP8 E4M3、TurboQuant K4V4、KVarN K4V2/G128。',
        'GPU event累计/busy union、各phase/category/kernel耗时与次数、host关联比例、CPU inclusive与CUDA runtime、profiler请求latency/first_chunk_latency/time_per_output_token、输入输出token。',
        '四方法首轮672题对齐完成；额外repeat与未完成attempt排除。',
        '累计GPU秒FP16 2379.36、KVarN2349.90、TQ2989.43、FP81887.66；诊断条件下FP8较低但有后端差异，不提供质量或生产吞吐排名。',
        ['configs/operator_profile.json','results/experiment_analysis_20261022/validated_results.json','results/experiment_analysis_20261022/raw_kernel_totals.csv'])
    add('N02','四 本工作区在线任务与算子实验','10月26日四策略64步闭环pilot',
        '检验离线局部误差改善能否在跨层、跨token传播后提高轨迹稳定性。',MODERN,
        '六任务各取测试第一/最后，共12请求，每策略64步。',
        '原Jacobian首seed；FP16 prefill、decode全部36层；greedy；参考适配器保留FP16源cache并重编码；每策略独立轨迹。',
        'FP16/full4/uniform3/query_curve3。',
        'token一致率、first divergence、64步全一致数、文本/IDs、采样局部MSE、参考decode墙钟/TPOT/peak allocated。',
        '48条生成完成，FP16逐token复现门槛通过。',
        '动态3位一致率45.05%高于uniform38.15%，但CI[−13.41,+29.04]pp跨0，3改善3退化6持平；不是官方任务准确率或实际系统性能。',
        [ctrl+'PROTOCOL.md',ctrl+'results/online/complete.json',ctrl+'results/online/结果分析.md'])
    fq='experiments/full_quality_20261027/'
    add('N03','四 本工作区在线任务与算子实验','10月27至29日原Jacobian官方任务质量',
        '检验相同三位平均预算的动态读取是否提高最终代码/数学质量，而不只改善局部误差。',MODERN+' 官方评分在.venv-eval。',
        'HumanEval164×5全部完成；MATH500仅FP16/full4完整，uniform另生成170未评分；原LongBench21全4750计划未启动。',
        'batch1；non-thinking，greedy seed42；自然EOS；HumanEval cap2048、数学8192，window40960；原Jacobian种子20261024、FP16 prefill、全36层decode量化。',
        'FP16/full4/uniform3/query_only3/query_curve3。',
        '官方pass@1/数学等价分；完整性/EOS/触顶/截断/长度；逐题gain-loss、bootstrap CI、语法/运行/超时分类、McNemar探索p；参考资源。',
        '1990生成、1820评分；暂停。','HumanEval78.05/76.83/52.44/71.34/68.29%；动态优于uniform但未证明等价full4；数学81.60→69.00%，不能把代码近似保持推广到数学。',
        [fq+'PROTOCOL.md',fq+'config.json',fq+'results/task_scores.csv',fq+'results/humaneval_analysis/paired_comparisons.json'])
    add('N04','四 本工作区在线任务与算子实验','10月27至29日单层读取量接续与硬件探针',
        '在正式LongBench任务上把逻辑账本与匹配单层attention的warm/evicted物理采样对应起来。',MODERN+' Nsight Compute2024.3.2；普通用户counter权限不足。',
        '计划LongBench4750×5；旧HumanEval/MATH只读链接；正式LongBench0题。',
        '每请求L17首次decode调用fixture，输出校验后warm预热5次与128MiB evict；ncu app replay；采集失败保留null，质量继续；已有GPU原生worker不打断。',
        '五策略；warm/evicted；逻辑K/V与物理DRAM分别。',
        'logical bits/padded bytes/metadata allowance、physical status/coverage、DRAM read/write、no_decode/partial/unavailable、postprocess/request seconds、探针错误与原始报告。',
        '接续/账本检查与真实探针已做，正式采集未开始；ERR_NVGPUCTRPERM。',
        '没有正式物理流量结果，不能把unavailable写成零；仅是单层调用计划，不等于完整生成计数。',
        ['experiments/full_quality_io_20261027/PROTOCOL.md','experiments/full_quality_io_20261027/hardware_capability.json','experiments/full_quality_io_20261027/io_checks.json'])
    win='experiments/winner_quality_20261029/'
    add('N05','四 本工作区在线任务与算子实验','10月29日JacK MixV阈值与代码质量',
        '验证离线最佳局部编码的代码任务表现，并测固定风险容忍量的位宽—质量折中。',MODERN+' .venv-eval隔离官方HumanEval评分。',
        'HumanEval同164题，四策略656条全部评分。',
        'JacK_MixV，首seed；non-thinking greedy，cap2048自然EOS；阈值.05/.15/.30，K后V顺序；full4-relative代理额外风险，无固定总位宽。',
        'full4/adaptive005/adaptive015/adaptive030。',
        'pass@1、correct数/长度/EOS、K/V depth elements/units、logical bits/padded bytes、元素加权平均bit、对各自轨迹full4 payload比例。',
        '全部完成。','pass@1为71.95/71.34/65.24/52.44%，均位4/2.9504/2.7147/2.5092；阈值放宽少读但质量下降。full4也低于N03原Jacobian76.83%，局部最优不代表任务最优。',
        [win+'PROTOCOL.md',win+'results/summary.json'])
    add('N06','四 本工作区在线任务与算子实验','11月2至5日无任务输出cap greedy基线',
        '先筛查完整4位编码基础质量，再计划0—4位曲线Controller与metadata消融，区别算法和工程阶段。',MODERN,
        '同464面板：HumanEval164、MATH100、LongBench21任务200；排除校准60。',
        'non-thinking greedy，无输入截断、无分任务输出cap，统一40960总窗口；prefill512、全36层decode。0—4位后续阶段仅有协议。',
        'FP16与五种fixed4编码；实际仅FP16和Jacobian产生部分结果。',
        '官方分、完整性/EOS/上下文limit、逻辑读取/mean bits/depth counts、metadata账本和参考资源。',
        '573条生成评分，随后为sampling+thinking暂停。',
        'FP16 HumanEval78.05%、数学77%；Jacobian代码76.83%、数学99题暂61.62%；LongBench多题耗尽边界，尚无完整编码比较；没有0—4位Controller最终结果。',
        ['experiments/staged_algorithm_20261102/PROTOCOL.md','experiments/staged_algorithm_20261102/analysis/task_metrics.csv'])
    add('N07','四 本工作区在线任务与算子实验','11月5日sampling计数与后端过渡',
        '实现完整thinking生成与同次现场硬件计数，并修复长range profiler阻塞和推理后端不一致。',MODERN+' 首版Nsight app-range，第二版direct CUPTI窗口。',
        '沿用464 ID；首版1正式结果，窗口版64生成63评分。',
        'thinking .6/.95/20/min_p0，主seed20261105；统一40960；Transformers参考适配器；输入不截断，无分任务输出cap。',
        '计划FP16+五编码，但只有FP16部分，不形成正式对照。',
        'generated/thinking/final token、finish、质量、execution ID、prefill/decode DRAM读写、counter故障/范围/库hash。',
        '两版均暂停，diagnostics/smoke另存。',
        '首版故障在生成EOS后的profiler range收尾；窗口版为vLLM更正暂停。两版样本不与下一原生vLLM合并最终排名。',
        ['experiments/staged_sampling_20261105/PROTOCOL.md','experiments/staged_sampling_20261105/diagnostics_profiler_stall_20261105.json','experiments/staged_sampling_windowed_20261105/stage_status.json'])
    add('N08','四 本工作区在线任务与算子实验','11月5至9日原生vLLM编码质量与完整DRAM',
        '在原生scheduler/sampler/paged KV闭环下检验完整4位质量，并给出同次真实生成完整prefill与decode的物理流量。',
        GPU+' 项目.venv；native vLLM Qwen3/GPUModelRunner，direct CUPTI单pass；root计数、普通用户隔离评分。',
        'HumanEval164、MATH100、LongBench21共200；同464题，各请求固定派生seed。',
        'max_seqs1、page128、prefill512、APC off、async off、40960可用容量+null block；thinking .6/.95/20/min_p0，主seed20261105；decode FULL graph；树uint8码、FP16页用于prefill，无实际位平面少读。',
        'FP16、Jacobian、JacK_MixV、Data、Gaussian、U_Clipped fixed4。',
        '23任务官方分与配对CI；EOS/context/thinking完整、输入输出长度；K/V逻辑bits/depth/mean bits；reference GPU/host峰值及耗时；prefill/decode/read/write/total DRAM整数字节、窗口units/count/time、execution ID/CSV/counterdata/library hash。',
        '统一保留11月9日11:06快照：四方案各464完整；Gaussian252生成251评分；U未开始；继承419FP16结果不重复计。',
        'HumanEval FP16/Jac/Mix/Data为96.95/89.02/92.68/84.76%；数学94/86/88/90%。树参考实现全程物理流量32.95—47.06PB高于FP16 12.30PB，不能称硬件少读。不同输出量和工作路径仍需区分。',
        ['experiments/staged_vllm_recovery_20261106/run_manifest.json','results/experiment_record_20261109/current_task_metrics.csv','results/experiment_record_20261109/current_run_totals.csv','results/experiment_record_20261109/current_physical_totals.json'])
    add('N09','四 本工作区在线任务与算子实验','11月7日起原生K4V4对照准备',
        '把当前树参考编码与原生TurboQuant/KVarN缓存路径比较完整质量，避免只在自定义backend间选择。',GPU,
        '同464题/方法；CPU/scorer checks已完成；正式生成0题。',
        '同N08采样与FP16 prefill；TQ K4V4 norm correction；KVarN K4V4 G128、sink128、Sinkhorn8、原生尾部保护；不测物理性能。',
        'turboquant_4bit_nc、kvarn_k4v4_g128；FP16/Jacobian冻结参照。',
        '计划官方分、长度/完成率/思考闭合、native dispatch counts、逻辑/保护元数据与参考耗时；现有CPU tests/scorer equivalence/source hashes。',
        '截至11:06等待GPU锁，尚无正式结果。',
        '不能给出两原生方法的实测质量结论；sink/tail与全KV树量化不同，未来属于实现包对照。',
        ['experiments/native_k4v4_quality_20261107/experiment.json','experiments/native_k4v4_quality_20261107/validation/cpu_checks.json','experiments/native_k4v4_quality_20261107/status.json'])


def write():
    for r in records:
        for field in ['purpose','model','environment','data','configuration','compared_methods',
                      'recorded_metrics','completion','conclusion','evidence']:
            assert r[field],(r['id'],field)
        for source in r['evidence']:
            assert (ROOT/source).exists(),(r['id'],source)
    out=['# 总实验记录\n\n',
         '本记录按**每一个实验问题**建立档案，每项均直接写明实验目的、模型、软硬件环境、数据、运行配置、对比方法、全部指标入口、完成情况、实测结果和结论。范围包括前置项目`kvbench-3090-cu126`的历史离线/在线实验，以及本工作区的编码、Controller、任务质量与计数实验。\n\n',
         '已落盘数据的统一截点仍为**2026年11月9日11:06（北京时间）**；本次补录的是更早的既有实验，没有重新运行模型。历史实验按原运行日期记录，归档时间与运行时间分开。共有'+str(len(records))+'份问题级档案；多个档案可共享一批测量，不按档案数重复累计样本。\n\n',
         '数值解释与完整当前23任务表另见[综合分析](实验综合分析_20261109.md)和[数据附录](results/experiment_record_20261109/数据附录.md)。前置项目新增[历史来源索引](results/experiment_record_20261109_legacy/历史来源索引.md)、[41个批次原始运行条件](results/experiment_record_20261109_legacy/runs.json)、[全部历史指标字段](results/experiment_record_20261109_legacy/metric_fields.csv)。\n\n',
         '## 实验目录\n\n',table(['编号','实验','类型','完成情况'],[[r['id'],r['name'],r['group'].split(' ',1)[1],r['completion']] for r in records])]
    group=None
    for r in records:
        if r['group']!=group:
            group=r['group'];out.append('## '+group+'\n\n')
        out.append('### '+r['id']+' '+r['name']+'\n\n')
        labels=[('实验目的','purpose'),('模型与结构','model'),('环境条件','environment'),
                ('数据和样本','data'),('实验配置','configuration'),('对比方法','compared_methods'),
                ('记录的指标','recorded_metrics'),('实际完成情况','completion')]
        out.append(table(['档案字段','具体记录'],[[label,r[key]] for label,key in labels]))
        if r['results']:
            out.append('**实测结果**\n\n'+r['results'])
        out.append('**结果与结论分析**：'+r['conclusion']+'\n\n')
        out.append('**配置、原始指标和复核来源**：'+'；'.join(link(p) for p in r['evidence'])+'。\n\n')
    out+=['## 五 全部批次与失败尝试\n\n',
          '前置项目41个运行目录的完整plan/suite/settings/status、全部成功/失败重试及其来源见[历史来源索引](results/experiment_record_20261109_legacy/历史来源索引.md)。H05、H08、H09、P04、P08等保留失败状态，失败启动不进入成功方法结果。`pipeline01/pipeline02_unbounded`的32题pilot已分别列入P00a/P00b，不将pilot完成写成全量任务完成。\n\n',
          '旧不对称全量批次存在“status1235题，而comparison437题”的报告滞后，分别记录；旧Progressive相对变化存在跨任务基线错误，新增[同任务重算CSV](results/experiment_record_20261109_legacy/progressive_online_same_dataset.csv)，保留原列并另写recomputed列，不覆盖旧结果。\n\n',
          '## 六 指标值与复现资料\n\n',
          '每项档案的来源链接包含该实验的全部汇总列。逐请求、逐层或逐kernel大表不在正文省略后冒充全量，按下列索引查到全部保留值；未记录的指标保持缺失，未执行的计划不得计为实测。\n\n',
          table(['资料','内容'],[
            [link('results/experiment_record_20261109_legacy/metric_artifacts.csv'),'前置项目全部已索引配置、汇总、CSV和原路径/大小/hash；超过10MB文件保留原位置引用'],
            [link('results/experiment_record_20261109_legacy/metric_fields.csv'),'历史指标CSV精确字段名'],
            [link('results/experiment_record_20261109/metric_fields.csv'),'本工作区汇总与逐题JSON的字段、嵌套路径和类型'],
            [link('results/experiment_record_20261109/raw_metric_artifacts.csv'),'1802原始指标文件索引，包括逐层CSV压缩分片及Parquet/JSONL位置'],
            [link('results/experiment_record_20261109/raw_metric_fields.csv'),'逐层原始CSV表头，不只保留主要输出指标'],
            [link('results/experiment_record_20261109/generation_metrics.csv'),'5811跨版本生成记录主要数值；含继承/复用，非独立样本总数'],
            [link('results/experiment_record_20261109/generation_metrics.jsonl.gz'),'逐题完整嵌套测量指标与原输出hash；文本/token arrays按原路径保留'],
            [link('results/experiment_record_20261109/current_task_metrics.csv'),'当前各方案23任务得分、EOS、上下文触顶、thinking与长度'],
            [link('results/experiment_record_20261109/current_physical_totals.json'),'当前逐题重新求和的prefill/decode硬件DRAM整数字节'],
            [link('实验综合分析_20261109.md'),'全部19编码方法主要误差表、K/V分层、Controller、旧/新质量与流量综合解释']]),
          '各档案的结构化副本见[实验档案JSON](results/experiment_record_20261109_legacy/实验档案.json)，字段完整性、来源hash和同任务差值复核见[核查记录](results/experiment_record_20261109_legacy/validation.json)。CPU-only整理脚本见[build_experiment_dossiers.py](scripts/build_experiment_dossiers.py)；旧项目归档脚本见[archive_legacy_experiment_records.py](scripts/archive_legacy_experiment_records.py)。\n']
    (ROOT/'总实验记录.md').write_text(''.join(out))
    (LEG/'实验档案.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(dossiers=len(records),groups=dict(collections.Counter(r['group'] for r in records)),
                         characters=sum(map(len,out))),ensure_ascii=False,indent=2))


if __name__=='__main__':
    build();write()
