# 多服务器同步与分片

远程仓库：`https://github.com/LinWang1225/test-V1`。当前论文方法入口为
`experiments/paper_k4v4_quality_20261109`。TurboQuant保留论文QJL，KVarN使用作者官方K4V4实现。

## 获取代码

旧备份历史已经包含环境和大量结果。建议使用部分克隆与代码稀疏检出，
保留实验源码、冻结样本、码本和评分器，避免将原服务器的环境和生成缓存下载到新服务器。

```bash
git clone --filter=blob:none --no-checkout https://github.com/LinWang1225/test-V1.git kvarn-profile
cd kvarn-profile
git config core.sparseCheckout true
git config core.sparseCheckoutCone false
git show origin/main:configs/sync/code-only.sparse-checkout > .git/info/sparse-checkout
git checkout main
```

已有克隆可正常`git pull --ff-only`。每个服务器重新建立Python环境并安装本仓库的
`src/KVarN`，不要直接复用Git历史中的`.venv`；不要用其他版本的上游vLLM替代此fork。
现有服务环境版本见`locks/operator-serving.freeze.txt`，CUDA 12.6核心约束见
`locks/serve-cu126.txt`。CUDA、驱动及编译选项仍须符合目标服务器GPU。
原机器专用的`install_environment.sh`含本地代理和路径，新服务器按其依赖步骤配置本地路径。

模型权重在仓库之外，须另行准备同一个Qwen3-4B revision：
`1cfa9a7208912126459214e8b04321603b3df60c`。
运行入口接受本机模型目录，不要求复刻`/home/wanglin`。

## 单方法、单服务器

```bash
export KV_MODEL_DIR=/本机模型目录/Qwen3-4B
.venv/bin/python scripts/run_quality_shard.py \
  --method turboquant_paper_k4v4 --model "$KV_MODEL_DIR" \
  --output portable_runs/turboquant-all --dry-run

.venv/bin/python scripts/run_quality_shard.py \
  --method turboquant_paper_k4v4 --model "$KV_MODEL_DIR" \
  --output portable_runs/turboquant-all --gpu 0
```

另一台服务器可同时运行`--method kvarn_k4v4_g128`，并使用自己的输出目录。
也支持`fp16`、`Jacobian_Tree`、`JacK_MixV`、`Data_Tree`、`Gaussian_Tree`。
U_Clipped已按用户要求从新入口中排除。

## 同一方法分到两台服务器

服务器A：

```bash
.venv/bin/python scripts/run_quality_shard.py \
  --method turboquant_paper_k4v4 --model "$KV_MODEL_DIR" \
  --num-shards 2 --shard-index 0 --gpu 0 --output portable_runs/tq-shard0
```

服务器B使用相同命令，将`--shard-index`改为`1`，输出改为`portable_runs/tq-shard1`。
按固定样本index取模分片，2片各232题；每题沿用原请求种子。
需要更多机器时，两边统一修改`--num-shards`，每台分配不同index。
同机多卡可用不同`--gpu`；同一GPU有运行中的实验时，入口会拒绝并发占用。

输出目录包含独立清单和逐题结果，重新执行相同命令会跳过已完成且身份一致的题目。
一次中断只会重跑尚未完整落盘的当前题。不要调用旧服务器的root服务恢复入口，
那些入口绑定了原机路径、完整队列与硬件计数权限。

## 评分与合并

评分可放到CPU服务器；先准备评分环境：

```bash
uv venv .venv-eval --python 3.12
uv pip install --python .venv-eval/bin/python -r locks/quality-eval.freeze.txt
.venv-eval/bin/python scripts/run_quality_shard.py \
  --method turboquant_paper_k4v4 --model "$KV_MODEL_DIR" \
  --num-shards 2 --shard-index 0 --output portable_runs/tq-shard0 --score-only
```

评分时沿用生成时的method、shard参数与输出目录；模型目录只需可读取同版配置文件。
输入文件已冻结，HumanEval仍使用原隔离评分器，数学及LongBench沿用相同官方评分实现。

将各服务器的分片结果目录汇集到同一台机器，然后：

```bash
.venv/bin/python scripts/merge_quality_shards.py --output portable_runs/tq-merged \
  portable_runs/tq-shard0 portable_runs/tq-shard1
```

合并会核对方法、样本、模型配置、源码及生成环境版本，拒绝冲突题目和混合协议；
缺题、缺评分会明确列出，只有全部预期题目生成且评分后才标记complete。
不同方法分别合并。若GPU硬件不同，保留逐题GPU UUID与运行环境，比较时单列硬件差异。

## 数据与计量口径

每方法使用相同464题：HumanEval164、数学100、LongBench21子任务共200题。
支持`--dataset humaneval`等任务过滤；不同过滤范围不能直接合并。
所有thinking、采样、停止规则沿用现有协议。

分片入口用于真实闭环生成的任务质量比较，**不启用CUPTI、不需要root，
也不宣称有完整生成DRAM测量**。它生成独立的portable运行指纹，不能冒充原服务断点，
也不自动汇入原服务的物理流量报告。现机冻结源码与运行队列不受影响。
