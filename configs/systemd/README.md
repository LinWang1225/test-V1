# 实验开机恢复

多服务器分工请使用[独立分片入口](../sync/README.md)。本目录的日期命名unit是原服务器部署快照，
含原机绝对路径；新服务器使用分片入口不需要安装这些root服务。2026-11-09的交接服务在Gaussian完成后跳过U_Clipped，
再启动论文TurboQuant与作者官方KVarN；旧native两组的用户服务已经停用。

用户级服务`kvarn-staged-experiment.service`与wanglin用户的linger配合，在开机后无需SSH登录即可启动。脚本、环境和模型路径必须保持可用。守护脚本放在实验源码目录之外，不改变已冻结的算法指纹。

本机安装：

```sh
mkdir -p ~/.config/systemd/user
cp configs/systemd/kvarn-staged-experiment.service ~/.config/systemd/user/
loginctl enable-linger wanglin
systemctl --user daemon-reload
systemctl --user enable --now kvarn-staged-experiment.service
```

核对：

```sh
loginctl show-user wanglin -p Linger
systemctl --user is-enabled kvarn-staged-experiment.service
systemctl --user status kvarn-staged-experiment.service
journalctl --user -u kvarn-staged-experiment.service -n 30 --no-pager
```

启动后若已有手动队列，守护服务只观察其文件锁，不中断或重复启动。当前手动队列不属于服务cgroup；停止守护服务不会停止这份已存在的手动任务。开机时没有队列，则等待GPU驱动就绪，用前台方式启动恢复队列；此后子进程由systemd统一管理。

异常退出等待60秒重启；10分钟内最多3次启动，重复失败后保留日志并停止重试，防止反复运行错误实验。修复后执行：

```sh
systemctl --user reset-failed kvarn-staged-experiment.service
systemctl --user start kvarn-staged-experiment.service
```

2026-11-05起恢复入口默认指向`experiments/staged_sampling_windowed_20261105`，旧greedy实验已暂停并保留。新队列为thinking+固定请求种子采样，包含全部464题（LongBench 200题、21子任务）。

物理计数器由独立root系统服务`kvarn-sampling-dram.service`提供；普通用户服务调度并评分，root服务在本次真实生成现场计数。root服务先运行实际DRAM探针；用户服务等待其心跳后才启动。管理员可使用`sudo .venv/bin/python scripts/sampling_dram_admin.py --install`安装并启动采集器。认证信息不写入代码或配置。

物理测量在每题真实端到端生成现场进行，prefill与完整decode的硬件聚合值和最终回答属于同一次执行。执行标记拒绝工具额外启动生成，不使用参考token或事后重跑。评分使用同次回答，计数开启时的耗时不作为无profiler服务性能。结果见新目录`analysis/physical_totals.json`。

本服务只覆盖目前已实现的第一阶段六方案队列。队列正常完成后退出成功，后续开机识别完成标记，不重复全量评测。后续阶段需先按实验依赖准备和冻结方案。关闭本实验自启：

```sh
systemctl --user disable --now kvarn-staged-experiment.service
```

无需关闭整个用户的linger，以免影响其他用户服务。实验生成日志位于`experiments/staged_sampling_windowed_20261105/logs/stage1_queue.log`；守护状态另存`autostart_status.json`。未重启服务器进行破坏性测试，采用配置验证、实际服务启停及队列PID一致性检查。

2026-11-05修复：原巨大NVTX区间已停用，改为direct CUPTI单pass连续窗口；每512个输入token或32个decode步骤读取一次真实计数并完整相加，无参考轨迹。
