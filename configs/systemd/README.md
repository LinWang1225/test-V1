# 实验开机恢复

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

本服务只覆盖目前已实现的第一阶段六方案队列。队列正常完成后退出成功，后续开机识别完成标记，不重复全量评测。后续阶段需先按实验依赖准备和冻结方案。关闭本实验自启：

```sh
systemctl --user disable --now kvarn-staged-experiment.service
```

无需关闭整个用户的linger，以免影响其他用户服务。实验生成日志仍位于`experiments/staged_algorithm_20261102/logs/stage1_queue.log`；守护状态另存`autostart_status.json`。未重启服务器进行破坏性测试，采用配置验证、实际服务启停及队列PID一致性检查。
