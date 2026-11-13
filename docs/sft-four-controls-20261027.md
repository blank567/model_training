# 本轮SFT：2/3轮、有效Batch8/16

2026-10-27用户选定先运行四组。四组已提交单GPU后台串行队列；第一组启动后，后续仅在前一组完整完成并核对步数后自动开始。

固定学习率`5e-5`、同一300M预训练Base、同一19,991条SFT训练集、seed42、warmup比例0.03、weight decay0.01。每组独立初始化，保留各自best和完整last。test不参与超参选择。

| 顺序 | 变量 | 实际轮数 | Micro × 累积 | 有效Batch | 最终更新数 |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | 训练量2轮 | 2 | 8 × 4 | 32 | 1,250 |
| 2 | 训练量3轮 | 3 | 8 × 4 | 32 | 1,875 |
| 3 | Batch8 | 1 | 8 × 1 | 8 | 2,499 |
| 4 | Batch16 | 1 | 8 × 2 | 16 | 1,250 |

训练量2/3轮共用三轮cosine调度：1,875步、57步warmup，通过`stop_epochs=2/3`控制实际训练量，因此配置名为`ep3_stop2`、`ep3_stop3`。比较同一学习率轨迹上的训练量，避免同时改变训练量与调度。旧两轮调度的2轮候选不纳入本轮，不能充当这次2轮结果。

2轮处理39,982条样本，3轮处理59,973条。比较最终验证Loss、最佳验证Loss与原语料保持曲线；仅看best不能判断增加训练量是否造成过拟合。既有一轮调度结果也不能直接充当三轮调度下的1轮结果。

Batch8/16固定一轮、19,991条样本和相同学习率。Micro固定8，累积1/2；验证/保存间隔400/200步，日志40/20步，即每约3,200样本验证、320样本记日志。既有相同数值配置的Batch32一轮结果可作为Batch轴的控制组。曲线按`examples_seen`或监督Token数比较。

这两条轴用于初筛，不是2×2组合实验，也不保证分阶段选择达到全局最优。5e-5沿用折中工作基线：1e-4虽然指令验证Loss更低，但原语料小窗口退化更明显；这些Loss不是固定任务准确率。

## 计划与配置

远程计划：`/root/autodl-tmp/model_training/runs/plans/sft_epochs2_3_batch8_16_20261027_194551/plan.json`。

远程队列目录：`/root/autodl-tmp/model_training/runs/assistant/system/sft_epochs2_3_batch8_16_queue_20261027_194551/`，内有queue.json、state.json、controller.log和CPU控制器测试日志。正式模型输出在runs根目录，各目录带参数和时间戳。

[四组冻结配置镜像与命令索引](E:/codex/model_training/runs/assistant/system/sft_four_controls_controller_20261027/README.md)保存了JSON、identity和env内容。训练使用远程原路径，不修改这些冻结文件。

新选择指针为`runs/plans/latest_sft_four_controls.json`。旧`runs/plans/sft_epochs_batch_controls_20261027_153648/`仍保存原0.5/1/2轮、共享两轮调度的候选及交付记录；旧训练量候选本轮均不启动。Batch8/16冻结文件直接复用，内容未改。

## 查看与停止

第一组实时Loss，在远程Bash执行：

```bash
source /root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr5e-5_mb8_ga4_b32_ep3_stop2_wu0p03_wd0p01_seed42_20261027_194551.env
tail -n 30 -F "$RUN_LOG"
```

`Ctrl+C`只退出tail，训练继续。该命令固定看第一组；下一组开始后source对应env。各组有独立W&B在线记录。

查看队列状态：

```bash
cd /root/autodl-tmp/model_training
.venv/bin/python -m json.tool runs/assistant/system/sft_epochs2_3_batch8_16_queue_20261027_194551/state.json
```

队列state中的step在每组结束时更新，实时step读取该组日志、metrics.jsonl或W&B。启动快照中控制器PID为2391、第一组训练PID为2403，实时情况以进程和日志为准。

停止整个队列时，先用`ps -p 2391 -o pid,args`确认仍是该控制器，再执行`kill -TERM 2391`。控制器通知当前训练保存后停止，并停止后续；不要只结束训练子进程后就假定队列已停止。中断恢复使用原配置与完整last，不改epochs或调度。再次提交同一队列会跳过已完整完成的组，有本组last时使用resume；恢复前必须确认原控制器已退出，不能重复提交。

## 容量与验证

启动实查数据盘110GiB、空闲48.35GiB；四组稳定权重预计18.08GiB。按正式预检与保存余量估计，串行运行四组需初始空闲约30.60GiB，容量足够。逐组仍检查磁盘、预算、GPU锁与冻结身份，不删除模型，队列结束不自动关闭计费实例。

控制器本地与远程各20项CPU/mock测试通过，覆盖不同目标步数、共同调度、串行子进程、恢复、空间不足、超时和停止信号。冻结训练源码保持`0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`。真实GPU进度快照另存于队列目录`status_snapshot.json`，不能当作四组最终完成或效果评测。
