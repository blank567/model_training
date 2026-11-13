# SFT训练量与有效Batch对照计划

> 2026-10-27更新：用户选定先跑实际2/3轮、有效Batch8/16四组，已提交后台串行队列。当前配置、进度与命令见[本轮四组实验](sft-four-controls-20261027.md)。下文保留此前五候选准备阶段的历史快照；旧0.5/1/2轮候选不在本轮队列中，不要重复运行旧启动命令。

`5e-5` 已完成625步，完整best/last已核验。三组学习率结果如下；预训练保持指标来自固定的小规模validation窗口，两列含义不同，均为越低越好。

| 学习率 | SFT回答验证macro Loss | 原语料小窗口Loss |
| --- | ---: | ---: |
| `1e-5` | 2.2394 | 3.3437 |
| `2e-5` | 2.1847 | 3.4156 |
| `5e-5` | 2.1222 | 3.6014 |
| `1e-4` | 2.0968 | 3.8516 |

四组均已完整完成相同625步预算。1e-4的指令验证Loss更低，但原语料小窗口Loss也更高；它比5e-5多降低约0.0254的SFT Loss、同时保持指标多上升约0.2502。这是两个不同目标上的变化，不能直接合并成综合分数。

后续配置先保留5e-5作为折中工作基线，已经恢复为 `ready_not_started`，可以由用户按下面命令直接训练。它不是经固定任务评测确认的综合最优，SFT Loss也不能代替任务准确率。1e-4完整模型保留，若以后选择它作为基线，应另生成新配置，不编辑当前已冻结JSON。

## 第一条轴：0.5、1、2轮训练量

各候选从同一预训练Base开始，固定B32、seed42、数据和模板。三组共同使用 `epochs=2` 的1250步cosine调度、38步warmup，只改变 `stop_epochs`。

| stop_epochs | 调度轮数 | 实际停止步数 | 处理样本数 |
| --- | ---: | ---: | ---: |
| 0.5 | 2 | 313 | 10,016 |
| 1 | 2 | 625 | 19,991 |
| 2 | 2 | 1,250 | 39,982 |

半轮在完整参数更新边界取整。比较同一调度轨迹上的不同训练量，记录最终验证Loss、最佳验证Loss与原能力保持曲线。仅比较best不能判断何时过拟合，因为最佳值会保留此前结果。

这条轴的1轮候选采用两轮学习率调度，需要独立训练。既有625步的一轮调度结果不能复用，也不能编辑旧配置的epochs后直接resume。

已冻结候选环境文件（当前学习率5e-5，尚未启动）：

| stop_epochs | 远程env路径 |
| --- | --- |
| 0.5 | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr5e-5_mb8_ga4_b32_ep2_stop0p5_wu0p03_wd0p01_seed42_20261027_153648.env` |
| 1 | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr5e-5_mb8_ga4_b32_ep2_stop1_wu0p03_wd0p01_seed42_20261027_153649.env` |
| 2 | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr5e-5_mb8_ga4_b32_ep2_stop2_wu0p03_wd0p01_seed42_20261027_153650.env` |

## 第二条轴：有效Batch 8、16、32

该初筛固定1轮、19,991条样本和同一学习率，不做线性LR放大。Micro Batch固定8，梯度累积为1/2/4；这样比较有效Batch的更新效果。显存峰值主要由Micro Batch决定，扩大累积次数不会按比例增加激活显存。

| 有效Batch | Micro × 累积 | 一轮更新数 | 验证/保存间隔 | 日志间隔 |
| --- | --- | ---: | ---: | ---: |
| 8 | 8 × 1 | 2,499 | 400步 | 40步 |
| 16 | 8 × 2 | 1,250 | 200步 | 20步 |
| 32 | 8 × 4 | 625 | 100步 | 10步 |

每约3200个样本验证/保存，每约320个样本记录训练日志，末尾另验证。按样本数对齐选模机会与监测开销；曲线比较优先使用 `examples_seen` 或真实监督Token数作横轴。固定同样的step数会使各Batch看到不同数量的数据，不适合本轮比较。

Batch32与已完成的5e-5一轮控制组完全同配置，复用它的完整记录，不再训练一遍。另两组已冻结、尚未启动：

| 有效Batch | 远程env路径 |
| --- | --- |
| 8 | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr5e-5_mb8_ga1_b8_ep1_wu0p03_wd0p01_seed42_20261027_153648.env` |
| 16 | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr5e-5_mb8_ga2_b16_ep1_wu0p03_wd0p01_seed42_20261027_153649.env` |

当前Batch草案以1轮为基准。若根据训练量实验改变后续基线，需重新生成并核对epochs、stop_epochs与学习率调度，不能直接把旧结果当成新控制组。

## 执行与资源

两个轴分别做单变量比较，不运行全部组合。所有新训练从同一预训练Base初始化，不继承上一组SFT权重；独立test不参与超参选择。当前共需5个新增独立实验：3个训练量候选、2个Batch候选，Batch32复用旧控制组。组合优选后还需要少量复验，不能把分阶段初筛当作全局最优搜索。

扩容已实查为110GiB，四组学习率实验完成后空闲约48.35GiB。五个后续实验稳定权重占用估计22.60GiB，顺序运行所需初始空闲约35.12GiB，当前容量足够；仍需逐组检查磁盘和预算，不自动删除模型。

计划总表位于远程 `runs/plans/sft_epochs_batch_controls_20261027_153648/plan.json`，校验记录为同目录 `verification.json`。可执行清单为同目录 `ready_to_train.json`。五份冻结配置均已回读身份/步数检查，五份启动脚本通过 `bash -n`，正式候选输出目录尚未创建，没有自动启动这些候选。

## 后面直接训练的命令

在AutoDL Bash终端中，每次只执行以下表格中的一条命令；完成一组后再启动下一组。脚本会加载对应env、先check，再用nohup独立后台start并打印PID、Config和Log。

| 实验 | 单条命令 |
| --- | --- |
| 训练量0.5轮 | `bash /root/autodl-tmp/model_training/runs/plans/sft_epochs_batch_controls_20261027_153648/commands/start_epochs_0p5.sh` |
| 训练量1轮（共用两轮调度） | `bash /root/autodl-tmp/model_training/runs/plans/sft_epochs_batch_controls_20261027_153648/commands/start_epochs_1.sh` |
| 训练量2轮 | `bash /root/autodl-tmp/model_training/runs/plans/sft_epochs_batch_controls_20261027_153648/commands/start_epochs_2.sh` |
| 有效Batch8 | `bash /root/autodl-tmp/model_training/runs/plans/sft_epochs_batch_controls_20261027_153648/commands/start_batch_8.sh` |
| 有效Batch16 | `bash /root/autodl-tmp/model_training/runs/plans/sft_epochs_batch_controls_20261027_153648/commands/start_batch_16.sh` |

Batch32直接复用已完成5e-5控制组，不提供重复启动脚本。脚本不负责自动启动其他组；进程提交后通过status和日志核验实际更新。中断后使用原配置resume，不再运行start脚本。

[本地配置副本及逐组命令索引](E:/codex/model_training/runs/assistant/system/sft_controls_handoff_20261027/README.md) 可直接查看五份JSON；这些是远程冻结文件的镜像，训练使用远程原路径。

正式执行时，先确认基线选择，再source对应env、运行check并用nohup后台start；停止和resume方式见 [SFT训练手册](sft-training.md)。配置名称包含LR、Micro/累积/有效Batch、调度轮数与stop量、warmup、weight decay、seed和时间戳。训练代码指纹保持 `0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`。
