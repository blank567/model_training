# 300M LoRA 单轮 rank 与 dropout 对照

本轮固定已选学习率 `2e-4`。先串行运行 `r8/alpha16` 和 `r32/alpha64`，再固定这两组中开发集表现较好的 rank，分别新增 `dropout=0.05`、`dropout=0.1`。胜出 rank 的第一组结果作为 `dropout=0` 基线复用，共运行四组，不为胜出配置追加训练轮数。

## 固定条件

| 项目 | 设置 |
| --- | --- |
| 初始化 | 同一预训练 303M Base，每个实验重新初始化 Adapter |
| 数据 | 同一 100,000 条正式训练集，中文为简体中文 |
| 学习率 | `2e-4` |
| 轮数 | `epochs=1`，`stop_epochs=null` |
| 更新数 / 调度长度 | 均为 3,125 步 |
| Warmup | `0.03`，即 94 步；Cosine 最低比例 `0.1` |
| Batch | Micro 8 × 梯度累积 4 = 有效 Batch 32 |
| 目标层 | Attention q/k/v/out 与 MLP gate/up/down |
| 精度 / seed | BF16 / 42 |
| 优化器 | AdamW，betas `[0.9,0.95]`，weight decay `0.01`，梯度裁剪 `1.0` |
| 记录 | W&B 离线、TensorBoard、JSONL；每 100 步验证及保存 |

## 顺序与选型

1. rank 对照：`r8/alpha16/dropout0` → `r32/alpha64/dropout0`。
2. 两组必须都完整训练一轮，再按独立辅助开发集 2,000 条的终点 macro Loss 选 rank；相同值按候选 key 排序确定。
3. 固定胜出 rank 和 `alpha=2r`，独立训练 `dropout0.05` → `dropout0.1`；复用已完成的 `dropout0`，按同一开发集指标比较三组。
4. 每组内部仍按主验证集保留 `best.pt`，另保留支持完整断点恢复的 `last.pt`。开发集终点用于配置排名，主验证 best 用于该组模型保留，两者分别记录。

2026-10-28 的 r16 实验使用三轮调度，不纳入本轮一轮调度的 rank 排名；它只作为历史参考。这次没有新增 r16 对照，因此结论只比较 r8 与 r32，不能宣称选出了 r8/r16/r32 三者中的最优值。

本轮不读取已使用的最终 1,000 题测试结果来调参，也不自动重复最终测试。开发集 Loss、原语料保持 Loss 和实际生成任务效果含义不同，不把开发集 Loss 下降写成任务准确率提高。

首次在线启动在 W&B 初始化时超时，未进行参数更新。已核对直连超时、AutoDL 加速线路返回 HTTP 403，因此为本轮重新冻结离线 W&B 配置；不改旧失败配置或旧模型。离线记录保存在各实验的 `wandb/offline-run-*`，网络恢复后可执行 `.venv/bin/wandb sync /完整路径/offline-run-...` 上传；当前 W&B 网页不会显示本轮实时曲线，可用 TensorBoard 或 JSONL 查看。

2026-11-01补充：网络已通过专用SSH隧道修复，四组完整离线记录均已补传并逐条核对到3,125步。上段保留初次运行状态；本轮冻结配置仍为offline，后续未冻结模板恢复online。具体链接、依赖和启动命令见[W&B网络修复](wandb-network-20261101.md)。

## 入口和产物

- `scripts/train_lora_one_epoch.py`：单轮 prepare/check/start/resume；配置包含该入口 SHA，目录和 W&B 名称包含 `r`、`alpha`、`ld`、LR、Batch、`ep1` 和时间戳。
- `scripts/run_lora_ablation.py`：两组 rank 完成后才创建胜出 rank 的 dropout 配置，任一组失败或未完成一轮则停止后续，不自动扩大训练量。
- 正式配置在 `runs/configs/`，权重在 `runs/` 根目录，控制台日志在 `runs/logs/`。控制器、计划、rank 选择和汇总在 `runs/assistant/system/lora_rank_dropout_<时间戳>/`。
- 完整结果为控制器目录的 `comparison.json`，包含四组的开发集、主验证、原语料保持、可训练参数和 Checkpoint 路径；不覆盖上一轮学习率筛选指针或产物。
- 旧 `src/llm_lab`、`src/llm_sft`、`src/llm_lora` 和旧训练入口保持原 SHA。新入口仅在本进程调用时扩展命名，并在结束后还原；它复用相同的配置、来源、Checkpoint、GPU 锁和预算核验。

在 AutoDL 查看实时进度：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
.venv/bin/python scripts/lora_ablation_status.py
```

恢复某一组时使用该组原冻结配置：

```bash
.venv/bin/python -u scripts/train_lora_one_epoch.py resume --config /完整路径/该组配置.json
```

仅在原控制器和训练进程均已退出后手动恢复。恢复继续原定一轮的剩余更新，不增加一轮；完整队列也可按原 `plan.json` 重新启动控制器，它会核验并复用已完整完成的组。
