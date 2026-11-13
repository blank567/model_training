# 手动运行 100M / 1B 预训练

数据已准备并通过 86 项独立验收；本文命令由用户在 AutoDL 远程终端执行。助手没有启动本轮 GPU 训练。

2026-10-24 原实例重新开机后，远程 CUDA/BF16 与真实训练加载器预检已通过，中英 train/validation 的 CPU Batch 均为 `[8, 1024]`。冻结配置时间戳为 **`20261024_192116`（北京时间）**。用户已完成 32 步中断检查并自行恢复训练；截至本次核查已超过 8,000 步，累计约 7.7 万 Token/s。实时状态见 `runs/pretrain_100m_1b_20261024_192116/metrics.jsonl`，这不是最终运行报告。

模板配置为 `configs/pretrain/100m_1b.json`，本轮实际使用下方带时间戳的冻结副本，读取 `artifacts/pretrain/wikipedia_1b/manifest.json`。模型为 100,092,672 参数、上下文 1024、BF16；micro batch 8、梯度累积 2、关闭激活重算。本轮已在目标 GPU 运行，峰值已分配显存约 8.93 GiB，CUDA 缓存及其他进程显存另计。下方初次启动流程留作说明，当前正在训练时只查看日志。

每次更新处理 16,384 Token，中英各半。总计 61,036 次更新，实际为 **1,000,013,824 Token**（中英各 500,006,912）；warmup 1000 步，之后 cosine decay，峰值学习率 0.0003。每 1000 步验证和保存，每次验证每语言 16,384 Token。启用 W&B 和 TensorBoard。

## 环境与会话

在远程终端加载本次已经配置好的环境。环境文件会切换到项目根目录、加载项目环境并导出本次的 `RUN_STAMP`、`RUN_DIR`、`RUN_CONFIG`、`RUN_LOG`：

```bash
source /root/autodl-tmp/model_training/runs/configs/pretrain_100m_1b_20261024_192116.env
nvidia-smi
ls -lh artifacts/pretrain/wikipedia_1b/manifest.json
.venv/bin/python -m llm_lab.cli pretrain --config "$RUN_CONFIG" --validate-only
```

`--validate-only` 只检查配置，不启动训练。本次独立配置已创建且已运行；下方“先跑 32 步”仅供首次启动时参考。当前不要重复启动第二个训练进程。现有进程停止后才使用原冻结配置和 `last.pt` 继续。

## 为新实验生成固定时间戳

以下生成步骤用于下一次独立新实验。本次使用上面的已准备环境文件即可。

新实验启动时生成一次北京时间戳，例如 `20261024_193000`。训练目录、独立配置、控制台日志和 W&B 名称共用它；后面的 32 步检查与正式续训使用同一个实验身份。

```bash
export RUN_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
export RUN_DIR="runs/pretrain_100m_1b_${RUN_STAMP}"
export RUN_CONFIG="runs/configs/pretrain_100m_1b_${RUN_STAMP}.json"

.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path.cwd().resolve()
config = json.loads((root / "configs/pretrain/100m_1b.json").read_text())
config["project_root"] = str(root)
config["output_dir"] = os.environ["RUN_DIR"]
config["train"]["wandb_run_name"] += "-" + os.environ["RUN_STAMP"]
path = root / os.environ["RUN_CONFIG"]
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print("本次配置：", path)
print("训练目录：", config["output_dir"])
PY

.venv/bin/python -m llm_lab.cli pretrain --config "$RUN_CONFIG" --validate-only
```

配置以独占创建模式写入，已有同名配置时会报错并保留原文件。首次训练前只创建配置父目录；训练器会创建本次训练目录。基础 `configs/pretrain/100m_1b.json` 仍作为模板保留。

## 先跑 32 步，再按同一计划继续

以下调用保持 61,036 步的正式学习率计划，只在第 32 步保存并正常停止，不是把总计划改成 32 步：

```bash
.venv/bin/python -u - <<'PY'
import os
from llm_lab.config import load_config
from llm_lab.engine.pretrain import run_pretrain
print(run_pretrain(load_config(os.environ["RUN_CONFIG"]), stop_after_steps=32))
PY
```

预期 summary 为 `status=simulated_interruption`、`step=32`、`tokens_seen=524288`。查看 Loss、显存和 W&B，确认没有 OOM 或非有限数值后，执行：

```bash
.venv/bin/python -u -m llm_lab.cli pretrain \
  --config "$RUN_CONFIG" \
  --resume "$RUN_DIR/last.pt"
```

此命令恢复模型、AdamW、学习率调度、随机数和 W&B 原运行，不会从第 0 步重训。不要将短测写成 `--max-steps 32`，那会改变计划与恢复身份。第 32 步前没有正式 validation（首次在第 1000 步），此短测用于检查稳定性与资源，不证明模型效果。

希望断开 SSH 后继续运行时，正式续训可以使用下面的后台命令，替代上面的前台命令，二者只执行一个：

```bash
nohup .venv/bin/python -u -m llm_lab.cli pretrain \
  --config "$RUN_CONFIG" \
  --resume "$RUN_DIR/last.pt" \
  >> "$RUN_DIR/console_${RUN_STAMP}.log" 2>&1 < /dev/null &
echo "训练 PID: $!"

tail -f "$RUN_DIR/console_${RUN_STAMP}.log"
```

记录打印的训练 PID；需要停止后台训练时，对这个 PID 发送 `kill -INT`，等待当前更新结束并保存。`tail -f` 中按 `Ctrl+C` 只退出日志查看，后台训练继续。

## 暂停、恢复与监控

按一次 `Ctrl+C`，等待当前更新结束并保存 `last.pt` 后再关闭终端；恢复仍用上面的 `--resume` 命令。不要在同一输出目录并发启动两个进程。恢复期间保持训练源码、模型配置、Batch、数据和环境身份不变；如需调整它们，使用独立新实验。

重新打开终端时，先加载项目环境，再填写原实验的时间戳并恢复变量；保留磁盘上的原配置，只执行续训命令：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export RUN_STAMP="20261024_193000"  # 替换为要恢复的原实验时间戳。
export RUN_DIR="runs/pretrain_100m_1b_${RUN_STAMP}"
export RUN_CONFIG="runs/configs/pretrain_100m_1b_${RUN_STAMP}.json"
```

恢复时不要重新执行 `date` 或生成新配置。W&B 名称也属于恢复身份，应继续沿用原时间戳。

```bash
watch -n 1 nvidia-smi
```

W&B 面板：[bilingual-training-lab](https://wandb.ai/mblankm-peking-university/bilingual-training-lab)。

Checkpoint 位于 `$RUN_DIR/last.pt` 和 `$RUN_DIR/best.pt`，曲线位于同目录的 `metrics.jsonl`、`tensorboard/`，控制台日志为 `$RUN_DIR/console_${RUN_STAMP}.log`。固定的 Checkpoint 文件名配合带时间戳的父目录，保留现有恢复流程。`best.pt` 只由 validation 选择，独立 test 留待最终评测。

本运行费用软上限 60 元，跨恢复累计；每次启动最长 18 小时。项目仍遵守 500 元总预算和 100 元 reserve。账本只估计训练进程费用，实例空闲、数据准备和存储需要结合平台账单核对；训练退出不会自动关机。旧小 Batch 基线约 11–12 小时的外推不代表新配置实测耗时。
