# 远程单卡运行手册

本文包含早期部署与可选 Qwen 工作流的历史说明。自研 303M 已完成 GPU 预训练及 SFT / LoRA / 阅读 RL，当前入口见[项目 README](../README.md)和[实验结果](experiment-results.md)。文中实例“当前”状态、容量、价格与待验收描述均需在新实例重新核查；新建后训练实验只用一轮，不复用旧多轮计划。

## 准备环境前先明确边界

正式训练在远程 Linux 单卡运行，当前已租用 AutoDL vGPU 48 GB，GPU 训练能力仍需实际验收。本地 Windows 负责源码开发、自动同步、数据 fixture 和 CPU 正确性检查。

远程将整个仓库放在平台提供的可写数据盘目录，例如 `/data/model_training`，不要默认放进系统盘。Windows 的 E: 策略适用于本机；Linux 没有盘符，需要明确所选数据盘挂载点。确保数据、`.cache`、`.venv`、`tmp`、`runs`、`artifacts` 和 Checkpoint 都位于该目录或明确选择的持久化数据盘。

安装前区分三个组件：NVIDIA 驱动由平台提供，负责 GPU 访问；PyTorch CUDA wheel 携带所需的用户态 CUDA 运行库；系统 CUDA Toolkit 包括 `nvcc` 等编译工具。它们显示的版本含义不同。这里使用预编译 wheel，不修改系统 Toolkit，也不通过本项目更新驱动。`nvidia-smi` 中的 CUDA 字样不能单独证明 PyTorch 能用 GPU。

## 环境变量与安装

远程推荐 Python 3.10 或 3.11。脚本默认只验证，显式 `--install` 才创建项目 `.venv` 和安装依赖。安装包含 PyTorch、CUDA wheel 运行库、数据与后训练依赖，可能下载数 GB；不包括预训练语料或 Qwen 模型权重。

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
PYTHON_BIN=/root/miniconda3/bin/python bash scripts/setup_remote.sh --install
source .venv/bin/activate
```

`env_remote.sh` 将 `HF_HOME`、`HF_DATASETS_CACHE`、`PIP_CACHE_DIR`、`TORCH_HOME`、`XDG_CACHE_HOME`、W&B 配置/缓存/凭据和临时目录指向项目内。正式模板已启用 W&B，在线训练前先按 [W&B 指南](wandb.md) 登录；TensorBoard/JSONL 同时可用。若使用平台预装环境，可先激活该环境，然后执行 `bash scripts/setup_remote.sh`，它不会默认安装或升级。

固定远程 PyTorch 起点为 `torch==2.9.0` 的 CUDA 12.8 wheel。官方也提供该版本的其他 CUDA wheel；如平台驱动不兼容，先选择兼容 wheel 或平台镜像再运行检查，不能把导入成功当作兼容。版本来源：[PyTorch 官方历史安装命令](https://pytorch.org/get-started/previous-versions/)。

`requirements/remote.txt` 固定直接依赖版本；它不是包含全部传递依赖和 wheel 哈希的完整锁文件。安装后保存 `runs/pip-freeze.txt`，运行 manifest 同时记录实际包版本，恢复要求环境一致。

## GPU 验收

```bash
python -m pip check
python -m llm_lab.runtime.probe --device cuda --require-cuda --output runs/environment.json
```

探针记录 Python/包版本、GPU 型号、显存、BF16 支持、`torch.version.cuda`、`torch.cuda.is_available()` 和经过 CUDA 同步的真实矩阵乘法耗时。CUDA 不可用、矩阵不有限或执行失败时，要求 CUDA 的命令返回非零状态。它不会训练模型，也不会修改驱动或 Toolkit。

这份报告只证明小型 GPU 工作能运行。完整模型的峰值显存、tokens/s、Activation Checkpointing 和后训练依赖仍需短程真实实验。不要用矩阵耗时推算正式训练速度。

## 先校验再训练

数据准备命令见 [data-guide.md](data-guide.md)。获取使用固定数据快照/版本，模型训练需要固定模型 revision；先核验来源和许可，再显式执行下载。有条件时在 CPU 实例或上传前完成这些工作。

后训练先运行不下载模型的校验：

```bash
python -m llm_lab.workflows.posttrain --config configs/posttrain/sft_full.json --validate-only
python -m llm_lab.workflows.posttrain --config configs/posttrain/sft_lora.json --validate-only
```

校验通过意味着本地配置、数据结构和审计条件通过，不代表模型权重已经缓存或 GPU 训练已通过。DPO 必须先有完整的 SFT 来源目录，不能直接从 Base 把合成偏好当成第一阶段训练。

预训练短程测速使用正式 100M 配置，但用较少步数覆盖上限，并填写实际小时价：

```bash
python -m llm_lab.cli pretrain --config configs/pretrain/100m.json --max-steps 20 --output-dir runs/probe_100m --hourly-rate-cny 2.0
```

`2.0` 只是命令语法示例，必须换成当前所租实例的实际人民币小时价。数据、模型维度和上下文应保持为待测速配置。模板的 1000 步是启动候选，不代表充分预训练；预算和时间可能先结束。

`--max-steps` 改变学习率调度终点，短程 Checkpoint 不能当作调度相同的正式训练前段。正式训练使用新的 `output_dir` 并从新初始化开始，避免与短程运行混淆；先复制配置并记录变更，再运行不带短步数覆盖的命令。选择正式总步数后应保持调度配置不变再做断点恢复。

预训练 `metrics.jsonl` 中的 `tokens_per_second_including_eval` 是本次训练段累计吞吐，包含验证开销；不是纯前向/反向速度。对比时必须采用相同记录口径。`peak_cuda_bytes` 是 PyTorch 已分配内存的峰值，不等于 `nvidia-smi` 显示的全部 GPU 进程占用。

预训练生成与验证命令使用实际运行的 Checkpoint 路径：

```bash
python -m llm_lab.cli evaluate --checkpoint runs/pretrain_100m/last.pt --split validation --output runs/pretrain_100m/validation.json
python -m llm_lab.cli generate --checkpoint runs/pretrain_100m/last.pt --prompt "语言模型" --max-new-tokens 64
```

独立测试需要在最终阶段显式使用 `--split test --allow-test`。调参阶段禁止把测试集改称验证集。正式结果的路径以实际配置 `output_dir` 为准。

## 继续训练与后训练

```bash
python -m llm_lab.cli pretrain --config configs/pretrain/100m.json --resume runs/pretrain_100m/last.pt --hourly-rate-cny 2.0
python -m llm_lab.workflows.posttrain --config configs/posttrain/sft_lora.json
python -m llm_lab.workflows.posttrain --config configs/posttrain/sft_lora.json --resume
```

正式后训练前在 JSON 配置填写模型 revision、数据路径、预算小时价和本次上限。`--resume` 不带路径时选当前运行目录中最新的完整 Trainer Checkpoint；也可传入具体目录。Adapter、Optimizer、Scheduler、RNG 和运行清单需要同时存在，只有一个 adapter 文件不能恢复完整训练。

DPO 配置提供 LoRA 与全参数两条对应路线。先检查来源路径指向选定的 SFT 结果并经过验证集选择，再运行 `--validate-only`，最后启动 DPO。SFT-LoRA 的 adapter 应继续绑定原 Base，不能当作完整 Base 路径随意替换。

后训练最终评测：

```bash
python -m llm_lab.evaluation.task_eval --checkpoint artifacts/runs/sft_lora/final --data artifacts/tasks/massive/sft/test.jsonl --split test --final --output artifacts/evaluations/sft_lora_test --hourly-rate-cny 2.0
```

评测输出存在时会拒绝覆盖，便于保留历史。真实生成使用远程 CUDA；离线评分可用 `--predictions` 替代 `--checkpoint`，但离线预测文件缺少模型训练来源时不能宣称自动完成训练数据隔离审计。

这里的 `final` 是最后一步导出，并不自动等于验证集最优模型。正式使用时应按验证结果选定具体 `checkpoint-N`，然后替换上面的路径。生成评测也需要填写实际小时价并记入共享预算；预算不足时保留部分预测，不将未完成评测报告成完整测试结果。

## 查看训练曲线

正式模板默认向 W&B 记录训练和验证指标，面板横轴为 `optimizer_step`。首次登录、离线模式、恢复身份与补传命令见 [W&B 指南](wandb.md)。开启 W&B 后，本地 JSONL 仍是实验产物的一部分。

预训练配置启用 `tensorboard` 后，运行目录包含事件文件。安装了项目 tracking 或 posttrain extra 的远程环境可启动：

```bash
tensorboard --logdir runs/pretrain_100m/tensorboard --host 127.0.0.1 --port 6006
```

通过平台的端口访问机制或 SSH 端口转发查看；无需把服务直接暴露公网。主要核对 Train/Validation Loss、学习率、梯度范数和吞吐。后训练模板默认不开 TensorBoard，可在预算允许时显式启用；Trainer 状态与 JSON 产物仍需保留。

## 预算、停止与交付

总上限 500 元包括 GPU、存储、传输等。配置中的小时价/时间/单次和项目预算用于进程级控制，训练结束后仍要核对云平台实例存活时间与实际账单。为环境准备、评测、Checkpoint 保存和未知开销预留预算。

训练进程退出不等于实例停止或释放；磁盘也可能继续收费。脚本不自动关机、不删除数据、不释放实例。完成后手动确认日志和 Checkpoint 持久化、备份必要产物，再到平台执行所需的停机或释放操作。

公开交付前包含：实际配置、依赖版本、环境报告、数据 manifest、许可证和署名、验证模型选择理由、最终独立指标、失败样例、费用与复现命令。不能把 CPU 测试通过写成“4090 训练已验证”。
