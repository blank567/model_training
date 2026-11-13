# 单卡训练运行手册

本手册对应自研 303M 的预训练、通用 SFT / LoRA 和阅读专项 RL 主线。实验结果见[完整报告](experiment-results.md)，数据来源、许可和构建见[数据手册](data-guide.md)，模块见[架构](architecture.md)。命令从项目根目录执行，`/path/to/...` 和时间戳占位符必须换为自己的实际路径。

## 1. 环境与目录

正式训练使用远程 Linux / NVIDIA GPU；Windows 用于开发与 CPU 验证。GPU 锁、进程扫描和队列依赖 Linux / POSIX，不能直接把 Windows 测试当作远程验收。项目放在可写数据盘，例如 `/root/autodl-tmp/model_training`；系统盘、持久数据盘和实例生命周期分别核对。

| 目录 | 内容与约束 |
| --- | --- |
| `data/`、`.cache/` | 原始来源与下载缓存，不进入 Git |
| `artifacts/` | 验收后的 manifest、Tokenizer、Token 流和数据发布 |
| `runs/<参数与时间戳>/` | 用户正式实验；保留原始路径和身份 |
| `runs/configs/`、`runs/logs/` | 冻结配置 / `.env` / 身份文件；控制台日志 |
| `runs/assistant/{smoke,benchmarks,inference,data_prepare,system}/` | 助手短测、测速、推理、数据准备与诊断 |
| `runs/budget_ledger.json`、`.gpu_training.lock` | 共享账本和单卡互斥身份，不随整理改名 |
| `docs/results/` | 公开汇总 CSV / 图表，不是权重或原始预测备份 |

历史训练基线为 Python 3.10、Ubuntu 22.04、单张 vGPU 48 GB；未确认底层物理 GPU 型号。新实例先检查实际资源：

```bash
nvidia-smi
python --version
df -h /root/autodl-tmp
```

历史直接依赖为 PyTorch 2.9.0 / CUDA 12.8 wheel、NumPy 1.26.4、Tokenizers 0.22.2、Transformers 4.57.1、PEFT 0.17.1、TRL 0.24.0、Accelerate 1.11.0、Datasets 4.3.0、OpenCC 1.4.2、W&B 0.30.0、TensorBoard 2.20.0。安装清单见 [requirements/remote.txt](../requirements/remote.txt)，并非完整传递依赖锁文件。

阅读数据发布另使用隔离的 OpenCC 1.1.9 工具，不能用全局版本替换后冒充同一数据身份；具体见[数据指南](data-guide.md)。

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
# 在平台已经可用的 Python 3.10/3.11 下显式安装新项目环境：
PYTHON_BIN=python bash scripts/setup_remote.sh --install
source .venv/bin/activate
python -m pip check
```

不传 `--install` 时安装脚本只检查。它不修改 NVIDIA 驱动或系统 CUDA Toolkit；驱动、镜像 Toolkit 和 wheel 携带的运行库是三个层次。`env_remote.sh` 把缓存、凭据和临时文件指向项目盘，安装后记录 `runs/pip-freeze.txt`。不要为了复现覆盖已有训练环境。

## 2. 分清 fixture、新实验与原实验恢复

1. **CPU fixture**：使用仓库原创小样本，只验证流程，不下载正式语料或权重。
2. **原实验复现 / 恢复**：必须持有匹配的 Base、Tokenizer、数据与冻结身份；公共仓库不包含这些产物。
3. **新数据 / 新模型实验**：使用新模板、输出名和身份；不能将历史入口的固定 SHA 校验删掉后当作原实验继续。

部分高层入口是历史实验的受限启动器：`train_300m.py` 固定原 3B manifest / Tokenizer；`train_reading_sft.py` 固定原 Full SFT100k 父模型；RL 还要求 comparison plan 和校准身份。仅替换 JSON 路径不能让任意新数据通过这些入口。源码中的约束见 [train_300m.py](../scripts/train_300m.py)、[train_reading_sft.py](../scripts/train_reading_sft.py) 和 [train_reading_rl.py](../scripts/train_reading_rl.py)。

CPU fixture 的最小流程：

```bash
python -m pip install -e '.[dev]'
python -m llm_lab.data.acquire --config configs/data/acquire_smoke.json
python -m llm_lab.data.prepare --config configs/data/prepare_smoke.json
python -m llm_lab.cli pretrain --config configs/pretrain/smoke.json --validate-only
python -m llm_lab.cli pretrain --config configs/pretrain/smoke.json \
  --output-dir "runs/assistant/smoke/manual_$(date +%Y%m%d_%H%M%S)"
```

该配置仅训练 4 步。`examples/_outputs/` 是可重建 fixture，正式产物默认拒绝覆盖。Windows 使用自己的 Python，并将 `TEMP` / `TMP` 放 E:；设置 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`，避免随机 Byte-BPE 输出触发 GBK 编码错误。

GPU 探针只验证真实小矩阵计算，不启动模型训练：

```bash
python -m llm_lab.runtime.probe --device cuda --require-cuda \
  --output runs/assistant/system/environment.json
```

CUDA / BF16 探针通过后，还需对目标模型测显存和吞吐；不能从矩阵乘法耗时推算长跑速度。

## 3. 共同的冻结、启动和单轮规则

`prepare` 读取模板并写入带参数 / 时间戳的配置、`.identity.json` 和 `.env`；不启动训练。`check` 校验身份及相应资源，`start` 新建运行，`resume` 从原目录 `last.pt` 继续。配置、Base、数据、Tokenizer、关键源码或调度改变时另建实验。

**所有新建后训练实验统一 `epochs=1`、`stop_epochs=null`，Warmup / 调度按真实一轮更新数计算。** 历史多轮结果保留，不调用旧 `--axis epochs` 计划，也不把多轮调度配合一轮停止点当作新默认。PPO 每批 rollout 内部两次优化不等于数据训练两轮。

开始前核对：数据验收、磁盘余量、预算、GPU 空闲和源码同步状态。只允许一组训练持有 GPU 锁；排队控制器运行时不要旁路手动启动同组。

## 4. 预训练

持有原 3B 验收产物时，使用受限启动器：

```bash
.venv/bin/python scripts/train_300m.py prepare \
  --template configs/pretrain/300m_3b_optimized.json
# 换成 prepare 实际返回的 env 文件：
source runs/configs/pretrain_300m_3b_YYYYmmdd_HHMMSS.env
.venv/bin/python scripts/train_300m.py check --config "$RUN_CONFIG"
.venv/bin/python -u scripts/train_300m.py probe --config "$RUN_CONFIG" --steps 32
```

`probe` 保持正式总学习率计划并保存前 32 步，随后用 `resume` 继续。不要把 `max_steps` 改为 32 再称作相同调度的正式前段。

```bash
nohup .venv/bin/python -u scripts/train_300m.py resume \
  --config "$RUN_CONFIG" >> "$RUN_LOG" 2>&1 < /dev/null &
echo "训练 PID: $!"
tail -f "$RUN_LOG"
```

新验收语料可使用通用预训练 CLI，并自行建立新配置和输出；它不等于历史启动器的 GPU 锁 / 正式计划全部校验：

```bash
python -m llm_lab.cli pretrain --config /path/to/new_pretrain.json --validate-only
python -m llm_lab.cli pretrain --config /path/to/new_pretrain.json
# 已有原配置与同一输出时：
python -m llm_lab.cli pretrain --config /path/to/new_pretrain.json \
  --resume /path/to/original_run/last.pt
```

新通用配置仍需遵守数据身份、配比、预算和单卡互斥要求，不能与其他训练同时占用 GPU。

## 5. 通用 SFT、LoRA 与阅读专项 SFT

通用 Full SFT / LoRA 从已完成的 303M 预训练 Base 开始。先在**新模板**填写自己的验收输入、对应哈希、监控帐号及预算，核对真正一轮调度。已有模板绑定的旧哈希不能与新路径混用。

```bash
.venv/bin/python scripts/train_sft.py prepare \
  --template configs/posttrain/custom_300m_sft.json
.venv/bin/python scripts/train_lora_one_epoch.py prepare \
  --template configs/posttrain/custom_300m_lora_r8_one_epoch.json
```

上面分别创建两组配置；选择一组返回的 `.env` 后，只启动这一组：

```bash
source /path/to/selected_run.env
.venv/bin/python scripts/train_sft.py check --config "$RUN_CONFIG"
nohup .venv/bin/python -u scripts/train_sft.py start \
  --config "$RUN_CONFIG" >> "$RUN_LOG" 2>&1 < /dev/null &
echo "训练 PID: $!"
tail -f "$RUN_LOG"
```

LoRA 将命令中的 `train_sft.py` 替换为 `train_lora_one_epoch.py`。该单轮入口提供 `check / start / resume / status`，没有 `probe` 子命令；Full SFT 的 `probe --steps 16` 则保持原计划短跑，短跑后必须 `resume`。LoRA Adapter-only 权重不是完整 Base，加载和恢复始终需要同一预训练 Base。

阅读专项使用独立 [train_reading_sft.py](../scripts/train_reading_sft.py)，从已选 Full SFT100k 而非 LoRA 结果初始化；该历史入口固定父模型 SHA，要求验收的 50k train / 2k development 数据：

```bash
.venv/bin/python scripts/train_reading_sft.py prepare \
  --template configs/posttrain/custom_reading_full_50k.json
# 取得本组配置后：
.venv/bin/python scripts/train_reading_sft.py check --config /path/to/frozen.json
.venv/bin/python -u scripts/train_reading_sft.py start --config /path/to/frozen.json
```

需要后台时按上面的 `nohup` 方式重定向至本组日志；中断后各入口统一使用原配置的 `resume`，不重复 prepare。

## 6. DPO / PPO / GRPO

三种算法分别从同一已完成阅读 SFT 起点开始，不接着前一算法继续。准备条件为：冻结 comparison plan、通过隔离审计的数据 arm、初始模型与数值 / 奖励校准；DPO 额外需要 preference manifest。`prepare` 会读取并校验这些产物，不能用空配置占位。

```bash
.venv/bin/python scripts/train_reading_rl.py --help
.venv/bin/python scripts/train_reading_rl.py prepare \
  --plan /path/to/plan.json --arm old_matched --algorithm ppo \
  --calibration /path/to/calibration.json
```

数据 arm 可选 `old_matched / new_matched / old_full_50k`；算法可选 `ppo / grpo / dpo`。DPO 另传 `--preferences /path/to/preference_manifest.json`。旧 / 新 matched 的 prompt 数相同，不代表 DPO 筛选后的偏好对也等于原始 prompt 数。

```bash
source /path/to/rl_run.env
mkdir -p "$(dirname "$RUN_LOG")"
.venv/bin/python scripts/train_reading_rl.py check --config "$RUN_CONFIG"
nohup .venv/bin/python -u scripts/train_reading_rl.py start \
  --config "$RUN_CONFIG" >> "$RUN_LOG" 2>&1 < /dev/null &
echo "训练 PID: $!"
```

恢复时仍用同一配置，将 `start` 换为 `resume`。RL 在采样 Batch 边界处理停止，单批可能比普通 SFT 更新更慢。初始策略无正确奖励、或 GRPO 无组内混合奖励时先做任务校准，不盲目长跑。

该历史 RL 生成入口的 W&B entity 来自 [DEFAULT_TRAIN](../src/llm_rl/config.py)，没有 CLI 覆盖参数。迁移到其他帐号需使用显式覆盖帐号且重新冻结的新配置生成流程，不能改写生成后的 config 绕过身份检查，也不能误以为环境变量一定覆盖显式 entity。

## 7. 日志、停止、恢复与监控

`nohup` 使进程在断开终端后继续；`tail -f` 只查看日志，其中 `Ctrl+C` 不会停止训练。前台训练进程中的 `Ctrl+C` 则会请求训练退出。停止前先核实 PID，避免使用历史记录里的旧 PID：

```bash
TRAIN_PID=12345   # 替换为刚启动并确认的训练 PID
ps -fp "$TRAIN_PID"
kill -INT "$TRAIN_PID"
```

等待本组 `summary.json` 和 `last.pt` 写完、进程退出后，再恢复或关闭实例；不要用强制结束代替正常保存。RL 也支持在本组输出放置 `STOP`，恢复前移除该停止标记。控制器的 STOP 只阻止后续组，不代表当前组已退出。

`best.pt` 用于验证 / 开发规则选模，`last.pt` 用于完整恢复；后者还需 optimizer、调度进度、RNG、采样游标和适用的 GradScaler。只有 adapter 或一份日志不能精确续训。不要在原运行旁复制一个 config 再额外加一轮。

```bash
source scripts/env_remote.sh
.venv/bin/wandb login
# TensorBoard 示例，RUN_DIR 为所选实验目录：
.venv/bin/tensorboard --logdir "$RUN_DIR/tensorboard" --host 127.0.0.1 --port 6006
```

Key 只输入登录提示，凭据和原始 W&B 事件不提交 Git。离线模式在**冻结前**设 `train.wandb_mode="offline"`；补传用 `.venv/bin/wandb sync /path/to/offline-run-...`。通用 SFT / LoRA 模板可在冻结前配置自己的 `wandb_entity`；部分历史运维脚本绑定原帐号，不直接用于新项目。

查看训练 / 验证 Loss、学习率、梯度、Token/s 和显存，保留本地 JSONL；W&B 网页登录成功不等于远程 SDK 已上传。TensorBoard 通过受控端口或 SSH 转发访问。

## 8. 推理与最终评测

使用模型对应的 manifest / Tokenizer，Base 续写不等于聊天：

```bash
.venv/bin/python -m llm_lab.cli generate --checkpoint /path/to/base/best.pt \
  --device cuda --prompt "机器学习是" --max-new-tokens 64
.venv/bin/python scripts/chat_sft.py --checkpoint /path/to/sft/best.pt \
  --device cuda --prompt "请简要解释梯度下降。" --max-new-tokens 128 --greedy
.venv/bin/python scripts/chat_reading_rl.py --checkpoint /path/to/rl/best.pt \
  --device cuda --prompt "材料：北京是中国的首都。问题：中国的首都是哪里？只输出原文片段。"
```

Base CLI 是贪心重算窗口路径，SFT 生成提供采样 / 重复惩罚，RL 有独立 KV 适配器；不能把不同解码设置的输出差异都归为训练效果。

通用最终评测使用 [evaluate_lora.py](../scripts/evaluate_lora.py) 的 `--selection`，全面统一评测使用 [evaluate_posttrain_comprehensive.py](../scripts/evaluate_posttrain_comprehensive.py) 的 `--plan`。这些计划须固定模型清单、数据身份、解码协议和输出；不是“给一个权重路径就自动有测试集”。预训练验证可用 `llm_lab.cli evaluate --split validation`，最终 test 另需 `--allow-test`。

按开发集选模型后再运行冻结最终测试；best 与 last 结果分开，不能看过测试后用 last 替换 best。CE、EM、F1 的口径与污染 / 公平性限制见[实验报告](experiment-results.md)。

## 9. 同步、存储和费用

源码同步是本地到远程单向快照，配置示例见 [sync_autodl.example.json](../configs/deployment/sync_autodl.example.json)，入口为 [sync.ps1](../scripts/sync.ps1) 的 `Start / Status / Pause / Resume / Stop / Once`。实际配置与密钥放忽略目录，核实主机公钥；训练前暂停同步并在远程设置 `.sync-pause`，避免中途改变源码身份。运行结束后再同步新版本。

打包 / 同步工具有自己的白名单，不遵循 `.gitignore`；本地部署包可能带机器配置，不能直接当公共源码发行包。公开资料应来自审查后的 Git 提交。

历史模板含 3.03 元/小时和 500 元项目预算 / 100 元余量等研究约束，部分由启动器固定校验。它们不是当前云平台报价或可随意覆盖的预算承诺。进程估费不含所有闲置、存储和传输费用；训练退出不会自动关机或释放实例。

磁盘不足时扩容或停在更新边界，不自动删除权重。保留选定 best、可恢复 last、配置、manifest、Tokenizer、环境和必要日志，核实备份后再处理实例。冻结模型、配置和输入不为目录整理搬动。
