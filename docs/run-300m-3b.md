# 手动运行 300M / 3B 预训练

2026-10-25 状态变更：用户已要求停止并清空 `20261025_140717` 这轮 300M 输出。该轮在 step 1,155 正常保存停止，随后权重/日志目录已删除；100M、全部语料、预算历史和小型清理记录保留。旧时间戳已标记 `cleared`，下方该时间戳命令属于历史交接，不能再次启动或恢复。受限 GPU 测速已完成，新正式配置为 **8×4、关闭重算、时间戳 `20261025_150858`**；当前执行入口请使用 [测速后的运行说明](run-300m-optimized.md)。

2026-10-25：100M 运行 `20261024_192116` 已完成 61,036 步、1,000,013,824 个处理 Token，`summary.status=completed`；读取远程时 GPU 空闲。下一轮是从随机参数独立训练 303,350,784 参数模型，不加载或覆盖 100M 权重。

配置模板为 `configs/pretrain/300m_3b.json`，启动代码为 `scripts/train_300m.py`。模型实现仍在 `src/llm_lab/models/decoder.py`，训练循环仍在 `src/llm_lab/engine/pretrain.py`；本次只新增配置和外部启动器，保留核心源码身份，使旧模型仍可按原条件加载/恢复。旧 `300m_3b.candidate.json` 保留为历史候选，不用于本轮正式启动。

## 配置

| 项目 | 值 |
| --- | --- |
| 参数量 | **303,350,784** |
| 模型 | Dense Decoder，GQA、RoPE、RMSNorm、SwiGLU、tied embedding |
| dim / layers / Q heads / KV heads / FFN | 1024 / 24 / 16 / 4 / 2816 |
| 词表 / 上下文 | 32,000 / 1024 |
| 精度 / 激活重算 | BF16 / 开启 |
| Micro batch / 梯度累积 | 4 / 4 |
| 每步处理量 | 16,384 Token，中英各 8,192 |
| 步数 / 处理量 | **183,106 / 3,000,008,704 Token** |
| 学习率 | 峰值 0.0003，warmup 3000，cosine，最低比例 0.1 |
| AdamW | betas=(0.9, 0.95)，weight decay 0.1，梯度裁剪 1.0 |
| 验证 / 保存 | 每 1000 步，每语言验证 16,384 Token |
| 日志 | 每 10 步，W&B online 和 TensorBoard |
| 费用与时间停止条件 | 180 元进程估费累计上限；每次启动最多 72 小时 |

单价暂沿用原实例的 **3.03 元/小时**，这是前次已提供的租价，不是此次自动读取的平台账单。租价变更时在首次训练前通过新的配置准备步骤填写；运行成本仍与数据准备、空闲及存储账单区分。项目共用 500 元上限、100 元余量和同一个 `runs/budget_ledger.json`，新旧运行各自累计。180 元按 3.03 元/小时约能覆盖 59.4 小时，费用阈值可能先于 72 小时时间阈值触发；这些是停止条件，不是全程耗时或费用保证。

实际训练池为 **3,100,003,021 Token**，中英各约 15.5 亿，中文统一简体。数据入口 `artifacts/pretrain/bilingual_3b/manifest.json` 的 SHA256 为 `f1dbbba35db3f414a36359f96c3a8a6a3b4bba1fb94b09284a5a0fef75c6c7b1`。Tokenizer、验证/测试集合与已验收版本固定，数据制备详情见 [data-3b.md](data-3b.md)。现有采样器随机有放回取窗，处理 30 亿 Token 不代表无重复覆盖 30 亿独立位置。

## 本次已准备好的实验

本次已在远程独占生成配置和环境文件，时间戳为 **`20261025_140717`（北京时间）**：

- 配置：`runs/configs/pretrain_300m_3b_20261025_140717.json`
- 环境：`runs/configs/pretrain_300m_3b_20261025_140717.env`
- 输出：`runs/pretrain_300m_3b_20261025_140717/`
- 控制台日志：`console_20261025_140717.log`，位于本次输出目录

本地启动器测试 13 passed、1 skipped（Linux 锁）；远程 CPU **14 passed**，补齐排他锁分支。远程 check 与原 CLI 的配置校验通过，可用磁盘约 25.387 GiB；读取时没有训练或 GPU 计算进程。旧 100M 的源码指纹、原冻结配置与 checkpoint 状态保持相同。非敏感交接报告为 `tmp/pretrain/handoff_300m_20261025_140717.json`。这些检查没有启动 GPU 模型训练，输出目录尚未创建。

在 AutoDL 终端直接加载本次环境，先检查条件，再做 32 步短测：

```bash
source /root/autodl-tmp/model_training/runs/configs/pretrain_300m_3b_20261025_140717.env
.venv/bin/python scripts/train_300m.py check --config "$RUN_CONFIG"
.venv/bin/python -u scripts/train_300m.py probe --config "$RUN_CONFIG" --steps 32
```

首次启动会先校验约 12.4 GB 的训练流和初始化 W&B，前 10 步后才输出第一条训练指标；初始化阶段暂时没有新指标不等于进程失效。

## 未来独立实验的准备方法

下面只用于未来新实验。本次沿用上述 `20261025_140717` 环境即可。生成新配置的步骤只准备文件，不训练：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
.venv/bin/python scripts/train_300m.py prepare --template configs/pretrain/300m_3b.json
```

准备结果会打印 `runs/configs/pretrain_300m_3b_时间戳.env` 的完整路径。生成一次后沿用该路径，后续不要重复 prepare 或创建新时间戳。配置、环境文件、输出目录、控制台日志和 W&B run 共用同一个时间戳。

加载生成的 `.env` 后先检查条件，再做 32 步短测：

```bash
source /root/autodl-tmp/model_training/runs/configs/pretrain_300m_3b_时间戳.env
.venv/bin/python scripts/train_300m.py check --config "$RUN_CONFIG"
.venv/bin/python -u scripts/train_300m.py probe --config "$RUN_CONFIG" --steps 32
```

`probe` 调用原训练器的 `stop_after_steps=32`，正式总计划仍为 183,106 步。它从随机参数开始，并保存完整 `last.pt`。不能用 `--max-steps 32` 改短学习率调度。短测主要确认无 OOM、Loss/梯度有限、checkpoint 写入及断点状态；32 步未覆盖第 1000 步的验证周期，不能据此承诺最终吞吐、质量或费用。

本次未由助手运行 300M 的 GPU 训练。4×4 与激活重算是保守起点，实际峰值显存和速度由用户短测记录。若需更换 Batch、重算开关、学习率或数据，先建立独立新配置；不能修改已训练配置后强行恢复。

## 正式继续与后台运行

32 步短测成功退出后，接着同一 `last.pt` 后台训练：

```bash
nohup .venv/bin/python -u scripts/train_300m.py resume \
  --config "$RUN_CONFIG" \
  >> "$RUN_LOG" 2>&1 < /dev/null &
echo "训练 PID: $!"
tail -f "$RUN_LOG"
```

恢复模型、AdamW、学习率调度、RNG、采样器和原 W&B run，从已保存步数继续。SSH 断开后后台进程继续运行，`tail` 中 `Ctrl+C` 只退出日志查看。

如果选择跳过短测，首次新训练可使用 `start --config "$RUN_CONFIG"`；它会拒绝非空输出目录。已完成短测或已有 checkpoint 时使用 `resume`。同一个实验只运行一个训练进程。

## 停止与恢复

对启动时打印的 PID 执行 `kill -INT PID`，等待训练器保存 `last.pt` 并正常退出；`summary.json` 的 `signal_stop` 表示已完成保存停止。未来重新连接后 source 原 `.env`，再执行上述 `resume`，保持同一配置、数据、源码和环境身份。

查看状态不启动训练：

```bash
.venv/bin/python scripts/train_300m.py status --config "$RUN_CONFIG"
```

## 与 100M 的隔离及资源边界

- 两轮分别实例化模型，各用带时间戳的输出目录、checkpoint、控制台日志和 W&B run。300M 不传入 100M checkpoint；词表与评测数据以只读方式复用。
- 启动器在训练全程持有项目排他锁，启动前还检查旧训练进程和 GPU 计算进程，防止新命令重复启动。旧 CLI 没有采用新锁，因此运行 300M 时仍应只使用一个训练入口，不手动另开旧 CLI；共享预算账本按单进程设计。
- 同一张 GPU 同时训练会争抢显存/算力，还可能造成无锁预算更新冲突；新流程会拒绝检测到的并行训练。当前 100M 已完成，因此按此流程启动 300M 不会中断 100M。
- 准备新 run 时建议可用空间至少 20 GiB。300M 完整单 checkpoint 约 3.39 GiB，`last/best/atomic tmp` 峰值约 10.17 GiB；启动器根据本轮已有 checkpoint 的独立 inode 占用调整恢复时的新增空间门槛，并保留至少 8 GiB 余量。实际日志与元数据还会占空间。
- 本机仅进行配置、meta 参数计数和 CPU 测试；远程 check 不执行模型训练。300M 的 CUDA/BF16、激活重算、显存和训练速度需要用户实际短测确认。
