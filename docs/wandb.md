# W&B 训练监控

项目固定 `wandb==0.30.0`。2026-10-23 已在 AutoDL 项目的 `.venv` 安装并通过 `pip check`，真实 SDK 离线写入、远程账号登录和在线上传自检均通过。自检中的三条 Loss 是合成指标，不是模型效果。

已验证的[项目面板](https://wandb.ai/mblankm-peking-university/bilingual-training-lab)与[在线自检运行](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/9852d3e2792c)。训练配置默认使用当前登录账号的默认空间，团队空间可显式配置。

安装后的完整远程 CPU 回归为 **113 passed**，包括真实 tiny PEFT 恢复；这不表示已经完成正式 GPU 预训练或后训练。

## 远程登录与在线验收

在 VS Code 已连接的 AutoDL 远程终端执行，API Key 只输入终端的交互提示，不放进命令参数、代码或聊天：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
.venv/bin/wandb login
.venv/bin/python -m llm_lab.runtime.wandb_check --mode online
```

默认项目名为 `bilingual-training-lab`，SDK 自检运行名为 `sdk-smoke-synthetic-metrics`。在线验收成功时 JSON 中的 `url` 是这次运行的面板地址。团队空间可在训练配置设置 `wandb_entity`；自检使用 `--entity <团队slug>`。没有账号时先在 [W&B 官网](https://wandb.ai/) 注册。

首次在线训练前需要完成登录并确认网络可达。在线模式要求认证，初始化最多等待 30 秒，结束上传最多等待 15 秒；不会静默把在线配置改成离线。初始化失败时尚未进入训练循环；训练输出已有元信息而无可恢复 Checkpoint 时，修复登录后选择新的输出目录。

## 配置与曲线

预训练的 `train` 对象、后训练 JSON 顶层均支持：

```json
{
  "wandb": true,
  "wandb_project": "bilingual-training-lab",
  "wandb_entity": null,
  "wandb_mode": "online",
  "wandb_run_name": null
}
```

正式 100M、全参 SFT、LoRA SFT、全参 DPO 和 LoRA DPO 模板已启用；CPU smoke 默认关闭。`wandb_run_name` 为空时使用输出目录名。配置中不接受 API Key；凭据由 SDK 的远程登录或运行环境提供。

预训练与后训练统一调用 `common/tracking.py`。后训练沿用现有 Trainer `Telemetry` callback 记录，没有额外添加会重复初始化的内置 W&B callback。TensorBoard 开关独立，本地 `metrics.jsonl` 仍保留。

| 训练线 | 主要指标 |
| --- | --- |
| 预训练 | `train_loss`、`learning_rate`、`grad_norm_before_clip`、`tokens_seen`、`tokens_per_second_including_eval`、`peak_cuda_bytes` |
| 预训练验证 | `validation/en/loss`、`validation/zh/loss`、对应 Perplexity 与 `validation/macro_loss` |
| SFT / LoRA | Trainer 的 `loss`、`eval_loss`、`grad_norm`、`learning_rate`、Token 吞吐、显存峰值与估计费用 |
| DPO | SFT 通用运行指标，以及 Trainer 实际返回的偏好 reward、margin、accuracy 等数值项 |

横轴定义为 `optimizer_step`；同一步先记录训练再记录验证不会丢掉其中一条。SDK 默认系统监控还可采集 CPU/GPU 使用情况；vGPU 下可见的指标受平台限制，以实际面板为准。

业务记录发送已选择的模型/训练超参数和数值指标；SDK 还会发送运行环境和依赖等元信息。不自动保存源码、Git diff、训练样例、梯度图或模型权重。日志目录、配置、NETRC 凭据和缓存分别位于运行目录及项目 `.cache/wandb`，没有改变系统级 W&B 配置。`source scripts/env_remote.sh` 会清除旧的禁用变量；运行模式由本项目 JSON 决定。

## 断点恢复

每个训练输出目录保存 `wandb_run.json`，将 W&B 身份和训练实验关联。

- 在线恢复复用原 run id，并使用 `resume="must"`，防止意外新建看似连续的曲线。
- 离线恢复创建新的 run 分段并沿用原 group；W&B 离线模式不支持原地恢复，不能把多个离线段描述为一条已经在线续写的记录。
- SDK 主线程调用前后保存并恢复已加载的 Python、NumPy、Torch RNG。训练 Checkpoint 的配置、源码、依赖、数据身份校验照常执行，不能通过开启监控绕过。
- 初始化失败会报错；训练中 SDK 日志异常则写入 `wandb_error.json` 并保留本地 JSONL 继续训练。结束上传受时限约束，未上传的本地记录保留供后续同步。

修改已有实验的监控配置或依赖属于身份变化，应新建实验，或按原配置、原代码、原环境恢复。安装 W&B 后的依赖快照为 `runs/pip-freeze-after-wandb.txt`。

## 离线记录与补传

不登录、不联网的 SDK 自检：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
.venv/bin/python -m llm_lab.runtime.wandb_check --mode offline
```

训练离线记录时，复制实验配置至新文件，将 `wandb_mode` 明确改为 `offline` 并使用新的输出目录。之后登录成功可补传某个离线运行：

```bash
.venv/bin/wandb sync <运行目录/wandb/offline-run-...>
```

`wandb sync` 的上传与训练 Checkpoint 恢复是两个独立操作。离线 run 分段可按相同 group 汇总，横轴继续使用每段记录的实际 `optimizer_step`。

官方接口依据：[W&B init / resume](https://docs.wandb.ai/ref/python/experiments/init/)、[Run.log 与自定义横轴](https://docs.wandb.ai/ref/python/experiments/run/)、[固定 0.30.0 的 Settings 源码](https://github.com/wandb/wandb/blob/v0.30.0/wandb/sdk/wandb_settings.py)。这些文档不代替本项目的远程 SDK 和训练验收。
