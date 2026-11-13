# 本次 AutoDL 实例部署

本文为 2026-10 的历史实例记录，不代表当前容量、价格或进度；当前项目结果见[实验汇总](experiment-results.md)。下面 SSH 示例须替换成自己的连接信息。部署打包器不遵循 `.gitignore`，本地包只用于受控部署，不可直接作为公共发行包。

实例初始信息来自用户在 2026-10-23 提交的终端检查：Python 3.10.8、驱动 570.124.04、驱动报告 CUDA 12.8、可见显存 49,140 MiB。当日已完成源码同步、CUDA/BF16 探针和旧 1000 步预训练，详细证据见 [验证记录](verification.md)。2026-10-24 扩容并重启后，文件系统实际容量已确认为 **70.0 GiB**；当前按 [1B 数据方案](data-1b.md) 准备简体中文与英文语料，新训练尚未启动。

| 项目 | 用户提供的配置 |
| --- | --- |
| 镜像 | Miniconda conda3，Python 3.10，Ubuntu 22.04 |
| 系统 CUDA Toolkit | 11.8 |
| GPU | vGPU 48 GB × 1；底层商品型号与算力份额未单独确认，BF16 探针已通过 |
| CPU / RAM | 20 vCPU，Xeon 8470Q，90 GB 内存 |
| 系统盘 / 数据盘 | 30 GB / 免费 50 GB + 付费 20 GB；重启后实际 70.0 GiB，可用空间以 `df` 为准 |
| 按量计费 | **3.03 元/小时**，使用该实例本次实际价格 |

本机源项目是 `E:\codex\model_training`。远程建议上传到 **`/root/autodl-tmp/model_training`**。AutoDL 将 `/root/autodl-tmp` 作为数据盘挂载目录；数据盘内容不随系统镜像保存，重要产物需要单独备份。[AutoDL 官方目录与存储说明](https://api.autodl.com/docs/env/)

## 1. 先只读检查实例

通过平台提供的 SSH 或终端进入实例后，先执行以下只读命令；不要先重装环境或启动下载：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python --version
command -v python
python -c 'import sys; print(sys.executable); import venv, ensurepip; print("ensurepip:", ensurepip.version())'
df -h / /root/autodl-tmp
```

需要确认：真实 GPU 型号、驱动版本、可见显存、Python 确实为 3.10、数据盘存在且有空间，以及 Miniconda Python 能导入 `venv` 和 `ensurepip`。平台的“vGPU 48 GB”标签没有给出独享计算份额，也不能单独证明 BF16 可用或训练速度。

本镜像没有承诺安装 Python 3.11，后续使用 **`PYTHON_BIN=python`**。若 `ensurepip` 缺失，当前脚本的 `python -m venv` 可能失败；先采用数据盘上的独立 Conda 环境或其他明确支持的环境创建方式，不要为绕过失败把依赖装进系统 Python，也不要重复覆盖半成品环境。环境创建方案和最终解释器路径需要记录。

## 2. 理解三个 CUDA 组件

- **NVIDIA 驱动**：由平台主机提供，决定 GPU 访问和运行库兼容范围。先读 `nvidia-smi`，不在项目部署中升级驱动。
- **系统 CUDA Toolkit 11.8**：镜像中的 `nvcc` 等开发工具。本项目使用预编译 wheel，不需要因此删除或升级它。
- **PyTorch wheel 的 CUDA 运行库**：当前标准脚本安装 `torch==2.9.0` 的 CUDA 12.8 wheel。它和系统 Toolkit 11.8 的版本含义不同；是否可运行还取决于宿主驱动和真实 GPU 工作。

因此，“镜像 CUDA 11.8”既不是需要重装系统的理由，也不能用来直接断言 CUDA 12.8 wheel 一定可用。驱动兼容性不满足时，先选择兼容的 wheel 或平台环境，并作为新的环境配置记录；不静默修改系统 Toolkit。固定 PyTorch wheel 的来源见 [PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)。

## 3. 上传项目并固定数据盘路径

日常开发已采用[本地保存后实时同步](source-sync.md)，源码主版本在 E:，无需每次手动打包上传。下面保留首次部署或手工恢复所用的源码包流程。

源码包位于本机 `E:\codex\model_training\dist\model_training_autodl.tar.gz`，由 `scripts/package_project.py` 生成，包含源码、配置、文档、测试和必要 fixture；不包含本机环境、缓存、训练产物或 `tmp/ssh` 连接文件。每次改动源码后重新打包。远程依赖重新安装。

用户可以在 VS Code 已连接的远程资源管理器中打开 `/root/autodl-tmp`，将这个压缩包拖入后上传；也可在本机 PowerShell 中使用已准备的 E: 盘 SSH 配置上传：

```powershell
$trainingSshPort = 22
$trainingSshHost = 'your-instance.example.com'
scp -F E:/ssh/config -P $trainingSshPort -o UserKnownHostsFile=E:/codex/model_training/tmp/ssh/known_hosts E:/codex/model_training/dist/model_training_autodl.tar.gz "root@${trainingSshHost}:/root/autodl-tmp/"
```

密码仅输入 SSH/VS Code 的登录提示，不写进命令、源码或聊天。上传完成后，在远程终端解压至项目目录：

```bash
cd /root/autodl-tmp
mkdir -p model_training
tar -xzf model_training_autodl.tar.gz -C model_training
```

固定数据盘路径：

```bash
cd /root/autodl-tmp/model_training
export LLM_LAB_ROOT=/root/autodl-tmp/model_training
source scripts/env_remote.sh
export CONDA_PKGS_DIRS="$LLM_LAB_ROOT/.cache/conda/pkgs"
export CONDA_ENVS_PATH="$LLM_LAB_ROOT/.conda/envs"
```

`env_remote.sh` 将 Hugging Face、pip、PyTorch、XDG、W&B 缓存/配置/凭据和 `TEMP`/`TMP`/`TMPDIR` 放在项目内。正式模板已启用 W&B，使用前按 [W&B 指南](wandb.md) 远程登录。若需要创建 Conda 环境，上述 Conda 变量也将包缓存和环境放进数据盘。每次新终端先重新加载环境；不要默认在 `/root/miniconda3` 或系统 `/tmp` 中安装大型依赖或保存训练数据。

当前标准安装把虚拟环境放在项目 `.venv`；它不复用镜像预装的 PyTorch，会重新下载 CUDA wheel 与依赖。因此实际空间需求包含环境文件、下载缓存和解压临时文件。

## 4. 环境安装与真实 GPU 探针

先确认第 1 步信息满足安装前提，再运行：

```bash
PYTHON_BIN=python bash scripts/setup_remote.sh --install
source .venv/bin/activate
python -m pip check
LLM_LAB_REQUIRE_PEFT_TEST=1 python -m pytest tests/test_posttrain.py -q --basetemp tmp/pytest-posttrain
python -m llm_lab.runtime.probe --device cuda --require-cuda --output runs/autodl_environment.json
cat runs/autodl_environment.json
```

`--install` 才会安装；不带该参数只检查现有环境。安装可能下载数 GB 并占用计费时间。现有 `.venv` 存在时，脚本默认优先检查它，不能仅靠激活其他 Conda 环境就断定检查对象已改变。

上述测试包含无联网、无预训练权重下载的微型 Qwen2/PEFT 恢复回归。`LLM_LAB_REQUIRE_PEFT_TEST=1` 将依赖缺失或版本不符视为失败，防止本机跳过的测试在远程仍被静默跳过。

探针读取 PyTorch CUDA 运行库、CUDA 可用性、真实 GPU 型号、总显存、计算能力和 BF16 标记，并执行经过 CUDA 同步的小矩阵乘法。查看 `runs/autodl_environment.json` 中的 `gpu_test.status` 和 `gpus[*].bf16_supported`。

**探针在不支持 BF16 时会使用 FP32 测矩阵；探针通过不等于默认 BF16 训练配置可运行。** 当前预训练和后训练模板都要求 BF16；不支持时必须明确选择其他精度、保留新配置并重新测试。小矩阵通过仍不能代替真实模型的显存、吞吐或恢复验证。

## 5. 数据盘的空间控制

开始获取完整数据或模型前先查看实际用量：

```bash
df -h /root/autodl-tmp
du -sh .venv .cache runs artifacts tmp 2>/dev/null
python -m pip cache info
```

初始免费容量为 50 GB，现已扩至实际 70.0 GiB。容量不能全部分配给训练数据；本轮制备保留至少 8 GiB 可用空间。还需预留依赖解压、日志、Checkpoint 原子写入临时副本和最终导出空间。含 Optimizer 的全参数 Checkpoint 会明显大于单份模型权重；多个运行的 `last`、`best`、Trainer Checkpoint 和 `final` 会累计占用磁盘。

优先使用配置中的有界数据获取和 Token 预算，控制保留的 Trainer Checkpoint 数量，避免复制整个 Hugging Face 缓存、重复下载 Base、保存过多全参数运行。需要降低空间时先列出本项目目录实际用量，确认恢复和报告所需文件，再决定归档或清理；不自动清理用户数据、平台目录或其他项目。不能假定整个 Wikipedia 快照和所有模型版本都适合放入这块盘。

AutoDL 官方说明数据盘通常没有冗余副本；保存系统镜像也不包含该数据盘。选定的配置、manifest、账本、Checkpoint 和报告应另行备份到本机 E: 或明确选择的持久存储，不能把“已保存镜像”当作训练数据已备份。[AutoDL 官方存储说明](https://api.autodl.com/docs/env/)

## 6. 探针通过后再准备数据与短程训练

按 [数据指南](data-guide.md) 分阶段获取与制备 Wikipedia 和 MASSIVE，先检查配置上限、网络可达性和剩余空间；不默认下载全量数据。模型 revision 和数据 revision 保持固定。当前 Wikipedia 方案先完整缓存两份已固定 SHA256 的首分片，共约 1 GB，再从本地 Parquet 流式读取，输出 JSONL 最多 100 MB、扫描文档最多 20000 篇；下载量和输出上限不同。

本实例优先使用官方 Hub 加平台学术加速。已查明默认 Miniconda CA 集合缺少平台已安装的根证书，加载加速后曾出现 SSL 错误；系统 `/etc/ssl/certs/ca-certificates.crt` 已含该证书。使用系统 CA 后，两个官方分片均已通过大小与 SHA256 校验，语料与 32K Tokenizer 已制备完成。

只在数据获取子 shell 中加载加速和现有系统 CA，保留原有 40 位 revision、分片 SHA256 与 TLS 校验；不使用 `verify=False`，不安装新 CA：

```bash
(
  source /etc/network_turbo
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
  HF_ENDPOINT=https://huggingface.co python -m llm_lab.data.acquire --config configs/data/acquire_wikipedia.json
)
python -m llm_lab.data.prepare --config configs/data/prepare_wikipedia.json
```

加速代理仅作用于获取子 shell，避免改变后续 W&B 连接的网络环境。镜像 `HF_ENDPOINT=https://hf-mirror.com` 仍可作为备选，本实例实测较慢。完整缓存并校验两份 Parquet 后，获取过程只读取本地文件，不再依赖外网。[AutoDL 官方加速说明](https://www.autodl.com/docs/network_turbo/)

以下短测协议已在本实例通过：第 10 步保存，重启后恢复至第 20 步。正式 1000 步基线已于 2026-10-23 20:27（北京时间）在独立目录启动；这不表示正式训练已完成。结果与状态快照见 [验收记录](verification.md#2026-10-23-真实数据与-gpu-预训练验收)。新实例仍需先确认制备 manifest 及中英各分区、BF16、依赖和 W&B 登录，再使用独立配置 **`configs/pretrain/100m_probe.json`**：

| 短测条件 | 固定值 |
| --- | --- |
| 总调度 | 20 个 optimizer steps，warmup 2 步 |
| 上下文 / micro batch / 累积 | 1024 / 2 / 8 |
| 精度 / Activation Checkpointing | BF16 / 开启 |
| 监控 | TensorBoard 和在线 W&B 开启 |
| 时间 / 本次 run 费用上限 | 每段进程 600 秒 / 跨恢复累计 2 元；实际小时价 3.03 元 |
| 输出目录 | `runs/pretrain_100m_probe20` |

第一段用训练引擎现有的 `stop_after_steps` 在第 10 个完整更新后退出并保存；保持配置中的总调度为 20，不能用 `--max-steps 10` 替代：

```bash
python - <<'PY'
from llm_lab.config import load_config
from llm_lab.engine.pretrain import run_pretrain

config = load_config("configs/pretrain/100m_probe.json")
run_pretrain(config, stop_after_steps=10)
PY
```

确认 `last.pt` 已保存、第一段进程正常结束后，在相同代码、配置、数据与依赖下启动新的进程，恢复至第 20 步：

```bash
python -m llm_lab.cli pretrain --config configs/pretrain/100m_probe.json --resume runs/pretrain_100m_probe20/last.pt
```

这次操作用于检查 GPU Checkpoint 保存后能否重新加载并继续执行；没有与连续执行 20 步的 GPU 对照运行，不能宣称已验证数值相同。预算先到时可能达不到第 10 或第 20 步，必须按实际步数和停止原因记录。W&B 可视化与本地日志保留实际 optimizer step 和恢复来源。

短测不是充分预训练。短测的 20 步调度与 `100m.json` 的 1000 步调度不同，**不能从这个 Checkpoint 直接恢复到正式配置**；本次正式训练使用独立输出目录并重新初始化。后续实验仍需根据实际吞吐、峰值显存、验证与保存开销确定规模和预算。

后训练的 `--validate-only` 不下载模型；真实训练前在本次运行配置的 `budget.hourly_rate_cny` 填 **3.03**，并设置运行限额。全参数 SFT、LoRA 和 DPO 按既定依赖顺序执行，不因为显存标签较大就跳过校验或同时启动全部实验。

## 7. TensorBoard 使用实际可达入口

项目 TensorBoard 监听本机回环地址：

```bash
tensorboard --logdir runs/pretrain_100m_probe20/tensorboard --host 127.0.0.1 --port 6006
```

先检查 6006 是否已被镜像内其他服务占用；不要终止不属于本项目的进程。端口有冲突时选择另一端口并同步调整访问方式。

优先在本机通过实际 SSH 连接建立 `-L 6006:127.0.0.1:6006` 转发，再访问本机 `http://127.0.0.1:6006`。SSH 主机和端口使用实例真实连接信息，不在文档记录密码或 Token。

AutoDL 官方文档提供 6006/6008 的自定义服务入口和 SSH 隧道方式；入口权限与实际代理网络路径需要在该实例上核验。监听 `127.0.0.1` 的服务不保证被平台代理直接访问，不能将控制台显示入口等同于连接已通。不要为了省去核验默认改为公网监听。[AutoDL 官方端口说明](https://www.autodl.com/docs/port/)

## 8. 费用与结束检查

本次实例按用户提供的 **3.03 元/小时**核算；不是手册中的示例报价。总预算 **500 元**包含 GPU、存储、传输等，默认 **100 元 reserve**用于计算外开销与重跑，不构成平台额外赠送的算力，也不保证足够完成全部实验。

训练进程的共享账本只估算可见运行成本，环境安装、下载、等待以及实例空闲时间同样可能产生实际费用。程序到预算退出不等于实例停止，停止计算也不代表扩容存储不会继续收费；以控制台实际账单为准。

结束前检查：日志与 Checkpoint 完整、选定产物已备份、共享账本已保存、没有本项目遗留进程，再在平台执行实际需要的关机或释放操作。本项目不自动关机、释放实例或删除数据。关于额外付费存储的计费边界，以 [AutoDL 官方计费说明](https://www.autodl.com/docs/price/) 和该实例控制台为准。

部署验收需要填写：远程实际型号/驱动/Python、探针结果、安装依赖快照、数据清单、短程吞吐与显存、实际时长和费用。完成之前均标记为待验证。
