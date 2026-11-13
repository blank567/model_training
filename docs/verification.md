# 本次实现与验证记录

2026-10-25 补充：用户授权停止并清空旧 300M 输出后，已执行三组串行、各 24 步的 GPU 短测（总 159.29 秒），配置保护与测速 CPU 检查 43 passed；测试权重已全部删除。最快安全方案是 8×4、关闭重算，约 31,674 Token/s、peak reserved 20.72 GiB；已冻结新实验 `20261025_150858`，没有自动启动正式长跑。旧源码身份、100M 与正式语料保留，详情见 [优化后的运行说明](run-300m-optimized.md)。

最新状态：2026-10-25，100M 已完成 **61,036 步、1,000,013,824 个处理 Token**；扩容后新训练池 **3,100,003,021 Token** 已完成独立发布验收。300M 的模板、独立启动器和本次冻结配置 `20261025_140717` 已准备，本地启动器测试 13 passed/1 skipped，远程 CPU **14 passed**；资源检查和严格配置校验通过，尚未执行 300M GPU 短测或正式训练。旧源码指纹与 100M checkpoint 状态保持相同，GPU 命令由用户执行，见 [300M 运行说明](run-300m-3b.md)。下文按时间保留历史记录。

日期：2026-10-22。项目在 `E:\codex\model_training` 全新搭建，没有以旧项目源码为基础。

## 已完成的本地验证

本次使用 `E:\conda_envs\llm\python.exe`，Python 3.10.19、PyTorch 2.10.0+cpu、NumPy 1.26.4、Tokenizers 0.22.2。没有升级或替换本机环境。临时文件与测试产物均位于 E:。

初次实现的全套测试结果：**70 passed in 7.65s**。2026-10-23 部署前复查增加 DPO adapter 恢复回归后，结果为 **73 passed, 1 skipped in 23.32s**；本机 Transformers 为 5.1.0，且没有目标 PEFT 依赖，真实 tiny Qwen2/PEFT CPU 回归明确跳过。远程安装固定依赖后使用 `LLM_LAB_REQUIRE_PEFT_TEST=1` 强制执行该项。

```powershell
$env:TEMP='E:\codex\tmp'
$env:TMP='E:\codex\tmp'
$env:PYTHONPATH='E:\codex\model_training\src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
& 'E:\conda_envs\llm\python.exe' -m pytest -q --basetemp artifacts/test-tmp/all-posttrain-final
```

覆盖范围：

- 因果注意力隔离未来信息、next-token 目标错位与交叉熵。
- 梯度累积与完整 Batch 的梯度一致；Activation Checkpointing 在 dropout 下保持梯度一致。
- 含 dropout、验证和重计算的微型训练中，连续执行与断点恢复的模型、优化器、调度器、采样器状态一致。
- 真实 CPU GradScaler 溢出回退；FP32 非有限梯度拒绝更新。
- 配置错误、独立测试集误用、跨语言源 ID 泄漏、数据和恢复身份变化的拒绝行为。
- 有界获取、文档去重、train-only Tokenizer、校验和、归档路径/链接/大小限制。
- SFT completion-only mask、EOS、完整样本 Token 配额和中英近似均衡。
- JSON 严格结构、重复槽位、意图/槽位指标和中英等权汇总。
- CUDA 不可用时不把 CPU 探针伪装成 GPU 通过。

约 100M 默认模型通过 meta device 统计为 **100,092,672 参数**，未为此在本机分配完整模型或运行正式训练。

## 集成流程

实际运行三条 fixture 数据命令：

```text
python -m llm_lab.data.acquire --config configs/data/acquire_smoke.json
python -m llm_lab.data.prepare --config configs/data/prepare_smoke.json
python -m llm_lab.data.prepare --config configs/data/prepare_massive_smoke.json
```

产物位于 `examples/_outputs/`，全部来自仓库内原创小样例。预训练 fixture 的实际词表为 512，六个 train/validation/test × en/zh Token 流均非空。MASSIVE 形状的 fixture 只验证接口，不冒充实际 MASSIVE 数据。

随后用 `configs/pretrain/smoke.json` 的 **34,976 参数微型模型**做 4 个 CPU optimizer update，共 128 个有效 Token，验证保存、重开、validation 和生成 CLI 的连接。产物位于 `tmp/integration_pretrain/`；这些数值不是 100M 模型的训练结果或性能基准。

后训练 `--validate-only` 已对 fixture 实际执行，不加载模型、不创建训练输出。离线评分 CLI 用金标复制成预测验证指标连接，得到满分只是评分器自检，不是模型准确率。实验计划 CLI 已生成 5 个等 Token 试验配置，未执行这些 GPU 试验。

两份远程 Bash 脚本均通过 Git Bash `-n` 语法检查，文件为 LF。

## 远程尚需验证

以下是 2026-10-22 至 10-23 部署前的待办快照；之后完成的项目见后续带日期的记录，不用这份历史清单覆盖新验收：

1. 按 `requirements/remote.txt` 安装后 `pip check`，保存完整 `pip-freeze.txt`；直接依赖版本表不是全量传递依赖锁文件。
2. 已租用的 AutoDL vGPU-48GB、PyTorch CUDA、BF16 和目标模型的前向/反向工作。用户提供的终端初检显示驱动 570.124.04、驱动报告 CUDA 12.8、Python 3.10.8 和约 50G 可用数据盘；这不代替 PyTorch 实际运算。
3. 固定版本 Wikipedia 与 MASSIVE 的真实下载、完整解析与数据审计；词表不足时补充语料并重新冻结配置。
4. Transformers 4.57.1 / PEFT 0.17.1 / TRL 0.24.0 / Accelerate 1.11.0 / Datasets 4.3.0 的完整训练与恢复集成。
5. 同环境 GPU 上连续训练与恢复训练的数值容差验收。
6. 显存峰值、吞吐、真实费用、验证集选模与最终独立测试。

截至 2026-10-23，用户已租用 AutoDL 实例并提交基础检查，尚未完成本项目的远程安装或真实 GPU 运行。本次没有下载正式语料或模型权重，也没有执行正式预训练或后训练。后续依据短程测速确定训练规模，不从 CPU 测试外推 GPU 性能或任务效果。

## 固定接口依据

后训练依赖范围核对 [TRL v0.24.0 pyproject](https://github.com/huggingface/trl/blob/v0.24.0/pyproject.toml)，DPO 预编码和 reference adapter 语义核对[对应文档](https://huggingface.co/docs/trl/v0.24.0/dpo_trainer)及[对应源码](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/dpo_trainer.py)。这属于接口核验，不替代目标环境集成测试。

Base 固定为 [Qwen2.5-0.5B 的完整 commit](https://huggingface.co/Qwen/Qwen2.5-0.5B/commit/060db6499f32faf8b98477b0a26969ef7d8b9987)。数据和许可证记录见 [data-guide.md](data-guide.md)。

## 2026-10-23 源码实时同步验收

公钥 SSH 已连接，远程非交互解释器为 `/root/miniconda3/bin/python`（3.10.8）。本机后台同步器使用标准库轮询和 OpenSSH，不额外安装监控依赖。全套本地结果为 **99 passed, 1 skipped in 8.27s**，其中新增 26 项源码同步行为测试。

覆盖白名单和凭据排除、首次源码包 manifest 接续、远程独立修改拒绝、路径/软链接/哈希校验、删除/重命名归档、原子替换故障后的保留与重试、断线后发送最新快照、保存合并、暂停和单实例锁；同时覆盖 AutoDL 容器中无关非 Python 进程元信息不可读的情况。

实际 AutoDL 验收：后台新增、修改、重命名、删除均通过，观测延迟约 2.3–2.9 秒。一个真实的 8 秒项目 Python 等待进程成功阻止同步，退出后自动补齐；验收未运行 GPU 训练。报告位于本机 `tmp/sync/integration_report.json`、`tmp/sync/safety_report.json`。接收端保存 manifest 并只归档已管理的旧源码，现有远程 `.venv` 保留。

同步器已作为隐藏窗口的后台进程运行。Windows 退出登录或重启后需要重新 `Start`；没有设置系统开机任务。后续训练环境的 pip check、真实 tiny PEFT 测试、CUDA/BF16 探针和目标模型短程训练仍需要单独验收。

## 2026-10-23 W&B 安装与在线验收

远程 `.venv` 已安装 `wandb==0.30.0` 及解析出的依赖，`pip check` 返回 `No broken requirements found`。实例直连官方 PyPI 的初次元数据请求超时，后使用实例已有的阿里云 pip 源完成安装；实际包 URL、版本和校验信息记录在 `runs/wandb_install.json`，日志为 `runs/wandb_install.log`，依赖快照为 `runs/pip-freeze-after-wandb.txt`。未为此改动 PyTorch/CUDA wheel 或系统 Toolkit。

真实 SDK 离线写入自检通过；用户在远程终端登录后，在线自检运行 `9852d3e2792c` 正常结束，确认上传 3 条合成指标和 SDK 元信息。[在线自检面板](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/9852d3e2792c)。这些点不是模型 Loss，不属于正式训练结果。

本机测试 **112 passed, 1 skipped in 10.56s**；随后在远程固定依赖中强制开启真实 PEFT 测试，使用 `CUDA_VISIBLE_DEVICES=""` 执行完整 CPU 回归，结果 **113 passed in 4.32s**。远程日志为 `runs/wandb_remote_tests.log`。此前跳过的 tiny Qwen2/PEFT adapter 恢复已经实际通过；完整 SFT/DPO 训练、真实 GPU 吞吐和恢复数值容差仍未以这些 CPU 测试替代。

W&B 已接入预训练日志与 SFT/LoRA/DPO 的 Trainer Telemetry，使用自定义 optimizer step 横轴、在线稳定 run id、离线分段 group；SDK 失败时保留本地 JSONL。新增测试覆盖同一步多事件、身份不符拒绝、SDK RNG 隔离和监控开启时的预训练恢复一致性。使用方式见 [W&B 指南](wandb.md)。

## 2026-10-23 真实数据与 GPU 预训练验收

本次使用已租用的单张 NVIDIA vGPU-48GB，计算能力 8.9，驱动 570.124.04，项目 Python 3.10.8、PyTorch 2.9.0+cu128、W&B 0.30.0。实际 BF16 小矩阵测试及完整 100,092,672 参数模型的前向、反向、梯度裁剪和 AdamW 更新均通过；未更改系统 Toolkit。

### 数据与正确性

官方 Hub 直连不可达；镜像可访问但元数据请求超时、分片吞吐较慢。AutoDL 已提供的系统 CA 可以校验平台学术加速证书，使用其官方代理取得固定 revision 的下载地址，再通过 HTTPS 下载并校验两个完整首分片，共 1,007,405,476 字节，实际下载约 128.69 秒。没有禁用 TLS 校验，也没有安装新 CA。可选显式分片流程已接入 `acquire.py`，避免全仓元数据发现；下载命中缓存时复验大小与 SHA256。

原始 JSONL 共 5,610 篇、99,981,818 字节；清洗扫描 5,610 篇，接受 5,564 篇，长度规则丢弃 46 篇。仅用 train 文档拟合 32,000 词表 BPE。六条流均通过哈希、长度与词表校验，test 只做完整性检查，没有运行模型测试评测。

| 分区 | 英文 Token | 中文 Token |
| --- | ---: | ---: |
| train | 10,974,692 | 11,725,725 |
| validation | 61,714 | 87,174 |
| test | 152,406 | 167,218 |

本机完整回归 **134 passed, 1 skipped in 10.54s**；跳过项需要远程固定 Transformers/PEFT 版本。远程禁用 CUDA 的完整 CPU 回归 **135 passed in 4.28s**，包括真实 tiny PEFT 测试。新增分片测试覆盖固定路径、大小与下载预算、缓存校验和原子完成。CPU 回归与下面的真实 GPU 训练分别记录。

### 独立 20 步短测

配置为 `configs/pretrain/100m_probe.json`：BF16、1024 上下文、micro batch 2、梯度累积 8、确定性与 Activation Checkpointing 开启。总调度 20 步、warmup 2；新进程在第 10 步保存后退出，另一新进程加载同配置 Checkpoint 并恢复到第 20 步。

- 实际完成 327,680 个训练 Token，中英各半，20 条日志的 optimizer step 连续且全部 Loss 有限。
- Train Loss 从 10.5113 到 8.4352；第 20 步 validation 英文 8.2582、中文 8.6608、均值 8.4595。每种语言仅评估 2,048 Token，这是短测窗口，不能视作最终质量结论。
- `torch.cuda.max_memory_allocated` 峰值 2,649,708,032 字节，约 2.47 GiB；该指标不包含驱动和全部保留显存。
- 首段第 2–9 步的纯训练窗口约 28,285 Token/s；整段包含初始化、W&B、验证与 Checkpoint 的端到端吞吐更低，不用单个窗口保证正式速度。
- 训练预算账本计入约 40.15 秒、0.0338 元；安装、下载、等待、实例空闲及部分初始化不在此计时内，实际费用以平台为准。
- [短测 W&B](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/e788aef39d73) 在恢复后仍使用同一个在线 run。

本次验证了 GPU 恢复可以继续执行；没有运行连续 20 步的 GPU 对照，因此不宣称已验证连续与恢复的数值等价。两个短测 Checkpoint 保留在远程 `runs/pretrain_100m_probe20/`；其 20 步调度不能直接续接正式 1000 步配置。

### 正式基线启动快照

2026-10-23 20:27（北京时间）使用原始 `configs/pretrain/100m.json` 从随机参数启动，目标 1000 步、16,384,000 个训练 Token，中英文各 8,192,000。W&B 为 [正式预训练 ca5a7695a5b7](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/ca5a7695a5b7)，输出为远程 `runs/pretrain_100m/`；后台监督状态和日志为 `runs/pretrain_100m_job/`。

首次验收快照已到第 50 步，Train Loss 8.1441、累计吞吐约 17,230 Token/s。此为运行中快照，正式训练尚未完成，也未进行最终测试。软时间上限 1800 秒，外层 2100 秒超时保护；退出训练不会自动关闭实例。程序在验证集上选择 best，独立 test 保留到最终报告。

本机非敏感验收报告为 `tmp/pretrain/verification_report.json`，包括数据清单指纹、运行状态、环境、短测结果和正式 W&B 链接。本地与远程训练源码指纹、正式运行记录中的源码和数据指纹已核对一致。运行期间源码同步按保护规则暂缓，结束后会补齐待同步文档。

## 2026-10-24 基线完成与扩展前验收

只读查询 AutoDL 后确认，`runs/pretrain_100m/` 与用户手动运行的 `runs/pretrain_manual_20261023_204911/` 均已完成 1000 步，各累计处理 16,384,000 Token；best validation loss 均为 5.978839457。训练预算计时分别约 666.09 秒与 689.73 秒，记录费用约 0.5606 元与 0.5805 元。这些不是包含实例空闲和数据准备的真实账单。查询时 GPU 为 0% 利用率、1MiB 显存，不能用历史运行快照描述当前状态。

这两轮仍归类为 100M 的流程与初步学习曲线基线，没有完成充分预训练，也没有据此进行最终测试。扩参数前的计划见 [scaling-plan.md](scaling-plan.md)。

新增 `configs/experiments/pretrain_throughput_48gb.json`，准备四项 32 步等 Token 对照：micro/accum 为 2×8、4×4、8×2，以及 8×2 关闭激活检查点。每次更新均为 16,384 Token，每次验证每语言也均为 16,384 Token。单项 524,288 Token；使用新输出目录，保持旧实验。

本机已执行标准库配置生成与严格 schema 校验，报告为 `tmp/scaling_preflight/validation_report.json`；未实例化训练模型、未启动 GPU 试验，当时也没有生成可执行长跑配置。吞吐提升、峰值显存和更大模型质量仍需由用户运行实测。

## 2026-10-24 1B 数据准备与验收边界

本轮确认的训练目标是 100,092,672 参数模型从随机参数累计处理 1B 有效 Token，中英文采样各占 50%，中文统一简体；每语言独立 train 语料需至少 500M Token，制备上限各 550M。后续约 300M 是独立实验，结构、训练 Token、显存与费用须重新验收，不从本轮 Checkpoint 扩容续训。约 20 Token/参数只保留为理论参考，不是本轮训练目标。

远程固定依赖的完整 CPU 回归快照为 **182 passed in 5.02s，无 skip**，覆盖真实 PyArrow 与官方 C++ OpenCC 绑定。小 fixture 的中文简体转换、旧中文 holdout 原始 Token 身份复算与转换后重建已通过，英文 holdout 保持字节一致。这些是测试与小样例证据，不是正式语料容量、模型能力或 GPU 长跑结果；后续代码变更仍需单独保留对应回归记录。

首轮正式数据制备 `runs/data_1b_prepare_20261024T060038Z` 在 **6.74 秒**发现真实文本的单次 OpenCC t2s 非幂等，按规则失败；当时尚未编码、未发布 manifest，失败输出已保留归档。随后增加有界固定点转换、循环检测和 provenance v2，并纳入真实 `浮於潛` 回归样例；远程针对性回归为 **32 passed in 0.65s**。此前的 182 项完整回归是修复前快照，不冒充修复后的全套验收。

有界固定点修复后，曾在远程 `runs/data_1b_prepare_20261024T060332Z` 重开正式制备，使用 `configs/data/prepare_wikipedia_1b.json`；该任务后来受控中断并归档。阶段进度、已下载 raw 或部分 bin 当时未算作完成数据；最终成功运行与 manifest 见末尾发布记录，新的 GPU 训练没有启动。

中文采用固定官方 `OpenCC==1.4.2` 的 C++ 绑定，以有界固定点 t2s 达到稳定规范，最多 8 次转换，循环或不收敛则失败；配置、递归词典、native binary 和算法身份写入数据 manifest。旧原文保持不变，新的中文 train/validation/test 采用相同规则，原始与转换后 hash 都纳入隔离。

Tokenizer 复用初版原始 train 拟合的 32,000 BPE，并核对原 SHA256。本轮不重新拟合，但不能据此宣称新转换后的 holdout 文本从未参与旧 BPE 拟合：转换可能合并原来不同的历史 train/holdout 文本。原始分区的 train-only 拟合证据与新规范形态的隔离证据分别报告。中文评测已改变，新旧 Loss 不能直接横向比较。

当前实例数据盘容量为 **70 GiB**；获取与制备均继续保留 **8 GiB** 可用空间水位，实际写入以实时磁盘检查为准。500 元总预算和 100 元余量保持不变。按旧 1000 步速度外推，1B 训练进程约 34–36 元，仅为旧短跑估算，未包含完整实例账单；300M 必须重测。正式数据现已发布，新 GPU 长跑仍须先测速和冻结运行配置，再由用户自行启动。

更多协议与入口见 [data-1b.md](data-1b.md)、[scaling-plan.md](scaling-plan.md)。本次文档更新没有启动下载、制备或 GPU 训练，也没有新增测试或修改训练源码。

## 2026-10-24 CPU 优化验收与制备历史

本地与远程同步源码 SHA 已核对一致。当前版本完整回归结果为：

| 环境 | 完整回归 | 可支持的结论 |
| --- | --- | --- |
| 远程固定依赖、CPU 测试 | **215 passed in 9.71s，无 skip** | 真实 PyArrow、官方 OpenCC、固定 Transformers/PEFT adapter 等受测路径已执行 |
| 本地开发环境 | **212 passed, 3 skipped in 30.05s** | 本地缺 PyArrow/真实 OpenCC；Transformers 非固定 4.57.1，真实 adapter 项跳过，其余受测逻辑通过 |

两个环境的结果不互相替代，也不覆盖上文 70、182、32 等历史测试快照。完整 CPU 测试不证明正式语料容量、GPU 长跑吞吐或最终模型能力。

类别过滤优化只替换 `common.normalize_text` 的逐字符类别判断，使用当前 UCD 惰性生成的允许范围否定正则；全部 Unicode 码点（含 surrogate）、整串与混合样例均严格等价。所有规范化调用次数、NFC/空白规则、过滤与哈希顺序保持原样；英文两次规范化继续保留。远程中英文各 1,000 篇样本的双次规范化约快 **2.95× / 3.74×**，这是清洗函数计时，未外推为全流水线加速。

独立 Tokenizer 批编码样本约快 **4.62×**，**8,000 次逐文 Token ID 对照全部一致**，逐文 EOS 与整篇 cap 模拟结果一致；不包含清洗、OpenCC、SQLite 与正式输出 I/O。正式编码采用 batch 64、累计输入 4MiB 边界，超阈值单文独占批次，仍按原文档顺序处理配额；4MiB 不是 RSS 上限。数据进程单独设置 `TOKENIZERS_PARALLELISM=true`、`RAYON_NUM_THREADS=8`，未修改全局 `env_remote.sh`。官方 `OpenCC==1.4.2` 的有界固定点 t2s 与 provenance v2 保持不变。

旧任务 `runs/data_1b_prepare_20261024T060332Z` 由主进程受控 SIGINT 停止，约 533k 文档、2.1GB 的部分处理进度保留于 `wikipedia_1b.interrupted_20261024T060332Z`，未发布为完成数据。最终任务 **`runs/data_1b_prepare_20261024T062547Z`，supervisor PID 5452** 已 CPU-only 成功结束；没有启动新的 GPU 训练。

中文抽样外推约 **714.27M Token** 保留为历史估计，不代替实际编码与每语言 500M 门槛；最终产物以以下独立验收为准。原 raw 获取统计不变。运行入口使用项目 `.venv/bin/python`，进程级环境示例见 [data-1b.md](data-1b.md)。

## 2026-10-24 正式数据发布与独立验收

非敏感报告为 `tmp/data_1b/verification_report.json`，状态 `passed`。正式入口为 `artifacts/pretrain/wikipedia_1b/manifest.json`，SHA256 为：

`9ccf48cd0e0d09df001ee50cae62e134e80cd4b770e0ec13c829c8f29b22999a`

| 分区 | en Token | zh Token |
| --- | ---: | ---: |
| train | 549,999,999 | 549,999,984 |
| validation | 61,714 | 87,116 |
| test | 152,406 | 167,086 |

train 合计 **1,099,999,983**，所有分区合计 **1,100,468,305** Token。实际写入 train 的文档为 en **563,366**、zh **895,049**。独立验收 **86/86 passed，38.9358 秒**；六条流的 TokenCorpus CPU 采样均为 `[2, 1024]`，没有运行模型测试集评测。

验收确认完整 raw 来源链、固定 BPE/OpenCC 身份、全部流的大小/SHA/合法 ID、审计/索引等产物指纹；全部受保护中文 holdout 原始/转换 hash 已找齐，保护源与保护 hash 的活跃 train 交集均为零。英文 holdout 字节保持不变，中文旧原始 bin 和新简体 bin 的独立重建均通过。旧 BPE 历史三项交集统计也为零，但只覆盖已保护 hash 与旧原始 train hash 的精确比较；范围与未独立全量重编码 train、语义近重复等限制集中记录在 [data-1b.md 的独立验收范围](data-1b.md#独立验收的范围与限制)。

最终制备耗时 **2,177.46 秒（约 36 分 17 秒）**，按实例时间估费 **1.8327 元**，仅代表本次进程时段，不能当作全部平台账单。产物目录 **12,136,828,541 bytes**，包含 cleaned/audit/index/bin 等；验收采集时磁盘为 **70 GiB 总量、28.717 GiB 可用**，GPU 为 **1MiB / 0%**。

正式数据可用于 100M/1B 的中英 50/50 有放回随机取窗训练，容量不等于无重复覆盖。新 GPU 训练未启动；用户自行启动前仍需独立测速并冻结长跑配置。后续 300M 保持独立实验。本次收尾仅更新文档，没有修改源码、新增长跑配置或运行额外测试。

## 2026-10-24 手动训练配置交接

按用户随后要求，新增 `configs/pretrain/100m_1b.json` 与 [手动运行说明](run-100m-1b.md)。本地 CLI `pretrain --validate-only` 已通过严格配置校验；没有实例化训练模型或启动 GPU 训练。候选使用 micro 8 / accumulation 2、关闭激活重算、BF16 和上下文 1024，总步数 61,036，计划累计 1,000,013,824 Token。每 1000 步验证与保存，启用 W&B；运行累计费用软限额 60 元，每会话最长 18 小时，项目 500 元 / reserve 100 元规则保持。

先以 Python API `stop_after_steps=32` 保持正式学习率计划并保存，再使用同一配置 `--resume`。这组 Batch 的 GPU 稳定性与吞吐尚未实测。原 SSH 17712 端口在交接时拒绝连接，未完成关机后的远程 BinaryCorpus 预检；用户确认原实例已关机且数据盘保留。配置曾由同步器记录同步，重新开机后仍须检查实际文件和当前 SSH 信息，再由用户执行训练。

同日重新开机后的远程预检已完成：Python 3.10.8、PyTorch 2.9.0+cu128、CUDA 与原生 BF16 可用；BinaryCorpus 实际读取并校验正式 manifest，中英 train/validation 四个 CPU Batch 均为 `[8,1024]`。查询时 vGPU 48GB 空闲，没有已有时间戳实验或训练进程，数据盘可用约 28.717 GiB。已在 `runs/configs/` 独占创建 `pretrain_100m_1b_20261024_192116.json` 和同名 `.env`，JSON SHA256 为 `0780684205683180f2780e7cff27521b768c5b51d8c75c54b882671b8f31f270`。真实 source 环境文件后，CLI 配置校验通过；本地非敏感交接记录为 `tmp/data_1b/handoff_20261024_192116.json`。训练输出目录尚未创建，未启动 GPU 训练，新 Batch 的显存和速度仍待用户运行 32 步实测。

## 2026-10-26 自研 300M 的 SFT 数据发布

正式数据为 `artifacts/posttrain/sft_300m_20261026_170531/`：train 19,991、validation 995、test 1,000，pilot 2,000 为 train 子集。训练完整输入 4,392,427 Token，回答监督 3,525,424 Token（含 EOS），中文回答占比约 50.011%，所有样本完整输入均不超过 512 Token。源和处理范围见 [数据构建](sft-data-build.md)。

本地 CPU 专项测试 **64 passed in 2.98s**；远程相同四个测试文件 **64 passed in 0.65s**。覆盖质量拒收回归、同问题/同材料分组、BPE边界、EOS监督、修改holdout后拒绝、跨split泄漏、pilot混入外部样本、原始身份改动、抽审不得针对test和已有发布拒绝覆盖等情况。

真实远程独立验收重编码全部四个分区，核对文件SHA、简体固定点、prompt/回答mask、末尾EOS、词表范围、声明group/精确键隔离、pilot子集及语言配额。本地与远程分别执行抽审发布，得到相同数据身份 `b85ba15fe4ccdb35e796a5e4a1767f07ea918c35ab0d2c85ac2ca61a432cef6a`，所有核心文件SHA一致，原预训练源码树操作前后身份相同。

候选版本的结构验收并未替代语义检查。后续分层LLM抽审发现截断、缺原文摘要、数学错误等问题，已扩大规则、隔离原始行并保留旧版废弃标记。最终源构建版的40条train/validation抽审又保守剔除14条疑点或待专业复核样本，因此正式train/validation略少于目标；test字节未改，不用于内容选择反馈。LLM抽审不是人工专家金标，未验证所有回答的事实正确或完整语义/跨语言去重。

正式manifest、acceptance、quality_review、curation_config、publication_receipt和原始许可归属均已保留。此数据验收只支持已受测的数据与CPU逻辑；SFT模型接入和受限GPU验证见后续记录，尚无正式SFT效果结论。

## 2026-10-26 自研300M SFT入口与受限GPU验证

新增独立 `src/llm_sft`，复用冻结Decoder并保护全部36个 `llm_lab` 源文件，远程预训练源码指纹仍为 `85cdd6e9c4625709931dfc1faa87ad5e1d0e9651738d16cb0d93c25ac8ec406b`。SFT运行指纹为 `145f8c3a1053a9f186404dc5f4ca7a50047dc462aeba61686b559e3456b9390b`。Base预训练已完成91553步、3,000,008,704tokens，SFT统一使用best step91500，SHA `69b24779315561390bddbef18ce5ce9b505052a3ca697a0a651dfa6ea3c2bf45`。

新增59项入口测试远程CPU全部通过：17项数据、7项真实tinyDecoder引擎、20项启动/控制变量计划、15项生成。覆盖回答/EOS监督、一次标签错位、padding、不等长回答Token加权累积、完整尾组、带dropout连续与恢复全状态逐项精确一致、Base权重不变、model-only恢复拒绝以及配置/数据/跨实验身份拒绝。完整本地回归416 passed、5 skipped；跳过项为未安装pyarrow的两项、当前transformers版本不符的一项、远程官方OpenCC一项、Windows不支持fcntl的一项。新增入口59项无跳过。

远程 vGPU-48GB、Torch2.9.0+cu128、BF16 先执行16次更新再保存，由独立进程恢复至17次更新后停止，累计544条样本。两个会话记录73.27秒、估算计算费0.0617元；不计实例闲置等其他费用。峰值allocated显存12.76GiB，剔除验证/保存的前15步间隔中位数0.486秒。回答验证macro Loss从2.5766458降至第16步2.4509776；固定预训练validation小窗口macro Loss由3.2655317升至3.2847193，继续观察退化。两者监督区域不同，不能直接比较数值大小。短测不是完整SFT，也不证明任务准确率或聊天能力改善。

W&B在线上传和相同run id恢复通过；TensorBoard及本地JSONL留存。`chat_sft.py`成功加载step16的模型权重并在CPU以相同模板生成16Token，达到max_new_tokens边界，该检查不评估完整回答或EOS成功率。短测控制脚本首次在同一进程复用launcher时被GPU占用保护拦截（该进程仍持有CUDA上下文）；改用独立进程后恢复通过，正式CLI执行方式本就使用独立进程。首次报告收集脚本的字段名错误已修正，训练断点和正式数据均未受影响。

仅短测 `runs/sft_300m_full_20261026_190101/best.pt` 和 `last.pt` 已清理，冻结配置标记cleared，禁止误续跑；日志、曲线、summary和短测receipt保留。100M与300M预训练、正式数据保持不变。清理后可用约17.97GiB，新单轮检查需17.04GiB。正式 `1e-5/2e-5/5e-5` 三组配置时间戳分别为 `20261026_213513/213514/213515`，固定同Base/数据/种子/B32/一轮625步/warmup19，均只准备未启动。多轮完整last占用与原子保存余量不能保证同时满足，后续逐次检查，不自动删除正式产物。

任务报告：本地与远程 `tmp/sft_entry/entry_ready.json`；CPU结果 `tmp/sft_entry/remote_tests.txt`；启动、恢复和控制变量命令见 [SFT训练手册](sft-training.md)。LoRA、DPO、正式长跑、完整训练时长和最终独立测试尚未验证。

## 2026-10-26 前台中断清理、日志破管修复与参数命名

用户前台启动旧默认实验 `20261026_213514`，Ctrl+C同时影响Python和tee，在第100步附近触发BrokenPipeError，完整last不存在；该次实验的best不是可完整恢复的状态。按用户“删掉这次实验的所有内容”，已删除唯一匹配的输出目录、冻结JSON/env/identity/state、控制台日志及W&B运行 `cb3150d8fcf0`。释放约1.1303GiB，复查余量17.97GiB。10个保护文件的inode、大小、mtime保持不变，包含100M/300M预训练模型、正式SFT数据manifest、其他两组冻结配置和共享预算账本；已发生费用未抹除。删除报告为 `tmp/sft_entry/deletion_20261026_213514.json`。

修复后信号处理器只置停止标志，不在信号回调内输出。控制台BrokenPipe/EPIPE会解除已断开的输出管道，只有stderr也指向同一管道时一同处理；其他I/O错误继续抛出。更新、JSONL与完整last保存不因该控制台异常中断。默认操作文档改为nohup后台直接写日志、tail单独查看；Ctrl+C只退出日志查看，不把tee当作后台工具。

所有新实验统一命名为 `sft_300m_full_lr2e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_<时间戳>` 一类参数名称，输出目录、冻结文件、RUN_LOG、W&B保持一致；训练量对照同时带调度ep与stop量。名称绑定实际配置并核验，旧名称不再允许执行；另外两份旧未执行配置保留并标记superseded。

当前SFT代码指纹 `0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`，原预训练源码指纹未变。本地435 passed / 6 skipped：新增入口78 passed / 1 skipped（实际OS pipe用例仅POSIX，Windows映射EINVAL）；远程Linux入口79 passed / 无跳过，4.24秒。覆盖实际关闭stdout/stderr读端的子进程能保存完整checkpoint并正常退出、日志/停止信号两种位置发生两类破管异常时保存step2并只恢复step3，以及ENOSPC未被错误吞掉。此次控制台与命名改动只做CPU验证，没有新GPU训练。

新三组LR配置时间戳为 `20261026_221316/221317/221318`，均包含完整参数名并只准备未启动；默认2e-5预检通过，所需17.04GiB、实有17.97GiB。报告为 `tmp/sft_entry/entry_ready_named.json`，远程CPU结果为本地 `tmp/sft_entry/remote_tests_params.txt`。旧213514启动命令已失效，请使用训练手册中的新版后台命令。

## 2026-10-26 按实验归属归档助手记录

用户要求其手动结果保留runs根目录，助手代跑的记录分入子目录；若改变路径会影响运行则保持原位。盘点时无训练进程，归档前扫描源码、配置、冻结配置、发布数据元信息以及进程引用。仅将21项独立的助手历史记录移入 `runs/assistant/benchmarks`、`inference`、`data_prepare`、`system`，预留 `smoke` 用于后续短测。

277个归档文件在移动前后逐字节SHA256一致；505个保护文件的inode、大小、mtime不变。训练代码指纹保持 `0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`。用户正式模型、全部权重和语料、冻结配置、共享账本和GPU锁未改变，未启动训练，未删除文件。保留被引用的旧probe、三个SFT数据制备目录、旧SFT状态目录和含恢复身份的早期模型原址。

旧→新映射和逐文件摘要为远程 `runs/assistant/organization_manifest.json`；根目录索引为 `runs/README.md`；本地结果为 `tmp/sft_entry/runs_organization_result.json`。历史报告中的旧路径保持原样，按映射查当前位置，不批量改动哈希绑定JSON。当前操作规范见 [runs目录说明](runs-layout.md)。

## 2026-10-27 第一组学习率正式启动

用户授权启动三组学习率实验后，预检发现70GiB数据盘仅余17.9678GiB。保留每组best+完整last约4.5203GiB，加新组17.0406GiB预检余量，三组顺序全部保留需初始至少26.0811GiB空闲（另计日志）。用户明确选择“先跑第一组，其余两组暂停”，因此仅后台启动 `1e-5`，没有启用自动队列，也没有删除权重或降低磁盘水位。

输出保留在根目录 `runs/sft_300m_full_lr1e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261026_221316`，PID启动快照为2113；独立session、stdin关闭、stdout/stderr直接追加正式日志，断开SSH不依赖前台tee。目标为625步，初次实查已推进到160步，GPU进程与训练PID一致；这是运行快照，尚无完成声明。W&B为 [第一组学习率00360a591a1e](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/00360a591a1e)。`2e-5/5e-5` 维持未启动、按用户要求暂停，需用户后续明确要求才启动。

监督记录位于 `runs/assistant/system/launch_lr1e-5_20261027_134434/launch.json`，本地启动与状态快照为 `tmp/sft_entry/first_lr_launch.json` 和 `first_lr_status.json`。全部正式配置、625步计划、同Base及数据身份、共享账本与原运行代码保持不变。未采用的顺序队列实现只做过8项mock CPU测试，未部署或执行，已移至本地助手system归档，不参与本次运行。

## 2026-10-27 用户授权继续其余两组学习率

用户随后要求继续 `2e-5/5e-5`。首组已完成625步、19,991样本，best step625，回答验证macro Loss为2.2394065；完整last3,640,493,931字节、best1,213,489,547字节均实际读取元信息，模型/配置/来源/游标身份通过。没有重跑或删除首组。预检时70GiB盘余19.2528GiB，够第二组；保留第二组后预计余14.73GiB，比第三组17.0406GiB预检门槛少约2.31GiB（还需考虑日志）。已提示用户额外准备至少3GiB空间。

独立控制器位于 `runs/assistant/system/lr_queue_2e-5_5e-5_b32_ep1_20261027_135836/`，controller PID启动快照2771。用户正式输出仍位于runs根目录，冻结配置及训练源码未变，runtime指纹仍为 `0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`。控制器先跳过已完成的1e-5，再顺序运行2e-5、5e-5；自身不初始化CUDA，每个check/start/resume是独立子进程。验证同Base、数据、Batch、seed、训练量及其他训练参数一致，只允许LR和W&B名称不同。

控制器新增外层超时与停止竞态保护，14项mock CPU测试本地与远程通过。check最多90秒；训练按冻结max_runtime_seconds+600秒监督，超时先TERM等待完整保存120秒，仍挂住才KILL并停止后续。退出0但summary未完整completed625也不会接着跑。空间不足至多等待600秒，随后记blocked_disk_space退出；不删除任何权重、不降低8GiB水位、不自动关闭实例。

启动后实查2e-5 PID2785在GPU训练，已到60步；W&B为 [4698fd0c406d](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/4698fd0c406d)。5e-5当时尚未启动，只排队，须空间就绪且2e-5正常完成才启动。队列状态state.json、controller.log与每组正式日志保持独立。本地记录为 `tmp/sft_entry/remaining_lr_queue_launch.json`、`remaining_lr_queue_status.json`、`first_lr_completion.json`；快照不代表最终完成状态。

后续用户回复“暂时无法增加空间”。只读实查2e-5已完成625步，best step625、回答验证macro Loss2.1846965；1e-5对应2.2394065。两组best及完整last均以CPU mmap实际加载，schema/stage/kind/run_id、步数和完整恢复字段通过。队列状态已blocked_disk_space且controller进程不存在，5e-5未启动，GPU计算进程为空，磁盘余14.7313GiB。没有中断第二组、没有发送终止信号、没有删改权重，也没有自动重启第三组。

已在队列目录写入 `user_decision.json`，记录第三组按用户空间约束暂缓；本地与远程收据为 `tmp/sft_entry/third_lr_paused.json`。这些为有限验证集上的单种子学习率对照结果，不代表完成独立测试或最终聊天能力评测。后续第三组需在空间就绪后由用户明确重新启动。

## 2026-10-27 空间清理与校验

用户要求清理无必要内容，并明确同意删除 `pretrain_100m` 与 `pretrain_manual_20261023_204911` 两组1000步基线的best/last权重。按精确白名单删除这些旧权重、SFT133649/140730已废弃的四分区载荷、三个固定版本可重下载源缓存、pytest/fixture/字节码/pip缓存和.sync旧备份。用户现有100M/1B、300M/3B及两组SFT模型和完整last、正式数据与父链、环境、凭据、冻结配置、账本和GPU锁保留。624个保护文件元数据未变，源码指纹仍0890…3872。

实际释放2.6637GiB，空闲由14.7312增至17.3949GiB。SFT全部分区摘要及train/validation加载、1B/3B validation、旧数据拒绝逻辑通过；pip check无依赖问题。原5e-5冻结配置预检通过，要求17.0406GiB，实际可用17.3949GiB。未启动第三组或其他训练。

明细在远程 `runs/assistant/system/space_cleanup_20261027_150505/{plan,report,verification}.json`，本地在 `tmp/sft_entry/space_cleanup_result.json` 与 `space_cleanup_verification.json`。两组旧1000步基线仅剩日志/配置，目录内weights_removed.json明确标记不可推理/续训；两版废弃SFT目录保留SUPERSEDED及PURGED_PAYLOAD标记。重建SFT时下载源缓存需重新获取。完整说明见 [清理报告](storage-cleanup-20261027.md)。

## 2026-10-27 5e-5完成、epochs/Batch准备及1e-4扩展

用户扩容后实查110GiB、余57.3949GiB，原5e-5实验后台启动并完成625步。最佳步625，SFT验证macro Loss2.1222417，固定原语料小窗口Loss3.6014293；进程计时396.99秒，估算计算费0.3341元。best与完整last实际CPU mmap读取，kind/schema/run_id/source与完整恢复字段通过。1e-5/2e-5结果保留，源码指纹未变，完成后余52.8737GiB。

按三组中SFT验证Loss最优5e-5生成epochs0.5/1/2及有效Batch8/16/32单轴计划，三个epochs候选共用两轮1250步/38步warmup，在313/625/1250步停止；Batch固定Micro8、accum1/2/4、一轮2499/1250/625步。验证/保存均对齐每3200样本，日志对齐每320样本。Batch32数值设置与既有控制组完全一致，直接复用；只新增5份冻结配置，逐项身份/训练计划/独立输出检查通过，没有启动这些候选。

计划在 `runs/plans/sft_epochs_batch_controls_20261027_153648/{plan,verification}.json`，本地在 `tmp/sft_entry/epochs_batch_prepared.json`、`epochs_batch_verification.json`。新增5组稳定权重估计22.60GiB、顺序启动所需初始空闲35.12GiB，当前扩容足够。原一轮调度结果不能复用作两轮调度下stop_epochs1的结果。

用户随后授权更大LR，对照新增1e-4，保持其他所有数值训练参数不变；独立后台PID启动快照2958，目标625步，W&B [5a30eeb43dc3](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/5a30eeb43dc3)。初次核验已到110步，日志Loss和梯度均有限。阶段计划标为held_pending_extended_learning_rate_result，epochs/Batch不启动。原语料保持指标随前三档LR增大而变差，后续不能只看SFT Loss宣布综合最优。独立test未使用；实际扩展结果见后续记录。

扩展1e-4已完整完成625步，best step625，SFT验证macro Loss2.0968458、同checkpoint原语料小窗口Loss3.8515981；所有已记录Loss和梯度范数均有限，裁剪前梯度范数最大2.4836。进程计时393.47秒、估算计算费0.3312元。CPU mmap实际读取best与完整last，schema/kind/run_id/source与完整恢复字段通过。四组同预算学习率对照均完成，独立test和任务准确率尚未评测。

对比5e-5的2.1222417/3.6014293，1e-4仅在SFT验证Loss上进一步降低，保持信号更差，不宣称综合能力更强。按折中工作基线保留5e-5的后续候选，计划状态改为ready_not_started；五份冻结配置已回读校验，五份逐实验后台启动脚本通过bash语法检查，Batch32复用已有完整控制组。未启动epochs或Batch，当前GPU无计算进程，余48.35GiB。

远程清单为 `runs/plans/sft_epochs_batch_controls_20261027_153648/ready_to_train.json`，脚本位于同目录commands。每次只执行一条，脚本先check再独立后台start；中断后用resume，不能重复start。镜像副本为本地 `runs/assistant/system/sft_controls_handoff_20261027/`，仅供查看，实际训练用远程冻结文件。学习率比较数据与图在本地 `runs/assistant/inference/lr_extension_comparison_20261027/`，原始读取记录在 `tmp/sft_entry/extended_lr_review.json`。

## 2026-10-27 用户选定2/3轮与Batch8/16并提交后台队列

用户明确要求先运行实际2轮、3轮以及有效Batch8、16共四组。沿用5e-5工作学习率、同一预训练Base和正式SFT数据；训练量轴使用共同三轮1875步cosine/57步warmup、在1250/1875步停止，Batch轴固定一轮、Micro8、累积1/2，目标2499/1250步。旧两轮调度下stop2的候选不冒充共同三轮调度下stop2；旧0.5/1轮不排队。原Batch8/16冻结文件直接复用，内容未改。

控制器本地和远程20项CPU/mock检查通过，覆盖四个不同步数、串行check/start/resume、跳过完整完成组、配置身份、共同调度、错步数/非完整退出时停止、空间水位与超时/停止信号。不初始化控制器自身CUDA，不修改scripts/train_sft.py或src/llm_lab、src/llm_sft；冻结运行指纹仍0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872。

启动前实查GPU空闲，110GiB盘余48.3526GiB；四组稳定权重预计18.0811GiB，串行需初始空闲约30.60GiB。控制器启动PID2391、第一组训练PID2403；后台目录runs/assistant/system/sft_epochs2_3_batch8_16_queue_20261027_194551。总计划runs/plans/sft_epochs2_3_batch8_16_20261027_194551/plan.json，正式输出全部保留在runs根目录。每组best和完整last独立保留，前一组完整完成后才开始下一组，失败停止后续，不删除权重、不关闭实例。

真实GPU复查第一组到200/1250步，6,400样本，记录的Loss与梯度均有限，SFT验证macro Loss2.2195875，best和完整last均已存在；W&B在线https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/7e0e8a72a7d4。此为启动后的有限进度快照，后三组仍排队，不代表四组训练全部完成或综合效果结论。详细进度以各组metrics/W&B为准；队列state的step在组结束时更新。

本地控制器、四份JSON/identity/env镜像、命令索引及启动/状态快照位于runs/assistant/system/sft_four_controls_controller_20261027/。用户运行说明见docs/sft-four-controls-20261027.md。原计划文档加了本轮入口与历史快照标记，防止再次启动旧候选。

## 2026-10-27 中英SFT候选原始储备下载

按用户要求预先下载后续SFT候选，不更换当前训练数据。已下载15个固定版本文件、1,346,228,580字节（约1.254GiB），保存在artifacts/posttrain/sft_reserve_20261027/raw/。Chinese SmolTalk八个任务文件262,746行，Smol-SmolTalk四个train分片460,341行，Dolly15,011行，Alpaca-GPT4中文48,818行，OASST2 train为128,575条消息，不能当作同数完整问答。公开test/validation分片未下载入候选池。

Chinese SmolTalk包含格式约束38,434、改写23,795、摘要26,687、文档问答66,441、编辑72,696、日常7,388、信息问答16,881、头脑风暴10,424行。上游指令评分不等于答案金标；英文Smol后续仍按原四来源白名单筛选。中文进入训练前须用既有官方OpenCC统一简体；原文按字节保留以复核来源。原始记录含可能重复及旧数据，不能视作等量新增可训练样本。

下载子进程CUDA不可见、两CPU核、低CPU优先级，限8MiB/s，保留24GiB磁盘。最初GitHub整文件请求超时，14个已完成文件保留；新增4MiB精确Range方式续取最后一文件，不重复下载其他源。最终下载器本地及远程各13项CPU/mock检查通过。源文件按冻结大小与SHA256提交后，另独立重算全部15个SHA256，源卡及Alpaca数据许可证、清单和inventory摘要均通过。

独立复查当前SFT manifest及train/validation/test文件摘要保持不变，不解析测试题；训练源码指纹保持0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872。现有四组GPU队列持续推进，本操作不启动新的GPU实验、不修改权重或冻结实验配置。

冻结清单configs/data/sft_reserve_20261027.json，SHA256为c82ea98556730917dc18ecfc9d4b44950757afeea7d0342e812fd3576d3b58f4。最终远程记录runs/assistant/data_prepare/sft_reserve_20261027_201653/，首个失败下载记录保留在sft_reserve_20261027_200817/。manifest标记training_ready=false；后续需繁转简、任务与答案检查、跨源和既有分区去重/污染检查、按监督Token配比及独立验收，再发布新数据集。详细目录见docs/sft-reserve-20261027.md。

## 2026-10-27 中英SFT扩充数据整理与发布

15个固定版本原始文件共915,491条记录/消息；先得到338,191条去重候选，再对64条分层文本作LLM内容抽审，逐条记录34条拒收。第二轮全量规则筛选与保守来源准入后保留327,209条候选（中文150,103、英文177,106）。分层抽审不估计全库错误率，不属于专家金标。

修复任务识别把classification说明、information子串与extracted误分到操作任务的问题；检查可明确解析的关键词完整词频、单词/条目/占位符数量、大小写和附言，隔离缺原文URL、角色串入及范围外内容。第二阶段只修改task，最终全部候选再次复查规则，并与原始第一阶段逐条核对所有其他字段，包括正文、Token和Mask。OASST只收本次逐条审过的3条，其余原始消息保留。

- 50,000条版：新增30,009条；完整输入Token 11,184,083，回答含EOS监督Token 6,090,534，data_identity=6e090dc2c2d72054ac7aa5fc943c8738888eb22542bd2280cf16f2fc4682fe58。
- 100,000条版：新增80,009条；完整输入Token 21,761,826，回答含EOS监督Token 10,553,126，data_identity=ad2e58f22c741c22ed029e2b53d67d5722b93e0c9dcbb88617598abcb4a6a3e3。

两版均保留原19,991条训练正文，5万为10万子集。主验证995、主测试1,000和pilot2,000按字节复用，旧latest_sft.json不改变。32K tokenizer、简体固定点、因果标签、回答及EOS监督保持原契约。每版独立全量重编码和Mask/EOS/摘要/精确隔离/pilot子集检查通过，正式SFTCorpus读取train/validation/pilot通过。

辅助评测另冻结验证2,000条、测试1,000条，来自训练及全部64条抽审均未使用的候选；逐条重编码，核对与训练及彼此的ID/group/规范化问题和材料隔离，按回答Token平衡中英。测试内容未提交内容审阅或超参数选择。它们不替换当前正式SFT入口的主评测。

本地新增流程35项通过、1项因未装PyArrow跳过；远程同一OASST用例实际通过，初始整理20项、第二阶段14项、辅助隔离2项，共36项通过。全部检查均为CPU数据/读取器验证，不代表新数据上的GPU效果实验。

发布根目录/root/autodl-tmp/model_training/artifacts/posttrain/sft_expansion_20261027_215627；新指针artifacts/posttrain/latest_sft_expansion.json；助手记录/root/autodl-tmp/model_training/runs/assistant/data_prepare/sft_expansion_20261027_215627。pool_sha256=c798696d47b88a2d1fed7dd3fb878ef3f6748dbe32844b64d4b810d83d580bc3；第二阶段脚本SHA256=4f6bbb3cc614d85a5c42fb908ce69e9ba932ccc2fed9e215b373fe220e8663ab。原始文件、两阶段候选与抽审、不可变运行快照、配置、正式数据及全部模型保留；所有验收通过后仅清理本次可重建candidates.sqlite并记录大小/摘要。

最终数据盘总容量110.00GiB，空闲25.68GiB。训练源码指纹仍为0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872；本次没有启动新GPU训练。详情见docs/sft-expansion-20261027.md。

## 2026-10-28 后训练测试比较与100k扩充启动

8组完成的SFT best与Base统一测试，主测试1,000条，固定64条生成和同一小窗口保持。按事前macro测试回答Loss选择lr5e-5_mb8_ga4_b32_ep3_stop2，Loss=2.04180737，英/中=1.89949644/2.18411829。EOS停止率=0.796875。原测试现用于配置选择，辅助测试保持封存。

用户明确选择沿用胜出配置，从相同预训练Base以100k重新SFT。已后台启动/root/autodl-tmp/model_training/runs/sft_300m_full_lr5e-5_mb8_ga4_b32_ep3_stop2_wu0p03_wd0p01_seed42_20261028_172855；实际2轮，B32、LR5e-5，共6,250更新，沿用3轮cosine共9,375调度步及282 warmup。初查最新step=190，W&B=https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/2946d6022bce。本记录不是长跑完成或最终效果结论。

评测与控制/辅助协议本地及远程15项CPU检查通过。完整结束后监督器自动运行预留测试，失败/不足步数/中断不触发；原权重全部保留。生成检查发现大小写违约、重复标题和误拒答，不能把Loss下降或EOS停止率解释成任务正确率。

训练源码仍为0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872。详情及入口见docs/sft-model-comparison-20261028.md。

## 2026-10-28 自研300M LoRA接入与三组学习率启动

新增独立llm_lora包和train_lora入口。原Full运行指纹保持0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872，LoRA指纹899d9750e6b9963cf0e5f60f22a3faf51655780d9519f919396da50fb0d3f914。本地及远程16项CPU检查通过；三独立GPU子进程连续33步与32+1恢复的最大浮点状态差0，冻结Base未改变，Adapter/AdamW/调度器/RNG/Sampler通过。

新最终测试已在LoRA调参前冻结1000条，协议/root/autodl-tmp/model_training/artifacts/posttrain/lora_evaluation_20261028_190311/protocol.json，SHA=9a26d3e02011f69b978f0fc79a440bba8052cae193c231eff13500fc9a881f11；中英回答Token接近50/50，覆盖格式177、一般问答176、材料问答177、改写235、摘要235。分类/抽取独立余量不足，本轮没有凑数。未把最终题目显示给助手或用于调参。

正式三组LR5e-5/1e-4/2e-4、r16/alpha32、全部7类线性层、B32、100k数据，同一预训练Base。初筛各3125步按辅助开发2000条macro Loss选配置，winner从原last继续到6250步。控制器PID启动快照6867，第一组已记录到90步，W&B=https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/b42623376a7c。这是启动快照，不是三组完成声明。

最终阶段将Full100k best与LoRA best对新1000题全部生成，客观约束仅统计可识别的要求和覆盖题数；未使用旧1000主测试继续选型，也不将EOS或ROUGE称为真实正确率。旧模型、所有语料及短测权重保留。实际CPU/GPU边界、断点和观察指令见docs/lora-training-300m.md。

## 2026-11-01 LoRA单轮rank与dropout队列

按用户要求固定LR2e-4，先运行r8/alpha16与r32/alpha64；均epochs=1、stop_epochs=null、100k数据、Micro8×累积4、3125更新、94 warmup。仅在两组完整结束后按辅助开发集macro Loss选rank，再对胜出rank分别从同一Base训练dropout0.05与0.1，复用其dropout0基线，共四组单轮训练。旧r16三轮调度结果不混入排名；不追加第二轮，不使用历史最终测试选型。

新增独立train_lora_one_epoch与run_lora_ablation入口，名称包含rank/alpha/dropout，额外绑定入口SHA。旧Full与LoRA数值代码指纹保持不变。新入口本地及远程各10项CPU检查通过，包括真实tiny模型完整一轮、拒绝多轮调度及stop覆盖、冻结命名和身份、rank变量隔离、两组完成前不选型、失败后不启动dropout等。

首次在线启动因W&B GraphQL超时退出，尚无参数更新；已核对直连超时、AutoDL加速HTTP403。重新冻结W&B离线配置，保留失败配置及日志，没有改写旧实验身份或删除权重。正式后台队列为runs/assistant/system/lora_rank_dropout_20261101_134339/plan.json，启动PID快照2689；权重输出仍在runs根目录。W&B离线、TensorBoard和JSONL继续记录，网络恢复后可同步离线数据。启动记录不代表四组训练完成，实际进度使用scripts/lora_ablation_status.py核对。

## 2026-11-01 中英数学后训练数据发布

按用户最终要求发布50,000条训练、2,000条开发、1,000条同领域测试和1,000条未见模板测试，均中英各半，同题双语保持同组。每个分区提供RLVR问题/答案和DPO偏好对两个视图；DPO负例来源明确为程序构造，并非人工或当前策略采样。另有嵌套2,000/10,000条训练版本，不自动启动训练。

GSM8K与TAL-SCQ5K固定官方提交原始文件全部下载。原始测试5,319条，排除259条归一化重复和1条标签/选项异常后，外部可评测5,059条：GSM8K英文1,319、TAL中文1,917、TAL英文1,823。没有按模型得分筛选题目，也没有截断问题。测试按语言、题型、难度和数据源分别报告，原始文件和排除记录完整保留。

发布artifacts/posttrain/math_alignment_20261101_162944/manifest.json；SHA=c913208b5372d25dc33c3044792cb75919a6062776574d75d6f4b4e28d6abf44，data_identity=42a70fb38bf74d0bed69a398a999c1d75fa585f9a2eb640d3b431a03e9b62497。受控样本经白名单AST有理数重算答案、错误偏好验证、同题双语/算式/模板分区隔离、OpenCC简体固定点和原32K分词器重编码检查；已知项目105,995个问题摘要用于精确排除，未宣称完全语义或预训练去污染。

本地与远程各19项CPU检查通过，全量独立审计均通过。数据发布约117.8MiB；远程同步49个文件、147,490,120字节，记录在runs/assistant/data_prepare/math_alignment_20261101_162944/。旧Full与LoRA数值代码SHA分别仍为0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872及899d9750e6b9963cf0e5f60f22a3faf51655780d9519f919396da50fb0d3f914；本次没有启动新的GPU训练。

## 2026-11-01 W&B网络恢复与四组历史补传

确认本机直连、AutoDL直连及AutoDL网络加速访问W&B均返回403，本机已有7890代理可正常连接。保留原控制公钥restrict，新增禁远程命令且限制TCP反向监听的独立隧道公钥；服务器实际仅监听127.0.0.1:17890，本机通过仅允许W&B相关443目标的中继连接现有代理。API Key仍保留在原服务器NETRC中。

服务器通过该链路真实认证HTTP200。SDK0.30.0未升级，纯CPU在线检查run onelf2mw已成功写入并从云端读回optimizer_step1。12项CPU/回环检查通过，远程命令与未批准TCP端口的密钥限制也实际验证通过。W&B专用网络配置由env_remote.sh加载，不给其他下载工具强加通用代理；CLI使用独立wandb_online.sh。

四组已完成的单轮LoRA记录全部补传：r8_ld0=a3422e345dab，r32_ld0=ca000af78489，r32_ld0.05=db047cc280e2，r32_ld0.1=77c20211b8d7。每组315条云端历史与本地315条optimizer_step序列逐条相同，最终3125，状态finished。完整证据runs/assistant/system/wandb_backfill_20261101_173528/report.json。历史冻结配置保持offline身份，cloud_sync.json单独提供云端链接；未冻结的后续模板恢复online。

此方案依赖本机开机、原7890代理及后台SSH隧道；Windows重启/注销后需执行scripts/wandb_network.ps1 Start，SSH暂时断线会自动重连。不是服务器脱离本机独立联网的方案。没有启动或重跑任何模型训练，旧Full与LoRA数值源码身份未改。操作和链接见docs/wandb-network-20261101.md。
