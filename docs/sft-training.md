# 自研 300M 模型的全参数 SFT

本入口使用已有的自研模型、32K BPE 和正式中英指令数据，训练代码为 [`src/llm_sft/engine.py`](../src/llm_sft/engine.py)，启动器为 [`scripts/train_sft.py`](../scripts/train_sft.py)。先完成全参数 SFT 的控制变量实验，再实现 LoRA 与 DPO。正式长跑由用户手动启动。

## 固定模型与数据

预训练实验 `pretrain_300m_3b_20261025_150858` 已完成 91,553 次更新、3,000,008,704 个 Token。SFT 从验证集选出的 **step 91,500 的 `best.pt`** 开始，各组独立重新加载同一份 Base；不会覆盖预训练模型。

| 项目 | 本轮取值 |
| --- | --- |
| Base | `/root/autodl-tmp/model_training/runs/pretrain_300m_3b_20261025_150858/best.pt` |
| Base SHA256 | `69b24779315561390bddbef18ce5ce9b505052a3ca697a0a651dfa6ea3c2bf45` |
| 参数量 | 303,350,784，全部参与更新 |
| 数据版本 | `sft_300m_20261026_170531` |
| 正式 manifest | `/root/autodl-tmp/model_training/artifacts/posttrain/sft_300m_20261026_170531/manifest.json` |
| 训练 / 验证 / 独立测试 | 19,991 / 995 / 1,000 条；另有 2,000 条训练内 pilot 子集 |
| 训练数据完整序列 | 4,392,427 Token，包含指令、模板和回答 |
| 回答监督 | **3,525,424 Token，含 EOS**；中文为简体，中英回答 Token 接近各半 |
| 长度 | 每条完整序列不超过 512 Token，Base 上下文上限 1,024 |

完整序列 Token 数与回答监督 Token 数用途不同：前者影响计算量，后者是损失归一化与监督预算的依据。标签只监督回答和 EOS，指令、系统提示、模板及 padding 不进入损失；数据在训练入口只移位一次。数据来源、许可和抽审边界见 [SFT 数据构建](sft-data-build.md)。数据通过结构检查和有限抽审，仍包含未被专家逐条核验的自动生成回答。

## 第一轮配置

配置模板为 [`configs/posttrain/custom_300m_sft.json`](../configs/posttrain/custom_300m_sft.json)。建议先单独完成默认 `2e-5`，检查曲线、停止行为和样例回答，再逐个跑另外两个学习率。

| 参数 | 默认值 |
| --- | --- |
| 学习率 | `2e-5`；首轮对照为 `1e-5 / 2e-5 / 5e-5` |
| Micro Batch / 累积次数 / 有效 Batch | `8 / 4 / 32` 条样本 |
| 训练量 | `epochs=1`，共 **625 次优化器更新**，最后一次为 23 条样本 |
| Warmup | `0.03`，向上取整为 **19 次更新** |
| 调度 | Warmup 后 Cosine Decay，最低学习率比例 `0.1` |
| 优化 | AdamW，betas=`[0.9,0.95]`，weight decay=`0.01`，梯度裁剪=`1.0` |
| 精度 / 随机种子 | BF16 / `42`，启用确定性算法 |
| 验证 / 保存 / 训练日志 | 每 `100 / 100 / 10` 次更新；训练结束再验证和保存 |
| 监测 | W&B 在线项目 `bilingual-training-lab`，同时写 TensorBoard 与本地 JSONL |

有效 Batch 按样本数定义，变长样本的实际 Token 数会变化。梯度累积按整次更新的回答监督 Token 总数归一化，最后不足四个 microbatch 的更新仍使用实际监督量。

## 先生成对照计划：不启动 GPU 训练

以下命令都在 AutoDL 的 **Bash 终端**执行，使用远程项目的绝对路径。

```bash
cd /root/autodl-tmp/model_training
source /root/autodl-tmp/model_training/scripts/env_remote.sh

/root/autodl-tmp/model_training/.venv/bin/python \
  /root/autodl-tmp/model_training/scripts/train_sft.py plan \
  --baseline /root/autodl-tmp/model_training/configs/posttrain/custom_300m_sft.json \
  --axis lr
```

`plan` 只在 `/root/autodl-tmp/model_training/runs/plans/sft_controls_lr_<时间戳>/` 写入 `plan.json` 和三份候选模板，**不占用 GPU、不启动训练，也尚未冻结运行源码身份**。`candidate_1/2/3.json` 分别对应 `1e-5/2e-5/5e-5`。之后一次只选择一份模板执行 `prepare`。

## 手动开始默认的 2e-5 实验

默认使用 `nohup` 在后台运行，日志直接写入文件，再单独查看。以下代码只准备一轮默认实验，正式训练仍由用户执行启动命令。

先生成本次时间戳并冻结实验：

```bash
cd /root/autodl-tmp/model_training
source /root/autodl-tmp/model_training/scripts/env_remote.sh
SFT_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)

/root/autodl-tmp/model_training/.venv/bin/python -u \
  /root/autodl-tmp/model_training/scripts/train_sft.py prepare \
  --template /root/autodl-tmp/model_training/configs/posttrain/custom_300m_sft.json \
  --stamp "$SFT_STAMP"

SFT_NAME="sft_300m_full_lr2e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_${SFT_STAMP}"
source "/root/autodl-tmp/model_training/runs/configs/${SFT_NAME}.env"
```

`prepare` 检查 Base、manifest、分区文件和 tokenizer，冻结配置、数据身份、Base SHA 与运行源码指纹。输出目录、冻结配置、日志和W&B采用同一个带参数的名称，例如 `sft_300m_full_lr2e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_<时间戳>`。其中mb是micro batch、ga是累积次数、b是有效Batch、ep是调度轮数、wu是warmup比例、wd是weight decay；小数点写成p。训练量对照还会带 `stop0p5` 等实际停止量。完整参数以 `config.json` 为准。

上述 `SFT_NAME` 适用于未修改的默认模板；改动超参数后应使用 `prepare` 返回的 `env` 路径，名称由配置自动生成。`.env` 设定本次 `RUN_NAME`、`RUN_CONFIG`、`RUN_DIR`、`RUN_LOG`，并把缓存和临时文件放到数据盘项目内。准备阶段仍未启动GPU训练。

先做启动检查，成功后再启动：

```bash
/root/autodl-tmp/model_training/.venv/bin/python -u \
  /root/autodl-tmp/model_training/scripts/train_sft.py check \
  --config "$RUN_CONFIG"
```

```bash
nohup /root/autodl-tmp/model_training/.venv/bin/python -u \
  /root/autodl-tmp/model_training/scripts/train_sft.py start \
  --config "$RUN_CONFIG" >>"$RUN_LOG" 2>&1 </dev/null &
SFT_PID=$!
printf '%s\n' "$SFT_PID" > "${RUN_LOG}.pid"
echo "训练进程 PID=$SFT_PID"
```

`check` 检查 CUDA/BF16、已有训练进程、GPU 占用、磁盘、共享预算和冻结身份，不更新模型参数。`start` 再检查一次并持有项目GPU锁。`nohup` 忽略终端断开信号，末尾 `&` 让进程在后台运行；`-u` 及时输出，`>>` 追加日志，`2>&1` 合并错误日志，`</dev/null` 关闭交互输入。PID返回只说明进程已创建，是否真正开始需查看日志与status。日志放在独立 `runs/logs/`，不污染训练输出目录。

单独查看日志：

```bash
tail -n 50 -F "$RUN_LOG"
```

此时按 `Ctrl+C` 只结束 `tail`，后台训练继续。不要把 `python ... | tee ...` 当作后台运行命令；它默认在前台，Ctrl+C会同时发送给Python和tee。

需要试跑时，可把 `start` 换成 `probe --steps 16`。这会真实执行前 16 次更新、写入断点，**沿用完整的 625 步学习率计划**。接着使用 `resume` 跑完同一实验；不要改成 16 步调度，也不要向已非空目录再次执行 `start`。

## 状态、停止和断点恢复

同一终端已加载 `.env` 时，可直接查询：

```bash
/root/autodl-tmp/model_training/.venv/bin/python \
  /root/autodl-tmp/model_training/scripts/train_sft.py status \
  --config "$RUN_CONFIG"
```

新终端先加载**原实验**的环境文件，把下面的占位时间戳替换成准备时输出的时间戳，不能重新运行 `date` 生成一个新的实验：

```bash
source /root/autodl-tmp/model_training/runs/configs/<RUN_NAME>.env
```

后台任务在启动它的原终端中用以下命令请求停止并等待完整保存。先退出 `tail`，再执行：

```bash
kill -TERM "$SFT_PID"
wait "$SFT_PID"
```

新终端先用status核对原实验的训练PID及进程命令，再发送SIGTERM；不要直接信任可能过期的 `.pid` 文件。`wait` 只适用于启动该进程的原shell。训练在当前完整更新结束后保存last再退出；强制kill-9、容器掉电或异常只能回到最近成功写入的断点。前台tee管道曾因Ctrl+C产生BrokenPipeError导致完整断点未写出，新版已补控制台破管保护，默认仍使用上面的后台直接写文件方式。

恢复原实验：

```bash
nohup /root/autodl-tmp/model_training/.venv/bin/python -u \
  /root/autodl-tmp/model_training/scripts/train_sft.py resume \
  --config "$RUN_CONFIG" >>"$RUN_LOG" 2>&1 </dev/null &
SFT_PID=$!
printf '%s\n' "$SFT_PID" > "${RUN_LOG}.pid"
```

`resume` 只读取该实验目录自己的完整 `last.pt`，恢复模型、AdamW、调度器、精度状态、随机数和数据游标，并续接监测记录。配置、数据、Base、训练源码或受核验的环境发生变化时会拒绝恢复。修改超参数应在新模板中进行，重新 `prepare` 创建独立实验；不要编辑已冻结的配置，也不要把另一组的 `last.pt` 拷贝过来续跑。

## 结果与前后对照

每轮先测 step 0 的 Base，再开始 SFT。模型选择使用验证集的**中英文回答 Loss 均值 `macro_loss`**，同时记录各语言 Loss 和监督 Token 加权 Loss。训练入口不使用独立测试集选模型。预训练验证语料的固定少量样本用于观察语言建模能力是否退化，这只是小规模保留能力检查。

| 文件 | 用途 |
| --- | --- |
| `best.pt` | 验证 Loss 最好的模型权重，用于生成；不含优化器，不能恢复训练 |
| `last.pt` | 最近保存的完整训练状态，用于 `resume` |
| `metrics.jsonl` | 持续追加的训练、分语言验证、保留能力、Token、吞吐、预算记录 |
| `summary.json` | 退出原因、步数、训练量、最佳步数和最新验证结果 |
| `run_manifest.json`、`config.json`、`environment.json` | 数据、模型、代码、配置与环境身份 |
| `tensorboard/`、`wandb_run.json` | 本地曲线与 W&B 运行关联 |

**如果所有 SFT 验证结果都没有优于 Base，`best.pt` 的 step 会保持为 0。** 这表示本轮未改善该选模指标，不能把这个文件当作已经选中的有效微调结果。

训练结束后，使用相同提示、模板和生成参数比较。下面默认 CPU 推理，避免与正在训练的 GPU 任务竞争；只有 GPU 空闲时才改成 `--device cuda`。

SFT 模型：

```bash
/root/autodl-tmp/model_training/.venv/bin/python \
  /root/autodl-tmp/model_training/scripts/chat_sft.py \
  --checkpoint "$RUN_DIR/best.pt" \
  --device cpu --greedy --max-new-tokens 128 --seed 42 \
  --prompt '请用三点解释机器学习与传统规则程序的区别。' \
  --output "$RUN_DIR/chat_sft_001.json"
```

同模板的 Base 对照：

```bash
/root/autodl-tmp/model_training/.venv/bin/python \
  /root/autodl-tmp/model_training/scripts/chat_sft.py \
  --checkpoint /root/autodl-tmp/model_training/runs/pretrain_300m_3b_20261025_150858/best.pt \
  --data-manifest /root/autodl-tmp/model_training/artifacts/posttrain/sft_300m_20261026_170531/manifest.json \
  --device cpu --greedy --max-new-tokens 128 --seed 42 \
  --prompt '请用三点解释机器学习与传统规则程序的区别。' \
  --output "$RUN_DIR/chat_base_001.json"
```

生成器从 manifest 读取与训练相同的模板，SFT 自动核验绑定的数据和 tokenizer；Base 对照必须显式给出 SFT manifest。输出到 EOS 即停止，终端只打印回答，JSON 记录停止原因、模型 SHA、步数、模板和数据身份。提示与生成预算超过上下文时直接报错，不会静默裁剪。报告文件拒绝覆盖，重复试验需换文件名。

EOS 监督并不保证小模型每次都能正确结束；触达 `max-new-tokens` 仍可能发生。Loss 改善也不等同于任务正确率或通用聊天能力提升，应同时检查固定的中英样例与保留能力指标，最后对冻结的优选方案做独立测试。

## 控制变量的执行顺序

先比较学习率，再把验证集选定的配置保存为新的基线模板，生成下一条轴的计划。不要直接运行所有变量的笛卡尔积。每组必须从同一 Base 开始，固定数据、随机种子、模板和评价方法。

| 顺序 | `plan --axis` | 对比值 | 公平性约束 |
| --- | --- | --- | --- |
| 1 | `lr` | `1e-5 / 2e-5 / 5e-5` | 固定 B32、一轮、warmup 0.03、weight decay 0.01 |
| 2 | `epochs` | `0.5 / 1 / 2` 轮 | 三组共同设 `epochs=2`，仅 `stop_epochs` 不同，使用同一条两轮调度 |
| 3 | `batch` | `8 / 16 / 32` 条/更新 | micro 固定 8，累积为 1/2/4；匹配处理样本位置的验证与保存频率 |
| 4 | `warmup` | `0.01 / 0.03 / 0.05` | 固定其他已选参数，观察早期稳定性 |
| 5 | `weight_decay` | `0 / 0.01 / 0.1` | 固定其他已选参数，观察泛化与过拟合 |

例如生成训练量计划，只生成文件，不运行：

```bash
/root/autodl-tmp/model_training/.venv/bin/python \
  /root/autodl-tmp/model_training/scripts/train_sft.py plan \
  --baseline /root/autodl-tmp/model_training/configs/posttrain/custom_300m_sft.json \
  --axis epochs
```

以上示例仍基于初始 `2e-5`；学习率优选改变后，应将 `--baseline` 指向已选基线模板的绝对路径。候选文件再分别 `prepare --template <候选绝对路径>`，每组使用新的时间戳，顺序执行。

训练量轴的共同两轮调度为 1,250 步、warmup 38 步。B32 下，`stop_epochs=0.5/1/2` 分别在 313/625/1,250 步停止；半轮按完整优化器更新取整，实际处理 10,016 条样本。这样比较的是同一优化轨迹的不同训练量，不能与把每组 `epochs` 分别设成 0.5/1/2、各自压缩 cosine 调度混为一谈。

该轴的 `1×` 使用两轮调度，不能直接复用初筛时采用一轮调度的 `1×` 结果。当前入口生成三个独立实验；每组从同一 Base 开始，以验证集确定最佳停止点。

Batch 轴的一轮更新数为 2,499/1,250/625，默认验证与保存间隔对应 400/200/100 步，即每约 3,200 条样本比较一次，末尾另测。小 Batch 不会因为更新步数更多而得到四倍选模机会；梯度噪声和学习率的最佳组合仍可能不同。

LoRA 的学习率、rank 和训练容量在全参数基线之后单独研究，再考虑 DPO。数据质量实验应预先固定共同的独立评测，匹配样本语言、长度、来源和回答监督 Token 预算；带 `SUPERSEDED.json` 的旧数据含已知隔离问题及不同发布条件，不能直接当作公平的“未过滤组”。

## 磁盘、预算和运行边界

本次准备时远程空闲约 **17.97 GiB**，当前检查为一轮新实验预留约 **17.04 GiB**，包含 checkpoint 原子替换时的临时空间与至少 8 GiB 水位。实际以每次 `check` 为准。

一轮完整 AdamW `last.pt` 约 3.39 GiB，模型 `best.pt` 约 1.13 GiB，稳定占用合计约 **4.52 GiB**，此外还有日志。当前空间无法让多轮完整输出及写入余量全部并存。每轮结束后先查看 `summary.json` 与磁盘，再决定是否归档或扩容；**脚本不会自动删除正式模型、旧实验或语料**。只保留 best 会失去该轮恢复能力，应由用户明确决定。

配置当前限制单轮估算计算费 15 元、单次运行两小时，项目共享总额 500 元并保留 100 元余量，小时价按 3.03 元记录。限制在更新边界检查，评测和保存仍有额外耗时；记录也不等于平台实际账单。退出不会自动关机或释放实例。训练时长与显存需以当前环境的实测记录为准，不能从 CPU 正确性测试推算。

## 2026-10-26 远程短测与已准备的实验

远程 vGPU-48GB、Torch 2.9.0+cu128、BF16 完成 16 次更新，再由独立进程从完整 `last.pt` 恢复到第 17 次更新后停止。模型权重读取、CPU 生成、TensorBoard 与 W&B 在线续接均通过。本地完整回归 416 passed / 5 skipped；远程新增入口 CPU 回归 59 passed。

| 短测项目 | 实测结果 |
| --- | --- |
| 累计处理样本 | 544 条，保持正式 625 步调度 |
| 峰值 allocated 显存 | 12.76 GiB；不是整机 `nvidia-smi` 占用 |
| 更新耗时中位数 | 0.486 秒；剔除基线、验证和保存，短测样本有限 |
| 回答验证 macro Loss | Base 2.5766 → 第 16 步 2.4510 |
| 固定预训练验证窗口 macro Loss | 3.2655 → 3.2847，出现小幅上升，正式训练继续观察 |
| 两次训练会话记录时长 / 估算费 | 73.27 秒 / 0.0617 元；不含实例空闲与其他进程 |
| W&B | [短测记录](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/f6ce19a85f0b) |

这些结果只证明入口、训练与恢复可用，不能当作正式超参对比或最终能力结论。16 Token 的生成检查触达了长度上限，未验证完整答案质量或 EOS 命中率。根据短测更新速度与验证/保存开销，一轮先按约 7–12 分钟预留；真实完整轮次尚未测量。

仅该短测的 `best.pt/last.pt` 已清理，日志和报告保留。随后旧命名的2e-5实验 `20261026_213514` 由用户前台启动，在第100步附近Ctrl+C，tee管道关闭触发BrokenPipeError，best已保存但完整last未写出。用户要求删除该次实验后，其权重、日志、冻结配置和W&B运行均已删除，释放约1.13GiB；预算账本保留已发生的估算费用。原100M、300M预训练及全部语料保持不变。

新版修复控制台破管保存问题并采用参数命名，旧的另外两组未启动配置标为superseded，仅供历史查看。当前运行指纹为 `0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`；最新本地完整回归435 passed / 6 skipped，远程入口CPU回归79 passed，其中实际关闭stdout/stderr管道的Linux子进程能够完整保存并以0状态退出。命名与控制台修复版本未另跑GPU；前述GPU数值属于修复前版本的受限短测。磁盘复查约17.97GiB，新三组仅准备，均未启动：

| 学习率 | `.env` 绝对路径 |
| --- | --- |
| `1e-5` | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr1e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261026_221316.env` |
| `2e-5`，建议先运行 | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr2e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261026_221317.env` |
| `5e-5` | `/root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr5e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261026_221318.env` |

新版计划在 `/root/autodl-tmp/model_training/runs/plans/sft_controls_lr_20261026_221316/plan.json`。直接在交互式Bash终端启动已准备的默认组，不用再次prepare：

```bash
source /root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr2e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261026_221317.env
/root/autodl-tmp/model_training/.venv/bin/python -u /root/autodl-tmp/model_training/scripts/train_sft.py check --config "$RUN_CONFIG"
nohup /root/autodl-tmp/model_training/.venv/bin/python -u /root/autodl-tmp/model_training/scripts/train_sft.py start --config "$RUN_CONFIG" >>"$RUN_LOG" 2>&1 </dev/null &
SFT_PID=$!
printf '%s\n' "$SFT_PID" > "${RUN_LOG}.pid"
tail -n 50 -F "$RUN_LOG"
```

上述配置绑定代码指纹，日后修改运行源码后应重新准备新实验。后续组启动前需逐次处理磁盘容量问题；当前空间适合先运行一组，不承诺三组完整恢复状态能够并存。

阅读代码可按 [`config.py`](../src/llm_sft/config.py) → [`data.py`](../src/llm_sft/data.py) → [`engine.py`](../src/llm_sft/engine.py) → [`train_sft.py`](../scripts/train_sft.py) → [`chat_sft.py`](../scripts/chat_sft.py) 的顺序。旧预训练代码与新 SFT 包共同进入运行指纹，正式训练和恢复期间不要同步修改这些运行源码；文档与测试不进入该指纹。

## 2026-10-27 本次正式运行状态

用户最终选择仅运行第一组 `1e-5`，`2e-5` 与 `5e-5` 暂停；没有自动顺序队列。第一组已由独立后台session启动，目标625步，保留best与完整last。W&B链接为 [00360a591a1e](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/00360a591a1e)，运行状态以实际summary/进程为准。

当前磁盘仅支持先跑一组；保留所有完整状态再启动后续组需要额外空间，不会自动删除前一组权重。不要对已启动的第一组再次start；停止后需要续训时使用其原配置resume。

在远程新终端查看本次日志：

```bash
source /root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr1e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261026_221316.env
tail -n 50 -F "$RUN_LOG"
```

Ctrl+C仅结束日志查看。与后台任务分离的启动记录为 `runs/assistant/system/launch_lr1e-5_20261027_134434/launch.json`。

## 2026-10-27 继续2e-5与5e-5

用户已重新授权后两组。1e-5完整完成625步，best与last保留并核验；队列会跳过它。2e-5已后台启动（W&B：[4698fd0c406d](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/4698fd0c406d)），5e-5按顺序排队，各组仍从相同Base开始。

队列状态位于 `/root/autodl-tmp/model_training/runs/assistant/system/lr_queue_2e-5_5e-5_b32_ep1_20261027_135836/state.json`。若完成2e-5后空间不足17.04GiB，控制器会最多等待10分钟，之后停止并记录blocked_disk_space，不删权重。按本次启动前容量，第三组还需额外至少约3GiB空闲；扩容或腾出空间后，等待期内可自动继续，超时退出后需重新启动队列。

查看第二组日志：

```bash
source /root/autodl-tmp/model_training/runs/configs/sft_300m_full_lr2e-5_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261026_221317.env
tail -n 50 -F "$RUN_LOG"
```

停止整个顺序任务应向当前控制器PID发SIGTERM，控制器会通知当前训练进程完整保存后退出，并停止后续组。PID需先核验对应的queue配置，不能直接照抄历史快照。当前快照controller PID2771，实际以state.json为准。

## 2026-10-27 空间无法增加后的最终状态

用户回复暂时无法增加空间，`5e-5` 暂缓。核验时 `1e-5/2e-5` 均已完整完成625步，best与含optimizer/scheduler/RNG/scaler的完整last均实际加载元信息通过；两组最佳回答验证macro Loss分别为2.2394065和2.1846965，不能据此代替最终任务准确率评测。

控制器已因blocked_disk_space超时退出，进程已核实不存在，5e-5未创建训练输出，不会因以后空间增加自行重启。当前余量14.73GiB，GPU无计算进程。未删除任何权重；以后要跑5e-5需在空间就绪后明确重新启动。用户决定记录为上述队列目录内的 `user_decision.json`。

## 2026-10-27 扩容后完成5e-5并扩展学习率

用户扩容后实查110GiB、启动前空闲57.39GiB。5e-5按原冻结配置完成625步，最佳步625，SFT验证macro Loss2.1222417，原语料小窗口Loss3.6014293。进程内计时396.99秒、估算计算费0.3341元，完整best/last已核验。W&B为 [6c73343449e7](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/6c73343449e7)。

用户随后要求尝试更大学习率，已额外启动1e-4，固定原Base、数据、B32、seed42、1轮625步与相同调度形状，只改变LR；W&B为 [5a30eeb43dc3](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/5a30eeb43dc3)。完整结果以实际summary为准，早期收敛不直接代表最终更好。输出目录为 `runs/sft_300m_full_lr1e-4_mb8_ga4_b32_ep1_wu0p03_wd0p01_seed42_20261027_154305`。

已准备并回读核验epochs与Batch五份新增候选，Batch32复用既有完整一轮控制组。加入1e-4后，候选暂不执行，等待扩展学习率比较和任务表现确认。具体矩阵、共享两轮调度及配置路径见 [训练量与Batch计划](sft-epochs-batch-plan.md)。

1e-4随后已完整完成625步，best step625，SFT验证Loss2.0968458、同checkpoint原语料小窗口Loss3.8515981，完整best/last已读取核验。它在指令拟合上略好，保持指标退化更明显，暂不将其认定为综合最佳。后续五组配置仍以5e-5作为折中工作基线，已整理ready_to_train.json和单条后台启动脚本，尚未执行；详见上方训练量与Batch计划中的命令表。四组模型均保留。
