# 自研300M LoRA：入口、运行与验证

## 后续新实验规则（2026-10-31）

用户已明确要求后续只训练一轮：新配置统一 `train.epochs=1`，`train.stop_epochs` 省略或为 `null`，调度与 Warmup 均按一轮计算。不再自动将胜出配置继续到第二轮，也不安排两轮/三轮对照。断点恢复只完成这一轮剩余步数。

100,000 条数据、有效 Batch 32 时，一轮为 3,125 次参数更新。下文的三轮调度、两轮终点及续训指令属于 2026-10-28 的历史实验记录，不作为新实验模板；原冻结配置和恢复身份保持不变。

2026-10-28已实现独立LoRA-SFT入口，CPU及GPU连续/恢复检查通过；三组学习率已提交串行后台流程。最新快照阶段为`screening`，第一组记录到890步；这不是三组全部完成声明。

## 2026-10-28 历史三组配置

同一预训练Base、同一100k训练集；rank16、alpha32、标准alpha/r缩放、LoRA dropout0、bias none；覆盖每层Attention q/k/v/out与MLP gate/up/down。Embedding、共享输出层、Norm及其他Base权重均冻结。

| 顺序 | 学习率 | 初筛步数 | 最终目标 |
| --- | ---: | ---: | --- |
| 1 | 5e-05 | 3,125（1轮） | 只将开发集选出的配置继续到6,250（2轮） |
| 2 | 0.0001 | 3,125（1轮） | 只将开发集选出的配置继续到6,250（2轮） |
| 3 | 0.0002 | 3,125（1轮） | 只将开发集选出的配置继续到6,250（2轮） |

精度BF16，seed42，Micro8×累积4=有效Batch32，保留原Base dropout。所有候选预先固定3轮cosine共9,375调度步、warmup282步、实际停止2轮；初筛只是暂停，恢复沿用原调度、AdamW、随机数及采样游标。

每100更新验证/保存，每10更新记录；W&B、TensorBoard与JSONL同时保存。rank8/32和注入范围的可选扩展尚未排队。

## 选模和最终测试

每组内部按主验证995条保存best。三组均完成3,125更新后，按独立辅助开发2,000条的macro回答Loss选择学习率；原语料保持Loss单独报告，不编造综合分。胜出者从完整Adapter last继续到6,250更新，最后使用其验证best参与方法比较。

新最终测试在LoRA调参前已冻结：1,000条，中英回答Token占比50.00%/50.00%；排除100k训练、已有验证/测试、内容抽审及共享重复组/精确问题和材料。

任务条数：{"format_following": 177, "short_qa_or_other": 176, "grounded_qa": 177, "rewriting": 235, "summarization": 235}。分类和抽取没有足够独立余量，本轮不补造或重标这两类；结果不能代表完整的分类/抽取评测。

冻结协议：`/root/autodl-tmp/model_training/artifacts/posttrain/lora_evaluation_20261028_190311/protocol.json`，SHA256=`9a26d3e02011f69b978f0fc79a440bba8052cae193c231eff13500fc9a881f11`。

配置锁定且胜出者完整训练结束后，当前100k Full best与LoRA best对同一新测试1,000条全部计算回答Loss，并各自对全部1,000题自由生成（最多256新Token，保留完整问题）。不以64条抽样替代。

生成结果保存题目ID、参考、回答、停止原因与Token；单列可识别的大小写、JSON、单词/列表/占位符数量、完整关键词词频约束的通过率及覆盖题数。未解析到的要求不记作已通过；格式通过率、EOS、引用重叠和Token准确率都不等于通用任务正确率。原始候选不是全量专家金标，近重复隔离有既定启发式边界。

## 查看实时进度

在AutoDL远程Bash执行：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
.venv/bin/python scripts/lora_status.py
```

该只读入口会核对实际进程，并区分正在恢复的进程与旧summary快照。当前第一组曲线：

```bash
source /root/autodl-tmp/model_training/runs/configs/lora_300m_data100000_r16_a32_all_lr5e-5_mb8_ga4_b32_ep3_stop2_wu0p03_wd0p01_seed42_20261028_191900.env
tail -n 30 -F "$RUN_LOG"
```

Ctrl+C仅结束tail，训练继续。切换实验时source该组env，所有正式输出均在runs根目录，包含data100000/rank/alpha/目标范围/LR/Batch/调度及时间戳。

第一组W&B：[在线曲线](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/b42623376a7c)。后两组实际启动后会各自创建运行记录。

控制器状态：`/root/autodl-tmp/model_training/runs/assistant/system/lora_lr_screen_20261028_191900/state.json`；启动PID快照6867。停止整个流程前先用ps核对该PID仍是本控制器；TERM会通知当前训练按完整更新保存，并停止后续。不要只结束tail或只根据旧PID操作。

## Checkpoint与恢复

- best.pt：项目Adapter-only Checkpoint，含LoRA参数、模型配置、Base/数据/分词器/代码身份。
- last.pt：另含Adapter AdamW、调度器、GradScaler、Python/NumPy/Torch/CUDA随机状态与数据游标；只有它可以完整resume。
- adapter_model.safetensors：仅Adapter张量。adapter_config.json为自研元数据协议，含rank/alpha、目标层及Base SHA；加载使用llm_lora.inference，而不是直接当作HF AutoPeftModel导出。
- 原303M Base不会被每个Checkpoint重复保存；它的文件与冻结参数摘要保持绑定。新rank、覆盖范围或数据版本必须新建实验，不能修改已冻结配置续训。

若流程中断，先确认控制器及原训练已经退出，再使用本组原配置恢复；筛选阶段限定到3125步：

```bash
source /root/autodl-tmp/model_training/runs/configs/lora_300m_data100000_r16_a32_all_lr5e-5_mb8_ga4_b32_ep3_stop2_wu0p03_wd0p01_seed42_20261028_191900.env
.venv/bin/python -u scripts/train_lora.py resume --config "$RUN_CONFIG" --until-step 3125
```

胜出组继续阶段使用resume且不传until-step，按原计划到6250。不要在仍运行的控制器旁并行手工启动。控制器失败会保存原因，当前实现不会自动重提失败的整条队列。

## 正确性和实际硬件证据

本地隔离依赖CPU与远程CPU共16项检查实际通过，覆盖零初始化等价、冻结梯度与Base张量不变、Adapter严格装载、合并推理、带Dropout的精确恢复、冻结配置与开发集不打开最终测试等。

三个独立GPU子进程：连续33步，与32步后恢复到33步对比；总66次GPU更新。Adapter、优化器、调度器、RNG与Sampler一致，最大浮点状态差0，验证阈值atol=1e-6/rtol=1e-5；Base摘要未变化。

实际可训练参数6,979,584，168个目标Linear，冻结Base 303,350,784。短测峰值allocated约9.47GiB，仅覆盖前33步，不代表全量长序列峰值或通用速度。

本地torch2.10与远程torch2.9分别执行CPU检查；GPU一致性结论来自同一远程环境，不承诺跨硬件/驱动逐位一致。LoRA仍可能过拟合和遗忘，应结合指令效果与保持指标比较。

旧Full运行源码指纹仍为`0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`；LoRA运行身份为`899d9750e6b9963cf0e5f60f22a3faf51655780d9519f919396da50fb0d3f914`，同时绑定新包、入口、评测依赖源码和包版本。

## 代码阅读顺序

1. src/llm_lora/adapters.py：冻结Base，注入A[r,in]和B[out,r]。输入[B,T,in]经过A/B成为[B,T,out]，乘alpha/r后加到原Linear结果；B零初始化使初始更新为零。
2. src/llm_lora/config.py与scripts/train_lora.py：独立协议、参数化命名、冻结及共用GPU锁；没有修改原llm_sft或llm_lab。
3. src/llm_lora/engine.py：复用原SFT的Mask、一次错位、按监督Token归一化和采样；新引擎只优化Adapter并保存其完整恢复状态。
4. src/llm_lora/data.py：辅助开发与最终测试独立协议，训练/开发阶段不读取最终测试题。
5. scripts/run_lora_experiments.py：等预算串行筛选，锁定配置，继续训练；失败、预算停止或不足步数不会进入最终测试。
6. src/llm_lora/evaluate.py和metrics.py：Full/LoRA逐题生成与受限客观评分，明确统计覆盖和语义边界。

本地启动快照与冻结配置镜像：`E:\codex\model_training\runs\assistant\system\lora_lr_screen_20261028_191900`；远程Controller记录：`/root/autodl-tmp/model_training/runs/assistant/system/lora_lr_screen_20261028_191900`；GPU短测：`/root/autodl-tmp/model_training/runs/assistant/smoke/lora_correctness_20261028_191340`。

最终报告目录预留为`/root/autodl-tmp/model_training/runs/assistant/inference/lora_vs_full_final_20261028_191900`，仅在配置锁定且胜出训练结束后创建。没有自动删除旧模型或关闭计费实例。
