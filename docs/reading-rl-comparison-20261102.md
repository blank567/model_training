# 阅读RL三组数据、三种算法对照（2026-11-02）

用户已授权顺序运行九组并保留全部结果；数据仅一轮，PPO每批采样允许两次内部更新。首组PPO已在后台训练，W&B实验为 `c12bf95cf966`。启动不等于完成；实时状态以服务器队列、指标、summary为准。

## 数据和起点

| 数据组 | 题数 | 与领域SFT的关系 |
| --- | ---: | --- |
| old_matched | 25,000 | 原50,000题中的固定子集，明确属于已见题对照 |
| new_matched | 25,000 | DuReader-robust、Quoref公开人工跨度；与既有SFT问题、材料及已知文章隔离 |
| old_full_50k | 50,000 | 原始全部SFT训练题，用于观察数据量扩展 |

新源清单为 `artifacts/posttrain/reading_rl_independent_25k_20261102_v5/manifest.json`。中文11,336、英文13,664；答案+EOS Token分别55,246、56,360，中文占49.5009%。v4抽查发现临床问题与所标材料对应问题不同，因此在候选池中额外排除临床建议/医患聚合1,161条和主观推荐160条。v4保留原文件并加隔离标记，未训练新组。为保持25,000题及中英监督Token均衡，英文候选按答案长度做了1,189次集合交换；不使用模型得分选题。

原始人工标注、跨度、简体规范、编码、EOS、掩码、文件摘要及隔离均全量审计。v5另阅读15条来源/答案长度样本的完整问题、材料及答案，并记录每题判断和局限。该检查不保证全部语义正确或原文世界事实正确；历史、口语化内容仍按所给材料做抽取任务。文章标识缺失时只能验证材料级隔离，不能声称完全排除跨语言语义重复或预训练污染。

比较计划v2在 `artifacts/posttrain/reading_rl_compare_25k_50k_20261102_v2/plan.json`。旧两组索引与v1逐字节相同；已经启动的旧组保留自己的v1冻结配置。修订只影响尚未启动的新数据组。

所有九组分别从已完成的阅读全参数SFT `best.pt` 初始化，SHA256：

```
7b13791c6afb50eeb09db33c750e40e6cab61d9fc26cd08ea2f8aa72ce58eca4
```

原模型路径：

```
/root/autodl-tmp/model_training/runs/reading_sft_300m_full_data50000_lr5e-5_mb8_ga4_b32_ctx1024_ep1_20261101_205704/best.pt
```

不会从前一算法的输出接着训练后一算法。参考策略始终冻结为此SFT；保留303,350,784参数的自研Decoder，原预训练/SFT/LoRA源码及历史身份不改动。

## 单轮配置

| 参数 | DPO | PPO+RLVR | GRPO+RLVR |
| --- | --- | --- | --- |
| actor学习率 | 5e-6 | 1e-6 | 1e-6 |
| 每个数据Batch的问题数 | 16 | 16 | 16 |
| 评分/反向传播微批回答数 | 16 | 16 | 16 |
| 每题策略采样数 | 离线构建 | 1 | 4 |
| 一批采样的内部优化次数 | 1 | **2** | 1 |
| 训练数据epochs / stop_epochs | 1 / null | 1 / null | 1 / null |
| 精度 / 上下文 / 最大回答 | BF16 / 1024 / 64 Token | 同左 | 同左 |
| warmup / 最低LR比例 | 0.03 / 0.1 | 同左 | 同左 |
| 梯度裁剪 / weight decay | 1 / 0 | 同左 | 同左 |
| 特有参数 | beta=0.1 | clip=0.2，KL=0.02，gamma=1，lambda=0.95 | clip=0.2，KL=0.02 |

PPO使用独立线性value head，输入为detach后的actor特征，critic LR=1e-4、value coefficient=0.5、value clip=0.2；不是另一个完整303M critic模型。有限回答到EOS或64 Token截断均按当前有限回合结束处理，终止bootstrap=0。采样温度1、完整softmax，无top-p截断；actor和参考策略评分都禁用dropout以保证新旧概率一致。

25,000题共有1,563个数据Batch；PPO是3,126次optimizer更新，GRPO是1,563次。50,000题共有3,125个数据Batch；PPO是6,250次更新，GRPO是3,125次。最后不足Batch保留。PPO每批两次更新不代表遍历数据两轮；Warmup和调度按各自真实optimizer总步数计算。

DPO先由同一个初始SFT为训练问题生成负回答，chosen始终为原始人工答案。每题最多16个候选，优先另一材料片段；也可接收明确不满足“只输出原文片段”要求的回答，分别记录 `wrong_material_span` 与 `not_a_material_span`。后者是抽取格式负例，不宣称已由人工判定为语义错误。排除空答、缺少EOS、特殊Token、疑似别名/包含关系、重音/标点差异及高F1重叠。偏好属于规则派生标注，**不冒充人工成对偏好数据**。

旧/新主对照按合格偏好实际数量取同量，并重新检查中英监督Token配比，不复制问题或伪造负例凑齐。旧50k的DPO记录数也可能少于50k。正式目录的 `n...` 显示实际对数；不足20,000个均衡主对照时停止等待复核。数据准备完成前，不能宣称已获得完整DPO对。

## 验证和真实GPU证据

- 20项数据/目标函数/KV缓存CPU检查以及4项三算法完整训练/精确断点恢复检查均在远程通过。
- 后续负例规则/队列检查及偏好发布-读取联调另行记录；不把fixture当作GPU训练效果。
- 初始SFT在真实GPU上完成三算法更新短测；原Decoder与缓存前向BF16相对L2误差最大约1.12%，检查的argmax一致。最长输入微批反向传播检查通过，短测峰值allocated约25.31GiB，不能当作正式训练的全部reserved显存。
- 三组各抽32个训练问题、每题4次真实采样，均得到正确及混合奖励。修订v5从这些真实采样中仅保留输入与金标准逐字节不变的交集，明确记录复用数值检查及保留样本数；不声称又跑了新GPU推理。
- 正式训练每100个数据Batch在固定256题开发子集生成评测，按中英macro EM，再以自定义F1作同分区分来保存best。第一组初始开发EM=26.9531%；这不是最终测试成绩。
- 全部实验选型结束后再用固定2,000题最终测试统一评测。256题开发子集、GPU短测32题与最终测试2,000题分别报告。

## 队列、日志和磁盘

顺序：PPO旧25k→PPO新25k→PPO旧50k→GRPO旧25k→GRPO新25k→GRPO旧50k→偏好准备→DPO旧匹配组→DPO新匹配组→DPO旧50k。九组训练串行，共用原 `.gpu_training.lock`；偏好准备也占该锁。每组保留独立best、可恢复last、optimizer/RNG状态、配置、日志、TensorBoard与W&B。

正式实验输出在 `runs/` 根目录，名称包含算法、数据、数量、Batch、LR、PPO更新数或GRPO组数、ep1、seed与时间；助手诊断和数据准备位于 `runs/assistant/` 分类下。

服务器控制文件：

```
/root/autodl-tmp/model_training/runs/assistant/system/reading_rl_queue_20261102/queue.json
/root/autodl-tmp/model_training/runs/assistant/system/reading_rl_queue_20261102/queue.log
```

以queue.json实际状态为准：`running`、`waiting_for_disk`、`all_nine_completed` 或 `failed_requires_review`。未完成一轮、人工停止、预算触发、缺少权重或进程异常均不会当作成功自动跳过。

首次检查九组全保留约需53GiB空闲，用户选择扩容约40GiB。每个新训练任务开始前检查8GiB预留、best/last及原子保存临时空间；未生效则最多等待1小时，期间不删除已有模型。等待超时后队列退出，扩容生效后可重新启动同一队列，已完成任务不会重跑：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
mkdir -p runs/assistant/system/reading_rl_queue_20261102
nohup .venv/bin/python -u scripts/run_reading_rl_queue.py \
  --state runs/assistant/system/reading_rl_queue_20261102/queue.json \
  >> runs/assistant/system/reading_rl_queue_20261102/queue.log 2>&1 < /dev/null &
```

已有队列运行时互斥锁会拒绝第二个队列。队列目录放置 `STOP` 会在当前任务结束后阻止下一项启动；当前训练需在自己的正式输出目录放 `STOP` 才会在采样Batch边界保存并停止。

每组默认计算预算12元、4小时，偏好准备20元、6小时；项目500元并保留100元。预算按运行时间估算计算费用，不是云平台账单，也不包含闲置实例与磁盘费用。触发上限会保存可恢复状态并停止后续队列，不自动追加数据轮数或预算。

等题数、等一轮并不等于等Token、等计算或等难度。旧/新来源、答案长度及预训练可见性也不同；结果应解释为数据方案对照，不能把差异全部归因于是否复用SFT。旧50k是数据量扩展；固定最终测试来自旧来源分布，也须在报告中说明。不同算法loss不具有直接可比性，主要比较同一独立评测指标、生成行为、实际Token与成本。
