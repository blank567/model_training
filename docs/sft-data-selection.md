# 自研 300M 首轮 SFT 数据筛选

核查日期：2026-10-26（Asia/Shanghai）。本文保留候选研究阶段记录；后续已完成正式数据构建与抽审发布，最新结果见 [SFT 数据构建](sft-data-build.md)，正式版本为 `20261026_170531`。下文的“尚未完成”描述研究阶段快照。

当前建议：中文优先筛原作者 Alpaca-GPT4 中文研究用数据，英文优先筛 Smol-SmolTalk 的指定来源；Dolly 的有上下文任务作补充，OASST2 中文作备选。COIG-CQIA 因原始数据卡未给出完整许可，暂不进入默认训练混合。

## 与当前预训练的关系

本轮讨论的后训练起点是自研 303,350,784 参数模型，预训练计划为约 3B Token，使用 32k BPE、1024 上下文。首轮 SFT 先验证短回答、基础指令、回答结束和简单任务，不预设基础模型已经具备稳定的通用问答能力。

原项目的 Qwen2.5-0.5B + MASSIVE 对照仍独立保留。当前通用指令候选需要新增数据协议，不能直接放进要求 `intent/slots` JSON 的旧 MASSIVE 入口。SFT 训练还需单独接入自研权重及 BPE；本次没有修改预训练源码、权重、依赖或训练配置，也没有启动新的 GPU 任务。

## 候选来源与决定

| 来源 | 当前决定 | 适合的部分 | 已确认边界 |
| --- | --- | --- | --- |
| [原作者 Alpaca-GPT4 中文](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/80cda626ea305004be42426671c66efebbf22144) | 中文主候选，限本轮非商业研究范围 | 简单问答、带输入的改写/提取、基础格式要求 | 合成数据；数据 CC BY-NC 4.0，代码 Apache-2.0 不覆盖数据限制；发现错误和未完成回答，不能整集直接放行 |
| [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | 英文主候选，按来源和样本双重筛选 | `smol-magpie-ultra-short`、`smollm-rewrite-30k`、少量 `smol-summarize-20k`、`smol-contraints` | 新生成来源采用 Apache-2.0，导入的其他公开来源仍保留原许可；来源名和长度合格不代表回答遵守指令 |
| [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 英文补充 | 有给定上下文的短阅读问答、提取、摘要 | CC BY-SA 3.0；人工数据仍存在上下文不支持的事实，分类标签也可能不准确 |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | 中文人工对话备选 | 完整、自足的中文首轮问答 | Apache-2.0；合格中文对数尚未统计；必须重建父子关系、审核质量并按对话树分组 |
| [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) | 暂缓进入默认混合 | 仅保留研究观察 | 官方卡仍为 `License: [More Information Needed]`；知乎子集较长，存在考试/医疗/专业内容 |

SmolTalk 的来源许可说明见[官方数据卡](https://huggingface.co/datasets/HuggingFaceTB/smoltalk#license)。Alpaca 数据许可证和作者用途说明已与原始文件一起保存；未来若改变用途，必须重新核对数据条件，不以镜像标签覆盖原作者声明。

## 已完成的实际抽样

从远程只读取得当前词表，核验 SHA256：`6040c90c045836982a47bc0544942f122d0446bf9ed4d2ac1929befd0d4b86b3`。全部长度由此 BPE 计算。

临时计数模板为 `User: {prompt}\nAssistant: {answer}` 再加一个 EOS；正式聊天模板尚未冻结。中文统计采用原文，尚未完成官方 OpenCC 简体转换；SmolTalk 只测首个完整 user→assistant 往返，正式多轮样本需要重新计数。

| 实际检查的样本 | 数量 | 完整输入 Token 中位数 | P90 | ≤512 Token | ≤1024 Token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alpaca-GPT4 中文 | 500 | 160 | 320 | 493/500 | 500/500 |
| Smol-SmolTalk 混合预览 | 300 | 378 | 602 | 244/300 | 293/300 |
| Dolly | 100 | 108 | 444 | 92/100 | 97/100 |
| COIG-CQIA `10why` | 100 | 232 | 343 | 100/100 | 100/100 |
| COIG-CQIA 知乎 9–10 分子集 | 100 | 567 | 1382 | 43/100 | 85/100 |

合计 1,100 条完成长度核查。Alpaca、Dolly 与两个 CQIA 文件采用 seed=42 从固定文件随机取样；SmolTalk 是训练 viewer 的 0、100000、300000 偏移各 100 条，**不是全量随机样本，也不是固定 revision 的正式训练输入**。其长度表还包含将来会被来源白名单排除的内容，不能当作最终混合分布。

表中的比例只表示长度适配，不能解释为数据质量通过率。当前尚未证明任何候选已经提供了 2 万条符合全部要求的样本。

## 发现的具体质量问题

- Alpaca 中文原始零基行 512：关于 1981 年 NBA MVP 的回答给错人物，已用 [NBA 官方历年名单](https://www.nba.com/news/history-mvp-award-winners)核对，拒收该条。
- Alpaca 中文原始零基行 752：回答以逗号结束，最后一句没有完成。应拒收，不能补 EOS 后当成完整回答。
- SmolTalk viewer 全局行 77：样本总长恰好 1024，但回答违反提示的句数和人称限制。说明只按长度或 source 放行不够。
- Dolly 原始零基行 1535：回答补充了给定材料不支持的创办者关系，拒收该条。
- 部分 Dolly 分类标签与实际任务不一致，因此任务配比需要核对文本，不能只按 `category` 计数。

完整定性记录见研究产物中的 `en_sample_review.md`、`zh_sample_review.json`。这些观察不能外推成全量错误率；未使用模型自动评分冒充人工金标。

## 首轮数据构建建议

目标仍为 20,000 条训练、1,000 条验证、1,000 条独立测试，另从训练分区取 2,000 条作 pilot。正式训练从同一个 Base 重新开始。数量是上限目标，质量不够时先报告缺口，不用重复或未审核来源补足。

1. 固定原始 commit、文件名、文件 SHA、原始行号或对话树 ID、原文 hash、来源和许可。SmolTalk 正式制备读取固定版本的原始 train Parquet，不从 viewer 导出后冒充固定训练集。
2. 中文统一简体后再做去重与分区，并保留转换前后身份；复用原 32k BPE，不重训分词器。先选择短而完整的任务，长度偏好 ≤512，硬上限 1024，避免截断回答后造成错误 EOS 标签。
3. 优先简单问答、输入支持的提取/改写、可核对的格式约束；摘要额外检查句数、人称、长度和事实忠实性。首轮不以复杂数学、长思维链、长代码、实时信息、医疗和法律建议为主任务。
4. 精确去重与近重复分组后划分；同一上下文、对话树、翻译或改写版本放在同一分区。公开原始 test 不回流训练；自建测试在定稿后封存。
5. 检查与现有预训练 validation/test 的交叉。原预训练含 Wikipedia，Dolly 的部分上下文也来自 Wikipedia；不能未经检查声称完全没有知识或文本暴露。检查必须记录范围和遗漏，不能把精确去重等同语义去重。
6. 按最终回答监督 Token 控制中英接近 1:1，同时分别报告样本数、完整输入 Token、回答 Token 和任务分布，不预先假定各 1 万条就能平衡。
7. 对来源、任务、长度分层抽审并记录拒收原因；独立评测题优先逐题核对。自动清洗和抽审的实际覆盖率必须报告。
8. 模板、回答 Loss Mask、EOS、padding 与标签错位验证通过后才发布训练 manifest；当前研究目录没有可直接执行训练的 splits。

## 本地文件与后续下载范围

- [候选清单](../artifacts/research/sft_300m_20261026/source_catalog.json)：固定 revision、文件 SHA、来源白名单和待完成事项，`training_ready=false`。
- [长度与下载证据](../artifacts/research/sft_300m_20261026/summary.json)、[证据文件校验清单](../artifacts/research/sft_300m_20261026/evidence_manifest.json)。
- 研究原始缓存位于 `tmp/sft_research/`；已取得 Alpaca 中文全文件 35,100,559 bytes / 48,818 条、Dolly 全文件 13,085,339 bytes / 15,011 条，以及两个 CQIA 研究文件和 SmolTalk 少量 viewer 响应。原文件取得不等于全部可用于训练。
- SmolTalk 四个 train 分片合计约 922 MB；官方大小与 SHA 已记录，尚未全量下载。可先读取一个约 230 MB 的固定 train 分片，过滤后不足再扩展，不必下载 test 分片或整个大混合集。
- OASST2 尚未下载全量，中文质量筛选数量未核实。当前不需要为了这次来源研究扩容远程数据盘；后续远程制备仍检查实际剩余空间并保留约定水位。

当前研究文件全部位于本地 E:；没有把候选数据混入正在使用的预训练语料。下一步是按上述规则实现并审计正式 SFT 数据制备，然后才进行训练测速。
