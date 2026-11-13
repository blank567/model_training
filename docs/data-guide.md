# 数据来源、构建与验收

本手册合并预训练、通用 SFT 和阅读后训练的数据要求。所列版本和数量描述已经完成的实验；重建时须取得相同输入并重新验收，不能把历史路径或哈希当作已下载数据。实际模型成绩见[实验结果](experiment-results.md)，训练入口见[运行手册](remote-runbook.md)。

仓库只分发原创 fixture、处理代码与脱敏汇总，不分发真实语料、Tokenizer 产物、权重或完整逐题预测。新建后训练实验只训练一轮：`epochs=1`、`stop_epochs=null`；数据准备本身不启动模型训练。

## 1. 公共数据契约

每份正式发布至少保存：上游名称与链接、不可变 revision 或发行文件 SHA256、原文件与行号、原文/规范化文本身份、分区/来源组、Tokenizer 身份、Token 数、过滤原因、许可及独立验收结果。

- 先清洗、归组和分区，再仅用 train 拟合预训练 Tokenizer。后训练复用原 32K BPE，不更换词表。
- 中文统一简体，转换后重新检查答案、去重和编码；中英配比按有效 Token 或回答监督 Token 控制，不能用样本数各半替代。
- 来源组、文章、材料、精确问题及已声明的近重复检查共同约束隔离。精确/有界近重复检查不证明不存在语义改写、翻译副本或预训练污染。
- 正式输出默认拒绝覆盖；输入版本、规则或 Tokenizer 改变时新建发布目录。带 `SUPERSEDED.json` 或隔离标记的候选不作训练入口。
- 下载保持 TLS 校验，按固定版本核对大小和 SHA。缓存命中也须校验，不用 `main` 代替已冻结 revision。

### 离线验证

项目根目录、已经安装依赖的 Python 环境中执行：

```bash
python -m llm_lab.data.acquire --config configs/data/acquire_smoke.json
python -m llm_lab.data.prepare --config configs/data/prepare_smoke.json
python -m llm_lab.cli pretrain --config configs/pretrain/smoke.json --validate-only
```

fixture 是仓库原创少量文本，产物进入 `examples/_outputs/`；它只能验证接口、分区、BPE 和训练闭环，不能证明真实语言质量。Windows 临时目录、UTF-8 环境和安装方式见[项目 README](../README.md)。

## 2. 预训练：约 3.1B Token 语料池

### 固定来源

| 来源 | 固定版本 | 用途与许可边界 |
| --- | --- | --- |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | `20231101.en` / `20231101.zh`；revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | 原约 1.1B Token；保留文章来源和适用署名/共享条件 |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | `cmn_Hani`；revision `af9c13333eb981300149d5ca60a8e9d659b276b9` | 新增中文网页；源卡标注 ODC-By 1.0，原网页权利仍须分别考虑 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `sample-10BT`；revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 新增英文网页；同样不能把来源数据库许可解释为重新授权全部网页 |

首分片有限实验使用 `configs/data/acquire_wikipedia.json`；约 1.1B 初始语料使用 `acquire_wikipedia_1b.json` / `prepare_wikipedia_1b.json`。扩容的显式分片、大小和 SHA 记录在 [expand_bilingual_3b.json](../configs/data/expand_bilingual_3b.json)。获取 JSONL 的文档/字节上限不是远程完整 Parquet 下载量上限，必须另留缓存和临时空间。

### 处理与存储

1. Unicode/空白规范化、长度/单字符重复/重复行过滤；正文与源 ID 精确去重。
2. 中文采用官方 OpenCC 1.4.2 的有界固定点 t2s；按文档隔离 train / validation / test。
3. 在原 train 拟合 Byte-level BPE，词表 32,000；后续扩容复用该 Tokenizer，每篇末尾追加 EOS。
4. 扩容排除 Wikipedia/Wikimedia 镜像域名，按规范 URL、原文及简体正文 hash 对照旧语料去重。
5. 旧验证/测试额外使用去空白、casefold 后的任意偏移 100 字符指纹；命中再核真实子串。该规则不覆盖所有语义近重复。
6. 增量写入使用 `.part`、fsync 与 SQLite 提交位置；恢复只回到已提交边界。合并发布保留旧 train 前缀、Tokenizer 和旧 holdout 身份。

输出为各分区、各语言的 little-endian uint32 Token 流，另有 manifest、审计和偏移索引。发布验收逐篇核对连续偏移、词表范围、EOS 与来源唯一性，并抽样解码核对正文 hash。

| 语言 | 原训练 Token | 新增 Token | 最终训练 Token |
| --- | ---: | ---: | ---: |
| 英文 | 549,999,999 | 1,000,000,429 | 1,550,000,428 |
| 简体中文 | 549,999,984 | 1,000,002,609 | 1,550,002,593 |
| 合计 | 1,099,999,983 | 2,000,003,038 | **3,100,003,021** |

历史正式 manifest 为 `artifacts/pretrain/bilingual_3b/manifest.json`，SHA256 为 `f1dbbba35db3f414a36359f96c3a8a6a3b4bba1fb94b09284a5a0fef75c6c7b1`。303M 实际累计处理 3,000,008,704 Token，采用随机有放回窗口采样；不能解释为全部独立位置恰好遍历一次。

已有父语料并核对输入后，扩容入口为：

```bash
python scripts/expand_corpus.py --config configs/data/expand_bilingual_3b.json --validate-only
# 确认新输出位置和空间预算后才实际制备。
python scripts/expand_corpus.py --config /path/to/your_expansion_config.json
python scripts/publish_expanded_corpus.py --output /path/to/your_expansion_output
```

扩容 manifest 本身不含全部旧验证/测试引用；必须通过 publisher 合并验收，才能用作完整训练 manifest。不能将下载完成或增量完成等同于正式发布。

## 3. 通用指令 SFT：20k 到 100k

### 来源和固定身份

| 来源 | 固定 revision | 筛选与许可 |
| --- | --- | --- |
| [Alpaca-GPT4 中文](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | `80cda626ea305004be42426671c66efebbf22144` | 合成参考回答；**数据 CC BY-NC 4.0**，代码 Apache-2.0 不替代数据限制 |
| [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc` | 指定新生成来源白名单；其 Apache-2.0 不覆盖导入来源的其他许可 |
| [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | 仅有完整给定上下文的任务；CC BY-SA 3.0 |
| [Chinese SmolTalk](https://huggingface.co/datasets/opencsg/smoltalk-chinese) | `5edf5d0d7b06794755cf78b274a150a4efe1c9d4` | 指令评分≥4及内容过滤；保留源卡的 Apache 元数据和 OpenCSG Community License 附加条件 |
| [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | 同树同语言首轮问答，审核/质量门槛；本发布仅收逐条内容抽审通过部分；Apache-2.0 |

文件 SHA、明确下载路径及来源规则在 [SFT 源配置](../configs/data/prepare_sft_300m_20261026_164414.json)和[扩充储备配置](../configs/data/sft_reserve_20261027.json)。许可不完整、门控未接受或仅为候选观察的来源不自动进入训练。

### 筛选和分区

- 复用原 32K BPE，先转简体再去重；拒收截断、缺材料、明显约束违例、超长、语言不符及当前短任务范围外的专业/代码/复杂数学回答。
- 按精确问题、材料、来源组和有界近重复分量归组；同组只留代表，并保护全部已有评测分区。
- 扩充使用有界 5 字符 shingle / Jaccard 去重，阈值 0.8；对已知 holdout 执行 100 字符任意偏移检查，不声称全语义隔离。
- 按回答监督 Token 平衡中英。自动质量规则、LLM 抽审、全量结构验收是不同证据；LLM 抽审不是专家金标，其疑点比例不能当作全量错误率。

| 发布 | train | validation | 主 test | 完整输入 / 回答监督 Token |
| --- | ---: | ---: | ---: | --- |
| 初版正式 `20261026_170531` | 19,991 | 995 | 1,000 | 4,392,427 / 3,525,424（train） |
| 扩充 50k | 50,000 | 复用 995 | 复用 1,000 | 11,184,083 / 6,090,534 |
| 扩充 100k | 100,000 | 复用 995 | 复用 1,000 | 21,761,826 / 10,553,126 |

50k / 100k 保留原 19,991 条训练文本，且 50k 是 100k 子集；pilot 2,000 来自 train。扩充另有独立辅助开发 2,000、最终测试 1,000，不与主验证/主测试混称。训练中英回答 Token 约各半，但样本数量并不各半。

历史主测试曾用于配置选择，已经不是完全未见的最终测试；后续辅助最终测试单独冻结。完整评测与已暴露测试的使用边界见[实验结果](experiment-results.md)。

### 模板与 Loss Mask

```text
System:
{可选 system}

User:
{prompt}

Assistant:
{response}{EOS}
```

无 system 时省略整个 System 段。prefix 与 response **分开编码后拼接**，避免 BPE 跨边界合并。PAD=0、UNK=1、EOS=2；prefix/padding 的 mask 为 0，回答与 EOS 为 1。训练层只因果移位一次：`input_ids[:, :-1]` 预测 `input_ids[:, 1:]`，监督使用 `loss_mask[:, 1:]`。

首轮正式数据完整序列≤512；扩充完整序列≤1024、回答含 EOS≤512，优先短而完整的样本，不截断回答后伪造结束标签。与 20k 比较时数据量、长度和任务覆盖同时变化，不能当作只增加条数的纯单变量实验。

### 构建入口

先复制配置并更换自己新建的输出目录和输入身份；以下名称仅标明代码入口，不覆盖历史发布：

```bash
python scripts/run_sft_data_build.py --config /path/to/your_sft_build_config.json
python scripts/publish_sft_curated.py --config /path/to/your_review_publication_config.json
python scripts/audit_sft_data.py --data-dir /path/to/your_sft_release
python scripts/prepare_sft_expansion.py --help
python scripts/curate_sft_expansion_pool.py --help
python scripts/build_sft_expansion_eval.py --help
```

扩充先 scan、内容抽审，再整理/发布及独立验收；不是只执行下载后直接训练。每份数据验收绑定文件 SHA、数据身份、重新 BPE 编码、Mask/EOS、简体固定点、分区隔离和 pilot 子集关系。

## 4. 阅读专项 SFT 与 RL

任务是**给定完整材料，输出原文答案片段**，不是开放知识问答或长推理链。阅读 SFT 从通用 Full SFT100k 接着训练，之后三种 RL 算法独立从同一阅读 SFT best 初始化。

### 阅读专项 50k

| 来源 | 固定版本 | 训练条数 | 保留的许可说明 |
| --- | --- | ---: | --- |
| [CMRC2018](https://github.com/ymcui/cmrc2018) | `c0eb1b6ba219847457e6af3180da722bbeb656af` | 6,556 | CC BY-SA 4.0 |
| [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) | `b944790de5af02c5fbb7cd9cb1473d27d169eebf` | 17,450 | CC BY-SA 3.0；原文/问题/答案共同繁转简 |
| [SQuAD 1.1](https://huggingface.co/datasets/rajpurkar/squad) | `7b6d24c440a36b6815f21b70d25016731768db1f` | 25,994 | CC BY-SA 4.0；未混入 2.0 无答案题 |

正式身份为 `reading_qa_50k_zh_en_20261101_v3`：train 50,000，中文 24,006 / 英文 25,994，答案+EOS Token 为 164,430 / 164,433；全轮输入 14,988,172 Token。短答案监督量较小，不能将条数解释为长推理数据量。

- 原始答案偏移先验证，再共同转简并重建/复查跨度。此发布固定 **OpenCC 1.1.9**，使用隔离工具目录；不同于预训练/通用 SFT 的 1.4.2，不要替换全局环境后冒充相同数据身份。
- 提示≤960 Token，参考答案含 EOS≤64；保留完整上下文，整序列≤1024。提示不计 Loss，答案及 EOS 监督。
- 按规范化文章标题和段落连通归组；训练、验证、测试隔离已知文章、材料与精确问题。
- validation 2,000 来自官方 train 留出的文章；最终 test 为 CMRC dev 500、DRCD test 500、SQuAD validation 1,000。它是自定义固定子集，不是隐藏排行榜。
- RL 选型使用固定 256 题开发子集；2,000 题阅读最终测试单独报告，不把短测/开发正确率冒充最终测试。

读取和验收代码为 [llm_reading_data](../src/llm_reading_data/)；数据来源 catalog 与验收材料须由自己的固定源文件准备：

```bash
python scripts/fetch_reading_sources.py --help
python scripts/prepare_reading_alignment.py --cache /path/to/source_cache \
  --tokenizer /path/to/tokenizer.json --output /path/to/new_release --work /path/to/work
python scripts/audit_reading_alignment.py --manifest /path/to/new_release/manifest.json \
  --cache /path/to/source_cache
```

上面的 audit 默认仅验收，不发布；完成代表性内容抽审并取得绑定该数据身份的回执后，才显式添加 `--publish --review /path/to/review_receipt.json`。专项训练器要求 manifest 为 `published`，未发布的候选会被拒绝。

### 三组 RL 数据与奖励

| 条件 | 在线 RL prompt 数 | 与专项 SFT 的关系 | DPO 合格偏好对 |
| --- | ---: | --- | ---: |
| `old_matched` | 25,000 | 原 50k 的固定已见子集 | 20,462 |
| `new_matched` | 25,000 | DuReader-robust / Quoref 新来源，按已知文章/材料/问题隔离 | 20,462 |
| `old_full_50k` | 50,000 | 原专项 SFT 全部训练题复用 | 40,871 |

新组正式版本为 `reading_rl_independent_25k_20261102_v5`，中文 11,336 / 英文 13,664，答案+EOS Token 为 55,246 / 56,360；中文约 49.50%。旧 v4 因内容抽查问题隔离，不能作为等价新组。新组排除临床建议/聚合及主观推荐，再按答案长度平衡，不使用模型得分选题。

新来源固定的是**官方发行文件 URL 与内容 SHA，而非 Git commit**：DuReader-robust `train.json` SHA256 为 `137e65191e2f0895d0b3ae8a2af2387b2aa2342642476a616287e150740016ce`；Quoref `train.json` 为 `5f974478561ee3c7e5d8c99773472741f6c382543358470afb3797dd0367c50d`。上游为 [DuReader](https://github.com/baidu/DuReader) 和 [Quoref](https://allenai.org/data/quoref)；正式 manifest 保留发行 URL 和来源行号。内容 SHA 不等于完整许可证明，重新分发前仍须核对上游条件。

PPO / GRPO 使用人工答案别名严格匹配的 0/1 奖励；可能漏掉合理同义答案，不能把 reward=0 都当成语义错误。整段复制和解释性长答不符合“只输出答案跨度”的任务。自定义 F1 仅作重合度诊断，不等于官方 SQuAD F1。

DPO 的 chosen 为原人工答案，rejected 由同一初始 SFT 在**训练题**上采样筛选；排除空答、缺 EOS、别名/包含关系及高 F1 近答案。原文其他片段和非跨度回答分别标注负例类型；格式负例不等于人工证明语义错误。这是规则派生偏好，不能称为人工成对偏好。没有可靠负例就保留缺口，不重复凑数。

准备代码为 [llm_reading_independent](../src/llm_reading_independent/) 和 `scripts/prepare_reading_comparison.py`；比较计划须绑定源 manifest、共同起点 SHA 和 `--ppo-updates 2`。DPO 构造入口 `scripts/prepare_reading_dpo_pairs.py` 会占 GPU 锁并实际生成，**不是纯 CPU 下载命令**。新实验输入审计、奖励校准通过后才显式执行。

三算法各遍历训练题一轮；PPO 每批 rollout 两次内部优化，GRPO 每题采样四个回答。等题数/等一轮不等于等 Token、难度或算力。旧新来源分布、SFT 可见性和最终测试来源不同，报告须披露这些差异。

## 5. 候选、许可与最终验收

数学候选包含 NuminaMath、CMM-Math 等公开来源及历史合成算术，但质量、中文供应与奖励校准不足，未形成本文已完成的九组 RL 任务；保留候选代码不等于已完成数学推理训练。早期 MASSIVE / Qwen 工作流也不是当前自研 303M 结果来源。

正式放行前核对：源文件固定身份；Tokenizer/特殊 Token 与模型兼容；原始及转简跨度正确；重新编码与 mask/EOS 一致；有效样本与 Token 数达标；精确问题/材料/已知来源组隔离；抽审覆盖与拒收记录；验收报告绑定当前 `data_identity`。测试只进入隔离索引，不进入负例构造或奖励规则拟合。

混合来源保留各自许可，不统一改标为 MIT、Apache 或“可商用”。上游数据许可、依赖代码许可和潜在模型权重分发条件分别处理；研究用数据允许下载不代表允许任意重新分发。完整结构验收与有限抽审均不能保证所有参考回答事实正确、无隐私或完全无污染。
