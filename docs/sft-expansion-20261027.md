# 中英SFT数据扩充与整理：2026-10-27

全量处理15个固定版本原始文件。第一阶段自动过滤和有界近重复归并后保留338,191条；对64条分层候选做LLM内容抽审，排除34条疑点，并将能确定的错误模式加入第二阶段全量复查。最终保留327,209条新候选。

扩充版本保留原19,991条训练样本，逐级追加新候选；主验证995条、主测试1,000条和pilot2,000条按字节复用。较小训练版本是较大版本的子集。原始储备与原SFT发布版、旧训练配置和模型权重保留。

| 版本 | 训练条数 | 新增条数 | 完整输入Token | 回答监督Token，含EOS | 中文监督Token占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sft_50k | 50,000 | 30,009 | 11,184,083 | 6,090,534 | 50.00% |
| sft_100k | 100,000 | 80,009 | 21,761,826 | 10,553,126 | 50.00% |

## 文件入口

远程根目录：`/root/autodl-tmp/model_training/artifacts/posttrain/sft_expansion_20261027_215627`。

- `sft_50k`训练入口：`/root/autodl-tmp/model_training/artifacts/posttrain/sft_expansion_20261027_215627/sft_50k/manifest.json`；[本地manifest副本](E:/codex/model_training/runs/assistant/data_prepare/sft_expansion_20261027_215627/manifests/sft_50k/manifest.json)。
- `sft_100k`训练入口：`/root/autodl-tmp/model_training/artifacts/posttrain/sft_expansion_20261027_215627/sft_100k/manifest.json`；[本地manifest副本](E:/codex/model_training/runs/assistant/data_prepare/sft_expansion_20261027_215627/manifests/sft_100k/manifest.json)。

候选池：`/root/autodl-tmp/model_training/artifacts/posttrain/sft_expansion_20261027_215627/candidates.jsonl`，仅作后续筛选储备；pool_manifest.json记录来源、任务分布、过滤原因和候选身份。
新发布清单：`/root/autodl-tmp/model_training/artifacts/posttrain/sft_expansion_20261027_215627/release.json`；新指针为`artifacts/posttrain/latest_sft_expansion.json`，旧latest_sft.json保持原样。

辅助评测位于auxiliary_evaluation/：从两份训练集与全部内容抽审都未使用的余量中冻结，额外提供2,000条验证及1,000条测试。原主验证/主测试继续保留；辅助集补充新增任务覆盖，参考答案仍不是专家金标。辅助测试没有进入内容抽审或超参选择。

## 清洗、去重与任务配比

- 中文采用原项目官方OpenCC 1.4.2的有界固定点t2s；先统一简体和Unicode/空白，再计算身份及分词。
- Chinese SmolTalk要求指令评分至少4；缺评分、低评分及明显范围外的难任务隔离。该评分不能证明回答事实正确。
- OASST按message_id/parent_id重建同树、同语言的首轮问答，要求两端审核通过、未删除及质量评分门槛，按合格同胞回答的rank选择；第二阶段只收录本轮逐条内容抽审通过的OASST样本，其余保留在原始储备。不把消息数当作问答数。
- 保留原Smol-SmolTalk四来源白名单及Dolly有上下文任务要求，延续旧抽审14条拒收记录；保留原来源许可。
- 本批供研究学习使用，混合来源保留各自许可及源卡附加条件，不能把整体视为统一的商业授权；来源条款见sft-reserve-20261027.md。
- 拒收缺原材料、疑似截断、重复、明确格式约束违例、超长、语言不符，以及当前短任务范围外的代码、复杂数学和专业建议等。范围或长度拒收不等同答案错误。
- 精确来源行、问题、材料和长回答对照旧SFT全部分区；主评测材料/答案与预训练holdout做100字符任意偏移子串检查，hash命中后确认真实子串。旧测试只进入自动隔离索引，不进入抽审清单。
- 对保护集和新候选共同进行有界5字符shingle/Jaccard近重复分组，剔除与旧分区同组的新候选，每个新重复分量只保留一个代表。此方法不保证语义改写、跨语言翻译的完全隔离。
- 按任务缺口追加摘要、改写、格式遵循、给定材料问答等，按回答监督Token平衡中英。旧训练集固定保留，因此最终任务比例受其组成及候选供应量影响，目标权重不是精确配额。

第二阶段修正了classification名词误当分类操作、information中的子串误当格式要求，以及extracted误当文本抽取等任务标签。新增候选仅修改task；原文本、Token、Loss Mask和身份字段保持原样。旧19,991条基准正文及其历史任务标签完整保留，因此任务统计仍属规则标签，不是人工金标。

新增池中的分类仅316条、抽取仅366条；没有为满足任务权重而误标问答或重复填充。这两类后续仍需要有针对性地补数据。

全量硬约束复查包含可明确解析的英文关键词完整词频、单词数、项目符号数量、占位符和关键词、大小写及P.S.要求，以及明确中文逐项字数范围。词数和字数采用pool_curation.json中的固定口径，未解析到的自然语言约束不声称已验证。另隔离缺少已获取原文的URL任务、角色串入及范围外的脚本/专业建议。

## 分词与训练适配

沿用32K BPE、原聊天前缀、PAD=0/UNK=1/EOS=2；不改词表或模型结构。prefix与回答分别编码后拼接，回答及EOS监督，prefix和padding不计Loss，因果移位仅执行一次。完整序列硬上限1,024 Token、回答含EOS上限512，优先选择完整序列不超过512的候选，不截断回答凑长度。

本批数据的样本长度和任务组成与旧2万条不同。旧2万与新版本的比较同时包含数据量和任务覆盖变化，不能当作只改变样本数的纯单变量实验。

## 验收、抽审与复现

远程20项数据流程CPU检查通过；本地19项通过，OASST Parquet用例因本机未装PyArrow跳过，该用例在远程已实际通过。第二阶段整理14项、辅助评测隔离2项检查在本地及远程通过。
每份发布数据经独立全量重编码、Mask/EOS、简体固定点、文件摘要、精确分区隔离和pilot子集验收；再由正式SFTCorpus读取train/validation/pilot，核对旧训练正文保留、嵌套关系、候选来源与抽审排除。全部最终候选再次执行第二阶段规则，并与归档的第一阶段逐条核对除task外的所有字段。辅助评测逐条重编码，并检查与训练、抽审及彼此的隔离。
有限LLM抽审共64条，排除34条；分层抽样不是全量错误率估计，结构验收不证明所有参考答案语义正确。

运行记录：`/root/autodl-tmp/model_training/runs/assistant/data_prepare/sft_expansion_20261027_215627`，保留不可变runtime快照、输入配置、progress、过滤理由、抽审决定、发布与独立验收报告。
本次可重建candidates.sqlite在发布及辅助验收通过后清理，清理收据保留大小与SHA256；原始数据、两个阶段的候选JSONL、正式数据和所有权重保留。
模型训练源码指纹仍为`0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`。本次仅CPU数据整理，没有启动新GPU训练。

## 各版本任务分布

| 任务 | 50000 | 100000 |
| --- | ---: | ---: |
| classification | 705 | 705 |
| extraction | 834 | 834 |
| format_following | 6,538 | 19,436 |
| grounded_qa | 6,534 | 16,252 |
| rewriting | 8,711 | 21,669 |
| short_qa_or_other | 17,960 | 19,436 |
| summarization | 8,718 | 21,668 |

## 第二阶段全量复查隔离原因

| 原因 | 条数 |
| --- | ---: |
| outside_bilingual_script_curriculum | 6,826 |
| required_keyword_missing | 1,606 |
| required_keyword_frequency_violated | 654 |
| oasst_not_individually_reviewed_for_this_release | 621 |
| required_postscript_missing | 606 |
| explicit_placeholder_budget_violated | 176 |
| explicit_word_budget_violated | 158 |
| specialist_health_claims_not_verified | 151 |
| user_prompt_contains_assistant_role_or_answer | 63 |
| current_recommendations_not_verified | 45 |
| documented_content_review_exclusion | 34 |
| external_material_not_fetched_or_verified | 30 |
| technical_execution_outside_initial_curriculum | 10 |
| explicit_bullet_budget_violated | 2 |

第一阶段池SHA256：`e672a9609a2c89bf8bb5f910b508b20f1171049024ec376219b7bb2b962f6616`。最终池SHA256：`c798696d47b88a2d1fed7dd3fb878ef3f6748dbe32844b64d4b810d83d580bc3`。第二阶段脚本SHA256：`4f6bbb3cc614d85a5c42fb908ce69e9ba932ccc2fed9e215b373fe220e8663ab`。
