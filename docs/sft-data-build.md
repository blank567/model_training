# 自研 300M 的首轮 SFT 数据构建

本次构建使用固定版本的 Alpaca-GPT4 中文、Smol-SmolTalk 指定新生成来源，以及 Dolly 有给定上下文的任务。来源许可与抽样研究见 [数据筛选](sft-data-selection.md)。中文数据限原作者声明的非商业研究用途；整体数据不应被重新标为单一 Apache 许可。

## 当前版本与质量边界

首版 `20261026_133649` 通过文件、分词、Loss Mask 和分区结构检查，但后续 LLM 抽审发现截断、缺上下文摘要、错误代数和统计回答。因此保留为问题归档，带有 `SUPERSEDED.json`，不作为正式训练入口。

第二版 `20261026_140730` 又在抽审中发现了缺参考材料的转换任务、回答中出现未核实数学公式、范围之外的编程回答，亦保留废弃标记。

源构建版时间戳为 `20261026_164414`，对应配置 [prepare_sft_300m_20261026_164414.json](../configs/data/prepare_sft_300m_20261026_164414.json)。对它的 40 条 train/validation 分层 LLM 抽审中，另有 14 条存在疑点或需要专业复核，按冻结清单隔离，测试文件按字节保留。

**正式发布版为 `20261026_170531`**，位于 `artifacts/posttrain/sft_300m_20261026_170531/`。抽审发布配置见 [curate_sft_300m_20261026_170531.json](../configs/data/curate_sft_300m_20261026_170531.json)，当前入口记录在 `artifacts/posttrain/latest_sft.json`。

| 分区 | 样本数 | 完整输入 Token | 回答监督 Token（含 EOS） |
| --- | ---: | ---: | ---: |
| train | 19,991 | 4,392,427 | 3,525,424 |
| validation | 995 | 218,235 | 175,289 |
| test | 1,000 | 219,898 | 180,176 |
| pilot，来自 train | 2,000 | 439,621 | 353,193 |

训练含中文 12,918 条、英文 7,073 条；回答监督 Token 的中文占比约 50.011%，全部完整输入不超过 512 Token。实际训练和验证分别比目标少 9 条、5 条，没有为补足条数加入未经复核的替代样本。

本地与远程独立发布得到同一数据身份：`b85ba15fe4ccdb35e796a5e4a1767f07ea918c35ab0d2c85ac2ca61a432cef6a`。64 项 CPU 测试及正式数据全量结构验收通过；原预训练源码树在操作前后身份相同。最终 [manifest](../artifacts/posttrain/sft_300m_20261026_170531/manifest.json)、[验收](../artifacts/posttrain/sft_300m_20261026_170531/acceptance.json)、[发布凭据](../artifacts/posttrain/sft_300m_20261026_170531/publication_receipt.json)均已保留。

质量判断分三层：固定来源与文件身份、自动规则和分区检查、train/validation 的分层 LLM 抽审。LLM 抽审不是人工专家金标；结构验收也不证明全部公开回答的事实正确。独立 test 的题目不参与本次语料选择反馈。

## 构建流程

1. 下载固定 commit 的源文件，检查大小与 SHA256；缓存仅写 `.cache/data/sft_300m/`。读取原始 train，公开 test 分片不进入训练。
2. 核对预训练 manifest、原 BPE 及 91 篇冻结 holdout 正文的身份。先按旧文本规则核原 SHA，再用本轮统一规范作污染检查。
3. 使用官方 OpenCC 1.4.2 C++ 绑定的有界固定点繁转简，并进行 NFKC/空白规范化。保留来源、原行号、原始 hash 和转换后 hash。
4. 拒收明显截断、重复、代码/复杂数学、当前范围之外的专业建议、年份事实、缺原材料摘要、无依据外部执行声明、语言不符和超长回答。来源或长度符合条件不能代替内容检查。
5. 比较预训练 validation/test 中任意偏移的连续 100 个规范化字符；滚动 hash 命中后再确认真实子串，避免 hash 碰撞造成错误结论。此检查不等于语义、翻译或知识层面的完全隔离。
6. 按精确问题、非空原材料、来源组和有界字符近重复分量归组，每组只保留一个代表，然后划分 train/validation/test。近重复检查有桶上限及长文本边界，不声称完整语义去重。
7. 按回答监督 Token 平衡中英；训练 20,000 条，validation/test 各 1,000 条。pilot 从 train 中取 2,000 条，不是额外独立测试。
8. 独立验收所有输出：文件 hash、重新 BPE 编码、简体固定点、回答 mask、EOS、ID/group/精确问题和材料跨分区隔离，以及 pilot 子集关系。报告绑定 `data_identity`。

## Token 与训练适配契约

复用现有 32k BPE：PAD=0、UNK=1、EOS=2，不增添新的角色特殊 Token。

有 system 时使用：

```text
System:
{system}

User:
{prompt}

Assistant:
```

无 system 时省略整个 System 段。**prefix 与 response 分开编码后拼接，再追加一个 EOS**，避免 BPE 跨边界合并造成 mask 错位。生成端必须采用同一 prefix 协议。

- `input_ids` 是完整 prefix + response + EOS。
- `loss_mask` 与 `input_ids` 同位置，prefix 为 0，response 和 EOS 为 1。
- 该文件还没有执行因果错位。训练适配层必须只错位一次：用 `input_ids[:, :-1]` 预测 `input_ids[:, 1:]`，对应监督位为 `loss_mask[:, 1:]`。
- 动态 padding 的补齐位置必须为 PAD，且不参与 Loss。梯度累积应明确回答 Token 的归一化方式，避免把长短样本简单当作等量监督。
- 原自研 Decoder 的 labels 接口和 HF Trainer 的自动错位不同，接入时必须有实际测试；不能直接把旧 MASSIVE 配置的模型路径换成 `best.pt`。

每条样本包含正文、Token、mask、语言、来源、固定 revision、原文件及行号、原始和规范化文本 hash、分组 ID、许可证。任务类别只是基于文本的启发式标签，用于抽审和统计，不当成任务金标。

## CPU 运行与复现

构建入口和独立验收位于 `scripts/`，没有修改 `src/llm_lab`，以保护正在运行的预训练恢复身份。

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

# 新数据输出目录必须不存在；重复构建应复制配置并更换时间戳。
.venv/bin/python -u scripts/run_sft_data_build.py \
  --config configs/data/prepare_sft_300m_20261026_164414.json

# 源构建完成后，按照冻结抽审清单生成独立发布目录。
.venv/bin/python scripts/publish_sft_curated.py \
  --config configs/data/curate_sft_300m_20261026_170531.json
```

只复核已有数据，不重新构建：

```bash
.venv/bin/python scripts/audit_sft_data.py \
  --data-dir artifacts/posttrain/sft_300m_20261026_170531 \
  --output artifacts/posttrain/sft_300m_20261026_170531/acceptance_recheck.json
```

下载时若需平台加速，仅在获取子 shell 中加载 `/etc/network_turbo`，指定已验证的系统 CA `/etc/ssl/certs/ca-certificates.crt`。不关闭 TLS 校验。下载失败可从本地 E: 的已核验缓存传到同一远程缓存路径，再由构建程序重新验 SHA。

监督器在远程最多使用两个允许的 CPU 核，CUDA 不可见；记录运行快照、源文件身份、测试、日志、过滤原因与停止状态。磁盘保留 8 GiB 水位，输出目录存在时拒绝覆盖。它只准备数据，不启动 SFT 或改变预训练预算记录。

## 验证与下一步

本轮数据规则、标签边界、holdout 规范化和验收工具测试由真实 CPU 测试支持；全量数据的验收结果另存于数据目录。训练速度、显存和任务效果需要 SFT 适配完成后的真实 GPU 短测，不能由这些数据测试替代。

本次发布已经绑定正式验收报告；上述独立重建命令默认生成待验收版本，不能将仅完成 `publish_sft_curated.py` 的输出当成已验收数据。最终身份、文件 SHA 和验收报告必须一致。原始参考回答没有经过全量人工专家核验；分层抽审刻意覆盖小来源与不同任务，其疑点比例不能当作全量错误率。

正式训练实验从同一个完成预训练的 Base 开始。pilot 和正式训练应分别记录；用 pilot 调整配置后，正式实验重新从 Base 启动，独立测试只在方案冻结后评测。
