# SFT中英候选储备：2026-10-27

已下载并逐文件核验15个固定版本原始文件，共1,346,228,580字节，约1.254GiB。

| 来源 | 文件数 | 原始记录数 | 适用部分与计数边界 |
| --- | ---: | ---: | --- |
| Chinese SmolTalk 八个任务文件 | 8 | 262,746 | 格式约束、改写、摘要、文档问答、编辑与短问答候选 |
| Smol-SmolTalk 四个训练分片 | 4 | 460,341 | 后续限既有四个新生成来源白名单；其余来源不默认入训练 |
| Dolly | 1 | 15,011 | 优先有给定上下文的抽取、摘要与阅读问答 |
| OpenAssistant OASST2 训练分区 | 1 | 128,575 | 这是消息数；后续先按父子关系重建中英文问答，保留树ID |
| Alpaca-GPT4 中文 | 1 | 48,818 | 合成数据；保留已知拒收行，非商业研究用途 |

**这是原始候选储备，不是新的可直接训练数据集。** 没有修改当前SFT发布版、latest_sft.json、冻结配置、权重、预算身份或正在运行的GPU队列。原始条数包括旧训练样本、可能的重复与尚未审核的回答；不能视为等量新增高质量问答。

## 存放与身份

远程储备根目录：`/root/autodl-tmp/model_training/artifacts/posttrain/sft_reserve_20261027`。
原始文件在raw/中，按来源及固定commit前缀分目录；source_cards/保存官方数据卡，Alpaca另保留DATA_LICENSE.txt。
reserve_config.json保存原始URL、完整revision、文件大小和SHA256；manifest.json标记training_ready=false，inventory.json记录实际Parquet/JSON行数与来源、评分分布。
冻结下载清单：[sft_reserve_20261027.json](../configs/data/sft_reserve_20261027.json)，SHA256：`c82ea98556730917dc18ecfc9d4b44950757afeea7d0342e812fd3576d3b58f4`。
远程下载记录：`/root/autodl-tmp/model_training/runs/assistant/data_prepare/sft_reserve_20261027_201653`，包含state.json、download.log、下载器快照和13项CPU/mock测试日志。

## 后续使用前的处理

- 原始来源按字节保留；中文进入训练前用现有官方OpenCC统一为简体，再去重与分词。
- 检查原材料是否齐全、参考答案是否支持、格式要求是否满足、回答是否完整；保留短任务，按现有32K BPE计数，不截断回答凑长度。
- Chinese SmolTalk的score是上游对首条指令的评分，不是参考答案正确性的金标。OASST的审核与rank也不能代替内容检查。
- 跨来源、现有SFT train/validation/test与预训练holdout检查重复或污染；同材料、对话树、翻译或改写版本按组隔离。不得把旧评测样本回流到训练中。
- 中英按回答监督Token配比，按任务和长度分层抽审；独立验收通过后才发布新的训练manifest。

## 来源与使用条件

- [Chinese SmolTalk官方卡](https://huggingface.co/datasets/opencsg/smoltalk-chinese)：保留Apache-2.0元数据及正文OpenCSG社区许可条件，不以页面顶部标签覆盖正文。当前仅作教学研究储备。
- [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)：当前下载完整训练分片，后续仅默认使用smol-magpie-ultra-short、smollm-rewrite-30k、smol-summarize-20k、smol-contraints四族；导入来源另保留原许可。
- [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)：CC-BY-SA-3.0。
- [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2)：Apache-2.0；仅下载train Parquet，不下载其validation。
- [Alpaca-GPT4原作者](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/80cda626ea305004be42426671c66efebbf22144)：数据CC-BY-NC-4.0，原作者限研究使用；不能用代码Apache许可覆盖数据限制。

Magpie中文与COIG-CQIA因本次官方卡未给出清晰数据许可，未纳入下载；Infinity-Instruct当前需要接受访问条件，未下载或绕过。

## 下载验证与资源

本地与远程各13项CPU/mock检查通过，覆盖精确Range续传、GitHub分段GET、服务端拒绝续传、损坏/超大内容、空间不足、路径穿越、现有文件与权重保留、停止信号。每个正式原始文件大小与SHA256均按冻结清单核验后才提交。
下载进程CUDA不可见，最多两个CPU核、低CPU优先级、8MiB/s下载上限，磁盘保留24GiB。下载和统计完成不证明所有参考答案正确，也不说明训练效果。
完成复查时空闲33.53GiB，训练源码指纹仍为`0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`。

逐文件实际行数：

| 原始文件 | 行数 |
| --- | ---: |
| `smoltalk_chinese/format-constrain.parquet` | 38,434 |
| `smoltalk_chinese/rewrite.parquet` | 23,795 |
| `smoltalk_chinese/summary.parquet` | 26,687 |
| `smoltalk_chinese/document-qa.parquet` | 66,441 |
| `smoltalk_chinese/editing.parquet` | 72,696 |
| `smoltalk_chinese/everyday.parquet` | 7,388 |
| `smoltalk_chinese/information-seeking.parquet` | 16,881 |
| `smoltalk_chinese/brainstorming.parquet` | 10,424 |
| `dolly/databricks-dolly-15k.jsonl` | 15,011 |
| `oasst2/data/train-00000-of-00001-88ba0162028a73fc.parquet` | 128,575 |
| `smol_smoltalk/data/train-00000-of-00004.parquet` | 115,086 |
| `smol_smoltalk/data/train-00001-of-00004.parquet` | 115,085 |
| `smol_smoltalk/data/train-00002-of-00004.parquet` | 115,085 |
| `smol_smoltalk/data/train-00003-of-00004.parquet` | 115,085 |
| `alpaca_gpt4_zh/data/alpaca_gpt4_data_zh.json` | 48,818 |
