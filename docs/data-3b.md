# 约 3.1B 中英语料扩容

2026-10-24 **21:16:35（北京时间）已完成远程 CPU 制备、校验与发布**，总耗时约 81.33 分钟。正式入口为 `artifacts/pretrain/bilingual_3b/manifest.json`，训练语料共 **3,100,003,021 Token**；原有 1,099,999,983 加新增 2,000,003,038 Token。没有裁切整篇文档、补齐或用重复采样凑容量。

| 语言 | 原有 train Token | 新增 train Token | 最终 train Token |
| --- | ---: | ---: | ---: |
| 英文 | 549,999,999 | 1,000,000,429 | 1,550,000,428 |
| 简体中文 | 549,999,984 | 1,000,002,609 | 1,550,002,593 |
| 合计 | 1,099,999,983 | 2,000,003,038 | 3,100,003,021 |

正式 manifest SHA256 为 `f1dbbba35db3f414a36359f96c3a8a6a3b4bba1fb94b09284a5a0fef75c6c7b1`。本次下载 3 个已校验 Parquet，共 9,146,666,208 bytes；编码新增 1,831,451 篇文档。发布验收逐篇检查偏移/EOS/词表/正文与来源唯一性，并对英文 182、中文 185 篇抽样解码核对正文 hash；原 train 前缀和验证/测试流哈希保持相同。另用真实 `BinaryCorpus` 在 CPU 读取中英 train/validation，各得到 `[4,1024]` 的输入和目标。远程验收报告位于 `runs/data_3b_prepare/final_verification.json`，本地副本位于 `tmp/data_3b/final_verification.json`。验收时剩余磁盘约 20.85 GiB，这是运行时快照。

当前 100M GPU 训练仍使用原有 `wikipedia_1b`，不改训练进程、源码、冻结配置或 manifest。300M 候选参见 [300m-3b.md](300m-3b.md)，数据准备不会启动新的 GPU 训练。

## 数据来源

现有固定 Wikipedia 快照的中文 6 个分片已用完。新增中文使用 [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/af9c13333eb981300149d5ca60a8e9d659b276b9/README.md) 的 `cmn_Hani`，英文使用 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md) 的 `sample-10BT`。中文配置名不代表全部简体，仍用官方 OpenCC 1.4.2 的同一有界固定点 t2s 规则。

两仓固定 revision、显式分片路径、大小和 SHA256 见 `configs/data/expand_bilingual_3b.json`。官方标注 ODC-By 1.0，原始网页内容同时受 CommonCrawl 条款及来源站点相应权利约束；保留仓库、版本、分片、原网页 URL 与归属，不等同于重新许可全部网页。

## 处理与隔离

1. 通过平台已有网络加速访问官方 HF；官方 HEAD 核对固定身份，签名文件地址只留内存。TLS 验证开启，支持精确 Range 恢复，分片完成后校验大小和 SHA256。
2. SQLite 种入旧语料全部审计 source ID、原始与简体正文 hash。新数据保留原 URL，排除 Wikipedia/Wikimedia 系域名，按规范 URL 和正文 hash 去重。
3. 规范化→t2s→规范化，中文逐篇检查幂等；过滤长度、单字符重复和重复行。用现有 32K BPE 编码，每篇加 EOS，不重新训练 tokenizer。
4. 对旧实际验证/测试正文额外构造去空白、casefold 后的每字符滑动 100 字符指纹，拒绝包含这些片段的新网页。该方法不能识别所有语义改写、所有近重复或同 URL 的全部历史版本，不作完全无污染保证。
5. 每个分片先写 `.part` 并 fsync，追加输出后 fsync，再提交 SQLite。恢复按数据库提交位置截尾，未提交 seen 回滚，避免重复或丢失。完成后先核验身份直接返回，不改 publisher 产物。
6. 完整 train 文件以旧 train 的已校验前缀开始，后接真实新增 Token。旧 validation/test、tokenizer 和 holdout 正文在校验后使用硬链接，哈希保持相同。独立 publisher 检查前缀、连续偏移、EOS、词表及抽样解码正文 hash，最后原子发布完整 manifest。

逐篇审计保存原 repo/revision/shard/row/id/url、正文 hash、token count 和全局偏移；不保存另一份巨大的 raw/cleaned JSONL。原分片可按固定身份重新下载。原本的 Wiki raw/cleaned/index/audit 与所有 checkpoint 均保留。

## 运行位置与查看命令

任务快照位于 `runs/data_3b_prepare/`，独立脚本不属于当前训练的 `src/llm_lab` 指纹。当前任务监督 PID 为 5111，worker 初始 PID 为 5337，PID 是启动快照，重启后以状态文件为准。任务在 SSH 断开后继续。

在 AutoDL 终端查看状态：

```bash
cd /root/autodl-tmp/model_training
cat runs/data_3b_prepare/job_status.json
cat artifacts/pretrain/bilingual_3b/progress.json
cat artifacts/pretrain/bilingual_3b/acceptance.json
tail -n 5 runs/data_3b_prepare/worker.log
```

历史进度中的 `tentative_tokens` 是当时分片暂存数量，`committed_tokens` 是已提交新增数量，`total_committed_tokens` 包含原 1.1B。当前任务已结束，日志不会继续增长；最终容量以正式 manifest 为准。

整个任务有 4 小时时间边界，始终保留 12 GiB 数据盘余量；达到边界会停止并保留可恢复的状态，不发布不完整语料。CPU 低优先级，Arrow 单批 64、tokenizer 4 线程，不使用 GPU。若任务失败，先查看日志与 writer PID，确认原进程停止后再处理 stale lock，并使用同一运行快照、配置和 `--resume`；不要同时启动第二个 writer。

任务成功需同时满足：`job_status.json` 的 `status=completed`、完整 manifest 存在且验收 `passed`。仅下载完成或 `expansion_manifest.json` 存在，均不能替代合并发布成功。

## 验证与磁盘

本地新增测试 13 passed、1 skipped（本机没有 PyArrow）；远程 CPU 14 passed，补齐真实 Parquet 读取。包含真实 BPE、EOS、旧前缀、镜像/重复排除、三处崩溃提交恢复、父前缀复制中断、完成恢复不修改发布文件、合并 BinaryCorpus 读取等。fixture 的繁简转换使用测试替身；生产 worker 另强制核对真实 OpenCC 的完整 provenance，不能把 fixture 当成 2B 实际数据验收。

按用户要求已清理旧失败/中断数据、pip 下载缓存、已转为 raw JSONL 的 Parquet 缓存、已完成的 tokenizer benchmark 暂存；记录为 `runs/storage_cleanup_20261024T114253Z.json`。共回收 10,602,418,176 bytes（约 9.87 GiB），清理后可用约 36.35 GiB；这些是历史快照，随后下载与编码会继续使用空间。所有正式模型 checkpoint、语料与原始 Wiki JSONL 保留。

新增 Token 本体约 8 GB，旧前缀副本约 4.4 GB，另需审计/index、单片暂存及提交时临时重复空间。分片提交后释放专属下载缓存。300M 开始前再核对磁盘，完整模型 `last/best/atomic tmp` 约 10.17 GiB，不能把眼下剩余空间当作下轮训练可用量。
