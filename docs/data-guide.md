# 数据获取、制备与来源审计

数据分成两条：预训练使用纯文本，研究语言建模；后训练使用带标注的中英任务，研究指令跟随和偏好优化。两者各有固定版本、分区和独立评测，不能把任务测试答案混入预训练学习数据。

## 先在本地跑通 fixture

fixture 是仓库里的少量自建示例，只验证处理逻辑，不用来证明真实语言能力。以下命令不联网、不下载模型：

```powershell
$env:TEMP='E:\codex\tmp'
$env:TMP='E:\codex\tmp'
$env:PYTHONPATH='E:\codex\model_training\src'
& 'E:\conda_envs\llm\python.exe' -m llm_lab.data.acquire --config configs/data/acquire_smoke.json
& 'E:\conda_envs\llm\python.exe' -m llm_lab.data.prepare --config configs/data/prepare_smoke.json
& 'E:\conda_envs\llm\python.exe' -m llm_lab.data.prepare --config configs/data/prepare_massive_smoke.json
```

产物在 `examples/_outputs/`：预训练输出 `pretrain/manifest.json`、Tokenizer 和各语言各分区的 Token 流；任务输出 `tasks/task_manifest.json`、`sft/{train,validation,test}.jsonl` 和对应 `dpo/` 文件。命令默认拒绝覆盖已有正式产物；重新处理时选择新的输出目录并保留来源记录。

## Wikipedia 预训练数据

来源为 [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)，使用 `20231101.en`、`20231101.zh` 快照，固定 revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`。当前 `configs/data/acquire_wikipedia.json` 明确选择中英文各第一个 Parquet 分片，并固定文件路径、完整 SHA256 和字节数：

| 语言 | 固定分片路径 | 文件大小（bytes） | SHA256 |
| --- | --- | ---: | --- |
| en | `20231101.en/train-00000-of-00041.parquet` | 420296449 | `382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677` |
| zh | `20231101.zh/train-00000-of-00006.parquet` | 587109027 | `853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56` |

两个完整分片合计 **1007405476 bytes，约 1 GB**，先下载到项目 `.cache/data/wikipedia/<sha256>.parquet`，核验大小和 SHA256 后，再从本地 Parquet 流式解析。传给 Datasets 的是本地文件路径，避免为有限样本枚举 Hub 全仓库元数据；不是只下载 100 MB 的远程按需抽样。

获取配置的 `max_docs=20000`、`max_bytes=100000000` 仍分别限制扫描文档数和写出 JSONL 字节数；当前两路 source 各分配最多 10000 篇、50000000 bytes（50 MB）输出。**100 MB 是 JSONL 输出上限，不是下载量上限。** 缓存分片、临时下载、JSONL 与后续编码产物都需要空间；重试也可能增加网络读取。这里只覆盖首分片中的有限文章，不代表完整 Wikipedia 或全站均匀随机样本。

本次 AutoDL 实例优先使用官方 Hub 加平台学术加速。实际排查发现：默认 Miniconda CA 集合缺少平台已安装的根证书，加载 `/etc/network_turbo` 后曾出现 SSL 校验错误；系统 `/etc/ssl/certs/ca-certificates.crt` 已包含该证书。指定系统 CA 后，两个官方分片均已下载并通过大小与 SHA256 校验。已完成本批语料制备、32K Tokenizer 和六条 Token 流校验；实测结果见 [验收记录](verification.md#2026-10-23-真实数据与-gpu-预训练验收)。这只是有界首批语料，不是全量 Wikipedia。

在子 shell 中仅为获取进程加载加速和现有系统 CA，保持 TLS 校验，不使用 `verify=False`，不安装新的 CA：

```bash
(
  source /etc/network_turbo
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
  HF_ENDPOINT=https://huggingface.co python -m llm_lab.data.acquire --config configs/data/acquire_wikipedia.json
)
python -m llm_lab.data.prepare --config configs/data/prepare_wikipedia.json
```

子 shell 结束后，代理与 CA 变量不会写入父 shell，避免影响后续训练的 W&B 连接。平台加速的使用方式和适用范围见 [AutoDL 官方学术资源加速说明](https://www.autodl.com/docs/network_turbo/)。这项 CA 处理针对本实例已查明的证书差异，不代表其他环境都存在相同问题。

镜像端点保留为备选；本实例实测其下载较慢，不据此承诺其他时段或实例的速度：

```bash
HF_ENDPOINT=https://hf-mirror.com python -m llm_lab.data.acquire --config configs/data/acquire_wikipedia.json
```

端点只改变下载入口，原有 40 位 revision、分片路径、SHA256 和大小保持不变，不能改成 `main` 或省略内容校验。端点是否可达、下载是否完成和速度均以实际运行结果为准。两个分片都完整缓存并通过校验后，重新获取只读取本地 Parquet，不依赖外网；大小或哈希不符时会报错，不静默信任旧文件。

获取输出为 `artifacts/raw/wikipedia/records.jsonl` 及获取清单；制备输出为 `artifacts/pretrain/wikipedia/manifest.json` 及 Token 流。获取完成不等于制备完成：仍需清洗、划分、训练 Tokenizer、编码和校验。配置中的文档数、输出字节和 Token 预算是上限，需要结合远程 CPU、磁盘、吞吐和总费用确定正式量级。

制备先做 Unicode/空白规范化，再做长度、单字符重复、重复行等显式质量过滤和规范化文本精确去重。按源文档划分 train/validation/test，再训练 BPE 与编码。当前是基础过滤与精确去重；没有实现大规模语义近重复检测，百科文章的模板和引用重复仍需审计。

Tokenizer 只在训练分区拟合；文档末尾写入 EOS。预训练存储为各分区的 `en`、`zh` 两路 little-endian uint32 Token 流；manifest 写入 Tokenizer 哈希、词表大小、各流哈希、有效 Token 数、审计文件和完整数据配置。训练时按有效 Token 数平衡语言，不能用文档各半代替 Token 各半。

## MASSIVE 后训练数据

来源为 [MASSIVE 官方项目](https://github.com/alexa/massive)，第一版只选 `en-US`、`zh-CN`。保留官方 partition，同一源 ID 的中英文版本必须在相同分区。任务为意图识别与槽位抽取，输出结构：

```json
{"intent":"play_music","slots":[{"slot":"artist_name","value":"示例歌手"}]}
```

正式获取与制备的入口：

```bash
python -m llm_lab.data.acquire --config configs/data/acquire_massive.json
python -m llm_lab.data.prepare --config configs/data/prepare_massive.json
```

输出任务目录为 `artifacts/tasks/massive/`。制备解析原始槽位标注，记录异常；对跨分区重复输入显式报错或按配置保留评测分区并删除训练副本。标签空间与负例规则只从训练分区构建。

发行包固定官方 HTTPS 版本 URL。若有独立可信 SHA256，可配置校验；若没有，首次获取记录观察到的 SHA256、URL 和大小，后续复用按已记录哈希校验。观察值可用于追踪和检查缓存变化，不能称为独立来源认证。

偏好不是人工打分：chosen 来自正确任务标注，rejected 保持 JSON/schema/槽位，仅改成另一个训练意图标签。这样的控制便于观察意图偏好，但覆盖的错误分布很窄，不支持对通用人类偏好或所有格式错误的结论。

## 许可、署名和公开交付

代码许可证、原始数据许可证、Base 模型许可证和输出权重发布条件分别记录。MASSIVE 数据为 CC BY 4.0，上游代码许可另列；Wikipedia 需要保留原始来源和适用的署名/共享条件，不能把本仓库代码许可套到语料。

固定快照和哈希只支持可追踪，不代表数据一定无隐私、无错误或未被 Base 训练使用。下载前审查 dataset card 与上游许可；发布语料副本或权重前再根据具体用途审查适用条款。公开报告至少列出数据名称、版本/revision、来源链接、许可和处理方式。

## 数据准备验收

- `train/validation/test` 同源组和规范化内容不重叠；中英文配对保持分区一致。
- Tokenizer 仅来自 train，词表大小、EOS 和模型配置兼容。
- 各语言各分区有效 Token/样本均非空且达到配置所需长度。
- 清洗、丢弃、重复、截断与偏好错误类型可从清单和审计记录查回。
- 后训练的 JSON 标注可解析且满足 schema，prompt 不跨分区重复。
- 最终测试数据不进入调参、偏好构造规则拟合或模型选择。
