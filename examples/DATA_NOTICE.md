这里的两份 JSONL 均为本项目原创的少量中英文测试样本，不是下载的 Wikipedia 或 MASSIVE 内容。

它们仅验证数据接口、分区隔离、分词、训练命令及评测闭环，不提供模型语言能力或正式任务效果的证据。`massive_fixture.jsonl` 仿照公开 schema 编写；fixture 中的人名、地名、歌名不指代真实标注语料。

正式数据许可入口：

- Wikipedia 数据卡：https://huggingface.co/datasets/wikimedia/wikipedia
- 固定版本：https://huggingface.co/datasets/wikimedia/wikipedia/commit/b04c8d1ceb2f5cd4588862100d08de323dccfbaa
- MASSIVE 数据许可与上游来源：https://github.com/alexa/massive/blob/main/NOTICE.md
- MASSIVE 发行及字段说明：https://github.com/alexa/massive/blob/main/README.md

MASSIVE 发行包的独立 SHA256 暂未随项目提供。远程执行 `acquire_massive.json` 时，程序固定官方 HTTPS 1.1 URL，首次下载记录 `observed_sha256`、URL 和大小并落锁；以后复用缓存须通过该指纹校验。配置中的 `expected_sha256` 可选，若提供则另做校验。观测散列是版本指纹，不是独立的来源认证。本地 fixture 流程完全不下载发行包。
