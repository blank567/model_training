# 2026-10-27 数据盘清理

本次针对AutoDL项目 `/root/autodl-tmp/model_training`，未操作本机C盘。用户要求删除无必要内容，并明确确认删除两个早期1000步基线的best/last权重。

| 项目 | 结果 |
| --- | --- |
| 清理前空闲 | 14.7312 GiB |
| 清理后空闲 | 17.3949 GiB |
| 实际释放 | 2.6637 GiB |
| 保护文件 | 624个，inode、大小、mtime及链接数未变 |
| 训练源码指纹 | `0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872`，保持不变 |

## 已删除

- `runs/pretrain_100m/`、`runs/pretrain_manual_20261023_204911/` 中用户明确选择删除的 `best.pt/last.pt`。每组两文件原为硬链接，真实释放合计约2.24GiB，不能按四份文件大小重复计算。日志、配置、曲线和summary保留，新增 `weights_removed.json` 标记；这两组旧基线已不能推理或续训。
- 已SUPERSEDED的SFT `20261026_133649/140730` 四个数据分区载荷。保留manifest、验收/质量记录、许可、SUPERSEDED拒绝标记及新增的 `PURGED_PAYLOAD.json`，仍拒绝用它们训练。
- 三个可重新获取的SFT源下载副本：`.cache/data/sft_300m/{alpaca_gpt4_zh.json,dolly.jsonl,smol_smoltalk_train_0.parquet}`。删除前按固定配置核验大小和SHA；固定revision、URL、SHA配置仍在。以后重建SFT数据需要先重新下载，不能假设可以立即离线重建。
- pytest临时夹具、五个已结束构建/队列任务的fixture目录、项目Python字节码缓存、pip缓存及旧源码同步备份。内部tiny测试权重与用户模型区分；不整删tmp、.cache或.sync。

详细删除清单和文件摘要在远程 `runs/assistant/system/space_cleanup_20261027_150505/plan.json`、`report.json`。

## 保留

100M/1B、300M/3B及 `1e-5/2e-5` SFT的best和完整last全部保留。正式SFT170531及其164414父版本、四个split、原始Wikipedia、1B和3B语料、已绑定SHA的cleaned/audit/SQLite索引、分词器、许可和来源链均保留。

`.venv`运行环境、SSH密钥/同步状态、W&B登录凭据、正式曲线、共享账本、GPU锁和全部冻结训练配置保持原位。大体积cleaned与SQLite不是普通缓存，删除会破坏已发布数据的验收与复现链，本次没有删除。

## 清理后验证

最新SFT train19,991条、validation995条可正常加载，四个分区文件SHA全部通过；1B/3B预训练validation语料正常；两个清掉载荷的旧SFT版本仍被SUPERSEDED标记拒绝。`pip check` 返回 `No broken requirements found.`。原 `5e-5` 冻结配置检查通过，需空闲17.0406GiB，实际17.3949GiB。

清理没有启动任何GPU训练；第三组仍未启动，不因空间恢复而自行运行。当前可用余量仍有限，后续下载或新增实验需要继续逐次检查。

本地对应报告为 `tmp/sft_entry/space_cleanup_result.json`、`space_cleanup_verification.json`。
