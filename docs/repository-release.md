# 仓库整理与验证记录

整理日期：2026-11-13。本次整理更新公开入口、结果汇总和 Git 忽略规则，未迁移正式实验、改动历史训练源码或配置身份，也未启动 GPU 训练。

## 发布范围

提交源码、测试、依赖声明、配置模板、原创示例、技术文档及 `docs/results/` 中的脱敏汇总。原始训练数据和完整逐题预测不随仓库分发；公开结果通过来源文件哈希和稳定相对标识追踪，不能代替完整复现实验所需的输入。

`.gitignore` 排除 `runs/ data/ artifacts/ tmp/ .cache/ dist/`、权重、二进制 Token 流、W&B 原始文件、日志、环境、密钥和机器连接配置。个人面试交付包留本地；`.gitignore` 不删除这些文件。

`.gitattributes` 禁止自动换行转换，保留冻结源码和带发布 SHA 的 CSV 原始字节；否则 Windows 的 `autocrlf` 可能让克隆后的源码身份或 CSV 哈希不再匹配。历史冻结文件保留已有末尾空行，不为风格整理重写；Git 空白检查继续检查行尾空白及空格后 Tab。

实际连接配置 `configs/deployment/sync_autodl.json`、实例快照 `autodl_vgpu48.json`、本机网络设置 `configs/remote/wandb_network.json` 留本地。新用户复制 `.example.json` 并填写自己的 SSH 地址、端口和密钥位置；核实主机公钥，不关闭主机校验。密钥仅放忽略目录。同步器目标目录有显式约束，修改示例前先阅读 [source-sync.md](source-sync.md)。

部分历史源码和文档含原 W&B entity / 结果链接，它们不是认证凭据。为保留恢复身份没有批量替换；新配置应覆盖自己的帐号，历史 W&B 发布/补传脚本不能不经审查直接用于新项目。

`scripts/package_project.py` 与源码同步器采用自身白名单，**不以 `.gitignore` 作为过滤器**。本地部署包可能包含被 Git 忽略的机器配置；只能用于自己受控的部署目标。公共源码归档应从审查后的 Git 提交生成，不能直接公开 `dist/` 的部署包。

## 本地验证

环境为现有 Windows Python 环境，不是历史远程锁定组合：PyTorch 2.10.0、Transformers 5.1.0、W&B 0.24.2，缺少 PEFT、TRL、Datasets、TensorBoard 和 OpenCC 等组件。此次未为整理替换既有环境或安装依赖。

| 检查 | 实际结果 |
| --- | --- |
| 全套 pytest | 667 项；648 passed、8 failed、7 skipped、4 errors；约 73.70 秒 |
| 8 failed | 缺固定 PEFT 0.17.1，涉及冻结检查及真实 adapter 注入 / 训练 / 恢复 |
| 4 errors | 数学 fixture 缺少 OpenCC |
| 7 skipped | PyArrow 3 项、Transformers 版本 1 项、OpenCC 1 项、POSIX 平台 2 项 |
| CLI 帮助 | 预训练、SFT、LoRA、单轮 LoRA、阅读 RL 入口通过 |
| 离线 fixture | acquire → prepare → validate-only → 4 步 CPU 训练 → best.pt 生成 → validation 评测通过 |
| Linux 队列 | Windows 不支持 `fcntl`，未声称本地验证通过 |

CPU fixture 为 34,976 参数、128 个处理 Token，产物在助手 smoke 目录，使用隔离账本，不影响已有模型和共享预算。该检查只验证工作流，输出不代表语言能力。

本地完整日志、JUnit 和环境归因在 `runs/assistant/system/release_20261113/validation_summary.json`，属于忽略的本地验证记录。Windows 输出包含特殊字符时须设 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。不能把上述结果写成“全套测试通过”，也不能用 CPU 检查替代实际 GPU 依赖、数值或性能验收。

## 复现和许可

完整复现还需取得上游授权数据、重建验收产物，并拥有对应 Base / Tokenizer / Checkpoint；若原输入已清理或未发布，来源哈希不能还原文件。公开模板中保留的历史路径或哈希只是原身份示例，不代表 clone 后已经具备输入。

依赖库、语料来源与潜在权重发布需分别核对许可。部分通用 SFT 数据非商业，部分阅读数据要求署名与相同方式共享，不能统一标为 MIT 或可商用。本次未擅自为用户选择代码开源许可证；没有许可证意味着未授予一份新的通用使用许可。
