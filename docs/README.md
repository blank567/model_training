# 文档导航

仓库整理于 2026-11-13。先读当前入口，再按需要查历史记录；日期文档中的“当前 / 待执行 / 已启动”描述只代表记录当日，不代表今天的进度。

## 当前入口

| 文档 | 用途 |
| --- | --- |
| [项目 README](../README.md) | 已完成范围、主要结果、安装与运行入口 |
| [详细实验结果](experiment-results.md) | 预训练、SFT、LoRA、阅读专项与九组 RL 的数据、结果和局限 |
| [结果产物](results/) | 公开 CSV、图表与来源哈希，不含原文预测和权重 |
| [架构与源码](architecture.md) | 实际自研模型主线、模块、输入输出与恢复边界 |
| [仓库发布说明](repository-release.md) | Git 范围、私有配置、验证和许可证边界 |
| [运行目录约定](runs-layout.md) | 正式实验与助手记录分开；不移动冻结引用 |
| [源码学习指南](learning-guide.md) | 张量、公式和代码阅读顺序 |
| [预训练流程详解](预训练流程与源码详解.md) | 预训练源码逐步讲解 |

## 按阶段运行

| 阶段 | 文档 |
| --- | --- |
| 环境与监控 | [远程手册](remote-runbook.md)、[W&B](wandb.md)、[源码同步](source-sync.md) |
| 预训练 | [数据指南](data-guide.md)、[1B 数据](data-1b.md)、[3B 数据](data-3b.md)、[100M 运行](run-100m-1b.md)、[303M 正式配置与测速](run-300m-optimized.md) |
| 全参数 SFT | [数据构建](sft-data-build.md)、[数据扩充](sft-expansion-20261027.md)、[训练入口](sft-training.md) |
| LoRA | [设计](lora-design-300m-20261028.md)、[运行入口](lora-training-300m.md)、[单轮 rank / dropout](lora-rank-dropout-20261101.md) |
| 阅读专项 / RL | [专项数据](reading-alignment-20261101.md)、[独立 RL 数据](reading-rl-data-20261101.md)、[九组对照协议](reading-rl-comparison-20261102.md) |
| 推理 / 评测 | [生成策略](generation.md)、[历史通用评测](sft-model-comparison-20261028.md)、[实验协议](experiment-protocol.md)、[当前完整结果](experiment-results.md) |

## 历史记录怎么使用

- [PROJECT_CONSTRAINTS.md](PROJECT_CONSTRAINTS.md)、早期架构方案、Qwen / MASSIVE 配置保留设计过程；主线已发展为自研 303M，不把计划写成结果。
- [SFT 历史轮数计划](sft-epochs-batch-plan.md)、[四组历史对照](sft-four-controls-20261027.md)解释已经完成的多轮实验。**新实验一律 1 轮、无 stop_epochs 停止点**，不要直接执行旧多轮计划。
- [AutoDL 初始部署](autodl-deployment.md)、[W&B 网络修复](wandb-network-20261101.md)、[历史验收](verification.md)是当时环境记录。硬件、价格、磁盘和联网状态需在新实例重新检查。
- [数学候选](math-alignment-20261101.md)、[公开数学选源](math-public-sources-20261101.md)包含被替代的数据候选和校准，不代表已完成数学 RL。
- 个人面试题库、压缩包与其派生文件保留本地，不纳入本次 Git 发布；历史运行原件也不随仓库分发。

当前可信实验汇总以[详细结果](experiment-results.md)为入口；训练重启必须使用自己的已验收输入与原冻结身份，不能仅按历史文档复制路径。
