# 中英基础数学后训练数据：2026-11-01

按用户最终要求，训练规模固定为50,000条，不扩大到10万。数据面向同一SFT起点下的DPO、PPO+RLVR与GRPO+RLVR对照；本次准备不启动新的GPU训练。后续每个训练实验仍只允许epochs=1，stop_epochs省略或为null。

## 已发布数据

发布目录：`artifacts/posttrain/math_alignment_20261101_162944/`。

Manifest SHA256：`c913208b5372d25dc33c3044792cb75919a6062776574d75d6f4b4e28d6abf44`。

Data identity：`42a70fb38bf74d0bed69a398a999c1d75fa585f9a2eb640d3b431a03e9b62497`。

| 分区 | 中英记录数 | 双语题目组 | 用途 |
| --- | ---: | ---: | --- |
| train | 50,000 | 25,000 | 主训练；中文和英文各25,000条 |
| validation | 2,000 | 1,000 | 选配置；使用训练未用的表述模板 |
| test_id | 1,000 | 500 | 已知题型/表述下的新数值题，检查同领域增益 |
| test_template | 1,000 | 500 | 训练与开发均未用的表述模板，检查模板泛化 |

八类题目为加法、减法、乘法、整除、先乘后加、先乘后减、先加后乘、先减后乘。训练集每类6,250条；开发集每类250条。两个测试分区分别覆盖每类124或126条，中英严格各半。

同一数学题的中英文属于同一个group_id。算式按加法/乘法交换律规范化，避免仅交换操作数就跨分区重复。开发和测试不复用训练算式；归一化题目与现有项目105,995个已知题目摘要做精确排除。这里不宣称完成了全语义去重或预训练语料的完全去污染。

## 两种训练视图，共用同一题库

- `rlvr/train.jsonl`：prompt、ground_truth、reward_type及来源/分组/题型信息。
- `dpo/train.jsonl`：相同prompt和ID，增加chosen/rejected。
- 两个视图均有相同的validation、test_id、test_template，以及嵌套的train_2000、train_10000。小规模版本覆盖全部八类，并优先使用较小数值题。

RLVR与DPO不是两份独立题库：训练共50,000条语言记录、25,000个双语组。统计置信区间时应按group_id分组重采样。

**DPO的偏好对来源是程序构造。** chosen是算术正确答案，rejected是经验证的错误数值；两者采用相同的纯数值输出格式。它们不是人工偏好、不是当前SFT模型采样结果，也不代表开放问答的综合回答质量。适合先做可复现的数值正确性对照；正式研究模型自身错误时，可后续只对训练问题采样并构建另一版本的偏好数据。

比较算法时还应保留同领域SFT对照：同一份DPO数据的prompt+chosen即可作为正确答案监督数据，帮助区分领域补课与偏好优化的收益。该建议不表示本次自动增加了SFT训练。

ground_truth、metadata.expression和其他Oracle信息只供奖励与审计使用。模型输入只取prompt，并套用Manifest中冻结的聊天模板，不把答案字段拼入模型输入。

## 具有代表性的外部测试

公开原始数据按提交固定版本下载，官方训练部分仅作储备，本轮5万训练没有使用这些公开题目。外部测试的选择没有参考任何模型的得分。

| 来源 | 官方测试记录 | 去重及合法性检查后 | 说明 |
| --- | ---: | ---: | --- |
| GSM8K英文 | 1,319 | 1,319 | 多步小学应用题 |
| TAL-SCQ5K中文 | 2,000 | 1,917 | 选择题；保留原难度0至4分层 |
| TAL-SCQ5K英文 | 2,000 | 1,823 | 选择题；保留原难度0至4分层 |
| 合计 | 5,319 | 5,059 | 分数据源、语言、难度报告 |

排除了259条归一化重复题和1条选项/标签异常题。没有按模型表现或题目难易删除题目，没有截断公开题目；本次所有其余题目均满足完整问题加128生成Token不超过1,024的限制。原始5,319条官方测试记录完整留在来源缓存，排除ID及原因登记于provenance/external_exclusions.json。

同领域测试2,000条，加外部测试5,059条，共有7,059条可评测记录。TAL包含竞赛题，与受控基础算术分布不同，应作为外部能力边界分组报告，不混成一个不加说明的总准确率。

## 验证器和评测协议

`llm_math_data.verifier`使用严格数值/单选匹配。数值支持整数、有限小数和分数的精确有理数比较；拒绝多候选数字、解释文本、NaN、除零和可执行表达式。奖励只有正确为1、错误为0，不因格式标签、输出长度或思维链外观额外给分。

主指标为严格最终答案准确率：输出必须满足单一数值或选项字母约定，并与参考答案一致。另报格式通过率、中英文、一步/两步题型以及公共题库的难度分层结果。各方法使用同一冻结SFT检查点起步、同一分词器/模板、同一生成上限；DPO Loss与RL奖励不能直接比较数值。

原303M模型的32K分词器SHA为6040c90c045836982a47bc0544942f122d0446bf9ed4d2ac1929befd0d4b86b3。受控训练样本最长序列含回答/EOS为60 Token；中英回答Token数相同。中文使用官方OpenCC1.1.9转简体并验证固定点。

本地与远程各19项CPU检查通过，包括真实小型发布流程、数值奖励绕过、白名单AST、交换律去重、嵌套子集、更新文件摘要后仍能发现错误偏好、模板泄漏与官方样本排除分母。全量本地及远程独立审计均已通过；远程结果记录在同名助手数据准备目录。本任务只评最终答案，不评估可见推理过程是否合理。

公开题库的参考标签来自发布者，没有独立重新解答全部公开题。本次数据验收不代表模型已经答对题目，也不证明训练或泛化效果。

## 复现与远程使用

原始数据缓存：`artifacts/research/math_alignment_sources_20261101/`，包含sources.lock.json、固定提交、文件SHA256、原始训练/测试、许可证/发布者数据卡和已有项目排除摘要。

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
.venv/bin/python scripts/audit_math_alignment.py \
  --manifest artifacts/posttrain/math_alignment_20261101_162944/manifest.json
```

只准备新版本时使用scripts/prepare_math_alignment.py和configs/data/prepare_math_alignment_20261101.json，并换新的输出目录；已发布目录拒绝覆盖。主训练文件是rlvr/train.jsonl或dpo/train.jsonl，开发集是各自validation.jsonl。测试文件不参与训练和调参。

当前仓库的MASSIVE/Qwen DPO旧入口使用另一种任务协议，不能直接套用这些数学文件。自研300M后续需要独立的DPO或rollout训练入口；本轮只完成数据、奖励函数与验收，不自动启动后训练。

来源：[GSM8K官方仓库](https://github.com/openai/grade-school-math)，提交3101c7d5072418e28b9008a6636bde82a006892c；[TAL-SCQ5K官方仓库](https://github.com/math-eval/TAL-SCQ5K)，提交8a66d6e14c65ba002beda39747e6b7c43bd2d472。GSM8K的MIT许可证随原始文件保留；TAL的MIT声明来自[发布者HF数据卡](https://huggingface.co/datasets/math-eval/TAL-SCQ5K)，其原GitHub仓库未单列LICENSE文件。
