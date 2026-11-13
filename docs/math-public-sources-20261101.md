# 广泛数学公开语料准备：2026-11-01

任务范围已改为代数与方程、几何、数论、概率统计、组合计数、函数与三角、微积分、线性代数，并保留证明与不等式题。原先的程序合成基础算术不再作为正式数学 SFT 主语料。

本次未启动数学 SFT、DPO、PPO 或 GRPO。此前两次 GPU 作业均为不更新权重的开发集采样。后续起点确定为已完成的全参数 SFT，Checkpoint SHA256 为 `4a36a0862c9d0527a7ec3737c7bc9ed5c107a667d4c3378b0e14d120241c6060`。所有新实验仍只允许 `epochs=1`、`stop_epochs=null`。

## 已实际下载并校验的主要公开来源

| 来源 | 记录及用途 | 数学覆盖/限制 |
| --- | --- | --- |
| [NuminaMath-1.5](https://huggingface.co/datasets/project-numina/NuminaMath-1.5) | 官方训练896,215条，3个完整分片 | 提供领域、题型、标准答案及有效性标记；其中520,811条同时为非合成且问题/解答标记Yes。这些标记不是独立正确性证明。 |
| [CMM-Math](https://huggingface.co/datasets/ecnu-icalk/cmm-math) | 官方训练22,248条；原始无图记录15,976条 | 原生中文、小学至高中、题型/年级/参考解析；必须进一步排除缺图、缺解答及超长样本。没有下载图片包或把图片题强行变成文字题。 |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)及[公开中文版](https://huggingface.co/datasets/ALmonster/MathInstruct-Chinese) | 英文262,039条，中文256,294条，文件合计305,682,143字节 | 数学覆盖广，但抽检发现计算错误、翻译矛盾和上下文缺失；仅作为已下载储备，第一版候选禁止训练。 |
| [MATH-Hard-Chinese](https://huggingface.co/datasets/ALmonster/MATH-Hard-Chinese) | 已下载1,116,460字节公开翻译文件 | 属于公开模型翻译储备，未选入当前新的原生中文主来源。 |

所有文件保存在 `artifacts/research/` 的独立时间目录，`sources.lock.json` 登记官方固定提交、大小与SHA256。合计下载不等于正式训练条数；NuminaMath与CMM的数据文件合计约539MB，其他储备不混算进5万条。

## 当前两版候选的状态

`artifacts/posttrain/math_public_20261101_v1/`：MathInstruct双语配对候选50,000条，结构检查通过，但数学抽检失败，已写入 `DO_NOT_TRAIN.json` 隔离。不能把结构验收当成数学正确性验收。

`artifacts/posttrain/math_public_20261101_v2/`：从Numina非合成有效题与CMM原生中文无图题筛出的50,000条候选。训练中英文分别42,272和7,728条，中文回答监督Token占11.26%；另保留开发2,000条和内部最终测试2,000条。该语言分布并不满足中英Token均衡目标，尚待用户选择规模/配比路线。

v2的学科计数：代数9,526；几何8,349；函数三角6,554；概率统计4,559；数论4,370；组合4,370；微积分3,525；不等式1,834；逻辑989；线代682；其他数学5,242。这些细分标签包含启发式分类，不是人工金标准。发布者原始领域与年级同时保留。

v2抽检仍发现两类问题：证明引用未提供的编号定理；多答案题的源标准答案只保留最后一个boxed值，同时个别推导包含计算错误。已写入 `DO_NOT_TRAIN.json`，不得用当前候选直接启动训练。筛选规则已增加相应拒绝条件，并修正把“integral powers（整数次幂）”误判为微积分的问题；需按语言路线构建新版本，不能修改既有候选来伪装通过。

抽检样例及源行身份：`runs/assistant/data_prepare/math_public_20261101/v2_spotcheck_samples.json`。原始来源全量统计：同目录 `source_inventory.json`。数据审核记录与局部CPU检查不代表模型已经获得数学能力。

## 后续训练边界

先完成数据质量及中英配比选择，再从全参数SFT做一轮数学专项SFT。随后从同一专项SFT权重独立比较DPO、PPO+RLVR和GRPO+RLVR。

已有的严格数字验证器只适用于数值答案，不能直接评判这些证明、符号表达式或多答案题。DPO需要基于训练题采样并核验偏好对；当前没有把随机错误数值包装成新的广泛数学偏好数据。

独立数学全参数入口为 `scripts/train_math_sft.py`；CPU小模型验证了沿用SFT权重、真正单轮调度与精确断点恢复。模板在 `configs/posttrain/custom_math_full_50k.json`，目前指向禁止训练的待修订候选，prepare/check会拒绝它。尚未冻结正式配置或启动GPU长跑。
