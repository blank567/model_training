# 学习路线与代码阅读

## 先建立完整数据流

预训练流向：原始文档 → 清洗/去重 → 文档分区 → 训练 Tokenizer → Token ID → 文档边界与样本 → `[B,T]` 输入和右移目标 → `[B,T,C]` 隐状态 → `[B,H,T,D]` 注意力 → `[B,T,V]` logits → 标量 Loss → backward → 梯度裁剪 → AdamW 参数更新。

`B` 是 micro batch，`T` 是上下文长度，`C` 是隐藏维度，`H` 是注意力头数，`D=C/H`，`V` 是词表大小。GQA 的 Query 头数与 KV 头数不同；KV 共享减少 KV 投影与推理缓存，不能因此推断所有训练显存都会同比下降。

目标序列移动一位：输入 `x=[a,b,c]` 时目标为 `y=[b,c,d]`。模型在位置 t 只能访问不晚于 t 的输入，从而预测下一 Token。语言建模目标是：

\[
\mathcal L=-\frac{1}{N}\sum_{b,t:\,y_{b,t}\ne-100}\log p_\theta(y_{b,t}\mid x_{b,\le t})
\]

`-100` 表示不计入 Loss 的位置；训练需使用真正的有效目标数归一化。Embedding 与输出投影可能绑定权重，参数量统计要避免重复。

## 从一小步看预训练

先看配置、数据 manifest，再看模型前向和训练循环。弄清楚每个指标是在 micro step 还是 optimizer step 上记录，以及 accumulation 时 Loss 如何缩放。AdamW 的权重衰减分组、warmup 和 cosine 的步数都必须按 optimizer update 定义。

RoPE 把位置信息作用到 Q/K；RMSNorm 规范化隐藏状态；SwiGLU 是门控前馈层。因果 mask 决定能看到哪些位置。先用测试验证未来 Token 改变不会影响过去位置 logits，再跑长训练。

梯度累积增加一次参数更新看到的数据量，但通常不会缩短单 Token 计算成本。Activation Checkpointing 减少保存的中间激活、增加重算；它需要验证反向和随机状态语义。BF16/FP16 改变数值范围与精度；FP16 常使用 GradScaler，BF16 通常无需 Loss Scaling，具体路径以代码为准。

## 再看 SFT 与偏好优化

SFT 把任务写成聊天模板，prompt 与 assistant response 一起进入模型，但只对指定回答 Token 计算 Loss。检查模板、特殊 Token、截断、padding、回答边界及全被 mask 的异常样本。

全参数 SFT 更新 Base 权重；LoRA 冻结 Base 并训练低秩增量，显存与优化器状态下降，但基础权重和激活仍需存储。公平比较需要固定起点和数据，分别寻找合理学习率，记录可训练参数比例。

DPO 输入同一个 prompt 下的 chosen/rejected。它比较相对于参考策略的回答对数概率差异；`beta` 与参考策略来源都是实验条件。规则构造偏好研究的是特定任务正确性与格式，不能据此声称完成通用 RLHF。

建议依次做：CPU 测试 → 极小数据训练控制流检查 → 远程环境探针 → 短程真实吞吐测试 → 100M 预训练基线 → 受控对照 → Base/SFT/LoRA 任务评测 → DPO 与局限分析。

## 面试讲解时需要证据

能够回答：为何验证 Loss 与训练 Loss 分离？为什么按文档划分？如何发现 Tokenizer 泄漏？如何证明 Loss Mask 正确？梯度累积与大 Batch 有何条件差异？中断后还原了什么？峰值显存在哪里采集？成本是否包括实例空闲？LoRA 的比较是否公平？DPO 是否只学会合成错误的模式？

每个答案优先引用配置、测试、运行记录和结果表。尚未远程验证的性能必须写为待测；小模型预训练产生局部统计规律，也不等同于已经有稳定开放聊天和推理能力。
