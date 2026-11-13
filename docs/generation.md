# 预训练模型的生成与重复退化

现有 `python -m llm_lab.cli generate` 每步直接取 `logits.argmax()`，没有温度、Top-p、Top-k 或重复惩罚。这是一条确定性的贪心续写基线。相同 checkpoint 的不同解码策略会产生明显不同的输出，贪心长文本出现循环属于已知问题；参见 [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) 和 [Hugging Face 生成策略](https://huggingface.co/docs/transformers/main/en/generation_strategies)。

100M / 1B 模型是 Wikipedia 为主的基础语言模型，没有做指令微调；它可以学到部分中文句型，但参数、有效中文训练量和领域覆盖都有限。循环说明当前生成有问题，单条样例不能证明严重过拟合、目标错位或训练失败。CPU 推理也不是这一问题的根因；改变设备主要影响速度。改采样可以减轻循环，不会增加模型知识或保证事实正确。

## 2026-10-25 实际 CPU 对比

使用用户同一份 `best.pt`（step 61,000，训练处理量 999,424,000 Token），模型 100,092,672 参数，CPU 4 线程。固定三个 prompt（用户的机器学习中文续写、中国首都、机器学习英文续写），每个比较贪心、普通采样、采样+4-gram 限制；用户原 prompt 再补 seed 7/13，共 **11 份输出**，最多各 100 个新 Token，遇 EOS 自然停止。整个对比约 45.77 秒，没有 GPU 训练或权重更新。

- 原 prompt 的贪心输出完整复现用户提供的循环，表明新脚本加载了相同模型行为。
- 普通采样 seed 42 在 14 个新 Token 后遇 EOS，输出仍是“如何让机器学习……机器学习”，不是有效解释。
- 加 4-gram 限制后不再重复相同的 4 个 BPE Token 序列，但仍反复出现“机器学习”，句意空泛。
- 两个补充中文 seed 会离题或产生概念混淆；首都续写给出错误内容；英文样例虽较少重复，却生成了不正确的机器学习解释。

因此本次只确认解码层能控制部分形式重复，**没有解决该 checkpoint 的语义与知识能力不足**。普通采样的短 EOS 输出不能与 100 Token 贪心直接比较重复率；4-gram 指标为零也有硬约束的直接作用，不属于知识质量提升证据。

完整输出与设置：远程 `runs/generation_compare_20261025_141908/comparison.json`，本地 `tmp/pretrain/generation_comparison_20261025_141908.json`。采样代码测试本地及远程 CPU 均 **21 passed**。对比结束时旧训练源码指纹和 checkpoint 大小/修改时间保持相同。

## 使用独立采样入口

新入口是 `scripts/generate_text.py`，复用原模型和 checkpoint。保留 `src/llm_lab` 全部源码、旧 CLI 和训练文件，避免改变 100M 或已冻结 300M 训练的严格恢复身份。原 CLI 继续作为贪心基线。

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh

.venv/bin/python scripts/generate_text.py \
  --checkpoint runs/pretrain_100m_1b_20261024_192116/best.pt \
  --device cpu \
  --prompt "机器学习是人工智能的一个分支，它主要研究" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-p 0.9 \
  --top-k 50 \
  --repetition-penalty 1.1 \
  --no-repeat-ngram-size 4 \
  --seed 42
```

上面参数是可对比的起点，不是最优参数保证。若限制让表达生硬，先将 `--no-repeat-ngram-size` 改成 0，并比较不同 seed。

| 参数 | 作用与边界 |
| --- | --- |
| temperature=0.8 | 调整采样分布的尖锐程度；过低仍可能重复，过高容易胡写 |
| top_p=0.9 | 在累计概率覆盖阈值的候选集合中采样 |
| top_k=50 | 最多保留概率最高的 50 个候选，0 表示关闭 |
| repetition_penalty=1.1 | 对历史中已出现的 Token 适当降权；对负 logit 乘系数、正 logit 除系数 |
| no_repeat_ngram_size=4 | 禁止重复出现相同的 4 个 BPE Token 序列，包含 prompt 历史；不是中文四字词，可能妨碍正常复述 |
| seed=42 | 固定本次采样的独立随机数发生器；不承诺跨设备/软件版本逐位一致 |

脚本默认禁止生成 PAD/UNK，保留 EOS 结束条件。模型每步只读取最近一个上下文窗口，重新计算前向；本次没有引入 KV cache。全候选被约束屏蔽会明确结束，不偷偷生成 PAD；原始 logits 的 NaN/正无穷会报错。

要在同一个入口对照贪心生成，使用 `--greedy --repetition-penalty 1 --no-repeat-ngram-size 0`。PAD/UNK 仍被屏蔽，因此严格比较原 CLI 时应同时检查它是否曾选择这些特殊 Token。

## 对比和下一步

比较时固定 checkpoint、prompt、最大新 Token 数，保留贪心输出，再比较无硬约束采样和 4-gram 约束采样。至少看几个中英 prompt 与多个 seed。重复率下降不能单独证明生成质量提高，因为随机胡写也可以不重复；硬性禁止 4-gram 后，该指标下降也有约束本身的直接作用。

若仍离题、概念混淆或不连贯，继续评估训练数据质量/覆盖、验证曲线、参数量和 Token 预算。3B 训练池比 Wiki 更丰富，300M 更大，但这两项不保证消除重复。若最终目标是听懂指令和聊天，再进入高质量 SFT；SFT 不能替代基础语言与知识能力的预训练。

这条 prompt 是正文续写，优先处理解码与基础模型质量，不需要先把它包装成聊天模板。推理实验只读权重和正式语料身份，不训练、不写训练 checkpoint，也不更新预算账本。
