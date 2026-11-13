"""只验证新对齐入口的数值边界；小模型 CPU 正确性检查不代表正式训练效果。"""
from types import SimpleNamespace
import pytest
import torch

from llm_align.core import pack, token_scores, sequence_scores, generate, development_subset, MathCorpus
from llm_lab.config import ModelConfig
from llm_lab.models import Decoder


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return Decoder(ModelConfig(vocab_size=19, dim=16, n_layers=2, n_heads=2, n_kv_heads=1,
                               ffn_dim=32, max_seq_len=32, dropout=0.0,
                               activation_checkpointing=False)).eval()


def test_answer_mask_shifts_exactly_once_includes_eos():
    inputs, targets, mask = pack([[3, 4, 5], [6]], [[7, 2], [8, 9, 2]], 'cpu')
    assert inputs.tolist() == [[3, 4, 5, 7], [6, 8, 9, 0]]
    assert targets.tolist() == [[4, 5, 7, 2], [8, 9, 2, 0]]
    assert mask.tolist() == [[False, False, True, True], [True, True, True, False]]


def test_batched_logprob_matches_independent_unpadded(model):
    prompts, answers = [[3, 4, 5], [6]], [[7, 2], [8, 9, 2]]
    combined = sequence_scores(model, prompts, answers, 'cpu')
    single = torch.cat([sequence_scores(model, [p], [a], 'cpu') for p, a in zip(prompts, answers)])
    torch.testing.assert_close(combined, single, rtol=1e-6, atol=1e-6)
    combined.sum().backward()
    assert model.embedding.weight.grad.isfinite().all()


def test_critic_head_sees_pre_action_hidden_and_detaches_backbone(model):
    head = torch.nn.Linear(16, 1)
    logp, mask, values = token_scores(model, [[3, 4]], [[7, 2]], 'cpu', head)
    assert values.shape == logp.shape == mask.shape
    values[mask].sum().backward()
    assert head.weight.grad is not None
    assert all(p.grad is None for p in model.parameters())


def test_batched_greedy_matches_independent_variable_lengths(model):
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
    prompts = [[3, 4, 5], [7], [6, 8]]
    batch = generate(model, tokenizer, prompts, 'cpu', 4, sample=False)
    singles = [generate(model, tokenizer, [p], 'cpu', 4, sample=False) for p in prompts]
    assert batch['answers'] == [v['answers'][0] for v in singles]
    assert batch['eos'] == [v['eos'][0] for v in singles]


def test_eos_is_kept_for_training_but_removed_from_text():
    class EosModel(torch.nn.Module):
        config = SimpleNamespace(max_seq_len=32)
        def forward(self, x):
            logits = torch.full((*x.shape, 10), -100.0)
            logits[:, :, 2] = 100
            return SimpleNamespace(logits=logits)
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
    result = generate(EosModel(), tokenizer, [[3], [4, 5]], 'cpu', sample=False)
    assert result == {'answers': [[2], [2]], 'texts': ['[]', '[]'], 'eos': [True, True]}


@pytest.mark.parametrize('split', ['test_id', 'test_template', 'external_test', '../test_id'])
def test_training_corpus_rejects_test_partitions_before_io(split):
    with pytest.raises(ValueError, match='training/development'):
        MathCorpus('does_not_exist', split)


def test_development_subset_is_balanced_reproducible():
    rows = [{'id': f'{language}_{family}_{i}', 'language': language, 'metadata': {'family': family}}
            for language in ['zh', 'en'] for family in ['add', 'divide'] for i in range(10)]
    a = development_subset(rows, 2)
    assert a == development_subset(rows, 2)
    assert len(a) == 8 and len({r['id'] for r in a}) == 8


@pytest.mark.parametrize('prompts,answers', [([], []), ([[3]], [[]]), ([[]], [[2]]), ([[3], [4]], [[2]])])
def test_invalid_empty_batches_rejected(prompts, answers):
    with pytest.raises(ValueError):
        pack(prompts, answers, 'cpu')
