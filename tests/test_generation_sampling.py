"""独立解码契约：无 GPU、网络或真实权重，不以随机文本多样性冒充质量。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generation_under_test", ROOT / "scripts/generate_text.py")
generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation)


class Tokenizer:
    def token_to_id(self, token):
        return {"<|pad|>": 0, "<|unk|>": 1, "<|eos|>": 2}.get(token)

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[3 + ord(c) % 3 for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(i) for i in ids if not skip_special_tokens or i > 2)


class Model(nn.Module):
    def __init__(self, logits, context=3):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_seq_len=context)
        self.rows = logits
        self.contexts = []

    def forward(self, x):
        self.contexts.append(x.tolist()[0])
        row = self.rows[min(len(self.contexts) - 1, len(self.rows) - 1)]
        return SimpleNamespace(logits=torch.tensor(row, dtype=torch.float32)[None, None])


def neutral(logits, history=(), **kwargs):
    options = dict(temperature=1., top_p=1., top_k=0, repetition_penalty=1., no_repeat_ngram_size=0)
    options.update(kwargs)
    return generation.sampling_scores(torch.tensor(logits, dtype=torch.float32), history, **options)


def test_repetition_penalty_changes_signs_and_applies_once_per_unique_id():
    output = neutral([-2., 4., 3.], [0, 0, 1], repetition_penalty=2, greedy=True)
    assert torch.equal(output, torch.tensor([-4., 2., 3.]))


def test_top_p_retains_first_token_that_crosses_probability_threshold():
    logits = torch.tensor([.6, .3, .1]).log()
    result = generation.sampling_scores(logits, [], temperature=1, top_p=.7, top_k=0, repetition_penalty=1)
    assert torch.isfinite(result).tolist() == [True, True, False]
    result = generation.sampling_scores(logits, [], temperature=1, top_p=.01, top_k=0, repetition_penalty=1)
    assert torch.isfinite(result).tolist() == [True, False, False]


def test_neutral_settings_preserve_probabilities_and_top_k_clamps_to_vocab():
    logits = [1., -2., 0.]
    assert torch.allclose(neutral(logits).softmax(-1), torch.tensor(logits).softmax(-1))
    assert torch.isfinite(neutral(logits, top_k=999)).all()
    assert torch.isfinite(neutral(logits, top_k=1)).tolist() == [True, False, False]


def test_ngram_blocks_token_continuation_and_keeps_eos_available():
    assert generation.forbidden_ngram_tokens([3, 4, 5, 3, 4], 3) == {5}
    assert generation.forbidden_ngram_tokens([3, 4], 1) == {3, 4}
    assert generation.forbidden_ngram_tokens([3], 4) == set()
    scores = neutral([0., 0., 0., 0.], [2, 3, 2], no_repeat_ngram_size=1, eos_id=2)
    assert torch.isfinite(scores[2]) and torch.isneginf(scores[3])


def test_eos_stops_immediately_and_pad_unk_are_forbidden():
    model = Model([[99., 98., 10., 0., 0., 0.]])
    result = generation.generate(model, Tokenizer(), "a", max_new_tokens=9, greedy=True, repetition_penalty=1)
    assert result["stop_reason"] == "eos"
    assert result["generated_ids"] == [2]
    assert result["completion"] == ""
    assert len(model.contexts) == 1


def test_exhausted_constraints_stop_without_sampling_forbidden_pad():
    model = Model([[99., 98., -torch.inf, -torch.inf, -torch.inf, -torch.inf]])
    result = generation.generate(model, Tokenizer(), "a", max_new_tokens=2)
    assert result["stop_reason"] == "constraints_exhausted"
    assert result["generated_ids"] == []


def test_context_crops_while_history_and_completion_keep_all_tokens():
    model = Model([[0., 0., -100., 100., 0., 0.]], context=3)
    result = generation.generate(model, Tokenizer(), "abcdef", max_new_tokens=5, greedy=True, repetition_penalty=1)
    assert all(len(context) == 3 for context in model.contexts)
    assert model.contexts[0] == Tokenizer().encode("abcdef").ids[-3:]
    assert model.contexts[-1] == [3, 3, 3]
    assert len(result["input_ids"]) == 6 and result["generated_ids"] == [3] * 5


def test_seed_is_local_and_reproduces_sampling_without_global_rng_changes():
    settings = dict(max_new_tokens=12, seed=789, top_k=0, top_p=1, repetition_penalty=1)
    model = Model([[0., 0., -torch.inf, 1., 1., 1.]])
    before = torch.get_rng_state().clone()
    first = generation.generate(model, Tokenizer(), "a", **settings)
    assert torch.equal(before, torch.get_rng_state())
    second = generation.generate(model, Tokenizer(), "a", **settings)
    assert first["generated_ids"] == second["generated_ids"]


@pytest.mark.parametrize("value", [float('nan'), float('inf')])
def test_invalid_raw_logits_raise(value):
    with pytest.raises(ValueError, match="logits"):
        neutral([value, 0.])


@pytest.mark.parametrize("settings", [{"temperature": 0}, {"temperature": float('nan')},
    {"top_p": 0}, {"top_p": 1.1}, {"top_k": -1}, {"top_k": True},
    {"repetition_penalty": .9}, {"no_repeat_ngram_size": -1},
    {"max_new_tokens": -1}, {"seed": 2**64}])
def test_invalid_sampling_parameters_rejected_before_model_forward(settings):
    model = Model([[0.] * 6])
    with pytest.raises(ValueError):
        generation.generate(model, Tokenizer(), "a", **settings)
    assert model.contexts == []


def test_extreme_low_temperature_stays_a_valid_distribution():
    scores = neutral([2., 1., 0.], temperature=1e-40)
    assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
    assert torch.equal(scores.softmax(-1), torch.tensor([1., 0., 0.]))
