"""核心数值契约：未来信息隔离、目标错位、累积与重计算等价。"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from llm_lab.config import ModelConfig
from llm_lab.models import Decoder


def tiny(**updates):
    base = ModelConfig(vocab_size=32, dim=16, n_layers=2, n_heads=4, n_kv_heads=2,
                       ffn_dim=32, max_seq_len=12, dropout=0, activation_checkpointing=False)
    return replace(base, **updates)


def test_causal_attention_cannot_see_future():
    torch.manual_seed(7)
    model = Decoder(tiny()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 9, 10, 11]])
    assert torch.allclose(model(first).logits[:, :3], model(second).logits[:, :3], atol=1e-7)


def test_loss_uses_already_shifted_targets_once():
    model = Decoder(tiny())
    stream = torch.tensor([[1, 2, 3, 4, 5]])
    x, labels = stream[:, :-1], stream[:, 1:]
    output = model(x, labels)
    expected = F.cross_entropy(output.logits.reshape(-1, 32), labels.reshape(-1))
    assert torch.equal(output.loss, expected)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.embedding.weight.grad is not None


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(13)
    full, accumulated = Decoder(tiny()), Decoder(tiny())
    accumulated.load_state_dict(full.state_dict())
    tokens = torch.randint(0, 32, (2, 6))
    labels = torch.randint(0, 32, (2, 6))
    full(tokens, labels).loss.backward()
    for index in range(2):
        (accumulated(tokens[index:index + 1], labels[index:index + 1]).loss / 2).backward()
    for left, right in zip(full.parameters(), accumulated.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=3e-7, rtol=1e-5)


def test_activation_checkpointing_preserves_dropout_gradients():
    torch.manual_seed(19)
    plain = Decoder(tiny(dropout=0.2))
    recomputed = Decoder(tiny(dropout=0.2, activation_checkpointing=True))
    recomputed.load_state_dict(plain.state_dict())
    tokens = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))
    torch.manual_seed(42)
    first = plain(tokens, targets).loss
    first.backward()
    torch.manual_seed(42)
    second = recomputed(tokens, targets).loss
    second.backward()
    assert torch.equal(first, second)
    for left, right in zip(plain.parameters(), recomputed.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=1e-7, rtol=1e-5)


def test_100m_preset_parameter_count_without_allocating_weights():
    with torch.device("meta"):
        model = Decoder(ModelConfig())
    assert model.parameter_count() == 100_092_672


@pytest.mark.parametrize("updates", [{"n_kv_heads": 3}, {"dim": 17}, {"dropout": 1}])
def test_invalid_model_config_rejected(updates):
    with pytest.raises(ValueError):
        Decoder(tiny(**updates))

