import math

import pytest
import torch

from forgelm.model import GPT, Config, RMSNorm, rotate

SMALL = Config(vocab_size=300, dim=32, n_layers=2, n_heads=4, hidden=48, block_size=16)


def model(seed=0, cfg=SMALL):
    torch.manual_seed(seed)
    return GPT(cfg)


def test_logits_have_one_row_per_position_and_one_column_per_token():
    logits, loss = model()(torch.randint(0, 300, (3, 16)))
    assert logits.shape == (3, 16, 300)
    assert loss is None


def test_parameter_count_matches_the_architecture():
    cfg = SMALL
    per_layer = 4 * cfg.dim * cfg.dim + 3 * cfg.dim * cfg.hidden + 2 * cfg.dim
    assert model().n_params() == cfg.vocab_size * cfg.dim + cfg.n_layers * per_layer + cfg.dim


def test_there_are_no_bias_terms_and_no_separate_output_matrix():
    names = [name for name, _ in model().named_parameters()]
    assert not any("bias" in name for name in names)
    assert sum(name == "tok.weight" for name in names) == 1
    assert not any("head" in name or "out" in name for name in names)


def test_a_later_token_cannot_change_an_earlier_prediction():
    net = model().eval()
    ids = torch.randint(0, 300, (1, 16))
    changed = ids.clone()
    changed[0, 10] = (changed[0, 10] + 1) % 300
    with torch.no_grad():
        before, after = net(ids)[0], net(changed)[0]
    assert torch.allclose(before[0, :10], after[0, :10], atol=1e-6)
    assert not torch.allclose(before[0, 10:], after[0, 10:], atol=1e-6)


def test_loss_at_initialization_is_close_to_uniform():
    net = model()
    ids, targets = torch.randint(0, 300, (8, 16)), torch.randint(0, 300, (8, 16))
    loss = net(ids, targets)[1].item()
    assert abs(loss - math.log(300)) < 0.1


def test_initialization_is_reproducible():
    assert torch.equal(model(3).tok.weight, model(3).tok.weight)
    assert not torch.equal(model(3).tok.weight, model(4).tok.weight)


def test_a_sequence_longer_than_the_block_is_refused():
    with pytest.raises(ValueError):
        model()(torch.zeros(1, 17, dtype=torch.long))


def test_config_rejects_shapes_rotary_embeddings_cannot_use():
    with pytest.raises(ValueError):
        Config(dim=30, n_heads=4)
    with pytest.raises(ValueError):
        Config(dim=12, n_heads=4)


def test_rmsnorm_output_has_unit_mean_square():
    out = RMSNorm(64, 0.0)(torch.randn(5, 64) * 7)
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(5), atol=1e-5)


def test_rotation_keeps_length_and_depends_only_on_relative_position():
    torch.manual_seed(0)
    a, b = torch.randn(1, 8), torch.randn(1, 8)
    at = lambda x, pos: rotate(x, torch.tensor([pos]))
    assert torch.allclose(at(a, 17).norm(), a.norm(), atol=1e-5)
    near = (at(a, 5) * at(b, 3)).sum()
    far = (at(a, 9) * at(b, 7)).sum()
    assert torch.allclose(near, far, atol=1e-4)
    assert not torch.allclose(near, (at(a, 5) * at(b, 4)).sum(), atol=1e-4)


def test_generate_appends_tokens_and_is_reproducible_from_a_seed():
    net = model()
    prompt = torch.tensor([[1, 2, 3]])
    first = net.generate(prompt, 30, generator=torch.Generator().manual_seed(5))
    second = net.generate(prompt, 30, generator=torch.Generator().manual_seed(5))
    assert first.shape == (1, 33)
    assert torch.equal(first[:, :3], prompt)
    assert torch.equal(first, second)


def test_top_k_one_is_greedy():
    net = model().eval()
    prompt = torch.tensor([[7, 8]])
    out = net.generate(prompt, 5, temperature=1.0, top_k=1)
    with torch.no_grad():
        expected = net(out[:, :-1])[0][0, -1].argmax().item()
    assert out[0, -1].item() == expected
