import struct

import pytest
import torch

from forgelm import export
from forgelm.model import GPT, Config
from forgelm.tokenizer import Tokenizer
from forgelm.train import batch, fit, learning_rate, write_fixtures

TEXT = b"to be, or not to be, that is the question: whether 'tis nobler in the mind to suffer. " * 40


def tiny():
    torch.manual_seed(0)
    tokenizer = Tokenizer.train(TEXT, 300)
    net = GPT(Config(vocab_size=300, dim=48, n_layers=2, n_heads=4, hidden=64, block_size=24))
    ids = torch.tensor(tokenizer.encode(TEXT), dtype=torch.long)
    return net, tokenizer, ids


def test_learning_rate_warms_up_then_decays_to_a_tenth():
    rates = [learning_rate(step, 1000, 1e-3, 100) for step in range(1000)]
    assert rates[0] < rates[50] < rates[99]
    assert rates[99] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(rates[100:], rates[101:]))
    assert rates[-1] == pytest.approx(1e-4, rel=0.05)


def test_targets_are_the_inputs_shifted_by_one():
    data = torch.arange(1000)
    x, y = batch(data, 16, 4, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(y, x + 1)


def test_training_drives_the_loss_down_on_text_it_can_memorize():
    net, _, ids = tiny()
    history = fit(net, ids, ids, steps=300, batch_size=16, lr=3e-3, warmup=20, eval_every=150, log=lambda _: None)
    assert history[0]["train"] > 5.0
    assert history[-1]["train"] < 0.3
    assert [record["step"] for record in history] == [0, 150, 300]


def test_the_model_file_round_trips_exactly(tmp_path):
    net, tokenizer, ids = tiny()
    size = export.write(net, tokenizer, tmp_path / "m.flm")
    assert size == 4 + 6 * 4 + 4 + 4 + 8 * len(tokenizer.merges) + 4 * net.n_params()

    loaded, loaded_tokenizer = export.read(tmp_path / "m.flm")
    assert loaded.cfg == net.cfg
    assert loaded_tokenizer.merges == tokenizer.merges
    net.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(net(ids[None, :24])[0], loaded(ids[None, :24])[0])


def test_the_file_starts_with_the_magic_and_the_header(tmp_path):
    net, tokenizer, _ = tiny()
    export.write(net, tokenizer, tmp_path / "m.flm")
    data = (tmp_path / "m.flm").read_bytes()
    assert data[:4] == b"FLM1"
    assert struct.unpack_from("<6I", data, 4) == (300, 48, 2, 4, 64, 24)


def test_a_tokenizer_of_the_wrong_size_is_refused(tmp_path):
    net, _, _ = tiny()
    with pytest.raises(ValueError):
        export.write(net, Tokenizer([]), tmp_path / "m.flm")


def test_damaged_files_are_refused(tmp_path):
    net, tokenizer, _ = tiny()
    export.write(net, tokenizer, tmp_path / "m.flm")
    data = (tmp_path / "m.flm").read_bytes()
    (tmp_path / "magic.flm").write_bytes(b"NOPE" + data[4:])
    (tmp_path / "long.flm").write_bytes(data + b"\x00\x00\x00\x00")
    with pytest.raises(ValueError):
        export.read(tmp_path / "magic.flm")
    with pytest.raises(ValueError):
        export.read(tmp_path / "long.flm")


def test_fixtures_hold_the_prompt_ids_and_one_logit_row_per_position(tmp_path):
    net, tokenizer, _ = tiny()
    write_fixtures(net, tokenizer, tmp_path, "to be, or not")
    lines = (tmp_path / "reference_logits.txt").read_text().splitlines()
    ids = [int(token) for token in lines[0].split()]
    assert ids == tokenizer.encode("to be, or not")
    assert len(lines) == 1 + len(ids)
    assert all(len(line.split()) == 300 for line in lines[1:])

    for row in (tmp_path / "tokenizer_cases.txt").read_text().splitlines():
        text_hex, encoded = row.split("\t")
        text = b"" if text_hex == "-" else bytes.fromhex(text_hex)
        expected = [] if encoded == "-" else [int(token) for token in encoded.split()]
        assert tokenizer.encode(text) == expected
