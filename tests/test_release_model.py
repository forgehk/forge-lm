"""The model checked into models/ must load and reproduce its own reference values."""

import json
from pathlib import Path

import torch

from forgelm import export

MODELS = Path(__file__).resolve().parent.parent / "models"


def load():
    net, tokenizer = export.read(MODELS / "shakespeare.flm")
    return net.eval(), tokenizer


def test_the_released_model_reproduces_its_reference_logits():
    net, _ = load()
    lines = (MODELS / "reference_logits.txt").read_text().splitlines()
    ids = [int(token) for token in lines[0].split()]
    expected = torch.tensor([[float(value) for value in line.split()] for line in lines[1:]])
    with torch.no_grad():
        logits = net(torch.tensor([ids]))[0][0]
    assert torch.allclose(logits, expected, atol=1e-4)


def test_the_released_tokenizer_matches_its_cases():
    _, tokenizer = load()
    for row in (MODELS / "tokenizer_cases.txt").read_text().splitlines():
        text_hex, encoded = row.split("\t")
        text = b"" if text_hex == "-" else bytes.fromhex(text_hex)
        expected = [] if encoded == "-" else [int(token) for token in encoded.split()]
        assert tokenizer.encode(text) == expected
        assert tokenizer.decode_bytes(expected) == text


def test_the_training_log_describes_the_released_model():
    net, tokenizer = load()
    log = json.loads((MODELS / "training_log.json").read_text())
    assert log["parameters"] == net.n_params()
    assert log["config"]["vocab_size"] == tokenizer.vocab_size
    assert log["model_bytes"] == (MODELS / "shakespeare.flm").stat().st_size
    assert log["history"][-1]["val"] < log["history"][0]["val"] - 2.0


def test_the_released_model_writes_text_that_looks_like_its_training_data():
    net, tokenizer = load()
    prompt = torch.tensor([tokenizer.encode("ROMEO:\n")])
    out = net.generate(prompt, 80, temperature=0.8, top_k=40, generator=torch.Generator().manual_seed(0))
    text = tokenizer.decode(out[0].tolist())
    assert text.startswith("ROMEO:\n")
    words = text.split()
    assert len(words) > 20
    common = {"the", "and", "to", "of", "I", "my", "you", "a", "in", "is", "not", "that", "me", "with", "thou", "thy"}
    assert sum(word.strip(",.;:!?").lower() in {w.lower() for w in common} for word in words) >= 5
