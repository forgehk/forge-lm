"""The single-file model format the Rust engine reads.

One file holds everything inference needs: the shape of the network, the
tokenizer's merge table and every weight. All integers are little-endian u32
and all weights are little-endian f32, row-major.

    magic       4 bytes  "FLM1"
    header      6 x u32  vocab_size, dim, n_layers, n_heads, hidden, block_size
    eps         f32
    n_merges    u32
    merges      n_merges x (u32 left, u32 right)      token id = 256 + position
    tok         [vocab_size, dim]
    per layer   norm1 [dim], wq wk wv wo [dim, dim], norm2 [dim],
                w1 [hidden, dim], w2 [dim, hidden], w3 [hidden, dim]
    norm        [dim]

The output projection is the token embedding, so it is not stored twice.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

from .model import GPT, Config
from .tokenizer import Tokenizer

MAGIC = b"FLM1"


def _tensors(model: GPT) -> list[torch.Tensor]:
    out = [model.tok.weight]
    for block in model.blocks:
        out += [block.norm1.weight, block.wq.weight, block.wk.weight, block.wv.weight, block.wo.weight,
                block.norm2.weight, block.w1.weight, block.w2.weight, block.w3.weight]
    out.append(model.norm.weight)
    return out


def write(model: GPT, tokenizer: Tokenizer, path: str | Path) -> int:
    """Write the model to ``path`` and return the number of bytes written."""
    cfg = model.cfg
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError("the tokenizer and the model disagree about the vocabulary size")
    parts = [MAGIC, struct.pack("<6I", cfg.vocab_size, cfg.dim, cfg.n_layers, cfg.n_heads, cfg.hidden, cfg.block_size),
             struct.pack("<f", cfg.eps), struct.pack("<I", len(tokenizer.merges))]
    parts += [struct.pack("<2I", left, right) for left, right in tokenizer.merges]
    parts += [tensor.detach().to(torch.float32).cpu().numpy().astype("<f4").tobytes() for tensor in _tensors(model)]
    data = b"".join(parts)
    Path(path).write_bytes(data)
    return len(data)


def read(path: str | Path) -> tuple[GPT, Tokenizer]:
    """Load a model written by ``write``. Used by the tests to prove the format round-trips."""
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError("not a forge-lm model file")
    vocab_size, dim, n_layers, n_heads, hidden, block_size = struct.unpack_from("<6I", data, 4)
    (eps,) = struct.unpack_from("<f", data, 28)
    (n_merges,) = struct.unpack_from("<I", data, 32)
    offset = 36
    merges = [struct.unpack_from("<2I", data, offset + 8 * i) for i in range(n_merges)]
    offset += 8 * n_merges
    # eps was stored as an f32; recover the short decimal it was written from.
    model = GPT(Config(vocab_size, dim, n_layers, n_heads, hidden, block_size, float(f"{eps:.6g}")))
    with torch.no_grad():
        for tensor in _tensors(model):
            count = tensor.numel()
            values = np.frombuffer(data, dtype="<f4", count=count, offset=offset).reshape(tuple(tensor.shape))
            tensor.copy_(torch.from_numpy(values.copy()))
            offset += 4 * count
    if offset != len(data):
        raise ValueError(f"{len(data) - offset} unexpected trailing bytes in the model file")
    return model, Tokenizer([tuple(pair) for pair in merges])
