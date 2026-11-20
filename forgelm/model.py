"""A decoder-only transformer in the style of the Llama family.

Compared with the original GPT-2 block:

* RMSNorm before each sublayer in place of LayerNorm after it;
* rotary position embeddings applied to queries and keys in place of learned
  position vectors, so position enters attention as a relative rotation;
* a SwiGLU feed-forward layer, ``w2(silu(w1 x) * w3 x)``;
* no bias terms, and the output projection shares its weights with the token
  embedding.

Everything here is written out explicitly (no nn.MultiheadAttention, no
scaled_dot_product_attention) because the Rust engine has to reproduce each
operation, and a hidden default is a parity bug waiting to happen.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

ROPE_BASE = 10000.0


@dataclass(frozen=True)
class Config:
    vocab_size: int = 512
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    hidden: int = 344
    block_size: int = 128
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("the head dimension must be even for rotary embeddings")

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotary embedding. ``x`` is (..., time, head_dim); pair (2i, 2i+1) turns by pos * base^(-2i/head_dim)."""
    head_dim = x.shape[-1]
    freqs = ROPE_BASE ** (-torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim)
    angles = positions.to(torch.float32)[:, None] * freqs[None, :]
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.norm1 = RMSNorm(cfg.dim, cfg.eps)
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.norm2 = RMSNorm(cfg.dim, cfg.eps)
        self.w1 = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.w2 = nn.Linear(cfg.hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, time, dim = x.shape
        head_dim = dim // self.n_heads
        h = self.norm1(x)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, time, self.n_heads, head_dim).transpose(1, 2)

        q = rotate(heads(self.wq(h)), positions)
        k = rotate(heads(self.wk(h)), positions)
        v = heads(self.wv(h))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(mask, float("-inf"))
        attended = F.softmax(scores, dim=-1) @ v
        x = x + self.wo(attended.transpose(1, 2).reshape(batch, time, dim))

        h = self.norm2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class GPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.eps)
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            # Layers that write into the residual stream start smaller, so depth does not inflate it.
            scale = 1 / math.sqrt(2 * cfg.n_layers) if name.endswith(("wo.weight", "w2.weight")) else 1.0
            nn.init.normal_(param, mean=0.0, std=0.02 * scale)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        _, time = ids.shape
        if time > self.cfg.block_size:
            raise ValueError(f"sequence of {time} tokens is longer than the block size {self.cfg.block_size}")
        positions = torch.arange(time, device=ids.device)
        mask = torch.triu(torch.ones(time, time, dtype=torch.bool, device=ids.device), diagonal=1)
        x = self.tok(ids)
        for block in self.blocks:
            x = block(x, positions, mask)
        logits = self.norm(x) @ self.tok.weight.T
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_tokens: int, temperature: float = 0.8, top_k: int | None = 40,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(n_tokens):
            window = ids[:, -self.cfg.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            choice = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            ids = torch.cat([ids, choice], dim=1)
        return ids
