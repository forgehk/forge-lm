"""Train a model from a text file.

    python -m forgelm.train --data shakespeare.txt --out runs/shakespeare --steps 3000

Writes the tokenizer, a PyTorch checkpoint, the exported model file, a log of the
training and validation loss, and the fixtures the Rust engine's tests compare
against.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from . import export
from .model import GPT, Config
from .tokenizer import Tokenizer


def batch(data: torch.Tensor, block: int, size: int, generator: torch.Generator):
    starts = torch.randint(len(data) - block - 1, (size,), generator=generator)
    x = torch.stack([data[s:s + block] for s in starts])
    y = torch.stack([data[s + 1:s + block + 1] for s in starts])
    return x, y


def learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup, then a cosine down to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: GPT, data: torch.Tensor, block: int, size: int, batches: int, generator: torch.Generator) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = batch(data, block, size, generator)
        total += model(x, y)[1].item()
    model.train()
    return total / batches


def fit(model: GPT, train: torch.Tensor, val: torch.Tensor, *, steps: int, batch_size: int, lr: float,
        warmup: int = 100, weight_decay: float = 0.1, eval_every: int = 250, seed: int = 0, log=print) -> list[dict]:
    """Run the optimization loop and return the loss log."""
    generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed + 1)
    # Weight decay on matrices only: decaying the norm gains just fights the normalization.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    plain = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )
    block = model.cfg.block_size
    history: list[dict] = []
    started = time.time()
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            record = {
                "step": step,
                "train": round(evaluate(model, train, block, batch_size, 8, eval_generator), 4),
                "val": round(evaluate(model, val, block, batch_size, 8, eval_generator), 4),
                "seconds": round(time.time() - started, 1),
            }
            history.append(record)
            log(f"step {step:5d}  train {record['train']:.4f}  val {record['val']:.4f}  {record['seconds']:.0f}s")
            if step == steps:
                break
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, lr, warmup)
        x, y = batch(train, block, batch_size, generator)
        loss = model(x, y)[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return history


def write_fixtures(model: GPT, tokenizer: Tokenizer, out: Path, prompt: str) -> None:
    """Reference values the Rust engine must reproduce: token ids and logits for a prompt."""
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    lines = [" ".join(map(str, ids))]
    lines += [" ".join(f"{value:.6f}" for value in row) for row in logits[0].tolist()]
    (out / "reference_logits.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cases = [prompt, "Hello, world!", "  leading and trailing  ", "tabs\tand\nnewlines\r\n", "naïve café — 日本語 🙂",
             "ROMEO:\nBut, soft! what light through yonder window breaks?", "", "a", "   "]
    rows = [f"{case.encode('utf-8').hex() or '-'}\t{' '.join(map(str, tokenizer.encode(case))) or '-'}" for case in cases]
    (out / "tokenizer_cases.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a forge-lm model from a text file.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=344)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="ROMEO:\nWhat light")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    raw = Path(args.data).read_bytes()

    started = time.time()
    tokenizer = Tokenizer.train(raw, args.vocab)
    tokenizer.save(out / "tokenizer.json")
    ids = torch.tensor(tokenizer.encode(raw), dtype=torch.long)
    print(f"tokenizer: {tokenizer.vocab_size} tokens, {len(raw)} bytes -> {len(ids)} tokens "
          f"({len(raw) / len(ids):.2f} bytes per token) in {time.time() - started:.0f}s")

    split = int(len(ids) * 0.9)
    model = GPT(Config(tokenizer.vocab_size, args.dim, args.layers, args.heads, args.hidden, args.block))
    print(f"model: {model.n_params():,} parameters")
    history = fit(model, ids[:split], ids[split:], steps=args.steps, batch_size=args.batch, lr=args.lr, seed=args.seed)

    torch.save({"config": model.cfg.to_dict(), "state": model.state_dict()}, out / "model.pt")
    size = export.write(model, tokenizer, out / "model.flm")
    write_fixtures(model, tokenizer, out, args.prompt)
    summary = {
        "config": model.cfg.to_dict(), "parameters": model.n_params(), "data_bytes": len(raw), "data_tokens": len(ids),
        "steps": args.steps, "batch": args.batch, "lr": args.lr, "seed": args.seed, "model_bytes": size, "history": history,
    }
    (out / "training_log.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'model.flm'} ({size:,} bytes)")


if __name__ == "__main__":
    main()
