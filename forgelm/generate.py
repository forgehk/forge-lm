"""Generate text with PyTorch from an exported model file.

    python -m forgelm.generate models/shakespeare.flm --prompt "ROMEO:" --tokens 200
"""

from __future__ import annotations

import argparse

import torch

from . import export


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a forge-lm model file.")
    parser.add_argument("model")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = export.read(args.model)
    prompt = torch.tensor([tokenizer.encode(args.prompt)])
    out = model.generate(prompt, args.tokens, args.temperature, args.top_k, torch.Generator().manual_seed(args.seed))
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
