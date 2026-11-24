# forge-lm

> A small language model built end to end: a BPE tokenizer trained from bytes, a Llama-style transformer and its trainer in PyTorch, a single-file model format, and an inference engine in Rust with no dependencies that reproduces PyTorch's logits to five decimal places.

[![CI](https://github.com/forgehk/forge-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/forgehk/forge-lm/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Nothing here is downloaded pretrained. The tokenizer learns its merges from the training text, the network starts from random weights, and the model in `models/` was trained by the code in this repository in under eleven minutes on two CPU cores.

```console
$ cd engine && cargo run --release -- generate ../models/shakespeare.flm --prompt "First Citizen:" --tokens 120 --seed 7
First Citizen:
We will, as he are, at all the filliful.

CORIOLANUS:
Sir, we will go
Alack their consentsession.

BRUTUS:
Come, come, you almost well fly.

MENENIUS:
One friends, I'll depose him.

CORIOLANUS:
I know it, sir. Can you speak? You know you what
He has for your count
120 tokens in 0.10s (1222 tokens/s)
```

It is 857,216 parameters trained on one megabyte of Shakespeare. It has learned the shape of a play, who speaks in which one, and a lot of words. It has not learned meaning, and it invents some words. That is what a model this size does.

## The pieces

| Piece | File | What it is |
|---|---|---|
| Tokenizer | `forgelm/tokenizer.py` | Byte-level BPE. Starts from the 256 byte values, learns merges from the corpus, never fails to encode, decodes losslessly. Deterministic: ties go to the smallest pair. |
| Model | `forgelm/model.py` | Decoder-only transformer: RMSNorm before each sublayer, rotary position embeddings, SwiGLU feed-forward, no biases, output projection tied to the embedding. Every operation is written out, no `nn.MultiheadAttention`. |
| Trainer | `forgelm/train.py` | AdamW with weight decay on matrices only, linear warmup into a cosine schedule, gradient clipping, a held-out validation split. |
| Model file | `forgelm/export.py` | One `.flm` file: a 32 byte header, the merge table, then every weight as little-endian f32. The layout is documented at the top of the file. |
| Engine | `engine/src/lib.rs` | The forward pass again, by hand, in Rust: embedding, RMSNorm, rotary attention with a key-value cache, SwiGLU, sampling. The only crate in the dependency tree is this one. |

## The two implementations agree

The trainer writes the logits PyTorch computes for a prompt, one row per position. The engine's tests load the same model file, feed the same tokens, and compare:

```console
$ cargo test --release --test parity logits_match -- --nocapture
largest difference from PyTorch over 4608 logits: 1.1570752e-5
```

The tokenizer is held to the same standard. A fixture of awkward inputs (empty, whitespace only, tabs and CRLF, accented Latin, Japanese, an emoji) must encode to identical ids in Python and Rust. Both sides implement the same two rules: cut text into chunks of whitespace then non-whitespace, and apply merges inside a chunk in the order they were learned.

## The released model

| | |
|---|---|
| Parameters | 857,216 |
| Vocabulary | 512 tokens (256 bytes + 256 merges), 2.03 bytes per token on the corpus |
| Shape | 4 layers, 4 heads, width 128, feed-forward 344, context 128 tokens |
| Corpus | Tiny Shakespeare, 1,115,394 bytes, last 10% held out |
| Training | 2,500 steps of 32 x 128 tokens, 639 seconds on 2 CPU cores |
| File | `models/shakespeare.flm`, 3.4 MB |

| Step | Train loss | Validation loss |
|---|---|---|
| 0 | 6.270 | 6.274 |
| 250 | 3.203 | 3.524 |
| 500 | 2.733 | 3.096 |
| 1000 | 2.446 | 3.003 |
| 1500 | 2.218 | 3.060 |
| 2000 | 2.045 | 3.058 |
| 2500 | 1.976 | 3.109 |

Loss is in nats per token. Uniform guessing over 512 tokens is ln 512 = 6.24, which is where it starts.

Read the table honestly: validation loss bottoms out near step 1000 and training loss keeps falling. From there the model is memorizing its megabyte, not generalizing. The released file is the last step, not the best one. A larger corpus is the fix, not more steps. At the best point, 3.00 nats per token is 1.48 nats per byte, about where character-level models of this size land on the same text.

## Use

Generate with the engine:

```bash
cd engine
cargo run --release -- info ../models/shakespeare.flm
cargo run --release -- generate ../models/shakespeare.flm --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top-k 40 --seed 1
cargo run --release -- generate ../models/shakespeare.flm --prompt "ROMEO:" --tokens 300 --window slide
```

Train your own:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python -m forgelm.train --data input.txt --out runs/shakespeare --steps 2500
python -m forgelm.generate runs/shakespeare/model.flm --prompt "ROMEO:"
```

`--out` receives the tokenizer, a PyTorch checkpoint, `model.flm`, the loss log, and the two fixtures the engine's tests read.

## Tests

```bash
pytest -q                        # 45 tests: tokenizer, model, trainer, file format, the released model
cd engine && cargo test --release    # 22 tests: arithmetic, tokenizer, parity with PyTorch, damaged files
```

Some of what they check:

- A later token cannot change an earlier prediction (causality).
- Rotary embeddings preserve length and depend only on relative position, in both languages.
- Training drives the loss below 0.3 on text small enough to memorize, so the optimizer works.
- The model file round-trips bit for bit, and truncated, padded or corrupted files are rejected with an error instead of a panic.
- Generation is reproducible from a seed and runs past the context length.
- With `--window slide`, every token past the context is the one a fresh pass over the previous 128 tokens would sample.

CI runs the Python suite on 3.10 and 3.12 and the engine on Linux and Windows.

## Limits

- The model is tiny and its corpus is one megabyte. It is a complete, working pipeline, not a useful assistant.
- The engine is scalar f32 on one thread. No SIMD, no batching, no quantization. It is fast only because the model is small.
- When the context fills, the engine by default keeps the second half and re-feeds it from position zero, so it samples differently from PyTorch past 128 tokens. `--window slide` predicts every token from exactly the last 128, as PyTorch's `generate` does (a 260-token greedy run from `First Citizen:` matches PyTorch byte for byte), at about 26 tokens/s instead of 1,100, because every token past the context recomputes the whole window.
- Training is CPU only as written.

## License

MIT. The corpus is Shakespeare, in the public domain, and is not stored in this repository.
