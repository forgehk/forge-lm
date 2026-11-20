"""Byte-level BPE tokenizer, trained from scratch.

The base vocabulary is the 256 byte values, so any input can be encoded and
decoding is lossless. Training repeatedly merges the most frequent adjacent pair
of tokens into a new token.

Text is first cut into chunks (a run of whitespace followed by a run of
non-whitespace) and merges never cross a chunk boundary. That keeps training
fast, because identical chunks are counted once, and it makes encoding a
function of one chunk at a time. The Rust engine implements the same two rules,
and a shared fixture checks that both produce identical token ids.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

CHUNK = re.compile(rb"[ \n\t\r]*[^ \n\t\r]+|[ \n\t\r]+")
BASE = 256


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self.merges = [tuple(pair) for pair in merges]
        self.rank = {pair: index for index, pair in enumerate(self.merges)}
        self.vocab = [bytes([value]) for value in range(BASE)]
        for left, right in self.merges:
            self.vocab.append(self.vocab[left] + self.vocab[right])
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return BASE + len(self.merges)

    @classmethod
    def train(cls, data: bytes, vocab_size: int) -> "Tokenizer":
        """Learn ``vocab_size - 256`` merges from ``data``."""
        if vocab_size < BASE:
            raise ValueError("vocab_size cannot be smaller than the 256 byte values")
        words = [(list(chunk), count) for chunk, count in Counter(CHUNK.findall(data)).items()]
        merges: list[tuple[int, int]] = []
        while BASE + len(merges) < vocab_size:
            pairs: Counter[tuple[int, int]] = Counter()
            for tokens, count in words:
                for pair in zip(tokens, tokens[1:]):
                    pairs[pair] += count
            if not pairs:
                break
            # Most frequent pair; ties go to the numerically smallest pair so training is deterministic.
            best = max(pairs.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))[0]
            new_id = BASE + len(merges)
            merges.append(best)
            for tokens, _ in words:
                if len(tokens) > 1:
                    _merge(tokens, best, new_id)
        return cls(merges)

    def encode(self, text: str | bytes) -> list[int]:
        data = text.encode("utf-8") if isinstance(text, str) else text
        ids: list[int] = []
        for chunk in CHUNK.findall(data):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk)
                self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        tokens = list(chunk)
        while len(tokens) > 1:
            # Apply merges in the order they were learned: lowest rank first.
            ranked = [self.rank.get(pair) for pair in zip(tokens, tokens[1:])]
            known = [rank for rank in ranked if rank is not None]
            if not known:
                break
            rank = min(known)
            _merge(tokens, self.merges[rank], BASE + rank)
        return tokens

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(self.vocab[token] for token in ids)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": [list(pair) for pair in self.merges]}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(pair) for pair in data["merges"]])


def _merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> None:
    """Replace every non-overlapping occurrence of ``pair``, left to right, in place."""
    left, right = pair
    write = read = 0
    size = len(tokens)
    while read < size:
        if read + 1 < size and tokens[read] == left and tokens[read + 1] == right:
            tokens[write] = new_id
            read += 2
        else:
            tokens[write] = tokens[read]
            read += 1
        write += 1
    del tokens[write:]
