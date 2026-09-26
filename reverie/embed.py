"""Embedding providers.

The default is a deterministic hashing embedder: no model download, no API
key, no network, byte-identical across machines and runs. That last property
matters more than raw quality here -- it makes every experiment in
``experiments/`` exactly reproducible, and it keeps HLD G5 (runs fully local
with no API key) literally true rather than aspirationally true.

It captures lexical, not semantic, similarity. The FTS path already covers
lexical matching, so on a real deployment this should be swapped for a small
ONNX sentence encoder; the interface is one method wide.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence, runtime_checkable

__all__ = ["Embedder", "HashingEmbedder", "get_embedder"]

_TOKEN = re.compile(r"[a-z0-9_]+")

# Tokens too common to carry signal. Deliberately tiny -- an aggressive stop
# list would silently break domain vocabulary like "run" or "test".
_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been it its this that "
    "with by from as if then than so we you i".split()
)


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Bag-of-tokens hashed into a fixed-width vector, L2 normalised.

    Uses signed hashing (the sign bit of the digest decides +1/-1) so that
    unrelated tokens colliding into one bucket cancel on average instead of
    accumulating a spurious positive similarity.
    """

    def __init__(self, dim: int = 64, use_bigrams: bool = True) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"
        self.use_bigrams = use_bigrams

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        if not tokens:
            return vec

        for tok in tokens:
            idx, sign = self._hash(tok)
            vec[idx] += sign

        if self.use_bigrams:
            for a, b in zip(tokens, tokens[1:]):
                idx, sign = self._hash(f"{a}_{b}")
                vec[idx] += sign * 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class ConstantEmbedder:
    """Zero vectors. Used to isolate the FTS-only path in ablations."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.name = f"constant-{dim}"

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


_REGISTRY: dict[str, type] = {"hash": HashingEmbedder, "constant": ConstantEmbedder}


def get_embedder(spec: str = "hash-64") -> Embedder:
    kind, _, dim = spec.partition("-")
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown embedder {spec!r}; known: {sorted(_REGISTRY)}")
    return cls(dim=int(dim) if dim else 64)
