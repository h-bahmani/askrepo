"""Lexical (BM25) search over a set of code chunks.

Dense embeddings would rank near-duplicate phrasing better, but they need a
model download and a vector store. BM25 needs neither, runs in milliseconds
on a laptop, and is a solid baseline for the identifier- and keyword-heavy
text that source code actually is. See the README's Limitations section.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from askrepo.chunking import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Split identifiers/words out of code or prose, lowercased."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class CodeIndex:
    """A searchable BM25 index over a fixed list of chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if chunks else None

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # Classic BM25 IDF can go negative for a term that appears in most of a
        # *small* corpus (e.g. a handful of chunks in a test repo) -- a real match,
        # just not a "rare" one. So results are ranked, not filtered by sign.
        query_tokens = tokenize(query)
        if not self.chunks or not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [SearchResult(chunk=c, score=float(s)) for c, s in ranked[:top_k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: Path) -> CodeIndex:
        with path.open("rb") as f:
            chunks: list[Chunk] = pickle.load(f)
        return cls(chunks)
