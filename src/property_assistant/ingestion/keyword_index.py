"""BM25 keyword index for exact-name matching."""

import pickle
import re
from pathlib import Path
from typing import Protocol

_INDEX_FILE = "bm25.pkl"
_FORMAT_VERSION = 1
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "which",
        "who",
        "how",
        "about",
    }
)


class KeywordIndex(Protocol):
    """Strategy interface for sparse keyword search."""

    def build(self, ids: list[str], texts: list[str]) -> None:
        """Index a corpus of chunk texts."""
        ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top-k ``(chunk_id, score)`` pairs for a query."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the index."""
        ...

    def load(self, directory: Path) -> None:
        """Load a persisted index."""
        ...


class Bm25KeywordIndex:
    """BM25Okapi over lowercased, punctuation-stripped, stopword-filtered tokens.

    Exact project and brand names ("Trump Mansions", "AIDA") rank highly here
    even when the embedding model would miss them.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: object | None = None

    def build(self, ids: list[str], texts: list[str]) -> None:
        """Tokenize and index the corpus.

        Args:
            ids: Chunk ids aligned with ``texts``.
            texts: Chunk texts.
        """
        from rank_bm25 import BM25Okapi  # noqa: PLC0415 - heavyweight import

        self._ids = list(ids)
        self._corpus_tokens = [_tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Score the corpus against a query.

        Args:
            query: Search query.
            k: Number of results.

        Returns:
            ``(chunk_id, bm25_score)`` pairs, best first; zero-score hits dropped.
        """
        if self._bm25 is None or not self._ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)  # type: ignore[attr-defined]
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:k] if score > 0.0]

    def save(self, directory: Path) -> None:
        """Persist the tokenized corpus and ids (the index is rebuilt on load)."""
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "ids": self._ids,
            "corpus_tokens": self._corpus_tokens,
        }
        (directory / _INDEX_FILE).write_bytes(pickle.dumps(payload))

    def load(self, directory: Path) -> None:
        """Load the persisted corpus and rebuild the BM25 structure.

        Raises:
            ValueError: On a format version mismatch.
        """
        from rank_bm25 import BM25Okapi  # noqa: PLC0415 - heavyweight import

        payload = pickle.loads((directory / _INDEX_FILE).read_bytes())
        if payload.get("version") != _FORMAT_VERSION:
            raise ValueError(f"Unsupported BM25 index version: {payload.get('version')}")
        self._ids = payload["ids"]
        self._corpus_tokens = payload["corpus_tokens"]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation and drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
