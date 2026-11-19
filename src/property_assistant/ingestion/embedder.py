"""Embedding strategies."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float32]

_BATCH_SIZE = 64


class Embedder(Protocol):
    """Strategy interface for text embedding."""

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        """Embed a batch of passages into an ``(n, dim)`` float32 matrix."""
        ...

    def embed_query(self, text: str) -> FloatArray:
        """Embed a search query into a ``(dim,)`` float32 vector."""
        ...


class FastEmbedEmbedder:
    """ONNX-based local embeddings via fastembed (no PyTorch, no API cost).

    Vectors are L2-normalized float32, so inner-product search equals cosine
    similarity. ``query_embed`` applies the BGE query instruction prefix.
    """

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        """Args:
        model_name: fastembed model identifier, e.g. ``BAAI/bge-small-en-v1.5``.
        cache_dir: Optional model download/cache directory.
        """
        from fastembed import TextEmbedding  # noqa: PLC0415 - heavyweight import

        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        self._dimension = self._probe_dimension()

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        """Embed passages in batches.

        Args:
            texts: Passage texts.

        Returns:
            L2-normalized ``(n, dim)`` float32 matrix.
        """
        vectors = list(self._model.embed(list(texts), batch_size=_BATCH_SIZE))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> FloatArray:
        """Embed one query with the model's query-side instruction.

        Args:
            text: The search query.

        Returns:
            L2-normalized ``(dim,)`` float32 vector.
        """
        vector = next(iter(self._model.query_embed(text)))
        matrix = _normalize(np.asarray([vector], dtype=np.float32))
        return np.asarray(matrix[0], dtype=np.float32)

    def _probe_dimension(self) -> int:
        """Determine the output dimension with a tiny embed call."""
        probe = next(iter(self._model.embed(["probe"])))
        return int(np.asarray(probe).shape[0])


def _normalize(matrix: FloatArray) -> FloatArray:
    """L2-normalize matrix rows, guarding against zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)
