"""Vector index storage."""

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from property_assistant.ingestion.embedder import FloatArray

_INDEX_FILE = "faiss.index"
_IDS_FILE = "ids.json"


class VectorStore(Protocol):
    """Strategy interface for dense vector search."""

    def add(self, ids: list[str], vectors: FloatArray) -> None:
        """Add vectors with their chunk ids."""
        ...

    def search(self, vector: FloatArray, k: int) -> list[tuple[str, float]]:
        """Return the top-k ``(chunk_id, score)`` pairs for a query vector."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the index to a directory."""
        ...

    def load(self, directory: Path) -> None:
        """Load the index from a directory."""
        ...

    def __len__(self) -> int:
        """Number of stored vectors."""
        ...


class FaissVectorStore:
    """Exact cosine search with a FAISS flat inner-product index.

    Vectors must be L2-normalized so inner product equals cosine similarity.
    Chunk ids are kept in a parallel list persisted as ``ids.json``.
    """

    def __init__(self, dimension: int) -> None:
        """Args:
        dimension: Embedding dimensionality.
        """
        import faiss  # noqa: PLC0415 - heavyweight import

        self._faiss = faiss
        self._index: faiss.Index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []

    def add(self, ids: list[str], vectors: FloatArray) -> None:
        """Add normalized vectors and their chunk ids.

        Args:
            ids: Chunk ids, aligned with ``vectors`` rows.
            vectors: ``(n, dim)`` float32 matrix.
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have equal length")
        self._index.add(vectors)
        self._ids.extend(ids)

    def search(self, vector: FloatArray, k: int) -> list[tuple[str, float]]:
        """Search the index.

        Args:
            vector: Normalized ``(dim,)`` query vector.
            k: Number of neighbours.

        Returns:
            ``(chunk_id, cosine_score)`` pairs, best first.
        """
        if not self._ids:
            return []
        query = np.asarray([vector], dtype=np.float32)
        scores, indices = self._index.search(query, min(k, len(self._ids)))
        return [
            (self._ids[idx], float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
            if idx >= 0
        ]

    def save(self, directory: Path) -> None:
        """Write ``faiss.index`` and ``ids.json`` into a directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(directory / _INDEX_FILE))
        (directory / _IDS_FILE).write_text(json.dumps(self._ids))

    def load(self, directory: Path) -> None:
        """Load the index files written by :meth:`save`."""
        self._index = self._faiss.read_index(str(directory / _INDEX_FILE))
        self._ids = json.loads((directory / _IDS_FILE).read_text())

    def __len__(self) -> int:
        """Number of stored vectors."""
        return len(self._ids)
