"""Dense vector retrieval."""

import asyncio

from property_assistant.domain.models import RetrievedChunk
from property_assistant.ingestion.chunk_store import ChunkStore
from property_assistant.ingestion.embedder import Embedder
from property_assistant.ingestion.vector_store import VectorStore


class VectorRetriever:
    """Embeds the query and searches the FAISS index (cosine similarity)."""

    def __init__(
        self, embedder: Embedder, vector_store: VectorStore, chunk_store: ChunkStore
    ) -> None:
        """Args:
        embedder: Query embedding strategy.
        vector_store: Dense index.
        chunk_store: Chunk lookup by id.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_store = chunk_store

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Retrieve the k nearest chunks by cosine similarity.

        The ONNX embedding call is CPU-bound, so it runs in a worker thread
        to keep the event loop responsive.

        Args:
            query: The search query.
            k: Number of results.

        Returns:
            Retrieved chunks, best first.
        """
        vector = await asyncio.to_thread(self._embedder.embed_query, query)
        hits = self._vector_store.search(vector, k)
        results: list[RetrievedChunk] = []
        for chunk_id, score in hits:
            try:
                chunk = self._chunk_store.get(chunk_id)
            except KeyError:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=score, retriever="vector"))
        return results
