"""Sparse keyword retrieval."""

from property_assistant.domain.models import RetrievedChunk
from property_assistant.ingestion.chunk_store import ChunkStore
from property_assistant.ingestion.keyword_index import KeywordIndex


class KeywordRetriever:
    """Searches the BM25 index for exact-term matches."""

    def __init__(self, keyword_index: KeywordIndex, chunk_store: ChunkStore) -> None:
        """Args:
        keyword_index: Sparse index.
        chunk_store: Chunk lookup by id.
        """
        self._keyword_index = keyword_index
        self._chunk_store = chunk_store

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Retrieve the top-k BM25 matches.

        Args:
            query: The search query.
            k: Number of results.

        Returns:
            Retrieved chunks, best first.
        """
        hits = self._keyword_index.search(query, k)
        results: list[RetrievedChunk] = []
        for chunk_id, score in hits:
            try:
                chunk = self._chunk_store.get(chunk_id)
            except KeyError:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=score, retriever="keyword"))
        return results
