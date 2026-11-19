"""Hybrid retrieval: concurrent dense + sparse search fused with RRF."""

import asyncio
from collections.abc import Sequence

from property_assistant.domain.models import RetrievedChunk
from property_assistant.retrieval.base import Retriever
from property_assistant.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Runs several retrievers concurrently and fuses their rankings.

    Each chunk's reported ``score`` is its best vector cosine similarity when
    available (used downstream for the grounding check), otherwise 0.0 for
    keyword-only hits.
    """

    def __init__(self, retrievers: Sequence[Retriever], candidates: int, rrf_k: int) -> None:
        """Args:
        retrievers: Sub-retrievers to run (e.g. vector + keyword).
        candidates: How many candidates to pull from each sub-retriever.
        rrf_k: RRF dampening constant.
        """
        self._retrievers = retrievers
        self._candidates = candidates
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Retrieve and fuse.

        Args:
            query: The search query.
            k: Number of fused results to return.

        Returns:
            Top-k fused chunks, best first.
        """
        result_lists = await asyncio.gather(
            *(retriever.retrieve(query, self._candidates) for retriever in self._retrievers)
        )
        rankings = [[rc.chunk.id for rc in results] for results in result_lists]
        fused = reciprocal_rank_fusion(rankings, k=self._rrf_k)

        by_id: dict[str, RetrievedChunk] = {}
        vector_scores: dict[str, float] = {}
        for results in result_lists:
            for retrieved in results:
                by_id.setdefault(retrieved.chunk.id, retrieved)
                if retrieved.retriever == "vector":
                    existing = vector_scores.get(retrieved.chunk.id, float("-inf"))
                    vector_scores[retrieved.chunk.id] = max(existing, retrieved.score)

        top: list[RetrievedChunk] = []
        for chunk_id, _fused_score in fused[:k]:
            base = by_id[chunk_id]
            top.append(
                RetrievedChunk(
                    chunk=base.chunk,
                    score=vector_scores.get(chunk_id, 0.0),
                    retriever="hybrid",
                )
            )
        return top
