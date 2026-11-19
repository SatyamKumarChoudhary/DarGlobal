"""Retriever tests with in-memory indexes."""

import logging
from pathlib import Path

from property_assistant.ingestion.chunk_store import ChunkStore
from property_assistant.ingestion.keyword_index import Bm25KeywordIndex
from property_assistant.ingestion.vector_store import FaissVectorStore
from property_assistant.retrieval.hybrid import HybridRetriever
from property_assistant.retrieval.keyword import KeywordRetriever
from property_assistant.retrieval.vector import VectorRetriever
from tests.conftest import FakeEmbedder, FakeRetriever, make_chunk, make_retrieved

logger = logging.getLogger("test")


def build_stores(
    tmp_path: Path,
) -> tuple[FakeEmbedder, FaissVectorStore, Bm25KeywordIndex, ChunkStore]:
    texts = {
        "a:0": "Trump Mansions branded residences in Riyadh",
        "b:0": "AIDA project overlooking Muscat in Oman",
        "c:0": "Wasalt platform for Saudi real estate",
    }
    embedder = FakeEmbedder(dimension=16)
    vector_store = FaissVectorStore(dimension=16)
    ids = list(texts)
    vector_store.add(ids, embedder.embed_documents(list(texts.values())))
    keyword_index = Bm25KeywordIndex()
    keyword_index.build(ids, list(texts.values()))
    chunk_store = ChunkStore(tmp_path)
    chunk_store.save([make_chunk(cid, text=text) for cid, text in texts.items()])
    return embedder, vector_store, keyword_index, chunk_store


class TestVectorRetriever:
    async def test_exact_text_is_top_hit(self, tmp_path: Path) -> None:
        embedder, vector_store, _, chunk_store = build_stores(tmp_path)
        retriever = VectorRetriever(embedder, vector_store, chunk_store)
        results = await retriever.retrieve("Trump Mansions branded residences in Riyadh", 2)
        assert results[0].chunk.id == "a:0"
        assert results[0].retriever == "vector"


class TestKeywordRetriever:
    async def test_keyword_match(self, tmp_path: Path) -> None:
        _, _, keyword_index, chunk_store = build_stores(tmp_path)
        retriever = KeywordRetriever(keyword_index, chunk_store)
        results = await retriever.retrieve("Wasalt Saudi", 2)
        assert results[0].chunk.id == "c:0"
        assert results[0].retriever == "keyword"


class TestHybridRetriever:
    async def test_fuses_and_reports_vector_score(self) -> None:
        vector = FakeRetriever(
            [
                make_retrieved("a:0", score=0.9, retriever="vector"),
                make_retrieved("b:0", score=0.5, retriever="vector"),
            ]
        )
        keyword = FakeRetriever(
            [
                make_retrieved("b:0", score=7.0, retriever="keyword"),
                make_retrieved("c:0", score=3.0, retriever="keyword"),
            ]
        )
        hybrid = HybridRetriever([vector, keyword], candidates=10, rrf_k=60)
        results = await hybrid.retrieve("query", 3)
        assert results[0].chunk.id == "b:0"  # appears in both rankings
        assert results[0].score == 0.5  # best vector cosine, not BM25 score
        assert all(r.retriever == "hybrid" for r in results)
        keyword_only = next(r for r in results if r.chunk.id == "c:0")
        assert keyword_only.score == 0.0

    async def test_k_bounds_results(self) -> None:
        vector = FakeRetriever([make_retrieved(f"v{i}:0") for i in range(5)])
        hybrid = HybridRetriever([vector], candidates=10, rrf_k=60)
        assert len(await hybrid.retrieve("query", 2)) == 2
