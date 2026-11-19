"""Ingestion component tests: cleaner, stores, indexes and pipeline."""

import logging
from pathlib import Path

import numpy as np
import pytest

from property_assistant.domain.enums import SourceSite
from property_assistant.ingestion.chunk_store import ChunkStore
from property_assistant.ingestion.chunker import RecursiveTextChunker
from property_assistant.ingestion.cleaner import TextCleaner
from property_assistant.ingestion.keyword_index import Bm25KeywordIndex
from property_assistant.ingestion.pipeline import IngestionPipeline
from property_assistant.ingestion.vector_store import FaissVectorStore
from property_assistant.scraping.repository import DocumentRepository
from tests.conftest import FakeEmbedder, make_chunk, make_document

logger = logging.getLogger("test")


class TestTextCleaner:
    def test_unicode_normalized_and_whitespace_collapsed(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("ﬁne   text\u00a0here") == "fine text here"

    def test_boilerplate_lines_removed(self) -> None:
        cleaner = TextCleaner()
        text = "Real content line.\nWe use cookies on this site.\nDownload the app now!"
        assert cleaner.clean(text) == "Real content line."

    def test_frequent_lines_learned_and_dropped(self) -> None:
        cleaner = TextCleaner()
        nav = "Home About Projects Contact"
        corpus = [f"{nav}\nUnique content {i}." for i in range(20)]
        cleaner.fit(corpus)
        assert cleaner.clean(corpus[0]) == "Unique content 0."

    def test_small_corpus_skips_frequency_learning(self) -> None:
        cleaner = TextCleaner()
        cleaner.fit(["same line", "same line"])
        assert cleaner.clean("same line") == "same line"


class TestFaissVectorStore:
    def test_add_search_round_trip(self) -> None:
        embedder = FakeEmbedder(dimension=8)
        store = FaissVectorStore(dimension=8)
        texts = ["alpha", "beta", "gamma"]
        store.add([f"c{i}" for i in range(3)], embedder.embed_documents(texts))
        hits = store.search(embedder.embed_query("alpha"), k=1)
        assert hits[0][0] == "c0"
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dimension=8)
        store = FaissVectorStore(dimension=8)
        store.add(["x"], embedder.embed_documents(["hello"]))
        store.save(tmp_path)
        restored = FaissVectorStore(dimension=8)
        restored.load(tmp_path)
        assert len(restored) == 1
        assert restored.search(embedder.embed_query("hello"), 1)[0][0] == "x"

    def test_empty_store_returns_nothing(self) -> None:
        store = FaissVectorStore(dimension=8)
        assert store.search(np.zeros(8, dtype=np.float32), 5) == []

    def test_mismatched_lengths_rejected(self) -> None:
        store = FaissVectorStore(dimension=8)
        with pytest.raises(ValueError, match="equal length"):
            store.add(["a", "b"], np.zeros((1, 8), dtype=np.float32))


class TestBm25KeywordIndex:
    def test_exact_name_ranks_first(self) -> None:
        index = Bm25KeywordIndex()
        index.build(
            ["c1", "c2", "c3"],
            [
                "Trump Mansions is a branded residence in Riyadh",
                "AIDA is a project in Muscat Oman",
                "Generic page about services",
            ],
        )
        assert index.search("Trump Mansions", k=2)[0][0] == "c1"

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        index = Bm25KeywordIndex()
        index.build(
            ["c1", "c2", "c3"],
            ["unique vocabulary here", "different words entirely", "third filler text"],
        )
        index.save(tmp_path)
        restored = Bm25KeywordIndex()
        restored.load(tmp_path)
        assert restored.search("vocabulary", 1)[0][0] == "c1"

    def test_empty_query_tokens(self) -> None:
        index = Bm25KeywordIndex()
        index.build(["c1"], ["some text"])
        assert index.search("the and of", 5) == []

    def test_unbuilt_index_returns_nothing(self) -> None:
        assert Bm25KeywordIndex().search("anything", 5) == []


class TestChunkStore:
    def test_save_load_get(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path)
        chunks = [make_chunk("a:0"), make_chunk("a:1")]
        store.save(chunks)
        restored = ChunkStore(tmp_path)
        restored.load()
        assert len(restored) == 2
        assert restored.get("a:0").id == "a:0"
        assert [c.id for c in restored.get_many(["a:1", "missing"])] == ["a:1"]

    def test_get_unknown_raises(self, tmp_path: Path) -> None:
        store = ChunkStore(tmp_path)
        store.save([])
        with pytest.raises(KeyError):
            store.get("nope")


class TestIngestionPipeline:
    def test_end_to_end_with_fakes(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        index_dir = tmp_path / "index"
        repo = DocumentRepository(raw_dir, logger)
        repo.save_all(
            SourceSite.DARGLOBAL,
            [
                make_document("a", content="First document body. " * 30),
                make_document("b", content="Second document body. " * 30),
                make_document("dup", content="First document body. " * 30),
            ],
        )
        embedder = FakeEmbedder(dimension=8)
        pipeline = IngestionPipeline(
            repository=repo,
            cleaner=TextCleaner(),
            chunker=RecursiveTextChunker(size=400, overlap=50),
            embedder=embedder,
            vector_store=FaissVectorStore(dimension=8),
            keyword_index=Bm25KeywordIndex(),
            chunk_store=ChunkStore(index_dir),
            index_dir=index_dir,
            embedding_model="fake-model",
            logger=logger,
        )
        report = pipeline.run()
        assert report.n_documents == 2  # duplicate content dropped
        assert report.n_chunks > 0
        assert (index_dir / "manifest.json").exists()
        assert (index_dir / "faiss.index").exists()
        assert (index_dir / "bm25.pkl").exists()
        assert (index_dir / "chunks.jsonl").exists()

    def test_no_documents_raises(self, tmp_path: Path) -> None:
        pipeline = IngestionPipeline(
            repository=DocumentRepository(tmp_path / "raw", logger),
            cleaner=TextCleaner(),
            chunker=RecursiveTextChunker(size=400, overlap=50),
            embedder=FakeEmbedder(),
            vector_store=FaissVectorStore(dimension=8),
            keyword_index=Bm25KeywordIndex(),
            chunk_store=ChunkStore(tmp_path / "index"),
            index_dir=tmp_path / "index",
            embedding_model="fake-model",
            logger=logger,
        )
        with pytest.raises(ValueError, match="No raw documents"):
            pipeline.run()
