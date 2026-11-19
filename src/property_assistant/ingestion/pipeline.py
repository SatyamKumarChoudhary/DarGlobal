"""The ingestion pipeline facade: clean → chunk → embed → index → persist."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from property_assistant.domain.models import Chunk, Document
from property_assistant.ingestion.chunk_store import ChunkStore
from property_assistant.ingestion.chunker import Chunker
from property_assistant.ingestion.cleaner import TextCleaner
from property_assistant.ingestion.embedder import Embedder
from property_assistant.ingestion.keyword_index import KeywordIndex
from property_assistant.ingestion.vector_store import VectorStore
from property_assistant.scraping.repository import DocumentRepository

MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Summary of one ingestion run."""

    n_documents: int
    n_chunks: int
    by_site: dict[str, int]
    by_doc_type: dict[str, int]
    embedding_model: str
    dimension: int


class IngestionPipeline:
    """Builds the search indexes from the raw document snapshots."""

    def __init__(
        self,
        repository: DocumentRepository,
        cleaner: TextCleaner,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        chunk_store: ChunkStore,
        index_dir: Path,
        embedding_model: str,
        logger: logging.Logger,
    ) -> None:
        """Args:
        repository: Source of raw documents.
        cleaner: Text normalizer (fitted on the whole corpus).
        chunker: Chunking strategy.
        embedder: Embedding strategy.
        vector_store: Dense index to populate.
        keyword_index: Sparse index to populate.
        chunk_store: Chunk persistence.
        index_dir: Output directory for all index artifacts.
        embedding_model: Model name recorded in the manifest.
        logger: Structured logger.
        """
        self._repository = repository
        self._cleaner = cleaner
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._chunk_store = chunk_store
        self._index_dir = index_dir
        self._embedding_model = embedding_model
        self._logger = logger

    def run(self) -> IngestionReport:
        """Execute the full pipeline and persist every artifact.

        Returns:
            A report with corpus statistics.

        Raises:
            ValueError: When no documents are available to ingest.
        """
        documents = self._repository.load_all()
        if not documents:
            raise ValueError("No raw documents found; run the scraper first")
        documents = self._clean_and_dedupe(documents)
        chunks = self._chunk(documents)
        self._logger.info(
            "embedding chunks", extra={"n_docs": len(documents), "n_chunks": len(chunks)}
        )
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        chunk_ids = [chunk.id for chunk in chunks]
        self._vector_store.add(chunk_ids, vectors)
        self._keyword_index.build(chunk_ids, [chunk.text for chunk in chunks])
        self._chunk_store.save(chunks)
        self._vector_store.save(self._index_dir)
        self._keyword_index.save(self._index_dir)
        report = self._build_report(documents, chunks)
        self._write_manifest(report)
        self._logger.info("ingestion complete", extra={"n_chunks": report.n_chunks})
        return report

    def _clean_and_dedupe(self, documents: list[Document]) -> list[Document]:
        """Clean every document and drop empty or duplicate-content ones."""
        self._cleaner.fit([doc.content for doc in documents])
        seen_hashes: set[str] = set()
        kept: list[Document] = []
        for doc in documents:
            cleaned = self._cleaner.clean(doc.content)
            if not cleaned:
                continue
            content_hash = hashlib.sha256(cleaned.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            kept.append(doc.model_copy(update={"content": cleaned}))
        self._logger.info(
            "cleaning complete",
            extra={"n_in": len(documents), "n_kept": len(kept)},
        )
        return kept

    def _chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split every document into chunks."""
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self._chunker.split(doc))
        return chunks

    def _build_report(self, documents: list[Document], chunks: list[Chunk]) -> IngestionReport:
        """Aggregate corpus statistics."""
        by_site: dict[str, int] = {}
        by_doc_type: dict[str, int] = {}
        for doc in documents:
            by_site[doc.site.value] = by_site.get(doc.site.value, 0) + 1
            by_doc_type[doc.doc_type.value] = by_doc_type.get(doc.doc_type.value, 0) + 1
        return IngestionReport(
            n_documents=len(documents),
            n_chunks=len(chunks),
            by_site=by_site,
            by_doc_type=by_doc_type,
            embedding_model=self._embedding_model,
            dimension=self._embedder.dimension,
        )

    def _write_manifest(self, report: IngestionReport) -> None:
        """Write ``manifest.json`` describing the built index."""
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "embedding_model": report.embedding_model,
            "dim": report.dimension,
            "n_docs": report.n_documents,
            "n_chunks": report.n_chunks,
            "by_site": report.by_site,
            "by_doc_type": report.by_doc_type,
        }
        self._index_dir.mkdir(parents=True, exist_ok=True)
        (self._index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
