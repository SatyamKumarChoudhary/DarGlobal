"""Shared test fixtures and fakes. No test hits the network."""

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import HttpUrl

from property_assistant.core.exceptions import LLMError
from property_assistant.domain.enums import DocType, SourceSite
from property_assistant.domain.models import Chunk, Document, RetrievedChunk
from property_assistant.llm.base import LLMMessage, LLMResult, StreamChunk


def make_document(
    doc_id: str = "doc1",
    site: SourceSite = SourceSite.DARGLOBAL,
    content: str = "Some content about a luxury real-estate project located in Dubai, UAE.",
    title: str = "Test Project",
    doc_type: DocType = DocType.PROJECT,
    metadata: dict | None = None,
) -> Document:
    """Build a valid Document with sensible defaults."""
    return Document(
        id=doc_id,
        site=site,
        url=HttpUrl(f"https://example.com/{doc_id}"),
        title=title,
        content=content,
        doc_type=doc_type,
        metadata=metadata or {},
        scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def make_chunk(
    chunk_id: str = "doc1:0",
    text: str = "[Dar Global | project | Test Project]\nSome content.",
    title: str = "Test Project",
    site: SourceSite = SourceSite.DARGLOBAL,
    url: str = "https://example.com/doc1",
) -> Chunk:
    """Build a valid Chunk with sensible defaults."""
    doc_id, _, index = chunk_id.partition(":")
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        index=int(index or 0),
        site=site,
        url=HttpUrl(url),
        title=title,
        doc_type=DocType.PROJECT,
    )


def make_retrieved(
    chunk_id: str = "doc1:0",
    score: float = 0.8,
    retriever: str = "hybrid",
    **chunk_kwargs: str,
) -> RetrievedChunk:
    """Build a RetrievedChunk around a default chunk."""
    return RetrievedChunk(
        chunk=make_chunk(chunk_id, **chunk_kwargs), score=score, retriever=retriever
    )


class FakeLLM:
    """Scripted LLM: returns canned responses or raises scripted errors."""

    def __init__(
        self,
        response_text: str = "A grounded answer [1].",
        stream_deltas: Sequence[str] | None = None,
        error: LLMError | None = None,
        model: str = "fake/model:free",
    ) -> None:
        self.response_text = response_text
        self.stream_deltas = list(stream_deltas or ["A grounded ", "answer [1]."])
        self.error = error
        self.model = model
        self.complete_calls: list[Sequence[LLMMessage]] = []
        self.stream_calls: list[Sequence[LLMMessage]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        self.complete_calls.append(messages)
        if self.error is not None:
            raise self.error
        return LLMResult(
            text=self.response_text, model=self.model, prompt_tokens=10, completion_tokens=5
        )

    async def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls.append(messages)
        if self.error is not None:
            raise self.error
        for delta in self.stream_deltas:
            yield StreamChunk(delta=delta, model=self.model)


class FakeRetriever:
    """Returns a scripted list of retrieved chunks."""

    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results if results is not None else [make_retrieved()]
        self.queries: list[str] = []

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.results[:k]


class FakeEmbedder:
    """Deterministic hash-based embeddings (no model download)."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> "np.ndarray":
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> "np.ndarray":
        return self._vector(text)

    def _vector(self, text: str) -> "np.ndarray":
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.standard_normal(self._dimension).astype(np.float32)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A default scripted LLM."""
    return FakeLLM()


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    """A default scripted retriever."""
    return FakeRetriever()
