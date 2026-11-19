"""Frozen Pydantic domain models.

These are the only data shapes that cross layer boundaries, which keeps
scraping, ingestion, retrieval and chat decoupled from one another.
"""

from datetime import datetime

from pydantic import BaseModel, HttpUrl

from property_assistant.domain.enums import DocType, Role, SourceSite

type MetadataValue = str | int | float | bool | list[str]
type Metadata = dict[str, MetadataValue]


class Document(BaseModel, frozen=True):
    """One cleaned page scraped from a source site."""

    id: str
    site: SourceSite
    url: HttpUrl
    title: str
    content: str
    doc_type: DocType
    metadata: Metadata = {}
    scraped_at: datetime


class Chunk(BaseModel, frozen=True):
    """A retrieval unit: a slice of a document with a context header line."""

    id: str
    doc_id: str
    text: str
    index: int
    site: SourceSite
    url: HttpUrl
    title: str
    doc_type: DocType
    metadata: Metadata = {}


class RetrievedChunk(BaseModel, frozen=True):
    """A chunk returned by a retriever together with its relevance score."""

    chunk: Chunk
    score: float
    retriever: str


class ChatTurn(BaseModel, frozen=True):
    """One message in a conversation."""

    role: Role
    content: str
    created_at: datetime


class Source(BaseModel, frozen=True):
    """A citation shown to the user under an answer."""

    index: int
    title: str
    url: HttpUrl
    site: SourceSite
    snippet: str


class Answer(BaseModel, frozen=True):
    """A complete chat answer with its provenance."""

    text: str
    sources: list[Source]
    model: str
    standalone_query: str
    grounded: bool
    latency_ms: int
