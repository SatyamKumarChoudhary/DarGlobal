"""Request/response DTOs for the public API."""

from uuid import UUID

from pydantic import BaseModel, Field

from property_assistant.domain.models import Source


class ChatRequest(BaseModel):
    """A chat message from the client."""

    session_id: UUID | None = Field(
        default=None, description="Omit on the first message; the server assigns one."
    )
    message: str = Field(description="The user's question.")


class ChatResponse(BaseModel):
    """A complete (non-streamed) chat answer."""

    session_id: UUID
    answer: str
    sources: list[Source]
    model: str
    grounded: bool
    latency_ms: int


class StatsResponse(BaseModel):
    """Corpus and model statistics."""

    n_documents: int
    n_chunks: int
    by_site: dict[str, int]
    by_doc_type: dict[str, int]
    indexed_at: str
    embedding_model: str
    llm_models: list[str]


class SuggestionsResponse(BaseModel):
    """Starter questions shown in the empty chat state."""

    suggestions: list[str]
