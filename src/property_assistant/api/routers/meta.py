"""Public, non-secret metadata endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from property_assistant.api.dependencies import get_container
from property_assistant.api.schemas import StatsResponse, SuggestionsResponse
from property_assistant.container import AppContainer

router = APIRouter(prefix="/api/v1", tags=["meta"])

_SUGGESTIONS = [
    "What projects does Dar Global have in Oman?",
    "Tell me about Trump Mansions in Riyadh.",
    "Which luxury brands has Dar Global partnered with?",
    "What is Wasalt and what services does it offer?",
    "Show me villas for sale in Riyadh on Wasalt.",
    "What did Dar Global announce recently?",
]


@router.get("/stats", response_model=StatsResponse)
async def stats(
    container: Annotated[AppContainer, Depends(get_container)],
) -> StatsResponse:
    """Corpus statistics from the index manifest."""
    manifest = container.manifest
    return StatsResponse(
        n_documents=manifest.get("n_docs", 0),
        n_chunks=manifest.get("n_chunks", 0),
        by_site=manifest.get("by_site", {}),
        by_doc_type=manifest.get("by_doc_type", {}),
        indexed_at=manifest.get("created_at", ""),
        embedding_model=manifest.get("embedding_model", ""),
        llm_models=container.settings.llm_models,
    )


@router.get("/suggestions", response_model=SuggestionsResponse)
async def suggestions() -> SuggestionsResponse:
    """Starter questions for the empty chat state."""
    return SuggestionsResponse(suggestions=_SUGGESTIONS)
