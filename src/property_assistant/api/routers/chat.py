"""Chat endpoints: JSON, SSE streaming and session management."""

import dataclasses
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from sse_starlette.sse import EventSourceResponse

from property_assistant.api.dependencies import get_chat_service, get_conversation_store
from property_assistant.api.ratelimit import chat_rate_limit, limiter
from property_assistant.api.schemas import ChatRequest, ChatResponse
from property_assistant.chat.conversation import ConversationStore
from property_assistant.chat.events import (
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    SourcesEvent,
    TokenEvent,
)
from property_assistant.chat.service import ChatService

router = APIRouter(prefix="/api/v1", tags=["chat"])

_EVENT_NAMES: dict[type, str] = {
    MetaEvent: "meta",
    TokenEvent: "token",
    SourcesEvent: "sources",
    DoneEvent: "done",
    ErrorEvent: "error",
}


def _event_payload(event: object) -> str:
    """Serialize a chat event dataclass (possibly holding Pydantic models) to JSON."""
    data = {
        field.name: getattr(event, field.name)
        for field in dataclasses.fields(event)  # type: ignore[arg-type]
    }
    if isinstance(event, SourcesEvent):
        data["sources"] = [source.model_dump(mode="json") for source in event.sources]
    return json.dumps(data, default=str)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(chat_rate_limit)
async def chat(
    body: ChatRequest,
    request: Request,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Answer a message and return the complete response as JSON."""
    session_id = body.session_id or uuid.uuid4()
    answer = await service.answer(str(session_id), body.message)
    return ChatResponse(
        session_id=session_id,
        answer=answer.text,
        sources=answer.sources,
        model=answer.model,
        grounded=answer.grounded,
        latency_ms=answer.latency_ms,
    )


@router.post("/chat/stream")
@limiter.limit(chat_rate_limit)
async def chat_stream(
    body: ChatRequest,
    request: Request,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> EventSourceResponse:
    """Answer a message as Server-Sent Events (meta, token, sources, done, error)."""
    session_id = body.session_id or uuid.uuid4()

    async def event_source() -> AsyncIterator[dict[str, str]]:
        async for event in service.stream_answer(str(session_id), body.message):
            yield {"event": _EVENT_NAMES[type(event)], "data": _event_payload(event)}

    return EventSourceResponse(event_source())


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    store: Annotated[ConversationStore, Depends(get_conversation_store)],
) -> Response:
    """Clear a conversation's history."""
    await store.clear(str(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
