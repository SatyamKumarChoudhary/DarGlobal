"""Discriminated union of streaming chat events."""

from dataclasses import dataclass

from property_assistant.domain.models import Source


@dataclass(frozen=True, slots=True)
class MetaEvent:
    """Sent first: the resolved session and standalone query."""

    session_id: str
    standalone_query: str


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One streamed answer delta."""

    delta: str


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """The citations for the finished answer."""

    sources: list[Source]


@dataclass(frozen=True, slots=True)
class DoneEvent:
    """Sent last on success."""

    model: str
    latency_ms: int
    grounded: bool


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """Sent when the LLM fails; a fallback answer follows as tokens."""

    code: str
    message: str


type ChatEvent = MetaEvent | TokenEvent | SourcesEvent | DoneEvent | ErrorEvent
