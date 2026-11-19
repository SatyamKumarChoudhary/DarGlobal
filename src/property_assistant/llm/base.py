"""LLM client protocol and message/result value types."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from property_assistant.domain.enums import Role


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One chat-completion message."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResult:
    """A complete (non-streamed) LLM response."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One streamed delta."""

    delta: str
    model: str


class LLMClient(Protocol):
    """Strategy interface for chat-completion backends."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """Run a full completion and return the final text."""
        ...

    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion token by token."""
        ...
