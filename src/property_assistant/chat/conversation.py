"""Conversation memory.

``InMemoryConversationStore`` suits a single-process deployment. A Redis-backed
implementation can drop in through the same protocol for multi-worker scaling.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from property_assistant.domain.models import ChatTurn


class ConversationStore(Protocol):
    """Strategy interface for per-session chat history."""

    async def get_history(self, session_id: str, limit: int) -> list[ChatTurn]:
        """Return the most recent turns for a session, oldest first."""
        ...

    async def append(self, session_id: str, turn: ChatTurn) -> None:
        """Append a turn to a session."""
        ...

    async def clear(self, session_id: str) -> None:
        """Delete a session's history."""
        ...


@dataclass(slots=True)
class _Session:
    turns: deque[ChatTurn]
    last_used: float = field(default_factory=time.monotonic)


class InMemoryConversationStore:
    """Dict-backed store with per-session TTL and a bounded turn deque."""

    def __init__(self, ttl_minutes: int, max_turns: int) -> None:
        """Args:
        ttl_minutes: Idle minutes after which a session expires (lazy).
        max_turns: Maximum turns kept per session.
        """
        self._ttl_s = ttl_minutes * 60
        self._max_turns = max_turns
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    async def get_history(self, session_id: str, limit: int) -> list[ChatTurn]:
        """Return up to ``limit`` most recent turns, oldest first."""
        async with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_used = time.monotonic()
            turns = list(session.turns)
        return turns[-limit:] if limit else turns

    async def append(self, session_id: str, turn: ChatTurn) -> None:
        """Append a turn, creating the session when needed."""
        async with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session(turns=deque(maxlen=self._max_turns))
                self._sessions[session_id] = session
            session.turns.append(turn)
            session.last_used = time.monotonic()

    async def clear(self, session_id: str) -> None:
        """Delete a session's history (no-op when unknown)."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    def _expire(self) -> None:
        """Drop sessions idle beyond the TTL. Caller must hold the lock."""
        cutoff = time.monotonic() - self._ttl_s
        expired = [sid for sid, s in self._sessions.items() if s.last_used < cutoff]
        for session_id in expired:
            del self._sessions[session_id]
