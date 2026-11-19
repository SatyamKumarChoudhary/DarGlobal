"""Conversation store tests."""

import time
from datetime import UTC, datetime

from property_assistant.chat.conversation import InMemoryConversationStore
from property_assistant.domain.enums import Role
from property_assistant.domain.models import ChatTurn


def turn(content: str, role: Role = Role.USER) -> ChatTurn:
    return ChatTurn(role=role, content=content, created_at=datetime.now(UTC))


class TestInMemoryConversationStore:
    async def test_round_trip(self) -> None:
        store = InMemoryConversationStore(ttl_minutes=10, max_turns=10)
        await store.append("s1", turn("hello"))
        await store.append("s1", turn("hi there", Role.ASSISTANT))
        history = await store.get_history("s1", limit=10)
        assert [t.content for t in history] == ["hello", "hi there"]

    async def test_unknown_session_returns_empty(self) -> None:
        store = InMemoryConversationStore(ttl_minutes=10, max_turns=10)
        assert await store.get_history("nope", limit=10) == []

    async def test_limit_returns_most_recent(self) -> None:
        store = InMemoryConversationStore(ttl_minutes=10, max_turns=10)
        for i in range(5):
            await store.append("s1", turn(f"msg{i}"))
        history = await store.get_history("s1", limit=2)
        assert [t.content for t in history] == ["msg3", "msg4"]

    async def test_max_turns_bound(self) -> None:
        store = InMemoryConversationStore(ttl_minutes=10, max_turns=3)
        for i in range(6):
            await store.append("s1", turn(f"msg{i}"))
        history = await store.get_history("s1", limit=100)
        assert [t.content for t in history] == ["msg3", "msg4", "msg5"]

    async def test_clear(self) -> None:
        store = InMemoryConversationStore(ttl_minutes=10, max_turns=10)
        await store.append("s1", turn("hello"))
        await store.clear("s1")
        assert await store.get_history("s1", limit=10) == []

    async def test_ttl_expiry(self, monkeypatch) -> None:
        store = InMemoryConversationStore(ttl_minutes=1, max_turns=10)
        await store.append("s1", turn("hello"))
        original = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: original + 120)
        assert await store.get_history("s1", limit=10) == []

    async def test_sessions_are_isolated(self) -> None:
        store = InMemoryConversationStore(ttl_minutes=10, max_turns=10)
        await store.append("s1", turn("one"))
        await store.append("s2", turn("two"))
        assert [t.content for t in await store.get_history("s1", 10)] == ["one"]
        assert [t.content for t in await store.get_history("s2", 10)] == ["two"]
