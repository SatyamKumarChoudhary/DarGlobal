"""ChatService orchestration tests with fakes (no network, no index)."""

import pytest

from property_assistant.chat.cache import AnswerCache
from property_assistant.chat.citations import CitationFormatter
from property_assistant.chat.conversation import InMemoryConversationStore
from property_assistant.chat.events import (
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    SourcesEvent,
    TokenEvent,
)
from property_assistant.chat.guardrails import InputGuard
from property_assistant.chat.rewriter import QueryRewriter
from property_assistant.chat.service import NOT_GROUNDED_MESSAGE, ChatService
from property_assistant.core.config import Settings
from property_assistant.core.exceptions import GuardrailViolation, LLMRateLimitError
from property_assistant.llm.prompts import PromptBuilder
from tests.conftest import FakeLLM, FakeRetriever, make_retrieved


def make_settings() -> Settings:
    return Settings(
        openrouter_api_key="sk-or-test",  # type: ignore[arg-type]
        llm_primary_model="primary/model:free",
        min_relevance_score=0.30,
        _env_file=None,  # type: ignore[call-arg]
    )


def make_service(
    llm: FakeLLM | None = None, retriever: FakeRetriever | None = None
) -> tuple[ChatService, FakeLLM, FakeRetriever]:
    llm = llm or FakeLLM()
    retriever = retriever or FakeRetriever()
    settings = make_settings()
    prompt_builder = PromptBuilder(settings.max_history_turns)
    service = ChatService(
        guard=InputGuard(settings.max_message_chars),
        store=InMemoryConversationStore(ttl_minutes=10, max_turns=12),
        rewriter=QueryRewriter(llm, prompt_builder),
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        citations=CitationFormatter(),
        cache=AnswerCache(ttl_s=60, max_size=16),
        settings=settings,
    )
    return service, llm, retriever


class TestAnswer:
    async def test_grounded_path_calls_llm_and_cites(self) -> None:
        service, llm, _retriever = make_service()
        answer = await service.answer("s1", "What is the test project?")
        assert answer.text == "A grounded answer [1]."
        assert answer.grounded is True
        assert answer.model == "fake/model:free"
        assert len(answer.sources) == 1
        assert len(llm.complete_calls) == 1

    async def test_not_grounded_skips_the_llm(self) -> None:
        retriever = FakeRetriever([make_retrieved(score=0.1)])
        service, llm, _ = make_service(retriever=retriever)
        answer = await service.answer("s1", "Unrelated question")
        assert answer.grounded is False
        assert answer.text == NOT_GROUNDED_MESSAGE
        assert llm.complete_calls == []

    async def test_degradation_on_rate_limit(self) -> None:
        llm = FakeLLM(error=LLMRateLimitError("quota"))
        service, _, _ = make_service(llm=llm)
        answer = await service.answer("s1", "What is the test project?")
        assert "temporarily busy" in answer.text
        assert answer.sources  # fallback sources still provided
        assert answer.model == "none"

    async def test_cache_hit_skips_retrieval_and_llm(self) -> None:
        service, llm, retriever = make_service()
        first = await service.answer("s1", "What is the test project?")
        # Different session, same standalone question → cache hit.
        second = await service.answer("s2", "What is the test project?")
        assert second.text == first.text
        assert len(llm.complete_calls) == 1
        assert len(retriever.queries) == 1

    async def test_invalid_input_raises(self) -> None:
        service, _, _ = make_service()
        with pytest.raises(GuardrailViolation):
            await service.answer("s1", "   ")

    async def test_turns_persisted(self) -> None:
        service, _, _ = make_service()
        await service.answer("s1", "Question one?")
        history = await service._store.get_history("s1", 10)
        assert len(history) == 2


class TestStreamAnswer:
    async def test_event_order_on_grounded_path(self) -> None:
        service, _, _ = make_service()
        events = [e async for e in service.stream_answer("s1", "What is the test project?")]
        types = [type(e) for e in events]
        assert types[0] is MetaEvent
        assert TokenEvent in types
        assert types[-2] is SourcesEvent
        assert types[-1] is DoneEvent
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.grounded is True

    async def test_not_grounded_streams_polite_message(self) -> None:
        retriever = FakeRetriever([make_retrieved(score=0.05)])
        service, llm, _ = make_service(retriever=retriever)
        events = [e async for e in service.stream_answer("s1", "Mystery?")]
        tokens = [e.delta for e in events if isinstance(e, TokenEvent)]
        assert "".join(tokens) == NOT_GROUNDED_MESSAGE
        assert llm.stream_calls == []

    async def test_error_event_then_fallback_on_llm_failure(self) -> None:
        llm = FakeLLM(error=LLMRateLimitError("quota"))
        service, _, _ = make_service(llm=llm)
        events = [e async for e in service.stream_answer("s1", "What is the test project?")]
        types = [type(e) for e in events]
        error_index = types.index(ErrorEvent)
        assert TokenEvent in types[error_index:]
        assert types[-1] is DoneEvent

    async def test_cached_answer_replayed_as_events(self) -> None:
        service, llm, _ = make_service()
        await service.answer("s1", "What is the test project?")
        events = [e async for e in service.stream_answer("s2", "What is the test project?")]
        types = [type(e) for e in events]
        assert types == [MetaEvent, TokenEvent, SourcesEvent, DoneEvent]
        assert len(llm.stream_calls) == 0
