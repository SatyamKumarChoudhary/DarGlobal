"""The chat orchestrator."""

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from property_assistant.chat.cache import AnswerCache
from property_assistant.chat.citations import CitationFormatter
from property_assistant.chat.conversation import ConversationStore
from property_assistant.chat.events import (
    ChatEvent,
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    SourcesEvent,
    TokenEvent,
)
from property_assistant.chat.guardrails import InputGuard
from property_assistant.chat.rewriter import QueryRewriter
from property_assistant.core.config import Settings
from property_assistant.core.exceptions import LLMError
from property_assistant.domain.enums import Role
from property_assistant.domain.models import Answer, ChatTurn, RetrievedChunk, Source
from property_assistant.llm.base import LLMClient
from property_assistant.llm.prompts import PromptBuilder
from property_assistant.retrieval.base import Retriever

NOT_GROUNDED_MESSAGE = (
    "I couldn't find that in the Dar Global / Wasalt data I have. "
    "You could try asking:\n"
    "- What projects does Dar Global have, and where?\n"
    "- Which luxury brands has Dar Global partnered with?\n"
    "- What is Wasalt and what services does it offer?"
)
DEGRADED_MESSAGE = (
    "The AI model is temporarily busy (free tier). Here are the most relevant pages I found:"
)
_FALLBACK_MODEL_NAME = "none"


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Intermediate state shared by the sync and streaming paths."""

    message: str
    history: list[ChatTurn]
    standalone_query: str
    context: list[RetrievedChunk]
    grounded: bool
    cached: Answer | None


class ChatService:
    """Facade that orchestrates one chat turn end to end."""

    def __init__(
        self,
        guard: InputGuard,
        store: ConversationStore,
        rewriter: QueryRewriter,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm: LLMClient,
        citations: CitationFormatter,
        cache: AnswerCache,
        settings: Settings,
    ) -> None:
        """Args:
        guard: Input validation.
        store: Conversation memory.
        rewriter: Follow-up → standalone query rewriting.
        retriever: Hybrid retriever.
        prompt_builder: Grounded prompt assembly.
        llm: The LLM client.
        citations: Citation → source mapping.
        cache: Standalone-answer cache.
        settings: Application settings.
        """
        self._guard = guard
        self._store = store
        self._rewriter = rewriter
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._citations = citations
        self._cache = cache
        self._settings = settings
        self._logger = logging.getLogger(__name__)

    async def answer(self, session_id: str, message: str) -> Answer:
        """Produce a complete answer (non-streaming path).

        Args:
            session_id: Conversation identifier.
            message: Raw user message.

        Returns:
            The full answer with sources.

        Raises:
            GuardrailViolation: On invalid input.
        """
        started = time.perf_counter()
        prepared = await self._prepare(session_id, message)
        if prepared.cached is not None:
            return prepared.cached
        if not prepared.grounded:
            answer = self._not_grounded_answer(prepared, started)
            await self._persist_turn(session_id, prepared.message, answer.text)
            return answer
        try:
            result = await self._llm.complete(
                self._prompt_builder.build_answer_messages(
                    prepared.standalone_query, prepared.context, prepared.history
                )
            )
            text, model = result.text, result.model
            sources = self._citations.to_sources(prepared.context, text)
        except LLMError:
            text, model, sources = self._degraded(prepared)
        answer = Answer(
            text=text,
            sources=sources,
            model=model,
            standalone_query=prepared.standalone_query,
            grounded=prepared.grounded,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        await self._persist_turn(session_id, prepared.message, answer.text)
        self._maybe_cache(prepared, answer)
        return answer

    async def stream_answer(self, session_id: str, message: str) -> AsyncIterator[ChatEvent]:
        """Produce an answer as a stream of events.

        Args:
            session_id: Conversation identifier.
            message: Raw user message.

        Yields:
            ``MetaEvent``, then ``TokenEvent``s, then ``SourcesEvent`` and
            ``DoneEvent``; an ``ErrorEvent`` plus fallback tokens on LLM
            failure. The stream never dies silently.
        """
        started = time.perf_counter()
        prepared = await self._prepare(session_id, message)
        yield MetaEvent(session_id=session_id, standalone_query=prepared.standalone_query)

        if prepared.cached is not None:
            yield TokenEvent(delta=prepared.cached.text)
            yield SourcesEvent(sources=prepared.cached.sources)
            yield DoneEvent(
                model=prepared.cached.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                grounded=prepared.cached.grounded,
            )
            return

        if not prepared.grounded:
            answer = self._not_grounded_answer(prepared, started)
            yield TokenEvent(delta=answer.text)
            yield SourcesEvent(sources=answer.sources)
            yield DoneEvent(model=answer.model, latency_ms=answer.latency_ms, grounded=False)
            await self._persist_turn(session_id, prepared.message, answer.text)
            return

        collected: list[str] = []
        model = _FALLBACK_MODEL_NAME
        try:
            messages = self._prompt_builder.build_answer_messages(
                prepared.standalone_query, prepared.context, prepared.history
            )
            async for chunk in self._llm.stream(messages):
                if chunk.model:
                    model = chunk.model
                collected.append(chunk.delta)
                yield TokenEvent(delta=chunk.delta)
            text = "".join(collected)
            sources = self._citations.to_sources(prepared.context, text)
        except LLMError as exc:
            yield ErrorEvent(code=exc.code, message=exc.message)
            text, model, sources = self._degraded(prepared)
            yield TokenEvent(delta=text)

        latency_ms = int((time.perf_counter() - started) * 1000)
        yield SourcesEvent(sources=sources)
        yield DoneEvent(model=model, latency_ms=latency_ms, grounded=prepared.grounded)
        await self._persist_turn(session_id, prepared.message, text)
        answer = Answer(
            text=text,
            sources=sources,
            model=model,
            standalone_query=prepared.standalone_query,
            grounded=prepared.grounded,
            latency_ms=latency_ms,
        )
        self._maybe_cache(prepared, answer)

    async def _prepare(self, session_id: str, message: str) -> _Prepared:
        """Shared pre-LLM pipeline: validate → history → rewrite → retrieve."""
        validated = self._guard.validate(message)
        history = await self._store.get_history(session_id, self._settings.max_history_turns)
        query = await self._rewriter.rewrite(validated, history)
        cached = self._cache.get(query) if not history else None
        if cached is not None:
            self._logger.info("answer cache hit", extra={"query": query[:80]})
            return _Prepared(
                message=validated,
                history=history,
                standalone_query=query,
                context=[],
                grounded=cached.grounded,
                cached=cached,
            )
        context = await self._retriever.retrieve(query, self._settings.retrieval_top_k)
        grounded = any(c.score >= self._settings.min_relevance_score for c in context)
        self._logger.info(
            "retrieval done",
            extra={"query": query[:80], "n_ctx": len(context), "grounded": grounded},
        )
        return _Prepared(
            message=validated,
            history=history,
            standalone_query=query,
            context=context,
            grounded=grounded,
            cached=None,
        )

    def _not_grounded_answer(self, prepared: _Prepared, started: float) -> Answer:
        """Build the polite no-data answer without calling the LLM."""
        return Answer(
            text=NOT_GROUNDED_MESSAGE,
            sources=[],
            model=_FALLBACK_MODEL_NAME,
            standalone_query=prepared.standalone_query,
            grounded=False,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _degraded(self, prepared: _Prepared) -> tuple[str, str, list[Source]]:
        """Build the graceful-degradation answer from the top sources."""
        sources = self._citations.to_sources(prepared.context, "")
        lines = [DEGRADED_MESSAGE]
        for source in sources:
            lines.append(f"- {source.title}: {source.url}")
        return "\n".join(lines), _FALLBACK_MODEL_NAME, sources

    async def _persist_turn(self, session_id: str, message: str, answer_text: str) -> None:
        """Append the user and assistant turns to the conversation store."""
        now = datetime.now(UTC)
        await self._store.append(
            session_id, ChatTurn(role=Role.USER, content=message, created_at=now)
        )
        await self._store.append(
            session_id, ChatTurn(role=Role.ASSISTANT, content=answer_text, created_at=now)
        )

    def _maybe_cache(self, prepared: _Prepared, answer: Answer) -> None:
        """Cache grounded, history-free answers produced by a real model."""
        if not prepared.history and answer.grounded and answer.model != _FALLBACK_MODEL_NAME:
            self._cache.put(prepared.standalone_query, answer)
