"""Query rewriter tests."""

from datetime import UTC, datetime

from property_assistant.chat.rewriter import QueryRewriter
from property_assistant.core.exceptions import LLMError
from property_assistant.domain.enums import Role
from property_assistant.domain.models import ChatTurn
from property_assistant.llm.prompts import PromptBuilder
from tests.conftest import FakeLLM


def _history() -> list[ChatTurn]:
    now = datetime.now(UTC)
    return [
        ChatTurn(role=Role.USER, content="Tell me about AIDA", created_at=now),
        ChatTurn(role=Role.ASSISTANT, content="AIDA is a project in Oman.", created_at=now),
    ]


class TestQueryRewriter:
    def _rewriter(self, llm: FakeLLM) -> QueryRewriter:
        return QueryRewriter(llm, PromptBuilder(max_history_turns=6))

    async def test_no_history_returns_question_without_llm_call(self) -> None:
        llm = FakeLLM()
        rewriter = self._rewriter(llm)
        result = await rewriter.rewrite("What is AIDA?", [])
        assert result == "What is AIDA?"
        assert llm.complete_calls == []

    async def test_with_history_uses_the_llm(self) -> None:
        llm = FakeLLM(response_text="Where is the AIDA project located?")
        rewriter = self._rewriter(llm)
        result = await rewriter.rewrite("where is it?", _history())
        assert result == "Where is the AIDA project located?"
        assert len(llm.complete_calls) == 1

    async def test_llm_error_falls_back_to_concatenation(self) -> None:
        llm = FakeLLM(error=LLMError("boom"))
        rewriter = self._rewriter(llm)
        result = await rewriter.rewrite("where is it?", _history())
        assert result == "Tell me about AIDA where is it?"

    async def test_empty_llm_output_falls_back(self) -> None:
        llm = FakeLLM(response_text="   ")
        rewriter = self._rewriter(llm)
        result = await rewriter.rewrite("where is it?", _history())
        assert result == "Tell me about AIDA where is it?"

    async def test_surrounding_quotes_stripped(self) -> None:
        llm = FakeLLM(response_text='"AIDA location Oman"')
        rewriter = self._rewriter(llm)
        assert await rewriter.rewrite("where?", _history()) == "AIDA location Oman"
