"""Prompt builder tests."""

from datetime import UTC, datetime

from property_assistant.domain.enums import Role
from property_assistant.domain.models import ChatTurn
from property_assistant.llm.prompts import SYSTEM_PROMPT, PromptBuilder
from tests.conftest import make_retrieved


def _turn(content: str, role: Role = Role.USER) -> ChatTurn:
    return ChatTurn(role=role, content=content, created_at=datetime.now(UTC))


class TestPromptBuilder:
    def setup_method(self) -> None:
        self.builder = PromptBuilder(max_history_turns=4)

    def test_system_prompt_first(self) -> None:
        messages = self.builder.build_answer_messages("q", [make_retrieved()], [])
        assert messages[0].role is Role.SYSTEM
        assert messages[0].content == SYSTEM_PROMPT

    def test_context_entries_are_numbered(self) -> None:
        context = [
            make_retrieved("a:0", title="Alpha", url="https://example.com/a"),
            make_retrieved("b:0", title="Beta", url="https://example.com/b"),
        ]
        messages = self.builder.build_answer_messages("q", context, [])
        user = messages[-1].content
        assert "[1] (Alpha — https://example.com/a)" in user
        assert "[2] (Beta — https://example.com/b)" in user
        assert user.rstrip().endswith("QUESTION: q")

    def test_history_window_applied(self) -> None:
        history = [_turn(f"msg{i}") for i in range(10)]
        messages = self.builder.build_answer_messages("q", [make_retrieved()], history)
        history_contents = [m.content for m in messages[1:-1]]
        assert history_contents == ["msg6", "msg7", "msg8", "msg9"]

    def test_context_trimmed_at_the_char_budget(self) -> None:
        big_text = "[H | project | T]\n" + "x" * 15_000
        context = [
            make_retrieved("a:0", text=big_text),
            make_retrieved("b:0", text=big_text),
            make_retrieved("c:0", text=big_text),
        ]
        messages = self.builder.build_answer_messages("q", context, [])
        user = messages[-1].content
        assert "[1]" in user
        assert "[3]" not in user  # lowest-ranked trimmed first

    def test_empty_context_is_explicit(self) -> None:
        messages = self.builder.build_answer_messages("q", [], [])
        assert "(no relevant context found)" in messages[-1].content

    def test_rewrite_messages_contain_history_and_question(self) -> None:
        history = [_turn("Tell me about AIDA")]
        messages = self.builder.build_rewrite_messages("where is it?", history)
        assert len(messages) == 2
        assert "Tell me about AIDA" in messages[1].content
        assert "where is it?" in messages[1].content
