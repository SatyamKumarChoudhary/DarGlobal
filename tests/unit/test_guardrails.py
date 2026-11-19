"""Input guard tests."""

import pytest

from property_assistant.chat.guardrails import InputGuard
from property_assistant.core.exceptions import GuardrailViolation


class TestInputGuard:
    def test_valid_message_passes_trimmed(self) -> None:
        guard = InputGuard(max_message_chars=100)
        assert guard.validate("  hello  ") == "hello"

    def test_empty_message_rejected(self) -> None:
        guard = InputGuard(max_message_chars=100)
        with pytest.raises(GuardrailViolation):
            guard.validate("   ")

    def test_too_long_message_rejected(self) -> None:
        guard = InputGuard(max_message_chars=10)
        with pytest.raises(GuardrailViolation):
            guard.validate("x" * 11)

    def test_control_characters_stripped(self) -> None:
        guard = InputGuard(max_message_chars=100)
        assert guard.validate("he\x00llo\x1f") == "hello"

    def test_newlines_and_tabs_preserved(self) -> None:
        guard = InputGuard(max_message_chars=100)
        assert guard.validate("line one\nline two\ttabbed") == "line one\nline two\ttabbed"

    def test_message_of_exactly_max_length_passes(self) -> None:
        guard = InputGuard(max_message_chars=5)
        assert guard.validate("abcde") == "abcde"
