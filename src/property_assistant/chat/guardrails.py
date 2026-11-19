"""Input validation guardrails."""

import re

from property_assistant.core.exceptions import GuardrailViolation

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputGuard:
    """Validates and sanitizes incoming chat messages."""

    def __init__(self, max_message_chars: int) -> None:
        """Args:
        max_message_chars: Maximum accepted message length.
        """
        self._max_message_chars = max_message_chars

    def validate(self, message: str) -> str:
        """Trim, strip control characters and enforce length bounds.

        Args:
            message: The raw user message.

        Returns:
            The sanitized message.

        Raises:
            GuardrailViolation: When the message is empty or too long.
        """
        cleaned = _CONTROL_CHARS_RE.sub("", message).strip()
        if not cleaned:
            raise GuardrailViolation("Message must not be empty")
        if len(cleaned) > self._max_message_chars:
            raise GuardrailViolation(
                f"Message exceeds the maximum length of {self._max_message_chars} characters"
            )
        return cleaned
