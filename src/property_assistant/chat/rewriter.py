"""Follow-up question rewriting."""

import logging

from property_assistant.core.exceptions import LLMError
from property_assistant.domain.enums import Role
from property_assistant.domain.models import ChatTurn
from property_assistant.llm.base import LLMClient
from property_assistant.llm.prompts import PromptBuilder

_REWRITE_MAX_TOKENS = 60


class QueryRewriter:
    """Turns follow-up messages into standalone search queries.

    Costs zero LLM calls for first messages (returned unchanged) and degrades
    to simple history concatenation when the LLM fails, protecting the free
    quota and keeping the chat responsive.
    """

    def __init__(self, llm: LLMClient, prompt_builder: PromptBuilder) -> None:
        """Args:
        llm: The LLM used for rewriting.
        prompt_builder: Builds the rewrite prompt.
        """
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._logger = logging.getLogger(__name__)

    async def rewrite(self, question: str, history: list[ChatTurn]) -> str:
        """Produce a standalone query for retrieval.

        Args:
            question: The raw user message.
            history: Prior conversation turns (may be empty).

        Returns:
            The question unchanged when there is no history; otherwise the
            LLM rewrite, or a concatenation fallback on LLM failure.
        """
        if not history:
            return question
        messages = self._prompt_builder.build_rewrite_messages(question, history)
        try:
            result = await self._llm.complete(
                messages, max_tokens=_REWRITE_MAX_TOKENS, temperature=0.0
            )
        except LLMError as exc:
            self._logger.warning("rewrite failed, using fallback", extra={"error": exc.message})
            return self._fallback(question, history)
        rewritten = result.text.strip().strip('"')
        return rewritten or self._fallback(question, history)

    @staticmethod
    def _fallback(question: str, history: list[ChatTurn]) -> str:
        """Concatenate the last user message with the follow-up."""
        last_user = next((turn.content for turn in reversed(history) if turn.role is Role.USER), "")
        return f"{last_user} {question}".strip()
