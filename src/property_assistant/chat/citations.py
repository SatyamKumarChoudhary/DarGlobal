"""Citation extraction and source mapping."""

import re

from property_assistant.domain.models import RetrievedChunk, Source

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_SNIPPET_CHARS = 200
_DEFAULT_TOP_SOURCES = 3


class CitationFormatter:
    """Maps ``[n]`` markers in the answer back to their context sources."""

    def to_sources(self, context: list[RetrievedChunk], answer_text: str) -> list[Source]:
        """Build the source list for an answer.

        Returns the sources the answer actually cites, de-duplicated by URL
        while keeping their original numbers. When the answer cites nothing,
        the top three context sources are returned as a fallback.

        Args:
            context: The retrieved chunks that were shown to the LLM
                (1-indexed by position in the prompt).
            answer_text: The final answer text.

        Returns:
            Ordered, de-duplicated sources.
        """
        cited_numbers = []
        seen_numbers: set[int] = set()
        for match in _CITATION_RE.finditer(answer_text):
            number = int(match.group(1))
            if 1 <= number <= len(context) and number not in seen_numbers:
                seen_numbers.add(number)
                cited_numbers.append(number)
        if cited_numbers:
            return self._build(context, cited_numbers)
        default_numbers = list(range(1, min(_DEFAULT_TOP_SOURCES, len(context)) + 1))
        return self._build(context, default_numbers)

    @staticmethod
    def _build(context: list[RetrievedChunk], numbers: list[int]) -> list[Source]:
        """Construct Source models, de-duplicating by URL."""
        sources: list[Source] = []
        seen_urls: set[str] = set()
        for number in numbers:
            chunk = context[number - 1].chunk
            url = str(chunk.url)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            body = chunk.text.split("\n", 1)[-1]  # skip the header line
            sources.append(
                Source(
                    index=number,
                    title=chunk.title,
                    url=chunk.url,
                    site=chunk.site,
                    snippet=body[:_SNIPPET_CHARS].strip(),
                )
            )
        return sources
