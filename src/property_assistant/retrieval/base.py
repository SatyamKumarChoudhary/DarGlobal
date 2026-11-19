"""Retriever strategy interface."""

from typing import Protocol

from property_assistant.domain.models import RetrievedChunk


class Retriever(Protocol):
    """Strategy interface: find relevant chunks for a query."""

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, best first."""
        ...
