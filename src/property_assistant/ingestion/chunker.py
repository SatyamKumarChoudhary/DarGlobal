"""Document chunking strategies."""

import re
from typing import Protocol

from property_assistant.domain.models import Chunk, Document

_MIN_CHUNK_CHARS = 50
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Chunker(Protocol):
    """Strategy interface: split a document into retrieval chunks."""

    def split(self, doc: Document) -> list[Chunk]:
        """Split one document into chunks."""
        ...


class RecursiveTextChunker:
    """Splits text on paragraph → line → sentence → word boundaries.

    Every chunk is prefixed with a context header line, e.g.
    ``[Dar Global | project | AIDA, Oman]``, so both the embedding model and
    the LLM always see which site/page the text belongs to. Chunks under
    50 characters of body text are dropped.
    """

    def __init__(self, size: int, overlap: int) -> None:
        """Args:
        size: Target maximum chunk size in characters (body, excl. header).
        overlap: Characters of trailing context carried into the next chunk.
        """
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self._size = size
        self._overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        """Split a document into header-prefixed chunks.

        Args:
            doc: The cleaned source document.

        Returns:
            Ordered chunks; possibly empty for very short documents.
        """
        header = self._header(doc)
        pieces = self._split_text(doc.content)
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            if len(piece) < _MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc.id}:{index}",
                    doc_id=doc.id,
                    text=f"{header}\n{piece}",
                    index=index,
                    site=doc.site,
                    url=doc.url,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    metadata=doc.metadata,
                )
            )
        return chunks

    @staticmethod
    def _header(doc: Document) -> str:
        """Build the context header line for a document's chunks."""
        place_bits: list[str] = []
        cities = doc.metadata.get("cities")
        if isinstance(cities, list) and cities:
            place_bits.append(str(cities[0]))
        countries = doc.metadata.get("countries")
        if isinstance(countries, list) and countries:
            place_bits.append(str(countries[0]))
        city = doc.metadata.get("city")
        if isinstance(city, str) and city and not place_bits:
            place_bits.append(city)
        place = f", {' / '.join(place_bits)}" if place_bits else ""
        return f"[{doc.site.display_name} | {doc.doc_type.value} | {doc.title}{place}]"

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text, then re-pack into sized windows with overlap."""
        fragments = self._fragment(text)
        pieces: list[str] = []
        current = ""
        for fragment in fragments:
            candidate = f"{current}\n{fragment}".strip() if current else fragment
            if len(candidate) <= self._size:
                current = candidate
                continue
            if current:
                pieces.append(current)
                tail = current[-self._overlap :] if self._overlap else ""
                current = f"{tail}\n{fragment}".strip()
                if len(current) > self._size:
                    pieces.extend(self._hard_split(current))
                    current = ""
            else:
                pieces.extend(self._hard_split(fragment))
        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _fragment(text: str) -> list[str]:
        """Split into natural units: paragraphs, then lines, then sentences."""
        fragments: list[str] = []
        for paragraph in text.split("\n\n"):
            for line in paragraph.splitlines():
                line = line.strip()  # noqa: PLW2901 - normalizing the loop value
                if line:
                    fragments.append(line)
        return fragments

    def _hard_split(self, text: str) -> list[str]:
        """Split an oversized fragment on sentences, then words as last resort."""
        pieces: list[str] = []
        current = ""
        for sentence in _SENTENCE_RE.split(text):
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self._size:
                current = candidate
                continue
            if current:
                pieces.append(current)
            if len(sentence) <= self._size:
                current = sentence
            else:
                words = sentence.split(" ")
                current = ""
                for word in words:
                    candidate = f"{current} {word}".strip() if current else word
                    if len(candidate) <= self._size:
                        current = candidate
                    else:
                        if current:
                            pieces.append(current)
                        current = word
        if current:
            pieces.append(current)
        return pieces
