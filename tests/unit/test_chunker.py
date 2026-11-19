"""Chunker tests."""

import pytest

from property_assistant.domain.enums import DocType, SourceSite
from property_assistant.ingestion.chunker import RecursiveTextChunker
from tests.conftest import make_document


class TestRecursiveTextChunker:
    def test_every_chunk_has_the_header_prefix(self) -> None:
        chunker = RecursiveTextChunker(size=200, overlap=40)
        doc = make_document(content="One sentence. " * 60)
        chunks = chunker.split(doc)
        assert chunks
        for chunk in chunks:
            assert chunk.text.startswith("[Dar Global | project | Test Project]")

    def test_header_includes_location_when_known(self) -> None:
        chunker = RecursiveTextChunker(size=500, overlap=50)
        doc = make_document(metadata={"cities": ["Muscat"], "countries": ["Oman"]})
        chunks = chunker.split(doc)
        assert "[Dar Global | project | Test Project, Muscat / Oman]" in chunks[0].text

    def test_chunk_bodies_respect_the_size_bound(self) -> None:
        size = 300
        chunker = RecursiveTextChunker(size=size, overlap=50)
        doc = make_document(content="A sentence here. " * 100)
        for chunk in chunker.split(doc):
            body = chunk.text.split("\n", 1)[1]
            # Overlap stitching can slightly exceed size; never more than size + overlap.
            assert len(body) <= size + 50

    def test_no_tiny_chunks(self) -> None:
        chunker = RecursiveTextChunker(size=1000, overlap=100)
        doc = make_document(content="tiny")
        assert chunker.split(doc) == []

    def test_consecutive_chunks_share_overlap(self) -> None:
        chunker = RecursiveTextChunker(size=150, overlap=60)
        doc = make_document(content="Word salad test. " * 80)
        chunks = chunker.split(doc)
        assert len(chunks) >= 2
        first_body = chunks[0].text.split("\n", 1)[1]
        second_body = chunks[1].text.split("\n", 1)[1]
        # The second chunk starts with trailing content carried from the first.
        overlap_tail = first_body[-60:]
        assert second_body[:20] in overlap_tail or overlap_tail.split()[-1] in second_body[:80]

    def test_ids_and_indexes_are_sequential(self) -> None:
        chunker = RecursiveTextChunker(size=150, overlap=30)
        doc = make_document(content="Many words repeated here. " * 50)
        chunks = chunker.split(doc)
        for position, chunk in enumerate(chunks):
            assert chunk.index == position
            assert chunk.id == f"{doc.id}:{position}"
            assert chunk.doc_id == doc.id

    def test_metadata_propagates(self) -> None:
        chunker = RecursiveTextChunker(size=500, overlap=50)
        doc = make_document(
            site=SourceSite.WASALT,
            doc_type=DocType.LISTING,
            metadata={"city": "Riyadh"},
        )
        chunk = chunker.split(doc)[0]
        assert chunk.site is SourceSite.WASALT
        assert chunk.doc_type is DocType.LISTING
        assert chunk.metadata["city"] == "Riyadh"

    def test_overlap_must_be_smaller_than_size(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            RecursiveTextChunker(size=100, overlap=100)
