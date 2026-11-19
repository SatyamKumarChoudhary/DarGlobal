"""Extractor tests against saved fixture pages from both real sites."""

from pathlib import Path

import pytest

from property_assistant.scraping.extractors import (
    CompositeExtractor,
    NextDataExtractor,
    TrafilaturaExtractor,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "html"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestNextDataExtractor:
    def test_darglobal_project_page(self) -> None:
        result = NextDataExtractor().extract(
            "https://darglobal.co.uk/rayana/trump-mansions", fixture("darglobal_project.html")
        )
        assert result is not None
        assert "Trump Mansions" in result.title
        assert "Trump" in result.text

    def test_wasalt_home_page(self) -> None:
        result = NextDataExtractor().extract("https://wasalt.sa/en", fixture("wasalt_home.html"))
        assert result is not None
        assert "Wasalt" in result.title
        assert "Saudi" in result.text

    def test_no_next_data_returns_meta_only_or_none(self) -> None:
        html = "<html><head><title>T</title></head><body><p>text</p></body></html>"
        result = NextDataExtractor().extract("https://example.com", html)
        assert result is None or result.text == ""


class TestTrafilaturaExtractor:
    def test_darglobal_press_body_text(self) -> None:
        result = TrafilaturaExtractor().extract(
            "https://darglobal.co.uk/press/x", fixture("darglobal_press.html")
        )
        assert result is not None
        assert "Jeddah" in result.text

    def test_empty_page_returns_none(self) -> None:
        assert TrafilaturaExtractor().extract("https://example.com", "<html></html>") is None


class TestCompositeExtractor:
    @pytest.fixture
    def extractor(self) -> CompositeExtractor:
        return CompositeExtractor([NextDataExtractor(), TrafilaturaExtractor()])

    def test_merges_structured_facts_and_body_text(self, extractor: CompositeExtractor) -> None:
        result = extractor.extract(
            "https://darglobal.co.uk/press/x", fixture("darglobal_press.html")
        )
        assert result is not None
        assert "Jeddah" in result.text
        assert result.structured  # facts mined from __NEXT_DATA__

    def test_deduplicates_repeated_lines(self, extractor: CompositeExtractor) -> None:
        result = extractor.extract("https://darglobal.co.uk/about", fixture("darglobal_about.html"))
        assert result is not None
        lines = [line.lower() for line in result.text.splitlines()]
        assert len(lines) == len(set(lines))

    def test_short_content_rejected(self, extractor: CompositeExtractor) -> None:
        html = "<html><head><title>T</title></head><body><p>too short</p></body></html>"
        assert extractor.extract("https://example.com", html) is None
