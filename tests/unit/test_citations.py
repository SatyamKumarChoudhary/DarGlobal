"""Citation formatter tests."""

from property_assistant.chat.citations import CitationFormatter
from tests.conftest import make_retrieved


class TestCitationFormatter:
    def setup_method(self) -> None:
        self.formatter = CitationFormatter()
        self.context = [
            make_retrieved("a:0", url="https://example.com/a", title="Page A"),
            make_retrieved("b:0", url="https://example.com/b", title="Page B"),
            make_retrieved("c:0", url="https://example.com/c", title="Page C"),
            make_retrieved("d:0", url="https://example.com/d", title="Page D"),
        ]

    def test_only_cited_sources_returned(self) -> None:
        sources = self.formatter.to_sources(self.context, "Fact one [1]. Fact three [3].")
        assert [s.index for s in sources] == [1, 3]
        assert [s.title for s in sources] == ["Page A", "Page C"]

    def test_citation_order_follows_the_answer(self) -> None:
        sources = self.formatter.to_sources(self.context, "See [2] and then [1].")
        assert [s.index for s in sources] == [2, 1]

    def test_duplicate_citations_deduplicated(self) -> None:
        sources = self.formatter.to_sources(self.context, "[1] again [1] and [1].")
        assert [s.index for s in sources] == [1]

    def test_same_url_cited_twice_deduplicated(self) -> None:
        context = [
            make_retrieved("a:0", url="https://example.com/same"),
            make_retrieved("a:1", url="https://example.com/same"),
        ]
        sources = self.formatter.to_sources(context, "[1] and [2].")
        assert len(sources) == 1

    def test_out_of_range_citations_ignored(self) -> None:
        sources = self.formatter.to_sources(self.context, "Fake [9] but real [2].")
        assert [s.index for s in sources] == [2]

    def test_no_citations_returns_top_three(self) -> None:
        sources = self.formatter.to_sources(self.context, "No markers here.")
        assert [s.index for s in sources] == [1, 2, 3]

    def test_empty_context(self) -> None:
        assert self.formatter.to_sources([], "anything [1]") == []

    def test_snippet_skips_header_line(self) -> None:
        sources = self.formatter.to_sources(self.context, "[1]")
        assert not sources[0].snippet.startswith("[Dar Global")
