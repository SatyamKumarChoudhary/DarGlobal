"""Scraper hook, registry, repository and discovery tests (no network)."""

import logging
import time
from pathlib import Path

import pytest

from property_assistant.core.exceptions import ScrapingError
from property_assistant.domain.enums import DocType, SourceSite
from property_assistant.scraping.base import BaseScraper
from property_assistant.scraping.darglobal import DarGlobalScraper
from property_assistant.scraping.discovery import SitemapDiscoverer
from property_assistant.scraping.extractors import ExtractedContent
from property_assistant.scraping.fetcher import FetchResult, RateLimiter
from property_assistant.scraping.registry import ScraperRegistry
from property_assistant.scraping.repository import DocumentRepository
from property_assistant.scraping.wasalt import WasaltScraper
from tests.conftest import make_document

logger = logging.getLogger("test")


class StubFetcher:
    """Returns canned HTML for every URL."""

    def __init__(self, html: str = "x" * 200) -> None:
        self.html = html
        self.fetched: list[str] = []

    async def fetch_text(self, url: str) -> FetchResult | None:
        self.fetched.append(url)
        return FetchResult(url=url, status=200, html=self.html, final_url=url)


class StubExtractor:
    """Returns fixed extracted content."""

    def extract(self, url: str, html: str) -> ExtractedContent | None:
        return ExtractedContent(title="Title", text="Body text " * 20, structured={})


def make_darglobal(urls: list[str], max_pages: int = 10) -> DarGlobalScraper:
    async def discover(_base: str) -> list[str]:
        return urls

    return DarGlobalScraper(
        fetcher=StubFetcher(),
        discover=discover,
        extractor=StubExtractor(),
        max_pages=max_pages,
        logger=logger,
    )


class TestDarGlobalScraper:
    def test_classification_by_url_family(self) -> None:
        scraper = make_darglobal([])
        content = ExtractedContent(title="t", text="x")
        cases = {
            "https://darglobal.co.uk/rayana/trump-mansions": DocType.PROJECT,
            "https://darglobal.co.uk/landing-page/aida": DocType.PROJECT,
            "https://darglobal.co.uk/projects/properties-in-oman": DocType.PROJECT,
            "https://darglobal.co.uk/press/some-announcement": DocType.PRESS,
            "https://darglobal.co.uk/partners/missoni": DocType.PARTNER,
            "https://darglobal.co.uk/about": DocType.ABOUT,
            "https://darglobal.co.uk/blog/post": DocType.ARTICLE,
            "https://darglobal.co.uk/faq": DocType.PAGE,
        }
        for url, expected in cases.items():
            assert scraper.classify(url, content) is expected, url

    def test_include_url_filters_noise(self) -> None:
        scraper = make_darglobal([])
        assert scraper.include_url("https://darglobal.co.uk/about")
        assert not scraper.include_url("https://darglobal.co.uk/ar/about")
        assert not scraper.include_url("https://darglobal.co.uk/thank-you")
        assert not scraper.include_url("https://other.com/about")

    def test_prioritise_puts_projects_first(self) -> None:
        scraper = make_darglobal([])
        ordered = scraper.prioritise(
            [
                "https://darglobal.co.uk/press/news",
                "https://darglobal.co.uk/about",
                "https://darglobal.co.uk/rayana/trump-mansions",
            ]
        )
        assert ordered[0].endswith("trump-mansions")
        assert ordered[1].endswith("/about")

    def test_metadata_extracts_locations(self) -> None:
        scraper = make_darglobal([])
        content = ExtractedContent(
            title="Villas in Marbella", text="A project on the Costa del Sol in Spain."
        )
        metadata = scraper.build_metadata("https://darglobal.co.uk/x", content)
        assert "Spain" in metadata["countries"]
        assert "Marbella" in metadata["cities"]

    async def test_run_caps_at_max_pages_and_counts(self) -> None:
        urls = [f"https://darglobal.co.uk/press/item-{i}" for i in range(9)]
        scraper = make_darglobal(urls, max_pages=4)
        docs = [doc async for doc in scraper.run()]
        assert len(docs) == 4
        assert scraper.stats.scraped == 4
        assert scraper.stats.by_doc_type == {"press": 4}

    async def test_run_survives_extractor_returning_none(self) -> None:
        class NoneExtractor:
            def extract(self, url: str, html: str) -> ExtractedContent | None:
                return None

        async def discover(_base: str) -> list[str]:
            return ["https://darglobal.co.uk/about"]

        scraper = DarGlobalScraper(
            fetcher=StubFetcher(),
            discover=discover,
            extractor=NoneExtractor(),
            max_pages=5,
            logger=logger,
        )
        docs = [doc async for doc in scraper.run()]
        assert docs == []
        assert scraper.stats.skipped == 1


class TestWasaltScraper:
    def _scraper(self) -> WasaltScraper:
        async def discover(_base: str) -> list[str]:
            return []

        return WasaltScraper(
            fetcher=StubFetcher(),
            discover=discover,
            extractor=StubExtractor(),
            max_pages=10,
            logger=logger,
        )

    def test_only_english_pages_included(self) -> None:
        scraper = self._scraper()
        assert scraper.include_url("https://wasalt.sa/en/about")
        assert not scraper.include_url("https://wasalt.sa/ar/about")
        assert not scraper.include_url("https://wasalt.com/en/about")

    def test_listing_classification(self) -> None:
        scraper = self._scraper()
        content = ExtractedContent(title="t", text="x")
        assert (
            scraper.classify("https://wasalt.sa/en/property/sale/villa-123", content)
            is DocType.LISTING
        )
        assert (
            scraper.classify("https://wasalt.sa/en/dailyrental/riyadh/apt-1", content)
            is DocType.LISTING
        )
        assert scraper.classify("https://wasalt.sa/en/about", content) is DocType.ABOUT

    def test_listing_metadata_from_structured_data(self) -> None:
        scraper = self._scraper()
        content = ExtractedContent(
            title="Villa",
            text="x",
            structured={"city": "Riyadh", "price": 2500000, "bedrooms": 4},
        )
        metadata = scraper.build_metadata("https://wasalt.sa/en/property/sale/villa-123", content)
        assert metadata["city"] == "Riyadh"
        assert metadata["price"] == 2500000
        assert metadata["purpose"] == "sale"

    def test_prioritise_bounds_listings(self) -> None:
        scraper = self._scraper()
        listings = [f"https://wasalt.sa/en/property/sale/x-{i}" for i in range(500)]
        ordered = scraper.prioritise(["https://wasalt.sa/en", *listings])
        assert ordered[0] == "https://wasalt.sa/en"
        assert len(ordered) < 500


class TestScraperRegistry:
    def test_create_registered_scraper(self) -> None:
        registry = ScraperRegistry()
        scraper = make_darglobal([])
        registry.register(SourceSite.DARGLOBAL, lambda: scraper)
        assert registry.create(SourceSite.DARGLOBAL) is scraper
        assert registry.available() == [SourceSite.DARGLOBAL]

    def test_unknown_site_raises(self) -> None:
        registry = ScraperRegistry()
        with pytest.raises(ScrapingError):
            registry.create(SourceSite.WASALT)


class TestDocumentRepository:
    def test_save_load_round_trip_and_dedupe(self, tmp_path: Path) -> None:
        repo = DocumentRepository(tmp_path, logger)
        docs = [make_document("a"), make_document("b"), make_document("a")]
        written = repo.save_all(SourceSite.DARGLOBAL, docs)
        assert written == 2
        loaded = repo.load_all()
        assert {doc.id for doc in loaded} == {"a", "b"}

    def test_count_by_site(self, tmp_path: Path) -> None:
        repo = DocumentRepository(tmp_path, logger)
        repo.save_all(SourceSite.DARGLOBAL, [make_document("a")])
        repo.save_all(SourceSite.WASALT, [make_document("w", site=SourceSite.WASALT)])
        assert repo.count_by_site() == {"darglobal": 1, "wasalt": 1}

    def test_load_with_no_files(self, tmp_path: Path) -> None:
        repo = DocumentRepository(tmp_path, logger)
        assert repo.load_all() == []


class TestSitemapDiscoverer:
    async def test_simple_sitemap(self) -> None:
        async def fetch(url: str) -> str | None:
            return (
                "<urlset><url><loc>https://x.com/a</loc></url>"
                "<url><loc>https://x.com/b</loc></url></urlset>"
            )

        discoverer = SitemapDiscoverer(fetch, logger)
        urls = await discoverer.discover("https://x.com")
        assert urls == ["https://x.com/a", "https://x.com/b"]

    async def test_sitemap_index_recursion(self) -> None:
        async def fetch(url: str) -> str | None:
            if url.endswith("/sitemap.xml"):
                return (
                    "<sitemapindex><sitemap><loc>https://x.com/child.xml</loc>"
                    "</sitemap></sitemapindex>"
                )
            return "<urlset><url><loc>https://x.com/page</loc></url></urlset>"

        discoverer = SitemapDiscoverer(fetch, logger)
        assert await discoverer.discover("https://x.com") == ["https://x.com/page"]

    async def test_unreachable_sitemap(self) -> None:
        async def fetch(url: str) -> str | None:
            return None

        discoverer = SitemapDiscoverer(fetch, logger)
        assert await discoverer.discover("https://x.com") == []


class TestRateLimiter:
    async def test_acquire_spaces_requests(self) -> None:
        limiter = RateLimiter(rate_per_sec=50)
        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.03  # at least 2 intervals of 20ms


class TestTemplateMethodIsFinal:
    def test_subclasses_do_not_override_run(self) -> None:
        for subclass in (DarGlobalScraper, WasaltScraper):
            assert subclass.run is BaseScraper.run
