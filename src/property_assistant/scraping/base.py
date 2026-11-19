"""Template-method base scraper: discover → fetch → extract → build Document."""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import HttpUrl

from property_assistant.domain.enums import DocType, SourceSite
from property_assistant.domain.models import Document
from property_assistant.scraping.extractors import ContentExtractor, ExtractedContent
from property_assistant.scraping.fetcher import Fetcher

type UrlDiscoverer = Callable[[str], Awaitable[list[str]]]


@dataclass(slots=True)
class ScrapeStats:
    """Counters reported after a scrape run."""

    discovered: int = 0
    scraped: int = 0
    skipped: int = 0
    failed: int = 0
    by_doc_type: dict[str, int] = field(default_factory=dict)


class BaseScraper(ABC):
    """Fixes the scraping algorithm; subclasses fill in the site-specific hooks.

    The :meth:`run` template method is final: discover URLs, filter and
    prioritise them, cap at ``max_pages``, then fetch → extract → classify →
    build a :class:`~property_assistant.domain.models.Document` for each. One
    bad page never aborts the run.
    """

    site: ClassVar[SourceSite]
    base_url: ClassVar[str]

    def __init__(
        self,
        fetcher: Fetcher,
        discover: UrlDiscoverer,
        extractor: ContentExtractor,
        max_pages: int,
        logger: logging.Logger,
    ) -> None:
        """Args:
        fetcher: Polite page fetcher (HTTP or browser-backed).
        discover: Coroutine producing candidate URLs for a base URL.
        extractor: Content extraction strategy.
        max_pages: Hard cap on pages fetched in one run.
        logger: Structured logger.
        """
        self._fetcher = fetcher
        self._discover = discover
        self._extractor = extractor
        self._max_pages = max_pages
        self._logger = logger
        self.stats = ScrapeStats()

    async def run(self) -> AsyncIterator[Document]:
        """Execute the full scrape and yield documents as they are built."""
        urls = await self.discover_urls()
        urls = [u for u in dict.fromkeys(urls) if self.include_url(u)]
        urls = self.prioritise(urls)[: self._max_pages]
        self.stats.discovered = len(urls)
        self._logger.info("scrape starting", extra={"site": self.site.value, "n_urls": len(urls)})
        for position, url in enumerate(urls, start=1):
            document = await self._scrape_one(url)
            if document is not None:
                self.stats.scraped += 1
                self.stats.by_doc_type[document.doc_type.value] = (
                    self.stats.by_doc_type.get(document.doc_type.value, 0) + 1
                )
                yield document
            if position % 25 == 0:
                self._logger.info(
                    "scrape progress",
                    extra={
                        "site": self.site.value,
                        "done": position,
                        "total": len(urls),
                        "scraped": self.stats.scraped,
                    },
                )
        self._logger.info(
            "scrape finished",
            extra={
                "site": self.site.value,
                "scraped": self.stats.scraped,
                "skipped": self.stats.skipped,
                "failed": self.stats.failed,
            },
        )

    async def _scrape_one(self, url: str) -> Document | None:
        """Fetch and convert one URL; log-and-continue on any failure."""
        try:
            result = await self._fetcher.fetch_text(url)
        except Exception:
            self._logger.exception("unexpected fetch error", extra={"url": url})
            self.stats.failed += 1
            return None
        if result is None:
            self.stats.skipped += 1
            return None
        content = self._extractor.extract(result.final_url, result.html)
        if content is None:
            self.stats.skipped += 1
            self._logger.debug("no content extracted", extra={"url": url})
            return None
        doc_type = self.classify(url, content)
        metadata = self.build_metadata(url, content)
        return Document(
            id=hashlib.sha256(url.encode()).hexdigest()[:16],
            site=self.site,
            url=HttpUrl(url),
            title=content.title or url,
            content=content.text,
            doc_type=doc_type,
            metadata=metadata,
            scraped_at=datetime.now(UTC),
        )

    @abstractmethod
    async def discover_urls(self) -> list[str]:
        """Produce the candidate URL list for this site."""

    @abstractmethod
    def include_url(self, url: str) -> bool:
        """Decide whether a discovered URL is worth scraping."""

    @abstractmethod
    def classify(self, url: str, content: ExtractedContent) -> DocType:
        """Assign a document type to a scraped page."""

    def prioritise(self, urls: list[str]) -> list[str]:
        """Order URLs so the most valuable pages fit under ``max_pages``.

        Default: keep discovery order. Subclasses may override.
        """
        return urls

    def build_metadata(self, url: str, content: ExtractedContent) -> dict[str, Any]:
        """Build per-document metadata. Default: empty. Subclasses may override."""
        return {}
