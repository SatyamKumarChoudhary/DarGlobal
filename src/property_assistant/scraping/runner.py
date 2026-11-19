"""Composition and execution of scrape runs (CLI-side wiring).

The runner builds a browser-backed fetcher (both target sites sit behind bot
challenges), loads robots.txt, wires the per-site scrapers through the
registry, and persists the results through the repository.
"""

import logging

from property_assistant.core.config import Settings
from property_assistant.domain.enums import SourceSite
from property_assistant.domain.models import Document
from property_assistant.scraping.base import ScrapeStats
from property_assistant.scraping.browser import BrowserFetcher
from property_assistant.scraping.darglobal import DarGlobalScraper
from property_assistant.scraping.discovery import SitemapDiscoverer
from property_assistant.scraping.extractors import (
    CompositeExtractor,
    ContentExtractor,
    NextDataExtractor,
    TrafilaturaExtractor,
)
from property_assistant.scraping.fetcher import RateLimiter, RobotsPolicy
from property_assistant.scraping.repository import DocumentRepository
from property_assistant.scraping.wasalt import WASALT_SITEMAPS, WasaltScraper

# Browsers must present a browser UA or the anti-bot layer blocks them outright;
# the crawl still identifies itself to robots.txt via the configured bot UA.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


async def run_scrape(
    settings: Settings, sites: list[SourceSite], logger: logging.Logger
) -> dict[SourceSite, ScrapeStats]:
    """Scrape the requested sites and persist JSONL snapshots.

    Args:
        settings: Application settings (rate limits, page caps, paths).
        sites: Which sites to scrape.
        logger: Structured logger.

    Returns:
        Per-site scrape statistics.
    """
    repository = DocumentRepository(settings.raw_data_dir, logger)
    rate_limiter = RateLimiter(settings.scraper_rate_per_sec)
    robots = RobotsPolicy(settings.scraper_user_agent)
    stats: dict[SourceSite, ScrapeStats] = {}

    async with BrowserFetcher(rate_limiter, robots, _BROWSER_UA, logger) as fetcher:
        for site in sites:
            base_url = {
                SourceSite.DARGLOBAL: DarGlobalScraper.base_url,
                SourceSite.WASALT: WasaltScraper.base_url,
            }[site]
            await robots.load(fetcher.fetch_resource, base_url)
            extra_sitemaps = WASALT_SITEMAPS if site is SourceSite.WASALT else []
            discoverer = SitemapDiscoverer(fetcher.fetch_resource, logger, extra_sitemaps)
            scraper = _build_scraper(site, fetcher, discoverer, settings, logger)
            documents: list[Document] = [doc async for doc in scraper.run()]
            repository.save_all(site, documents)
            stats[site] = scraper.stats
    return stats


def _build_scraper(
    site: SourceSite,
    fetcher: BrowserFetcher,
    discoverer: SitemapDiscoverer,
    settings: Settings,
    logger: logging.Logger,
) -> DarGlobalScraper | WasaltScraper:
    """Instantiate the right scraper with its page budget."""
    if site is SourceSite.DARGLOBAL:
        return DarGlobalScraper(
            fetcher=fetcher,
            discover=discoverer.discover,
            extractor=_default_extractor(),
            max_pages=settings.scraper_max_pages_darglobal,
            logger=logger,
        )
    return WasaltScraper(
        fetcher=fetcher,
        discover=discoverer.discover,
        extractor=_default_extractor(),
        max_pages=settings.scraper_max_pages_wasalt,
        logger=logger,
    )


def _default_extractor() -> ContentExtractor:
    """Composite NextData + Trafilatura extractor."""
    return CompositeExtractor([NextDataExtractor(), TrafilaturaExtractor()])
