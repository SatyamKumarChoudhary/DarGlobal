"""Factory registry mapping a site to its scraper constructor."""

from collections.abc import Callable

from property_assistant.core.exceptions import ScrapingError
from property_assistant.domain.enums import SourceSite
from property_assistant.scraping.base import BaseScraper

type ScraperFactory = Callable[[], BaseScraper]


class ScraperRegistry:
    """Open/closed extension point: register a factory per source site."""

    def __init__(self) -> None:
        self._factories: dict[SourceSite, ScraperFactory] = {}

    def register(self, site: SourceSite, factory: ScraperFactory) -> None:
        """Register the factory that builds the scraper for ``site``."""
        self._factories[site] = factory

    def create(self, site: SourceSite) -> BaseScraper:
        """Instantiate the scraper for a site.

        Raises:
            ScrapingError: When no factory is registered for the site.
        """
        factory = self._factories.get(site)
        if factory is None:
            raise ScrapingError(f"No scraper registered for site {site.value!r}")
        return factory()

    def available(self) -> list[SourceSite]:
        """Return the sites that have a registered scraper."""
        return list(self._factories)
