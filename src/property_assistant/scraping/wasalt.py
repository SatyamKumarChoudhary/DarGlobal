"""Scraper for wasalt.sa (English pages only, bounded listing sample)."""

import re
from typing import Any, ClassVar
from urllib.parse import urlsplit

from property_assistant.domain.enums import DocType, SourceSite
from property_assistant.scraping.base import BaseScraper
from property_assistant.scraping.extractors import ExtractedContent

# The CDN hosts the sitemaps; HTML lives on wasalt.sa. The 8.6k-page category
# sitemap is intentionally excluded: those are thin search-result pages.
WASALT_SITEMAPS = [
    "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_en_sa.xml.gz",
]
# Hand-picked overview and major-city category pages.
_SEED_URLS = [
    "https://wasalt.sa/en",
    "https://wasalt.sa/en/about",
    "https://wasalt.sa/en/s/wasalt-business",
    "https://wasalt.sa/en/s/masterplan",
    "https://wasalt.sa/en/properties-for-sale-in-riyadh",
    "https://wasalt.sa/en/properties-for-sale-in-jeddah",
    "https://wasalt.sa/en/properties-for-sale-in-dammam",
    "https://wasalt.sa/en/properties-for-rent-in-riyadh",
    "https://wasalt.sa/en/properties-for-rent-in-jeddah",
]
_MAX_LISTINGS = 120
_PRIORITY_CITIES = ("riyadh", "jeddah", "dammam", "khobar", "makkah", "madinah")
_SKIP_PATTERNS = re.compile(r"/(privacy-policy|terms-|cdn-cgi)/?", re.IGNORECASE)
_NUMERIC_FACTS = ("price", "bedrooms", "bathrooms", "area", "builtuparea")


class WasaltScraper(BaseScraper):
    """Scrapes platform pages, service pages and a bounded listing sample."""

    site: ClassVar[SourceSite] = SourceSite.WASALT
    base_url: ClassVar[str] = "https://wasalt.sa"

    async def discover_urls(self) -> list[str]:
        """Discover URLs from CDN sitemaps plus the seed pages."""
        discovered = await self._discover(self.base_url)
        return [*_SEED_URLS, *discovered]

    def include_url(self, url: str) -> bool:
        """Keep English wasalt.sa pages only."""
        parts = urlsplit(url)
        if parts.netloc != "wasalt.sa":
            return False
        if not parts.path.startswith("/en"):
            return False
        return not _SKIP_PATTERNS.search(parts.path)

    def classify(self, url: str, content: ExtractedContent) -> DocType:
        """Classify by URL family."""
        path = urlsplit(url).path
        if "/property/" in path or "/dailyrental" in path:
            return DocType.LISTING
        if "/project" in path:
            return DocType.PROJECT
        if "/blog" in path or "/post" in path:
            return DocType.ARTICLE
        if "/about" in path or "/corporate" in path or "/s/wasalt-business" in path:
            return DocType.ABOUT
        return DocType.PAGE

    def prioritise(self, urls: list[str]) -> list[str]:
        """Platform pages first, then a balanced, city-diverse listing sample.

        Sale listings (``/property/``) and daily rentals are sampled
        separately so one sitemap cannot crowd out the other; both favour
        major cities.
        """
        overview: list[str] = []
        sale_listings: list[str] = []
        rental_listings: list[str] = []
        other: list[str] = []
        for url in urls:
            path = urlsplit(url).path
            if "/property/" in path:
                sale_listings.append(url)
            elif "/dailyrental" in path:
                rental_listings.append(url)
            elif path.rstrip("/") in ("/en", "/en/about") or "/s/" in path:
                overview.append(url)
            else:
                other.append(url)

        def by_city(url: str) -> int:
            return 0 if any(city in url.lower() for city in _PRIORITY_CITIES) else 1

        sampled_sales = sorted(sale_listings, key=by_city)[:_MAX_LISTINGS]
        sampled_rentals = sorted(rental_listings, key=by_city)[: _MAX_LISTINGS // 2]
        return [*overview, *other, *sampled_sales, *sampled_rentals]

    def build_metadata(self, url: str, content: ExtractedContent) -> dict[str, Any]:
        """Lift listing facts (city, price, rooms, ...) from the structured data."""
        metadata: dict[str, Any] = {}
        structured = content.structured
        for key in ("city", "district", "propertytype", "purpose", "currency"):
            value = structured.get(key)
            if isinstance(value, str) and value:
                metadata[key] = value
        for key in _NUMERIC_FACTS:
            value = structured.get(key)
            if isinstance(value, int | float | str) and value != "":
                metadata[key] = value
        path = urlsplit(url).path
        if "/sale/" in path:
            metadata.setdefault("purpose", "sale")
        elif "/rent" in path:
            metadata.setdefault("purpose", "rent")
        rega = structured.get("rega") or structured.get("regaverified")
        if rega is not None:
            metadata["rega_verified"] = bool(rega)
        return metadata
