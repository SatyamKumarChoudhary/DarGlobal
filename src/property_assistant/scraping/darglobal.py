"""Scraper for darglobal.co.uk (English pages only)."""

import re
from typing import Any, ClassVar
from urllib.parse import urlsplit

from property_assistant.domain.enums import DocType, SourceSite
from property_assistant.scraping.base import BaseScraper
from property_assistant.scraping.extractors import ExtractedContent

_PROJECT_PREFIXES = ("/landing-page/", "/lp/", "/rayana/", "/amaya/", "/one-of-one/")
_ABOUT_PATHS = ("/about", "/investor", "/why-invest")
_SKIP_PATTERNS = re.compile(
    r"/(ar|thank-you|success|pay-online|bookyourunit|internship-application|"
    r"become-a-broker|become-an-agent|get-in-touch|campaigns|privacy-policy|"
    r"terms-conditions|terms-of-uses|sitemap)",
    re.IGNORECASE,
)
# Known locations mentioned in DarGlobal titles/bodies, mapped to a country.
_LOCATIONS: dict[str, str] = {
    "dubai": "UAE",
    "abu dhabi": "UAE",
    "uae": "UAE",
    "riyadh": "Saudi Arabia",
    "jeddah": "Saudi Arabia",
    "saudi arabia": "Saudi Arabia",
    "doha": "Qatar",
    "qatar": "Qatar",
    "muscat": "Oman",
    "oman": "Oman",
    "marbella": "Spain",
    "costa del sol": "Spain",
    "spain": "Spain",
    "london": "UK",
    "uk": "UK",
    "united kingdom": "UK",
    "athens": "Greece",
    "greece": "Greece",
    "maldives": "Maldives",
    "bodrum": "Turkey",
    "turkey": "Turkey",
}
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})")


class DarGlobalScraper(BaseScraper):
    """Scrapes projects, press, partners and corporate pages from DarGlobal."""

    site: ClassVar[SourceSite] = SourceSite.DARGLOBAL
    base_url: ClassVar[str] = "https://darglobal.co.uk"

    async def discover_urls(self) -> list[str]:
        """Discover URLs from the sitemap."""
        return await self._discover(self.base_url)

    def include_url(self, url: str) -> bool:
        """Keep same-host English content pages; drop utility/Arabic pages."""
        parts = urlsplit(url)
        if parts.netloc != "darglobal.co.uk":
            return False
        path = parts.path.rstrip("/") or "/"
        return not _SKIP_PATTERNS.search(path)

    def classify(self, url: str, content: ExtractedContent) -> DocType:
        """Classify by URL family."""
        path = urlsplit(url).path
        if any(path.startswith(p) for p in _PROJECT_PREFIXES) or path.startswith("/projects"):
            return DocType.PROJECT
        if path.startswith("/press"):
            return DocType.PRESS
        if path.startswith("/partners"):
            return DocType.PARTNER
        if any(path.startswith(p) for p in _ABOUT_PATHS):
            return DocType.ABOUT
        if path.startswith(("/blog", "/insights")):
            return DocType.ARTICLE
        return DocType.PAGE

    def prioritise(self, urls: list[str]) -> list[str]:
        """Projects first, then corporate pages, partners, press, then the rest."""

        def rank(url: str) -> int:
            path = urlsplit(url).path
            if any(path.startswith(p) for p in _PROJECT_PREFIXES) or path.startswith("/projects"):
                return 0
            if any(path.startswith(p) for p in _ABOUT_PATHS):
                return 1
            if path.startswith("/partners"):
                return 2
            if path.startswith("/press"):
                return 3
            if path.startswith(("/blog", "/insights")):
                return 5
            return 4

        return sorted(urls, key=rank)

    def build_metadata(self, url: str, content: ExtractedContent) -> dict[str, Any]:
        """Infer city/country from the title and body, plus a press date."""
        metadata: dict[str, Any] = {}
        haystack = f"{content.title}\n{content.text[:2000]}".lower()
        cities: list[str] = []
        countries: list[str] = []
        for place, country in _LOCATIONS.items():
            if re.search(rf"\b{re.escape(place)}\b", haystack):
                if place != country.lower():
                    cities.append(place.title())
                if country not in countries:
                    countries.append(country)
        if cities:
            metadata["cities"] = cities
        if countries:
            metadata["countries"] = countries
        published = content.structured.get("publishedat") or content.structured.get("date")
        if isinstance(published, str):
            match = _DATE_RE.search(published)
            if match:
                metadata["date"] = match.group(1)
        return metadata
