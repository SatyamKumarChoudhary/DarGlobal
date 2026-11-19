"""URL discovery: sitemap parsing with a same-domain crawler as fallback."""

import gzip
import logging
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin, urlsplit

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from property_assistant.scraping.fetcher import Fetcher

_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.IGNORECASE | re.DOTALL)
_SITEMAP_TAG_RE = re.compile(r"<sitemap>", re.IGNORECASE)

type ResourceFetcher = Callable[[str], Awaitable[str | None]]


class SitemapDiscoverer:
    """Discovers page URLs from robots.txt sitemap entries and /sitemap.xml."""

    def __init__(
        self,
        fetch_resource: ResourceFetcher,
        logger: logging.Logger,
        extra_sitemaps: list[str] | None = None,
    ) -> None:
        """Args:
        fetch_resource: Coroutine returning a URL's text body or None.
        logger: Structured logger.
        extra_sitemaps: Known sitemap URLs to read in addition to defaults.
        """
        self._fetch_resource = fetch_resource
        self._logger = logger
        self._extra_sitemaps = extra_sitemaps or []

    async def discover(self, base_url: str) -> list[str]:
        """Collect de-duplicated page URLs for a site.

        Reads ``{base}/sitemap.xml`` plus any explicitly supplied sitemap
        URLs, recursing into sitemap indexes.

        Args:
            base_url: Site root, e.g. ``https://example.com``.

        Returns:
            Unique page URLs in first-seen order.
        """
        candidates = [f"{base_url.rstrip('/')}/sitemap.xml", *self._extra_sitemaps]
        seen: dict[str, None] = {}
        for sitemap_url in candidates:
            for url in await self._read_sitemap(sitemap_url, depth=0):
                seen.setdefault(url, None)
        self._logger.info(
            "sitemap discovery complete", extra={"base_url": base_url, "n_urls": len(seen)}
        )
        return list(seen)

    async def _read_sitemap(self, url: str, depth: int) -> list[str]:
        """Read one sitemap (or sitemap index) URL, recursing one level deep."""
        max_depth = 2
        if depth > max_depth:
            return []
        body = await self._fetch_resource(url)
        if body is None:
            return []
        if _SITEMAP_TAG_RE.search(body):
            nested: list[str] = []
            for child in _LOC_RE.findall(body):
                nested.extend(await self._read_sitemap(child, depth + 1))
            return nested
        return list(_LOC_RE.findall(body))


def decompress_if_gzip(data: bytes) -> str:
    """Return text content from possibly gzip-compressed sitemap bytes."""
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data).decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")


class LinkCrawler:
    """Breadth-first same-domain crawler used when a sitemap is unavailable."""

    def __init__(
        self,
        fetcher: Fetcher,
        logger: logging.Logger,
        allow_patterns: list[str] | None = None,
        deny_patterns: list[str] | None = None,
    ) -> None:
        """Args:
        fetcher: Polite page fetcher.
        logger: Structured logger.
        allow_patterns: Regexes a URL must match (any) to be followed.
        deny_patterns: Regexes that exclude a URL when matched.
        """
        self._fetcher = fetcher
        self._logger = logger
        self._allow = [re.compile(p) for p in (allow_patterns or [])]
        self._deny = [re.compile(p) for p in (deny_patterns or [])]

    async def crawl(self, seeds: list[str], max_pages: int) -> list[str]:
        """BFS over same-domain links starting from the seed URLs.

        Args:
            seeds: Starting URLs; their host defines the crawl domain.
            max_pages: Hard cap on pages fetched.

        Returns:
            The URLs visited, in BFS order.
        """
        if not seeds:
            return []
        domain = urlsplit(seeds[0]).netloc
        queue = list(seeds)
        visited: dict[str, None] = {}
        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited or not self._accepts(url):
                continue
            visited[url] = None
            result = await self._fetcher.fetch_text(url)
            if result is None:
                continue
            for link in self._extract_links(result.final_url, result.html):
                if urlsplit(link).netloc == domain and link not in visited:
                    queue.append(link)
        return list(visited)

    def _accepts(self, url: str) -> bool:
        """Apply the allow/deny regex lists to a URL."""
        if any(p.search(url) for p in self._deny):
            return False
        return not self._allow or any(p.search(url) for p in self._allow)

    @staticmethod
    def _extract_links(page_url: str, html: str) -> list[str]:
        """Pull absolute, fragment-free hrefs out of a page."""
        tree = HTMLParser(html)
        links: list[str] = []
        for node in tree.css("a[href]"):
            href = node.attributes.get("href")
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            links.append(urljoin(page_url, href).split("#")[0])
        return links
