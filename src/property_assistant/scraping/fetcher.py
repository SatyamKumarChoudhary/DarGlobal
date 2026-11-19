"""Polite HTTP fetching: rate limiting, robots.txt compliance and retries."""

import asyncio
import logging
import time
import urllib.robotparser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from property_assistant.core.exceptions import RobotsDisallowedError

_HTML_CONTENT_TYPES = ("text/html", "application/xhtml", "text/xml", "application/xml")


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of a successful page fetch."""

    url: str
    status: int
    html: str
    final_url: str


class RateLimiter:
    """Async token bucket limiting outbound requests per second."""

    def __init__(self, rate_per_sec: float) -> None:
        """Args:
        rate_per_sec: Maximum sustained requests per second.
        """
        self._interval = 1.0 / rate_per_sec
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        """Block until the next request slot is available."""
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


class RobotsPolicy:
    """Caches one robots.txt parser per host and answers allow/deny queries."""

    def __init__(self, user_agent: str) -> None:
        """Args:
        user_agent: The User-Agent string the crawler identifies as.
        """
        self._user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    async def load(self, fetch_text: Callable[[str], Awaitable[str | None]], base_url: str) -> None:
        """Fetch and parse robots.txt for a site.

        Args:
            fetch_text: Coroutine returning the body of a URL, or None.
            base_url: Site root, e.g. ``https://example.com``.
        """
        host = urlsplit(base_url).netloc
        parser = urllib.robotparser.RobotFileParser()
        body = await fetch_text(f"{base_url.rstrip('/')}/robots.txt")
        if body is None:
            # No robots.txt reachable: default to allow-all per the RFC.
            parser.parse([])
        else:
            parser.parse(body.splitlines())
        self._parsers[host] = parser

    def is_allowed(self, url: str) -> bool:
        """Return True when the configured user agent may fetch ``url``."""
        host = urlsplit(url).netloc
        parser = self._parsers.get(host)
        if parser is None:
            return True
        return parser.can_fetch(self._user_agent, url)

    def sitemaps(self, base_url: str) -> list[str]:
        """Return the ``Sitemap:`` URLs declared in a site's robots.txt."""
        host = urlsplit(base_url).netloc
        parser = self._parsers.get(host)
        if parser is None:
            return []
        return list(parser.site_maps() or [])


class Fetcher(Protocol):
    """A polite page fetcher usable by scrapers and the sitemap discoverer."""

    async def fetch_text(self, url: str) -> FetchResult | None:
        """Fetch a text/HTML resource, or return None when unusable."""
        ...


def _is_retryable(exc: BaseException) -> bool:
    """Retry on connection problems, timeouts, 5xx and 429 responses."""
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code >= 500 or exc.response.status_code == 429  # noqa: PLR2004
    )


class HttpFetcher:
    """Robots-aware, rate-limited httpx fetcher with exponential-backoff retries."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        rate_limiter: RateLimiter,
        robots: RobotsPolicy,
        logger: logging.Logger,
    ) -> None:
        """Args:
        client: Shared async HTTP client (owns connection pooling).
        rate_limiter: Token bucket applied before every request.
        robots: Robots.txt policy; disallowed URLs are skipped.
        logger: Structured logger.
        """
        self._client = client
        self._rate_limiter = rate_limiter
        self._robots = robots
        self._logger = logger

    async def fetch_text(self, url: str) -> FetchResult | None:
        """Fetch one page politely.

        Args:
            url: Absolute URL to fetch.

        Returns:
            A :class:`FetchResult`, or ``None`` on 404 / non-HTML / robots denial.
        """
        if not self._robots.is_allowed(url):
            self._logger.info("robots disallowed, skipping", extra={"url": url})
            return None
        await self._rate_limiter.acquire()
        try:
            response = await self._get_with_retry(url)
        except (httpx.HTTPError, RobotsDisallowedError) as exc:
            self._logger.warning("fetch failed", extra={"url": url, "error": str(exc)})
            return None
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in _HTML_CONTENT_TYPES):
            self._logger.debug("non-HTML content, skipping", extra={"url": url})
            return None
        return FetchResult(
            url=url, status=response.status_code, html=response.text, final_url=str(response.url)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def _get_with_retry(self, url: str) -> httpx.Response:
        """GET with retries on transient failures; raises on 5xx for retry."""
        response = await self._client.get(url, follow_redirects=True)
        if response.status_code >= 500 or response.status_code == 429:  # noqa: PLR2004
            response.raise_for_status()
        return response
