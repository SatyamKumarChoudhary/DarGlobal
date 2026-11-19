"""Playwright-backed fetcher for bot-protected sites (scrape-time only).

Both target sites sit behind JavaScript challenges (Incapsula on
darglobal.co.uk, Cloudflare on wasalt.sa), so plain HTTP requests receive a
challenge page instead of content. This fetcher drives a real headless
browser, which solves the challenges transparently. It implements the same
``Fetcher`` protocol as :class:`~property_assistant.scraping.fetcher.HttpFetcher`,
so the rest of the scraping pipeline is unaware of the difference.

Playwright is an optional dependency group (``uv sync --group scrape``) and is
never part of the runtime Docker image.
"""

import logging
from types import TracebackType
from typing import TYPE_CHECKING, Self

from property_assistant.scraping.fetcher import FetchResult, RateLimiter, RobotsPolicy

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

_CHALLENGE_MARKERS = ("_Incapsula_Resource", "cf-chl", "Just a moment...", "challenge-platform")
_MAX_CHALLENGE_RETRIES = 3
_SETTLE_MS = 2500


class BrowserFetcher:
    """Fetches pages with headless Chromium, retrying through bot challenges.

    Use as an async context manager so the browser is always closed::

        async with BrowserFetcher(rate_limiter, robots, ua, logger) as fetcher:
            result = await fetcher.fetch_text(url)
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        robots: RobotsPolicy,
        user_agent: str,
        logger: logging.Logger,
    ) -> None:
        """Args:
        rate_limiter: Token bucket applied before every navigation.
        robots: Robots.txt policy; disallowed URLs are skipped.
        user_agent: Browser user agent to present.
        logger: Structured logger.
        """
        self._rate_limiter = rate_limiter
        self._robots = robots
        self._user_agent = user_agent
        self._logger = logger
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __aenter__(self) -> Self:
        """Start Playwright and open one shared browser context."""
        from playwright.async_api import async_playwright  # noqa: PLC0415 - optional dep

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        await self._new_context()
        return self

    async def _new_context(self) -> None:
        """(Re)create the browser context; a fresh context clears a stuck session."""
        if self._browser is None:
            raise RuntimeError("BrowserFetcher must be used as an async context manager")
        if self._context is not None:
            await self._context.close()
        self._context = await self._browser.new_context(user_agent=self._user_agent, locale="en-US")
        self._page = await self._context.new_page()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the browser and stop Playwright."""
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def fetch_text(self, url: str) -> FetchResult | None:
        """Navigate to a page and return its rendered HTML.

        Retries a few times when a bot-challenge page is returned, because the
        challenge cookie occasionally needs a fresh warm-up navigation.

        Args:
            url: Absolute URL to fetch.

        Returns:
            A :class:`FetchResult` with fully rendered HTML, or ``None`` when
            the page stayed blocked, errored or returned a non-success status.
        """
        if self._page is None:
            raise RuntimeError("BrowserFetcher must be used as an async context manager")
        if not self._robots.is_allowed(url):
            self._logger.info("robots disallowed, skipping", extra={"url": url})
            return None
        for attempt in range(_MAX_CHALLENGE_RETRIES):
            await self._rate_limiter.acquire()
            try:
                response = await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await self._page.wait_for_timeout(_SETTLE_MS)
                html = await self._page.content()
            except Exception as exc:  # Playwright raises many unrelated error types
                self._logger.warning(
                    "browser fetch failed, recreating context",
                    extra={"url": url, "attempt": attempt + 1, "error": str(exc)[:200]},
                )
                # A hung navigation poisons the page; start a fresh context.
                await self._new_context()
                continue
            if not _is_challenge(html):
                status = response.status if response is not None else 200
                if status >= 400:  # noqa: PLR2004
                    self._logger.info("http error", extra={"url": url, "status": status})
                    return None
                return FetchResult(url=url, status=status, html=html, final_url=self._page.url)
            self._logger.info(
                "bot challenge page, retrying", extra={"url": url, "attempt": attempt + 1}
            )
            await self._new_context()
            await self._page.wait_for_timeout(_SETTLE_MS * (attempt + 1))
        self._logger.warning("page could not be fetched", extra={"url": url})
        return None

    async def fetch_resource(self, url: str) -> str | None:
        """Fetch a non-HTML resource (robots.txt, sitemap XML) through the
        browser context so the challenge cookies are reused.

        Args:
            url: Absolute URL of the resource.

        Returns:
            The response body, or ``None`` when blocked or failed.
        """
        if self._context is None or self._page is None:
            raise RuntimeError("BrowserFetcher must be used as an async context manager")
        for attempt in range(_MAX_CHALLENGE_RETRIES):
            await self._rate_limiter.acquire()
            try:
                response = await self._context.request.get(url)
                body = await response.text()
            except Exception as exc:  # Playwright raises many unrelated error types
                self._logger.warning(
                    "resource fetch failed", extra={"url": url, "error": str(exc)[:200]}
                )
                return None
            if response.ok and not _is_challenge(body):
                return body
            # Re-warm the challenge cookie by visiting the site root.
            root = url.split("/", 3)
            await self._page.goto(f"{root[0]}//{root[2]}/", wait_until="domcontentloaded")
            await self._page.wait_for_timeout(_SETTLE_MS * (attempt + 1))
        self._logger.warning("resource stayed behind bot challenge", extra={"url": url})
        return None


def _is_challenge(html: str) -> bool:
    """Heuristically detect Incapsula / Cloudflare challenge pages."""
    head = html[:4000]
    return any(marker in head for marker in _CHALLENGE_MARKERS)
