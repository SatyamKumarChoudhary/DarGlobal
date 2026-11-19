"""Content extraction strategies.

``NextDataExtractor`` mines the Next.js ``__NEXT_DATA__`` JSON payload (both
target sites are Next.js apps backed by a CMS), ``TrafilaturaExtractor`` pulls
the readable body text, and ``CompositeExtractor`` merges the two.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import trafilatura
from selectolax.lexbor import LexborHTMLParser as HTMLParser

_MIN_MERGED_TEXT_CHARS = 100
_MAX_HARVEST_DEPTH = 12

# JSON keys whose string values usually carry user-visible content.
_TEXT_KEYS = frozenset(
    {
        "title",
        "heading",
        "subheading",
        "subtitle",
        "description",
        "description2",
        "text",
        "content",
        "body",
        "summary",
        "label",
        "name",
        "question",
        "answer",
        "address",
        "caption",
        "metatitle",
        "metadescription",
        "alt",
    }
)
# JSON keys that carry structured facts worth keeping as key/value pairs.
_FACT_KEYS = frozenset(
    {
        "location",
        "city",
        "district",
        "country",
        "price",
        "startingprice",
        "currency",
        "bedrooms",
        "bathrooms",
        "area",
        "builtuparea",
        "carpetarea",
        "propertytype",
        "purpose",
        "developer",
        "brand",
        "status",
        "handover",
        "completiondate",
        "launchdate",
        "date",
        "publishedat",
        "furnishingtype",
        "rega",
        "regaverified",
        "propertyfor",
        "noofbedrooms",
        "noofbathrooms",
    }
)
# Subtrees that never carry content.
_SKIP_KEYS = frozenset(
    {
        "image",
        "images",
        "img",
        "dimg",
        "mimg",
        "icon",
        "logo",
        "video",
        "textures",
        "formbgcolor",
        "bgcolor",
        "color",
        "css",
        "styles",
        "url",
        "href",
        "link",
        "localizations",
        "menuheader",
        "footer",
        "homefooter",
        "header",
        "navigation",
        "nav",
        "menu",
        "seo",
        "structureddata",
        "scripts",
        "query",
        "runtimeconfig",
        "buildid",
        "_sentrytracedata",
        "_sentrybaggage",
        # Site-wide subtrees repeated on every page (forms, footers, legal text).
        "globaldata",
        "thankyoupagedata",
        "countries",
        "countrylist",
        "formfields",
        "bannerdata",
        "popup",
        "cookie",
    }
)
_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Asset filenames that leak through generic keys like "name".
_ASSET_RE = re.compile(r"\.(?:png|jpe?g|webp|gif|svg|mp4|webm|pdf|ico|woff2?)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ExtractedContent:
    """Normalized output of a content extractor."""

    title: str
    text: str
    structured: dict[str, Any] = field(default_factory=dict)
    lang: str | None = None


class ContentExtractor(Protocol):
    """Strategy interface: turn raw HTML into extracted content."""

    def extract(self, url: str, html: str) -> ExtractedContent | None:
        """Extract content from a page, or return None when nothing useful."""
        ...


class NextDataExtractor:
    """Harvests readable text and structured facts from ``__NEXT_DATA__``.

    Works generically for any Next.js/CMS payload: it walks the JSON tree,
    keeps string values under content-ish keys, collects fact-ish keys into a
    structured dict, and skips style/asset/navigation subtrees.
    """

    def extract(self, url: str, html: str) -> ExtractedContent | None:
        """Extract from the Next.js data script plus meta/JSON-LD tags.

        Args:
            url: The page URL (unused, kept for the protocol).
            html: Raw page HTML.

        Returns:
            Extracted content, or None when there is no usable payload.
        """
        tree = HTMLParser(html)
        payload = self._next_data_payload(tree)
        lines: list[str] = []
        facts: dict[str, Any] = {}
        if payload is not None:
            page_props = payload.get("props", {}).get("pageProps", {})
            self._harvest(page_props, lines, facts, depth=0)
        meta_description = self._meta_description(tree)
        if meta_description:
            lines.insert(0, meta_description)
        self._harvest_json_ld(tree, lines, facts)
        title = self._page_title(tree) or next(iter(lines), "")[:120]
        text = _dedupe_lines(lines)
        if not text and not facts:
            return None
        if facts:
            fact_lines = [f"{key}: {value}" for key, value in facts.items()]
            text = text + ("\n" if text else "") + "\n".join(fact_lines)
        lang = None
        html_node = tree.css_first("html")
        if html_node is not None:
            lang = html_node.attributes.get("lang")
        return ExtractedContent(title=title, text=text, structured=facts, lang=lang)

    @staticmethod
    def _next_data_payload(tree: HTMLParser) -> dict[str, Any] | None:
        """Parse the ``__NEXT_DATA__`` script tag if present."""
        node = tree.css_first("script#__NEXT_DATA__")
        if node is None or not node.text():
            return None
        try:
            parsed: dict[str, Any] = json.loads(node.text())
        except json.JSONDecodeError:
            return None
        return parsed

    @classmethod
    def _harvest(cls, obj: Any, lines: list[str], facts: dict[str, Any], depth: int) -> None:
        """Recursively collect text lines and structured facts from JSON."""
        if depth > _MAX_HARVEST_DEPTH:
            return
        if isinstance(obj, dict):
            for raw_key, value in obj.items():
                key = raw_key.lower()
                if key in _SKIP_KEYS:
                    continue
                if key in _FACT_KEYS and isinstance(value, str | int | float) and value != "":
                    facts.setdefault(key, value)
                    continue
                if key in _TEXT_KEYS and isinstance(value, str):
                    cleaned = _clean_fragment(value)
                    if cleaned and not _ASSET_RE.search(cleaned):
                        lines.append(cleaned)
                    continue
                cls._harvest(value, lines, facts, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                cls._harvest(item, lines, facts, depth + 1)

    @staticmethod
    def _meta_description(tree: HTMLParser) -> str | None:
        """Read <meta name=description> or the OpenGraph description."""
        for selector in ('meta[name="description"]', 'meta[property="og:description"]'):
            node = tree.css_first(selector)
            if node is not None:
                content = node.attributes.get("content", "")
                if content and content.strip():
                    return _clean_fragment(content)
        return None

    @classmethod
    def _harvest_json_ld(cls, tree: HTMLParser, lines: list[str], facts: dict[str, Any]) -> None:
        """Mine ``application/ld+json`` blocks for additional facts."""
        for node in tree.css('script[type="application/ld+json"]'):
            if not node.text():
                continue
            try:
                data = json.loads(node.text())
            except json.JSONDecodeError:
                continue
            cls._harvest(data, lines, facts, depth=8)  # shallow harvest only

    @staticmethod
    def _page_title(tree: HTMLParser) -> str:
        """Best-effort page title: og:title, <title>, then first <h1>."""
        og = tree.css_first('meta[property="og:title"]')
        if og is not None and og.attributes.get("content"):
            return _clean_fragment(og.attributes["content"] or "")
        title_node = tree.css_first("title")
        if title_node is not None and title_node.text().strip():
            return _clean_fragment(title_node.text())
        h1 = tree.css_first("h1")
        if h1 is not None:
            return _clean_fragment(h1.text())
        return ""


class TrafilaturaExtractor:
    """Extracts the readable main body text with trafilatura."""

    def extract(self, url: str, html: str) -> ExtractedContent | None:
        """Extract body text.

        Args:
            url: The page URL, passed to trafilatura for context.
            html: Raw page HTML.

        Returns:
            Extracted content, or None when no main text was found.
        """
        text = trafilatura.extract(
            html, url=url, include_tables=True, favor_recall=True, include_comments=False
        )
        if not text or not text.strip():
            return None
        tree = HTMLParser(html)
        title_node = tree.css_first("title")
        title = _clean_fragment(title_node.text()) if title_node is not None else ""
        return ExtractedContent(title=title, text=text.strip())


class CompositeExtractor:
    """Runs every extractor and merges their output.

    Structured facts come from the Next.js payload; body text is the union of
    all extracted text with duplicate lines removed. Returns None when the
    merged text is too short to be useful.
    """

    def __init__(self, extractors: Sequence[ContentExtractor]) -> None:
        """Args:
        extractors: Strategies to run, in priority order for the title.
        """
        self._extractors = extractors

    def extract(self, url: str, html: str) -> ExtractedContent | None:
        """Merge the results of all extractors.

        Args:
            url: The page URL.
            html: Raw page HTML.

        Returns:
            Merged content, or None when under the minimum useful length.
        """
        results = [r for e in self._extractors if (r := e.extract(url, html)) is not None]
        if not results:
            return None
        title = next((r.title for r in results if r.title), "")
        merged_text = _dedupe_lines(
            [line for result in results for line in result.text.splitlines()]
        )
        structured: dict[str, Any] = {}
        for result in results:
            for key, value in result.structured.items():
                structured.setdefault(key, value)
        if len(merged_text) < _MIN_MERGED_TEXT_CHARS:
            return None
        lang = next((r.lang for r in results if r.lang), None)
        return ExtractedContent(title=title, text=merged_text, structured=structured, lang=lang)


def _clean_fragment(value: str) -> str:
    """Strip HTML tags and collapse whitespace in a text fragment."""
    return _WS_RE.sub(" ", _HTML_TAG_RE.sub(" ", value)).strip()


def _dedupe_lines(lines: list[str]) -> str:
    """Normalize, de-duplicate (case-insensitive) and re-join text lines."""
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        normalized = _WS_RE.sub(" ", line).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            kept.append(normalized)
    return "\n".join(kept)
