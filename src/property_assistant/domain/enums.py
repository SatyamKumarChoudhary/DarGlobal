"""Domain enumerations."""

from enum import StrEnum


class SourceSite(StrEnum):
    """Which public site a document was scraped from."""

    DARGLOBAL = "darglobal"
    WASALT = "wasalt"

    @property
    def display_name(self) -> str:
        """Human-friendly site name used in chunk headers and citations."""
        return {SourceSite.DARGLOBAL: "Dar Global", SourceSite.WASALT: "Wasalt"}[self]


class DocType(StrEnum):
    """Coarse content category of a scraped page."""

    PROJECT = "project"
    PRESS = "press"
    PARTNER = "partner"
    ABOUT = "about"
    LISTING = "listing"
    ARTICLE = "article"
    PAGE = "page"


class Role(StrEnum):
    """Chat message author role (OpenAI-compatible)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
