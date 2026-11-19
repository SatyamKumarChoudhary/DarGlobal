"""Per-IP rate limiting shared by the app factory and the chat router."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from property_assistant.core.config import get_settings

limiter = Limiter(key_func=get_remote_address)


def chat_rate_limit() -> str:
    """The configured chat rate limit, resolved lazily so tests can override."""
    return get_settings().rate_limit
