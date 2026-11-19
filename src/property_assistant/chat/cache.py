"""TTL + LRU cache for answers to standalone questions."""

import re
import time
from collections import OrderedDict

from property_assistant.domain.models import Answer

_NORMALIZE_RE = re.compile(r"[^\w\s]")


class AnswerCache:
    """Bounded TTL LRU cache keyed by the normalized standalone query.

    Only answers produced without conversation history are cached, because a
    follow-up's answer depends on its session context.
    """

    def __init__(self, ttl_s: int, max_size: int) -> None:
        """Args:
        ttl_s: Entry lifetime in seconds.
        max_size: Maximum number of cached answers (LRU eviction).
        """
        self._ttl_s = ttl_s
        self._max_size = max_size
        self._entries: OrderedDict[str, tuple[float, Answer]] = OrderedDict()

    def get(self, query: str) -> Answer | None:
        """Return the cached answer for a query, or None on miss/expiry."""
        key = _normalize(query)
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, answer = entry
        if time.monotonic() > expires_at:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return answer

    def put(self, query: str, answer: Answer) -> None:
        """Cache an answer, evicting the least recently used when full."""
        key = _normalize(query)
        self._entries[key] = (time.monotonic() + self._ttl_s, answer)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)


def _normalize(query: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    return " ".join(_NORMALIZE_RE.sub(" ", query.lower()).split())
