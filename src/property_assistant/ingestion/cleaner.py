"""Text normalization and boilerplate removal."""

import re
import unicodedata
from collections import Counter

_WS_RE = re.compile(r"[ \t]+")
_BOILERPLATE_PATTERNS = (
    re.compile(r"cookie", re.IGNORECASE),
    re.compile(r"download the app", re.IGNORECASE),
    re.compile(r"subscribe to (our )?newsletter", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"^\s*(accept|decline|ok|close)\s*$", re.IGNORECASE),
)
# A line repeated in more than this share of documents is navigation/footer noise.
_REPEAT_THRESHOLD = 0.30
_MIN_DOCS_FOR_FREQUENCY = 10


class TextCleaner:
    """Normalizes unicode and strips boilerplate lines.

    Call :meth:`fit` with the whole corpus first so lines repeated across many
    documents (navigation, footers) can be detected by frequency; then
    :meth:`clean` drops them per document.
    """

    def __init__(self) -> None:
        self._frequent_lines: set[str] = set()

    def fit(self, texts: list[str]) -> None:
        """Learn which exact lines recur across the corpus.

        Args:
            texts: Every document's raw text.
        """
        if len(texts) < _MIN_DOCS_FOR_FREQUENCY:
            return
        line_doc_counts: Counter[str] = Counter()
        for text in texts:
            unique_lines = {line.strip() for line in text.splitlines() if line.strip()}
            line_doc_counts.update(unique_lines)
        threshold = len(texts) * _REPEAT_THRESHOLD
        self._frequent_lines = {
            line for line, count in line_doc_counts.items() if count > threshold
        }

    def clean(self, text: str) -> str:
        """Normalize one document's text.

        Applies NFKC unicode normalization, collapses whitespace, and removes
        boilerplate lines plus lines learned as cross-document repeats.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.
        """
        normalized = unicodedata.normalize("NFKC", text)
        kept: list[str] = []
        for raw_line in normalized.splitlines():
            line = _WS_RE.sub(" ", raw_line).strip()
            if not line:
                continue
            if line in self._frequent_lines:
                continue
            if any(p.search(line) for p in _BOILERPLATE_PATTERNS):
                continue
            kept.append(line)
        return "\n".join(kept)
