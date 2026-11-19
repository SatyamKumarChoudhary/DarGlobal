"""JSONL persistence for scraped documents."""

import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from property_assistant.domain.enums import SourceSite
from property_assistant.domain.models import Document


class DocumentRepository:
    """Stores one JSONL snapshot file per site under ``data/raw/``."""

    def __init__(self, raw_dir: Path, logger: logging.Logger) -> None:
        """Args:
        raw_dir: Directory holding ``{site}.jsonl`` snapshot files.
        logger: Structured logger.
        """
        self._raw_dir = raw_dir
        self._logger = logger

    def save_all(self, site: SourceSite, docs: Iterable[Document]) -> int:
        """Write a site's documents atomically, de-duplicated by document id.

        Args:
            site: The source site the documents belong to.
            docs: Documents to persist.

        Returns:
            The number of unique documents written.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        unique: dict[str, Document] = {}
        for doc in docs:
            unique[doc.id] = doc
        target = self._path_for(site)
        fd, tmp_path = tempfile.mkstemp(dir=self._raw_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for doc in unique.values():
                    handle.write(doc.model_dump_json() + "\n")
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        self._logger.info("documents saved", extra={"site": site.value, "n_docs": len(unique)})
        return len(unique)

    def load_all(self) -> list[Document]:
        """Load every document from every site snapshot found on disk."""
        documents: list[Document] = []
        for site in SourceSite:
            path = self._path_for(site)
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line:
                        documents.append(Document.model_validate_json(line))
        return documents

    def count_by_site(self) -> dict[str, int]:
        """Return the number of stored documents per site."""
        counts: dict[str, int] = {}
        for doc in self.load_all():
            counts[doc.site.value] = counts.get(doc.site.value, 0) + 1
        return counts

    def _path_for(self, site: SourceSite) -> Path:
        """The snapshot path for a site."""
        return self._raw_dir / f"{site.value}.jsonl"
