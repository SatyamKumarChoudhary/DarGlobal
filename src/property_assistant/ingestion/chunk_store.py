"""Chunk persistence and id lookup."""

from pathlib import Path

from property_assistant.domain.models import Chunk

_CHUNKS_FILE = "chunks.jsonl"


class ChunkStore:
    """Maps chunk ids to chunks, persisted as ``chunks.jsonl``."""

    def __init__(self, directory: Path) -> None:
        """Args:
        directory: The index directory holding ``chunks.jsonl``.
        """
        self._path = directory / _CHUNKS_FILE
        self._chunks: dict[str, Chunk] = {}

    def save(self, chunks: list[Chunk]) -> None:
        """Persist chunks and refresh the in-memory map."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        self._chunks = {chunk.id: chunk for chunk in chunks}

    def load(self) -> None:
        """Load chunks from disk into memory."""
        chunks: dict[str, Chunk] = {}
        with self._path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    chunk = Chunk.model_validate_json(line)
                    chunks[chunk.id] = chunk
        self._chunks = chunks

    def get(self, chunk_id: str) -> Chunk:
        """Return one chunk by id.

        Raises:
            KeyError: When the id is unknown.
        """
        return self._chunks[chunk_id]

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Return chunks for the given ids, skipping unknown ids."""
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def __len__(self) -> int:
        """Number of loaded chunks."""
        return len(self._chunks)
