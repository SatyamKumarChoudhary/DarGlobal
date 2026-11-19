"""Golden-set retrieval evaluation.

Reports hit@k and MRR against ``data/eval/golden_questions.yaml`` and writes
the results table to ``docs/EVALUATION.md``.

Usage:
    uv run python scripts/evaluate.py [--k 6] [--with-llm]
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from property_assistant.core.config import get_settings  # noqa: E402
from property_assistant.ingestion.chunk_store import ChunkStore  # noqa: E402
from property_assistant.ingestion.embedder import FastEmbedEmbedder  # noqa: E402
from property_assistant.ingestion.keyword_index import Bm25KeywordIndex  # noqa: E402
from property_assistant.ingestion.vector_store import FaissVectorStore  # noqa: E402
from property_assistant.retrieval.hybrid import HybridRetriever  # noqa: E402
from property_assistant.retrieval.keyword import KeywordRetriever  # noqa: E402
from property_assistant.retrieval.vector import VectorRetriever  # noqa: E402

GOLDEN_PATH = Path("data/eval/golden_questions.yaml")
OUTPUT_PATH = Path("docs/EVALUATION.md")


def build_retriever() -> HybridRetriever:
    """Load the persisted indexes and assemble the hybrid retriever."""
    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    vector_store = FaissVectorStore(embedder.dimension)
    vector_store.load(settings.index_dir)
    keyword_index = Bm25KeywordIndex()
    keyword_index.load(settings.index_dir)
    chunk_store = ChunkStore(settings.index_dir)
    chunk_store.load()
    return HybridRetriever(
        retrievers=[
            VectorRetriever(embedder, vector_store, chunk_store),
            KeywordRetriever(keyword_index, chunk_store),
        ],
        candidates=settings.retrieval_candidates,
        rrf_k=settings.rrf_k,
    )


async def evaluate(k: int) -> None:
    """Run retrieval for every golden question and report hit@k and MRR."""
    questions = yaml.safe_load(GOLDEN_PATH.read_text())["questions"]
    retriever = build_retriever()
    rows: list[tuple[str, str, bool, float]] = []
    for item in questions:
        question = item["question"]
        expected = item["expected_url_contains"].lower()
        results = await retriever.retrieve(question, k)
        rank = next(
            (
                position
                for position, retrieved in enumerate(results, start=1)
                if expected in str(retrieved.chunk.url).lower()
            ),
            0,
        )
        rows.append((question, expected, rank > 0, 1.0 / rank if rank else 0.0))

    hits = sum(1 for _, _, hit, _ in rows if hit)
    hit_rate = hits / len(rows)
    mrr = sum(rr for _, _, _, rr in rows) / len(rows)

    lines = [
        "# Retrieval evaluation",
        "",
        f"_Run: {datetime.now(UTC).isoformat(timespec='seconds')} — "
        f"hybrid retrieval, k={k}_",
        "",
        f"**hit@{k}: {hit_rate:.2f}** ({hits}/{len(rows)})  |  **MRR: {mrr:.2f}**",
        "",
        "| Question | Expected URL fragment | Hit | RR |",
        "|---|---|---|---|",
    ]
    for question, expected, hit, rr in rows:
        lines.append(f"| {question} | `{expected}` | {'✅' if hit else '❌'} | {rr:.2f} |")
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"hit@{k}: {hit_rate:.2f} ({hits}/{len(rows)})  MRR: {mrr:.2f}")
    for question, expected, hit, _ in rows:
        print(f"  {'✅' if hit else '❌'}  {question}  [{expected}]")
    print(f"\nWrote {OUTPUT_PATH}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6, help="Retrieval depth for hit@k.")
    args = parser.parse_args()
    asyncio.run(evaluate(args.k))


if __name__ == "__main__":
    main()
