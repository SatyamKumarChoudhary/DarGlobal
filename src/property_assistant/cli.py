"""Typer CLI entry points: scrape | ingest | serve | evaluate."""

import asyncio
import logging

import typer

from property_assistant.core.config import get_settings
from property_assistant.core.logging import setup_logging
from property_assistant.domain.enums import SourceSite

app = typer.Typer(name="property-assistant", help="DarGlobal & Wasalt property assistant CLI.")


@app.callback()
def _init() -> None:
    """Configure logging before any command runs."""
    setup_logging(get_settings().log_level)


@app.command()
def scrape(
    site: str = typer.Option("all", help="Site to scrape: darglobal | wasalt | all."),
    max_pages: int | None = typer.Option(None, help="Override the per-site page cap."),
) -> None:
    """Scrape the configured public sites and store JSONL snapshots."""
    from property_assistant.scraping.runner import run_scrape  # noqa: PLC0415 - optional deps

    settings = get_settings()
    if max_pages is not None:
        settings = settings.model_copy(
            update={
                "scraper_max_pages_darglobal": max_pages,
                "scraper_max_pages_wasalt": max_pages,
            }
        )
    sites = list(SourceSite) if site == "all" else [SourceSite(site)]
    logger = logging.getLogger("property_assistant.scrape")
    stats = asyncio.run(run_scrape(settings, sites, logger))
    for scraped_site, site_stats in stats.items():
        typer.echo(
            f"{scraped_site.value}: discovered={site_stats.discovered} "
            f"scraped={site_stats.scraped} skipped={site_stats.skipped} "
            f"failed={site_stats.failed} by_type={site_stats.by_doc_type}"
        )


@app.command()
def ingest() -> None:
    """Build the search indexes from the raw data snapshots."""
    from property_assistant.ingestion.chunk_store import ChunkStore  # noqa: PLC0415
    from property_assistant.ingestion.chunker import RecursiveTextChunker  # noqa: PLC0415
    from property_assistant.ingestion.cleaner import TextCleaner  # noqa: PLC0415
    from property_assistant.ingestion.embedder import FastEmbedEmbedder  # noqa: PLC0415
    from property_assistant.ingestion.keyword_index import Bm25KeywordIndex  # noqa: PLC0415
    from property_assistant.ingestion.pipeline import IngestionPipeline  # noqa: PLC0415
    from property_assistant.ingestion.vector_store import FaissVectorStore  # noqa: PLC0415
    from property_assistant.scraping.repository import DocumentRepository  # noqa: PLC0415

    settings = get_settings()
    logger = logging.getLogger("property_assistant.ingest")
    embedder = FastEmbedEmbedder(settings.embedding_model)
    pipeline = IngestionPipeline(
        repository=DocumentRepository(settings.raw_data_dir, logger),
        cleaner=TextCleaner(),
        chunker=RecursiveTextChunker(settings.chunk_size_chars, settings.chunk_overlap_chars),
        embedder=embedder,
        vector_store=FaissVectorStore(embedder.dimension),
        keyword_index=Bm25KeywordIndex(),
        chunk_store=ChunkStore(settings.index_dir),
        index_dir=settings.index_dir,
        embedding_model=settings.embedding_model,
        logger=logger,
    )
    report = pipeline.run()
    typer.echo(
        f"documents={report.n_documents} chunks={report.n_chunks} "
        f"by_site={report.by_site} by_type={report.by_doc_type} "
        f"model={report.embedding_model} dim={report.dimension}"
    )


@app.command()
def serve(port: int | None = typer.Option(None, help="Override the configured port.")) -> None:
    """Run the API server."""
    import uvicorn  # noqa: PLC0415 - deferred so other CLI commands start fast

    settings = get_settings()
    uvicorn.run(
        "property_assistant.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=port or settings.port,
    )


if __name__ == "__main__":
    app()
