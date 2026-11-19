"""Typer CLI entry points: scrape | ingest | serve | evaluate."""

import typer

from property_assistant.core.config import get_settings
from property_assistant.core.logging import setup_logging

app = typer.Typer(name="property-assistant", help="DarGlobal & Wasalt property assistant CLI.")


@app.callback()
def _init() -> None:
    """Configure logging before any command runs."""
    setup_logging(get_settings().log_level)


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
