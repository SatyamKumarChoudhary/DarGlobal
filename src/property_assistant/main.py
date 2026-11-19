"""FastAPI application factory."""

from fastapi import FastAPI

from property_assistant.core.config import get_settings
from property_assistant.core.logging import setup_logging


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return app
