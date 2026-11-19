"""FastAPI application factory and lifespan wiring."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from property_assistant.api.errors import error_body, register_exception_handlers
from property_assistant.api.middleware import RequestContextMiddleware
from property_assistant.api.ratelimit import limiter
from property_assistant.api.routers import chat, health, meta
from property_assistant.container import AppContainer
from property_assistant.core.config import get_settings
from property_assistant.core.logging import setup_logging

_WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the container at startup and release it at shutdown."""
    settings = get_settings()
    logger = logging.getLogger("property_assistant.main")
    container = AppContainer.build(settings)
    app.state.container = container
    logger.info(
        "startup complete",
        extra={"manifest": container.manifest, "llm_models": settings.llm_models},
    )
    try:
        yield
    finally:
        await container.aclose()


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=_lifespan)

    app.state.limiter = limiter

    def _rate_limit_handler(request: Request, exc: Exception) -> Response:
        """Map slowapi's exception to the standard error envelope."""
        return JSONResponse(
            status_code=429,
            content=error_body("rate_limited", "Too many requests; please slow down"),
        )

    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    register_exception_handlers(app)

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(meta.router)

    app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the chat UI."""
        return FileResponse(_WEB_DIR / "templates" / "index.html")

    return app
