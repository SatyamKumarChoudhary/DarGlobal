"""Consistent error responses for the whole API."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from property_assistant.core.exceptions import AppError
from property_assistant.core.logging import request_id_var

logger = logging.getLogger(__name__)


def error_body(code: str, message: str) -> dict[str, dict[str, str | None]]:
    """The error envelope used by every failure path."""
    return {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers mapping AppError (and unexpected errors) to JSON bodies."""

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        """Map application errors to their declared HTTP status."""
        return JSONResponse(status_code=exc.http_status, content=error_body(exc.code, exc.message))

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        """Never leak internals from unexpected failures."""
        logger.exception("unhandled error", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content=error_body("internal_error", "An unexpected error occurred"),
        )
