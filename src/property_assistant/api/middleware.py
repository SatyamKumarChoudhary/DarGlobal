"""Request-id propagation and structured access logging."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from property_assistant.core.logging import request_id_var

_REQUEST_ID_HEADER = "X-Request-ID"

logger = logging.getLogger("property_assistant.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, propagates it to logs and emits one access line."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap the request with id propagation and latency logging."""
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[_REQUEST_ID_HEADER] = request_id
        latency_ms = int((time.perf_counter() - started) * 1000)
        if request.url.path not in ("/health", "/ready"):
            logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                },
            )
        return response
