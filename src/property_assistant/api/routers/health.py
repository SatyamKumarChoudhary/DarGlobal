"""Liveness and readiness probes."""

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, str]:
    """Readiness probe: 200 when the index is loaded, 503 otherwise."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not ready"}
    return {"status": "ready"}
