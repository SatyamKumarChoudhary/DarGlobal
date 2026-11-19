# ---------- builder ----------
FROM python:3.12-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-group scrape --no-install-project
COPY src ./src
COPY data/raw ./data/raw
RUN uv sync --frozen --no-dev --no-group scrape
# Bake the embedding model and build the index at image build time
# (no runtime scraping or model downloads; no API key needed here).
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN uv run --no-dev --no-group scrape property-assistant ingest

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" FASTEMBED_CACHE_PATH=/app/.cache/fastembed PORT=7860
# HF Spaces requires uid 1000.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv ./.venv
COPY --from=builder --chown=appuser:appuser /app/src ./src
COPY --from=builder --chown=appuser:appuser /app/data ./data
COPY --from=builder --chown=appuser:appuser /app/.cache ./.cache
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health')" || exit 1
CMD ["sh", "-c", "uvicorn property_assistant.main:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
