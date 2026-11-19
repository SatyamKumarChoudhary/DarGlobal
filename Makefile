.PHONY: install lint format typecheck test scrape ingest eval run docker-build docker-run free-models

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

test:
	uv run pytest --cov=src --cov-report=term-missing

scrape:
	uv run property-assistant scrape --site all

ingest:
	uv run property-assistant ingest

eval:
	uv run python scripts/evaluate.py

run:
	uv run uvicorn property_assistant.main:create_app --factory --reload --port 7860

docker-build:
	docker build -t property-assistant .

docker-run:
	docker run --rm -p 7860:7860 --env-file .env property-assistant

free-models:
	uv run python scripts/list_free_models.py
