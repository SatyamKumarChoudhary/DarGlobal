"""API test fixtures: a real app with the container faked out."""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from property_assistant.api.dependencies import (
    get_chat_service,
    get_container,
    get_conversation_store,
)
from property_assistant.chat.cache import AnswerCache
from property_assistant.chat.citations import CitationFormatter
from property_assistant.chat.conversation import InMemoryConversationStore
from property_assistant.chat.guardrails import InputGuard
from property_assistant.chat.rewriter import QueryRewriter
from property_assistant.chat.service import ChatService
from property_assistant.core.config import Settings, get_settings
from property_assistant.llm.prompts import PromptBuilder
from property_assistant.main import create_app
from tests.conftest import FakeLLM, FakeRetriever


class FakeContainer:
    """Container stand-in exposing only what the routers use."""

    def __init__(self, chat_service: ChatService, store: InMemoryConversationStore) -> None:
        self.chat_service = chat_service
        self.conversation_store = store
        self.manifest: dict[str, Any] = {
            "n_docs": 42,
            "n_chunks": 256,
            "by_site": {"darglobal": 20, "wasalt": 22},
            "by_doc_type": {"project": 15, "listing": 27},
            "created_at": "2026-01-01T00:00:00+00:00",
            "embedding_model": "BAAI/bge-small-en-v1.5",
        }
        self.settings = _test_settings()


def _test_settings() -> Settings:
    return Settings(
        openrouter_api_key="sk-or-test",  # type: ignore[arg-type]
        llm_primary_model="primary/model:free",
        environment="test",
        rate_limit="1000/minute",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def app() -> FastAPI:
    """A real app instance with dependencies overridden by fakes."""
    get_settings.cache_clear()
    settings = _test_settings()
    application = create_app()

    llm = FakeLLM()
    retriever = FakeRetriever()
    store = InMemoryConversationStore(ttl_minutes=10, max_turns=12)
    prompt_builder = PromptBuilder(settings.max_history_turns)
    chat_service = ChatService(
        guard=InputGuard(settings.max_message_chars),
        store=store,
        rewriter=QueryRewriter(llm, prompt_builder),
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        citations=CitationFormatter(),
        cache=AnswerCache(ttl_s=60, max_size=16),
        settings=settings,
    )
    container = FakeContainer(chat_service, store)
    application.dependency_overrides[get_container] = lambda: container
    application.dependency_overrides[get_chat_service] = lambda: chat_service
    application.dependency_overrides[get_conversation_store] = lambda: store
    # Simulate a completed startup without running the real lifespan.
    application.state.container = container
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """A test client that does NOT run the lifespan (no index needed)."""
    return TestClient(app)
