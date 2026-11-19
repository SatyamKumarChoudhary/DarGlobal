"""Composition root: builds the full object graph once at startup."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Self

import httpx

from property_assistant.chat.cache import AnswerCache
from property_assistant.chat.citations import CitationFormatter
from property_assistant.chat.conversation import InMemoryConversationStore
from property_assistant.chat.guardrails import InputGuard
from property_assistant.chat.rewriter import QueryRewriter
from property_assistant.chat.service import ChatService
from property_assistant.core.config import Settings
from property_assistant.core.exceptions import ConfigurationError, IndexNotReadyError
from property_assistant.ingestion.chunk_store import ChunkStore
from property_assistant.ingestion.embedder import FastEmbedEmbedder
from property_assistant.ingestion.keyword_index import Bm25KeywordIndex
from property_assistant.ingestion.pipeline import MANIFEST_FILE
from property_assistant.ingestion.vector_store import FaissVectorStore
from property_assistant.llm.openrouter import OpenRouterClient
from property_assistant.llm.prompts import PromptBuilder
from property_assistant.retrieval.hybrid import HybridRetriever
from property_assistant.retrieval.keyword import KeywordRetriever
from property_assistant.retrieval.vector import VectorRetriever

_ANSWER_CACHE_MAX_SIZE = 256


@dataclass(slots=True)
class AppContainer:
    """Holds every long-lived service for the API process."""

    settings: Settings
    http_client: httpx.AsyncClient
    chat_service: ChatService
    conversation_store: InMemoryConversationStore
    manifest: dict[str, Any]

    @classmethod
    def build(cls, settings: Settings) -> Self:
        """Construct and wire the full object graph.

        Args:
            settings: Validated application settings.

        Returns:
            A ready container with the indexes loaded.

        Raises:
            IndexNotReadyError: When the index artifacts are missing.
            ConfigurationError: When the index was built with another
                embedding model than the one configured.
        """
        logger = logging.getLogger("property_assistant.container")
        manifest = cls._load_manifest(settings)

        embedder = FastEmbedEmbedder(settings.embedding_model)
        vector_store = FaissVectorStore(embedder.dimension)
        vector_store.load(settings.index_dir)
        keyword_index = Bm25KeywordIndex()
        keyword_index.load(settings.index_dir)
        chunk_store = ChunkStore(settings.index_dir)
        chunk_store.load()
        logger.info(
            "index loaded",
            extra={
                "n_chunks": len(chunk_store),
                "embedding_model": settings.embedding_model,
                "llm_models": settings.llm_models,
            },
        )

        retriever = HybridRetriever(
            retrievers=[
                VectorRetriever(embedder, vector_store, chunk_store),
                KeywordRetriever(keyword_index, chunk_store),
            ],
            candidates=settings.retrieval_candidates,
            rrf_k=settings.rrf_k,
        )

        http_client = httpx.AsyncClient()
        llm = OpenRouterClient(http_client, settings)
        prompt_builder = PromptBuilder(settings.max_history_turns)
        conversation_store = InMemoryConversationStore(
            ttl_minutes=settings.session_ttl_minutes,
            max_turns=settings.max_history_turns * 2,
        )
        chat_service = ChatService(
            guard=InputGuard(settings.max_message_chars),
            store=conversation_store,
            rewriter=QueryRewriter(llm, prompt_builder),
            retriever=retriever,
            prompt_builder=prompt_builder,
            llm=llm,
            citations=CitationFormatter(),
            cache=AnswerCache(ttl_s=settings.answer_cache_ttl_s, max_size=_ANSWER_CACHE_MAX_SIZE),
            settings=settings,
        )
        return cls(
            settings=settings,
            http_client=http_client,
            chat_service=chat_service,
            conversation_store=conversation_store,
            manifest=manifest,
        )

    async def aclose(self) -> None:
        """Release shared resources."""
        await self.http_client.aclose()

    @staticmethod
    def _load_manifest(settings: Settings) -> dict[str, Any]:
        """Load and validate the index manifest."""
        manifest_path = settings.index_dir / MANIFEST_FILE
        if not manifest_path.exists():
            raise IndexNotReadyError(
                f"Index manifest not found at {manifest_path}; run `property-assistant ingest`"
            )
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        built_with = manifest.get("embedding_model")
        if built_with != settings.embedding_model:
            raise ConfigurationError(
                f"Index was built with {built_with!r} but settings require "
                f"{settings.embedding_model!r}; re-run ingestion"
            )
        return manifest
