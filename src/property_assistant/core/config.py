"""Application settings loaded from the environment (12-factor style)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from property_assistant.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Typed application configuration.

    Every value can be overridden through environment variables or a local
    ``.env`` file. Secrets are wrapped in :class:`~pydantic.SecretStr` so they
    never leak through ``repr`` or logs.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # App
    app_name: str = "DarGlobal Property Assistant"
    environment: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    port: int = 7860
    app_public_url: str = "http://localhost:7860"
    cors_origins: list[str] = ["*"]

    # OpenRouter
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_primary_model: str = "nvidia/nemotron-3.5-lightning:free"
    llm_fallback_models: list[str] = [
        "google/gemma-4-31b-it:free",
        "baidu/ernie-5.5-flash-preview:free",
        "openrouter/free",
    ]
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_dir: Path = Path("data/index")
    raw_data_dir: Path = Path("data/raw")
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    retrieval_top_k: int = 6
    retrieval_candidates: int = 20
    rrf_k: int = 60
    min_relevance_score: float = 0.30

    # Chat
    max_history_turns: int = 6
    session_ttl_minutes: int = 60
    max_message_chars: int = 1000
    answer_cache_ttl_s: int = 3600
    rate_limit: str = "10/minute"

    # Scraping
    scraper_user_agent: str = "DarGlobalAssignmentBot/1.0 (+contact: shivreti@gmail.com)"
    scraper_rate_per_sec: float = 1.0
    scraper_max_pages_darglobal: int = 300
    scraper_max_pages_wasalt: int = 400

    @field_validator("llm_primary_model")
    @classmethod
    def _primary_must_be_free(cls, value: str) -> str:
        """Reject paid model IDs to protect against accidental billing."""
        if not _is_free_model(value):
            raise ConfigurationError(f"llm_primary_model must be a ':free' model, got {value!r}")
        return value

    @field_validator("llm_fallback_models")
    @classmethod
    def _fallbacks_must_be_free(cls, value: list[str]) -> list[str]:
        """Reject paid model IDs in the fallback list."""
        for model in value:
            if not _is_free_model(model):
                raise ConfigurationError(
                    f"llm_fallback_models must contain only ':free' models, got {model!r}"
                )
        return value

    @property
    def llm_models(self) -> list[str]:
        """The ordered model list: primary first, then the fallbacks."""
        ordered = [self.llm_primary_model]
        ordered.extend(m for m in self.llm_fallback_models if m not in ordered)
        return ordered


def _is_free_model(model_id: str) -> bool:
    """Return True when a model ID is guaranteed free on OpenRouter."""
    return model_id.endswith(":free") or model_id == "openrouter/free"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
