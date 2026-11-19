"""Application exception hierarchy.

Every error carries a stable machine-readable ``code`` and the HTTP status the
API layer should map it to, so error handling stays consistent end to end.
"""


class AppError(Exception):
    """Base class for all application errors."""

    code: str = "app_error"
    http_status: int = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__


class ConfigurationError(AppError):
    """Raised when the configuration is invalid or inconsistent."""

    code = "configuration_error"
    http_status = 500


class ScrapingError(AppError):
    """Raised by the scraping pipeline (CLI only, never surfaced via HTTP)."""

    code = "scraping_error"


class RobotsDisallowedError(ScrapingError):
    """Raised when robots.txt forbids fetching a URL."""

    code = "robots_disallowed"


class IndexNotReadyError(AppError):
    """Raised when the search index has not been built or loaded yet."""

    code = "index_not_ready"
    http_status = 503


class LLMError(AppError):
    """Raised when the upstream LLM call fails."""

    code = "llm_error"
    http_status = 502


class LLMRateLimitError(LLMError):
    """Raised when the upstream LLM reports a rate limit (HTTP 429)."""

    code = "llm_rate_limited"
    http_status = 429


class LLMUnavailableError(LLMError):
    """Raised when every configured model failed to answer."""

    code = "llm_unavailable"
    http_status = 502


class GuardrailViolation(AppError):  # noqa: N818 - name is part of the public contract
    """Raised when user input fails validation (empty, too long, ...)."""

    code = "guardrail_violation"
    http_status = 400


class SessionNotFoundError(AppError):
    """Raised when a chat session ID is unknown."""

    code = "session_not_found"
    http_status = 404
