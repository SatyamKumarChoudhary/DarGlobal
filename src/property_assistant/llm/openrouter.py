"""OpenRouter chat-completions client with model fallback and SSE streaming."""

import json
import logging
import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from property_assistant.core.config import Settings
from property_assistant.core.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from property_assistant.llm.base import LLMMessage, LLMResult, StreamChunk

_MAX_RETRY_AFTER_S = 5.0
_RETRYABLE_STATUSES = frozenset({408, 500, 502, 503, 504})
# OpenRouter rejects requests whose native-fallback `models` array exceeds 3.
_MAX_MODELS_ARRAY = 3
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class _ThinkStripper:
    """Stateful ``<think>...</think>`` remover for streamed deltas."""

    def __init__(self) -> None:
        self._inside = False
        self._buffer = ""

    def feed(self, delta: str) -> str:
        """Consume a delta and return only the visible (non-reasoning) part."""
        self._buffer += delta
        output: list[str] = []
        while self._buffer:
            if self._inside:
                end = self._buffer.find("</think>")
                if end < 0:
                    self._buffer = self._buffer[-8:]  # keep a potential partial tag
                    break
                self._buffer = self._buffer[end + len("</think>") :]
                self._inside = False
            else:
                start = self._buffer.find("<think>")
                if start < 0:
                    # Hold back a possible partial opening tag at the end.
                    safe_len = len(self._buffer)
                    for i in range(1, min(len("<think>"), safe_len) + 1):
                        if "<think>".startswith(self._buffer[-i:]):
                            safe_len = len(self._buffer) - i
                            break
                    output.append(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                    break
                output.append(self._buffer[:start])
                self._buffer = self._buffer[start + len("<think>") :]
                self._inside = True
        return "".join(output)


class OpenRouterClient:
    """Calls OpenRouter's OpenAI-compatible API with client-side model fallback.

    Sends the full ``models`` array so OpenRouter can route natively, and also
    iterates the model list client-side when a model fails outright. Retries
    transient errors with exponential backoff; respects short ``Retry-After``
    waits on 429 and raises :class:`LLMRateLimitError` for longer ones.
    """

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        """Args:
        client: Shared async HTTP client.
        settings: Application settings (key, base URL, model lists, limits).
        """
        self._client = client
        self._settings = settings
        self._logger = logging.getLogger(__name__)

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """Run one completion, falling back across the configured models.

        Args:
            messages: Conversation messages.
            max_tokens: Optional completion token cap.
            temperature: Optional sampling temperature.

        Returns:
            The completion result (with any ``<think>`` blocks stripped).

        Raises:
            LLMRateLimitError: When rate-limited beyond a short wait.
            LLMUnavailableError: When every configured model failed.
        """
        models = self._settings.llm_models
        last_error: LLMError | None = None
        for model in models:
            started = time.perf_counter()
            try:
                data = await self._post_json(
                    self._payload(messages, model, models, max_tokens, temperature, stream=False)
                )
                text = self._extract_text(data)
            except LLMRateLimitError:
                raise
            except LLMError as exc:
                self._logger.warning(
                    "model failed, trying next", extra={"model": model, "error": exc.message}
                )
                last_error = exc
                continue
            usage = data.get("usage") or {}
            latency_ms = int((time.perf_counter() - started) * 1000)
            answered_model = data.get("model", model)
            self._logger.info(
                "llm completion",
                extra={
                    "model": answered_model,
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            )
            return LLMResult(
                text=_THINK_RE.sub("", text).strip(),
                model=answered_model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        raise last_error or LLMUnavailableError("All configured models failed")

    async def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion, falling back across models before first token.

        Args:
            messages: Conversation messages.
            max_tokens: Optional completion token cap.
            temperature: Optional sampling temperature.

        Yields:
            Visible text deltas (``<think>`` blocks are stripped statefully).

        Raises:
            LLMRateLimitError: When rate-limited beyond a short wait.
            LLMUnavailableError: When every configured model failed.
        """
        models = self._settings.llm_models
        last_error: LLMError | None = None
        for model in models:
            payload = self._payload(messages, model, models, max_tokens, temperature, stream=True)
            stripper = _ThinkStripper()
            emitted = False
            try:
                async for chunk in self._stream_once(payload):
                    visible = stripper.feed(chunk.delta)
                    if visible:
                        emitted = True
                        yield StreamChunk(delta=visible, model=chunk.model or model)
                return
            except LLMRateLimitError:
                raise
            except LLMError as exc:
                if emitted:
                    # Tokens already reached the user; do not restart mid-answer.
                    raise
                self._logger.warning(
                    "stream failed, trying next model",
                    extra={"model": model, "error": exc.message},
                )
                last_error = exc
        raise last_error or LLMUnavailableError("All configured models failed")

    def _payload(
        self,
        messages: Sequence[LLMMessage],
        model: str,
        models: list[str],
        max_tokens: int | None,
        temperature: float | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the chat-completions request body.

        The native-fallback ``models`` array starts at the currently attempted
        model and is capped at OpenRouter's limit; the client-side loop still
        covers the full configured list.
        """
        start = models.index(model) if model in models else 0
        native_fallbacks = models[start : start + _MAX_MODELS_ARRAY]
        return {
            "model": model,
            "models": native_fallbacks,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": (
                temperature if temperature is not None else self._settings.llm_temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else self._settings.llm_max_tokens,
            "stream": stream,
        }

    def _headers(self) -> dict[str, str]:
        """Auth and attribution headers required by OpenRouter."""
        return {
            "Authorization": f"Bearer {self._settings.openrouter_api_key.get_secret_value()}",
            "HTTP-Referer": self._settings.app_public_url,
            "X-Title": self._settings.app_name,
            "Content-Type": "application/json",
        }

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retries on transient failures; normalize errors."""
        url = f"{self._settings.openrouter_base_url}/chat/completions"
        attempts = self._settings.llm_max_retries + 1
        delay = 1.0
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=httpx.Timeout(self._settings.llm_timeout_s, connect=10.0),
                )
            except httpx.HTTPError as exc:
                if attempt + 1 == attempts:
                    raise LLMError(f"Connection to OpenRouter failed: {exc}") from exc
                await _sleep(delay)
                delay *= 2
                continue
            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                await self._handle_rate_limit(response)
                continue
            if response.status_code in _RETRYABLE_STATUSES and attempt + 1 < attempts:
                await _sleep(delay)
                delay *= 2
                continue
            if response.status_code != httpx.codes.OK:
                raise LLMError(f"OpenRouter returned HTTP {response.status_code}")
            data: dict[str, Any] = response.json()
            if "error" in data:
                raise LLMError(str(data["error"].get("message", "unknown provider error")))
            return data
        raise LLMError("Retries exhausted")

    async def _handle_rate_limit(self, response: httpx.Response) -> None:
        """Wait for a short Retry-After, otherwise surface the rate limit."""
        retry_after = _parse_retry_after(response)
        if retry_after is not None and retry_after <= _MAX_RETRY_AFTER_S:
            self._logger.info("rate limited, waiting", extra={"retry_after_s": retry_after})
            await _sleep(retry_after)
            return
        raise LLMRateLimitError("OpenRouter rate limit exceeded")

    async def _stream_once(self, payload: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Stream one request, parsing OpenRouter's SSE protocol."""
        url = f"{self._settings.openrouter_base_url}/chat/completions"
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers(),
                timeout=httpx.Timeout(self._settings.llm_timeout_s, connect=10.0),
            ) as response:
                if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                    await response.aread()
                    await self._handle_rate_limit(response)
                    raise LLMError("Rate limited during stream start")
                if response.status_code != httpx.codes.OK:
                    await response.aread()
                    raise LLMError(f"OpenRouter returned HTTP {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue  # ignore ": OPENROUTER PROCESSING" comments
                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if "error" in event:
                        raise LLMError(
                            str(event["error"].get("message", "mid-stream provider error"))
                        )
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        yield StreamChunk(delta=delta, model=event.get("model", ""))
        except httpx.HTTPError as exc:
            raise LLMError(f"Stream connection failed: {exc}") from exc

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull the assistant text out of a completion response."""
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Empty choices in OpenRouter response")
        content = (choices[0].get("message") or {}).get("content")
        if not content:
            raise LLMError("Empty content in OpenRouter response")
        return str(content)


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Parse the Retry-After header (seconds form only)."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def _sleep(seconds: float) -> None:
    """Module-level sleep hook (patchable in tests)."""
    import asyncio  # noqa: PLC0415 - trivial, keeps the hook self-contained

    await asyncio.sleep(seconds)
