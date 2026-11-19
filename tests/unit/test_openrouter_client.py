"""OpenRouter client tests with respx-mocked HTTP."""

import json

import httpx
import pytest
import respx

from property_assistant.core.config import Settings
from property_assistant.core.exceptions import LLMError, LLMRateLimitError
from property_assistant.domain.enums import Role
from property_assistant.llm.base import LLMMessage
from property_assistant.llm.openrouter import OpenRouterClient, _ThinkStripper

BASE = "https://openrouter.ai/api/v1"
MESSAGES = [LLMMessage(role=Role.USER, content="hello")]


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "openrouter_api_key": "sk-or-test",
        "llm_primary_model": "primary/model:free",
        "llm_fallback_models": ["backup/model:free", "openrouter/free"],
        "llm_max_retries": 0,
        "app_public_url": "https://app.example.com",
        "_env_file": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def completion_body(text: str = "Hi!", model: str = "primary/model:free") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


@pytest.fixture
def settings() -> Settings:
    return make_settings()


class TestComplete:
    @respx.mock
    async def test_success_includes_required_headers(self, settings: Settings) -> None:
        route = respx.post(f"{BASE}/chat/completions").respond(json=completion_body())
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            result = await client.complete(MESSAGES)
        assert result.text == "Hi!"
        assert result.model == "primary/model:free"
        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer sk-or-test"
        assert request.headers["http-referer"] == "https://app.example.com"
        assert "x-title" in request.headers
        payload = json.loads(request.content)
        assert payload["model"] == "primary/model:free"
        assert payload["models"] == [
            "primary/model:free",
            "backup/model:free",
            "openrouter/free",
        ]

    @respx.mock
    async def test_fallback_to_next_model_on_provider_error(self, settings: Settings) -> None:
        route = respx.post(f"{BASE}/chat/completions")
        route.side_effect = [
            httpx.Response(200, json={"error": {"message": "model down"}}),
            httpx.Response(200, json=completion_body("From backup", "backup/model:free")),
        ]
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            result = await client.complete(MESSAGES)
        assert result.text == "From backup"
        assert result.model == "backup/model:free"

    @respx.mock
    async def test_all_models_failing_raises_unavailable(self, settings: Settings) -> None:
        respx.post(f"{BASE}/chat/completions").respond(json={"error": {"message": "nope"}})
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            with pytest.raises(LLMError):
                await client.complete(MESSAGES)

    @respx.mock
    async def test_429_with_long_retry_after_raises_rate_limit(self, settings: Settings) -> None:
        respx.post(f"{BASE}/chat/completions").respond(
            status_code=429, headers={"Retry-After": "120"}
        )
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            with pytest.raises(LLMRateLimitError):
                await client.complete(MESSAGES)

    @respx.mock
    async def test_429_with_short_retry_after_waits_and_retries(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("property_assistant.llm.openrouter._sleep", fake_sleep)
        route = respx.post(f"{BASE}/chat/completions")
        route.side_effect = [
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json=completion_body()),
        ]
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, make_settings(llm_max_retries=1))
            result = await client.complete(MESSAGES)
        assert result.text == "Hi!"
        assert sleeps == [1.0]

    @respx.mock
    async def test_think_blocks_stripped(self, settings: Settings) -> None:
        respx.post(f"{BASE}/chat/completions").respond(
            json=completion_body("<think>reasoning...</think>The answer.")
        )
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            result = await client.complete(MESSAGES)
        assert result.text == "The answer."


def sse_bytes(*events: str) -> bytes:
    return ("\n".join(f"data: {event}" for event in events) + "\n").encode()


def stream_event(delta: str, model: str = "primary/model:free") -> str:
    return json.dumps({"model": model, "choices": [{"delta": {"content": delta}}]})


class TestStream:
    @respx.mock
    async def test_stream_parses_sse_and_stops_at_done(self, settings: Settings) -> None:
        body = (
            b": OPENROUTER PROCESSING\n"
            + sse_bytes(stream_event("Hel"), stream_event("lo!"))
            + b"data: [DONE]\n"
            + sse_bytes(stream_event("IGNORED"))
        )
        respx.post(f"{BASE}/chat/completions").respond(
            stream=body, headers={"content-type": "text/event-stream"}
        )
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            deltas = [chunk.delta async for chunk in client.stream(MESSAGES)]
        assert "".join(deltas) == "Hello!"

    @respx.mock
    async def test_stream_falls_back_before_first_token(self, settings: Settings) -> None:
        route = respx.post(f"{BASE}/chat/completions")
        route.side_effect = [
            httpx.Response(500),
            httpx.Response(
                200,
                stream=sse_bytes(stream_event("ok", "backup/model:free")) + b"data: [DONE]\n",
                headers={"content-type": "text/event-stream"},
            ),
        ]
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            chunks = [chunk async for chunk in client.stream(MESSAGES)]
        assert chunks[0].delta == "ok"

    @respx.mock
    async def test_stream_think_blocks_stripped(self, settings: Settings) -> None:
        body = (
            sse_bytes(
                stream_event("<thi"),
                stream_event("nk>hidden</th"),
                stream_event("ink>visible"),
            )
            + b"data: [DONE]\n"
        )
        respx.post(f"{BASE}/chat/completions").respond(
            stream=body, headers={"content-type": "text/event-stream"}
        )
        async with httpx.AsyncClient() as http:
            client = OpenRouterClient(http, settings)
            deltas = [chunk.delta async for chunk in client.stream(MESSAGES)]
        assert "".join(deltas) == "visible"


class TestThinkStripper:
    def test_passthrough_without_tags(self) -> None:
        stripper = _ThinkStripper()
        assert stripper.feed("plain text") == "plain text"

    def test_strips_complete_block(self) -> None:
        stripper = _ThinkStripper()
        assert stripper.feed("<think>hidden</think>shown") == "shown"

    def test_strips_block_split_across_deltas(self) -> None:
        stripper = _ThinkStripper()
        output = stripper.feed("<th") + stripper.feed("ink>secret</think>ok")
        assert output == "ok"
