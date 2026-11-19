"""Chat endpoint contract tests."""

import json
import uuid

from fastapi.testclient import TestClient


class TestChatJson:
    def test_chat_returns_full_contract(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat", json={"message": "What is the test project?"})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "session_id",
            "answer",
            "sources",
            "model",
            "grounded",
            "latency_ms",
        }
        assert body["answer"] == "A grounded answer [1]."
        assert body["grounded"] is True
        assert uuid.UUID(body["session_id"])  # server-generated
        assert body["sources"][0]["index"] == 1

    def test_chat_respects_a_provided_session_id(self, client: TestClient) -> None:
        session_id = str(uuid.uuid4())
        response = client.post("/api/v1/chat", json={"session_id": session_id, "message": "hello"})
        assert response.json()["session_id"] == session_id

    def test_empty_message_gives_400_error_body(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat", json={"message": "   "})
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "guardrail_violation"
        assert "message" in error
        assert "request_id" in error

    def test_missing_message_gives_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422


class TestChatStream:
    def test_event_order(self, client: TestClient) -> None:
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "What is the test project?"}
        ) as response:
            assert response.status_code == 200
            events: list[tuple[str, dict]] = []
            event_name = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((event_name, json.loads(line.split(":", 1)[1])))
        names = [name for name, _ in events]
        assert names[0] == "meta"
        assert "token" in names
        assert names[-2:] == ["sources", "done"]
        meta = events[0][1]
        assert uuid.UUID(meta["session_id"])
        done = events[-1][1]
        assert done["grounded"] is True

    def test_sources_event_contains_structured_sources(self, client: TestClient) -> None:
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "What is the test project?"}
        ) as response:
            sources_payload: dict | None = None
            event_name = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event_name == "sources":
                    sources_payload = json.loads(line.split(":", 1)[1])
        assert sources_payload is not None
        source = sources_payload["sources"][0]
        assert source["index"] == 1
        assert source["url"].startswith("https://")
        assert source["title"]

    def test_tokens_concatenate_to_the_answer(self, client: TestClient) -> None:
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "What is the test project?"}
        ) as response:
            tokens: list[str] = []
            event_name = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event_name == "token":
                    tokens.append(json.loads(line.split(":", 1)[1])["delta"])
        assert "".join(tokens) == "A grounded answer [1]."


class TestSessions:
    def test_delete_session_returns_204(self, client: TestClient) -> None:
        response = client.delete(f"/api/v1/sessions/{uuid.uuid4()}")
        assert response.status_code == 204


class TestMeta:
    def test_stats(self, client: TestClient) -> None:
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        body = response.json()
        assert body["n_documents"] == 42
        assert body["by_site"] == {"darglobal": 20, "wasalt": 22}
        assert body["llm_models"][0] == "primary/model:free"

    def test_suggestions(self, client: TestClient) -> None:
        response = client.get("/api/v1/suggestions")
        assert response.status_code == 200
        assert len(response.json()["suggestions"]) == 6
