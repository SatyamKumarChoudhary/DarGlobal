"""Health and readiness endpoint tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestHealth:
    def test_health_always_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_200_when_container_present(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_ready_503_before_startup(self, app: FastAPI) -> None:
        del app.state.container
        bare_client = TestClient(app)  # no lifespan → no container
        response = bare_client.get("/ready")
        assert response.status_code == 503
