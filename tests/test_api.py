from pathlib import Path

from fastapi.testclient import TestClient

from askrepo.api import build_app


def test_health_endpoint(tmp_path: Path) -> None:
    client = TestClient(build_app(tmp_path))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_without_index_returns_409(tmp_path: Path) -> None:
    client = TestClient(build_app(tmp_path))
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 409
