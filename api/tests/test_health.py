from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok_without_database() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dataset_endpoints_are_stubbed() -> None:
    assert client.get("/datasets").status_code == 501
    assert client.post("/datasets/import", json={"hf_repo_id": "lerobot/pusht"}).status_code == 501
