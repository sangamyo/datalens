from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok_without_database() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_import_rejects_invalid_repo_id_without_database() -> None:
    # Dataset endpoints are implemented now (they used to return 501); validation needs no DB.
    assert client.post("/datasets/import", json={"hf_repo_id": "not a repo"}).status_code == 422
