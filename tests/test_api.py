from pathlib import Path

from fastapi.testclient import TestClient

from limbus_librarian.api.app import create_app
from limbus_librarian.config import Settings


def test_api_ask_health_configs(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(data_dir=tmp_path, cors_origins="http://testserver")
    monkeypatch.setattr(
        type(settings),
        "fixtures_dir",
        property(lambda self: root / "data" / "fixtures"),
    )
    monkeypatch.setattr(
        type(settings),
        "configs_dir",
        property(lambda self: root / "configs"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        assert "not affiliated" in health.json()["disclaimer"].lower()
        configs = client.get("/v1/configs")
        assert "hybrid" in configs.json()["configs"]
        resp = client.post(
            "/v1/ask",
            json={"query": "Who is Dongrang?", "config_id": "bm25_only", "debug": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["citations"]
        assert body["trace"]["config_id"] == "bm25_only"
        chunk_id = body["citations"][0]["chunk_id"]
        src = client.get(f"/v1/sources/{chunk_id}")
        assert src.status_code == 200
        assert src.json()["chunk_id"] == chunk_id
