from pathlib import Path

from fastapi.testclient import TestClient

from limbus_librarian.api.app import create_app
from limbus_librarian.config import Settings


def test_api_ask_health_configs(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        data_dir=tmp_path,
        cors_origins="http://testserver",
        openai_api_key="",
    )
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
        assert health.json()["llm_configured"] is False
        ui = client.get("/")
        assert ui.status_code == 200
        assert "Limbus Librarian" in ui.text
        assert "Unofficial fan project" in ui.text
        assert "official logo" not in ui.text.lower()
        configs = client.get("/v1/configs")
        assert "hybrid" in configs.json()["configs"]
        assert "hybrid_graph" in configs.json()["configs"]
        reloaded = client.post("/v1/reload")
        assert reloaded.status_code == 200
        assert reloaded.json()["chunk_count"] > 0

        catalog = client.get("/v1/documents", params={"type": "character,sinner"})
        assert catalog.status_code == 200
        assert {item["title"] for item in catalog.json()["items"]} == {
            "Dongrang",
            "Yi Sang",
        }
        assert catalog.json()["total"] == 2
        searched = client.get("/v1/documents", params={"q": "mirror"})
        assert searched.status_code == 200
        assert {item["title"] for item in searched.json()["items"]} >= {
            "Dongrang",
            "The Mirror",
        }
        assert client.get(
            "/v1/documents", params={"canto": "Canto V"}
        ).json()["total"] == 0
        document = client.get(
            f"/v1/documents/{catalog.json()['items'][0]['doc_id']}"
        )
        assert document.status_code == 200
        assert document.json()["summary"]
        assert document.json()["sections"]
        assert document.json()["url"].startswith("https://")
        assert document.json()["related"]
        related = client.get(
            f"/v1/documents/{catalog.json()['items'][0]['doc_id']}/related"
        )
        assert related.status_code == 200
        assert related.json()["items"]

        resp = client.post(
            "/v1/ask",
            json={
                "query": "Who is Dongrang?",
                "config_id": "bm25_only",
                "debug": True,
                "document_types": ["character"],
                "cantos": ["Canto IV"],
                "max_canto": 4,
                "history": ["What happened in Canto IV?"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["citations"]
        assert "[cite:" not in body["answer"]
        assert "[1]" in body["answer"]
        assert body["trace"]["config_id"] == "bm25_only"
        retrieve = next(
            step for step in body["trace"]["steps"] if step["name"] == "retrieve"
        )
        assert retrieve["detail"]["filters"]["document_types"] == ["character"]
        assert retrieve["detail"]["filters"]["cantos"] == ["Canto IV"]
        assert all(
            hit["metadata"]["document_type"] == "character"
            for hit in body["trace"]["hits"]
        )
        chunk_id = body["citations"][0]["chunk_id"]
        src = client.get(f"/v1/sources/{chunk_id}")
        assert src.status_code == 200
        assert src.json()["chunk_id"] == chunk_id

        app_js = client.get("/static/app.js")
        styles_css = client.get("/static/styles.css")
        assert "inline-citation" in app_js.text
        assert "source-${number}" in app_js.text
        assert "localStorage" in app_js.text
        assert "history: recentHistory" in app_js.text
        assert "data-related-doc" in app_js.text
        assert '<svg viewBox="0 0 48 56"' in ui.text
        assert 'id="ask-status"' in ui.text
        assert "prefers-reduced-motion" in styles_css.text
        assert "skeleton-shimmer" in styles_css.text
        assert '["Searching…", "Reading sources…", "Writing…"]' in app_js.text
        assert "query.value = \"\";" in app_js.text
        assert "restoreSession();" in app_js.text
        assert 'setAttribute("aria-busy", "true")' in app_js.text
        assert "data-stub" not in ui.text

        too_much_history = client.post(
            "/v1/ask",
            json={"query": "Who is Yi Sang?", "history": ["question"] * 5},
        )
        assert too_much_history.status_code == 422

        spoiler_limited = client.post(
            "/v1/ask",
            json={
                "query": "Who is Dongrang?",
                "config_id": "bm25_only",
                "max_canto": 3,
                "debug": True,
            },
        )
        assert spoiler_limited.status_code == 200
        assert spoiler_limited.json()["trace"]["hits"] == []
