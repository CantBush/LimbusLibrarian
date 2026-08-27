import sqlite3
from pathlib import Path

from limbus_librarian.config import Settings
from limbus_librarian.config_loader import load_named_config
from limbus_librarian.eval import evaluate_retrieval, load_gold
from limbus_librarian.graph import build_ask_graph, run_ask
from limbus_librarian.graph.heuristics import analyze_query
from limbus_librarian.graph.store import GraphStore
from limbus_librarian.ingest.pipeline import load_documents
from limbus_librarian.models import SourceDocument
from limbus_librarian.runtime import bootstrap_from_fixtures


def _graph_document(
    doc_id: str,
    title: str,
    *,
    page_id: int,
    entities: list[str] | None = None,
    infobox: dict[str, str] | None = None,
) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        source_id="fixture",
        url="https://example.test/wiki/" + title.replace(" ", "_"),
        title=title,
        page_id=page_id,
        revision_id=1,
        document_type="character",
        retrieved_at="2026-01-01T00:00:00Z",
        entities=entities or [],
        infobox=infobox or {},
    )


def test_search_and_eval_and_graph(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(data_dir=tmp_path, openai_api_key="")
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
    monkeypatch.setattr(
        type(settings),
        "gold_path",
        property(lambda self: root / "data" / "eval" / "gold" / "v1.jsonl"),
    )
    searcher = bootstrap_from_fixtures(settings)
    cfg = load_named_config(root / "configs", "hybrid")
    hits = searcher.search("Who is Dongrang?", cfg)
    assert hits
    assert any(h.title == "Dongrang" for h in hits)

    gold = load_gold(root / "data" / "eval" / "gold" / "v1.jsonl")
    summary = evaluate_retrieval(gold, lambda q: searcher.search(q, cfg), k=8)
    assert summary["recall@k"] > 0.5
    assert summary["mrr"] > 0.4

    answer = run_ask("Who is Dongrang?", searcher, cfg, api_key="", debug=True)
    assert "Dongrang" in answer.answer
    assert answer.citations
    assert answer.trace is not None
    assert answer.trace.hops <= 1
    assert answer.trace.analysis is not None
    assert answer.trace.analysis.entities == ["Dongrang"]
    assert "character" in answer.trace.analysis.document_types
    step_names = [s.name for s in answer.trace.steps]
    assert "retrieve" in step_names

    league_answer = run_ask(
        "What was the League of Nine?",
        searcher,
        cfg,
        api_key="",
        debug=True,
    )
    assert league_answer.trace is not None
    assert league_answer.trace.analysis is not None
    assert league_answer.trace.analysis.entities == ["League of Nine"]
    assert "faction" in league_answer.trace.analysis.document_types

    compiled = build_ask_graph(searcher)
    result = compiled.invoke(
        {
            "query": "Who is Dongrang?",
            "config": cfg,
            "hops": 0,
            "gen_retries": 0,
            "api_key": "",
            "generate_model": "gpt-5.6-terra",
        }
    )
    assert result.get("hops", 0) <= 1


def test_sqlite_graph_related_pages_and_third_rrf_list(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(data_dir=tmp_path, embedding_dims=32, openai_api_key="")
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
    searcher = bootstrap_from_fixtures(settings)
    documents = load_documents(settings.documents_path)
    yi_sang = next(document for document in documents if document.title == "Yi Sang")
    dongrang = next(document for document in documents if document.title == "Dongrang")
    store = GraphStore(settings.catalog_path)
    assert store.match_entities("How did the League of Nine know Dongrang?") == [
        ("League of Nine", "faction"),
        ("Dongrang", "character"),
    ]
    assert store.match_entities("Explain The Mirror.") == [("The Mirror", "world")]
    unmatched = analyze_query(
        "Explain Professor Zorblax.",
        store.match_entities("Explain Professor Zorblax."),
    )
    assert unmatched.entities == []
    assert unmatched.document_types == []

    related = store.related(yi_sang.doc_id)
    assert {item["title"] for item in related} >= {"Dongrang", "League of Nine", "The Mirror"}

    with sqlite3.connect(settings.catalog_path) as connection:
        before = connection.execute(
            "SELECT src, rel, dst, doc_id FROM edges ORDER BY src, rel, dst, doc_id"
        ).fetchall()
    store.rebuild(list(reversed(documents)))
    with sqlite3.connect(settings.catalog_path) as connection:
        after = connection.execute(
            "SELECT src, rel, dst, doc_id FROM edges ORDER BY src, rel, dst, doc_id"
        ).fetchall()
    assert after == before
    assert ("Dongrang", "affiliated_with", "K Corp.", dongrang.doc_id) in after

    config = load_named_config(settings.configs_dir, "hybrid_graph")
    hits = searcher.search("How are Yi Sang and Dongrang related?", config)
    assert hits
    assert any("graph" in hit.score_components for hit in hits)
    assert any(hit.title in {"Dongrang", "League of Nine", "Yi Sang"} for hit in hits)


def test_rebuild_deduplicates_case_variant_edges(tmp_path: Path):
    store = GraphStore(tmp_path / "graph.sqlite")
    documents = [
        _graph_document(
            "doc:yi-sang",
            "Yi Sang",
            page_id=1,
            entities=["League of Nine", "league of nine", "Dongrang"],
            infobox={"affiliation": "K Corp.", "employer": "k corp."},
        ),
        _graph_document(
            "doc:dongrang",
            "Dongrang",
            page_id=2,
            entities=["Yi Sang", "yi sang"],
        ),
    ]

    store.rebuild(documents)

    with sqlite3.connect(store.path) as connection:
        edges = connection.execute(
            "SELECT src, rel, dst, doc_id FROM edges ORDER BY src, rel, dst, doc_id"
        ).fetchall()
        entity_titles = [
            row[0]
            for row in connection.execute(
                "SELECT title FROM entities ORDER BY title COLLATE NOCASE"
            ).fetchall()
        ]

    identities = {(src.casefold(), rel, dst.casefold(), doc_id) for src, rel, dst, doc_id in edges}
    assert len(identities) == len(edges) == 4
    assert ("Yi Sang", "links_to", "League of Nine", "doc:yi-sang") in edges
    assert ("Yi Sang", "affiliated_with", "K Corp.", "doc:yi-sang") in edges
    assert ("Yi Sang", "links_to", "Dongrang", "doc:yi-sang") in edges
    assert ("Dongrang", "links_to", "Yi Sang", "doc:dongrang") in edges
    assert "League of Nine" in entity_titles
    assert "K Corp." in entity_titles
    store.rebuild(list(reversed(documents)))
    with sqlite3.connect(store.path) as connection:
        again = connection.execute(
            "SELECT src, rel, dst, doc_id FROM edges ORDER BY src, rel, dst, doc_id"
        ).fetchall()
    assert again == edges

