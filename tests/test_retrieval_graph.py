from pathlib import Path

from limbus_librarian.config import Settings
from limbus_librarian.config_loader import load_named_config
from limbus_librarian.eval import evaluate_retrieval, load_gold
from limbus_librarian.graph import build_ask_graph, run_ask
from limbus_librarian.runtime import bootstrap_from_fixtures


def test_search_and_eval_and_graph(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(data_dir=tmp_path)
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
    step_names = [s.name for s in answer.trace.steps]
    assert "retrieve" in step_names

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
