import json
from pathlib import Path
from types import SimpleNamespace

from limbus_librarian import cli
from limbus_librarian.eval import GoldItem, evaluate_retrieval, ndcg_at_k, resolve_gold
from limbus_librarian.models import RetrievalHit, SourceDocument


def _document(doc_id: str, title: str) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        source_id="fixture",
        url="https://example.test/wiki/" + title.replace(" ", "_"),
        title=title,
        page_id=1,
        revision_id=1,
        document_type="world",
        retrieved_at="2026-01-01T00:00:00Z",
    )


def _hit(doc_id: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        text="text",
        title="Title",
        url="https://example.test",
        section_path="Title",
        score=1.0,
        rank=1,
        retriever_name="test",
    )


def test_resolve_gold_supports_ids_and_normalized_titles():
    docs = [_document("wiki:10", "The Head"), _document("wiki:11", "Smoke War")]
    gold = [
        GoldItem(
            id="q1",
            question="question",
            question_type="what",
            relevant_doc_ids=["wiki:10", "wiki:missing"],
            relevant_doc_titles=["Smoke_War", "Unknown Page"],
        )
    ]

    [resolved] = resolve_gold(gold, docs)

    assert resolved.relevant_doc_ids == ["wiki:10", "wiki:11"]
    assert resolved.resolved_titles == {"Smoke_War": "wiki:11"}
    assert resolved.unresolved_doc_ids == ["wiki:missing"]
    assert resolved.unresolved_titles == ["Unknown Page"]
    assert resolved.status == "partial"


def test_unresolved_items_are_reported_without_retrieval():
    item = GoldItem(
        id="q1",
        question="Who?",
        question_type="who",
        relevant_doc_titles=["Not In Corpus"],
    )
    resolved = resolve_gold([item], [])

    def unexpected_retrieval(_query: str):
        raise AssertionError("retrieval must not run for unresolved labels")

    summary = evaluate_retrieval(resolved, unexpected_retrieval)

    assert summary["n"] == 1
    assert summary["n_evaluated"] == 0
    assert summary["n_unresolved"] == 1
    assert summary["rows"][0]["unresolved_titles"] == ["Not In Corpus"]


def test_metrics_use_resolved_title_doc_id():
    item = GoldItem(
        id="q1",
        question="What is the Head?",
        question_type="what",
        relevant_doc_titles=["The Head"],
    )
    resolved = resolve_gold([item], [_document("wiki:42", "The Head")])

    summary = evaluate_retrieval(resolved, lambda _query: [_hit("wiki:42")], k=8)

    assert summary["n_evaluated"] == 1
    assert summary["recall@k"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["ndcg@k"] == 1.0
    assert ndcg_at_k({"a", "b"}, ["a", "x"], 2) < 1.0
    assert summary["by_question_type"]["what"]["n_evaluated"] == 1
    assert summary["by_question_type"]["what"]["ndcg@k"] == 1.0


def test_wiki_gold_has_curated_phase_three_coverage():
    path = Path(__file__).resolve().parents[1] / "data" / "eval" / "gold" / "wiki_v1.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 40
    assert {record["question_type"] for record in records} >= {
        "who",
        "what",
        "relationship",
        "event",
        "where_established",
    }
    text = path.read_text(encoding="utf-8")
    assert "Canto V" in text
    assert "Smoke War" in text
    assert "The Head" in text
    assert all(record.get("relevant_doc_titles") for record in records)


def test_comparison_cli_skips_index_loading_when_all_labels_unresolved(
    tmp_path, monkeypatch, capsys
):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "wiki_v1.jsonl").write_text(
        GoldItem(
            id="q1",
            question="Who?",
            question_type="who",
            relevant_doc_titles=["Future Wiki Page"],
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        documents_path=tmp_path / "missing-documents.jsonl",
        configs_dir=Path(__file__).resolve().parents[1] / "configs",
        gold_path_for=lambda name: gold_dir / f"{name}.jsonl",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "searcher_from_disk",
        lambda _settings: (_ for _ in ()).throw(AssertionError("indexes should not load")),
    )

    cli.main(["eval", "--gold", "wiki_v1", "--compare"])

    output = capsys.readouterr().out
    assert all(config_id in output for config_id in cli.COMPARISON_CONFIGS)
    assert "who" in output
    assert "no wiki IDs were fabricated" in output
    report = json.loads(
        (tmp_path / "data" / "eval" / "runs" / "wiki_v1.comparison.json").read_text()
    )
    assert [row["config_id"] for row in report["configs"]] == list(cli.COMPARISON_CONFIGS)
    assert all(row["n_unresolved"] == 1 for row in report["configs"])
