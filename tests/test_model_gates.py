import json
from pathlib import Path
from types import SimpleNamespace

from limbus_librarian import cli
from limbus_librarian.eval import GoldItem, ResolvedGoldItem
from limbus_librarian.eval.model_gates import (
    LunaStructuredRewriter,
    StructuredRewrite,
    run_luna_experiment,
)
from limbus_librarian.index.searcher import HybridSearcher
from limbus_librarian.models import RetrievalConfig, RetrievalHit


def _hit(doc_id: str = "doc:1", chunk_id: str = "chunk:1") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="Dante and Vergilius work together at Limbus Company.",
        title="Dante",
        url="https://example.test",
        section_path="Dante",
        score=1.0,
        rank=1,
        retriever_name="test",
    )


class _Retriever:
    def retrieve(self, _query, _k, _filters):
        return [_hit()]


def test_cross_encoder_is_config_gated_and_lazy(monkeypatch):
    calls = {"init": 0, "rerank": 0}

    class FakeReranker:
        def __init__(self):
            calls["init"] += 1

        def rerank(self, _query, hits):
            calls["rerank"] += 1
            return hits

    monkeypatch.setattr("limbus_librarian.index.searcher.CrossEncoderReranker", FakeReranker)
    searcher = HybridSearcher(_Retriever(), None)
    base = RetrievalConfig(id="base", use_dense=False, use_bm25=True)

    searcher.search("Dante", base)
    assert calls == {"init": 0, "rerank": 0}

    explicitly_disabled = base.model_copy(
        update={"id": "disabled", "rerank_backend": "none", "use_rerank": True}
    )
    searcher.search("Dante", explicitly_disabled)
    assert calls == {"init": 0, "rerank": 0}

    cross = base.model_copy(
        update={"id": "cross", "rerank_backend": "cross_encoder", "use_rerank": True}
    )
    searcher.search("Dante", cross)
    searcher.search("Vergilius", cross)
    assert calls == {"init": 1, "rerank": 2}


def test_luna_rewriter_uses_bounded_structured_call():
    request = {}

    class Responses:
        def create(self, **kwargs):
            request.update(kwargs)
            return SimpleNamespace(
                output_text=StructuredRewrite(
                    rewritten_query="Dante Vergilius relationship",
                    relevant_chunk_ids=["chunk:1", "invented"],
                    rationale="Both entities are named.",
                ).model_dump_json()
            )

    rewriter = LunaStructuredRewriter(
        "test-key",
        "gpt-test",
        client=SimpleNamespace(responses=Responses()),
    )
    result = rewriter.rewrite("How are Dante and Vergilius connected?", [_hit()])

    assert result.relevant_chunk_ids == ["chunk:1"]
    assert request["max_output_tokens"] == 600
    assert request["text"]["format"]["type"] == "json_schema"


def test_luna_experiment_only_selects_relationship_or_empty_slice():
    relationship = ResolvedGoldItem(
        item=GoldItem(
            id="relationship",
            question="How are Dante and Vergilius connected?",
            question_type="relationship",
        ),
        relevant_doc_ids=["doc:1"],
    )
    ordinary = ResolvedGoldItem(
        item=GoldItem(id="ordinary", question="Who is Dante?", question_type="who"),
        relevant_doc_ids=["doc:1"],
    )

    class FakeRewriter:
        def rewrite(self, _question, _hits):
            return StructuredRewrite(
                rewritten_query="Dante Vergilius relationship",
                relevant_chunk_ids=["chunk:1"],
                rationale="The candidate directly describes the relationship.",
            )

    report = run_luna_experiment(
        [relationship, ordinary],
        lambda _query: [_hit()],
        FakeRewriter(),
    )

    assert report["status"] == "completed"
    assert [row["id"] for row in report["rows"]] == ["relationship"]
    assert report["luna_rewrite_and_grade"]["mrr"] == 1.0


def test_luna_cli_without_key_records_not_executed(tmp_path, monkeypatch, capsys):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "wiki_v1.jsonl").write_text(
        GoldItem(
            id="q1",
            question="How are A and B connected?",
            question_type="relationship",
            relevant_doc_titles=["Missing"],
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        documents_path=tmp_path / "documents.jsonl",
        configs_dir=Path(__file__).resolve().parents[1] / "configs",
        utility_model="gpt-5.6-luna",
        openai_api_key="",
        gold_path_for=lambda name: gold_dir / f"{name}.jsonl",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "searcher_from_disk",
        lambda _settings: (_ for _ in ()).throw(AssertionError("must not load indexes")),
    )

    cli.main(["eval", "--gold", "wiki_v1", "--luna-experiment"])

    report_path = tmp_path / "data" / "eval" / "runs" / "wiki_v1.luna_experiment.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "not_executed"
    assert "not executed" in capsys.readouterr().out
