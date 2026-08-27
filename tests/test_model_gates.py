import json
from pathlib import Path
from types import SimpleNamespace

from limbus_librarian import cli
from limbus_librarian.eval import GoldItem, ResolvedGoldItem
from limbus_librarian.eval.model_gates import (
    GENERATION_EVAL_LIMIT_CAP,
    GenerationJudgment,
    JudgedClaim,
    LunaGenerationJudge,
    LunaStructuredRewriter,
    StructuredRewrite,
    lexical_coverage,
    run_generation_eval,
    run_luna_experiment,
)
from limbus_librarian.index.searcher import HybridSearcher
from limbus_librarian.models import RetrievalConfig, RetrievalHit


def _hit(
    doc_id: str = "doc:1",
    chunk_id: str = "chunk:1",
    text: str = "Dante and Vergilius work together at Limbus Company.",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
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


def test_lexical_coverage_is_casefold_substring():
    result = lexical_coverage(
        "Vergilius guides Dante and the Sinners. They both work for Limbus Company.",
        ["Vergilius guides Dante and the Sinners", "both work for Limbus Company"],
    )
    assert result["coverage"] == 1.0
    assert result["missed"] == []


def test_generation_judge_drops_invented_chunk_ids():
    request = {}

    class Responses:
        def create(self, **kwargs):
            request.update(kwargs)
            return SimpleNamespace(
                output_text=GenerationJudgment(
                    claims=[
                        JudgedClaim(
                            text="Vergilius guides Dante.",
                            supported=True,
                            cited_chunk_ids=["chunk:1", "invented"],
                            supporting_chunk_ids=["chunk:1", "nope"],
                        )
                    ]
                ).model_dump_json()
            )

    judge = LunaGenerationJudge(
        "test-key",
        "gpt-test",
        client=SimpleNamespace(responses=Responses()),
    )
    result = judge.judge(
        "How are Dante and Vergilius connected?",
        "Vergilius guides Dante [cite:chunk:1].",
        [_hit()],
    )

    claim = result.claims[0]
    assert claim.cited_chunk_ids == ["chunk:1"]
    assert claim.supporting_chunk_ids == ["chunk:1"]
    assert claim.supported is True
    assert request["max_output_tokens"] == 1600
    assert "invent" in request["input"][0]["content"].lower()


def test_generation_eval_skips_unresolved_and_scores_extractive():
    unresolved = ResolvedGoldItem(
        item=GoldItem(
            id="missing",
            question="What is missing?",
            question_type="what",
            expected_answer_points=["never retrieved"],
        ),
        relevant_doc_ids=[],
        unresolved_titles=["Missing"],
    )
    resolved = ResolvedGoldItem(
        item=GoldItem(
            id="q1",
            question="How are Dante and Vergilius connected?",
            question_type="relationship",
            expected_answer_points=["work together", "missing point"],
        ),
        relevant_doc_ids=["doc:1"],
    )

    def unexpected_model(_query, _hits):
        raise AssertionError("model arm must not run without a judge")

    report = run_generation_eval(
        [unresolved, resolved],
        lambda _query: [_hit()],
        limit=GENERATION_EVAL_LIMIT_CAP,
        model_fn=unexpected_model,
        judge=None,
    )

    assert report["n"] == 1
    assert report["n_skipped_unresolved"] == 1
    assert [row["id"] for row in report["rows"]] == ["q1"]
    assert report["extractive"]["coverage"] == 0.5
    assert report["model"]["status"] == "not_executed"
    assert report["rows"][0]["extractive"]["matched"] == ["work together"]


def test_generation_eval_model_arm_uses_judge():
    item = ResolvedGoldItem(
        item=GoldItem(
            id="q1",
            question="How are Dante and Vergilius connected?",
            question_type="relationship",
            expected_answer_points=["work together"],
        ),
        relevant_doc_ids=["doc:1"],
    )

    class FakeJudge:
        def judge(self, _question, _answer, hits):
            return GenerationJudgment(
                claims=[
                    JudgedClaim(
                        text="They work together.",
                        supported=True,
                        cited_chunk_ids=[hits[0].chunk_id],
                        supporting_chunk_ids=[hits[0].chunk_id],
                    )
                ]
            )

    report = run_generation_eval(
        [item],
        lambda _query: [_hit()],
        model_fn=lambda _query, _hits: "Dante and Vergilius work together [cite:chunk:1].",
        judge=FakeJudge(),
    )

    assert report["model"]["status"] == "completed"
    assert report["model"]["coverage"] == 1.0
    assert report["model"]["faithfulness"] == 1.0
    assert report["model"]["citation_claim_overlap"] == 1.0


def test_generation_eval_prints_progress_and_keeps_partial_on_interrupt():
    first = ResolvedGoldItem(
        item=GoldItem(
            id="q1",
            question="Who is Dante?",
            question_type="who",
            expected_answer_points=["work together"],
        ),
        relevant_doc_ids=["doc:1"],
    )
    second = ResolvedGoldItem(
        item=GoldItem(
            id="q2",
            question="Who is Vergilius?",
            question_type="who",
            expected_answer_points=["guide"],
        ),
        relevant_doc_ids=["doc:1"],
    )
    messages: list[str] = []

    class InterruptJudge:
        def judge(self, _question, _answer, _hits):
            raise KeyboardInterrupt

    report = run_generation_eval(
        [first, second],
        lambda _query: [_hit()],
        model_fn=lambda _query, _hits: "Dante and Vergilius work together.",
        judge=InterruptJudge(),
        progress=messages.append,
    )

    assert report["status"] == "interrupted"
    assert report["n"] == 1
    assert report["rows"][0]["id"] == "q1"
    assert "extractive" in report["rows"][0]
    assert report["model"]["status"] == "interrupted"
    assert any("generating" in message for message in messages)
    assert any("judging" in message for message in messages)
    assert any("Interrupted" in message for message in messages)


def test_generation_eval_cli_without_key_writes_extractive(
    tmp_path, monkeypatch, capsys
):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "wiki_v1.jsonl").write_text(
        GoldItem(
            id="q1",
            question="Who is Dante?",
            question_type="who",
            relevant_doc_titles=["Dante"],
            expected_answer_points=["clock-headed"],
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        json.dumps(
            {
                "doc_id": "wiki:1",
                "source_id": "fixture",
                "url": "https://example.test/Dante",
                "title": "Dante",
                "page_id": 1,
                "revision_id": 1,
                "document_type": "character",
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        documents_path=documents,
        configs_dir=Path(__file__).resolve().parents[1] / "configs",
        utility_model="gpt-5.6-luna",
        generate_model="gpt-5.6-terra",
        openai_api_key="",
        gold_path_for=lambda name: gold_dir / f"{name}.jsonl",
    )

    class FakeSearcher:
        def search(self, _query, _config):
            return [_hit(text="Dante is a clock-headed manager.")]

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "searcher_from_disk", lambda _settings: FakeSearcher())

    cli.main(
        ["eval", "--gold", "wiki_v1", "--generation-eval", "--experiment-limit", "20"]
    )

    report_path = tmp_path / "data" / "eval" / "runs" / "wiki_v1.generation_eval.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert report["baseline_config"] == "vector_only"
    assert report["extractive"]["coverage"] == 1.0
    assert report["model"]["status"] == "not_executed"
    assert "model arm not executed" in output


def test_generation_eval_cli_skips_indexes_when_unresolved(
    tmp_path, monkeypatch, capsys
):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "wiki_v1.jsonl").write_text(
        GoldItem(
            id="q1",
            question="Who?",
            question_type="who",
            relevant_doc_titles=["Missing"],
            expected_answer_points=["never"],
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        documents_path=tmp_path / "missing-documents.jsonl",
        configs_dir=Path(__file__).resolve().parents[1] / "configs",
        utility_model="gpt-5.6-luna",
        generate_model="gpt-5.6-terra",
        openai_api_key="",
        gold_path_for=lambda name: gold_dir / f"{name}.jsonl",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "searcher_from_disk",
        lambda _settings: (_ for _ in ()).throw(AssertionError("must not load indexes")),
    )

    cli.main(["eval", "--gold", "wiki_v1", "--generation-eval"])

    report = json.loads(
        (tmp_path / "data" / "eval" / "runs" / "wiki_v1.generation_eval.json").read_text()
    )
    assert report["n"] == 0
    assert report["n_skipped_unresolved"] == 1
    assert report["model"]["status"] == "not_executed"
    assert "no wiki IDs were fabricated" in capsys.readouterr().out
