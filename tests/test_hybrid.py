from limbus_librarian.index.hybrid import reciprocal_rank_fusion
from limbus_librarian.models import RetrievalHit


def _hit(chunk_id: str, rank: int, name: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id="d",
        text="t",
        title="T",
        url="http://example.test",
        section_path="T",
        score=1.0,
        rank=rank,
        retriever_name=name,
    )


def test_rrf_prefers_agreement():
    fused = reciprocal_rank_fusion(
        [
            [_hit("a", 1, "bm25"), _hit("b", 2, "bm25")],
            [_hit("b", 1, "dense"), _hit("a", 2, "dense")],
        ],
        k=60,
        limit=10,
    )
    assert fused[0].chunk_id in {"a", "b"}
    assert fused[0].retriever_name == "hybrid_rrf"
    assert "rrf_bm25" in fused[0].score_components or "rrf_dense" in fused[0].score_components
