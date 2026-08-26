from limbus_librarian.generate.citations import validate_citations
from limbus_librarian.models import RetrievalHit


def test_drops_unknown_citation_ids():
    hit = RetrievalHit(
        chunk_id="abc123",
        doc_id="d",
        text="Dongrang is a researcher.",
        title="Dongrang",
        url="https://example.test/dongrang",
        section_path="Dongrang",
        score=1.0,
        rank=1,
        retriever_name="bm25",
    )
    text, citations, ok = validate_citations("He is a researcher [cite:abc123] [cite:FAKE]", [hit])
    assert text == "He is a researcher [1]"
    assert citations[0].chunk_id == "abc123"
    assert citations[0].url.endswith("dongrang")
    assert not ok


def test_rewrites_repeated_and_adjacent_citations_in_metadata_order():
    def hit(chunk_id: str, title: str) -> RetrievalHit:
        return RetrievalHit(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            text=f"{title} text",
            title=title,
            url=f"https://example.test/{title.lower()}",
            section_path=title,
            score=1.0,
            rank=1,
            retriever_name="bm25",
        )

    text, citations, ok = validate_citations(
        "Claim [cite:bbb222] [cite:aaa111]. Again [cite:bbb222].",
        [hit("aaa111", "Alpha"), hit("bbb222", "Beta")],
    )

    assert text == "Claim [1][2]. Again [1]."
    assert [citation.chunk_id for citation in citations] == ["bbb222", "aaa111"]
    assert ok
