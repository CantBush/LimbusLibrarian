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
    assert "FAKE" not in text
    assert citations[0].chunk_id == "abc123"
    assert citations[0].url.endswith("dongrang")
    assert ok
