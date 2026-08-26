from pathlib import Path

from limbus_librarian.chunking import chunk_documents
from limbus_librarian.ingest.classify import classify_document, is_lore_first
from limbus_librarian.ingest.pipeline import ingest_connector
from limbus_librarian.sources.fixture import FixtureSourceConnector


def test_identity_pages_are_not_lore_first():
    assert classify_document("Seven Assoc. South Section 6 Yi Sang", ["Identities"]) == "identity"
    assert not is_lore_first("identity")


def test_fixture_ingest_skips_identity(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "data" / "fixtures"
    docs = ingest_connector(
        FixtureSourceConnector(fixtures),
        tmp_path / "documents.jsonl",
        tmp_path / "catalog.sqlite",
        lore_first=True,
    )
    titles = {d.title for d in docs}
    assert "Dongrang" in titles
    assert "Yi Sang" in titles
    assert "Canto IV" in titles
    assert "League of Nine" in titles
    assert "The Mirror" in titles
    assert "Seven Assoc. South Section 6 Yi Sang" not in titles
    assert all(d.license for d in docs)
    chunks = chunk_documents(docs)
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    again = chunk_documents(docs)
    assert [c.chunk_id for c in again] == ids
