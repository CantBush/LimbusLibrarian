import json
from pathlib import Path

import pytest

from limbus_librarian.chunking import chunk_document, chunk_documents
from limbus_librarian.ingest.classify import (
    classify_document,
    detect_cantos,
    is_lore_first,
    is_media_subpage,
)
from limbus_librarian.ingest.pipeline import ingest_connector, ingest_incremental
from limbus_librarian.models import SourceDocument
from limbus_librarian.sources.fixture import FixtureSourceConnector


def test_identity_pages_are_not_lore_first():
    assert classify_document("Seven Assoc. South Section 6 Yi Sang", ["Identities"]) == "identity"
    assert classify_document("Yi Sang/Identity", ["Identities"]) == "identity"
    assert not is_lore_first("identity")


def test_overview_identity_and_ego_titles_are_world():
    assert classify_document("Identity", ["Identities"]) == "world"
    assert classify_document("E.G.O.", ["E.G.O"]) == "world"
    assert classify_document("E.G.O", []) == "world"
    assert classify_document("EGO", []) == "world"
    assert is_lore_first("world")


def test_gallery_and_sprite_subpages_are_not_lore_first():
    assert is_media_subpage("Faust/Gallery")
    assert is_media_subpage("Gregor/Sprites")
    assert not is_media_subpage("Gregor/Story")
    assert not is_media_subpage("Faust")
    assert classify_document("Faust/Gallery", ["Characters", "Sinners"]) == "other"
    assert classify_document("Gregor/Sprites", ["Characters"]) == "other"
    assert classify_document("Gregor/Story", ["Characters"]) == "character"
    assert not is_lore_first("other")


def test_detect_cantos_normalizes_later_cantos_and_digits():
    assert detect_cantos("Canto V: The Evil Defining", [], "") == ["Canto V"]
    assert detect_cantos("Spoilers", [], "See Canto VIII and canto 4.") == [
        "Canto VIII",
        "Canto IV",
    ]


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
    assert "Identity" in titles
    assert "Seven Assoc. South Section 6 Yi Sang" not in titles
    assert all(d.license for d in docs)
    chunks = chunk_documents(docs)
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    again = chunk_documents(docs)
    assert [c.chunk_id for c in again] == ids


def test_ingest_resume_skips_completed_batches(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "data" / "fixtures"

    class BatchConnector(FixtureSourceConnector):
        def __init__(self, fail_after: int | None = None):
            super().__init__(fixtures)
            self.fail_after = fail_after
            self.calls: list[list[int]] = []
            self.list_calls = 0

        def list_pages(self):
            self.list_calls += 1
            return super().list_pages()

        def fetch_pages(self, page_ids: list[int]):
            self.calls.append(page_ids)
            if self.fail_after is not None and len(self.calls) > self.fail_after:
                raise RuntimeError("interrupted")
            return [self.fetch_page(page_id) for page_id in page_ids]

    documents = tmp_path / "processed" / "documents.jsonl"
    catalog = tmp_path / "processed" / "catalog.sqlite"
    state = tmp_path / "raw" / "ingest_state.json"
    interrupted = BatchConnector(fail_after=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        ingest_connector(
            interrupted,
            documents,
            catalog,
            state_path=state,
            batch_size=2,
        )

    first_batch = set(interrupted.calls[0])
    resumed = BatchConnector()
    docs = ingest_connector(
        resumed,
        documents,
        catalog,
        state_path=state,
        batch_size=2,
    )

    assert resumed.list_calls == 0
    assert all(first_batch.isdisjoint(call) for call in resumed.calls)
    assert {doc.title for doc in docs} >= {"Dongrang", "Yi Sang", "Canto IV"}
    saved_state = json.loads(state.read_text(encoding="utf-8"))
    assert saved_state["skipped_documents"] == [
        {
            "page_id": 106,
            "title": "Seven Assoc. South Section 6 Yi Sang",
            "document_type": "identity",
            "categories": ["Identities", "Yi Sang Identities"],
        }
    ]


def test_incremental_ingest_fetches_changed_and_removes_deleted(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "data" / "fixtures"
    documents = tmp_path / "processed" / "documents.jsonl"
    catalog = tmp_path / "processed" / "catalog.sqlite"
    base = FixtureSourceConnector(fixtures)
    initial = ingest_connector(base, documents, catalog)
    dongrang = next(document for document in initial if document.title == "Dongrang")
    yi_sang = next(document for document in initial if document.title == "Yi Sang")

    class IncrementalConnector(FixtureSourceConnector):
        def list_recent_changes(self, since: str):
            assert since == "2026-01-01T00:00:00Z"
            return [
                {
                    "page_id": dongrang.page_id,
                    "title": dongrang.title,
                    "revision_id": 2,
                    "timestamp": "2026-02-01T00:00:00Z",
                    "deleted": False,
                },
                {
                    "page_id": 0,
                    "title": yi_sang.title,
                    "revision_id": 0,
                    "timestamp": "2026-02-01T01:00:00Z",
                    "deleted": True,
                },
            ]

        def fetch_pages(self, page_ids: list[int]):
            assert page_ids == [dongrang.page_id]
            page = self.fetch_page(dongrang.page_id)
            return [
                page.model_copy(
                    update={
                        "revision_id": 2,
                        "wikitext": page.wikitext + "\n\n== Update ==\nNew fixture-only lore.",
                    }
                )
            ]

    result = ingest_incremental(
        IncrementalConnector(fixtures),
        documents,
        catalog,
        since="2026-01-01T00:00:00Z",
    )

    by_title = {document.title: document for document in result.documents}
    assert by_title["Dongrang"].revision_id == 2
    assert "Yi Sang" not in by_title
    assert dongrang.doc_id in result.changed_doc_ids
    assert yi_sang.doc_id in result.deleted_doc_ids
    assert result.until == "2026-02-01T01:00:00Z"


def test_chunking_merges_tiny_leftovers_and_strips_tables():
    doc = SourceDocument(
        doc_id="wiki:chunk-test",
        source_id="fixture",
        url="https://example.test/wiki/Test",
        title="Test",
        page_id=1,
        revision_id=1,
        document_type="world",
        retrieved_at="2026-01-01T00:00:00Z",
        raw_wikitext=(
            "A long enough intro about the City and its Nests.\n\n"
            "== Gallery ==\n"
            "{| class=\"wikitable\"\n| a || b\n|}\n"
            "[[File:Ignored.png|thumb]]\n"
            "== Background ==\n"
            "The Head enforces the City's laws across every Nest.\n"
        ),
    )

    chunks = chunk_document(doc)
    paths = [chunk.section_path for chunk in chunks]
    assert "Test/Gallery" not in paths
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "wikitable" not in combined
    assert "class=" not in combined
    assert "The Head enforces" in combined
    again = chunk_document(doc)
    assert [chunk.chunk_id for chunk in again] == [chunk.chunk_id for chunk in chunks]


def test_catalog_and_retrieval_hide_media_subpages(tmp_path: Path):
    from limbus_librarian.catalog import CatalogStore
    from limbus_librarian.index.common import matches_filters

    faust = SourceDocument(
        doc_id="wiki:faust",
        source_id="fixture",
        url="https://example.test/wiki/Faust",
        title="Faust",
        page_id=1,
        revision_id=1,
        document_type="sinner",
        retrieved_at="2026-01-01T00:00:00Z",
        plain_text="The Sinner who developed the engine of Mephistopheles.",
    )
    gallery = SourceDocument(
        doc_id="wiki:faust-gallery",
        source_id="fixture",
        url="https://example.test/wiki/Faust/Gallery",
        title="Faust/Gallery",
        page_id=2,
        revision_id=1,
        document_type="character",
        retrieved_at="2026-01-01T00:00:00Z",
        plain_text="__TOC__ CGs = Leviathan",
    )
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        faust.model_dump_json() + "\n" + gallery.model_dump_json() + "\n",
        encoding="utf-8",
    )
    catalog = CatalogStore(tmp_path / "missing.sqlite", documents)
    listed = catalog.list(document_types={"character", "sinner"})
    assert {item["title"] for item in listed["items"]} == {"Faust"}
    assert catalog.get(gallery.doc_id) is None

    gallery_chunk = chunk_document(gallery)[0]
    faust_chunk = chunk_document(faust)[0]
    assert matches_filters(gallery_chunk, None) is False
    assert matches_filters(faust_chunk, None) is True
