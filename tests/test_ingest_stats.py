from limbus_librarian.ingest.stats import build_ingest_stats, format_ingest_stats
from limbus_librarian.models import SourceDocument


def _document(title: str, document_type: str) -> SourceDocument:
    return SourceDocument(
        doc_id=f"fixture:{title}",
        source_id="fixture",
        url="https://example.test",
        title=title,
        page_id=1,
        revision_id=1,
        document_type=document_type,
        retrieved_at="2026-01-01T00:00:00Z",
    )


def test_ingest_stats_counts_saved_skips_and_categories():
    stats = build_ingest_stats(
        [_document("Yi Sang", "sinner")],
        {
            "listings": [{"page_id": page_id} for page_id in (1, 2, 3)],
            "skipped_page_ids": [2, 3],
            "skipped_documents": [
                {
                    "page_id": 2,
                    "title": "Seven Yi Sang",
                    "document_type": "identity",
                    "categories": ["Identities", "Yi Sang Identities"],
                },
                {
                    "page_id": 3,
                    "title": "Crow's Eye View",
                    "document_type": "ego",
                    "categories": ["E.G.O", "Yi Sang E.G.O"],
                },
            ],
        },
    )

    assert stats["document_type_counts"] == {"ego": 1, "identity": 1, "sinner": 1}
    assert stats["top_skipped_categories"] == [
        ("Identities", 1),
        ("Yi Sang Identities", 1),
        ("E.G.O", 1),
        ("Yi Sang E.G.O", 1),
    ]
    assert stats["identity_samples"] == ["Seven Yi Sang"]
    assert stats["ego_samples"] == ["Crow's Eye View"]


def test_ingest_stats_handles_legacy_state_without_network():
    stats = build_ingest_stats(
        [],
        {
            "listings": [
                {"page_id": 2, "title": "Seven Yi Sang/Identity Story"},
                {"page_id": 3, "title": "Crow's Eye View E.G.O"},
                {"page_id": 4, "title": "Unknown Enemy"},
            ],
            "skipped_page_ids": [2, 3, 4],
        },
    )

    assert stats["document_type_counts"] == {"ego": 1, "identity": 1}
    assert stats["skipped_unknown"] == 1
    assert "unavailable: this catalog predates" in format_ingest_stats(stats)
