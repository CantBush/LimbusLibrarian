from __future__ import annotations

from collections import Counter
from typing import Any

from limbus_librarian.ingest.classify import classify_document
from limbus_librarian.models import SourceDocument

_SKIPPED_TYPES = frozenset({"identity", "ego", "other"})


def build_ingest_stats(documents: list[SourceDocument], state: dict[str, Any]) -> dict[str, Any]:
    """Summarize a local ingest without fetching wiki pages."""
    type_counts = Counter(document.document_type for document in documents)
    saved_skips = list(state.get("skipped_documents", []))
    skipped_ids = {int(page_id) for page_id in state.get("skipped_page_ids", [])}

    if saved_skips:
        skipped = saved_skips
        skipped_unknown = max(0, len(skipped_ids) - len(saved_skips))
        categories_available = True
    else:
        listings = {
            int(item["page_id"]): item
            for item in state.get("listings", [])
            if "page_id" in item
        }
        skipped = []
        skipped_unknown = 0
        for page_id in sorted(skipped_ids):
            item = listings.get(page_id)
            if item is None:
                skipped_unknown += 1
                continue
            title = str(item.get("title", ""))
            document_type = classify_document(title, [])
            if document_type == "other":
                skipped_unknown += 1
            skipped.append(
                {
                    "page_id": page_id,
                    "title": title,
                    "document_type": document_type,
                    "categories": [],
                }
            )
        categories_available = False

    type_counts.update(
        str(item.get("document_type", "other"))
        for item in skipped
        if item.get("document_type") in _SKIPPED_TYPES
        and (categories_available or item.get("document_type") != "other")
    )
    category_counts = Counter(
        str(category)
        for item in skipped
        if item.get("document_type") in _SKIPPED_TYPES
        for category in item.get("categories", [])
    )

    def samples(document_type: str) -> list[str]:
        return sorted(
            {
                str(item.get("title", ""))
                for item in skipped
                if item.get("document_type") == document_type and item.get("title")
            },
            key=str.casefold,
        )[:5]

    return {
        "documents_retained": len(documents),
        "pages_listed": len(state.get("listings", [])),
        "document_type_counts": dict(sorted(type_counts.items())),
        "skipped_total": len(skipped_ids),
        "skipped_unknown": skipped_unknown,
        "top_skipped_categories": category_counts.most_common(10),
        "categories_available": categories_available,
        "identity_samples": samples("identity"),
        "ego_samples": samples("ego"),
    }


def format_ingest_stats(stats: dict[str, Any]) -> str:
    lines = [
        f"Catalog documents: {stats['documents_retained']}",
        f"Discovered pages: {stats['pages_listed']}",
        "Document types:",
    ]
    for document_type, count in stats["document_type_counts"].items():
        lines.append(f"  {document_type}: {count}")
    lines.append(f"Skipped pages: {stats['skipped_total']}")
    if stats["skipped_unknown"]:
        lines.append(f"  unknown from legacy state: {stats['skipped_unknown']}")
    lines.append("Top categories among other/skipped:")
    if stats["categories_available"]:
        categories = stats["top_skipped_categories"]
        lines.extend(f"  {category}: {count}" for category, count in categories)
        if not categories:
            lines.append("  (none)")
    else:
        lines.append("  (unavailable: this catalog predates skipped-page metadata)")
    for label, key in (("Identity", "identity_samples"), ("E.G.O.", "ego_samples")):
        lines.append(f"Sample skipped {label} titles:")
        samples = stats[key]
        lines.extend(f"  {title}" for title in samples)
        if not samples:
            lines.append("  (none)")
    return "\n".join(lines)
