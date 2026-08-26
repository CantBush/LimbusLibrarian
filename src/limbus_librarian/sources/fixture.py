from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from limbus_librarian.sources import RawPage, SourceConnector


class FixtureSourceConnector:
    source_id = "limbuscompany_wiki"

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = fixtures_dir
        self._pages: dict[int, RawPage] = {}
        for path in sorted(fixtures_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            page = RawPage(
                source_id=self.source_id,
                page_id=data["page_id"],
                revision_id=data["revision_id"],
                title=data["title"],
                url=data["url"],
                namespace=data.get("namespace", 0),
                wikitext=data["wikitext"],
                categories=data.get("categories", []),
                last_modified=data.get("last_modified"),
                retrieved_at=data.get("retrieved_at")
                or datetime.now(UTC).isoformat(),
            )
            self._pages[page.page_id] = page

    def list_pages(self) -> list[dict]:
        return [
            {"page_id": p.page_id, "title": p.title, "namespace": p.namespace}
            for p in self._pages.values()
        ]

    def fetch_page(self, page_id: int) -> RawPage:
        return self._pages[page_id]
