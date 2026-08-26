from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from limbus_librarian.sources import RawPage


class DumpSourceConnector:
    """Load a local JSONL dump of wiki pages (one RawPage per line)."""

    source_id = "limbuscompany_wiki"

    def __init__(self, dump_path: Path) -> None:
        self.dump_path = dump_path
        self._pages: dict[int, RawPage] = {}
        for line in dump_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            page = RawPage.model_validate(
                {
                    **data,
                    "retrieved_at": data.get("retrieved_at")
                    or datetime.now(UTC).isoformat(),
                }
            )
            self._pages[page.page_id] = page

    def list_pages(self) -> list[dict]:
        return [
            {"page_id": p.page_id, "title": p.title, "namespace": p.namespace}
            for p in self._pages.values()
        ]

    def fetch_page(self, page_id: int) -> RawPage:
        return self._pages[page_id]
