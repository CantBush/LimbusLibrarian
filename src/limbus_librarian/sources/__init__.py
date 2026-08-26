from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RawPage(BaseModel):
    source_id: str
    page_id: int
    revision_id: int
    title: str
    url: str
    namespace: int = 0
    wikitext: str
    categories: list[str] = Field(default_factory=list)
    last_modified: str | None = None
    retrieved_at: str


@runtime_checkable
class SourceConnector(Protocol):
    source_id: str

    def list_pages(self) -> list[dict]:
        """Return dicts with page_id, title, namespace."""

    def fetch_page(self, page_id: int) -> RawPage:
        ...
