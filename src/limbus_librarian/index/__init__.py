from __future__ import annotations

from typing import Protocol

from limbus_librarian.models import RetrievalHit


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[RetrievalHit]:
        ...
