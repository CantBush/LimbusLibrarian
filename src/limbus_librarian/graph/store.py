from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from limbus_librarian.index.common import chunk_to_hit, matches_filters
from limbus_librarian.models import Chunk, RetrievalHit, SourceDocument

_WORDS = re.compile(r"[a-z0-9]+")
_INFOBOX_RELATIONS = {
    "affiliation": "affiliated_with",
    "affiliations": "affiliated_with",
    "employer": "affiliated_with",
    "faction": "member_of",
    "group": "member_of",
    "member_of": "member_of",
    "organization": "member_of",
}


def _key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _clean_target(value: str) -> str:
    return " ".join(value.replace("<br>", " ").replace("<br/>", " ").split()).strip(" ,")


class GraphStore:
    """Deterministic SQLite entity graph derived from parsed wiki metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def rebuild(self, documents: list[SourceDocument]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entities: dict[str, tuple[str, str | None, str]] = {}
        edges: dict[tuple[str, str, str, str], tuple[str, str, str, str]] = {}

        def add_edge(src: str, rel: str, dst: str, doc_id: str) -> None:
            dst_key = _key(dst)
            entities.setdefault(dst_key, (dst, None, "other"))
            stored_dst = entities[dst_key][0]
            identity = (_key(src), rel, dst_key, doc_id)
            edges.setdefault(identity, (src, rel, stored_dst, doc_id))

        for document in sorted(documents, key=lambda item: (_key(item.title), item.doc_id)):
            title_key = _key(document.title)
            entities[title_key] = (document.title, document.doc_id, document.document_type)

        for document in sorted(documents, key=lambda item: (_key(item.title), item.doc_id)):
            src = entities[_key(document.title)][0]
            for raw_target in sorted(set(document.entities), key=lambda item: (_key(item), item)):
                target = _clean_target(raw_target)
                if not target or _key(target) == _key(src):
                    continue
                add_edge(src, "links_to", target, document.doc_id)
            for raw_key, raw_value in sorted(
                document.infobox.items(), key=lambda item: (_key(item[0]), item[0])
            ):
                relation = _INFOBOX_RELATIONS.get(_key(raw_key).replace(" ", "_"))
                target = _clean_target(raw_value)
                if relation is None or not target or len(target) > 160:
                    continue
                add_edge(src, relation, target, document.doc_id)

        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    title TEXT PRIMARY KEY COLLATE NOCASE,
                    doc_id TEXT,
                    type TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT NOT NULL COLLATE NOCASE,
                    rel TEXT NOT NULL,
                    dst TEXT NOT NULL COLLATE NOCASE,
                    doc_id TEXT NOT NULL,
                    PRIMARY KEY (src, rel, dst, doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
                CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
                CREATE INDEX IF NOT EXISTS idx_edges_doc ON edges(doc_id);
                """
            )
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM entities")
            connection.executemany(
                "INSERT OR IGNORE INTO entities(title, doc_id, type) VALUES (?, ?, ?)",
                [entities[key] for key in sorted(entities)],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO edges(src, rel, dst, doc_id) VALUES (?, ?, ?, ?)",
                sorted(edges.values(), key=lambda edge: tuple(_key(value) for value in edge)),
            )

    def related(self, doc_id: str, limit: int = 12) -> list[dict]:
        if not self.path.exists() or limit <= 0:
            return []
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT title FROM entities WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                return []
            title = str(row[0])
            edge_rows = connection.execute(
                """
                SELECT src, rel, dst, doc_id
                FROM edges
                WHERE src = ? COLLATE NOCASE OR dst = ? COLLATE NOCASE
                ORDER BY rel, src COLLATE NOCASE, dst COLLATE NOCASE, doc_id
                """,
                (title, title),
            ).fetchall()
            entity_rows = connection.execute(
                "SELECT title, doc_id, type FROM entities"
            ).fetchall()
        by_title = {
            _key(str(entity_title)): (str(entity_title), entity_doc_id, entity_type)
            for entity_title, entity_doc_id, entity_type in entity_rows
        }
        related: dict[str, dict] = {}
        for src, relation, dst, _edge_doc_id in edge_rows:
            other = str(dst) if _key(str(src)) == _key(title) else str(src)
            entity = by_title.get(_key(other))
            if entity is None or entity[1] is None or entity[1] == doc_id:
                continue
            item = related.setdefault(
                str(entity[1]),
                {
                    "doc_id": str(entity[1]),
                    "title": entity[0],
                    "document_type": entity[2],
                    "relations": [],
                },
            )
            direction = "outgoing" if _key(str(src)) == _key(title) else "incoming"
            descriptor = {"relation": str(relation), "direction": direction}
            if descriptor not in item["relations"]:
                item["relations"].append(descriptor)
        return sorted(related.values(), key=lambda item: (_key(item["title"]), item["doc_id"]))[
            :limit
        ]

    def query_entities(self, query: str, limit: int = 4) -> list[str]:
        if not self.path.exists() or limit <= 0:
            return []
        exact_mentions = [title for title, _entity_type in self.match_entities(query, limit)]
        if exact_mentions:
            return exact_mentions
        query_key = _key(query)
        query_words = set(_WORDS.findall(query_key))
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT title FROM entities ORDER BY length(title) DESC, title COLLATE NOCASE"
            ).fetchall()
        scored: list[tuple[int, int, str]] = []
        for row in rows:
            title = str(row[0])
            words = set(_WORDS.findall(_key(title)))
            overlap = len(words & query_words)
            if overlap and words:
                scored.append((overlap, len(words), title))
        scored.sort(key=lambda item: (-item[0], -item[1], _key(item[2])))
        return [item[2] for item in scored[:limit]]

    def match_entities(self, query: str, limit: int = 4) -> list[tuple[str, str]]:
        """Return non-overlapping catalog titles explicitly mentioned in a query."""
        if not self.path.exists() or limit <= 0:
            return []
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """
                SELECT title, type
                FROM entities
                ORDER BY length(title) DESC, title COLLATE NOCASE
                """
            ).fetchall()

        occupied: list[tuple[int, int]] = []
        matches: list[tuple[int, str, str]] = []
        for raw_title, raw_type in rows:
            title = str(raw_title)
            pattern = re.escape(title).replace(r"\ ", r"\s+")
            for mention in re.finditer(rf"(?<!\w){pattern}(?!\w)", query, re.IGNORECASE):
                span = mention.span()
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                occupied.append(span)
                matches.append((span[0], title, str(raw_type)))
                break
        matches.sort(key=lambda item: (item[0], _key(item[1])))
        return [(title, entity_type) for _, title, entity_type in matches[:limit]]

    def neighbor_doc_ids(self, titles: list[str], limit: int = 8) -> list[str]:
        if not self.path.exists() or not titles or limit <= 0:
            return []
        scores: dict[str, int] = defaultdict(int)
        with sqlite3.connect(self.path) as connection:
            for title in titles:
                rows = connection.execute(
                    """
                    SELECT e.doc_id, edge.doc_id
                    FROM edges AS edge
                    LEFT JOIN entities AS e
                      ON e.title = CASE
                        WHEN edge.src = ? COLLATE NOCASE THEN edge.dst
                        ELSE edge.src
                      END COLLATE NOCASE
                    WHERE edge.src = ? COLLATE NOCASE OR edge.dst = ? COLLATE NOCASE
                    """,
                    (title, title, title),
                ).fetchall()
                for neighbor_doc_id, evidence_doc_id in rows:
                    if neighbor_doc_id:
                        scores[str(neighbor_doc_id)] += 2
                    if evidence_doc_id:
                        scores[str(evidence_doc_id)] += 1
        return [
            doc_id
            for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]


class GraphRetriever:
    name = "graph"

    def __init__(self, chunks: list[Chunk], store: GraphStore, max_neighbors: int = 8) -> None:
        self.chunks = chunks
        self.store = store
        self.max_neighbors = max_neighbors

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict | None = None,
        max_neighbors: int | None = None,
        entity_titles: list[str] | None = None,
    ) -> list[RetrievalHit]:
        titles = entity_titles or self.store.query_entities(query)
        doc_ids = self.store.neighbor_doc_ids(
            titles,
            self.max_neighbors if max_neighbors is None else max_neighbors,
        )
        if not doc_ids or k <= 0:
            return []
        query_words = set(_WORDS.findall(_key(query)))
        candidates: list[tuple[int, int, int, Chunk]] = []
        doc_rank = {doc_id: rank for rank, doc_id in enumerate(doc_ids)}
        for chunk in self.chunks:
            if chunk.doc_id not in doc_rank or not matches_filters(chunk, filters):
                continue
            text_words = set(_WORDS.findall(_key(f"{chunk.title} {chunk.text}")))
            overlap = len(query_words & text_words)
            candidates.append((doc_rank[chunk.doc_id], -overlap, chunk.ordinal, chunk))
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].chunk_id))
        hits: list[RetrievalHit] = []
        per_document: dict[str, int] = defaultdict(int)
        for _, neg_overlap, _, chunk in candidates:
            if per_document[chunk.doc_id] >= 2:
                continue
            rank = len(hits) + 1
            score = 1.0 / rank + (-neg_overlap * 0.01)
            hits.append(chunk_to_hit(chunk, score, rank, self.name))
            per_document[chunk.doc_id] += 1
            if len(hits) >= k:
                break
        return hits
