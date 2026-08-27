from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from limbus_librarian.models import RetrievalHit, SourceDocument


class GoldItem(BaseModel):
    id: str
    question: str
    question_type: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_doc_titles: list[str] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)


class ResolvedGoldItem(BaseModel):
    item: GoldItem
    relevant_doc_ids: list[str]
    resolved_titles: dict[str, str] = Field(default_factory=dict)
    unresolved_doc_ids: list[str] = Field(default_factory=list)
    unresolved_titles: list[str] = Field(default_factory=list)

    @property
    def status(self) -> str:
        unresolved = self.unresolved_doc_ids or self.unresolved_titles
        if not self.relevant_doc_ids:
            return "unresolved"
        return "partial" if unresolved else "resolved"


def load_gold(path: Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(GoldItem.model_validate_json(line))
    return items


def resolve_gold(
    gold: list[GoldItem],
    documents: list[SourceDocument],
) -> list[ResolvedGoldItem]:
    """Resolve stable title labels against the local catalog without guessing IDs."""
    ids = {document.doc_id for document in documents}
    titles: dict[str, list[SourceDocument]] = {}
    for document in documents:
        titles.setdefault(_normalize_title(document.title), []).append(document)

    resolved: list[ResolvedGoldItem] = []
    for item in gold:
        relevant_ids = [doc_id for doc_id in item.relevant_doc_ids if doc_id in ids]
        unresolved_ids = [doc_id for doc_id in item.relevant_doc_ids if doc_id not in ids]
        resolved_titles: dict[str, str] = {}
        unresolved_titles: list[str] = []
        for title in item.relevant_doc_titles:
            matches = titles.get(_normalize_title(title), [])
            if len(matches) != 1:
                unresolved_titles.append(title)
                continue
            doc_id = matches[0].doc_id
            resolved_titles[title] = doc_id
            if doc_id not in relevant_ids:
                relevant_ids.append(doc_id)
        resolved.append(
            ResolvedGoldItem(
                item=item,
                relevant_doc_ids=relevant_ids,
                resolved_titles=resolved_titles,
                unresolved_doc_ids=unresolved_ids,
                unresolved_titles=unresolved_titles,
            )
        )
    return resolved


def _normalize_title(title: str) -> str:
    return " ".join(title.replace("_", " ").split()).casefold()


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(relevant & top) / len(relevant)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg(rels: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    rels = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
    ideal = [1.0] * min(len(relevant), k)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(rels) / denom


def evaluate_retrieval(
    gold: list[GoldItem] | list[ResolvedGoldItem],
    retrieve_fn: Callable[[str], list[RetrievalHit]],
    k: int = 8,
) -> dict:
    rows = []
    recs, mrrs, ndcgs = [], [], []
    unresolved_items: list[dict] = []
    for entry in gold:
        resolved = (
            entry
            if isinstance(entry, ResolvedGoldItem)
            else ResolvedGoldItem(item=entry, relevant_doc_ids=entry.relevant_doc_ids)
        )
        item = resolved.item
        if resolved.unresolved_doc_ids or resolved.unresolved_titles:
            unresolved_items.append(
                {
                    "id": item.id,
                    "unresolved_doc_ids": resolved.unresolved_doc_ids,
                    "unresolved_titles": resolved.unresolved_titles,
                }
            )
        if not resolved.relevant_doc_ids:
            row = {
                "id": item.id,
                "question_type": item.question_type,
                "status": resolved.status,
                "unresolved_doc_ids": resolved.unresolved_doc_ids,
                "unresolved_titles": resolved.unresolved_titles,
                "retrieved": [],
            }
            rows.append(row)
            continue
        hits: list[RetrievalHit] = retrieve_fn(item.question)
        retrieved_docs = []
        for hit in hits:
            if hit.doc_id not in retrieved_docs:
                retrieved_docs.append(hit.doc_id)
        rel = set(resolved.relevant_doc_ids)
        r = recall_at_k(rel, retrieved_docs, k)
        m = mrr(rel, retrieved_docs)
        n = ndcg_at_k(rel, retrieved_docs, k)
        recs.append(r)
        mrrs.append(m)
        ndcgs.append(n)
        rows.append(
            {
                "id": item.id,
                "question_type": item.question_type,
                "status": resolved.status,
                "recall@k": r,
                "mrr": m,
                "ndcg@k": n,
                "retrieved": retrieved_docs[:k],
                "resolved_titles": resolved.resolved_titles,
                "unresolved_doc_ids": resolved.unresolved_doc_ids,
                "unresolved_titles": resolved.unresolved_titles,
            }
        )
    summary = {
        "n": len(gold),
        "n_evaluated": len(recs),
        "n_unresolved": sum(row.get("status") == "unresolved" for row in rows),
        "n_partial": sum(row.get("status") == "partial" for row in rows),
        "n_with_unresolved_labels": len(unresolved_items),
        "recall@k": sum(recs) / len(recs) if recs else 0.0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "by_question_type": _metrics_by_question_type(rows),
        "unresolved_labels": unresolved_items,
        "rows": rows,
    }
    return summary


def _metrics_by_question_type(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("question_type") or "other")].append(row)
    by_type: dict[str, dict] = {}
    for question_type, type_rows in sorted(grouped.items()):
        evaluated = [row for row in type_rows if "ndcg@k" in row]
        by_type[question_type] = {
            "n": len(type_rows),
            "n_evaluated": len(evaluated),
            "n_unresolved": sum(row.get("status") == "unresolved" for row in type_rows),
            "n_partial": sum(row.get("status") == "partial" for row in type_rows),
            "recall@k": (
                sum(row["recall@k"] for row in evaluated) / len(evaluated) if evaluated else 0.0
            ),
            "mrr": sum(row["mrr"] for row in evaluated) / len(evaluated) if evaluated else 0.0,
            "ndcg@k": (
                sum(row["ndcg@k"] for row in evaluated) / len(evaluated) if evaluated else 0.0
            ),
        }
    return by_type


def write_eval_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
