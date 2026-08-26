from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from limbus_librarian.models import RetrievalHit


class GoldItem(BaseModel):
    id: str
    question: str
    question_type: str
    relevant_doc_ids: list[str]
    expected_answer_points: list[str] = Field(default_factory=list)


def load_gold(path: Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(GoldItem.model_validate_json(line))
    return items


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
    return sum(rel / (1 if i == 0 else __import__("math").log2(i + 1)) for i, rel in enumerate(rels))


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    rels = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(rels) / denom


def evaluate_retrieval(
    gold: list[GoldItem],
    retrieve_fn,
    k: int = 8,
) -> dict:
    rows = []
    recs, mrrs, ndcgs = [], [], []
    for item in gold:
        hits: list[RetrievalHit] = retrieve_fn(item.question)
        retrieved_docs = []
        for hit in hits:
            if hit.doc_id not in retrieved_docs:
                retrieved_docs.append(hit.doc_id)
        rel = set(item.relevant_doc_ids)
        r = recall_at_k(rel, retrieved_docs, k)
        m = mrr(rel, retrieved_docs)
        n = ndcg_at_k(rel, retrieved_docs, k)
        recs.append(r)
        mrrs.append(m)
        ndcgs.append(n)
        rows.append(
            {
                "id": item.id,
                "recall@k": r,
                "mrr": m,
                "ndcg@k": n,
                "retrieved": retrieved_docs[:k],
            }
        )
    summary = {
        "n": len(gold),
        "recall@k": sum(recs) / len(recs) if recs else 0.0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "rows": rows,
    }
    return summary


def write_eval_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
