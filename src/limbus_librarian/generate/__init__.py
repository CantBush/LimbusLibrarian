from __future__ import annotations

from limbus_librarian.models import RetrievalHit


def format_context(hits: list[RetrievalHit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"[cite:{hit.chunk_id}] {hit.title} / {hit.section_path}\n{hit.text}"
        )
    return "\n\n".join(blocks)


def heuristic_answer(query: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return (
            "I could not find this in the loaded corpus. "
            "Limbus Librarian only answers from retrieved source documents."
        )
    parts = [f"Based on the retrieved sources for {query!r}:"]
    for hit in hits[:4]:
        snippet = hit.text.replace("\n", " ")[:220]
        parts.append(f"- {hit.title}: {snippet} [cite:{hit.chunk_id}]")
    return "\n".join(parts)


def generate_answer(
    query: str,
    hits: list[RetrievalHit],
    model: str,
    api_key: str,
) -> str:
    from limbus_librarian.llm import LLMAdapter

    return LLMAdapter(api_key=api_key, generate_model=model).generate(query, hits)
