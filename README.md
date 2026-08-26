# Limbus Librarian

An unofficial, fan-made lore research assistant for *Limbus Company*.
**Not affiliated with Project Moon.**

Ask questions about characters, factions, and story events. The system retrieves
supporting passages from a local corpus and answers with citations. It is built
as an AI engineering project (hybrid retrieval, reranking, LangGraph, eval),
not as a simple “embed documents and call an LLM” demo.

Wiki writing, when ingested, is [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
from the [Limbus Company Wiki](https://limbuscompany.wiki.gg/). See [NOTICE.md](NOTICE.md).

## How it works

1. **Ingest** wiki-like pages (fixtures, a JSONL dump, or a polite MediaWiki API client).
   Pages are parsed, classified (character, canto, faction, …), and filtered to a
   **lore-first** set (combat Identity/E.G.O. pages are skipped in V1).
2. **Chunk** by section headings and store documents + chunks on disk.
3. **Retrieve** with BM25 (keywords), dense vectors (meaning), or **hybrid RRF**
   (both). Optional rerank, then a relevance filter.
4. **LangGraph** runs: analyze query → retrieve → grade → at most one query refine
   → generate → check that citation IDs actually exist in retrieved chunks.
5. **Answer** through the CLI, FastAPI, or the React chat UI, with source links.

Retrieval setups are YAML configs you can swap without code changes:
`bm25_only`, `vector_only`, `hybrid`, `hybrid_rerank`, `hybrid_rerank_refine`.

Without `OPENAI_API_KEY`, embeddings are deterministic hashed vectors and answers
are assembled from retrieved snippets (enough for tests and local demos). With a
key, OpenAI embeddings and generation are used.

## Quick start

```bash
python -m pip install -e ".[dev]"
copy .env.example .env
python -m limbus_librarian.cli ingest-fixtures
python -m limbus_librarian.cli ask "Who is Dongrang?"
python -m pytest -q
python -m uvicorn limbus_librarian.api.app:app --reload --port 8000
```

Web UI (requires Node.js):

```bash
cd apps/web
npm install
npm run dev
```

Optional vector DB: `docker compose up qdrant` (V1 tests use an in-process dense index).

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/health` | Liveness + unofficial disclaimer |
| GET | `/v1/configs` | Retrieval config ids |
| POST | `/v1/ask` | `{ "query", "config_id?", "debug?" }` |
| GET | `/v1/sources/{chunk_id}` | Full chunk + metadata |

```bash
python -m limbus_librarian.cli search "What is the League of Nine?" --config hybrid
python -m limbus_librarian.cli eval --config hybrid
```
