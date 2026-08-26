# Limbus Librarian

Unofficial, fan-made lore research assistant for *Limbus Company*.
**Not affiliated with Project Moon.**

Limbus Librarian is a citation-grounded RAG system over a lore-first corpus
derived from (or tested with fixtures inspired by) the
[Limbus Company Wiki](https://limbuscompany.wiki.gg/) (CC BY-SA 4.0).

See [NOTICE.md](NOTICE.md) for licensing and attribution.

## Quick start

```bash
uv venv
uv pip install -e ".[dev]"
copy .env.example .env   # optional OPENAI_API_KEY
uv run limbus ingest-fixtures
uv run limbus search "Who is Dongrang?" --config hybrid
uv run limbus ask "Who is Dongrang?" --debug
uv run limbus eval --config hybrid
uv run pytest
uv run uvicorn limbus_librarian.api.app:app --reload --port 8000
```

Frontend:

```bash
cd apps/web
npm install
npm run dev
```

Qdrant (optional; V1 tests use an in-process dense index):

```bash
docker compose up qdrant
```

## Retrieval experiments

Configs in `configs/retrieval/`:

- `vector_only`
- `bm25_only`
- `hybrid`
- `hybrid_rerank`
- `hybrid_rerank_refine`

Without `OPENAI_API_KEY`, embeddings are deterministic hashed vectors and
answers are assembled from retrieved snippets (good for tests and demos).

## Live wiki ingest

`MediaWikiSourceConnector` talks to the Action API with a slow rate limit.
Prefer a local dump plus fixtures for development. Do not HTML-scrape,
do not ingest images, and do not train models on wiki text.

## API

- `GET /v1/health`
- `GET /v1/configs`
- `POST /v1/ask` `{ "query": "...", "config_id": "hybrid", "debug": true }`
- `GET /v1/sources/{chunk_id}`
