# Limbus Librarian

An unofficial, fan-made lore research assistant for *Limbus Company*.
**Not affiliated with Project Moon.**

Ask questions about characters, factions, and story events. The system retrieves
supporting passages from a local corpus and answers with citations. It is built
as an AI engineering project (hybrid retrieval, reranking, LangGraph, eval),
not as a simple “embed documents and call an LLM” demo.

Wiki writing, when ingested, is [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
from the [Limbus Company Wiki](https://limbuscompany.wiki.gg/). See [NOTICE.md](NOTICE.md).

## One-command local prototype

Python 3.12+ is required. The dashboard and API run in one process; Node, Qdrant,
and a separate ingest step are not required.

```powershell
python -m pip install -e ".[dev]"
copy .env.example .env
# Paste your key after OPENAI_API_KEY= in .env (optional)
python -m limbus_librarian.cli serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

With a key, OpenAI embeddings and answer generation are selected automatically.
The first keyed start embeds the small fixture corpus once (typically cents or
less); later starts load `data/indexes/dense.npz`. Without a key, the app remains
fully usable with deterministic local embeddings and extractive answers.

Use `--host` and `--port` to change the bind address:

```powershell
python -m limbus_librarian.cli serve --host 0.0.0.0 --port 8080
```

## How it works

1. **Ingest** wiki-like pages (fixtures, a JSONL dump, or a polite MediaWiki API client).
   Pages are parsed, classified (character, canto, faction, …), and filtered to a
   **lore-first** set (combat Identity/E.G.O. pages and visual subpages such as
   `/Gallery` and `/Sprites` are skipped in V1; `/Story` tabs are kept).
2. **Chunk** by section headings and store documents + chunks on disk.
3. **Retrieve** with BM25 (keywords), dense vectors (meaning), or **hybrid RRF**.
   `hybrid_graph` adds a third deterministic list from wikilink/infobox neighbors.
4. **LangGraph** runs: analyze query → retrieve → grade → at most one query refine
   → generate → check that citation IDs actually exist in retrieved chunks.
5. **Answer** through the CLI or FastAPI-served dashboard, with source links.

Retrieval setups are YAML configs you can swap without code changes:
`bm25_only`, `vector_only`, `hybrid`, `hybrid_rerank`, `hybrid_rerank_refine`,
`hybrid_rerank_cross_encoder`, and `hybrid_graph`. `rerank_backend` is explicitly
`none`, `lexical`, or `cross_encoder`; the cross-encoder and its optional dependency
are loaded only when that backend is selected.

Without `OPENAI_API_KEY`, embeddings are deterministic hashed vectors and answers
are assembled from retrieved snippets (enough for tests and local demos). With a
key, OpenAI embeddings and generation are used.

## Development and verification

```bash
python -m pip install -e ".[dev]"
python -m limbus_librarian.cli ingest-fixtures
python -m limbus_librarian.cli ask "Who is Dongrang?"
python -m pytest -q
```

CI runs the same fixture pytest suite on Python 3.12 (`pip install -e ".[dev]"`);
it does not contact wiki.gg, OpenAI, or install the optional `[rerank]` extra.

## Lore-first wiki ingest

The optional live ingest uses MediaWiki's Action API and configured lore category
trees; it does not crawl HTML, download images, or enumerate every wiki page.
Requests are limited to one per second, page details are fetched in API batches,
and 429/server failures use bounded backoff. Progress is saved to
`data/raw/ingest_state.json`, so rerunning the command resumes an interrupted ingest.

```powershell
python -m limbus_librarian.cli ingest-wiki
# To intentionally discard progress and start a new corpus:
python -m limbus_librarian.cli ingest-wiki --restart
# After a full ingest, fetch only edits since its saved watermark:
python -m limbus_librarian.cli ingest-wiki --since
# Or provide an explicit MediaWiki timestamp:
python -m limbus_librarian.cli ingest-wiki --since 2026-08-01T00:00:00Z
```

The command classifies and stores lore pages, skips Identity/E.G.O. combat pages
and visual subpages (`/Gallery`, `/Sprites`, and similar image tabs), chunks
the corpus, and rebuilds BM25 plus the dense cache. `/Story` character tabs are
kept. If `OPENAI_API_KEY` is set,
dense embeddings use OpenAI in batches; otherwise they use the local deterministic
embedder. It does not perform model training.

Incremental ingest uses MediaWiki `recentchanges`, compares revision IDs, rechunks
only affected pages, removes stale/deleted chunks, and reuses dense vectors for
unchanged chunk IDs. BM25 is rewritten from the resulting local chunk set. The
same pass deterministically rebuilds SQLite `entities` and `edges` from parsed
wikilinks and selected infobox relationships; it does not use an LLM extractor.

After ingest, restart `limbus serve` or reload indexes in the running process:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/reload
```

wiki.gg's Terms of Service and robots policy restrict scraper behavior. Review and
honor the current policies before ingesting; this deliberately slow local fan-tool
path uses the Action API rather than an HTML scraper. Fixtures remain the only
corpus used by tests and CI, and tests never contact wiki.gg.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/health` | Liveness, key status + unofficial disclaimer |
| GET | `/v1/configs` | Retrieval config ids |
| POST | `/v1/ask` | `{ "query", "config_id?", "debug?" }` |
| POST | `/v1/reload` | Reload chunks and retrieval indexes from disk |
| GET | `/v1/sources/{chunk_id}` | Full chunk + metadata |
| GET | `/v1/documents/{doc_id}/related` | Deterministic graph-linked pages |

```bash
python -m limbus_librarian.cli search "What is the League of Nine?" --config hybrid
python -m limbus_librarian.cli eval --gold wiki_v1 --config hybrid
python -m limbus_librarian.cli eval --gold wiki_v1 --compare
# Explicit model gates (never run by normal asks/tests):
python -m pip install -e ".[rerank]"
python -m limbus_librarian.cli eval --gold wiki_v1 --rerank-compare
python -m limbus_librarian.cli eval --gold wiki_v1 --luna-experiment --experiment-limit 6
python -m limbus_librarian.cli eval --gold wiki_v1 --generation-eval --experiment-limit 20
```

The comparison command evaluates `bm25_only`, `vector_only`, `hybrid`,
`hybrid_rerank`, and `hybrid_graph`, prints overall and per-`question_type`
Recall/MRR/nDCG rows, and writes JSON reports under `data/eval/runs/`.
Gold labels may use exact `relevant_doc_ids`, stable `relevant_doc_titles`, or
both. Title labels are resolved against the locally ingested document catalog.
Until the first live ingest has populated that catalog, `wiki_v1` labels are
reported as unresolved and excluded from metric averages; the evaluator never
invents wiki page IDs. The legacy fixture set remains available as `--gold v1`.

After classifier changes that keep the Identity and E.G.O. overview pages,
run `ingest-wiki --since` (or a targeted re-fetch) so those titles enter the
local catalog. Combat Identity/E.G.O. pages stay skipped. Remaining unresolved
`wiki_v1` titles in the current catalog are The Pallid Whale, Pequod, and
Fixers (plus Identity / E.G.O. until that ingest lands). Partial items such as
wiki-q21 still resolve Queequeg without fabricating a Pequod id.

`limbus serve` does not use the Qdrant client or the legacy Vite app under
`apps/web`. `docker compose up --build` is an optional API-only container path.
Full deletion of those unused pieces can wait; they are not a prototype feature.

### Current retrieval decision

The local `wiki_v1` run (36 evaluable questions at K=8) makes `vector_only` the
evidence-backed default: Recall 0.875, MRR 0.766, and nDCG 0.790. Hybrid scored
0.852 / 0.713 / 0.741, while lexical hybrid reranking fell to
0.792 / 0.683 / 0.687. Lexical reranking therefore remains available but off by
default. The BAAI cross-encoder scored Recall 0.880 / MRR 0.819 / nDCG 0.815,
beating `vector_only` by 0.005 / 0.053 / 0.025 and lexical reranking by
0.088 / 0.137 / 0.128. However, the cold CPU comparison took 999.512 seconds
and requires the large optional Torch/model install. That latency is unsuitable
for default interactive asks, so `vector_only` remains the default and the
cross-encoder remains an explicit quality-over-latency option. Exact results are
in `data/eval/runs/wiki_v1.rerank_comparison.json`.

The Luna gate is also opt-in. It makes at most `--experiment-limit` structured
utility-model calls (maximum 8) on an empty-retrieval/relationship slice, using
`OPENAI_API_KEY` only when configured. It compares deterministic refinement with
the model rewrite/relevance result and writes
`data/eval/runs/wiki_v1.luna_experiment.json`; without a key it records
`not_executed`. Normal asks, offline eval, and tests never call Luna. Eval outputs
under `data/eval/runs/` are local ignored artifacts.

On the bounded six-question relationship slice, Luna rewrite/relevance improved
the heuristic arm from Recall 0.694 / MRR 0.389 / nDCG 0.484 to
0.778 / 0.583 / 0.602. This is promising but too small and model-dependent to
replace deterministic defaults, so Luna remains an explicitly invoked experiment.

`--compare` also prints per-`question_type` slices. On the local `wiki_v1`
relationship slice (7 evaluable items, including partials), `hybrid_graph`
beat `vector_only` on the primary metric: nDCG 0.691 vs 0.584 (MRR 0.786 vs
0.500; Recall 0.738 vs 0.810). That nDCG margin is large enough to overlay
graph retrieval for relationship questions when the loaded config is the
default `vector_only` (same `k_graph` as `hybrid_graph.yaml`). Explicit UI/CLI
`--config` values stay unchanged, and graph neighbors remain available on
catalog related-pages for every type. `vector_only` remains the default recipe
overall.

Generation eval is a separate opt-in gate:
`eval --gold wiki_v1 --generation-eval --experiment-limit 20`. It skips
unresolved gold items, scores extractive-answer coverage of
`expected_answer_points` with or without a key, and writes
`data/eval/runs/wiki_v1.generation_eval.json`. A local extractive run on 20
resolved questions scored coverage 0.000 (gold bullets are not verbatim in
retrieved snippets). Faithfulness and citation-claim overlap use one bounded
structured Luna call per selected question and require `OPENAI_API_KEY`;
without a key the model arm is `not_executed` while extractive coverage is
still recorded. With a key the command is sequential and slow: up to 20
questions × two OpenAI calls (generate, then judge), often a minute or more
each. It now prints per-question progress, times out a stuck request after
300s, and writes a partial report if you Ctrl+C. The judge may only see
retrieved chunk texts and cannot invent chunk IDs. Default generator, prompt,
and hop limit are unchanged.
