from __future__ import annotations

import argparse
import json
from pathlib import Path

from limbus_librarian.config import get_settings
from limbus_librarian.config_loader import load_named_config
from limbus_librarian.eval import evaluate_retrieval, load_gold, write_eval_report
from limbus_librarian.graph import run_ask
from limbus_librarian.runtime import bootstrap_from_fixtures, searcher_from_disk


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="limbus")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest-fixtures", help="Build corpus from local fixtures")
    search_p = sub.add_parser("search", help="Run retrieval")
    search_p.add_argument("query")
    search_p.add_argument("--config", default="hybrid")
    ask_p = sub.add_parser("ask", help="Run the LangGraph ask path")
    ask_p.add_argument("query")
    ask_p.add_argument("--config", default="hybrid_rerank_refine")
    ask_p.add_argument("--debug", action="store_true")
    eval_p = sub.add_parser("eval", help="Retrieval evaluation on gold set")
    eval_p.add_argument("--config", default="hybrid")
    eval_p.add_argument("--k", type=int, default=8)

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "ingest-fixtures":
        bootstrap_from_fixtures(settings)
        print(f"Wrote {settings.documents_path} and {settings.chunks_path}")
        return

    searcher = searcher_from_disk(settings)
    config = load_named_config(settings.configs_dir, args.config)

    if args.cmd == "search":
        hits = searcher.search(args.query, config)
        print(json.dumps([h.model_dump() for h in hits], indent=2))
        return
    if args.cmd == "ask":
        answer = run_ask(
            args.query,
            searcher,
            config,
            api_key=settings.openai_api_key,
            generate_model=settings.generate_model,
            debug=args.debug,
        )
        print(answer.model_dump_json(indent=2))
        return
    if args.cmd == "eval":
        gold = load_gold(settings.gold_path)
        summary = evaluate_retrieval(
            gold,
            lambda q: searcher.search(q, config),
            k=args.k,
        )
        out = Path(settings.data_dir) / "eval" / "runs" / f"{config.id}.json"
        write_eval_report(out, summary)
        print(json.dumps({k: summary[k] for k in ("n", "recall@k", "mrr", "ndcg@k")}, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
