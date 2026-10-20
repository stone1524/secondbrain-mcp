"""CLI adapter: index, search, and inspect without an MCP client."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.config import ConfigError, load_settings
from .mcp_stdio import build_registry, dispatch, tool_manifest


def _settings():
    return load_settings()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="secondbrain", description=__doc__)
    parser.add_argument("--vault", help="Vault root (defaults to SB_VAULT_ROOT)")
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="Rebuild the index")
    index_parser.add_argument("--force", action="store_true")

    search_parser = sub.add_parser("search", help="Search the index")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=8)
    search_parser.add_argument("--mode", default="hybrid", choices=["hybrid", "lexical"])
    search_parser.add_argument("--paths", default="", help="Semicolon-separated prefixes")
    search_parser.add_argument("--include-archive", action="store_true")
    search_parser.add_argument("--json", action="store_true")

    sub.add_parser("status", help="Index health")
    sub.add_parser("tools", help="Print the tool manifest as JSON")
    queue_parser = sub.add_parser("review", help="List items awaiting a human decision")
    queue_parser.add_argument("--limit", type=int, default=50)
    eval_parser = sub.add_parser(
        "eval",
        help="Build cases from the vault and score retrieval (recall@5 gates)",
    )
    eval_parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable the embedding backend (slower; model cache required)",
    )
    eval_parser.add_argument("--out", default="", help="Write the JSON report here")

    args = parser.parse_args(argv)
    try:
        settings = load_settings(vault_root=args.vault) if args.vault else _settings()
        tools, settings = build_registry(settings)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if args.command == "index":
        payload = dispatch("index_rebuild", {"force": args.force}, tools)
    elif args.command == "eval":
        # The nightly quality stage: build cases from the vault, score them
        # against the serving index, print the four gates. Delegates to the
        # harness so the CLI and the documented `python -m ...eval.run` path
        # can never disagree about what "passed" means.
        import tempfile
        from contextlib import ExitStack

        from ..eval.build_cases import build
        from ..eval.run import main as eval_main

        cases = build(settings.vault_root)
        with ExitStack() as stack:
            workdir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            cases_path = workdir / "cases.json"
            cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            report_path = Path(args.out) if args.out else workdir / "report.json"
            argv = ["--cases", str(cases_path), "--report", str(report_path)]
            if args.semantic:
                argv.append("--semantic")
            return eval_main(argv)
    elif args.command == "search":
        prefixes = [part for part in args.paths.split(";") if part]
        payload = dispatch(
            "search",
            {
                "query": args.query,
                "limit": args.limit,
                "mode": args.mode,
                "paths": prefixes,
                "include_archive": args.include_archive,
            },
            tools,
        )
    elif args.command == "status":
        payload = dispatch("index_status", {}, tools)
    elif args.command == "review":
        payload = dispatch("review_queue", {"limit": args.limit}, tools)
    else:
        payload = tool_manifest(tools)

    if args.command == "search" and not getattr(args, "json", False) and not payload.get("isError"):
        for hit in payload.get("hits", []):
            print(f"{hit['score']:.4f}  {hit['note']}" + (f"  [{hit['heading']}]" if hit["heading"] else ""))
        if not payload.get("hits"):
            reason = payload.get("empty_reason", {})
            print(f"(no hits) {reason.get('detail', '')}".strip())
        return 0

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    _ = settings
    return 0 if not payload.get("isError") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
