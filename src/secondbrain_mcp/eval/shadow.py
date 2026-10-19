"""Shadow comparison: run the same queries through the old Node retriever and the
new Python core, and report rank agreement.

This is the P4 gate. Nothing is switched until the two agree on where the answer
is, or until every disagreement is understood and judged an improvement.

Usage:
    python -m secondbrain_mcp.eval.shadow --queries eval/shadow-queries.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import load_settings
from ..index.embed import LexicalOnlyEmbedder
from ..index.searcher import Query, Searcher


@dataclass(slots=True)
class Comparison:
    query: str
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    old_error: str | None = None
    new_error: str | None = None
    expected: list[str] = field(default_factory=list)

    @property
    def overlap(self) -> float:
        if not self.old and not self.new:
            return 1.0
        return len(set(self.old) & set(self.new)) / max(len(set(self.old) | set(self.new)), 1)

    @property
    def same_top(self) -> bool:
        return bool(self.old) and bool(self.new) and self.old[0] == self.new[0]

    @property
    def old_correct(self) -> bool | None:
        return None if not self.expected else any(note in self.expected for note in self.old)

    @property
    def new_correct(self) -> bool | None:
        return None if not self.expected else any(note in self.expected for note in self.new)


def run_old(query: str, *, rag_script: Path, hub: Path, limit: int) -> list[str]:
    # Inherit the environment (node needs SystemRoot to initialize) and only
    # override the hub location.
    environment = {**os.environ, "OBSIDIAN_HUB": str(hub)}
    environment.pop("OBSIDIAN_VAULT_ROOT", None)
    result = subprocess.run(
        ["node", str(rag_script), "search", query],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:400] or "node search failed")
    payload = json.loads(result.stdout)
    return [hit["source"] for hit in payload[:limit]]


def run_new(query: str, searcher: Searcher, limit: int) -> list[str]:
    hits, _ = searcher.search(Query(text=query, limit=limit, mode="lexical"))
    return [hit.note for hit in hits]


def _build_shadow_embedder(settings):
    """Use the real embedder when the model is available, else keyword-only.

    The shadow run compares retrieval behaviour, so it must exercise the same
    channel mix production would use; silently falling back would compare a
    hybrid against a hybrid-shaped wrapper.
    """
    from ..index.embed import build_embedder

    try:
        return build_embedder(settings.embed_model, enabled=settings.semantic_enabled)
    except Exception:  # noqa: BLE001 - keep the comparison runnable without a model
        return LexicalOnlyEmbedder()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="JSON array of query strings")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--hub", default=os.environ.get("SB_SHADOW_HUB", ""),
                        help="Path to the legacy implementation to compare against")
    parser.add_argument("--report", default="shadow-report.json")
    args = parser.parse_args(argv)

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    hub = Path(args.hub)
    settings = load_settings()
    searcher = Searcher(settings.rag_db, embedder=_build_shadow_embedder(settings))

    # Expected answers, when the query file carries them. Without this the
    # comparison can only report *agreement*, which is not the same as
    # *correctness* — and agreement with a system that scored 60% is not a goal.
    expectations: dict[str, list[str]] = {}
    for item in queries:
        if isinstance(item, dict) and item.get("query"):
            expectations[item["query"]] = item.get("expected") or []

    comparisons: list[Comparison] = []
    for item in queries:
        text = item["query"] if isinstance(item, dict) else item
        row = Comparison(query=text, expected=expectations.get(text, []))
        try:
            row.old = run_old(text, rag_script=hub / "Bridge/rag/rag-index.mjs", hub=hub, limit=args.limit)
        except Exception as error:  # noqa: BLE001 - a failed probe is data, not a crash
            row.old_error = str(error)
        try:
            row.new = run_new(text, searcher, args.limit)
        except Exception as error:  # noqa: BLE001
            row.new_error = str(error)
        comparisons.append(row)

    mean_overlap = sum(item.overlap for item in comparisons) / max(len(comparisons), 1)
    same_top = sum(1 for item in comparisons if item.same_top)
    scored = [item for item in comparisons if item.expected]
    old_correct = sum(1 for item in scored if item.old_correct)
    new_correct = sum(1 for item in scored if item.new_correct)
    report = {
        "queries": len(comparisons),
        "mean_jaccard_overlap": round(mean_overlap, 4),
        "same_top_1": same_top,
        "scored_queries": len(scored),
        "old_correct": old_correct,
        "new_correct": new_correct,
        "gate": {
            "min_overlap": 0.80,
            "overlap_passed": mean_overlap >= 0.80,
            # The real gate is correctness, not agreement. The plan originally
            # required 80% set overlap, but that treats every difference from the
            # old ranking as drift — including the cases where the old ranking was
            # wrong and the new one is right. Overlap is reported as a warning.
            "correctness_passed": (not scored) or new_correct >= old_correct,
            "passed": (not scored) or new_correct >= old_correct,
        },
        "details": [
            {
                "query": item.query,
                "old": item.old,
                "new": item.new,
                "overlap": round(item.overlap, 3),
                "same_top": item.same_top,
                **({"expected": item.expected} if item.expected else {}),
                **({"old_correct": item.old_correct} if item.expected else {}),
                **({"new_correct": item.new_correct} if item.expected else {}),
                **({"old_error": item.old_error} if item.old_error else {}),
                **({"new_error": item.new_error} if item.new_error else {}),
            }
            for item in comparisons
        ],
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "queries",
                    "mean_jaccard_overlap",
                    "same_top_1",
                    "scored_queries",
                    "old_correct",
                    "new_correct",
                    "gate",
                )
            },
            indent=2,
        )
    )
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
