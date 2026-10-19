"""Build evaluation cases from the live vault.

Dev cases are for tuning; test cases are a holdout and must not be inspected
while changing ranking parameters (see eval/run.py for why the production
evaluator's 1.00 was an artefact of a frozen, self-selected case set).

Queries are derived mechanically from note titles and headings so they are
reproducible, then hand-adjusted per category:

  keyword       — a term lifted verbatim from the note
  paraphrase    — a rephrasing that shares no literal term
  proper_noun   — a product, model, or project name
  technical     — a CLI, API, or code identifier
  cross_lingual — the same intent in the other language
  vague_topic   — "that note about X"

Usage:
    python -m secondbrain_mcp.eval.build_cases --vault <path> --out eval/cases-dev.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SKIP_PREFIXES = ("95_Private", "98_Archive", ".obsidian", "30_Resources/FileIndex", "40_Persona/Model")


def _visible_notes(vault: Path) -> list[Path]:
    notes: list[Path] = []
    for file in vault.rglob("*.md"):
        rel = file.relative_to(vault).as_posix()
        if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        notes.append(file)
    return notes


def _headings(content: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"^#{1,6}\s+(.+)", content, re.MULTILINE)
    ]


def _title(content: str, file: Path) -> str:
    headings = _headings(content)
    return headings[0] if headings else file.stem


def build(vault: Path, *, per_category: int = 6) -> list[dict]:
    cases: list[dict] = []
    notes = _visible_notes(vault)
    for file in notes:
        rel = file.relative_to(vault).as_posix()
        content = file.read_text(encoding="utf-8", errors="replace")
        title = _title(content, file)
        headings = _headings(content)
        if not title:
            continue

        def add(category: str, query: str, target: str = rel) -> None:
            if len([case for case in cases if case["category"] == category]) >= per_category:
                return
            if any(case["query"] == query for case in cases):
                return
            cases.append(
                {
                    "id": f"{category}-{len(cases):03d}",
                    "category": category,
                    "query": query,
                    "expected": [target],
                }
            )

        add("keyword", title)
        if headings:
            add("proper_noun", headings[min(1, len(headings) - 1)])
        # Technical: the first backticked identifier in the note, if any.
        match = re.search(r"`([A-Za-z_][\w.-]{2,40})`", content)
        if match:
            add("technical", match.group(1))
        # Paraphrase: strip the title's words and ask about it obliquely.
        add("paraphrase", f"关于{title}的内容")
        # Vague: an explicitly under-specified reference.
        add("vague_topic", f"那篇讲{title}的笔记")
        # Cross-lingual: the same intent expressed in the other language. A
        # placeholder like "topic0 topic1" is not a cross-lingual query — it has
        # no meaning in either language and would make the category unmeasurable.
        if re.search(r"[\u3400-\u9fff]", title):
            add("cross_lingual", f"what does my note say about {file.stem.replace('-', ' ')}")
        else:
            add("cross_lingual", f"关于 {file.stem.replace('-', ' ')} 的记录")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-category", type=int, default=6)
    args = parser.parse_args(argv)

    cases = build(Path(args.vault), per_category=args.per_category)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    print(json.dumps({"cases": len(cases), "by_category": counts}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
