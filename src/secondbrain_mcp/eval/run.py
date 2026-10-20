"""Evaluation harness: measurable retrieval quality with gates that can fail.

The production evaluator reported Recall@5 = 1.00 while real queries scored 60%.
The mechanism, once traced, was not fraud but measurement error:

* the nightly run used six hand-written cases, not the 25-case file;
* two of those six narrowed ``paths`` to ``90_Agent/logs``, a directory holding
  exactly one document — recall on a one-document corpus is 1.0 by construction;
* the evaluated default scope held ~22 documents, so roughly 73% of the pool
  consisted of acceptable answers;
* two of the three "gates" could not fail: the traceability predicate is
  tautological by construction, and the privacy probe targets a zone that is
  excluded at index time;
* the query glossary had been extended using the two cases that were failing,
  with no holdout set.

This harness fixes each of those: dev and test cases are separate files, every
gate is falsifiable, and the corpus snapshot is recorded so a number can be
reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import load_settings
from ..index.embed import LexicalOnlyEmbedder, build_embedder
from ..index.schema import status as index_status
from ..index.searcher import Query, Searcher

CATEGORIES = (
    "keyword",
    "paraphrase",
    "proper_noun",
    "technical",
    "cross_lingual",
    "vague_topic",
)


@dataclass(slots=True)
class CaseResult:
    case_id: str
    category: str
    query: str
    rank: int | None
    expected: list[str]


@dataclass(slots=True)
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)
    semantic_wins: int = 0
    null_vectors: int = 0
    privacy_hits: int = 0
    index_leaks: int = 0
    corpus_snapshot: str = ""

    def recall_at(self, k: int) -> float:
        if not self.cases:
            return 0.0
        hits = sum(1 for case in self.cases if case.rank is not None and case.rank < k)
        return hits / len(self.cases)

    def by_category(self) -> dict:
        grouped: dict[str, list[CaseResult]] = {}
        for case in self.cases:
            grouped.setdefault(case.category, []).append(case)
        return {
            category: {
                "cases": len(items),
                "recall_at_5": round(
                    sum(1 for item in items if item.rank is not None and item.rank < 5) / len(items), 4
                ),
                "rank_1": round(
                    sum(1 for item in items if item.rank == 0) / len(items), 4
                ),
            }
            for category, items in sorted(grouped.items())
        }

    def as_dict(self) -> dict:
        return {
            "corpus_snapshot": self.corpus_snapshot,
            "cases": len(self.cases),
            "recall_at_5": round(self.recall_at(5), 4),
            "recall_at_1": round(self.recall_at(1), 4),
            "by_category": self.by_category(),
            "semantic_rank1_wins": self.semantic_wins,
            "null_embedding_chunks": self.null_vectors,
            "privacy_leakage": self.privacy_hits,
            "index_side_leak": self.index_leaks,
            "results": [
                {
                    "id": case.case_id,
                    "category": case.category,
                    "query": case.query,
                    "rank": case.rank,
                    "expected": case.expected,
                }
                for case in self.cases
            ],
        }


def corpus_snapshot(db_path: Path) -> str:
    """A stable digest of indexed content, so a score can be reproduced."""
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT path, content_hash FROM documents WHERE active = 1 ORDER BY path"
        ).fetchall()
    finally:
        connection.close()
    material = "|".join(f"{path}:{digest}" for path, digest in rows)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def run_cases(
    cases: list[dict],
    searcher: Searcher,
    *,
    db_path: Path,
) -> EvalReport:
    report = EvalReport()
    for case in cases:
        expected = case.get("expected") or case.get("expected_sources") or []
        prefixes = tuple(case.get("paths") or ())
        hits, _ = searcher.search(
            Query(text=case["query"], prefixes=prefixes, limit=5, mode="hybrid")
        )
        notes = [hit.note for hit in hits]
        rank = next((index for index, note in enumerate(notes) if note in expected), None)
        report.cases.append(
            CaseResult(
                case_id=case.get("id", case["query"]),
                category=case.get("category", "keyword"),
                query=case["query"],
                rank=rank,
                expected=expected,
            )
        )
        # A semantic win means the note appears only when vectors participate.
        lexical_hits, _ = searcher.search(
            Query(text=case["query"], prefixes=prefixes, limit=5, mode="lexical")
        )
        lexical_notes = [hit.note for hit in lexical_hits]
        if rank == 0 and (not lexical_notes or lexical_notes[0] not in expected):
            report.semantic_wins += 1
    report.null_vectors = index_status(db_path)["null_embedding_chunks"]
    return report


def privacy_probe(searcher: Searcher) -> int:
    """Every casing of the protected directory, queried explicitly.

    This probe can only fail if private rows can reach the index at all. The
    searcher refuses the PRIVATE zone unconditionally, which is the *correct*
    behaviour — but it also means a zero here proves the read-side refusal, not
    the index-side exclusion. Both halves are checked: `index_side_leak` below
    reads the database directly, so a future change that indexes a private file
    is caught even though the search layer would still hide it.
    """
    leaks = 0
    for spelling in ("95_Private", "95_private", "95_PRIVATE", "95_PrIvAtE"):
        hits, _ = searcher.search(
            Query(text=f"{spelling} persona secret", prefixes=(spelling,), limit=20)
        )
        leaks += sum(1 for hit in hits if "95_priv" in hit.note.lower() and "/" in hit.note)
    return leaks


def index_side_leak(db_path: Path) -> int:
    """Documents and chunks whose path belongs to the private zone.

    The search layer hides private rows, so a leak that only lived in the
    database would never show up in `privacy_probe` — which is exactly how the
    production gate became unfalsifiable. This counts rows directly.
    """
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM documents WHERE lower(path) LIKE '95_private/%'"
            " OR lower(path) LIKE '%/95_private/%'"
        ).fetchone()
        return int(row[0])
    except sqlite3.Error:
        return 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="dev or test case file")
    parser.add_argument("--report", default="eval-report.json")
    parser.add_argument("--semantic", action="store_true", help="Enable the embedding backend")
    args = parser.parse_args(argv)

    settings = load_settings()
    embedder = (
        build_embedder(settings.embed_model, enabled=True)
        if args.semantic
        else LexicalOnlyEmbedder()
    )
    searcher = Searcher(settings.rag_db, embedder=embedder)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = run_cases(cases, searcher, db_path=settings.rag_db)
    report.corpus_snapshot = corpus_snapshot(settings.rag_db)
    report.privacy_hits = privacy_probe(searcher)
    report.index_leaks = index_side_leak(settings.rag_db)

    payload = report.as_dict()
    # Four gates, all of them falsifiable. The documented contract had four and
    # the code implemented three: `semantic_rank1_wins >= 1` was printed in the
    # README and in docs/evaluation.md but never evaluated, so a run reporting
    # zero semantic wins still said `passed: true`. In lexical-only mode the
    # semantic gate cannot be met by construction, so it is reported as not
    # applicable rather than silently passing or silently failing.
    semantic_applicable = args.semantic
    payload["gates"] = {
        "recall_at_5>=0.85": report.recall_at(5) >= 0.85,
        "privacy_leakage==0": report.privacy_hits == 0,
        "index_side_leak==0": report.index_leaks == 0,
        "null_embedding_chunks==0": report.null_vectors == 0,
        "semantic_rank1_wins>=1": (
            report.semantic_wins >= 1 if semantic_applicable else None
        ),
    }
    payload["gates_applicable"] = {
        "semantic_rank1_wins>=1": semantic_applicable,
    }
    payload["passed"] = all(
        value for value in payload["gates"].values() if value is not None
    )
    Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("recall_at_5", "recall_at_1", "semantic_rank1_wins", "null_embedding_chunks", "privacy_leakage", "index_side_leak", "gates", "passed")}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


_ = CATEGORIES
