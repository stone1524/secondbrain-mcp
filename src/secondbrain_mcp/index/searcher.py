"""Hybrid retrieval: BM25 over a posting list, cosine over a vector matrix, RRF fusion.

Why RRF instead of the production formula
-----------------------------------------
Production scored ``lexical + cosine * 2``. Measured lexical scores ran 3–35 and
cosine 0–1, so the vector channel could contribute at most 2 points against a
channel that routinely produced 14. The vector side could never change the
ordering — which is exactly what the probe found: in 11 of 11 queries where both
engines returned the note, the hybrid and the keyword-only search produced
*identical* ranks. Reciprocal Rank Fusion compares positions, not magnitudes, so
neither channel can swamp the other by scale.

Why the gates
-------------
On a 22-document live corpus, near-empty persona placeholders sat close to the
corpus centroid. With lexical score 0 they still entered the candidate set
through cosine alone and out-ranked substantive notes on unrelated queries. Two
gates remove that: a semantic-only hit must clear both a similarity floor and a
minimum token count, and a near-empty body is down-weighted.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.policy import (
    Zone,
    in_default_scope,
    read_allowed,
    scope_prefixes_allows,
    zone_of,
)
from .embed import Embedder, LexicalOnlyEmbedder
from .schema import initialize

BM25_K1 = 1.2
BM25_B = 0.75
PHRASE_BONUS = 0.5
CANDIDATE_LIMIT = 200

# Conversational wrappers users actually type around a note's name. Stripping
# them recovers the *core* being asked for: 「关于X的内容」 and 「那篇讲X的笔记」
# are X, and "what does my note say about X" is X. The forensic evidence: every
# paraphrase/vague case lost rank-1 had its exact title inside such a wrapper,
# while a long note that merely *mentions* the title won on raw accumulation.
_QUERY_WRAPPERS = (
    re.compile(r"^关于(?P<core>.+?)的(?:内容|记录|笔记|说明|介绍)$"),
    re.compile(r"^那篇讲(?P<core>.+?)的笔记$"),
    re.compile(r"^那篇(?P<core>.+?)的笔记$"),
    re.compile(r"^what does my note say about (?P<core>.+)$", re.IGNORECASE),
)


def query_cores(text: str) -> list[str]:
    """The query plus each wrapper-stripped variant, longest-first unchanged.

    Kept as a module function so tests can pin the stripping table directly;
    this is retrieval behavior, not per-call state."""
    cores = [text]
    stripped = text.strip()
    for pattern in _QUERY_WRAPPERS:
        match = pattern.match(stripped)
        if match:
            core = match.group("core").strip()
            if core:
                cores.append(core)
    return cores


@dataclass(slots=True)
class Hit:
    note: str
    heading: str | None
    excerpt: str
    content_hash: str | None
    updated_at: str | None
    score: float
    lexical_score: float
    semantic_score: float
    lexical_rank: int | None
    semantic_rank: int | None
    matched_terms: list[str]
    trust: str = "untrusted_note"

    def as_dict(self) -> dict:
        return {
            "note": self.note,
            "heading": self.heading,
            "excerpt": self.excerpt,
            "content_hash": self.content_hash,
            "updated_at": self.updated_at,
            "score": round(self.score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "semantic_score": round(self.semantic_score, 6),
            "lexical_rank": self.lexical_rank,
            "semantic_rank": self.semantic_rank,
            "matched_terms": self.matched_terms,
            "trust": self.trust,
        }


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    prefixes: tuple[str, ...] = ()
    include_archive: bool = False
    limit: int = 8
    mode: str = "hybrid"  # hybrid | lexical


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    scope_reason: str
    lexical_candidates: int
    semantic_candidates: int
    semantic_enabled: bool
    gates_applied: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "scope_reason": self.scope_reason,
            "lexical_candidates": self.lexical_candidates,
            "semantic_candidates": self.semantic_candidates,
            "semantic_enabled": self.semantic_enabled,
            "gates_applied": list(self.gates_applied),
        }


class Searcher:
    def __init__(
        self,
        db_path: Path | str,
        *,
        embedder: Embedder | None = None,
        rrf_k: int = 60,
        rrf_depth: int = 50,
        semantic_floor: float = 0.55,
        semantic_min_tokens: int = 20,
        stub_body_chars: int = 120,
        stub_penalty: float = 0.5,
    ):
        self.db_path = Path(db_path)
        self.embedder = embedder or LexicalOnlyEmbedder()
        self.rrf_k = rrf_k
        self.rrf_depth = rrf_depth
        self.semantic_floor = semantic_floor
        self.semantic_min_tokens = semantic_min_tokens
        self.stub_body_chars = stub_body_chars
        self.stub_penalty = stub_penalty
        self._cache: dict = {"generation": None, "matrix": None, "ids": None, "vectors_by_chunk": None}

    # ---------- public ----------

    def search(self, query: Query) -> tuple[list[Hit], SearchDiagnostics]:
        connection = initialize(self.db_path)
        try:
            generation = self._generation(connection)
            tokens = self._tokenize(query.text)
            if not tokens:
                return [], SearchDiagnostics(
                    scope_reason="empty_query",
                    lexical_candidates=0,
                    semantic_candidates=0,
                    semantic_enabled=False,
                    gates_applied=(),
                )
            scope = self._scope(query)
            lexical = self._lexical(connection, query.text, tokens, scope)
            semantic_enabled = query.mode == "hybrid" and not isinstance(
                self.embedder, LexicalOnlyEmbedder
            )
            semantic = (
                self._semantic(connection, generation, query.text, scope)
                if semantic_enabled
                else []
            )
            gates: list[str] = []
            if semantic_enabled:
                semantic, gates = self._apply_semantic_gates(semantic)
            hits = self._fuse(
                lexical, semantic, scope, query, self._note_bodies(connection)
            )
            reason = (
                "explicit_prefixes"
                if query.prefixes
                else ("include_archive" if query.include_archive else "default")
            )
            return hits, SearchDiagnostics(
                scope_reason=reason,
                lexical_candidates=len(lexical),
                semantic_candidates=len(semantic),
                semantic_enabled=semantic_enabled,
                gates_applied=tuple(gates),
            )
        finally:
            connection.close()

    # ---------- scope ----------

    def _scope(self, query: Query) -> callable:
        prefixes = list(query.prefixes)

        def allowed(note: str) -> bool:
            zone = zone_of(note)
            if zone is Zone.PRIVATE:
                return False
            if zone is Zone.GENERATED:
                # Generated aggregates (audit, candidates, index-of-index) are
                # never answers; they exist so a caller can find them by name.
                return bool(prefixes) or query.include_archive
            if zone is Zone.ARCHIVE:
                return query.include_archive or any(
                    note.startswith(prefix.strip("/") + "/") for prefix in prefixes
                )
            if prefixes:
                return scope_prefixes_allows(note, prefixes)
            return in_default_scope(note)

        return allowed

    # ---------- lexical ----------

    def _lexical(
        self,
        connection: sqlite3.Connection,
        text: str,
        tokens: list[str],
        allowed,
    ) -> list[dict]:
        placeholders = ",".join("?" for _ in tokens)
        rows = connection.execute(
            "SELECT p.chunk_id, p.term, p.tf, c.path, c.heading, c.title, c.content,"
            " c.token_count, c.body_chars, d.content_hash, d.updated_at"
            " FROM postings p"
            " JOIN chunks c ON c.id = p.chunk_id AND c.active = 1"
            " JOIN documents d ON d.path = c.path AND d.active = 1"
            f" WHERE p.term IN ({placeholders})",
            tuple(tokens),
        ).fetchall()
        if not rows:
            return []

        # N counts *active* documents only. Counting retired chunks inflated it
        # and skewed idf toward zero on vaults with churn.
        document_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT c.path) AS n FROM chunks c"
                " JOIN documents d ON d.path = c.path AND d.active = 1"
                " WHERE c.active = 1"
            ).fetchone()["n"]
        ) or 1
        # df is the number of distinct *documents* containing the term, not the
        # number of posting rows. A note split into k matching chunks inflated df
        # by k; once df exceeded N the Robertson-Sparck Jones idf went negative and
        # the most-relevant multi-chunk note sank below single-chunk noise.
        document_frequency: dict[str, int] = {}
        token_set = set(tokens)
        df_rows = connection.execute(
            "SELECT DISTINCT p.term, c.path FROM postings p"
            " JOIN chunks c ON c.id = p.chunk_id AND c.active = 1"
            " JOIN documents d ON d.path = c.path AND d.active = 1"
            f" WHERE p.term IN ({placeholders})",
            tuple(tokens),
        ).fetchall()
        for row in df_rows:
            if row["term"] in token_set:
                document_frequency[row["term"]] = document_frequency.get(row["term"], 0) + 1

        # Length normalization uses real prose only: file stubs are a handful of
        # tokens each and otherwise crush the average toward zero.
        prose = [
            row
            for row in connection.execute(
                "SELECT body_chars FROM chunks WHERE active = 1"
            ).fetchall()
            if row["body_chars"] > 0
        ]
        average_body = max(
            1.0,
            sum(row["body_chars"] for row in prose) / max(len(prose), 1),
        )

        # A note's own title is usually stored with hyphens while a user retypes it
        # with punctuation ("（", "·", spaces). Comparing raw substrings misses
        # exactly the queries that name the note most precisely, so the phrase
        # test runs on a punctuation-free form of both sides.
        def phrase_key(value: str) -> str:
            return "".join(character for character in value.lower() if character.isalnum())

        # Stub detection must look at the *note*, not the matched section: the
        # production attractor was a persona file that is empty everywhere. A
        # substantive note with one small section (「背景」 at 61 chars) is not
        # a stub, but the chunk-level test halved its won rank anyway.
        note_bodies = self._note_bodies(connection)
        full_key = phrase_key(text)
        core_keys = [
            phrase_key(core)
            for core in query_cores(text)
            if core != text and len(phrase_key(core)) >= 4
        ]
        # Per-document identity fields: the H1 title and the file stem — what
        # the user *named*, not what a section happens to discuss. A long
        # meta-review that quotes another note's title in its body has zero
        # identity match; the note that is that title has one. The forensic
        # pattern driving this split: a cross-referencing note won 6 of 11
        # lost cases by accumulating per-chunk BM25 while the title-holder
        # failed the old body-only phrase test.
        best: dict[str, dict] = {}
        for row in rows:
            if not allowed(row["path"]):
                continue
            term = row["term"]
            frequency = int(row["tf"])
            df = max(1, document_frequency.get(term, 1))
            idf = np.log(1 + (document_count - df + 0.5) / (df + 0.5))
            length_ratio = row["body_chars"] / average_body
            score = idf * (
                frequency * (BM25_K1 + 1)
            ) / (frequency + BM25_K1 * (1 - BM25_B + BM25_B * length_ratio))
            entry = best.get(row["path"])
            if entry is None:
                entry = {
                    "note": row["path"],
                    "heading": row["heading"],
                    "content": row["content"],
                    "content_hash": row["content_hash"],
                    "updated_at": row["updated_at"],
                    "token_count": int(row["token_count"]),
                    "body_chars": int(row["body_chars"]),
                    "lexical": 0.0,
                    "matched": set(),
                    "struct": {
                        phrase_key(row["title"]),
                        phrase_key(Path(row["path"]).stem),
                        phrase_key(row["heading"] or ""),
                    },
                    "bodies": [],
                    "note_body": 0,
                }
                best[row["path"]] = entry
            entry["lexical"] += float(score)
            entry["matched"].add(term)
            entry["bodies"].append(row["content"])
            entry["note_body"] = note_bodies.get(row["path"], entry["body_chars"])

        for entry in best.values():
            # BM25 is length-normalized, so a whole-title query against a long
            # chunk can score below a short note sharing one token. A phrase hit
            # against a note's *name* is unambiguous evidence and must outrank
            # that noise; a phrase hit inside body text is weaker evidence.
            # Levels: 0 = whole query is (inside) the title/stem;
            # 1 = wrapper-stripped core is; 2 = whole query appears in body.
            struct = {key for key in entry["struct"] if key}
            body_key = phrase_key(" ".join(entry["bodies"])[:8000])
            level = 3
            if full_key and full_key in struct:
                level = 0
            elif full_key and any(core in struct for core in core_keys):
                level = 1
            elif full_key and full_key in body_key:
                level = 2
            entry["phrase_level"] = level
            entry["exact_phrase"] = level <= 1

        ranked = sorted(
            best.values(),
            key=lambda item: (item["phrase_level"], -item["lexical"], item["note"]),
        )
        for rank, entry in enumerate(ranked):
            entry["lexical_rank"] = rank
        return ranked[:CANDIDATE_LIMIT]

    # ---------- semantic ----------

    def _note_bodies(self, connection: sqlite3.Connection) -> dict[str, int]:
        return {
            row["path"]: int(row["n"])
            for row in connection.execute(
                "SELECT path, SUM(body_chars) AS n FROM chunks"
                " WHERE active = 1 GROUP BY path"
            ).fetchall()
        }

    def _generation(self, connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'generation'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def _load_vectors(self, connection: sqlite3.Connection, generation: int) -> None:
        if self._cache["generation"] == generation:
            return
        rows = connection.execute(
            "SELECT e.chunk_id, e.dim, e.vector, c.path"
            " FROM embeddings e JOIN chunks c ON c.id = e.chunk_id AND c.active = 1"
        ).fetchall()
        ids: list[str] = []
        paths: list[str] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if vector.size != row["dim"]:
                continue
            ids.append(row["chunk_id"])
            paths.append(row["path"])
            vectors.append(vector)
        matrix = np.vstack(vectors) if vectors else None
        self._cache = {
            "generation": generation,
            "matrix": matrix,
            "ids": ids,
            "paths": paths,
        }

    def _semantic(
        self,
        connection: sqlite3.Connection,
        generation: int,
        text: str,
        allowed,
    ) -> list[dict]:
        if isinstance(self.embedder, LexicalOnlyEmbedder):
            return []
        self._load_vectors(connection, generation)
        matrix = self._cache["matrix"]
        if matrix is None or not len(self._cache["ids"]):
            return []
        try:
            query_vector = self.embedder.encode_query(text)
        except Exception:  # noqa: BLE001 - degrade to lexical rather than fail the query
            return []
        if query_vector.size != matrix.shape[1]:
            return []
        similarities = matrix @ query_vector
        order = np.argsort(-similarities)
        meta = {
            row["id"]: row
            for row in connection.execute(
                "SELECT id, path, heading, content, token_count, body_chars"
                " FROM chunks WHERE active = 1"
            ).fetchall()
        }
        results: list[dict] = []
        seen: set[str] = set()
        for index in order:
            chunk_id = self._cache["ids"][int(index)]
            row = meta.get(chunk_id)
            if row is None or not allowed(row["path"]) or row["path"] in seen:
                continue
            seen.add(row["path"])
            results.append(
                {
                    "note": row["path"],
                    "heading": row["heading"],
                    "content": row["content"],
                    "token_count": int(row["token_count"]),
                    "body_chars": int(row["body_chars"]),
                    "semantic": float(similarities[int(index)]),
                }
            )
            if len(results) >= CANDIDATE_LIMIT:
                break
        for rank, entry in enumerate(results):
            entry["semantic_rank"] = rank
        return results

    def _apply_semantic_gates(self, semantic: list[dict]) -> tuple[list[dict], list[str]]:
        gates: list[str] = []
        kept: list[dict] = []
        dropped_floor = dropped_tokens = 0
        for entry in semantic:
            if entry["semantic"] < self.semantic_floor:
                dropped_floor += 1
                continue
            if entry["token_count"] < self.semantic_min_tokens:
                dropped_tokens += 1
                continue
            kept.append(entry)
        if dropped_floor:
            gates.append(f"semantic_floor>{self.semantic_floor}:dropped={dropped_floor}")
        if dropped_tokens:
            gates.append(f"min_tokens>={self.semantic_min_tokens}:dropped={dropped_tokens}")
        return kept, gates

    # ---------- fusion ----------

    def _fuse(
        self,
        lexical: list[dict],
        semantic: list[dict],
        allowed,
        query: Query,
        note_bodies: dict[str, int],
    ) -> list[Hit]:
        merged: dict[str, dict] = {}

        def slot(note: str) -> dict:
            entry = merged.get(note)
            if entry is None:
                entry = {
                    "note": note,
                    "heading": None,
                    "content": "",
                    "content_hash": None,
                    "updated_at": None,
                    "lexical": 0.0,
                    "semantic": 0.0,
                    "lexical_rank": None,
                    "semantic_rank": None,
                    "matched": set(),
                    "token_count": 0,
                    "body_chars": 0,
                    "phrase_level": 3,
                }
                merged[note] = entry
            return entry

        for entry in lexical[: self.rrf_depth]:
            target = slot(entry["note"])
            target["lexical"] = entry["lexical"]
            target["lexical_rank"] = entry["lexical_rank"]
            target["matched"] = entry["matched"]
            target["heading"] = entry["heading"]
            target["content"] = entry["content"]
            target["content_hash"] = entry["content_hash"]
            target["updated_at"] = entry["updated_at"]
            target["token_count"] = entry["token_count"]
            target["body_chars"] = entry["body_chars"]
            target["phrase_level"] = entry["phrase_level"]

        for entry in semantic[: self.rrf_depth]:
            target = slot(entry["note"])
            target["semantic"] = entry["semantic"]
            target["semantic_rank"] = entry["semantic_rank"]
            if not target["content"]:
                target["heading"] = entry["heading"]
                target["content"] = entry["content"]
            target["token_count"] = max(target["token_count"], entry["token_count"])
            target["body_chars"] = max(target["body_chars"], entry["body_chars"])

        hits: list[Hit] = []
        for note, entry in merged.items():
            if not read_allowed(note) or not allowed(note):
                continue
            score = 0.0
            if entry["lexical_rank"] is not None:
                score += 1.0 / (self.rrf_k + entry["lexical_rank"] + 1)
            if entry["semantic_rank"] is not None:
                score += 1.0 / (self.rrf_k + entry["semantic_rank"] + 1)
            # The title match must survive RRF. Without this multiplier a
            # wrapper query («关于X的内容») is won by any note clearing the
            # semantic floor with a weak lexical hit — two votes beat the title
            # holder's one, even against a BM25 of 2506. PHRASE_BONUS was defined
            # in v0.2 but never applied; it is load-bearing here. A full
            # two-channel score is 1/61 + 1/62 ≈ 0.0323; a structural-phrase
            # hit must clear it from one channel: 0.0164 × (1 + 1.0) > 0.0323.
            if entry["phrase_level"] <= 1:
                score *= 1.0 + PHRASE_BONUS * 2
            note_body = note_bodies.get(note, entry["body_chars"])
            if note_body < self.stub_body_chars:
                score *= self.stub_penalty
            hits.append(
                Hit(
                    note=note,
                    heading=entry["heading"] or None,
                    excerpt=entry["content"][:900],
                    content_hash=entry["content_hash"],
                    updated_at=entry["updated_at"],
                    score=score,
                    lexical_score=entry["lexical"],
                    semantic_score=entry["semantic"],
                    lexical_rank=entry["lexical_rank"],
                    semantic_rank=entry["semantic_rank"],
                    matched_terms=sorted(entry["matched"]),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.note))
        return hits[: max(1, min(query.limit, 50))]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        from .text import tokenize

        return tokenize(text)


def search_index(
    db_path: Path | str,
    query: str,
    *,
    embedder: Embedder | None = None,
    prefixes: tuple[str, ...] = (),
    include_archive: bool = False,
    limit: int = 8,
    mode: str = "hybrid",
) -> tuple[list[Hit], SearchDiagnostics]:
    searcher = Searcher(db_path, embedder=embedder)
    return searcher.search(
        Query(
            text=query,
            prefixes=prefixes,
            include_archive=include_archive,
            limit=limit,
            mode=mode,
        )
    )
