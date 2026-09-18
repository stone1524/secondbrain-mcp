"""Keyword retrieval over the vault: CJK-bigram tokenization + BM25 + privacy filtering.

Parity with the production retriever (see docs/behavior-inventory.md §3):

- chunking by markdown headings with an 1800-character flush threshold;
- tokens: unicode word terms (length > 1) plus CJK bigrams, deduplicated;
- BM25 (k1=1.2, b=0.75) where idf/avgLength are *global corpus* statistics and
  the length normalization averages over prose rows only — filename-stub rows
  would otherwise crush real chunks' tf weight;
- exact-phrase bonus (+0.5) and per-note best-chunk deduplication;
- hard-excluded zones (private areas, .obsidian, archive, regenerated persona
  views) are unreachable even through explicit scope prefixes — the privacy
  leak probe must always return zero hits;
- auxiliary areas (auto-generated file stubs, agent ops notes) are indexed but
  skipped in the default scope; they surface only when the caller narrows
  into them explicitly;
- a bilingual query glossary may only *add* expansion terms: the original
  wording is always kept, so an over-eager entry can never lose an exact match.

Documented divergence from the production lexicon: chunk tokens fold in the
heading. Production tokenizes chunk body only, which makes heading-only
queries unfindable through the lexical path; keyword-only v0.1 has no
semantic fallback to cover that, so the heading joins the tokens here.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .audit import AuditLog
from .hashing import sha256_file
from .paths import VaultPathError, VaultPaths

_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)
_CJK_TERM_RE = re.compile(r"[\u3400-\u9fff]+")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)")

# Generic defaults only — personal-domain entries must come from user config.
DEFAULT_GLOSSARY: tuple[tuple[str, str], ...] = (
    (r"how often", "frequency 周期 monthly weekly"),
    (r"多久(一次)?|多长时间|频率|周期", "frequency 周期 monthly weekly"),
    (r"清理|cleanup", "cleanup 清理"),
    (r"审核|审查|review", "review 审核 回顾"),
)


def tokenize(text: str) -> list[str]:
    terms = [term for term in _TOKEN_RE.findall(str(text).lower()) if len(term) > 1]
    tokens: set[str] = set()
    for term in terms:
        tokens.add(term)
        if _CJK_TERM_RE.fullmatch(term):
            tokens.update(term[i : i + 2] for i in range(len(term) - 1))
    return sorted(tokens)


def chunks_for(content: str) -> list[dict]:
    chunks: list[dict] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        text = " ".join(body).strip()
        if text:
            chunks.append({"heading": heading, "content": text})
        body.clear()

    for line in content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(1).strip()
        elif line.strip():
            body.append(line.strip())
        if len(" ".join(body)) > 1800:
            flush()
    flush()
    if not chunks and content.strip():
        # Documented divergence: a heading-only note would otherwise produce zero
        # chunks — invisible to retrieval *and* re-indexed by the self-heal pass
        # on every single run (the heuristic never converges for it). Production
        # bug; here the heading itself becomes the chunk.
        chunks.append({"heading": heading, "content": heading})
    return chunks


def _prefix_matches(note: str, prefix: str) -> bool:
    return bool(prefix) and (note == prefix or note.startswith(prefix + "/"))


class Retriever:
    schema_version = 1

    def __init__(
        self,
        paths: VaultPaths,
        db_path: Path | str,
        *,
        audit_log: AuditLog | None = None,
        auxiliary_prefixes: tuple[str, ...] = ("30_Resources/FileIndex", "90_Agent"),
        stub_prefixes: tuple[str, ...] = ("30_Resources/FileIndex",),
        excluded_files: tuple[str, ...] = (
            "40_Persona/Model/audit.md",
            "40_Persona/Model/candidates.md",
            "30_Resources/FileIndex/auto-knowledge-index.md",
        ),
        glossary: tuple[tuple[str, str], ...] = DEFAULT_GLOSSARY,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.paths = paths
        self.db_path = Path(db_path)
        self.audit = audit_log
        self.auxiliary_prefixes = tuple(p.strip("/") for p in auxiliary_prefixes)
        self.stub_prefixes = tuple(p.strip("/") for p in stub_prefixes)
        self.excluded_files = frozenset(excluded_files)
        self.glossary = glossary
        self.k1 = k1
        self.b = b
        self._corpus_cache: dict = {"generation": None, "rows": None, "df": {}, "avg_length": 0.0}

    # ---------- indexing ----------

    def index(self, *, force: bool = False) -> dict:
        db = self._open_db()
        try:
            stored_schema = self._meta(db, "schema_version")
            if stored_schema is not None and int(stored_schema) != self.schema_version:
                force = True
            generation = int(self._meta(db, "index_generation") or 0)
            changed: list[dict] = []
            seen: set[str] = set()
            unchanged = healed = skipped = vanished = 0
            for file in self._walk():
                note = self.paths.relative(file)
                if self._hard_excluded(note):
                    skipped += 1
                    continue
                try:
                    content = file.read_text(encoding="utf-8")
                except FileNotFoundError:
                    # Moved or deleted between the walk and the read: skip without
                    # marking seen so the old path is deactivated below.
                    vanished += 1
                    continue
                current = sha256_file(file)
                existing = db.execute(
                    "SELECT hash, active FROM documents WHERE path=?", (note,)
                ).fetchone()
                if (
                    not force
                    and existing is not None
                    and existing["hash"] == current
                    and existing["active"] == 1
                    and db.execute(
                        "SELECT COUNT(*) FROM chunks WHERE path=? AND active=1", (note,)
                    ).fetchone()[0]
                    > 0
                ):
                    unchanged += 1
                    seen.add(note)
                    continue
                if (
                    not force
                    and existing is not None
                    and existing["hash"] == current
                    and existing["active"] == 1
                ):
                    # Hash matches but no active chunks: an interrupted legacy run.
                    healed += 1
                seen.add(note)
                updated_at = datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat()
                changed.append(
                    {"path": note, "hash": current, "updated_at": updated_at, "content": content}
                )

            pending: list[dict] = []
            for item in changed:
                for chunk in chunks_for(item["content"]):
                    pending.append(
                        {
                            "id": uuid.uuid4().hex,
                            "path": item["path"],
                            "heading": chunk["heading"],
                            "content": chunk["content"],
                            # Heading folded in: see module docstring divergence note.
                            "tokens": tokenize(f"{chunk['heading']}\n{chunk['content']}"),
                        }
                    )

            deactivated = 0
            db.execute("BEGIN IMMEDIATE")
            try:
                for item in changed:
                    db.execute("DELETE FROM chunks WHERE path=?", (item["path"],))
                    db.execute(
                        "INSERT OR REPLACE INTO documents(path,hash,updated_at,active)"
                        " VALUES(?,?,?,1)",
                        (item["path"], item["hash"], item["updated_at"]),
                    )
                for chunk in pending:
                    db.execute(
                        "INSERT INTO chunks(id,path,heading,content,tokens,active)"
                        " VALUES(?,?,?,?,?,1)",
                        (
                            chunk["id"],
                            chunk["path"],
                            chunk["heading"],
                            chunk["content"],
                            json.dumps(chunk["tokens"]),
                        ),
                    )
                for (path,) in db.execute("SELECT path FROM documents WHERE active=1").fetchall():
                    if path not in seen:
                        # Scoped indexing would restrict this by scope; the whole
                        # vault is the scope today, exactly like the production
                        # default run.
                        db.execute("UPDATE documents SET active=0 WHERE path=?", (path,))
                        db.execute("UPDATE chunks SET active=0 WHERE path=?", (path,))
                        deactivated += 1
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                    (str(self.schema_version),),
                )
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('index_generation',?)",
                    (str(generation + 1),),
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
            stats = {
                "schema_version": self.schema_version,
                "indexed": len(changed),
                "unchanged": unchanged,
                "healed": healed,
                "skipped": skipped,
                "vanished": vanished,
                "deactivated": deactivated,
                "chunks": len(pending),
            }
            if self.audit is not None:
                self.audit.record("rag_index", "(vault)", **stats)
            return stats
        finally:
            db.close()

    def status(self) -> dict:
        db = self._open_db()
        try:
            documents = db.execute("SELECT COUNT(*) FROM documents WHERE active=1").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks WHERE active=1").fetchone()[0]
            clauses, params = [], []
            for prefix in self.auxiliary_prefixes:
                clauses.append("path LIKE ? OR path = ?")
                params.extend((f"{prefix}/%", prefix))
            auxiliary = 0
            if clauses:
                auxiliary = db.execute(
                    f"SELECT COUNT(*) FROM documents WHERE active=1 AND ({' OR '.join(clauses)})",
                    params,
                ).fetchone()[0]
            return {
                "schema_version": int(self._meta(db, "schema_version") or self.schema_version),
                "generation": int(self._meta(db, "index_generation") or 0),
                "documents": documents,
                "chunks": chunks,
                "auxiliary_documents": auxiliary,
            }
        finally:
            db.close()

    # ---------- search ----------

    def search(self, query: str, *, paths: list[str] | None = None, limit: int = 8) -> list[dict]:
        prefixes = self._normalize_scope(paths)
        base_tokens = tokenize(query)
        if not base_tokens:
            return []
        expansion = self._expand_query(query)
        query_tokens = list(dict.fromkeys([*base_tokens, *tokenize(expansion)]))

        db = self._open_db()
        try:
            generation = int(self._meta(db, "index_generation") or 0)
            rows = self._corpus(db, generation)
        finally:
            db.close()
        corpus_size = len(rows)
        avg_length = max(self._corpus_cache["avg_length"], 1.0)
        df = self._corpus_cache["df"]

        lowered_query = str(query).lower()
        scoped = [
            row
            for row in rows
            if self._in_scope(row["path"], prefixes) and not self._hard_excluded(row["path"])
        ]
        best: dict[str, dict] = {}
        for row in scoped:
            tokens = row["tokens"]
            counts = {token: tokens.count(token) for token in query_tokens if token in tokens}
            if not counts:
                # No lexical match and (in v0.1) no semantic vector: skip the row,
                # exactly like the production gate before cosine fusion.
                continue
            lexical = 0.5 if lowered_query in row["content"].lower() else 0.0
            for term, tf in counts.items():
                term_df = df.get(term, 0)
                idf = math.log(1 + (corpus_size - term_df + 0.5) / (term_df + 0.5))
                lexical += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * len(tokens) / avg_length)
                )
            previous = best.get(row["path"])
            if previous is None or lexical > previous["lexical_score"]:
                best[row["path"]] = {
                    "source": row["path"],
                    "heading": row["heading"],
                    "excerpt": row["content"][:900],
                    "content_hash": row["hash"],
                    "updated_at": row["updated_at"],
                    "retrieval_score": round(lexical, 4),
                    "lexical_score": round(lexical, 4),
                    "semantic_score": 0.0,
                    "matched_terms": sorted(counts),
                }
        ranked = sorted(
            best.values(), key=lambda item: (-item["retrieval_score"], item["source"])
        )
        results = ranked[: max(1, min(int(limit), 50))]
        if self.audit is not None:
            self.audit.record(
                "rag_search",
                "(vault)",
                scope_decision="default" if not paths else f"scoped:{','.join(prefixes)}",
                hits=len(results),
            )
        return results

    # ---------- internals ----------

    def _open_db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA busy_timeout=5000")
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY, hash TEXT NOT NULL, updated_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, path TEXT NOT NULL, heading TEXT, content TEXT NOT NULL,
                tokens TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
            """
        )
        return db

    def _meta(self, db: sqlite3.Connection, key: str) -> str | None:
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _walk(self) -> list[Path]:
        files: list[Path] = []
        for directory, dirnames, filenames in os.walk(self.paths.root):
            dirnames[:] = [name for name in dirnames if name != ".obsidian"]
            for name in filenames:
                if name.lower().endswith(".md"):
                    files.append(Path(directory) / name)
        return files

    def _hard_excluded(self, note: str) -> bool:
        if self.paths.is_private(note):
            return True
        if ".obsidian" in note.split("/"):
            return True
        if note == "98_Archive" or note.startswith("98_Archive/"):
            return True
        return note in self.excluded_files

    def _is_auxiliary(self, note: str) -> bool:
        return any(_prefix_matches(note, prefix) for prefix in self.auxiliary_prefixes)

    def _in_scope(self, note: str, prefixes: list[str]) -> bool:
        if not any(prefix == "" or _prefix_matches(note, prefix) for prefix in prefixes):
            return False
        if any(_prefix_matches(note, prefix) for prefix in prefixes):
            return True
        return not self._is_auxiliary(note)

    def _normalize_scope(self, paths: list[str] | None) -> list[str]:
        if not paths:
            return [""]
        prefixes = set()
        for item in paths:
            raw = str(item)
            probe = Path(raw)
            # Reject before stripping: a rooted "/abs" would otherwise lose its
            # leading slash and sneak through as a plain prefix.
            if probe.is_absolute() or probe.root or (len(raw) > 1 and raw[1] == ":"):
                raise VaultPathError("paths entries must be vault-relative prefixes")
            cleaned = raw.replace("\\", "/").strip("/")
            if not cleaned or ".." in cleaned.split("/"):
                raise VaultPathError("paths entries must be vault-relative prefixes")
            prefixes.add(cleaned)
        return list(prefixes) or [""]

    def _expand_query(self, query: str) -> str:
        additions = " ".join(
            extra
            for pattern, extra in self.glossary
            if re.search(pattern, query, re.IGNORECASE)
        )
        return additions.strip()

    def _corpus(self, db: sqlite3.Connection, generation: int) -> list[dict]:
        cache = self._corpus_cache
        if cache["generation"] == generation and cache["rows"] is not None:
            return cache["rows"]
        rows = [
            {
                "path": row["path"],
                "heading": row["heading"],
                "content": row["content"],
                "hash": row["hash"],
                "updated_at": row["updated_at"],
                "tokens": json.loads(row["tokens"]),
            }
            for row in db.execute(
                "SELECT chunks.path, heading, content, tokens, documents.hash, documents.updated_at"
                " FROM chunks JOIN documents ON documents.path = chunks.path"
                " WHERE chunks.active=1 AND documents.active=1"
            )
        ]
        df: dict[str, int] = {}
        for row in rows:
            for term in set(row["tokens"]):
                df[term] = df.get(term, 0) + 1
        prose_rows = [
            row
            for row in rows
            if not any(_prefix_matches(row["path"], prefix) for prefix in self.stub_prefixes)
        ]
        avg_length = sum(len(row["tokens"]) for row in prose_rows) / max(len(prose_rows), 1)
        cache.update(generation=generation, rows=rows, df=df, avg_length=avg_length)
        return rows
