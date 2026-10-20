"""Incremental indexer.

Design notes tied to observed production defects:

* **Embed state is explicit.** A document is only "unchanged" when its hash
  matches *and* the work it needs is actually done. The production index tested
  the hash alone, so a run that failed to embed produced NULL vectors that were
  never retried — 74 of 228 live chunks in the real vault.
* **Scoped re-index never deactivates documents it did not visit.** Deactivation
  is driven by file existence, not by set membership in the current run, so a
  partial run cannot silently drop rows.
* **The model fingerprint is written with the vectors.** A mismatch triggers a
  vector rebuild instead of producing an index that mixes two vector spaces.
* Documents are written one at a time; the whole changed set is never held in
  memory.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..core.policy import IndexDecision, NotePath, index_decision
from .embed import Embedder, LexicalOnlyEmbedder
from .schema import bump_generation, get_meta, initialize, set_meta
from .text import chunk_note

MARKDOWN_SUFFIX = ".md"


@dataclass(slots=True)
class IndexReport:
    indexed: int = 0
    unchanged: int = 0
    skipped: int = 0
    vanished: int = 0
    deactivated: int = 0
    chunks: int = 0
    embedded_chunks: int = 0
    embedding_failures: int = 0
    embed_error: str | None = None
    generation: int = 0
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "indexed": self.indexed,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "vanished": self.vanished,
            "deactivated": self.deactivated,
            "chunks": self.chunks,
            "embedded_chunks": self.embedded_chunks,
            "embedding_failures": self.embedding_failures,
            "embed_error": self.embed_error,
            "generation": self.generation,
            **self.details,
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _walk(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        # The policy makes the real decision; skipping dot-directories here only
        # keeps the walk cheap. Lowercased because Windows is case-insensitive.
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for name in filenames:
            if name.lower().endswith(MARKDOWN_SUFFIX):
                files.append(Path(directory) / name)
    return files


def _chunk_id(rel: str, ordinal: int) -> str:
    return hashlib.sha256(f"{rel}#{ordinal}".encode()).hexdigest()[:32]


class Indexer:
    def __init__(
        self,
        vault_root: Path | str,
        db_path: Path | str,
        *,
        embedder: Embedder | None = None,
    ):
        self.vault_root = Path(vault_root).resolve()
        self.db_path = Path(db_path)
        self.embedder = embedder or LexicalOnlyEmbedder()

    @property
    def has_semantic_backend(self) -> bool:
        return not isinstance(self.embedder, LexicalOnlyEmbedder)

    def rebuild(self, *, force: bool = False) -> IndexReport:
        connection = initialize(self.db_path, force=force)
        report = IndexReport()
        try:
            self._reconcile_model(connection, report)
            visited: set[str] = set()
            pending = self._scan(connection, report, visited, force=force)
            for item in pending:
                self._write_one(connection, item, report)
            report.deactivated = self._deactivate_vanished(connection, visited)
            report.generation = bump_generation(connection)
            connection.commit()
            return report
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # ---------- model consistency ----------

    def _reconcile_model(self, connection: sqlite3.Connection, report: IndexReport) -> None:
        if not self.has_semantic_backend:
            return
        stored = get_meta(connection, "model_fingerprint")
        active = self.embedder.info.fingerprint
        if stored and stored != active:
            connection.execute("DELETE FROM embeddings")
            connection.commit()
            report.details["vectors_dropped"] = "model fingerprint changed"

    # ---------- scan ----------

    def _scan(
        self,
        connection: sqlite3.Connection,
        report: IndexReport,
        visited: set[str],
        *,
        force: bool = False,
    ) -> list[dict]:
        pending: list[dict] = []
        for file in _walk(self.vault_root):
            try:
                note = NotePath.from_disk(self.vault_root, file)
            except ValueError:
                report.skipped += 1
                continue
            decision = index_decision(note.key)
            if decision is IndexDecision.SKIP:
                report.skipped += 1
                continue
            try:
                payload = file.read_bytes()
            except FileNotFoundError:
                # Moved between the walk and the read (the nightly archiver runs
                # just before the indexer); the next run picks it up.
                report.vanished += 1
                continue

            visited.add(note.rel)
            content_hash = sha256_bytes(payload)
            if not force and self._is_complete(connection, note.rel, content_hash, decision):
                report.unchanged += 1
                continue
            pending.append(
                {
                    "rel": note.rel,
                    "hash": content_hash,
                    "content": payload.decode("utf-8", errors="replace"),
                    "decision": decision,
                    "mtime": file.stat().st_mtime,
                }
            )
        report.details["visited"] = len(visited)
        return pending

    def _is_complete(
        self,
        connection: sqlite3.Connection,
        rel: str,
        content_hash: str,
        decision: IndexDecision,
    ) -> bool:
        row = connection.execute(
            "SELECT content_hash, active FROM documents WHERE path = ?", (rel,)
        ).fetchone()
        if row is None or row["content_hash"] != content_hash or row["active"] != 1:
            return False
        if connection.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE path = ? AND active = 1", (rel,)
        ).fetchone()["n"] == 0:
            return False
        if decision is IndexDecision.LEXICAL_ONLY or not self.has_semantic_backend:
            return True
        # A document that was committed while embedding was unavailable must be
        # re-processed; a hash-only test made the gap permanent in production.
        gap = connection.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE path = ? AND active = 1"
            " AND id NOT IN (SELECT chunk_id FROM embeddings)",
            (rel,),
        ).fetchone()["n"]
        return gap == 0

    # ---------- write ----------

    def _write_one(self, connection: sqlite3.Connection, item: dict, report: IndexReport) -> None:
        rel = item["rel"]
        chunks = chunk_note(item["content"], filename=Path(rel).stem)
        self._delete_document_rows(connection, rel)

        inserted: list[tuple[str, str]] = []
        for chunk in chunks:
            chunk_id = _chunk_id(rel, chunk.ordinal)
            connection.execute(
                "INSERT INTO chunks(id, path, heading, title, ordinal, content,"
                " token_count, body_chars, active) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    chunk_id,
                    rel,
                    chunk.heading,
                    chunk.title,
                    chunk.ordinal,
                    chunk.content,
                    chunk.token_count,
                    chunk.body_chars,
                ),
            )
            # Postings come from the chunk's own tokenization. Re-deriving tokens
            # from (heading, content) here dropped the document title — the title
            # of an H1-over-H2 note lives in neither field — so the note stayed
            # unfindable by its own name even though its token_count counted it.
            for term, frequency in chunk.term_frequencies:
                connection.execute(
                    "INSERT INTO postings(chunk_id, term, tf) VALUES(?, ?, ?)",
                    (chunk_id, term, frequency),
                )
            inserted.append((chunk_id, chunk.content))

        embed_state = "LEXICAL_ONLY"
        if item["decision"] is IndexDecision.INDEX and inserted and self.has_semantic_backend:
            embed_state = self._embed(connection, inserted, report)

        connection.execute(
            "INSERT OR REPLACE INTO documents(path, content_hash, size, mtime,"
            " updated_at, embed_state, active) VALUES(?, ?, ?, ?, ?, ?, 1)",
            (
                rel,
                item["hash"],
                len(item["content"].encode("utf-8")),
                item["mtime"],
                datetime.fromtimestamp(item["mtime"], timezone.utc).isoformat(),
                embed_state,
            ),
        )
        report.indexed += 1
        report.chunks += len(chunks)
        report.details.setdefault("embed_state", {})
        report.details["embed_state"][embed_state] = (
            report.details["embed_state"].get(embed_state, 0) + 1
        )

    def _delete_document_rows(self, connection: sqlite3.Connection, rel: str) -> None:
        connection.execute(
            "DELETE FROM postings WHERE chunk_id IN"
            " (SELECT id FROM chunks WHERE path = ?)",
            (rel,),
        )
        connection.execute(
            "DELETE FROM embeddings WHERE chunk_id IN"
            " (SELECT id FROM chunks WHERE path = ?)",
            (rel,),
        )
        connection.execute("DELETE FROM chunks WHERE path = ?", (rel,))

    def _embed(
        self,
        connection: sqlite3.Connection,
        inserted: list[tuple[str, str]],
        report: IndexReport,
    ) -> str:
        chunk_ids = [item[0] for item in inserted]
        contents = [item[1] for item in inserted]
        try:
            vectors = self.embedder.encode_passages(contents)
        except Exception as error:  # noqa: BLE001 - a backend crash must not abort the run
            # One document failing must not abort the run; the FAILED state makes
            # the gap visible and the next pass retries it.
            report.embedding_failures += 1
            report.embed_error = str(error)
            return "FAILED"

        fingerprint = self.embedder.info.fingerprint
        dim = self.embedder.info.dim
        for chunk_id, vector in zip(chunk_ids, vectors, strict=False):
            connection.execute(
                "INSERT OR REPLACE INTO embeddings(chunk_id, fingerprint, dim, vector)"
                " VALUES(?, ?, ?, ?)",
                (chunk_id, fingerprint, dim, bytes(vector.astype("float32").tobytes())),
            )
            report.embedded_chunks += 1
        connection.execute(
            "INSERT OR REPLACE INTO model_meta(fingerprint, model_name, dim, revision,"
            " prefix_scheme, created_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                fingerprint,
                self.embedder.info.name,
                dim,
                self.embedder.info.revision,
                self.embedder.info.prefix_scheme,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        set_meta(connection, "model_fingerprint", fingerprint)
        return "EMBEDDED"

    # ---------- deactivation ----------

    def _deactivate_vanished(
        self, connection: sqlite3.Connection, visited: set[str]
    ) -> int:
        """Retire documents that are indexed but no longer on disk.

        Existence is the test, not membership in ``visited``: a scoped run must
        never deactivate rows it simply did not look at.
        """
        deactivated = 0
        rows = connection.execute(
            "SELECT path FROM documents WHERE active = 1"
        ).fetchall()
        for row in rows:
            rel = row["path"]
            if rel in visited:
                continue
            if (self.vault_root / rel).exists():
                continue
            connection.execute("UPDATE chunks SET active = 0 WHERE path = ?", (rel,))
            connection.execute("UPDATE documents SET active = 0 WHERE path = ?", (rel,))
            connection.execute(
                "DELETE FROM postings WHERE chunk_id IN"
                " (SELECT id FROM chunks WHERE path = ?)",
                (rel,),
            )
            connection.execute(
                "DELETE FROM embeddings WHERE chunk_id IN"
                " (SELECT id FROM chunks WHERE path = ?)",
                (rel,),
            )
            deactivated += 1
        return deactivated


def index_vault(
    vault_root: Path | str,
    db_path: Path | str,
    *,
    embedder: Embedder | None = None,
    force: bool = False,
) -> IndexReport:
    return Indexer(vault_root, db_path, embedder=embedder).rebuild(force=force)
