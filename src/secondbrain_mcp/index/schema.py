"""Index schema and connection handling.

``schema_version = 2`` is deliberately incompatible with the production v6
database: the two differ in what a row means (v6 has no embed-state and no model
fingerprint), so mixing them would silently corrupt retrieval. A version bump
forces a full rebuild.

Version 2 changed what a postings row *means* — script-split terms, document
titles inside the term set, df per document — without changing any column. An
index built under version 1 returns silently wrong rankings for every note the
hash-check marks unchanged, so those rows must be rebuilt, not kept. Version 3
widens the row's term set again: hyphen segments and the file stem join the
index. Same trap, same cure.

Two columns carry the fixes for observed production defects:

``documents.embed_state``
    The old index marked a document unchanged on hash equality alone, so a run
    that failed to embed left NULL vectors that were never retried — 74 of 228
    live chunks, including every Inbox chunk, were permanently vectorless.
    ``FAILED`` makes the missing work visible and re-indexable.

``embeddings.fingerprint``
    The old index stored only a model *name*. Switching models re-embedded
    changed files and left the rest, producing an index that claimed a
    homogeneity it did not have. The fingerprint covers model, dimension,
    library revision and prefix scheme; a mismatch is refused rather than
    silently averaged.

Every SQL statement in this module is written inline as a literal and every
runtime value is bound as a parameter, so no data can reach query structure.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 3


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def initialize(db_path: Path | str, *, force: bool = False) -> sqlite3.Connection:
    connection = connect(db_path)
    if force:
        connection.executescript(
            """
            DROP TABLE IF EXISTS postings;
            DROP TABLE IF EXISTS embeddings;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS model_meta;
            DROP TABLE IF EXISTS meta;
            """
        )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            updated_at TEXT NOT NULL,
            embed_state TEXT NOT NULL DEFAULT 'LEXICAL_ONLY',
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_documents_active ON documents(active);

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            heading TEXT NOT NULL DEFAULT '',
            ordinal INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            body_chars INTEGER NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
        CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active);

        CREATE TABLE IF NOT EXISTS postings (
            chunk_id TEXT NOT NULL,
            term TEXT NOT NULL,
            tf INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
        CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);

        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_fingerprint ON embeddings(fingerprint);

        CREATE TABLE IF NOT EXISTS model_meta (
            fingerprint TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            dim INTEGER NOT NULL,
            revision TEXT NOT NULL,
            prefix_scheme TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    stored = get_meta(connection, "schema_version")
    if stored is not None and int(stored) != SCHEMA_VERSION:
        connection.close()
        return initialize(db_path, force=True)
    set_meta(connection, "schema_version", str(SCHEMA_VERSION))
    connection.commit()
    return connection


def get_meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM meta WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else None


def set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value)
    )


def bump_generation(connection: sqlite3.Connection) -> int:
    generation = int(get_meta(connection, "generation") or 0) + 1
    set_meta(connection, "generation", str(generation))
    return generation


def status(db_path: Path | str) -> dict:
    connection = initialize(db_path)
    try:
        total_chunks = int(
            connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE active = ?", (1,)
            ).fetchone()["n"]
        )
        embedded = int(
            connection.execute(
                "SELECT COUNT(*) AS n FROM embeddings"
                " WHERE chunk_id IN (SELECT id FROM chunks WHERE active = ?)",
                (1,),
            ).fetchone()["n"]
        )
        documents = int(
            connection.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE active = ?", (1,)
            ).fetchone()["n"]
        )
        documents_embedded = int(
            connection.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE active = ? AND embed_state = ?",
                (1, "EMBEDDED"),
            ).fetchone()["n"]
        )
        documents_failed = int(
            connection.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE active = ? AND embed_state = ?",
                (1, "FAILED"),
            ).fetchone()["n"]
        )
        # Only chunks belonging to a document that *should* carry vectors count as
        # null. Lexical-only documents (file stubs, generated views) legitimately
        # have none, and counting them made the gate impossible to satisfy. A
        # FAILED document counts: its vectors are owed, not waived.
        null_required = int(
            connection.execute(
                "SELECT COUNT(*) AS n FROM chunks c"
                " JOIN documents d ON d.path = c.path"
                " WHERE c.active = 1 AND d.embed_state IN (?, ?)"
                " AND c.id NOT IN (SELECT chunk_id FROM embeddings)",
                ("EMBEDDED", "FAILED"),
            ).fetchone()["n"]
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "generation": int(get_meta(connection, "generation") or 0),
            "documents": documents,
            "chunks": total_chunks,
            "embedded_chunks": embedded,
            "null_embedding_chunks": null_required,
            "lexical_only_chunks": max(0, total_chunks - embedded - null_required),
            "documents_embedded": documents_embedded,
            "documents_failed_embedding": documents_failed,
            "db_path": str(db_path),
        }
    finally:
        connection.close()
