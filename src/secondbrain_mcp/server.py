"""MCP stdio server shell (official MCP Python SDK).

Exposes the gateway as MCP tools over stdio. Configuration via environment:

- ``SB_VAULT_ROOT``: Markdown vault directory. Defaults to the bundled
  synthetic demo vault so a fresh install has something safe to serve.
- ``SB_RAG_DB``: index database path (default ``<vault>/.secondbrain/index.db``).
- ``SB_AGENT_LABEL``: caller label written into the audit log (e.g. ``claude``,
  ``cursor``, ``zcode``) so multi-client writes stay attributable.

Run: ``secondbrain-mcp`` or ``python -m secondbrain_mcp.server``.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .tools import Gateway


def build_gateway() -> Gateway:
    vault = os.environ.get("SB_VAULT_ROOT")
    if not vault:
        bundled = Path(__file__).resolve().parents[2] / "demo-vault"
        if bundled.exists():
            vault = str(bundled)
    if not vault:
        raise RuntimeError("Set SB_VAULT_ROOT to your Markdown vault directory")
    return Gateway(vault, db_path=os.environ.get("SB_RAG_DB") or None)


gw = build_gateway()
mcp = MCPServer(
    name="secondbrain",
    instructions=(
        "A local-first Markdown knowledge base with safety-first writes: "
        "hash-locked updates, no-delete archiving with rollback manifests, "
        "privacy zones excluded from every read path. Excerpts are evidence "
        "candidates only - call read_note before relying on them."
    ),
)


@mcp.tool()
def read_note(note: str, offset: int = 0, max_length: int = 50000) -> dict:
    """Read a standard note by vault-relative path; returns the full-content hash. Private notes are never readable."""
    return gw.read_note(note, offset=offset, max_length=max_length)


@mcp.tool()
def list_notes(folder: str | None = None, pattern: str | None = None, limit: int = 100) -> dict:
    """List note paths (private notes excluded), newest first; optional folder prefix and substring pattern."""
    return gw.list_notes(folder=folder, pattern=pattern, limit=limit)


@mcp.tool()
def search_notes(query: str, paths: list[str] | None = None, limit: int = 8) -> dict:
    """Keyword search over indexed notes; falls back to a linear vault scan when the index is empty."""
    return gw.search_notes(query, paths=paths, limit=limit)


@mcp.tool()
def rag_search(query: str, paths: list[str] | None = None, limit: int = 8) -> dict:
    """Retrieval with per-note scores and traceability fields. Excerpts are candidates: call read_note before citing."""
    return gw.rag_search(query, paths=paths, limit=limit)


@mcp.tool()
def rag_reindex(force: bool = False) -> dict:
    """Incrementally re-index the vault into a rebuildable derived index."""
    return gw.rag_reindex(force=force)


@mcp.tool()
def rag_status() -> dict:
    """Index health metadata (document/chunk counts, schema version) without exposing note contents."""
    return gw.rag_status()


@mcp.tool()
def create_inbox_note(folder: str, filename: str, content: str) -> dict:
    """Create a note in an inbox directory (Web/Conversations/Files). Never overwrites an existing file."""
    return gw.create_inbox_note(folder, filename, content)


@mcp.tool()
def append_note(note: str, content: str, expected_hash: str | None = None) -> dict:
    """Append to a note under a per-file lock with optimistic-concurrency check on expected_hash."""
    return gw.append_note(note, content, expected_hash=expected_hash)


@mcp.tool()
def update_metadata(note: str, fields: dict, expected_hash: str | None = None) -> dict:
    """Update an allowlisted set of frontmatter fields (status/tags/agent_generated/...); other keys are dropped."""
    return gw.update_metadata(note, fields, expected_hash=expected_hash)


@mcp.tool()
def move_note(note: str, destination: str, expected_hash: str | None = None) -> dict:
    """Move a note within the vault (never into a private area); destination must not exist."""
    return gw.move_note(note, destination, expected_hash=expected_hash)


@mcp.tool()
def archive_notes(entries: list[dict], dry_run: bool = False) -> dict:
    """Archive notes by (note, hash, category) entries into the review area with a batch rollback manifest. Run dry_run first."""
    return gw.archive_notes(entries, dry_run=dry_run)


@mcp.tool()
def list_archives(limit: int = 30) -> dict:
    """List archive batches with their restore accounting."""
    return gw.list_archives(limit=limit)


@mcp.tool()
def restore_archive(batch_id: str = "latest", dry_run: bool = False, only: str | None = None) -> dict:
    """Restore an archived batch back to its original paths; skips already-restored, missing, altered or occupied entries."""
    return gw.restore_archive(batch_id, dry_run=dry_run, only=only)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
