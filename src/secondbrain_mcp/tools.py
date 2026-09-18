"""Gateway tool surface: the operations an MCP client can invoke.

Plain methods on :class:`Gateway` (unit-testable without a wire); server.py
registers them onto the MCP app. Every tool mirrors the production gateway's
contract — see docs/behavior-inventory.md.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .archive import Archiver
from .audit import AuditLog
from .hashing import ensure_hash_matches, sha256_file
from .locking import FileLock
from .paths import VaultPaths
from .retrieval import Retriever

INBOX_FOLDERS = ("Web", "Conversations", "Files")
METADATA_FIELDS = frozenset(
    {"status", "tags", "updated_at", "agent_generated", "confidence", "state", "last_verified"}
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Gateway:
    def __init__(
        self,
        vault_root: Path | str,
        *,
        db_path: Path | str | None = None,
        agent_label: str | None = None,
        log_dir: Path | str | None = None,
    ):
        self.paths = VaultPaths(vault_root)
        state_dir = Path(db_path).parent if db_path else self.paths.root / ".secondbrain"
        self.db_path = Path(db_path) if db_path else state_dir / "index.db"
        log_dir = Path(log_dir) if log_dir else state_dir / "logs"
        self.audit = AuditLog(
            log_dir / "gateway-audit.jsonl", agent_label=agent_label or os.environ.get("SB_AGENT_LABEL")
        )
        self.retriever = Retriever(self.paths, self.db_path, audit_log=self.audit)
        self.archiver = Archiver(self.paths, agent_label=self.audit.agent_label)
        self.lock_dir = state_dir / "locks"

    # ---------- read ----------

    def read_note(self, note: str, offset: int = 0, max_length: int = 50000) -> dict:
        file = self.paths.resolve(note)
        self.paths.assert_readable(file)
        self.audit.record("read_note", self.paths.relative(file))
        content = file.read_text(encoding="utf-8")
        total = len(content)
        start = max(0, min(int(offset), total))
        window = min(max(1, int(max_length)), 200000)
        visible = content[start : start + window]
        return {
            "note": self.paths.relative(file),
            "content": visible,
            "offset": start,
            "total_length": total,
            "truncated": start + len(visible) < total,
            "hash": ensure_hash_matches(file, None),
        }

    def list_notes(self, folder: str | None = None, pattern: str | None = None, limit: int = 100) -> dict:
        base = self.paths.resolve(folder) if folder else self.paths.root
        self.paths.assert_readable(base)
        needle = pattern.lower() if pattern else None
        notes = []
        for directory, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if name != ".obsidian"]
            for name in filenames:
                if not name.lower().endswith(".md"):
                    continue
                file = Path(directory) / name
                if self.paths.is_private(file):
                    continue
                rel = self.paths.relative(file)
                if needle and needle not in rel.lower():
                    continue
                stat = file.stat()
                notes.append(
                    {"note": rel, "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "size": stat.st_size}
                )
        notes.sort(key=lambda item: item["updated_at"], reverse=True)
        result = {
            "folder": self.paths.relative(base) if base != self.paths.root else "",
            "total": len(notes),
            "count": min(len(notes), max(1, int(limit))),
            "notes": notes[: max(1, int(limit))],
        }
        self.audit.record("list_notes", result["folder"], returned=result["count"])
        return result

    # ---------- retrieval ----------

    def search_notes(self, query: str, paths: list[str] | None = None, limit: int = 8) -> dict:
        """Keyword search; falls back to a linear vault scan while the index is empty."""
        if self.retriever.status()["documents"] > 0:
            hits = self.retriever.search(query, paths=paths, limit=limit)
            return {
                "mode": "index-lexical",
                "hits": [
                    {"note": hit["source"], "excerpt": hit["excerpt"][:300], "matched_terms": hit["matched_terms"]}
                    for hit in hits
                ],
            }
        needle = str(query).lower()
        found: list[dict] = []
        for directory, dirnames, filenames in os.walk(self.paths.root):
            dirnames[:] = [name for name in dirnames if name != ".obsidian"]
            for name in filenames:
                if not name.lower().endswith(".md"):
                    continue
                file = Path(directory) / name
                if self.paths.is_private(file):
                    continue
                try:
                    content = file.read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                at = content.lower().find(needle)
                if at >= 0:
                    found.append(
                        {
                            "note": self.paths.relative(file),
                            "excerpt": content[max(0, at - 80) : at + len(needle) + 160],
                        }
                    )
                if len(found) >= max(1, int(limit)):
                    break
            if len(found) >= max(1, int(limit)):
                break
        self.audit.record("search_notes", "(vault)", mode="linear-fallback", hits=len(found))
        return {"mode": "linear-fallback", "hits": found}

    def rag_search(self, query: str, paths: list[str] | None = None, limit: int = 8) -> dict:
        hits = self.retriever.search(query, paths=paths, limit=limit)
        return {
            "query": query,
            "hits": hits,
            "requires_read_note_for_full_evidence": True,
            "fallback": None if hits else "Use search_notes, or pass paths to include auxiliary areas",
        }

    def rag_reindex(self, force: bool = False) -> dict:
        return self.retriever.index(force=force)

    def rag_status(self) -> dict:
        return self.retriever.status()

    # ---------- write ----------

    def create_inbox_note(self, folder: str, filename: str, content: str) -> dict:
        if folder not in INBOX_FOLDERS:
            raise ValueError(f"folder must be one of {INBOX_FOLDERS}")
        safe_name = Path(filename.replace("\\", "/")).name
        if not safe_name.lower().endswith(".md"):
            raise ValueError("Markdown filename required")
        file = self.paths.root / "00_Inbox" / folder / safe_name
        self.audit.record("create_inbox_note", self.paths.relative(file))
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ValueError("Note already exists; inbox creation never overwrites") from error
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
        return {"created": self.paths.relative(file)}

    def append_note(self, note: str, content: str, expected_hash: str | None = None) -> dict:
        file = self.paths.resolve(note)
        self.paths.assert_readable(file)
        self.audit.record("append_note", self.paths.relative(file))
        with FileLock(self.lock_dir, file, relative_name=self.paths.relative(file)):
            ensure_hash_matches(file, expected_hash)
            with file.open("a", encoding="utf-8") as stream:
                stream.write(f"\n{content}\n")
            return {"appended": self.paths.relative(file), "hash": sha256_file(file)}

    def update_metadata(self, note: str, fields: dict, expected_hash: str | None = None) -> dict:
        file = self.paths.resolve(note)
        self.paths.assert_readable(file)
        self.audit.record("update_metadata", self.paths.relative(file))
        allowed = {key: value for key, value in fields.items() if key in METADATA_FIELDS}
        dropped = sorted(set(fields) - set(allowed))
        with FileLock(self.lock_dir, file, relative_name=self.paths.relative(file)):
            ensure_hash_matches(file, expected_hash)
            content = file.read_text(encoding="utf-8")
            if not content.startswith("---"):
                raise ValueError("Note has no frontmatter")
            end = content.index("\n---", 3)
            front = content[:end]
            for key, value in allowed.items():
                line = f"{key}: {json.dumps(value, ensure_ascii=False)}"
                pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
                front = pattern.sub(line, front) if pattern.search(front) else f"{front}\n{line}"
            file.write_text(f"{front}{content[end:]}", encoding="utf-8")
            return {"updated": self.paths.relative(file), "fields": allowed, "dropped": dropped, "hash": sha256_file(file)}

    def move_note(self, note: str, destination: str, expected_hash: str | None = None) -> dict:
        file = self.paths.resolve(note)
        self.paths.assert_readable(file)
        target = self.paths.resolve(destination)
        self.paths.assert_move_destination(target)
        self.audit.record("move_note", self.paths.relative(file), destination=self.paths.relative(target))
        with FileLock(self.lock_dir, file, relative_name=self.paths.relative(file)):
            ensure_hash_matches(file, expected_hash)
            if target.exists():
                raise ValueError("Destination already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(file, target)
        return {"moved": self.paths.relative(file), "destination": self.paths.relative(target)}

    # ---------- archive / restore ----------

    def archive_notes(self, entries: list[dict], dry_run: bool = False) -> dict:
        requests = []
        for entry in entries:
            requests.append((entry["note"], entry["hash"], entry.get("category", "")))
        result = self.archiver.archive_notes(requests, dry_run=dry_run)
        if result["batch_id"]:
            result["rollback_hint"] = f"restore_archive(batch_id='{result['batch_id']}')"
        return result

    def list_archives(self, limit: int = 30) -> dict:
        return {"manifests": self.archiver.list_batches(limit=limit)}

    def restore_archive(self, batch_id: str = "latest", dry_run: bool = False, only: str | None = None) -> dict:
        return self.archiver.restore_batch(batch_id, dry_run=dry_run, only=only)
