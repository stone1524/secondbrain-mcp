"""Archive and restore: no-delete removal with a rollback manifest.

The v0.1 archiver's restore trusted manifest paths verbatim: ``root / entry["to"]``
silently discards ``root`` when ``to`` is absolute, and ``..`` segments walk out
of the vault. A manifest lives in a writable directory, so it is untrusted
input. Every entry path is re-parsed through :class:`NotePath` here, and the
archived side must resolve inside the archive directory.

The other production gap: ``auto-archive`` moved files first and wrote the
manifest afterwards, so a crash between the two produced moves nothing could
roll back. The manifest is written before the moves, in a ``pending`` state, and
finalised afterwards.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..core.atomic import write_atomic
from ..core.audit import AuditLog
from ..core.errors import HashRequired, InvalidArguments, NoteNotFound, PathDenied
from ..core.locking import FileLock, LockConflictError
from ..core.policy import NotePath, PrivateNoteError, VaultPathError, read_allowed

MANIFEST_SCHEMA_VERSION = 1


@dataclass(slots=True)
class ArchiveService:
    root: Path
    audit: AuditLog
    archive_dirname: str = "98_Archive"
    review_subdir: str = "Review"
    lock_dir: Path | None = None
    stale_lock_seconds: float = 300.0

    @property
    def archive_root(self) -> Path:
        return self.root / self.archive_dirname

    @property
    def manifest_root(self) -> Path:
        return self.archive_root / ".manifests"

    # ---------- archive ----------

    def archive(
        self,
        requests: list[dict],
        *,
        dry_run: bool = False,
        category: str = "",
    ) -> dict:
        if not requests:
            raise InvalidArguments("notes[] is required", "Pass at least one note with its hash")
        results: list[dict] = []
        planned: list[dict] = []

        for request in requests:
            note = str(request.get("note", ""))
            expected = request.get("hash")
            try:
                if not expected:
                    # Archiving is a mutation: the hash is what proves the caller
                    # saw the content it is about to remove from the live tree.
                    # The production archiver treated it as optional, so a
                    # caller could archive a file it had never read.
                    raise HashRequired()
                path = self._existing(note)
                category_name = str(request.get("category") or category or "manual")
                if path.key.startswith(NotePath.fold(f"{self.archive_dirname}/")):
                    raise InvalidArguments("note is already inside the archive area")
                destination = self._destination(path, category_name)
                entry = {
                    "from": path.rel,
                    "to": destination.relative_to(self.root).as_posix(),
                    "hash": expected,
                    "category": category_name,
                }
                # Re-verify under no lock but immediately before use; the strict
                # check happens inside the lock below.
                if self._digest(path.absolute) != expected:
                    results.append(
                        {
                            "note": note,
                            "status": "skipped",
                            "reason": "content changed since it was read",
                        }
                    )
                    continue
                if dry_run:
                    results.append(
                        {
                            "note": note,
                            "status": "would_archive",
                            "destination": entry["to"],
                            "category": category_name,
                        }
                    )
                    continue
                planned.append(entry)
            except (NoteNotFound, InvalidArguments, PathDenied, HashRequired) as error:
                results.append({"note": note, "status": "rejected", "reason": str(error)})

        if dry_run or not planned:
            self.audit.record(
                "archive",
                "(batch)",
                phase="outcome",
                ok=True,
                dry_run=dry_run,
                requested=len(requests),
                planned=len(planned),
            )
            return {
                "dry_run": dry_run,
                "archived": 0,
                "planned": len(planned),
                "manifest": None,
                "results": results,
            }

        batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:4]}"
        manifest_path = self.manifest_root / f"{batch_id}.json"
        # Written first, in `pending` state: if the process dies mid-move the
        # manifest already names every intended move.
        write_atomic(
            manifest_path,
            json.dumps(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "batch_id": batch_id,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "source": "gateway",
                    "agent": self.audit.agent_label,
                    "state": "pending",
                    "entries": planned,
                    "restores": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        moved: list[dict] = []
        for entry in planned:
            path = NotePath.parse(self.root, entry["from"])
            try:
                destination = NotePath.parse(self.root, entry["to"])
                if destination.absolute.exists():
                    raise InvalidArguments("archive destination already exists")
                # The lock and the in-lock re-verification are what make the move
                # atomic against a concurrent write. Without them a writer that
                # had already passed its own hash check could re-create the
                # source path right after the move, leaving the note present in
                # both places — archived and alive at once.
                lock = (
                    FileLock(
                        self.lock_dir,
                        path.absolute,
                        relative_name=path.rel,
                        stale_after_seconds=self.stale_lock_seconds,
                    )
                    if self.lock_dir is not None
                    else _null_lock()
                )
                with lock:
                    if self._digest(path.absolute) != entry["hash"]:
                        results.append(
                            {
                                "note": entry["from"],
                                "status": "skipped",
                                "reason": "content changed since it was read",
                            }
                        )
                        continue
                    destination.absolute.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(path.absolute, destination.absolute)
                moved.append({**entry, "size": destination.absolute.stat().st_size})
                results.append(
                    {
                        "note": entry["from"],
                        "status": "archived",
                        "destination": entry["to"],
                    }
                )
            except (OSError, InvalidArguments, VaultPathError, LockConflictError) as error:
                results.append({"note": entry["from"], "status": "skipped", "reason": str(error)})

        _rewrite_manifest(
            manifest_path,
            state="complete" if moved else "empty",
            entries=moved,
        )
        self.audit.record(
            "archive",
            "(batch)",
            phase="outcome",
            ok=True,
            batch_id=batch_id,
            archived=len(moved),
        )
        return {
            "dry_run": False,
            "archived": len(moved),
            "manifest": batch_id,
            "rollback": f"restore(batch_id='{batch_id}')",
            "results": results,
        }

    # ---------- restore ----------

    def restore(
        self, batch_id: str = "latest", *, dry_run: bool = False, only: str | None = None
    ) -> dict:
        manifest_path = self._resolve_manifest(batch_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The Node auto-archiver's manifests carry no schema_version at all.
        # Requiring the stamp refused every one of them — the nightly job
        # archived, and the no-delete guarantee's rollback half silently died
        # (10 of 10 live manifests were un-restorable). An absent stamp means
        # the legacy shape, which is identical here: entries are {from, to,
        # hash}, and both sides are re-validated as untrusted regardless.
        # The gate guards *interpretation*: a stamp from an unknown future
        # format is still refused.
        schema_version = manifest.get("schema_version", MANIFEST_SCHEMA_VERSION)
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise InvalidArguments(
                f"Unsupported manifest schema_version: {schema_version}",
                "Manifests from an older gateway cannot be restored safely",
            )
        already = {
            note
            for restore in manifest.get("restores", [])
            for note in restore.get("notes", [])
        }
        results: list[dict] = []
        restored: list[str] = []

        for entry in manifest.get("entries", []):
            source_raw = entry.get("from")
            archive_raw = entry.get("to")
            if only and only not in str(source_raw) + str(archive_raw):
                continue
            try:
                # Both sides are untrusted: a manifest is a file on disk that any
                # process with vault access could have written.
                if not NotePath.is_manifest_safe(source_raw) or not NotePath.is_manifest_safe(
                    archive_raw
                ):
                    raise PathDenied("manifest entry is not a vault-relative path")
                original = NotePath.parse(self.root, str(source_raw))
                archived = NotePath.parse(self.root, str(archive_raw))
                # A restore writes the original path back, so the destination
                # obeys the same writability rule as any other write. Without
                # this, a crafted manifest could place a file inside 95_Private
                # using nothing but the gateway's own tools — the private zone
                # was unreadable but still writable through this side door.
                try:
                    original.assert_writable()
                except PrivateNoteError as error:
                    raise PathDenied(
                        "restore destination is inside a non-writable zone"
                    ) from error
                archive_prefix = NotePath.fold(f"{self.archive_dirname}/")
                if not archived.key.startswith(archive_prefix):
                    raise PathDenied("archived path is outside the archive directory")
                if original.key.startswith(archive_prefix):
                    raise PathDenied("restore destination is inside the archive")
                if original.rel in already:
                    results.append(
                        {"note": original.rel, "status": "skipped", "reason": "already restored"}
                    )
                    continue
                if not archived.absolute.exists():
                    results.append(
                        {"note": original.rel, "status": "skipped", "reason": "archived file missing"}
                    )
                    continue
                expected = entry.get("hash")
                if expected and self._digest(archived.absolute) != expected:
                    results.append(
                        {
                            "note": original.rel,
                            "status": "skipped",
                            "reason": "archived content changed after archiving",
                        }
                    )
                    continue
                if original.absolute.exists():
                    results.append(
                        {"note": original.rel, "status": "skipped", "reason": "original path occupied"}
                    )
                    continue
                if dry_run:
                    results.append({"note": original.rel, "status": "would_restore"})
                    continue
                original.absolute.parent.mkdir(parents=True, exist_ok=True)
                os.replace(archived.absolute, original.absolute)
                restored.append(original.rel)
                results.append({"note": original.rel, "status": "restored"})
            except (VaultPathError, PathDenied, OSError) as error:
                results.append(
                    {"note": str(source_raw), "status": "error", "reason": str(error)}
                )

        if restored and not dry_run:
            manifest.setdefault("restores", []).append(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "count": len(restored),
                    "notes": restored,
                }
            )
            write_atomic(
                manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2)
            )
        self.audit.record(
            "restore",
            "(batch)",
            phase="outcome",
            ok=True,
            batch_id=batch_id,
            restored=len(restored),
            dry_run=dry_run,
        )
        return {
            "batch_id": manifest.get("batch_id"),
            "dry_run": dry_run,
            "restored": len(restored),
            "results": results,
        }

    # ---------- manifest listing ----------

    def list_batches(self, limit: int = 30) -> list[dict]:
        if not self.manifest_root.exists():
            return []
        batches: list[dict] = []
        for file in sorted(self.manifest_root.glob("batch-*.json"), reverse=True)[:limit]:
            try:
                manifest = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            batches.append(
                {
                    "batch_id": manifest.get("batch_id", file.stem),
                    "at": manifest.get("at"),
                    "state": manifest.get("state"),
                    "entries": len(manifest.get("entries", [])),
                    "restored": sum(
                        item.get("count", 0) for item in manifest.get("restores", [])
                    ),
                }
            )
        return batches

    # ---------- helpers ----------

    def _resolve_manifest(self, batch_id: str) -> Path:
        if not self.manifest_root.exists():
            raise InvalidArguments("No archive manifests exist yet")
        if batch_id == "latest":
            candidates = sorted(self.manifest_root.glob("batch-*.json"), reverse=True)
            if not candidates:
                raise InvalidArguments("No archive manifests exist yet")
            return candidates[0]
        # The id is an identifier, not a path: reject anything that could escape.
        if "/" in batch_id or "\\" in batch_id or ".." in batch_id:
            raise InvalidArguments("batch_id must be a plain identifier")
        candidate = self.manifest_root / f"{batch_id}.json"
        if not candidate.exists():
            raise InvalidArguments(f"Batch not found: {batch_id}")
        return candidate

    def _existing(self, note: str) -> NotePath:
        try:
            path = NotePath.parse(self.root, note)
        except VaultPathError as error:
            raise PathDenied(str(error)) from error
        if not read_allowed(path.key):
            raise PathDenied("Path is outside the permitted area")
        if not path.absolute.exists():
            raise NoteNotFound(note)
        return path

    def _destination(self, path: NotePath, category: str) -> Path:
        safe_category = Path(category).name or "manual"
        destination = self.archive_root / self.review_subdir / safe_category / Path(path.rel).name
        if destination.exists():
            stem, suffix = destination.stem, destination.suffix
            destination = destination.with_name(
                f"{stem}-{self._digest(path.absolute)[:10]}{suffix}"
            )
        return destination

    @staticmethod
    def _digest(file: Path) -> str:
        import hashlib

        return hashlib.sha256(file.read_bytes()).hexdigest()


def _null_lock():
    """A no-op context manager for callers that did not pass a lock directory.

    Services built without one (unit tests, one-shot CLI use) must not silently
    lose the in-lock re-verification, so the check still runs; only the
    cross-process mutex is absent."""

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    return _Null()


def _rewrite_manifest(path: Path, *, state: str, entries: list[dict]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["state"] = state
    manifest["entries"] = entries
    manifest["finalised_at"] = datetime.now(timezone.utc).isoformat()
    write_atomic(path, json.dumps(manifest, ensure_ascii=False, indent=2))
