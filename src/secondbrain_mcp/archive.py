"""No-delete archiving: move notes into the archive area with batch rollback manifests.

Production parity:
- an archived note keeps its vault-relative path under ``<archive>/Review/``;
- every move is re-verified against the content hash captured at scan time —
  a note edited between scan and archive is skipped, never mis-filed;
- name collisions in the archive get a short-hash suffix instead of overwrite;
- each non-dry batch writes a manifest (``batch-<ts>-<id>.json``) that a restore
  can reverse: skipped when already restored, when the archived file is missing,
  when archived content changed, or when the original path is occupied;
- nothing in this module ever unlinks a file.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .hashing import sha256_file
from .locking import FileLock, now_iso
from .paths import VaultPaths


@dataclass
class ArchiveResult:
    note: str
    status: str  # archived | would_archive | skipped
    destination: str | None = None
    reason: str | None = None
    category: str | None = None

    def as_dict(self) -> dict:
        out = {"note": self.note, "status": self.status}
        if self.destination is not None:
            out["destination"] = self.destination
        if self.reason is not None:
            out["reason"] = self.reason
        if self.category is not None:
            out["category"] = self.category
        return out


class Archiver:
    def __init__(
        self,
        paths: VaultPaths,
        *,
        archive_dir: str = "98_Archive",
        review_dir: str = "Review",
        agent_label: str | None = None,
    ):
        self.paths = paths
        self.archive_dir = archive_dir
        self.review_dir = review_dir
        self.manifest_dir = paths.root / archive_dir / ".manifests"
        self.agent_label = agent_label

    def archive_notes(self, requests, *, dry_run: bool = False) -> dict:
        """Archive ``(note, scan_hash, category)`` tuples in one reversible batch.

        ``scan_hash`` is the content hash captured when the note was selected for
        cleanup; the move is skipped when the file's current hash differs.
        """
        requests = list(requests)
        results: list[ArchiveResult] = []
        entries: list[dict] = []
        for note, scan_hash, category in requests:
            file = self.paths.resolve(note)
            rel = self.paths.relative(file)
            if rel.startswith(f"{self.archive_dir}/"):
                results.append(
                    ArchiveResult(note, "skipped", reason="note is already inside the archive area")
                )
                continue
            self.paths.assert_readable(file)
            try:
                current = sha256_file(file)
            except FileNotFoundError:
                results.append(ArchiveResult(note, "skipped", reason="note not found"))
                continue
            if current != scan_hash:
                results.append(
                    ArchiveResult(note, "skipped", reason="content changed since the last scan")
                )
                continue
            destination = self.paths.root / self.archive_dir / self.review_dir / rel
            if destination.exists():
                destination = destination.with_name(
                    f"{destination.stem}-{current[:10]}{destination.suffix}"
                )
            if dry_run:
                results.append(
                    ArchiveResult(
                        note,
                        "would_archive",
                        destination=self.paths.relative(destination),
                        category=category,
                    )
                )
                continue
            with FileLock(self.manifest_dir.parent / ".locks", file, relative_name=rel):
                if sha256_file(file) != scan_hash:
                    results.append(
                        ArchiveResult(
                            note, "skipped", reason="content changed since the last scan"
                        )
                    )
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(file, destination)
            entries.append(
                {
                    "from": rel,
                    "to": self.paths.relative(destination),
                    "hash": scan_hash,
                    "size": destination.stat().st_size,
                    "category": category,
                }
            )
            results.append(
                ArchiveResult(
                    note, "archived", destination=self.paths.relative(destination), category=category
                )
            )
        batch_id = None
        if entries:
            batch_id = (
                f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
            )
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "batch_id": batch_id,
                "at": now_iso(),
                "agent": self.agent_label,
                "summary": {"requested": len(requests), "archived": len(entries)},
                "entries": entries,
                "restores": [],
            }
            (self.manifest_dir / f"{batch_id}.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return {
            "batch_id": batch_id,
            "dry_run": dry_run,
            "archived": len(entries),
            "results": [item.as_dict() for item in results],
        }

    def list_batches(self, limit: int = 30) -> list[dict]:
        manifests = self._manifest_files()
        listing = []
        for path in manifests[: max(limit, 0)]:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            listing.append(
                {
                    "batch_id": manifest.get("batch_id"),
                    "at": manifest.get("at"),
                    "agent": manifest.get("agent"),
                    "archived": len(manifest.get("entries", [])),
                    "restored": sum(len(r.get("notes", [])) for r in manifest.get("restores", [])),
                }
            )
        return listing

    def restore_batch(self, batch_id: str = "latest", *, dry_run: bool = False, only: str | None = None) -> dict:
        manifest_path = self._resolve_manifest(batch_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        already = {note for r in manifest.get("restores", []) for note in r.get("notes", [])}
        results: list[dict] = []
        restored: list[str] = []
        def skip(outcome: dict, reason: str) -> dict:
            return {**outcome, "status": "skipped", "reason": reason}

        for entry in manifest.get("entries", []):
            if only and only not in entry["from"] and only not in entry["to"]:
                continue
            outcome = {"from": entry["from"], "note": entry["to"]}
            archived_file = self.paths.root / entry["to"]
            original = self.paths.root / entry["from"]

            if entry["from"] in already:
                results.append(skip(outcome, "already restored"))
            elif not archived_file.exists():
                results.append(skip(outcome, "archived file missing"))
            elif sha256_file(archived_file) != entry["hash"]:
                results.append(skip(outcome, "archived content changed after archiving"))
            elif original.exists():
                results.append(skip(outcome, "original path occupied"))
            elif dry_run:
                results.append({**outcome, "status": "would_restore"})
            else:
                original.parent.mkdir(parents=True, exist_ok=True)
                os.rename(archived_file, original)
                restored.append(entry["from"])
                results.append({**outcome, "status": "restored"})
        if restored and not dry_run:
            manifest.setdefault("restores", []).append(
                {"at": now_iso(), "count": len(restored), "notes": restored}
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return {
            "batch_id": manifest["batch_id"],
            "dry_run": dry_run,
            "restored": len(restored),
            "would_restore": sum(1 for r in results if r["status"] == "would_restore"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "results": results,
        }

    def _manifest_files(self) -> list[Path]:
        if not self.manifest_dir.exists():
            return []
        manifests = [p for p in self.manifest_dir.iterdir() if p.name.startswith("batch-") and p.name.endswith(".json")]
        return sorted(manifests, key=lambda p: p.name, reverse=True)

    def _resolve_manifest(self, batch_id: str) -> Path:
        if batch_id == "latest":
            manifests = self._manifest_files()
            if not manifests:
                raise FileNotFoundError("No archive batches found")
            return manifests[0]
        candidate = self.manifest_dir / f"{batch_id}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Batch not found: {batch_id}")
        return candidate
