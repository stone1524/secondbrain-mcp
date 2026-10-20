"""Notes service: read, list, create, append, patch, metadata, move.

Every mutation follows the same shape so the safety properties cannot drift
apart again:

1. resolve + zone check through :class:`NotePath` (the only containment decision);
2. audit ``intent`` before the lock;
3. acquire the ownership-tokened lock;
4. re-verify ``expected_hash`` *inside* the lock (the read that produced it was
   unlocked, so it is a snapshot, not a guarantee);
5. mutate with an atomic write;
6. audit ``outcome`` with ``ok`` or an error code.

``expected_hash`` is required for every mutation. Making it optional — as the
production gateway did — meant a caller could opt out of optimistic concurrency
entirely by simply omitting the argument.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.atomic import write_atomic
from ..core.audit import AuditLog
from ..core.errors import (
    GatewayError,
    HashConflict,
    HashRequired,
    InvalidArguments,
    NoteNotFound,
    PathDenied,
    PrivateDenied,
)
from ..core.locking import FileLock, LockConflictError
from ..core.policy import NotePath, PrivateNoteError, VaultPathError, read_allowed

MARKDOWN_SUFFIX = ".md"
METADATA_FIELDS = frozenset(
    {"status", "tags", "updated_at", "agent_generated", "confidence", "state", "last_verified"}
)
UNTRUSTED = "untrusted_note"


def read_preserving_newlines(path: Path) -> str:
    """Read text without newline translation.

    ``Path.read_text`` normalizes CRLF to LF, while a hash computed over the raw
    bytes does not — so a note read as text and hashed as bytes yields two
    different, equally plausible hash values. Every hash-relevant read goes
    through here so the two always agree.
    """
    with open(path, encoding="utf-8", newline="") as stream:
        return stream.read()


@dataclass(slots=True)
class NoteService:
    root: Path
    inbox_folders: tuple[str, ...]
    lock_dir: Path
    audit: AuditLog
    stale_lock_seconds: float = 300.0

    # ---------- read ----------

    def read(self, note: str, *, offset: int = 0, max_length: int = 50000) -> dict:
        path = self._existing(note)
        content = read_preserving_newlines(path.absolute)
        total = len(content)
        start = max(0, min(int(offset), total))
        window = min(max(1, int(max_length)), 200000)
        visible = content[start : start + window]
        self.audit.record("note_read", path.rel, phase="outcome", ok=True)
        return {
            "note": path.rel,
            "content": visible,
            "offset": start,
            "total_length": total,
            "truncated": start + len(visible) < total,
            "hash": self._digest(content),
            # Notes are data. A caller must not treat their contents as
            # instructions; this marker is what makes that explicit.
            "trust": UNTRUSTED,
        }

    def list(
        self,
        *,
        folder: str | None = None,
        pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        base = self.root if not folder else self._parse(folder).absolute
        if not base.exists():
            raise NoteNotFound(folder or "(vault root)")
        needle = pattern.lower() if pattern else None
        entries: list[dict] = []
        for directory, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                name
                for name in dirnames
                if name.lower() != ".obsidian"
            ]
            for name in filenames:
                if not name.lower().endswith(MARKDOWN_SUFFIX):
                    continue
                file = Path(directory) / name
                try:
                    path = NotePath.from_disk(self.root, file)
                except (VaultPathError, ValueError):
                    continue
                if not read_allowed(path.key):
                    continue
                rel = path.rel
                if needle and needle not in rel.lower():
                    continue
                stat = file.stat()
                entries.append(
                    {
                        "note": rel,
                        "updated_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(),
                        "size": stat.st_size,
                    }
                )
        entries.sort(key=lambda item: item["updated_at"], reverse=True)
        bounded = max(1, min(int(limit), 500))
        start = max(0, int(offset))
        page = entries[start : start + bounded]
        self.audit.record(
            "note_list", folder or "", phase="outcome", ok=True, returned=len(page)
        )
        return {
            "folder": folder or "",
            "total": len(entries),
            "offset": start,
            "count": len(page),
            "has_more": start + len(page) < len(entries),
            "notes": page,
        }

    # ---------- create ----------

    def create(
        self,
        *,
        folder: str,
        filename: str,
        content: str,
        overwrite: bool = False,
    ) -> dict:
        if folder not in self.inbox_folders:
            raise InvalidArguments(
                f"folder must be one of {list(self.inbox_folders)}",
                "Use one of the configured inbox folders",
            )
        safe_name = Path(filename.replace("\\", "/")).name
        if not safe_name.lower().endswith(MARKDOWN_SUFFIX):
            raise InvalidArguments("Markdown filename required", "Pass a name ending in .md")
        # The destination is derived from the policy, not from the caller's string:
        # a traversal in `filename` cannot escape because only the basename survives.
        destination = self.root / "00_Inbox" / folder / safe_name
        path = NotePath.from_disk(self.root, destination)
        path.assert_writable()
        self.audit.record("note_create", path.rel, phase="intent")
        path.absolute.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            write_atomic(path.absolute, content)
            outcome = "overwritten"
        else:
            try:
                handle = os.open(path.absolute, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                # Production hard-failed here, so a re-run could never recover.
                stem = path.absolute.stem
                suffix = path.absolute.suffix
                index = 2
                while True:
                    candidate = path.absolute.with_name(f"{stem}-{index}{suffix}")
                    try:
                        handle = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    except FileExistsError:
                        index += 1
                        continue
                    path = NotePath.from_disk(self.root, candidate)
                    break
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            outcome = "created"
        self.audit.record("note_create", path.rel, phase="outcome", ok=True, result=outcome)
        return {"created": path.rel, "result": outcome, "hash": self._digest(content)}

    # ---------- append ----------

    def append(self, note: str, content: str, *, expected_hash: str | None) -> dict:
        path = self._existing(note)
        self._require_hash(expected_hash, "note_append", path.rel)
        try:
            with FileLock(
                self.lock_dir,
                path.absolute,
                relative_name=path.rel,
                stale_after_seconds=self.stale_lock_seconds,
            ):
                current = read_preserving_newlines(path.absolute)
                self._assert_hash(current, expected_hash)
                updated = f"{current}\n{content}\n"
                write_atomic(path.absolute, updated)
        except LockConflictError as error:
            raise GatewayError("LOCK_BUSY", str(error), "Retry in a few seconds") from error
        self.audit.record("note_append", path.rel, phase="outcome", ok=True)
        return {"appended": path.rel, "hash": self._digest(updated)}

    # ---------- patch ----------

    def patch(
        self,
        note: str,
        *,
        expected_hash: str | None,
        operation: str,
        target: str,
        content: str,
        preview: bool = True,
    ) -> dict:
        """Edit a note body. Always previewable; never writes on preview.

        The anchor must be unique for ``replace``. Ambiguity is refused rather
        than resolved by picking the first occurrence: a model that cannot name
        its target precisely enough has not understood the note well enough to
        be allowed to rewrite it.
        """
        path = self._existing(note)
        self._require_hash(expected_hash, "note_patch", path.rel)
        if operation not in ("replace", "replace_section", "insert_after_heading"):
            raise InvalidArguments(
                f"Unknown operation: {operation}",
                "Use replace, replace_section, or insert_after_heading",
            )
        if not target:
            raise InvalidArguments("target is required", "Provide the exact text or heading to edit")

        original = read_preserving_newlines(path.absolute)
        self._assert_hash(original, expected_hash)
        updated = self._apply_patch(original, operation, target, content)
        diff = unified_diff(path.rel, original, updated)
        if preview:
            self.audit.record(
                "note_patch", path.rel, phase="preview", ok=True, patch_mode=operation
            )
            return {
                "note": path.rel,
                "preview": True,
                "operation": operation,
                "diff": diff,
                "hash_before": self._digest(original),
                "hash_after": self._digest(updated),
                "requires_confirmation": True,
            }
        try:
            with FileLock(
                self.lock_dir,
                path.absolute,
                relative_name=path.rel,
                stale_after_seconds=self.stale_lock_seconds,
            ):
                self._assert_hash(read_preserving_newlines(path.absolute), expected_hash)
                write_atomic(path.absolute, updated)
        except LockConflictError as error:
            raise GatewayError("LOCK_BUSY", str(error), "Retry in a few seconds") from error
        self.audit.record(
            "note_patch", path.rel, phase="outcome", ok=True, patch_mode=operation
        )
        return {
            "note": path.rel,
            "preview": False,
            "applied": True,
            "diff": diff,
            "hash": self._digest(updated),
        }

    def _apply_patch(self, original: str, operation: str, target: str, content: str) -> str:
        if operation == "replace":
            occurrences = original.count(target)
            if occurrences == 0:
                raise InvalidArguments(
                    "target not found in note",
                    "Read the note and copy the exact text to replace",
                )
            if occurrences > 1:
                raise InvalidArguments(
                    f"target is ambiguous ({occurrences} occurrences)",
                    "Include more surrounding text so the anchor matches exactly once",
                )
            return original.replace(target, content, 1)
        if operation == "replace_section":
            pattern = re.compile(rf"^(#{{1,6}})\s+{re.escape(target)}\s*$", re.MULTILINE)
            matches = list(pattern.finditer(original))
            if len(matches) != 1:
                raise InvalidArguments(
                    f"heading {target!r} matched {len(matches)} times",
                    "Pass the exact heading text; it must be unique",
                )
            start = matches[0].start()
            level = len(matches[0].group(1))
            following = re.search(
                rf"^#{{1,{level}}}\s+", original[matches[0].end() :], re.MULTILINE
            )
            end = (
                matches[0].end() + following.start()
                if following
                else len(original)
            )
            body = content.rstrip("\n")
            return f"{original[:start]}{target_heading(matches[0], body)}{original[end:]}"
        # insert_after_heading
        pattern = re.compile(rf"^(#{{1,6}})\s+{re.escape(target)}\s*$", re.MULTILINE)
        matches = list(pattern.finditer(original))
        if len(matches) != 1:
            raise InvalidArguments(
                f"heading {target!r} matched {len(matches)} times",
                "Pass the exact heading text; it must be unique",
            )
        insert_at = matches[0].end()
        return f"{original[:insert_at]}\n{content.rstrip()}\n{original[insert_at:]}"

    # ---------- metadata ----------

    def metadata(self, note: str, fields: dict, *, expected_hash: str | None) -> dict:
        path = self._existing(note)
        self._require_hash(expected_hash, "note_meta", path.rel)
        allowed = {key: value for key, value in fields.items() if key in METADATA_FIELDS}
        dropped = sorted(set(fields) - set(allowed))
        if not allowed:
            raise InvalidArguments(
                f"No allowed fields in {sorted(fields)}",
                f"Allowed fields: {sorted(METADATA_FIELDS)}",
            )
        try:
            with FileLock(
                self.lock_dir,
                path.absolute,
                relative_name=path.rel,
                stale_after_seconds=self.stale_lock_seconds,
            ):
                content = read_preserving_newlines(path.absolute)
                self._assert_hash(content, expected_hash)
                updated = apply_frontmatter(content, allowed)
                write_atomic(path.absolute, updated)
        except LockConflictError as error:
            raise GatewayError("LOCK_BUSY", str(error), "Retry in a few seconds") from error
        self.audit.record(
            "note_meta", path.rel, phase="outcome", ok=True, keys=sorted(allowed)
        )
        return {
            "updated": path.rel,
            "fields": allowed,
            "dropped": dropped,
            "hash": self._digest(updated),
        }

    # ---------- move ----------

    def move(self, note: str, destination: str, *, expected_hash: str | None) -> dict:
        source = self._existing(note)
        self._require_hash(expected_hash, "note_move", source.rel)
        target = self._writable(destination)
        if target.absolute.exists():
            raise InvalidArguments(
                "Destination already exists", "Choose a path that does not yet exist"
            )
        try:
            with FileLock(
                self.lock_dir,
                source.absolute,
                relative_name=source.rel,
                stale_after_seconds=self.stale_lock_seconds,
            ):
                self._assert_hash(
                    read_preserving_newlines(source.absolute), expected_hash
                )
                target.absolute.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source.absolute, target.absolute)
        except LockConflictError as error:
            raise GatewayError("LOCK_BUSY", str(error), "Retry in a few seconds") from error
        self.audit.record(
            "note_move", source.rel, phase="outcome", ok=True, destination=target.rel
        )
        return {"moved": source.rel, "destination": target.rel}

    # ---------- helpers ----------

    def _parse(self, note: str) -> NotePath:
        try:
            return NotePath.parse(self.root, note)
        except VaultPathError as error:
            raise PathDenied(str(error)) from error

    def _writable(self, note: str) -> NotePath:
        """Parse and prove the destination is writable. Parking a note inside
        the private zone is the relocation half of the same boundary as reading
        it: the refusal must carry the PRIVATE_DENY contract (and reach the
        caller with a recovery), not escape as a raw PermissionError the SDK
        flattens to 'Error executing tool'. The Node health check asserted this
        (moveIntoPrivateDenied) against the gateway it retired."""
        path = self._parse(note)
        try:
            path.assert_writable()
        except PrivateNoteError as error:
            raise PrivateDenied() from error
        return path

    def _existing(self, note: str) -> NotePath:
        path = self._parse(note)
        # Zone before existence: a private path must be refused as private, never
        # reported as missing (which would confirm whether a file exists).
        try:
            path.assert_readable()
        except PrivateNoteError as error:
            raise PrivateDenied() from error
        if not path.absolute.exists():
            raise NoteNotFound(note)
        return path

    def _require_hash(self, expected_hash: str | None, operation: str, rel: str) -> None:
        if not expected_hash:
            self.audit.record(
                operation, rel, phase="outcome", ok=False, error="HASH_REQUIRED"
            )
            raise HashRequired()

    def _assert_hash(self, content: str, expected_hash: str | None) -> None:
        actual = self._digest(content)
        if expected_hash != actual:
            raise HashConflict()

    @staticmethod
    def _digest(content: str) -> str:
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def digest_file(path: Path) -> str:
        """Hash over raw bytes.

        Reading with ``newline=""`` keeps CRLF intact, so the text hash equals the
        byte hash; without it a Windows note read as text and hashed as bytes
        would produce two different, both-plausible-looking values.
        """
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()


def target_heading(match: re.Match, body: str) -> str:
    return f"{match.group(0)}\n\n{body}\n"


_NEEDS_QUOTE = re.compile(r"""[:#\[\]{}&*!|>'"%@`\n]|^\s|\s$|^(true|false|null|~|-?\d+(\.\d+)?)$""", re.IGNORECASE)


def yaml_scalar(value: object) -> str:
    """Render a scalar the way a human would write it.

    Production emitted ``json.dumps(value)`` unconditionally, so every string
    became ``status: "inbox"`` — valid, but non-idiomatic YAML that a person
    editing the note by hand would never write.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    text = str(value)
    return json.dumps(text, ensure_ascii=False) if _NEEDS_QUOTE.search(text) else text


def apply_frontmatter(content: str, fields: dict) -> str:
    if not content.startswith("---"):
        raise InvalidArguments(
            "Note has no frontmatter", "Add a frontmatter block or use append instead"
        )
    end = content.find("\n---", 3)
    if end < 0:
        raise InvalidArguments(
            "Unterminated frontmatter", "Close the frontmatter block with ---"
        )
    front = content[:end]
    for key, value in fields.items():
        line = f"{key}: {yaml_scalar(value)}"
        pattern = re.compile(rf"^{re.escape(key)}:[^\n]*$", re.MULTILINE)
        front = pattern.sub(line, front) if pattern.search(front) else f"{front}\n{line}"
    return f"{front}{content[end:]}"


def unified_diff(note: str, before: str, after: str) -> str:
    import difflib

    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{note}",
        tofile=f"b/{note}",
        n=3,
    )
    return "".join(diff)


_ = PrivateDenied  # exported for adapters that map error classes to codes
