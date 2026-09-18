"""Per-file advisory locks with stale-lock rotation (production parity).

Lock files are created atomically with O_CREAT|O_EXCL: two agents racing on the
same note cannot both win. A lock older than the staleness window is rotated
into a history directory instead of being silently broken, so a crashed agent
never wedges a note forever — and every acquired/released lock leaves a trace.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self


class LockConflictError(RuntimeError):
    """Another agent is modifying the target note."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FileLock:
    def __init__(
        self,
        lock_dir: Path | str,
        target: Path | str,
        *,
        stale_after_seconds: float = 300.0,
        relative_name: str | None = None,
    ):
        self.lock_dir = Path(lock_dir)
        self.history_dir = self.lock_dir / "history"
        self.target = Path(target)
        self.key = sha256(str(self.target).lower().encode("utf-8")).hexdigest()
        self.lock_path = self.lock_dir / f"{self.key}.lock"
        self.stale_after_seconds = stale_after_seconds
        self._relative_name = relative_name

    def acquire(self) -> FileLock:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._rotate_stale_lock()
        payload = {
            "pid": os.getpid(),
            "file": self._relative_name or self.target.name,
            "at": now_iso(),
        }
        try:
            handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise LockConflictError(
                "Note is being modified by another agent; retry later"
            ) from error
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        return self

    def _rotate_stale_lock(self) -> None:
        try:
            age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if age > self.stale_after_seconds:
            rotated = self.history_dir / f"{self.key}-stale-{int(time.time() * 1000)}.lock"
            self.lock_path.rename(rotated)

    def release(self) -> None:
        try:
            released = self.history_dir / (
                f"{self.key}-{int(time.time() * 1000)}-{uuid.uuid4().hex}.lock"
            )
            self.lock_path.rename(released)
        except FileNotFoundError:
            pass

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
