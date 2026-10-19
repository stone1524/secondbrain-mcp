"""Per-file advisory locks with ownership tokens.

Two defects in the previous implementation are fixed here:

1. **No ownership check on release.** A writer whose lock was reaped as stale
   would rename away the *next* holder's lock in its ``finally`` block, silently
   destroying mutual exclusion. Every lock now carries a token, and only the
   owner retires it.
2. **Unchecked rotation race.** ``stat`` then ``rename`` let two contenders both
   observe the same stale lock and both rotate; the loser raised an uncaught
   ``FileNotFoundError``. Rotation now tolerates the loser.

A rotated lock is never deleted — it moves into ``history/`` so every acquisition
leaves a trace.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Self


class LockConflictError(RuntimeError):
    """Another agent is modifying the target note."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class _LockPayload:
    pid: int
    token: str
    file: str
    at: str


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
        # Case-folded key: on Windows one file must not acquire two distinct locks.
        self.key = sha256(str(self.target).lower().encode("utf-8")).hexdigest()
        self.lock_path = self.lock_dir / f"{self.key}.lock"
        self.stale_after_seconds = stale_after_seconds
        self._relative_name = relative_name
        self._token: str | None = None

    def acquire(self) -> FileLock:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._rotate_stale_lock()
        token = uuid.uuid4().hex
        payload = _LockPayload(
            pid=os.getpid(),
            token=token,
            file=self._relative_name or self.target.name,
            at=now_iso(),
        )
        try:
            handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise LockConflictError(
                "Note is being modified by another agent; retry later"
            ) from error
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(asdict(payload), stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        self._token = token
        return self

    def _rotate_stale_lock(self) -> None:
        try:
            age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if age <= self.stale_after_seconds:
            return
        rotated = self.history_dir / f"{self.key}-stale-{int(time.time() * 1000)}.lock"
        try:
            self.lock_path.rename(rotated)
        except (FileNotFoundError, OSError):
            # Another contender rotated it first; that is the outcome we wanted.
            pass

    def release(self) -> None:
        if self._token is None:
            return
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self._token = None
            return
        if payload.get("token") != self._token:
            # Our lock was reaped and someone else holds it now. Leave it alone.
            self._token = None
            return
        released = self.history_dir / (
            f"{self.key}-{int(time.time() * 1000)}-{self._token}.lock"
        )
        try:
            self.lock_path.rename(released)
        except FileNotFoundError:
            pass
        self._token = None

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
