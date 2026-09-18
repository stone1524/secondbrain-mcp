from __future__ import annotations

import json
import os
import time

import pytest

from secondbrain_mcp.locking import FileLock, LockConflictError


def test_acquire_creates_lock_with_metadata(tmp_path):
    target = tmp_path / "vault" / "note.md"
    lock = FileLock(tmp_path / "locks", target, relative_name="note.md").acquire()
    try:
        assert lock.lock_path.exists()
        payload = json.loads(lock.lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["file"] == "note.md"
        assert "at" in payload
    finally:
        lock.release()


def test_second_acquire_conflicts(tmp_path):
    target = tmp_path / "note.md"
    first = FileLock(tmp_path / "locks", target).acquire()
    with pytest.raises(LockConflictError, match="another agent"):
        FileLock(tmp_path / "locks", target).acquire()
    first.release()


def test_release_moves_lock_to_history(tmp_path):
    target = tmp_path / "note.md"
    lock = FileLock(tmp_path / "locks", target).acquire()
    lock.release()
    assert not lock.lock_path.exists()
    history = list((tmp_path / "locks" / "history").glob("*.lock"))
    assert len(history) == 1


def test_context_manager_releases_on_error(tmp_path):
    target = tmp_path / "note.md"
    with pytest.raises(RuntimeError), FileLock(tmp_path / "locks", target):
        raise RuntimeError("boom")
    assert not FileLock(tmp_path / "locks", target).lock_path.exists()


def test_stale_lock_is_rotated_and_broken(tmp_path):
    target = tmp_path / "note.md"
    lock = FileLock(tmp_path / "locks", target, stale_after_seconds=1.0)
    lock.acquire()
    old = time.time() - 60
    os.utime(lock.lock_path, (old, old))
    second = FileLock(tmp_path / "locks", target, stale_after_seconds=1.0).acquire()
    try:
        assert second.lock_path.exists()
        rotated = list((tmp_path / "locks" / "history").glob("*-stale-*.lock"))
        assert len(rotated) == 1
    finally:
        second.release()


def test_fresh_lock_is_not_stolen(tmp_path):
    target = tmp_path / "note.md"
    lock = FileLock(tmp_path / "locks", target, stale_after_seconds=300.0)
    lock.acquire()
    with pytest.raises(LockConflictError):
        FileLock(tmp_path / "locks", target, stale_after_seconds=300.0).acquire()
    lock.release()
