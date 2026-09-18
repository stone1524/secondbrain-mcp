"""Content hashing and optimistic-concurrency checks (production parity)."""

from __future__ import annotations

import hashlib
from pathlib import Path


class HashConflictError(RuntimeError):
    """The note changed since the caller last read it."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_hash_matches(path: Path | str, expected_hash: str | None) -> str:
    """Return the current content hash; raise when an expected hash no longer matches.

    The error message is part of the tool contract: it tells the agent exactly how
    to recover (re-read, then retry with the fresh hash).
    """
    current = sha256_file(path)
    if expected_hash and current != expected_hash:
        raise HashConflictError(
            "Note changed since it was read; retry with the current content hash"
        )
    return current
