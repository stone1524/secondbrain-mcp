"""Durable file writes: temp file, fsync, atomic replace.

Every mutation in the previous system used a plain ``write_file`` in place, so a
process killed mid-write left a truncated note. The temp file lives in the same
directory as the target (``os.replace`` is only atomic within a filesystem).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(target: Path | str, content: str, *, encoding: str = "utf-8") -> Path:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def append_durable(target: Path | str, content: str, *, encoding: str = "utf-8") -> Path:
    """Append + fsync. Used for the audit log, where a torn tail loses evidence."""
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding=encoding, newline="") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return path
