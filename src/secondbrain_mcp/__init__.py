"""secondbrain-mcp: safety-first building blocks for agent-writable knowledge bases."""

from .archive import Archiver
from .audit import AuditLog
from .hashing import HashConflictError, ensure_hash_matches, sha256_file
from .locking import FileLock, LockConflictError
from .paths import PrivateNoteError, VaultPathError, VaultPaths

__version__ = "0.1.0"

__all__ = [
    "Archiver",
    "AuditLog",
    "FileLock",
    "HashConflictError",
    "LockConflictError",
    "PrivateNoteError",
    "VaultPathError",
    "VaultPaths",
    "ensure_hash_matches",
    "sha256_file",
]
