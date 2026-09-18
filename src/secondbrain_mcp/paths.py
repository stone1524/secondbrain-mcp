"""Vault path containment and privacy boundaries.

Parity with the production gateway: absolute paths and paths that escape the
vault root are rejected before any filesystem access; notes under a private
prefix are never readable and never a valid move destination.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class VaultPathError(ValueError):
    """The provided path is not a usable relative vault path."""


class PrivateNoteError(PermissionError):
    """The operation would cross the private-zone boundary."""


class VaultPaths:
    def __init__(
        self,
        vault_root: Path | str,
        private_prefixes: tuple[str, ...] = ("95_Private",),
    ):
        self.root = Path(vault_root).resolve()
        self.private_prefixes = tuple(
            PurePosixPath(prefix).as_posix().strip("/") for prefix in private_prefixes
        )

    def resolve(self, note: str) -> Path:
        if not isinstance(note, str) or not note.strip():
            raise VaultPathError("A relative vault path is required")
        candidate = Path(note)
        if candidate.is_absolute() or candidate.root:
            raise VaultPathError("A relative vault path is required")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise VaultPathError("Path escapes the vault")
        return resolved

    def relative(self, file: Path | str) -> str:
        path = Path(file)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve().relative_to(self.root).as_posix()

    def is_private(self, file: Path | str) -> bool:
        rel = self.relative(file)
        return any(
            rel == prefix or rel.startswith(prefix + "/") for prefix in self.private_prefixes
        )

    def assert_readable(self, file: Path | str) -> None:
        if self.is_private(file):
            raise PrivateNoteError("Private notes are not readable through this gateway")

    def assert_move_destination(self, file: Path | str) -> None:
        if self.is_private(file):
            raise PrivateNoteError("Cannot move into a private area through this tool")
