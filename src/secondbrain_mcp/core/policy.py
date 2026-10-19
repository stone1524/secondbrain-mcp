"""Vault path policy — the single source of truth for every zone decision.

The production system implemented the same privacy rule four times (gateway,
RAG indexer, cleanup collector, maintenance) with divergent semantics: three
copies compared raw strings case-sensitively, so a vault whose directory was
spelled ``95_private`` silently disabled the private zone on Windows. This
module exists so that never happens again: every caller asks the same questions
and gets an answer derived from the resolved location, not from the caller's
spelling.

Layering rule: this module imports nothing from the package.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath


class VaultPathError(ValueError):
    """The provided path is not a usable relative vault path."""


class PrivateNoteError(PermissionError):
    """The operation would cross the private-zone boundary."""


class Zone(Enum):
    """Where a note sits relative to the gateway's access policy."""

    PRIVATE = "private"  # unreadable, unwritable, unindexed
    ARCHIVE = "archive"  # history: never indexed
    AUXILIARY = "auxiliary"  # indexed, but not in default results
    LEXICAL_ONLY = "lexical_only"  # indexed for keywords, never embedded
    GENERATED = "generated"  # machine-written aggregate; never retrievable
    STANDARD = "standard"


class IndexDecision(Enum):
    INDEX = "index"
    LEXICAL_ONLY = "lexical_only"
    SKIP = "skip"


# Order matters: first match wins. Patterns run against a lower-cased path, so
# no rule needs its own case handling.
_RULES: tuple[tuple[re.Pattern[str], Zone], ...] = (
    (re.compile(r"(?:^|/)95_private(?:/|$)"), Zone.PRIVATE),
    (re.compile(r"(?:^|/)\.obsidian(?:/|$)"), Zone.PRIVATE),
    (re.compile(r"^98_archive(?:/|$)"), Zone.ARCHIVE),
    (re.compile(r"^90_agent(?:/|$)"), Zone.AUXILIARY),
    (re.compile(r"^30_resources/fileindex/auto-knowledge-index\.md$"), Zone.GENERATED),
    (re.compile(r"^40_persona/model/(?:audit|candidates)\.md$"), Zone.GENERATED),
    (re.compile(r"^30_resources/fileindex(?:/|$)"), Zone.LEXICAL_ONLY),
)


def _normalize(rel: str) -> str:
    return rel.replace("\\", "/").strip("/").lower()


def zone_of(rel: str) -> Zone:
    """Classify a vault-relative path (forward slashes, no leading slash)."""
    normalized = _normalize(rel)
    for pattern, zone in _RULES:
        if pattern.search(normalized):
            return zone
    return Zone.STANDARD


def read_allowed(rel: str) -> bool:
    return zone_of(rel) is not Zone.PRIVATE


def write_allowed(rel: str) -> bool:
    return zone_of(rel) is not Zone.PRIVATE


def index_decision(rel: str) -> IndexDecision:
    zone = zone_of(rel)
    if zone is Zone.PRIVATE:
        return IndexDecision.SKIP
    # Archive content is history, not live knowledge: hundreds of machine-written
    # session artifacts would out-rank real notes on workflow-ish keywords.
    if zone is Zone.ARCHIVE:
        return IndexDecision.SKIP
    if zone is Zone.GENERATED:
        # An aggregate index repeats every other note's title, so it matches
        # almost any query and then out-ranks the note it was listing. It must
        # not be a retrievable document at all.
        return IndexDecision.SKIP
    if zone is Zone.LEXICAL_ONLY:
        return IndexDecision.LEXICAL_ONLY
    return IndexDecision.INDEX


def in_default_scope(rel: str) -> bool:
    """Auxiliary areas are searchable only when the caller narrows into them."""
    return zone_of(rel) in (Zone.STANDARD, Zone.LEXICAL_ONLY)


def scope_prefixes_allows(rel: str, prefixes: list[str]) -> bool:
    """Explicit prefixes re-enable auxiliary areas; they never re-enable private."""
    if not read_allowed(rel):
        return False
    if not prefixes:
        return in_default_scope(rel)
    normalized = [prefix.replace("\\", "/").strip("/") for prefix in prefixes]
    for prefix in normalized:
        if prefix == "" or rel == prefix or rel.startswith(prefix + "/"):
            return True
    return False


@dataclass(frozen=True, slots=True)
class NotePath:
    """A validated, vault-relative note location.

    Construction is the only place containment is decided. ``rel`` preserves the
    caller's spelling for display and for use as a database key; ``key`` is the
    case-folded form used for every comparison, so Windows casing differences can
    never produce a second identity for one file.
    """

    root: Path
    rel: str
    key: str
    absolute: Path

    @classmethod
    def parse(cls, root: Path | str, raw: str) -> NotePath:
        if not isinstance(raw, str) or not raw.strip():
            raise VaultPathError("A relative vault path is required")
        if Path(raw).is_absolute() or Path(raw).root:
            raise VaultPathError("A relative vault path is required")
        # Reject Windows drive-relative forms ("C:foo") that is_absolute misses.
        if len(raw) > 1 and raw[1] == ":":
            raise VaultPathError("A relative vault path is required")

        root_path = Path(root).resolve()
        candidate = (root_path / raw).resolve()
        try:
            relative = candidate.relative_to(root_path)
        except ValueError as error:
            raise VaultPathError("Path escapes the vault") from error
        if str(relative) in (".", ""):
            raise VaultPathError("A note path is required, not the vault root")

        rel = relative.as_posix()
        return cls(root=root_path, rel=rel, key=cls.fold(rel), absolute=candidate)

    @staticmethod
    def fold(value: str) -> str:
        """Comparison key: separators normalized, case folded."""
        return os.path.normcase(value.replace("\\", "/")).casefold()

    @property
    def zone(self) -> Zone:
        return zone_of(self.key)

    def assert_readable(self) -> NotePath:
        if not read_allowed(self.key):
            raise PrivateNoteError("Private notes are not readable through this gateway")
        return self

    def assert_writable(self) -> NotePath:
        if not write_allowed(self.key):
            raise PrivateNoteError("Cannot write into a private area through this gateway")
        return self

    @classmethod
    def from_disk(cls, root: Path | str, file: Path | str) -> NotePath:
        """Build from a path discovered on disk; already known to be inside."""
        root_path = Path(root).resolve()
        path = Path(file)
        if not path.is_absolute():
            path = root_path / path
        relative = path.resolve().relative_to(root_path).as_posix()
        return cls.parse(root_path, relative)

    @staticmethod
    def is_manifest_safe(value: object) -> bool:
        """Manifest entries are untrusted: only vault-relative, non-traversing."""
        if not isinstance(value, str) or not value.strip():
            return False
        normalized = value.replace("\\", "/")
        if normalized.startswith("/"):
            return False
        if Path(value).is_absolute() or Path(value).root:
            return False
        if len(value) > 1 and value[1] == ":":
            return False
        return ".." not in PurePosixPath(normalized).parts
