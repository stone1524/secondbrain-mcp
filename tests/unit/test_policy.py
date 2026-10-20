"""Adversarial tests for the core policy layer.

Every case here corresponds to a defect observed in the production system or in
the v0.1 package. They are written to fail loudly if the defect returns.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from secondbrain_mcp.core.audit import AuditLog
from secondbrain_mcp.core.locking import FileLock, LockConflictError
from secondbrain_mcp.core.policy import (
    IndexDecision,
    NotePath,
    PrivateNoteError,
    VaultPathError,
    Zone,
    in_default_scope,
    index_decision,
    read_allowed,
    scope_prefixes_allows,
    zone_of,
)


@pytest.mark.parametrize(
    "spelling",
    ["95_Private", "95_private", "95_PRIVATE", "95_PrIvAtE", "95_privATe"],
)
def test_every_casing_of_the_private_zone_is_private(spelling: str) -> None:
    """The production bug: isPrivate compared case-sensitively, so an on-disk
    '95_private' directory was readable through the gateway on Windows."""
    assert zone_of(f"{spelling}/Persona/personality.json") is Zone.PRIVATE
    assert read_allowed(f"{spelling}/Persona/personality.json") is False
    assert index_decision(f"{spelling}/anything.md") is IndexDecision.SKIP


@pytest.mark.parametrize("spelling", [".obsidian", ".Obsidian", ".OBSIDIAN"])
def test_obsidian_config_is_never_readable(spelling: str) -> None:
    assert zone_of(f"{spelling}/app.json.md") is Zone.PRIVATE
    assert index_decision(f"{spelling}/x.md") is IndexDecision.SKIP


@pytest.mark.parametrize("spelling", ["98_Archive", "98_archive", "98_ARCHIVE"])
def test_archive_is_case_insensitive_and_not_in_default_scope(spelling: str) -> None:
    assert zone_of(f"{spelling}/Review/old.md") is Zone.ARCHIVE
    assert in_default_scope(f"{spelling}/Review/old.md") is False


def test_nested_private_directory_is_caught_at_any_depth() -> None:
    """A private folder nested deeper than the vault root must not become a hole."""
    assert zone_of("30_Resources/95_private/leak.md") is Zone.PRIVATE


def test_auxiliary_areas_are_indexed_and_in_default_scope() -> None:
    """90_Agent used to be excluded from default results. On the live vault that
    was 19 of 44 documents — including the densest technical note — so a rule
    meant to keep ops noise out was hiding the answers."""
    assert zone_of("90_Agent/research/x.md") is Zone.AUXILIARY
    assert index_decision("90_Agent/research/x.md") is IndexDecision.INDEX
    assert in_default_scope("90_Agent/research/x.md") is True


def test_explicit_prefix_never_reaches_private_but_folds_case() -> None:
    assert scope_prefixes_allows("90_Agent/research/x.md", ["90_Agent"]) is True
    # Prefix matching folds case, like every other path decision in the system.
    assert scope_prefixes_allows("90_Agent/research/x.md", ["90_agent"]) is True
    assert scope_prefixes_allows("10_Notes/a.md", ["10_notes/"]) is True
    # No prefix list can reach the private zone.
    assert scope_prefixes_allows("95_Private/x.md", ["95_Private", ""]) is False
    assert scope_prefixes_allows("95_private/x.md", ["95_private"]) is False


def test_generated_views_are_never_retrievable() -> None:
    """An aggregate index repeats every note's title, so it matches almost any
    query and then out-ranks the note it was listing. It must not be indexed."""
    assert index_decision("40_Persona/Model/audit.md") is IndexDecision.SKIP
    assert index_decision("40_Persona/Model/candidates.md") is IndexDecision.SKIP
    assert index_decision("30_Resources/FileIndex/auto-knowledge-index.md") is IndexDecision.SKIP


def test_fileindex_is_lexical_only() -> None:
    assert zone_of("30_Resources/FileIndex/stub.md") is Zone.LEXICAL_ONLY
    assert index_decision("30_Resources/FileIndex/stub.md") is IndexDecision.LEXICAL_ONLY


def test_standard_note_is_standard() -> None:
    assert zone_of("40_Persona/Preferences/monthly-review.md") is Zone.STANDARD
    assert in_default_scope("40_Persona/Preferences/monthly-review.md") is True


# ---------- NotePath containment ----------


def test_notepath_rejects_absolute_and_traversal(tmp_path: Path) -> None:
    for bad in ["/etc/passwd", "C:/Windows/system32", "C:relative", "../outside.md", "a/../../b.md", ""]:
        with pytest.raises(VaultPathError):
            NotePath.parse(tmp_path, bad)


def test_notepath_containment_survives_case_variation(tmp_path: Path) -> None:
    (tmp_path / "95_Private").mkdir()
    note = NotePath.parse(tmp_path, "95_private/secret.md")
    assert note.zone is Zone.PRIVATE
    with pytest.raises(PrivateNoteError):
        note.assert_readable()


def test_notepath_key_is_case_folded_but_rel_preserves_spelling(tmp_path: Path) -> None:
    note = NotePath.parse(tmp_path, "40_Persona/Model/Profile.md")
    assert note.rel == "40_Persona/Model/Profile.md"
    assert note.key == NotePath.fold("40_Persona/Model/Profile.md")
    assert note.key == note.key.lower() if os.name == "nt" else True


def test_manifest_path_validation_rejects_the_two_escape_shapes() -> None:
    """path.join(root, absolute) discards root; '..' walks out. Both must be refused."""
    assert NotePath.is_manifest_safe("00_Inbox/Web/a.md") is True
    assert NotePath.is_manifest_safe("98_Archive/Review/a.md") is True
    assert NotePath.is_manifest_safe("/etc/passwd") is False
    assert NotePath.is_manifest_safe("C:/Windows/x.md") is False
    assert NotePath.is_manifest_safe("../../../etc/passwd") is False
    assert NotePath.is_manifest_safe("a/../../b.md") is False
    assert NotePath.is_manifest_safe("") is False
    assert NotePath.is_manifest_safe(None) is False


# ---------- Locking ----------


def test_second_acquirer_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("x", encoding="utf-8")
    first = FileLock(tmp_path / "locks", target).acquire()
    try:
        with pytest.raises(LockConflictError):
            FileLock(tmp_path / "locks", target).acquire()
    finally:
        first.release()


def test_release_does_not_retire_someone_elses_lock(tmp_path: Path) -> None:
    """The production race: a writer whose lock was reaped as stale renamed away
    the next holder's lock in its finally block."""
    target = tmp_path / "note.md"
    target.write_text("x", encoding="utf-8")
    first = FileLock(tmp_path / "locks", target).acquire()
    # Simulate the stale reaper stealing the lock and a new owner taking it.
    first.lock_path.unlink()
    second = FileLock(tmp_path / "locks", target).acquire()
    try:
        first.release()  # must be a no-op now
        assert second.lock_path.exists(), "the current owner's lock was destroyed"
    finally:
        second.release()


def test_stale_lock_is_rotated_not_deleted(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("x", encoding="utf-8")
    lock = FileLock(tmp_path / "locks", target, stale_after_seconds=0)
    lock.acquire()
    lock.lock_path.unlink()
    stale = FileLock(tmp_path / "locks", target, stale_after_seconds=0)
    stale.lock_path.write_text("{}", encoding="utf-8")
    os.utime(stale.lock_path, (0, 0))
    os.remove(stale.lock_path)
    stale.lock_path.write_text("{}", encoding="utf-8")
    os.utime(stale.lock_path, (0, 0))
    stale.acquire()
    try:
        rotated = list((tmp_path / "locks" / "history").glob("*-stale-*.lock"))
        assert rotated, "a rotated lock must be preserved in history"
    finally:
        stale.release()


# ---------- Audit ----------


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", agent_label="test")
    log.record("append_note", "a.md", phase="intent")
    log.record("append_note", "a.md", phase="outcome", ok=True)
    ok, _ = log.verify()
    assert ok is True

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["note"] = "b.md"
    lines[0] = json.dumps(tampered, ensure_ascii=False, sort_keys=True)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad = AuditLog(tmp_path / "audit.jsonl").verify()
    assert ok is False
    assert bad is not None


def test_audit_records_intent_and_outcome_separately(tmp_path: Path) -> None:
    """A rejected write must be distinguishable from an accepted one."""
    log = AuditLog(tmp_path / "audit.jsonl", agent_label="test")
    log.record("append_note", "a.md", phase="intent")
    log.record("append_note", "a.md", phase="outcome", ok=False, error="HASH_CONFLICT")
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["phase"] for r in records] == ["intent", "outcome"]
    assert records[1]["ok"] is False
    assert records[1]["error"] == "HASH_CONFLICT"
