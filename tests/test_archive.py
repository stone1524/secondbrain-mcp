from __future__ import annotations

import json

import pytest

from secondbrain_mcp.archive import Archiver
from secondbrain_mcp.hashing import sha256_file
from secondbrain_mcp.paths import PrivateNoteError


@pytest.fixture
def archiver(vault):
    return Archiver(vault, agent_label="pytest")


def scan(vault, rel):
    return sha256_file(vault.root / rel)


def test_archive_moves_notes_and_writes_manifest(archiver, vault):
    hash_a = scan(vault, "00_Inbox/Conversations/note-a.md")
    hash_b = scan(vault, "10_Notes/note-b.md")
    out = archiver.archive_notes(
        [
            ("00_Inbox/Conversations/note-a.md", hash_a, "inbox_aging"),
            ("10_Notes/note-b.md", hash_b, "exact_duplicate"),
        ]
    )
    assert out["batch_id"] and out["batch_id"].startswith("batch-")
    assert out["archived"] == 2
    assert not (vault.root / "00_Inbox" / "Conversations" / "note-a.md").exists()
    archived = vault.root / "98_Archive" / "Review" / "00_Inbox" / "Conversations" / "note-a.md"
    assert archived.exists()
    manifest = json.loads(
        (vault.root / "98_Archive" / ".manifests" / f"{out['batch_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["summary"] == {"requested": 2, "archived": 2}
    assert manifest["agent"] == "pytest"
    assert manifest["entries"][0]["hash"] == hash_a
    assert manifest["restores"] == []


def test_dry_run_moves_nothing(archiver, vault):
    hash_a = scan(vault, "00_Inbox/Conversations/note-a.md")
    out = archiver.archive_notes(
        [("00_Inbox/Conversations/note-a.md", hash_a, "inbox_aging")], dry_run=True
    )
    assert out["batch_id"] is None
    assert out["results"][0]["status"] == "would_archive"
    assert (vault.root / "00_Inbox" / "Conversations" / "note-a.md").exists()


def test_changed_since_scan_is_skipped(archiver, vault, make_note):
    file = make_note("10_Notes/late-edit.md", "scanned version")
    stale_hash = sha256_file(file)
    file.write_text("edited after the scan", encoding="utf-8")
    out = archiver.archive_notes([("10_Notes/late-edit.md", stale_hash, "exact_duplicate")])
    assert out["archived"] == 0
    assert out["results"][0] == {
        "note": "10_Notes/late-edit.md",
        "status": "skipped",
        "reason": "content changed since the last scan",
    }
    assert file.exists()


def test_note_already_in_archive_is_skipped(archiver, vault):
    out = archiver.archive_notes([("98_Archive/Review/old.md", "0" * 64, "orphan_index")])
    assert out["results"][0]["reason"] == "note is already inside the archive area"


def test_private_source_is_hard_rejected(archiver):
    with pytest.raises(PrivateNoteError):
        archiver.archive_notes([("95_Private/secrets.md", "0" * 64, "exact_duplicate")])


def test_name_collision_gets_short_hash_suffix(archiver, vault):
    # The note's archive slot is already occupied by a previous batch: the new
    # arrival must get a short-hash suffix instead of overwriting the old copy.
    occupied = vault.root / "98_Archive" / "Review" / "10_Notes" / "shared-name.md"
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_text("# previously archived copy\n", encoding="utf-8")
    hash_shared = scan(vault, "10_Notes/shared-name.md")
    out = archiver.archive_notes([("10_Notes/shared-name.md", hash_shared, "exact_duplicate")])
    assert out["archived"] == 1
    result = out["results"][0]
    assert result["status"] == "archived"
    suffixes = [p for p in (vault.root / "98_Archive" / "Review" / "10_Notes").glob("shared-name-*.md")]
    assert len(suffixes) == 1
    assert occupied.read_text(encoding="utf-8") == "# previously archived copy\n"


def test_restore_roundtrip(archiver, vault):
    hash_b = scan(vault, "10_Notes/note-b.md")
    batch = archiver.archive_notes([("10_Notes/note-b.md", hash_b, "exact_duplicate")])
    listing = archiver.list_batches()
    assert listing[0]["batch_id"] == batch["batch_id"]
    assert listing[0]["restored"] == 0

    out = archiver.restore_batch(batch["batch_id"])
    assert out["restored"] == 1
    assert (vault.root / "10_Notes" / "note-b.md").exists()
    assert not (vault.root / "98_Archive" / "Review" / "10_Notes" / "note-b.md").exists()

    again = archiver.restore_batch(batch["batch_id"])
    assert again["restored"] == 0
    assert again["results"][0]["reason"] == "already restored"


def test_restore_dry_run_changes_nothing(archiver, vault):
    hash_b = scan(vault, "10_Notes/note-b.md")
    batch = archiver.archive_notes([("10_Notes/note-b.md", hash_b, "exact_duplicate")])
    out = archiver.restore_batch(batch["batch_id"], dry_run=True)
    assert out["would_restore"] == 1
    assert (vault.root / "98_Archive" / "Review" / "10_Notes" / "note-b.md").exists()
    assert not (vault.root / "10_Notes" / "note-b.md").exists()


def test_restore_skips_when_original_occupied(archiver, vault):
    hash_b = scan(vault, "10_Notes/note-b.md")
    batch = archiver.archive_notes([("10_Notes/note-b.md", hash_b, "exact_duplicate")])
    (vault.root / "10_Notes").mkdir(exist_ok=True)
    (vault.root / "10_Notes" / "note-b.md").write_text("a new note took the spot", encoding="utf-8")
    out = archiver.restore_batch(batch["batch_id"])
    assert out["restored"] == 0
    assert out["results"][0]["reason"] == "original path occupied"


def test_restore_skips_when_archived_content_changed(archiver, vault):
    hash_b = scan(vault, "10_Notes/note-b.md")
    batch = archiver.archive_notes([("10_Notes/note-b.md", hash_b, "exact_duplicate")])
    archived = vault.root / "98_Archive" / "Review" / "10_Notes" / "note-b.md"
    archived.write_text("tampered", encoding="utf-8")
    out = archiver.restore_batch(batch["batch_id"])
    assert out["results"][0]["reason"] == "archived content changed after archiving"


def test_restore_latest(archiver, vault):
    hash_b = scan(vault, "10_Notes/note-b.md")
    archiver.archive_notes([("10_Notes/note-b.md", hash_b, "exact_duplicate")])
    out = archiver.restore_batch("latest")
    assert out["restored"] == 1


def test_missing_batch_raises(archiver):
    with pytest.raises(FileNotFoundError):
        archiver.restore_batch("batch-does-not-exist")
