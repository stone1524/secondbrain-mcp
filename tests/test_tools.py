from __future__ import annotations

import json

import pytest

from secondbrain_mcp.paths import PrivateNoteError
from secondbrain_mcp.tools import Gateway


@pytest.fixture
def gateway(vault, tmp_path):
    return Gateway(vault.root, db_path=tmp_path / "index.db", log_dir=tmp_path / "logs", agent_label="pytest")


def test_read_note_returns_hash_and_pagination(gateway):
    out = gateway.read_note("00_Inbox/Conversations/note-a.md")
    assert "Note A" in out["content"]
    assert len(out["hash"]) == 64
    assert out["truncated"] is False

    window = gateway.read_note("00_Inbox/Conversations/note-a.md", offset=0, max_length=5)
    assert window["truncated"] is True


def test_read_note_private_rejected(gateway):
    with pytest.raises(PrivateNoteError):
        gateway.read_note("95_Private/secrets.md")


def test_create_inbox_note_ok_and_never_overwrites(gateway):
    created = gateway.create_inbox_note("Conversations", "meeting.md", "# Meeting\n")
    assert created["created"] == "00_Inbox/Conversations/meeting.md"
    with pytest.raises(ValueError, match="never overwrites"):
        gateway.create_inbox_note("Conversations", "meeting.md", "# Again\n")


def test_create_inbox_note_rejects_bad_folder_and_non_md(gateway):
    with pytest.raises(ValueError):
        gateway.create_inbox_note("10_Notes", "x.md", "x")
    with pytest.raises(ValueError, match="Markdown"):
        gateway.create_inbox_note("Web", "x.txt", "x")


def test_append_note_hash_lock(gateway):
    gateway.create_inbox_note("Web", "log.md", "line1\n")
    first = gateway.read_note("00_Inbox/Web/log.md")
    out = gateway.append_note("00_Inbox/Web/log.md", "line2", expected_hash=first["hash"])
    assert len(out["hash"]) == 64
    with pytest.raises(RuntimeError, match="retry with the current content hash"):
        gateway.append_note("00_Inbox/Web/log.md", "line3", expected_hash=first["hash"])


def test_update_metadata_allowlist_and_apply(gateway, make_note):
    make_note("10_Notes/meta.md", "---\nstatus: draft\nsecret: keep\n---\n# Meta\n")
    out = gateway.update_metadata(
        "10_Notes/meta.md",
        {"status": "reviewed", "evil_key": "x"},
    )
    assert out["fields"] == {"status": "reviewed"}
    assert out["dropped"] == ["evil_key"]
    content = (gateway.paths.root / "10_Notes" / "meta.md").read_text(encoding="utf-8")
    assert "status: \"reviewed\"" in content
    assert "secret: keep" in content


def test_update_metadata_requires_frontmatter(gateway, make_note):
    make_note("10_Notes/plain.md", "# no frontmatter\n")
    with pytest.raises(ValueError, match="frontmatter"):
        gateway.update_metadata("10_Notes/plain.md", {"status": "x"})


def test_move_note_and_checks(gateway, vault):
    out = gateway.move_note("10_Notes/note-b.md", "20_Reference/note-b.md")
    assert out["destination"] == "20_Reference/note-b.md"
    with pytest.raises(ValueError, match="already exists"):
        gateway.move_note("00_Inbox/Conversations/note-a.md", "20_Reference/note-b.md")
    with pytest.raises(PrivateNoteError):
        gateway.move_note("00_Inbox/Conversations/note-a.md", "95_Private/hide.md")


def test_archive_and_restore_roundtrip_via_gateway(gateway, vault):
    from secondbrain_mcp.hashing import sha256_file

    file = vault.root / "10_Notes" / "note-b.md"
    hash_b = sha256_file(file)
    batch = gateway.archive_notes([{"note": "10_Notes/note-b.md", "hash": hash_b, "category": "exact_duplicate"}])
    assert batch["archived"] == 1
    assert batch["rollback_hint"]
    assert not file.exists()

    listing = gateway.list_archives()
    assert listing["manifests"][0]["restored"] == 0

    restored = gateway.restore_archive("latest")
    assert restored["restored"] == 1
    assert file.exists()


def test_search_falls_back_to_linear_scan_before_first_index(gateway):
    out = gateway.search_notes("Synthetic demo body")
    assert out["mode"] == "linear-fallback"
    assert any(hit["note"] == "10_Notes/note-b.md" for hit in out["hits"])


def test_search_and_reindex_after_indexing(gateway):
    gateway.rag_reindex()
    out = gateway.rag_search("synthetic demo body")
    assert out["hits"][0]["source"] == "10_Notes/note-b.md"
    assert out["requires_read_note_for_full_evidence"] is True
    assert gateway.rag_status()["documents"] >= 4


def test_audit_log_written(gateway, tmp_path):
    gateway.read_note("10_Notes/note-b.md")
    audit_file = tmp_path / "logs" / "gateway-audit.jsonl"
    entries = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["agent"] == "pytest"
    assert entries[0]["operation"] == "read_note"
