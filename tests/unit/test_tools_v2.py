"""Tool-layer tests: the error contract, mandatory hashes, and patch semantics.

These assert what crosses the *tool* boundary, not just what the service raises —
the production failure was that a correct Python exception became an unhelpful
wire message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from secondbrain_mcp.adapters.mcp_stdio import build_registry, dispatch, tool_manifest
from secondbrain_mcp.core.config import Settings


def make_settings(tmp_path: Path) -> Settings:
    vault = tmp_path / "vault"
    for relative in (
        "00_Inbox/Web",
        "00_Inbox/Conversations",
        "40_Persona/Model",
        "40_Persona/Preferences",
        "90_Agent/review",
        "95_Private",
        "98_Archive",
    ):
        (vault / relative).mkdir(parents=True, exist_ok=True)
    (vault / "95_Private/secret.md").write_text("# s\nPRIVATE\n", encoding="utf-8")
    (vault / "40_Persona/Preferences/monthly-review.md").write_text(
        "---\nstatus: confirmed\ntags: [review]\n---\n\n# 每月检视\n\n每月复盘一次。\n",
        encoding="utf-8",
    )
    (vault / "90_Agent/review/2026-10-cleanup-review.md").write_text(
        "---\nstatus: awaiting-user\n---\n\n# cleanup\n\nPending decision.\n",
        encoding="utf-8",
    )
    (vault / "40_Persona/Model/candidates.md").write_text(
        "---\nupdated_at: 2026-09-18T00:00:00Z\n---\n\n# 待确认人格候选\n\n- [ ] skills: 进入维护期\n",
        encoding="utf-8",
    )
    (vault / "90_Agent/review/cleanup-candidates-latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-10-19T06:35:33Z",
                "candidates": [
                    {"category": "agent_generated_stale", "note": "90_Agent/x.md", "reason": "stale"}
                ],
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state"
    return Settings(
        vault_root=vault.resolve(),
        state_root=state,
        agent_label="test",
        rag_db=state / "index.db",
        audit_log=state / "logs" / "gateway-audit.jsonl",
        lock_dir=state / "locks",
    )


@pytest.fixture()
def tools(tmp_path: Path):
    registry, settings = build_registry(make_settings(tmp_path))
    dispatch("index_rebuild", {}, registry)
    return registry, settings


# ---------- tool surface ----------


def test_tool_surface_is_narrow_and_complete(tools) -> None:
    registry, _ = tools
    names = set(registry)
    for required in (
        "note_read",
        "note_list",
        "search",
        "index_status",
        "index_rebuild",
        "review_queue",
        "note_create",
        "note_append",
        "note_patch",
        "note_meta",
        "note_move",
        "archive",
        "restore",
    ):
        assert required in names, f"missing tool: {required}"
    # The old surface exposed both search_notes and rag_search, leaving callers to
    # guess which one they wanted.
    assert "search_notes" not in names
    assert "rag_search" not in names
    manifest = tool_manifest(registry)
    assert all("inputSchema" in item for item in manifest)


# ---------- error contract ----------


def test_hash_required_error_reaches_the_caller(tools) -> None:
    registry, _ = tools
    payload = dispatch(
        "note_append",
        {"note": "40_Persona/Preferences/monthly-review.md", "content": "x"},
        registry,
    )
    assert payload["isError"] is True
    assert payload["error"]["code"] == "HASH_REQUIRED"
    assert "note_read" in payload["error"]["recovery"]


def test_hash_conflict_tells_the_caller_what_to_do(tools) -> None:
    registry, _ = tools
    path = "40_Persona/Preferences/monthly-review.md"
    payload = dispatch(
        "note_append",
        {"note": path, "content": "x", "expected_hash": "deadbeef"},
        registry,
    )
    assert payload["isError"] is True
    assert payload["error"]["code"] == "HASH_CONFLICT"
    assert "note_read" in payload["error"]["recovery"]


def test_private_read_is_refused_through_the_tool_boundary(tools) -> None:
    registry, _ = tools
    for spelling in ("95_Private/secret.md", "95_private/secret.md", "95_PRIVATE/secret.md"):
        payload = dispatch("note_read", {"note": spelling}, registry)
        assert payload["isError"] is True
        assert payload["error"]["code"] == "PRIVATE_DENY"


def test_path_traversal_is_refused(tools) -> None:
    registry, _ = tools
    payload = dispatch("note_read", {"note": "../../etc/passwd"}, registry)
    assert payload["isError"] is True
    assert payload["error"]["code"] == "PATH_DENIED"


def test_unknown_tool_lists_the_available_ones(tools) -> None:
    registry, _ = tools
    payload = dispatch("does_not_exist", {}, registry)
    assert payload["isError"] is True
    assert payload["error"]["code"] == "UNKNOWN_TOOL"
    assert "note_read" in payload["error"]["recovery"]


# ---------- patch ----------


def test_patch_preview_writes_nothing(tools) -> None:
    registry, settings = tools
    note = "40_Persona/Preferences/monthly-review.md"
    before = (settings.vault_root / note).read_text(encoding="utf-8")
    read = dispatch("note_read", {"note": note}, registry)
    payload = dispatch(
        "note_patch",
        {
            "note": note,
            "expected_hash": read["hash"],
            "operation": "replace",
            "target": "每月复盘一次。",
            "content": "每月复盘两次。",
            "preview": True,
        },
        registry,
    )
    assert payload.get("isError") is not True, payload
    assert payload["preview"] is True
    assert "-每月复盘一次。" in payload["diff"]
    assert "+每月复盘两次。" in payload["diff"]
    # The file must be byte-identical after a preview.
    assert (settings.vault_root / note).read_text(encoding="utf-8") == before


def test_patch_apply_requires_the_same_hash_as_the_preview(tools) -> None:
    registry, settings = tools
    note = "40_Persona/Preferences/monthly-review.md"
    read = dispatch("note_read", {"note": note}, registry)
    preview = dispatch(
        "note_patch",
        {
            "note": note,
            "expected_hash": read["hash"],
            "operation": "replace",
            "target": "每月复盘一次。",
            "content": "每月复盘两次。",
            "preview": True,
        },
        registry,
    )
    assert preview["hash_after"] != preview["hash_before"]
    applied = dispatch(
        "note_patch",
        {
            "note": note,
            "expected_hash": preview["hash_before"],
            "operation": "replace",
            "target": "每月复盘一次。",
            "content": "每月复盘两次。",
            "preview": False,
        },
        registry,
    )
    assert applied.get("isError") is not True, applied
    assert applied["applied"] is True
    assert "每月复盘两次。" in (settings.vault_root / note).read_text(encoding="utf-8")


def test_patch_refuses_an_ambiguous_anchor(tools) -> None:
    registry, settings = tools
    note = "40_Persona/Preferences/repeated.md"
    (settings.vault_root / note).write_text("---\nstatus: inbox\n---\n\ndup\n\ndup\n", encoding="utf-8")
    read = dispatch("note_read", {"note": note}, registry)
    payload = dispatch(
        "note_patch",
        {
            "note": note,
            "expected_hash": read["hash"],
            "operation": "replace",
            "target": "dup",
            "content": "changed",
            "preview": True,
        },
        registry,
    )
    assert payload["isError"] is True
    assert payload["error"]["code"] == "INVALID_ARGUMENTS"
    assert "ambiguous" in payload["error"]["message"]


def test_patch_replaces_a_whole_section(tools) -> None:
    """A section runs until the next heading of the *same or higher* level.

    `# 标题` is H1, so the following `## 下一节` is a child of that section and is
    replaced with it. A sibling H1 would survive.
    """
    registry, settings = tools
    note = "40_Persona/Preferences/sections.md"
    (settings.vault_root / note).write_text(
        "---\nstatus: inbox\n---\n\n# 标题\n\n旧内容\n\n## 下一节\n\n子内容\n\n# 另一个标题\n\n保留\n",
        encoding="utf-8",
    )
    read = dispatch("note_read", {"note": note}, registry)
    payload = dispatch(
        "note_patch",
        {
            "note": note,
            "expected_hash": read["hash"],
            "operation": "replace_section",
            "target": "标题",
            "content": "新内容",
            "preview": False,
        },
        registry,
    )
    assert payload.get("isError") is not True, payload
    result = (settings.vault_root / note).read_text(encoding="utf-8")
    assert "新内容" in result
    assert "旧内容" not in result
    assert "子内容" not in result, "a child heading belongs to the replaced section"
    assert "# 另一个标题" in result and "保留" in result, "a sibling section must survive"


def test_patch_section_stops_at_a_sibling_heading(tools) -> None:
    registry, settings = tools
    note = "40_Persona/Preferences/siblings.md"
    (settings.vault_root / note).write_text(
        "---\nstatus: inbox\n---\n\n## 第一节\n\n旧内容\n\n## 第二节\n\n保留\n",
        encoding="utf-8",
    )
    read = dispatch("note_read", {"note": note}, registry)
    payload = dispatch(
        "note_patch",
        {
            "note": note,
            "expected_hash": read["hash"],
            "operation": "replace_section",
            "target": "第一节",
            "content": "新内容",
            "preview": False,
        },
        registry,
    )
    assert payload.get("isError") is not True, payload
    result = (settings.vault_root / note).read_text(encoding="utf-8")
    assert "新内容" in result and "旧内容" not in result
    assert "## 第二节" in result and "保留" in result


# ---------- create ----------


def test_create_never_overwrites_and_suffixes_instead(tools) -> None:
    registry, settings = tools
    first = dispatch(
        "note_create",
        {"folder": "Conversations", "filename": "note.md", "content": "first"},
        registry,
    )
    second = dispatch(
        "note_create",
        {"folder": "Conversations", "filename": "note.md", "content": "second"},
        registry,
    )
    assert first["result"] == "created"
    assert second["result"] == "created"
    assert second["created"].endswith("note-2.md")
    assert (settings.vault_root / "00_Inbox/Conversations/note.md").read_text(encoding="utf-8") == "first"


def test_create_rejects_traversal_in_the_filename(tools) -> None:
    registry, settings = tools
    payload = dispatch(
        "note_create",
        {"folder": "Web", "filename": "../../escape.md", "content": "x"},
        registry,
    )
    assert payload.get("isError") is not True, payload
    assert payload["created"].startswith("00_Inbox/Web/")
    assert not (settings.vault_root.parent / "escape.md").exists()


# ---------- archive ----------


def test_archive_then_restore_round_trip(tools) -> None:
    registry, settings = tools
    note = "40_Persona/Preferences/monthly-review.md"
    read = dispatch("note_read", {"note": note}, registry)
    planned = dispatch("archive", {"notes": [{"note": note, "hash": read["hash"]}], "dry_run": True}, registry)
    assert planned["results"][0]["status"] == "would_archive"
    assert (settings.vault_root / note).exists()

    done = dispatch("archive", {"notes": [{"note": note, "hash": read["hash"]}]}, registry)
    assert done["archived"] == 1
    assert not (settings.vault_root / note).exists()

    restored = dispatch("restore", {"batch_id": "latest"}, registry)
    assert restored["restored"] == 1
    assert (settings.vault_root / note).exists()


def test_restore_refuses_a_hostile_manifest(tools) -> None:
    """Absolute and traversing paths in a manifest must never move a file."""
    registry, settings = tools
    outside = settings.vault_root.parent / "outside.md"
    outside.write_text("OUTSIDE", encoding="utf-8")
    manifest_dir = settings.vault_root / "98_Archive/.manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "batch-hostile.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "batch-hostile",
                "entries": [
                    {"from": "00_Inbox/Web/injected.md", "to": str(outside), "hash": "x"},
                    {"from": "00_Inbox/Web/traversed.md", "to": "../../../outside.md", "hash": "x"},
                    {"from": str(outside), "to": "98_Archive/Review/x.md", "hash": "x"},
                ],
                "restores": [],
            }
        ),
        encoding="utf-8",
    )
    payload = dispatch("restore", {"batch_id": "batch-hostile"}, registry)
    assert payload["restored"] == 0
    assert all(item["status"] != "restored" for item in payload["results"])
    assert outside.read_text(encoding="utf-8") == "OUTSIDE"
    assert not (settings.vault_root / "00_Inbox/Web/injected.md").exists()


def test_restore_rejects_an_unknown_manifest_schema(tools) -> None:
    registry, settings = tools
    manifest_dir = settings.vault_root / "98_Archive/.manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "batch-old.json").write_text(
        json.dumps({"batch_id": "batch-old", "entries": []}), encoding="utf-8"
    )
    payload = dispatch("restore", {"batch_id": "batch-old"}, registry)
    assert payload["isError"] is True
    assert payload["error"]["code"] == "INVALID_ARGUMENTS"


# ---------- review ----------


def test_review_queue_surfaces_the_backlog(tools) -> None:
    registry, _ = tools
    payload = dispatch("review_queue", {}, registry)
    kinds = {item["kind"] for item in payload["items"]}
    assert "persona_candidate" in kinds, "an unconfirmed candidate must appear"
    assert "cleanup_candidate" in kinds
    persona = next(item for item in payload["items"] if item["kind"] == "persona_candidate")
    assert persona["age_days"] is not None and persona["age_days"] > 0


def test_review_queue_reports_a_missing_scan_instead_of_throwing(tools) -> None:
    registry, settings = tools
    (settings.vault_root / "90_Agent/review/cleanup-candidates-latest.json").unlink()
    payload = dispatch("review_queue", {}, registry)
    assert payload.get("isError") is not True
    assert any(item["kind"] == "cleanup_scan_missing" for item in payload["items"])


# ---------- search ----------


def test_empty_search_explains_scope(tools) -> None:
    registry, _ = tools
    payload = dispatch("search", {"query": "zzzzqqqxxx"}, registry)
    assert payload["hits"] == []
    assert payload["empty_reason"]["reason"] == "no_match_in_scope"
    assert "90_Agent" in payload["empty_reason"]["detail"]


def test_search_returns_untrusted_marker(tools) -> None:
    registry, _ = tools
    payload = dispatch("search", {"query": "每月 复盘"}, registry)
    assert payload["hits"]
    assert all(hit["trust"] == "untrusted_note" for hit in payload["hits"])


def test_index_status_reports_null_vectors(tools) -> None:
    registry, _ = tools
    payload = dispatch("index_status", {}, registry)
    assert "null_embedding_chunks" in payload
    assert payload["schema_version"] == 1
