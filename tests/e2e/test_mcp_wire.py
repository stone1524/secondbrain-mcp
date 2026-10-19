"""End-to-end MCP test: a real stdio subprocess speaking the official protocol.

This is the layer the unit tests cannot cover. The production failure that made
this necessary: the SDK collapses plain exceptions into "Error executing tool X",
so the recovery hint (and the hash) never reached the model. Only a wire-level
test can prove the error contract actually survives the protocol.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    for relative in (
        "00_Inbox/Web",
        "00_Inbox/Conversations",
        "40_Persona/Preferences",
        "40_Persona/Model",
        "90_Agent/review",
        "95_Private",
        "98_Archive",
    ):
        (vault / relative).mkdir(parents=True, exist_ok=True)
    (vault / "95_Private/secret.md").write_text("# s\nPRIVATE\n", encoding="utf-8")
    (vault / "40_Persona/Preferences/monthly-review.md").write_text(
        "---\nstatus: confirmed\n---\n\n# 每月检视\n\n每月复盘一次。\n", encoding="utf-8"
    )
    (vault / "40_Persona/Model/candidates.md").write_text(
        "---\nupdated_at: 2026-09-18T00:00:00Z\n---\n\n- [ ] skills: 维护期\n", encoding="utf-8"
    )
    return vault


class Client:
    def __init__(self, vault: Path, state: Path):
        environment = {
            **os.environ,
            "SB_VAULT_ROOT": str(vault),
            "SB_STATE_ROOT": str(state),
            "SB_RAG_SEMANTIC": "0",
            "SB_AGENT_LABEL": "e2e",
        }
        self.process = subprocess.Popen(
            [sys.executable, "-m", "secondbrain_mcp.adapters.mcp_stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        self._next_id = 1

    def request(self, method: str, params: dict | None = None) -> dict:
        identifier = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": identifier, "method": method}
        if params is not None:
            payload["params"] = params
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise AssertionError(f"server closed the connection: {stderr[-2000:]}")
        return json.loads(line)

    def call(self, name: str, arguments: dict) -> dict:
        """Return the tool payload, normalizing the two error renderings.

        A ``GatewayError`` is returned by the handler as a normal result whose
        body carries ``isError`` — deliberately, so the message and recovery
        instruction survive. The protocol layer also has its own error shape for
        schema violations. Both are normalized here.
        """
        response = self.request("tools/call", {"name": name, "arguments": arguments})
        result = response.get("result", {})
        raw = result.get("content", [{}])[0].get("text", "")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"text": raw, "isError": bool(result.get("isError"))}
        if result.get("isError"):
            payload.setdefault("text", raw)
            payload["isError"] = True
        return payload

    def close(self) -> None:
        self.process.stdin.close() if self.process.stdin else None
        self.process.wait(timeout=20)


@pytest.fixture()
def client(tmp_path: Path):
    vault = make_vault(tmp_path)
    instance = Client(vault, tmp_path / "state")
    try:
        initialized = instance.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e2e", "version": "1"},
            },
        )
        assert "result" in initialized, initialized
        yield instance, vault
    finally:
        instance.close()


def test_tool_listing_declares_real_arguments(client) -> None:
    """A generic **kwargs wrapper would publish zero arguments here."""
    instance, _ = client
    response = instance.request("tools/list")
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    assert len(tools) == 13
    assert "note" in tools["note_read"]["inputSchema"]["properties"]
    assert "expected_hash" in tools["note_append"]["inputSchema"]["properties"]
    assert "operation" in tools["note_patch"]["inputSchema"]["properties"]
    assert tools["note_patch"]["inputSchema"]["required"]
    # The old surface is gone rather than aliased.
    assert "rag_search" not in tools and "search_notes" not in tools


def test_hash_required_error_survives_the_wire(client) -> None:
    """A missing required field is rejected by the protocol layer, before the
    handler runs — that is the strongest possible form of the guarantee."""
    instance, _ = client
    result = instance.call(
        "note_append", {"note": "40_Persona/Preferences/monthly-review.md", "content": "x"}
    )
    assert result.get("isError") is True
    assert "expected_hash" in result["text"]


def test_wrong_hash_reaches_the_model_with_recovery(client) -> None:
    """The action with a recovery instruction: correct hash, stale value."""
    instance, _ = client
    result = instance.call(
        "note_append",
        {
            "note": "40_Persona/Preferences/monthly-review.md",
            "content": "x",
            "expected_hash": "0" * 64,
        },
    )
    assert result.get("isError") is True
    assert result["error"]["code"] == "HASH_CONFLICT"
    assert "note_read" in result["error"]["recovery"]


def test_private_zone_is_refused_over_the_wire(client) -> None:
    instance, _ = client
    leaked = False
    for spelling in ("95_Private/secret.md", "95_private/secret.md", "95_PRIVATE/secret.md"):
        result = instance.call("note_read", {"note": spelling})
        assert result.get("isError") is True
        # The refusal may be rendered by the protocol layer or by the handler;
        # what must never appear is the note's content.
        leaked = leaked or "\nPRIVATE\n" in result.get("text", "")
    assert leaked is False


def test_read_returns_a_usable_hash_and_trust_marker(client) -> None:
    instance, _ = client
    result = instance.call("note_read", {"note": "40_Persona/Preferences/monthly-review.md"})
    assert result.get("isError") is not True, result
    assert len(result["hash"]) == 64
    assert result["trust"] == "untrusted_note"
    assert "每月复盘" in result["content"]


def test_patch_preview_then_apply_over_the_wire(client) -> None:
    instance, vault = client
    note = "40_Persona/Preferences/monthly-review.md"
    read = instance.call("note_read", {"note": note})
    preview = instance.call(
        "note_patch",
        {
            "note": note,
            "expected_hash": read["hash"],
            "operation": "replace",
            "target": "每月复盘一次。",
            "content": "每月复盘两次。",
            "preview": True,
        },
    )
    assert preview.get("isError") is not True, preview
    assert preview["preview"] is True
    assert "每月复盘两次。" not in (vault / note).read_text(encoding="utf-8")

    applied = instance.call(
        "note_patch",
        {
            "note": note,
            "expected_hash": read["hash"],
            "operation": "replace",
            "target": "每月复盘一次。",
            "content": "每月复盘两次。",
            "preview": False,
        },
    )
    assert applied.get("isError") is not True, applied
    assert "每月复盘两次。" in (vault / note).read_text(encoding="utf-8")


def test_search_and_empty_reason_over_the_wire(client) -> None:
    instance, _ = client
    instance.call("index_rebuild", {})
    found = instance.call("search", {"query": "每月 复盘"})
    assert found.get("isError") is not True, found
    assert found["hits"]
    assert found["hits"][0]["trust"] == "untrusted_note"

    missing = instance.call("search", {"query": "zzzqqqxxx"})
    assert missing["hits"] == []
    assert missing["empty_reason"]["reason"] == "no_match_in_scope"


def test_review_queue_over_the_wire(client) -> None:
    instance, _ = client
    result = instance.call("review_queue", {})
    assert result.get("isError") is not True, result
    assert any(item["kind"] == "persona_candidate" for item in result["items"])
