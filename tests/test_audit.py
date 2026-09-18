from __future__ import annotations

import json

from secondbrain_mcp.audit import AuditLog


def test_record_appends_jsonl_with_agent_label(tmp_path, monkeypatch):
    monkeypatch.setenv("SB_AGENT_LABEL", "pytest")
    log = AuditLog(tmp_path / "logs" / "audit.jsonl")
    log.record("read_note", "10_Notes/a.md", scope_decision="standard-vault")
    log.record("append_note", "10_Notes/a.md")
    lines = (tmp_path / "logs" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert entries[0]["agent"] == "pytest"
    assert entries[0]["operation"] == "read_note"
    assert entries[0]["scope_decision"] == "standard-vault"
    assert entries[1]["operation"] == "append_note"


def test_explicit_label_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SB_AGENT_LABEL", "from-env")
    log = AuditLog(tmp_path / "audit.jsonl", agent_label="explicit")
    log.record("rag_search", "(vault)")
    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert entry["agent"] == "explicit"


def test_label_can_be_none(tmp_path, monkeypatch):
    monkeypatch.delenv("SB_AGENT_LABEL", raising=False)
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("read_note", "a.md")
    entry = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert entry["agent"] is None
