"""Append-only JSONL audit log with per-agent attribution."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .locking import now_iso


class AuditLog:
    def __init__(self, log_path: Path | str, agent_label: str | None = None):
        self.log_path = Path(log_path)
        self.agent_label = agent_label or os.environ.get("SB_AGENT_LABEL")

    def record(self, operation: str, note: str, **details) -> None:
        entry = {
            "at": now_iso(),
            "agent": self.agent_label,
            "operation": operation,
            "note": note,
            **details,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
