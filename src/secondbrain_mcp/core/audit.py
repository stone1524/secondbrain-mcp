"""Hash-chained audit trail with intent/outcome pairs.

The previous audit wrote one line *before* the mutation and no line after it, so
a rejected write was indistinguishable from an accepted one, and there was no
fsync. Three changes:

* every mutation records ``intent`` then ``outcome`` (with ``ok`` or an error code);
* each record carries ``prev`` — the SHA-256 of the previous record — so a
  deleted or reordered line is detectable;
* appends are fsynced.

This is tamper-*evident*, not tamper-*proof*: the threat model is an agent
mishap or a duplicated write, not a local attacker rewriting history.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .atomic import append_durable


class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, log_path: Path | str, agent_label: str | None = None):
        self.log_path = Path(log_path)
        self.agent_label = agent_label
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.log_path.exists():
            return self.GENESIS
        last = self.GENESIS
        with self.log_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line).get("hash", last)
                except json.JSONDecodeError:
                    # A torn tail from an older, unfsynced writer.
                    continue
        return last

    def record(self, operation: str, note: str, *, phase: str = "intent", **details) -> str:
        # Reserved keys are set below and cannot be overridden by a caller's
        # detail payload; a collision would otherwise be a TypeError at runtime.
        extra = {
            key: value
            for key, value in details.items()
            if key not in ("operation", "phase", "note", "at", "agent", "prev", "hash")
        }
        payload = {
            "at": datetime.now(timezone.utc).isoformat(),
            "agent": self.agent_label,
            "operation": operation,
            "phase": phase,
            "note": note,
            **extra,
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = sha256(f"{self._last_hash}{body}".encode()).hexdigest()
        payload["prev"] = self._last_hash
        payload["hash"] = digest
        append_durable(self.log_path, json.dumps(payload, ensure_ascii=False) + "\n")
        self._last_hash = digest
        return digest

    def verify(self) -> tuple[bool, str | None]:
        """Recompute the chain. Returns (ok, first_bad_line_operation)."""
        expected = self.GENESIS
        if not self.log_path.exists():
            return True, None
        with self.log_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return False, "unparseable"
                claimed = record.pop("hash", None)
                prev = record.pop("prev", None)
                body = json.dumps(record, ensure_ascii=False, sort_keys=True)
                actual = sha256(f"{expected}{body}".encode()).hexdigest()
                if prev != expected or claimed != actual:
                    return False, str(record.get("operation"))
                expected = claimed
        return True, None
