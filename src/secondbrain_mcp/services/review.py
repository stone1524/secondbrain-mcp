"""Review queue: surface what is waiting on a human.

The production system accumulated decisions nobody ever saw: two persona
candidates unconfirmed since 2026-09-18, every cleanup review still marked
``awaiting-user`` from 2026-08, and nothing in any tool's output that would
prompt someone to look. A queue tool makes the backlog visible at the moment a
conversation touches the knowledge base, instead of waiting in a folder.

This module reads only; confirmation happens through the normal write tools.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PERSONA_CANDIDATES = "40_Persona/Model/candidates.md"
CLEANUP_LATEST = "90_Agent/review/cleanup-candidates-latest.json"
INBOX_PREFIX = "00_Inbox/"

PENDING_MARKER = re.compile(r"^- \[ \]\s*(.+)$", re.MULTILINE)
STATUS_LINE = re.compile(r"^status:\s*(.+)$", re.MULTILINE)


@dataclass(slots=True)
class ReviewService:
    root: Path
    inbox_aging_days: int = 90

    def queue(self, *, older_than_days: int = 0, limit: int = 50) -> dict:
        items: list[dict] = []
        items.extend(self._persona_candidates())
        items.extend(self._cleanup_candidates())
        items.extend(self._aging_inbox(older_than_days=older_than_days))
        # Items with an unknown age (a missing scan) sort last rather than
        # crashing the comparison against integer ages.
        items.sort(
            key=lambda item: (
                item.get("age_days") is None,
                -(item.get("age_days") or 0),
            )
        )
        now = datetime.now(timezone.utc)
        for item in items:
            item.setdefault("age_days", None)
        return {
            "generated_at": now.isoformat(),
            "total": len(items),
            "count": min(len(items), max(1, limit)),
            "items": items[: max(1, limit)],
        }

    # ---------- sources ----------

    def _persona_candidates(self) -> list[dict]:
        file = self.root / PERSONA_CANDIDATES
        if not file.exists():
            return []
        content = file.read_text(encoding="utf-8")
        updated = _frontmatter_value(content, "updated_at")
        age = _age_days(updated)
        return [
            {
                "kind": "persona_candidate",
                "summary": match.group(1).strip(),
                "source": PERSONA_CANDIDATES,
                "age_days": age,
                # The action must name a tool that exists. The previous text
                # pointed at "personality tools" the Python core never
                # implemented, so the two oldest decisions in the vault had no
                # way to be resolved by any caller — visible, and permanently
                # stuck.
                "action": (
                    "Decide, then record it with note_patch on "
                    f"{PERSONA_CANDIDATES} (tick the checkbox), passing "
                    "expected_hash from note_read"
                ),
            }
            for match in PENDING_MARKER.finditer(content)
        ]

    def _cleanup_candidates(self) -> list[dict]:
        file = self.root / CLEANUP_LATEST
        if not file.exists():
            # A missing scan is a state worth reporting, not an exception: the
            # production tool simply threw and told the caller to run a script.
            return [
                {
                    "kind": "cleanup_scan_missing",
                    "summary": "Nightly cleanup scan has not produced a candidate list",
                    "source": CLEANUP_LATEST,
                    "age_days": None,
                    "action": "Run the nightly collector, or ignore if cleanup is not in use",
                }
            ]
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return [
                {
                    "kind": "cleanup_scan_invalid",
                    "summary": "Cleanup candidate file is not valid JSON",
                    "source": CLEANUP_LATEST,
                    "age_days": None,
                    "action": "Re-run the nightly collector",
                }
            ]
        generated = data.get("generated_at")
        age = _age_days(generated)
        items: list[dict] = []
        for candidate in data.get("candidates", [])[:50]:
            items.append(
                {
                    "kind": "cleanup_candidate",
                    "summary": f"{candidate.get('category')}: {candidate.get('note')}",
                    "source": CLEANUP_LATEST,
                    "age_days": age,
                    "action": "Archive with archive() after review, or leave in place",
                    "detail": {
                        "category": candidate.get("category"),
                        "reason": candidate.get("reason"),
                    },
                }
            )
        return items

    def _aging_inbox(self, *, older_than_days: int) -> list[dict]:
        inbox = self.root / INBOX_PREFIX
        if not inbox.exists():
            return []
        threshold = older_than_days if older_than_days > 0 else self.inbox_aging_days
        items: list[dict] = []
        for file in inbox.rglob("*.md"):
            content = file.read_text(encoding="utf-8", errors="replace")
            status = (_frontmatter_value(content, "status") or "").strip().lower()
            if status and status != "inbox":
                continue
            modified = datetime.fromtimestamp(file.stat().st_mtime, timezone.utc)
            age = (datetime.now(timezone.utc) - modified).days
            if age < threshold:
                continue
            items.append(
                {
                    "kind": "inbox_aging",
                    "summary": file.relative_to(self.root).as_posix(),
                    "source": "00_Inbox",
                    "age_days": age,
                    "action": "Promote into a project/area, or archive if superseded",
                }
            )
        return items


def _frontmatter_value(content: str, key: str) -> str | None:
    if not content.startswith("---"):
        return None
    end = content.find("\n---", 3)
    if end < 0:
        return None
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(content[:end])
    return match.group(1).strip().strip('"').strip("'") if match else None


def _age_days(stamp: str | None) -> int | None:
    if not stamp:
        return None
    normalized = stamp.replace("Z", "+00:00")
    try:
        then = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - then).days)


_ = STATUS_LINE
