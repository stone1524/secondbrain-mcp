"""Shared fixtures: a synthetic demo vault with no real personal content."""

from __future__ import annotations

from pathlib import Path

import pytest

from secondbrain_mcp.paths import VaultPaths

SYNTHETIC_NOTES = {
    "00_Inbox/Conversations/note-a.md": (
        "---\nstatus: draft\nagent_generated: false\n---\n# Note A\n\nSynthetic demo content.\n"
    ),
    "10_Notes/note-b.md": "# Note B\n\nSynthetic demo body for tests.\n",
    "10_Notes/shared-name.md": "# Shared name in 10_Notes\n",
    "20_Reference/shared-name.md": "# Shared name in 20_Reference\n",
    "95_Private/secrets.md": "# Private\n\nThis content must never leave the private zone.\n",
}


@pytest.fixture
def vault(tmp_path: Path) -> VaultPaths:
    root = tmp_path / "vault"
    for rel, content in SYNTHETIC_NOTES.items():
        file = root / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    return VaultPaths(root)


@pytest.fixture
def make_note(vault: VaultPaths):
    def _make(rel: str, content: str | None = None) -> Path:
        file = vault.root / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        body = content if content is not None else f"# Synthetic {rel}\n\nDemo body.\n"
        file.write_text(body, encoding="utf-8")
        return file

    return _make
