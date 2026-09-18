from __future__ import annotations

import pytest

from secondbrain_mcp.paths import PrivateNoteError, VaultPathError, VaultPaths


def test_relative_path_resolves_inside_vault(vault: VaultPaths):
    resolved = vault.resolve("10_Notes/note-b.md")
    assert resolved == vault.root / "10_Notes" / "note-b.md"
    assert vault.relative(resolved) == "10_Notes/note-b.md"


def test_absolute_path_rejected(vault: VaultPaths, tmp_path):
    with pytest.raises(VaultPathError):
        vault.resolve(str(tmp_path / "outside.md"))


def test_rooted_path_rejected(vault: VaultPaths):
    with pytest.raises(VaultPathError):
        vault.resolve("/etc/passwd")


def test_escape_rejected(vault: VaultPaths):
    with pytest.raises(VaultPathError):
        vault.resolve("../outside.md")
    with pytest.raises(VaultPathError):
        vault.resolve("10_Notes/../../outside.md")


def test_empty_path_rejected(vault: VaultPaths):
    with pytest.raises(VaultPathError):
        vault.resolve("")
    with pytest.raises(VaultPathError):
        vault.resolve("   ")


def test_private_zone_detection(vault: VaultPaths):
    assert vault.is_private("95_Private/secrets.md")
    assert not vault.is_private("10_Notes/note-b.md")
    with pytest.raises(PrivateNoteError):
        vault.assert_readable(vault.resolve("95_Private/secrets.md"))
    vault.assert_readable(vault.resolve("10_Notes/note-b.md"))


def test_private_move_destination_rejected(vault: VaultPaths):
    destination = vault.resolve("95_Private/hideout.md")
    with pytest.raises(PrivateNoteError):
        vault.assert_move_destination(destination)


def test_custom_private_prefixes(tmp_path):
    paths = VaultPaths(tmp_path, private_prefixes=("Secret", "Nests/95"))
    (tmp_path / "Secret").mkdir()
    assert paths.is_private("Secret/a.md")
    assert paths.is_private("Nests/95/a.md")
    assert not paths.is_private("Nests/94/a.md")
