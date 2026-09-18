from __future__ import annotations

import pytest

from secondbrain_mcp.hashing import HashConflictError, ensure_hash_matches, sha256_file


def test_sha256_is_stable_and_hex(make_note):
    file = make_note("10_Notes/h.md", "stable content")
    assert sha256_file(file) == sha256_file(file)
    assert len(sha256_file(file)) == 64


def test_matching_expected_hash_passes(make_note):
    file = make_note("10_Notes/h.md", "stable content")
    current = sha256_file(file)
    assert ensure_hash_matches(file, current) == current


def test_missing_expected_hash_is_tolerated(make_note):
    file = make_note("10_Notes/h.md", "stable content")
    assert ensure_hash_matches(file, None) == sha256_file(file)


def test_stale_hash_rejected_with_recovery_hint(make_note):
    file = make_note("10_Notes/h.md", "before edit")
    stale = sha256_file(file)
    file.write_text("after edit", encoding="utf-8")
    with pytest.raises(HashConflictError, match="retry with the current content hash"):
        ensure_hash_matches(file, stale)
