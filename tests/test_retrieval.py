from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from secondbrain_mcp.audit import AuditLog
from secondbrain_mcp.hashing import sha256_file
from secondbrain_mcp.paths import VaultPathError, VaultPaths
from secondbrain_mcp.retrieval import Retriever, chunks_for, tokenize

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_retriever(tmp_path: Path) -> Retriever:
    paths = VaultPaths(REPO_ROOT / "demo-vault")
    return Retriever(paths, tmp_path / "demo.db")


# ---------- tokenization ----------


def test_tokenize_lowercases_and_dedupes():
    assert tokenize("Tea House tea") == ["house", "tea"]


def test_tokenize_cjk_produces_bigrams():
    tokens = tokenize("知识库")
    assert tokens == ["知识", "知识库", "识库"]


def test_tokenize_drops_single_characters():
    assert tokenize("a 翠 鸟 ab") == ["ab"]
    # 翠 and 鸟 are single CJK characters: no bigram can be formed.


def test_tokenize_keeps_hyphen_inside_terms():
    assert tokenize("8x42 spotting-scope") == ["8x42", "spotting-scope"]


# ---------- chunking ----------


def test_chunks_split_on_headings():
    content = "# One\nalpha\n## Two\nbeta\ngamma"
    chunks = chunks_for(content)
    assert [(c["heading"], c["content"]) for c in chunks] == [
        ("One", "alpha"),
        ("Two", "beta gamma"),
    ]


def test_chunks_flush_on_length_limit():
    content = "# Long\n" + "\n".join(["word"] * 500)
    chunks = chunks_for(content)
    assert len(chunks) >= 2
    assert all(c["content"] for c in chunks)


# ---------- incremental indexing ----------


def test_index_first_and_second_run(vault, tmp_path):
    retriever = Retriever(vault, tmp_path / "index.db")
    first = retriever.index()
    # 5 notes walked, 1 private note hard-excluded.
    assert first["indexed"] == 4
    assert first["skipped"] == 1

    second = retriever.index()
    assert second["indexed"] == 0
    assert second["unchanged"] == 4


def test_index_reindexes_edited_note(vault, tmp_path):
    retriever = Retriever(vault, tmp_path / "index.db")
    retriever.index()
    file = vault.root / "10_Notes" / "note-b.md"
    file.write_text("# Note B\n\nEdited body with new content.\n", encoding="utf-8")
    out = retriever.index()
    assert out["indexed"] == 1
    assert out["unchanged"] == 3


def test_index_deactivates_vanished_note(vault, tmp_path):
    retriever = Retriever(vault, tmp_path / "index.db")
    retriever.index()
    (vault.root / "20_Reference" / "shared-name.md").unlink()
    out = retriever.index()
    assert out["deactivated"] == 1
    hits = retriever.search("shared name")
    assert all(hit["source"] != "20_Reference/shared-name.md" for hit in hits)


def test_index_heals_hollow_document(vault, tmp_path):
    retriever = Retriever(vault, tmp_path / "index.db")
    retriever.index()
    db = sqlite3.connect(retriever.db_path)
    db.execute("UPDATE chunks SET active=0 WHERE path='10_Notes/note-b.md'")
    db.commit()
    db.close()
    out = retriever.index()
    assert out["healed"] == 1
    assert retriever.search("synthetic demo body")[0]["source"] == "10_Notes/note-b.md"


# ---------- privacy and exclusion ----------


def test_private_zone_is_never_searchable(demo_retriever):
    demo_retriever.index()
    default_probe = demo_retriever.search("私人日记")
    explicit_probe = demo_retriever.search("私人日记", paths=["95_Private"])
    assert default_probe == []
    assert explicit_probe == []


def test_archive_area_excluded_from_results(demo_retriever):
    demo_retriever.index()
    assert all(
        not hit["source"].startswith("98_Archive/") for hit in demo_retriever.search("旧清单 归档")
    )


def test_regenerated_persona_views_excluded(demo_retriever):
    demo_retriever.index()
    for query in ("待确认候选", "协会角色审计"):
        assert all(
            not hit["source"].startswith("40_Persona/Model/") for hit in demo_retriever.search(query)
        )


def test_dot_obsidian_directory_skipped(vault, tmp_path):
    (vault.root / ".obsidian").mkdir(exist_ok=True)
    (vault.root / ".obsidian" / "plugins.md").write_text("# plugins\n", encoding="utf-8")
    retriever = Retriever(vault, tmp_path / "index.db")
    retriever.index()
    assert all(not hit["source"].startswith(".obsidian") for hit in retriever.search("plugins"))


# ---------- scope model ----------


def test_auxiliary_areas_skipped_by_default_and_enabled_explicitly(demo_retriever):
    demo_retriever.index()
    assert all(
        not hit["source"].startswith("30_Resources/FileIndex/")
        for hit in demo_retriever.search("sighting")
    )
    scoped = demo_retriever.search("sighting", paths=["30_Resources/FileIndex"])
    assert any(hit["source"] == "30_Resources/FileIndex/sightings-2025.csv.md" for hit in scoped)


def test_scope_prefix_narrows_results(demo_retriever):
    demo_retriever.index()
    hits = demo_retriever.search("望远镜", paths=["10_Notes"])
    assert hits
    assert all(hit["source"].startswith("10_Notes/") for hit in hits)


def test_scope_rejects_escape_and_absolute(tmp_path):
    retriever = Retriever(VaultPaths(tmp_path), "index.db")
    with pytest.raises(VaultPathError):
        retriever.search("x", paths=["../evil"])
    with pytest.raises(VaultPathError):
        retriever.search("x", paths=["/abs"])


# ---------- ranking ----------


def test_cjk_bigram_query_finds_note(demo_retriever):
    demo_retriever.index()
    hits = demo_retriever.search("望远镜")
    assert hits
    assert hits[0]["source"] == "10_Notes/设备清单.md"
    assert any(hit["source"] == "00_Inbox/Web/镜头清洁指南.md" for hit in hits)


def test_bilingual_glossary_adds_recall_never_loses(demo_retriever):
    demo_retriever.index()
    # ZH query, EN note: the glossary expansion bridges the language boundary.
    hits = demo_retriever.search("多久浇一次水")
    assert any(hit["source"] == "10_Notes/watering-schedule.md" for hit in hits)
    # Original wording is always kept: an exact-match note is never displaced by
    # an over-eager expansion entry.
    exact = demo_retriever.search("设备清单")
    assert exact and exact[0]["source"] == "10_Notes/设备清单.md"


def test_phrase_bonus_ranks_contiguous_match_first(vault, make_note, tmp_path):
    make_note("10_Notes/hawk-a.md", "red tailed hawk")
    make_note("10_Notes/hawk-b.md", "hawk tailed red")
    retriever = Retriever(vault, tmp_path / "index.db")
    retriever.index()
    hits = retriever.search("red tailed hawk")
    sources = [hit["source"] for hit in hits]
    assert sources.index("10_Notes/hawk-a.md") < sources.index("10_Notes/hawk-b.md")
    assert hits[0]["lexical_score"] - hits[1]["lexical_score"] == pytest.approx(0.5)


def test_per_note_best_chunk_dedup(vault, make_note, tmp_path):
    make_note(
        "10_Notes/multi.md",
        "# Kingfisher\n翠鸟在北岸。\n## Second heading\n又见到翠鸟。",
    )
    retriever = Retriever(vault, tmp_path / "index.db")
    retriever.index()
    hits = retriever.search("翠鸟")
    assert [hit["source"] for hit in hits].count("10_Notes/multi.md") == 1


def test_empty_query_returns_nothing(demo_retriever):
    assert demo_retriever.search("   ") == []


# ---------- result shape and hygiene ----------


def test_result_carries_traceability_fields(demo_retriever):
    demo_retriever.index()
    hit = demo_retriever.search("设备清单")[0]
    assert hit["source"] == "10_Notes/设备清单.md"
    assert hit["content_hash"] == sha256_file(REPO_ROOT / "demo-vault" / "10_Notes" / "设备清单.md")
    assert hit["updated_at"]
    assert "设备清单" in hit["matched_terms"]
    assert hit["excerpt"]


def test_status_reflects_index(demo_retriever):
    stats = demo_retriever.index()
    status = demo_retriever.status()
    assert status["documents"] == stats["indexed"] == 9
    assert status["chunks"] == stats["chunks"] > 0
    assert status["generation"] == 1
    # Two FileIndex stubs are auxiliary.
    assert status["auxiliary_documents"] == 2


def test_audit_log_receives_index_and_search(demo_retriever, tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl", agent_label="pytest")
    retriever = Retriever(demo_retriever.paths, tmp_path / "demo.db", audit_log=audit)
    retriever.index()
    retriever.search("望远镜")
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    operations = [json.loads(line)["operation"] for line in lines]
    assert operations == ["rag_index", "rag_search"]
