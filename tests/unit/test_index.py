"""Indexing and retrieval tests, including the two production defects they encode."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from secondbrain_mcp.index.embed import LexicalOnlyEmbedder, ModelInfo
from secondbrain_mcp.index.indexer import Indexer, index_vault
from secondbrain_mcp.index.schema import status
from secondbrain_mcp.index.searcher import Query, Searcher
from secondbrain_mcp.index.text import body_length, chunk_note, tokenize


class FakeEmbedder:
    """Deterministic embeddings: identical text gives identical vectors.

    The real model is not required to test ranking, gates, or the fingerprint
    contract; using a fake keeps CI offline and fast.
    """

    def __init__(self, dim: int = 8, *, fail_on: str | None = None):
        self._info = ModelInfo(name="fake", dim=dim, revision="test", prefix_scheme="plain")
        self.fail_on = fail_on

    @property
    def info(self) -> ModelInfo:
        return self._info

    def _vector(self, text: str) -> np.ndarray:
        seed = sum(ord(character) for character in text) % 997
        generator = np.random.default_rng(seed)
        vector = generator.random(self._info.dim).astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    def encode_passages(self, texts):
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise RuntimeError("simulated embedding failure")
        return np.vstack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    notes = {
        "40_Persona/Preferences/monthly-review.md": "# 每月检视\n\n每月做一次复盘，检查目标与偏好是否仍然成立。\n",
        "90_Agent/research/2026-09-18-rag-model-pilot-report.md": (
            "# RAG model pilot\n\n"
            "The pilot compared multilingual-e5-large against bge-small-zh-v1.5 under\n"
            "L2 normalization. Recall@5 was the primary criterion.\n"
        ),
        "00_Inbox/Conversations/2026-09-19-planning.md": (
            "# Planning notes\n\nEmbedded work as the main line, with AI as a "
            "differentiating second track. One primary project and one supporting one.\n"
        ),
        "95_Private/Persona/personality.json.md": "# secret\n\nPRIVATE CONTENT\n",
        "98_Archive/Review/old.md": "# 旧笔记\n\n归档区里的历史内容。\n",
        "30_Resources/FileIndex/auto-knowledge-index.md": "# index\n\n",
    }
    for rel, content in notes.items():
        target = vault / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    # A placeholder persona note: frontmatter-only, the shape that caused the
    # boilerplate attractor in production.
    placeholder = vault / "40_Persona/Model/current-profile.md"
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text("---\ntype: snapshot\nstate: placeholder\n---\n", encoding="utf-8")
    return vault


def build(tmp_path: Path, embedder=None):
    vault = make_vault(tmp_path)
    db = tmp_path / "index.db"
    report = index_vault(vault, db, embedder=embedder or LexicalOnlyEmbedder())
    return vault, db, report


# ---------- text ----------


def test_cjk_is_tokenized_into_bigrams() -> None:
    tokens = tokenize("每月检视")
    assert "每月" in tokens and "月检" in tokens and "检视" in tokens


def test_mixed_script_term_splits_into_runs() -> None:
    """`Watering的内容` must index as Latin `watering` plus CJK bigrams. Left
    whole, it matches nothing on the document side — every document token is one
    script — and the most common query shape in a bilingual vault
    (English title wrapped in Chinese filler) is lexically dead."""
    tokens = tokenize("关于Watering Schedule的内容")
    assert "watering" in tokens
    assert "schedule" in tokens
    assert "内容" in tokens
    assert "关于" in tokens
    # and the document side produces the same Latin terms
    doc_tokens = tokenize("Watering Schedule")
    assert "watering" in doc_tokens and "schedule" in doc_tokens


def test_hyphenated_term_matches_spaced_query() -> None:
    """A file named 01-end-to-end.md queried as '01 end to end' shares no
    single term: one side is hyphen-joined, the other is words. The hyphen
    segments must be indexed alongside the whole."""
    doc_tokens = tokenize("end-to-end")
    query_tokens = tokenize("end to end")
    assert {"end", "to"} <= set(doc_tokens)
    assert {"end", "to"} <= set(query_tokens)


def test_filename_stem_joins_the_indexed_surface() -> None:
    """Users name notes first and type those names into search second. A note
    whose H1 is Chinese and whose *file stem* is English was unfindable by the
    English name — the stem must contribute terms like the title does."""
    chunks = chunk_note("# PCB 设计标准\n\n正文内容若干若干。\n", filename="00-index")
    assert chunks
    assert "index" in chunks[0].tokens
    assert "00" in chunks[0].tokens


def test_heading_only_note_still_yields_a_chunk() -> None:
    """Production produced zero chunks here, so the note was invisible *and*
    re-indexed on every single run."""
    chunks = chunk_note("# 只有标题\n")
    assert chunks, "a heading-only note must produce a chunk"
    assert chunks[0].heading == "只有标题"


def test_frontmatter_only_note_does_not_vanish() -> None:
    chunks = chunk_note("---\ntype: snapshot\n---\n")
    assert chunks, "a frontmatter-only note must still be indexable"


def test_body_length_ignores_frontmatter() -> None:
    assert body_length("---\ntype: x\n---\n") == 0
    assert body_length("---\ntype: x\n---\nhello") == 5


def test_chunk_tokens_include_the_document_title() -> None:
    """The postings table is written from Chunk.tokens. If a consumer ever
    re-tokenizes (heading, content) instead, an H1-over-H2 note loses its title
    from the index again — the token_count would still include it, so nothing
    downstream would notice."""
    chunks = chunk_note("# 从开题到成品的端到端流程\n\n## 准备\n调研与选题的工作记录。\n")
    assert chunks
    for chunk in chunks:
        assert "端到端" in chunk.tokens or "端到" in chunk.tokens or "到端" in chunk.tokens
        assert set(dict(chunk.term_frequencies)) == set(chunk.tokens)


# ---------- indexing ----------


def test_private_and_archive_are_never_indexed_but_auxiliary_is(tmp_path: Path) -> None:
    _, db, _ = build(tmp_path)
    rows = {row["path"] for row in _documents(db)}
    assert "95_Private/Persona/personality.json.md" not in rows
    assert "98_Archive/Review/old.md" not in rows
    assert "90_Agent/research/2026-09-18-rag-model-pilot-report.md" in rows
    # The generated aggregate is excluded entirely: it would out-rank the notes
    # it indexes for almost any query.
    assert "30_Resources/FileIndex/auto-knowledge-index.md" not in rows


def _documents(db: Path):
    import sqlite3

    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute("SELECT path, embed_state FROM documents WHERE active=1").fetchall()
    finally:
        connection.close()


def test_second_run_is_incremental(tmp_path: Path) -> None:
    vault, db, first = build(tmp_path)
    assert first.indexed > 0
    report = Indexer(vault, db).rebuild()
    assert report.indexed == 0
    assert report.unchanged > 0


def test_missing_vectors_are_retried_on_the_next_pass(tmp_path: Path) -> None:
    """The production 'unchanged' test looked at the hash only, so a failed
    embedding run left NULL vectors forever."""
    vault = make_vault(tmp_path)
    db = tmp_path / "index.db"
    index_vault(vault, db, embedder=FakeEmbedder(fail_on="multilingual-e5-large"))
    failed = status(db)
    assert failed["documents_failed_embedding"] >= 1
    # A FAILED document's chunks count as null precisely because they should have
    # vectors; lexical-only documents are excluded from this metric by design.
    assert failed["null_embedding_chunks"] > 0

    report = index_vault(vault, db, embedder=FakeEmbedder())
    assert report.indexed >= 1, "the failed document must be re-processed"
    healthy = status(db)
    assert healthy["documents_failed_embedding"] == 0
    # Lexical-only documents (generated views, file stubs) legitimately carry no
    # vector; what must not persist is a document that *should* be embedded.
    import sqlite3

    connection = sqlite3.connect(db)
    try:
        unembedded = connection.execute(
            "SELECT c.path FROM chunks c WHERE c.active = 1"
            " AND c.id NOT IN (SELECT chunk_id FROM embeddings)"
        ).fetchall()
    finally:
        connection.close()
    assert all(path.startswith(("30_Resources/FileIndex", "40_Persona/Model")) for (path,) in unembedded)


def test_changed_model_fingerprint_drops_stale_vectors(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    db = tmp_path / "index.db"
    index_vault(vault, db, embedder=FakeEmbedder(dim=8))
    before = status(db)
    assert before["embedded_chunks"] > 0

    other = FakeEmbedder(dim=4)
    other._info = ModelInfo(name="fake-other", dim=4, revision="test", prefix_scheme="plain")
    index_vault(vault, db, embedder=other)
    after = status(db)
    # Every document that should carry a vector must have been re-embedded under
    # the new fingerprint; nothing may be left from the old model.
    assert after["documents_failed_embedding"] == 0
    import sqlite3

    connection = sqlite3.connect(db)
    try:
        fingerprints = {
            row[0] for row in connection.execute("SELECT DISTINCT fingerprint FROM embeddings")
        }
    finally:
        connection.close()
    assert fingerprints == {other.info.fingerprint}


def test_deleted_note_is_deactivated(tmp_path: Path) -> None:
    vault, db, _ = build(tmp_path)
    (vault / "00_Inbox/Conversations/2026-09-19-planning.md").unlink()
    report = Indexer(vault, db).rebuild()
    assert report.deactivated >= 1
    rows = {row["path"] for row in _documents(db)}
    assert "00_Inbox/Conversations/2026-09-19-planning.md" not in rows


# ---------- retrieval ----------


def test_title_holder_beats_cross_referencer(tmp_path: Path) -> None:
    """The forensic pattern: a long meta-review that *quotes* a note's title
    accumulated per-chunk BM25 and outranked the note that *is* that title.
    A structural phrase match — the query equals the H1 or the file stem — is
    identity evidence and must win the fusion regardless of what the body
    contains or which channels the rival cleared."""
    vault = tmp_path / "vault"
    (vault / "10_Notes").mkdir(parents=True)
    (vault / "90_Agent").mkdir()
    (vault / "10_Notes" / "topic.md").write_text(
        "# 项目周记\n\n每周记录一次进度与风险。\n", encoding="utf-8"
    )
    review = "\n\n".join(
        f"## 第 {n} 节\n\n评审里引用了《项目周记》并展开讨论，这一段的正文相当长，"
        "反复出现项目周记四个字以便累积词法得分，越写越长越好。"
        for n in range(8)
    )
    (vault / "10_Notes" / "review.md").write_text(f"# 季度评审\n\n{review}\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index_vault(vault, db)
    hits, _ = Searcher(db).search(Query(text="项目周记"))
    assert hits
    assert hits[0].note == "10_Notes/topic.md", "the title holder must outrank the quoters"


def test_wrapper_query_resolves_to_named_note(tmp_path: Path) -> None:
    """「关于X的内容」/「那篇讲X的笔记」 are real user phrasings; the wrapper is
    noise and X — the note's name — is the request. Stripping it recovers the
    structural identity, which the whole-query phrase test missed."""
    vault = tmp_path / "vault"
    (vault / "10_Notes").mkdir(parents=True)
    (vault / "10_Notes" / "water.md").write_text(
        "# 浇水计划\n\n冬季每月浇一次，夏季每周一次。\n", encoding="utf-8"
    )
    (vault / "10_Notes" / "misc.md").write_text(
        "# 园艺备忘\n\n备忘里写了一句关于浇水计划的内容需要确认。\n", encoding="utf-8"
    )
    db = tmp_path / "index.db"
    index_vault(vault, db)
    searcher = Searcher(db)
    for query in ("关于浇水计划的内容", "那篇讲浇水计划的笔记"):
        hits, _ = searcher.search(Query(text=query))
        assert hits, query
        assert hits[0].note == "10_Notes/water.md", query


def test_substantive_note_not_stubbed_by_one_small_section(tmp_path: Path) -> None:
    """The stub gate exists for persona files that are empty everywhere. A real
    note whose *matched section* is tiny (proper_noun-007: a 61-char 背景) is
    not a stub; the penalty must look at the note's total body, not the chunk."""
    vault = tmp_path / "vault" / "10_Notes"
    vault.mkdir(parents=True)
    long_body = "这是一段足够长的正文内容用于让这篇笔记成为真实笔记而不是占位文件。" * 6
    (vault / "resume.md").write_text(
        f"# 简历\n\n## 背景\n\n教育经历。\n\n## 经历\n\n{long_body}\n", encoding="utf-8"
    )
    db = tmp_path / "index.db"
    index_vault(vault.parent, db)
    hits, _ = Searcher(db).search(Query(text="背景"))
    assert hits
    assert hits[0].note == "10_Notes/resume.md"
    assert hits[0].score > 0.032, "a substantive note must not be halved by the stub gate"


def test_section_heading_query_wins_by_identity(tmp_path: Path) -> None:
    """A proper-noun query that equals a section heading must rank that note
    first: the heading is part of the note's identity surface. (Regression:
    the first pass of the tiered rescue dropped headings.)"""
    vault = tmp_path / "vault" / "10_Notes"
    vault.mkdir(parents=True)
    (vault / "diag.md").write_text(
        "# 系统体检\n\n## 总诊断\n\n全部指标正常，无异常项需要跟进处理。\n\n"
        "## 细节\n\n其他若干内容若干内容。\n", encoding="utf-8"
    )
    (vault / "other.md").write_text(
        "# 周报\n\n本周完成了若干工作，其中提到总诊断一次带过而已。\n", encoding="utf-8"
    )
    db = tmp_path / "index.db"
    index_vault(vault.parent, db)
    hits, _ = Searcher(db).search(Query(text="总诊断"))
    assert hits
    assert hits[0].note == "10_Notes/diag.md"


def test_keyword_query_finds_its_note(tmp_path: Path) -> None:
    _, db, _ = build(tmp_path)
    searcher = Searcher(db)
    hits, _ = searcher.search(Query(text="每月 检视 复盘"))
    assert hits
    assert hits[0].note == "40_Persona/Preferences/monthly-review.md"


def test_title_only_query_finds_the_note(tmp_path: Path) -> None:
    """The production shape this system exists for: an H1 title over H2
    sections. The title characters appear in no chunk body, so a query that is
    just the note's own name must still resolve through the lexical channel —
    it lives in the postings via Chunk.tokens."""
    vault = tmp_path / "vault"
    note = vault / "10_Notes" / "equipment.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# 设备清单\n\n## 厨房\n锅碗瓢盆和烤箱的使用记录。\n\n## 书房\n显示器支架的采购记录。\n",
        encoding="utf-8",
    )
    db = tmp_path / "index.db"
    index_vault(vault, db)
    hits, _ = Searcher(db).search(Query(text="设备清单"))
    assert hits, "a note must be findable by its own title"
    assert hits[0].note == "10_Notes/equipment.md"


def test_idf_stays_positive_and_multichunk_note_outranks_single_match(tmp_path: Path) -> None:
    """df must count distinct documents, not posting rows. With a 3-chunk note
    matching the term in every chunk, row-counted df exceeded N and the Robertson
    idf went negative — the richest match sank to the bottom."""
    vault = tmp_path / "vault" / "10_Notes"
    vault.mkdir(parents=True)
    (vault / "single.md").write_text(
        "# Single\n\nalpha one body text body text body text\n", encoding="utf-8"
    )
    (vault / "multi.md").write_text(
        "# Multi\n\nbeta and alpha section one body text body text\n\n"
        "## Two\nbeta and alpha section two body text body text\n\n"
        "## Three\nbeta and alpha section three body text body text\n",
        encoding="utf-8",
    )
    db = tmp_path / "index.db"
    index_vault(vault, db)
    hits, _ = Searcher(db).search(Query(text="alpha"))
    assert len(hits) == 2
    assert all(hit.lexical_score > 0 for hit in hits), "idf must not go negative"


def test_auxiliary_notes_are_not_default_results(tmp_path: Path) -> None:
    _, db, _ = build(tmp_path)
    searcher = Searcher(db)
    hits, _ = searcher.search(Query(text="multilingual e5 pilot Recall"))
    assert all(not hit.note.startswith("90_Agent/") for hit in hits)


def test_explicit_prefix_reaches_auxiliary(tmp_path: Path) -> None:
    _, db, _ = build(tmp_path)
    searcher = Searcher(db)
    hits, _ = searcher.search(
        Query(text="multilingual e5 pilot Recall", prefixes=("90_Agent",))
    )
    assert hits
    assert hits[0].note.startswith("90_Agent/")


def test_archive_is_reachable_only_when_asked(tmp_path: Path) -> None:
    _, db, _ = build(tmp_path)
    searcher = Searcher(db)
    default_hits, _ = searcher.search(Query(text="归档区 历史内容"))
    assert all(not hit.note.startswith("98_Archive") for hit in default_hits)
    asked_hits, _ = searcher.search(Query(text="归档区 历史内容", include_archive=True))
    assert all(not hit.note.startswith("98_Archive") for hit in asked_hits)
    # The archive is skipped at index time by policy; an explicit ask cannot
    # manufacture hits for content that was never indexed, and must not leak the
    # private zone either.
    assert all("95_Private" not in hit.note.lower() for hit in asked_hits)


def test_private_content_is_never_returned(tmp_path: Path) -> None:
    _, db, _ = build(tmp_path)
    searcher = Searcher(db)
    hits, _ = searcher.search(Query(text="PRIVATE CONTENT secret"))
    assert all("95_Private" not in hit.note.lower() for hit in hits)


def test_boilerplate_placeholder_does_not_win_an_unrelated_query(tmp_path: Path) -> None:
    """The production attractor: an empty persona stub out-ranked real notes on
    queries containing none of its words, via the semantic channel alone."""
    _, db, _ = build(tmp_path, embedder=FakeEmbedder())
    searcher = Searcher(db, embedder=FakeEmbedder())
    hits, _ = searcher.search(Query(text="completely unrelated quantum chromodynamics"))
    assert all("current-profile" not in hit.note for hit in hits)


def test_rrf_ranks_by_position_not_magnitude(tmp_path: Path) -> None:
    """A document ranked #1 by both channels must beat one ranked #1 by lexical
    only, regardless of the raw score gap the old formula would have produced."""
    _, db, _ = build(tmp_path)
    searcher = Searcher(db)
    hits, diagnostics = searcher.search(Query(text="每月 检视"))
    assert hits
    top = hits[0]
    assert top.lexical_rank == 0
    assert diagnostics.semantic_enabled is False  # lexical-only embedder
    assert diagnostics.gates_applied == ()


def test_empty_query_returns_nothing(tmp_path: Path) -> None:
    _, db, _ = build(tmp_path)
    hits, diagnostics = Searcher(db).search(Query(text="  "))
    assert hits == []
    assert diagnostics.scope_reason == "empty_query"
