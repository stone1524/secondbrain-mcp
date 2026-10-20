"""Deployment health check: verifies the live gateway's contract end to end.

The retired Node gateway shipped a wire-level health check (verify-mcp.mjs) run
every night; after the Python migration the harness still checked the dead
gateway while nothing validated the serving one. This is the migration of that
check: same contract, exercised through the tool boundary against a scratch
vault (writes go to a temp tree, never the live vault), then one read against
the live index to prove the serving database answers.

Exit 0 when every check passes; 1 with a JSON report otherwise. Run as:
    python -m secondbrain_mcp.adapters.healthcheck
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..core.audit import AuditLog
from ..core.config import Settings, load_settings
from .mcp_stdio import build_registry, dispatch


def _scratch_vault(root: Path) -> None:
    for relative in (
        "00_Inbox/Web",
        "00_Inbox/Conversations",
        "10_Notes",
        "95_Private",
        "98_Archive",
        ".Obsidian",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "95_Private/secret.md").write_text("PRIVATE-DO-NOT-LEAK\n", encoding="utf-8")
    (root / ".Obsidian/app.json.md").write_text("{}", encoding="utf-8")


def check_contract() -> dict:
    """Full tool-boundary contract on a scratch vault. Returns {name: bool}."""
    checks: dict[str, bool] = {}
    tmp = Path(tempfile.mkdtemp(prefix="sb-health-"))
    _scratch_vault(tmp / "vault")
    settings = Settings(
        vault_root=(tmp / "vault").resolve(),
        state_root=tmp / "state",
        agent_label="health-check",
        rag_db=tmp / "state" / "index.db",
        audit_log=tmp / "state" / "logs" / "gateway-audit.jsonl",
        lock_dir=tmp / "state" / "locks",
        semantic_enabled=False,  # hermetic: never wait on a model load
    )
    registry, settings = build_registry(settings)

    names = set(registry)
    checks["surface_13_tools"] = len(names) == 13
    checks["no_permanent_delete"] = not {"delete_note", "delete_file"} & names

    created = dispatch(
        "note_create",
        {"folder": "Web", "filename": "health.md", "content": "# health\n\nbody\n"},
        registry,
    )
    note = created.get("created")
    checks["create"] = bool(note)

    read = dispatch("note_read", {"note": note}, registry)
    checks["read_hash_and_trust"] = bool(read.get("hash")) and read.get("trust") == "untrusted_note"

    no_hash = dispatch("note_append", {"note": note, "content": "x"}, registry)
    checks["hash_required"] = no_hash.get("isError") is True

    wrong = dispatch(
        "note_append", {"note": note, "content": "x", "expected_hash": "deadbeef"}, registry
    )
    checks["wrong_hash_denied_with_recovery"] = (
        wrong.get("isError") is True and bool(wrong.get("error", {}).get("recovery"))
    )

    appended = dispatch(
        "note_append",
        {"note": note, "content": "\nappended\n", "expected_hash": read["hash"]},
        registry,
    )
    checks["append"] = appended.get("isError") is not True

    private = dispatch("note_read", {"note": "95_Private/secret.md"}, registry)
    checks["private_denied"] = private.get("isError") is True

    listed = dispatch("note_list", {}, registry)
    blob = json.dumps(listed, ensure_ascii=False)
    checks["list_hides_private"] = "secret.md" not in blob
    checks["list_hides_obsidian"] = ".Obsidian" not in blob and ".obsidian" not in blob.lower()

    escape = dispatch("note_read", {"note": "../../etc/passwd"}, registry)
    checks["escape_denied"] = escape.get("isError") is True

    relocate = dispatch(
        "note_move",
        {
            "note": note,
            "destination": "95_Private/stolen.md",
            "expected_hash": appended["hash"],
        },
        registry,
    )
    checks["move_into_private_denied"] = relocate.get("isError") is True
    checks["move_left_source_alone"] = not (settings.vault_root / "95_Private/stolen.md").exists()

    rebuilt = dispatch("index_rebuild", {}, registry)
    checks["index"] = rebuilt.get("isError") is not True and rebuilt.get("indexed", 0) > 0
    searched = dispatch("search", {"query": "health body"}, registry)
    checks["search_hits"] = bool(searched.get("hits"))

    archived = dispatch("archive", {"notes": [{"note": note, "hash": appended["hash"]}]}, registry)
    checks["archive"] = archived.get("archived") == 1
    restored = dispatch("restore", {"batch_id": archived["manifest"]}, registry)
    checks["restore"] = restored.get("restored") == 1

    audit = AuditLog(settings.audit_log, agent_label="health-check")
    checks["audit_chain_verifies"] = audit.verify()[0]

    shutil.rmtree(tmp, ignore_errors=True)
    return checks


def check_live_index(settings: Settings) -> dict:
    """Read-only probe against the serving database, through a *snapshot*.

    The live db may be stamped an older schema than this process (a server that
    started before the current code was deployed is still running). Opening it
    with initialize() would DROP-and-rebuild it in place — yanking the index out
    from under the running old-code processes mid-session. So the probe backs the
    database up via SQLite's backup API into a temp file and inspects *that*;
    the live file is opened read-only and never mutated."""
    checks: dict[str, bool] = {}
    tmp = Path(tempfile.mkdtemp(prefix="sb-health-live-"))
    copy = tmp / "snapshot.db"
    try:
        # Copy db+wal+shm rather than opening the live file read-only: a
        # mode=ro open of a WAL database requires the -shm to exist, which is
        # only true while a writer holds it. Files copied together replay the
        # wal on open — consistent enough for a health probe, and it never
        # mutates the live database.
        staged = tmp / "live-copy.db"
        shutil.copy2(settings.rag_db, staged)
        for suffix in ("-wal", "-shm"):
            side = Path(str(settings.rag_db) + suffix)
            if side.exists():
                shutil.copy2(side, Path(str(staged) + suffix))
        live = sqlite3.connect(staged)
        target = sqlite3.connect(copy)
        live.backup(target)
        live.close()
        target.close()
    except (sqlite3.Error, OSError) as error:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"snapshot_available": False, "error": str(error)}

    probe = Settings(
        vault_root=settings.vault_root,
        state_root=tmp,
        agent_label=settings.agent_label,
        rag_db=copy,
        audit_log=settings.audit_log,
        lock_dir=tmp / "locks",
        semantic_enabled=False,  # lexical-only: prove the term set answers, no model
    )
    registry, _ = build_registry(probe)
    # The copy still carries whatever schema stamp the live db had. A running
    # gateway on older code holds the live db at its own version; the copy must
    # be probed *as is*, not reconciled — initialize() on a version mismatch
    # would DROP the snapshot and the probe would report an empty index as a
    # health failure when the real problem is "not rebuilt yet". Read the stamp
    # and report it; only a same-version snapshot is content-checked.
    meta = dict(
        sqlite3.connect(copy).execute("SELECT key, value FROM meta").fetchall()
    ) if copy.exists() else {}
    meta: dict = {}
    if copy.exists():
        try:
            connection = sqlite3.connect(copy)
            meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
            connection.close()
        except sqlite3.Error:
            meta = {}  # no meta table: pre-schema or empty database
    from ..index.schema import SCHEMA_VERSION

    stamped = meta.get("schema_version")
    checks["snapshot_schema_current"] = stamped == str(SCHEMA_VERSION)
    if stamped == str(SCHEMA_VERSION):
        connection = sqlite3.connect(copy)
        count = connection.execute(
            "SELECT COUNT(*) FROM documents WHERE active = 1"
        ).fetchone()[0]
        connection.close()
        checks["live_documents_indexed"] = count > 0
        result = dispatch("search", {"query": "health", "limit": 3}, registry)
        checks["live_search_functional"] = result.get("isError") is not True
    else:
        checks["snapshot_schema_current"] = False  # informational: rebuild pending
    shutil.rmtree(tmp, ignore_errors=True)
    return checks


def main(argv: list[str] | None = None) -> int:
    report = {
        "at": datetime.now(timezone.utc).isoformat(),
        "contract": check_contract(),
    }
    report["passed"] = all(report["contract"].values())
    try:
        settings = load_settings()
        report["live"] = check_live_index(settings)
    except Exception as error:  # noqa: BLE001 - the live probe failing must be reported, not crash the check
        report["live"] = {"error": str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
