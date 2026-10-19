"""Validated configuration.

Previously every knob was read ad hoc from ``os.environ`` at module import, with
no schema and no validation: a non-numeric ``max_per_run`` made ``slice(0, NaN)``
archive nothing while exiting 0, and five different environment variables named
the vault root with two scripts ignoring all of them.

All settings are resolved once, validated, and passed explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """A configuration value is missing or malformed."""


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw).expanduser()


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from error
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from error


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True, slots=True)
class Settings:
    vault_root: Path
    state_root: Path
    agent_label: str | None
    rag_db: Path
    audit_log: Path
    lock_dir: Path
    archive_dirname: str = "98_Archive"
    inbox_dirname: str = "00_Inbox"
    inbox_folders: tuple[str, ...] = ("Web", "Conversations", "Files")
    stale_lock_seconds: float = 300.0
    semantic_enabled: bool = True
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    embed_batch: int = 0
    # Retrieval knobs; calibrated by the evaluation suite, not guessed.
    rrf_k: int = 60
    rrf_depth: int = 50
    semantic_floor: float = 0.55
    semantic_min_tokens: int = 20
    stub_body_chars: int = 120
    stub_penalty: float = 0.5
    sensitive_paths: tuple[str, ...] = field(default_factory=tuple)


def load_settings(
    *,
    vault_root: Path | str | None = None,
    state_root: Path | str | None = None,
) -> Settings:
    resolved_vault = Path(
        vault_root
        or _env_path("SB_VAULT_ROOT", None)
        or _env_path("OBSIDIAN_VAULT_ROOT", None)
        or (Path.cwd() / "demo-vault")
    ).resolve()
    if not resolved_vault.exists():
        raise ConfigError(f"Vault root does not exist: {resolved_vault}")

    resolved_state = Path(
        state_root
        or _env_path("SB_STATE_ROOT", None)
        or (resolved_vault / ".secondbrain")
    ).resolve()

    db_path = _env_path("SB_RAG_DB", None) or (resolved_state / "index.db")

    return Settings(
        vault_root=resolved_vault,
        state_root=resolved_state,
        agent_label=os.environ.get("SB_AGENT_LABEL") or os.environ.get("OBSIDIAN_AGENT_LABEL"),
        rag_db=db_path,
        audit_log=resolved_state / "logs" / "gateway-audit.jsonl",
        lock_dir=resolved_state / "locks",
        archive_dirname=os.environ.get("SB_ARCHIVE_DIRNAME", "98_Archive"),
        inbox_dirname=os.environ.get("SB_INBOX_DIRNAME", "00_Inbox"),
        stale_lock_seconds=_env_float("SB_STALE_LOCK_SECONDS", 300.0),
        semantic_enabled=_env_bool("SB_RAG_SEMANTIC", True),
        embed_model=os.environ.get("SB_RAG_MODEL", "BAAI/bge-small-zh-v1.5"),
        embed_batch=_env_int("SB_RAG_EMBED_BATCH", 0),
        rrf_k=_env_int("SB_RRF_K", 60, minimum=1),
        rrf_depth=_env_int("SB_RRF_DEPTH", 50, minimum=1),
        semantic_floor=_env_float("SB_SEMANTIC_FLOOR", 0.55),
        semantic_min_tokens=_env_int("SB_SEMANTIC_MIN_TOKENS", 20),
        stub_body_chars=_env_int("SB_STUB_BODY_CHARS", 120),
        stub_penalty=_env_float("SB_STUB_PENALTY", 0.5),
        sensitive_paths=tuple(
            part.strip()
            for part in os.environ.get("SB_SENSITIVE_PATHS", "").split(";")
            if part.strip()
        ),
    )
