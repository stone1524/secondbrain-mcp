# Threat model

## What this defends against

**An agent making a mistake, or note content steering an agent.** The realistic
failure is a model that misreads a path, retries a write it already applied, or
follows an instruction embedded in a note it was asked to summarize. The system
is designed so those produce a refusal or an obvious error, never silent data
loss.

**Not defended:** a malicious local process with filesystem access. Such a
process can edit the vault, the audit log, and the manifests directly. This is a
single-user local tool; pretending otherwise would add cryptography without
adding safety.

## Threats and controls

| Threat | Control |
|---|---|
| Agent reads the private zone | Zone decided in one place, from the resolved path, case-folded. Probed under four casings in both unit and wire tests. |
| Agent writes outside the vault | `NotePath.parse` resolves then requires containment; absolute paths, drive-relative forms, and `..` are rejected. |
| Agent overwrites a concurrent edit | `expected_hash` is mandatory on every mutation and re-verified inside the lock. |
| Agent deletes a note | There is no delete operation. Removal is archive plus a rollback manifest; recycle-bin work is manual. |
| A crafted manifest moves files | Restore re-parses every path through the policy and requires the archived side to resolve inside the archive directory. |
| A crash truncates a note | Writes are `temp → fsync → os.replace`. |
| A crash loses the rollback record | The manifest is written in `pending` state *before* any move, then finalised. |
| Note content is treated as instructions | Every read and search result carries `trust: untrusted_note`. |
| A rejected write looks like an accepted one | Audit records intent and outcome separately, each with `ok` or an error code. |
| Audit history is edited | Records form a SHA-256 chain (`prev` → `hash`); `verify()` reports the first broken link. |
| A stale lock is stolen from a live writer | Five-minute staleness is time-based only; a slow-but-alive writer can lose its lock. Mitigated by ownership tokens so the loser cannot destroy the winner's lock, not eliminated. |

## Trust boundaries

```
untrusted:  note bodies, archive manifests, cleanup candidate files
trusted:    the vault root and state root from configuration
```

Manifests and candidate files are written by other processes into directories
inside the vault, so they are treated as input, not as trusted state. This is why
restore validates paths rather than trusting `from`/`to`.

## Known limitations

- **Staleness is time-based.** A five-minute lock window is a guess. A writer
  blocked longer than that can have its lock reaped.
- **Reads are unlocked.** `note_read` returns a hash that was true at read time.
  The write path re-verifies, so a lost update is refused — but a read can return
  content that is already stale.
- **The audit chain is tamper-evident, not tamper-proof.** A local process can
  rewrite the whole chain. It detects accidental corruption and naive editing.
- **`embed_state = FAILED` is retried on every index run.** A permanently broken
  model will cause a re-embedding attempt each time rather than an escalating
  backoff.
- **Dependency advisories are triaged, not absent.** The shipped surface depends
  on `mcp` (which pulls `pyjwt[crypto]` for its auth paths), plus optional
  `numpy`/`fastembed` for the semantic channel. `pip-audit` runs against both
  the dev and the serving venvs (last clean run: 2026-10-20, after upgrading
  pyjwt/pip/setuptools/urllib3). Upgrades on the `mcp` line are constrained by
  the `>=1.2,<3` pin, so triage reads which package an advisory sits in, not
  just the version number.
- **The nightly harness never writes the serving index.** It builds and scores
  against a staging state root, and its live probe inspects a *copy* of the
  serving database (SQLite backup API). A schema-mismatch DROP can therefore
  only ever hit a staging or already-reconciled state — never the index the
  running gateway processes depend on (`adapters/healthcheck.py`).
