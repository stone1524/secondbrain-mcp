# Migration from the Node gateway

The v1 gateway was a 366-line Node ESM script plus a 378-line indexer, with the
embedding done by a long-lived Python child exchanging newline-delimited JSON.
The v2 core is Python throughout. This document records what moved, what was
deliberately dropped, and how to go back.

## What the merge fixed structurally

The Node→Python embedding bridge had no request ids. Responses were correlated
purely by "one line per request", so a timed-out request's late reply could be
consumed by the next one; the only mitigation was killing the process. The child
also held the event loop open, requiring a ref/unref dance so one-shot CLIs could
exit, and `taskkill /T /F` on Windows because the venv trampoline's interpreter
survives a plain kill. In one language the provider is a function call.

## Behaviour changes

| Area | v1 | v2 |
|---|---|---|
| Tool names | `rag_search` + `search_notes` | one `search` with `mode` |
| Editing | append only | `note_patch` with mandatory preview |
| `expected_hash` | optional | required on every mutation |
| Private-zone casing | case-sensitive prefix compare | case-folded zone lookup |
| Manifest restore | paths trusted | every path re-validated |
| Error messages | collapsed to "Error executing tool X" | `code` + `message` + `recovery` |
| Vector gaps | permanent after one failure | `embed_state=FAILED`, retried next run |
| Model switch | silently mixed vector spaces | fingerprint mismatch forces rebuild |
| Archive indexing | excluded | excluded (unchanged) |
| Agent-ops notes | hidden by default | still hidden by default; pass `paths` |

`expected_hash` becoming mandatory is the one breaking change for clients. A call
that omits it now fails with `HASH_REQUIRED` and an instruction to call
`note_read` first.

## Running both side by side

Point the new server at the same vault but a different state root:

```json
{
  "command": "path/to/python.exe",
  "args": ["-m", "secondbrain_mcp.adapters.mcp_stdio"],
  "env": {
    "SB_VAULT_ROOT": "path/to/vault",
    "SB_STATE_ROOT": "path/to/state-v2"
  }
}
```

The old gateway keeps its own lock directory and index, so the two do not
interfere. During the shadow phase, writes go only through the old server and the
new one is queried read-only.

## Shadow gate

```bash
python -m secondbrain_mcp.eval.shadow \
    --queries eval/shadow-queries.json \
    --report eval/shadow-report.json
```

The gate is **correctness parity or better**, not set overlap. Requiring 80%
overlap with a system that scored 60% would ratify its mistakes; low overlap is
expected where the old ranking was the defect. Overlap is reported so a large
divergence is still visible for inspection.

## Rollback

The MCP registration is a single JSON entry. Restoring the previous command and
args returns the old gateway; nothing in the vault changed shape during the
migration, so the old index and its lock directory remain valid.

## What stayed where

The nightly collectors (`Bridge/collectors/`) are unchanged and still run on
Node. They communicate with the gateway only through files:

- `cleanup-candidates-latest.json` (consumed by `review_queue`)
- archive manifests under `98_Archive/.manifests/`
- generated notes the indexer skips by policy

Those files are a contract, not shared code. A future change should add a
`schema_version` field and have the gateway refuse unknown versions; the current
manifests already carry one.
