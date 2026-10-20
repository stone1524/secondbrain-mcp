# Evaluation

## Why this document exists

The production system reported **Recall@5 = 1.00**. The same vault, probed with
twenty realistic queries, returned the right note in the top 5 about **60%** of
the time. Both numbers were measured honestly; the difference is what the metric
was measuring.

Tracing the gap found five independent defects in the evaluation, not in the
retriever:

1. **The nightly run used six hand-written cases**, not the 25-case file that sat
   in the repository unused.
2. **Two of those six narrowed `paths` to `90_Agent/logs`** — a directory holding
   exactly one document. Recall@5 on a one-document corpus is 1.0 by
   construction.
3. **The evaluated default scope held about 22 documents.** With five result
   slots, roughly 73% of the pool consisted of acceptable answers.
4. **Two of the three gates could not fail.** The traceability predicate was
   tautological by construction, and the privacy probe targeted a zone that is
   excluded at index time, so it could never return a hit.
5. **The query glossary had been extended using the two failing cases**, with no
   holdout. The two misses (`per-06`, `per-09`) each had a dedicated entry added
   afterwards. The score measured fit to a set that had been tuned on itself.

A frozen eval set does not merely stop measuring; it keeps returning a number
that looks like a guarantee.

## What this harness does differently

| Risk | Mitigation |
|---|---|
| Cases selected to pass | Six categories, generated mechanically from note titles and headings, then scored without filtering |
| No holdout | `cases-dev.json` for tuning, `cases-test.json` kept unread during parameter work |
| Metric too lenient | Recall@5 **and** rank-1, reported per category |
| Gates that cannot fail | Every gate has a reachable failure state, verified by deliberately breaking it |
| Unreproducible numbers | Every report records a corpus snapshot hash |
| Testing only what is easy | Cases include `90_Agent` and `00_Inbox`, the areas the old evaluator never touched |

## Categories

| Category | What it tests | Example |
|---|---|---|
| `keyword` | literal term present in the note | a note's own title |
| `paraphrase` | same intent, different words | 「关于 X 的内容」 |
| `proper_noun` | product / model / project name | a section heading |
| `technical` | CLI, API, or code identifier | a backticked token |
| `cross_lingual` | same intent in the other language | English query, Chinese note |
| `vague_topic` | under-specified reference | 「那篇讲 X 的笔记」 |

## Gates

```
recall_at_5          >= 0.85
semantic_rank1_wins  >= 1        # proves the vector channel is not dead weight
null_embedding_chunks == 0       # every document that should be embedded, is
privacy_leakage      == 0        # probed under four casings of the protected dir
index_side_leak      == 0        # rows in the db whose path is inside 95_Private
```

The semantic gate runs only under `--semantic`. In lexical-only mode it is
reported as `null` (not applicable) rather than `true`, because a gate that
returns "pass" for a capability that was never exercised is how the previous
system's metrics became decorative. The `privacy_leakage` probe depends on the
search layer refusing the private zone, which makes it structurally unable to
fail on a read-side regression that still indexed the rows — `index_side_leak`
reads the database directly to cover that half.

Gates are executed by CI (`.github/workflows/ci.yml`) after the demo index is
built, so a change that breaks recall or leaks a private path into the index
cannot merge green.

The second gate is the one that would have caught the production regression. A
hybrid retriever whose semantic channel never changes a rank-1 is not a hybrid;
it is a keyword search with an expensive appendage. The production system would
have scored **0** on this gate across every probe.

## Current results

Baseline, 36 dev cases, lexical only:

```
recall_at_5   0.750
recall_at_1   0.444
semantic      0 wins
```

After the retrieval rewrite, same cases, semantic enabled:

```
recall_at_5   0.861
recall_at_1   0.611
semantic      7 rank-1 wins
null vectors  0
privacy       0
all gates     pass
```

After the schema-v2 posting fixes (title-in-index, df-per-document,
script-split tokens, hyphen segments, file stem in the surface), same cases on
the same real vault, semantic enabled. The prior code reproduces 0.861
exactly — the comparison is honest:

```
recall_at_5   1.000   (0.861 -> 1.000, zero regressed cases)
recall_at_1   0.694   (0.611 -> 0.694)
null vectors  0
privacy       0
```

After the structural-phrase identity fix (chunks.title, tiered identity vs
body mention, wrapper-stripped query cores, note-level stub gate, PHRASE_BONUS
applied at fusion, evidence tie-break on fused-score ties):

```
recall_at_5   1.000
recall_at_1   0.972   (0.694 -> 0.972)
null vectors  0
privacy       0
```

The single remaining rank-1 loss is `DRAFT` — a generic identifier present in
several PCB-standard notes, where the aggregate index note holds the strongest
lexical claim; no identity signal exists to disambiguate and none is
fabricated for it. Note: the hybrid rank@1 (0.972) now meets the lexical-only
rank@1 exactly; before the evidence tie-break, the semantic channel was
*demoting* correct rank-1 answers (0.944 < 0.972) by tying at 1/61 against
path-sorted noise. A tie at one-channel-RRF is structural, so ties break
toward literal matched evidence.

**A note the honest harness must record:** semantic rank-1 wins was 7 -> 5 ->
0 across these steps. Every case in this generator is *derived from a note's
title or heading*, so once the lexical channel legitimately recognizes that
identity, the vector channel can no longer flip a rank-1 on any of them. The
gate `semantic_rank1_wins >= 1` is not satisfied by this case set anymore —
not because the vector channel is dead weight (it still carries recall on
zero-overlap queries, which is exactly what the production regression gate
probes for), but because **a title-derived generator structurally cannot
measure it.** The fix is a separate no-shared-term probe set (see Roadmap);
loosening the gate without it would repeat the production sin of a number
that looks like a guarantee.

## Shadow comparison

Retrieval was also compared against the system it replaced, query by query, on
the same vault (`eval/shadow.py`). Agreement with a system that scores 60% is not
a goal, so the gate is **correctness**, not overlap:

```
queries        18
old correct    15 / 18
new correct    17 / 18
set overlap    0.44   (reported as a warning)
```

Low overlap with higher correctness is the expected outcome when the old ranking
was itself the defect. Three queries the old system missed, the new one finds;
one query it found, the new one does not — a long title retyped with full-width
punctuation, addressed by normalizing the phrase-match key.

## Running it

```bash
# Build cases from a vault
python -m secondbrain_mcp.eval.build_cases --vault "$SB_VAULT_ROOT" --out eval/cases-dev.json

# Score
python -m secondbrain_mcp.eval.run --cases eval/cases-dev.json --semantic \
    --report eval/report.json

# Compare against a Node implementation
python -m secondbrain_mcp.eval.shadow --queries eval/shadow-queries.json \
    --report eval/shadow-report.json
```

Thresholds are configuration (`SB_SEMANTIC_FLOOR`, `SB_RRF_K`, …), so a change to
ranking must be justified by a re-run, not by an argument.
