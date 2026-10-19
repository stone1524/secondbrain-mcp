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
```

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
