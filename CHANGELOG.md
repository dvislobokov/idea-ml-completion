# Changelog

Each experiment is one commit. Metrics are measured on the fixed 13-repository corpus per language
(`../ml-data/<lang>/sets/base13.txt`), split by file 90/10, unless stated otherwise. See README for the recipe.

## 0.1.0 — baseline (2026-10-05)

- ml-core: C#/Go lexers, vocabulary, modified Kneser–Ney n-gram LM (order 4, 64-bit hash keys, float values, gzip),
  listwise logistic-regression ranker with proxy example generator, `.cml` format v1.
- ml-train: `l2`, `l1`, `eval-lm`, `eval-rank`, `tokens`.
- tools/corpus: GitHub selection pipeline; reviewed lists of 737 C# and 900 Go repositories; sparse cloning with limits.

Baseline on 13 repos:

| metric | Go | C# |
|---|---|---|
| train tokens / vocab | 8.0 M / 50 k | 7.9 M / 50 k |
| LM size (no pruning) | 45 MB | 46 MB |
| perplexity | 8.3 | 9.9 |
| identifier OOV | 8.8 % | 17.5 % |
| next identifier top-1 / top-5 (LM) | 0.33 / 0.52 | 0.30 / 0.49 |
| ranker MRR / top-1 | 0.72 / 0.61 | 0.74 / 0.64 |
| LM-only baseline MRR | 0.63 | 0.59 |
| l2 time (train + eval) | 4 s + 214 s | 12 s + 263 s |

## e01 — fixed-context scorer for the n-gram LM

`NgramModel.Scorer`: for one completion position the hashes of the context suffixes and the cumulative backoff weights are
computed once; each candidate then costs ≤ `order` hash extensions + lookups instead of re-hashing the context for every
order. Quality is bit-identical (same probabilities). Also: `--repos <file>` fixes the repository set for experiments,
symlinks inside cloned repos are never followed, `tools/bench/run.sh` runs the standard measurement.

| metric | Go | C# |
|---|---|---|
| LM eval time (top-k over 50 k vocab, ~12 k / 10 k positions) | 214 s → **97 s** | 263 s → **86 s** |
| perplexity / top-1 / top-5 | 8.3 / 0.335 / 0.522 (same) | 9.9 / 0.304 / 0.487 (same) |
| ranker MRR | 0.715 (same) | 0.741 (same) |

(Go file count changed 6738 → 6727: symlinked files are now skipped.)

## e02 — compact model storage: 24-bit fingerprints + 8-bit quantised values (format v2)

`CompactFloatMap`: open addressing keyed by the 64-bit n-gram hash, but a slot stores only a 24-bit fingerprint (the slot
index fixes the other bits) and an 8-bit value index into 256 quantile bins. On disk: slot delta (varint) + 3 + 1 bytes per
entry. The exact `LongFloatMap` stays for training; `NgramModel.write` converts. Fingerprint collisions ≈ 1e-7 per lookup.

| metric | Go | C# |
|---|---|---|
| LM file, no pruning | 45.1 MB → **18.6 MB** | 46.5 MB → **19.1 MB** |
| RAM per entry | 12 B × 1.43 → 5 B × 1.43 | same |
| perplexity exact / compact | 8.3 / 8.3 | 9.9 / 9.9 |
| top-1 / top-5 exact | 0.335 / 0.522 | 0.304 / 0.487 |
| top-1 / top-5 compact | 0.333 / 0.522 | 0.303 / 0.487 |
| ranker MRR (LM feature from compact model) | 0.715 (same) | 0.741 (same) |

Quantisation to 256 bins is lossless for ranking purposes. Combined with `--min-count 1,1,2,2` the model would be ~7–8 MB.
(Eval time in this run is not comparable: both languages ran concurrently.)

## e03 — correct pruning (SRILM-style) and per-repository thresholds

Bug found while adding `--min-repos`: pruning was applied to the count tables *before* estimation, so (a) continuation counts
of lower orders were computed from already-pruned higher orders and (b) kept probabilities were re-estimated from the
reduced totals. Now the model is estimated from full counts; pruned n-grams are then removed and the backoff weight of their
context is recomputed as γ = (1 − Σ kept p) / (1 − Σ kept p_lower), so kept probabilities stay exactly as estimated and the
distribution still sums to one (test `prunedModelStillSumsToOneAndKeepsProbabilities`). `NgramTable` can track the number of
distinct repositories per n-gram (`--min-repos a,b,c,d`, plan §5.5 memorisation guard).

| variant | Go size | Go ppl | Go top-1/top-5 | Go ranker MRR | C# size | C# ppl | C# top-1/top-5 | C# ranker MRR |
|---|---|---|---|---|---|---|---|---|
| no pruning | 18.6 MB | 8.3 | 0.333 / 0.522 | 0.715 | 19.1 MB | 9.9 | 0.303 / 0.487 | 0.741 |
| `--min-count 1,1,2,2` (old, wrong pruning) | 18 MB* | 9.5 | 0.320 / 0.497 | — | 20 MB* | 11.9 | 0.284 / 0.456 | — |
| `--min-count 1,1,2,2` (fixed) | **7.7 MB** | **8.8** | **0.327 / 0.506** | **0.706** | **8.1 MB** | **10.8** | **0.292 / 0.466** | **0.731** |
| `--min-repos 1,1,1,2` (4-grams in ≥ 2 repos) | 8.7 MB | 10.7 | 0.262 / 0.438 | 0.687 | 8.9 MB | 13.3 | 0.222 / 0.404 | 0.712 |
| `--min-repos 1,1,2,2` | 3.7 MB | 15.3 | 0.222 / 0.349 | 0.630 | 3.8 MB | 21.0 | 0.167 / 0.276 | 0.652 |

\* old numbers were measured with format v1 (12 bytes/entry); the fixed pruning yields the same entry count at v2 size.

Singleton pruning of orders 3–4 is now nearly free: −59 % size for +0.5–0.9 perplexity and −0.6/−1.1 p.p. top-1.
The repository threshold is too aggressive on 13 repositories (most 4-grams live in one repo); it is meant for the full corpus
(≥ 3 of 300 repos) and must be re-measured there — keep it off until then.
