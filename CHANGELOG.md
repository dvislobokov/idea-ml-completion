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
