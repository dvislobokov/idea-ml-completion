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

## e04 — per-file cache LM mixed into the global model

`CacheLm` (order 3, interpolated absolute discounting over the tokens seen so far in the file) and `MixedLm`
(P = λ·P_cache + (1−λ)·P_global). Used both for LM evaluation and as the `lm_logprob` ranker feature (`--cache λ`).
This is the "localness of software" component (Tu, Su & Devanbu 2014); the IDE will feed the cache from the editor buffer.

| metric | Go, no cache | Go, λ=0.3 | Go, λ=0.5 | C#, no cache | C#, λ=0.3 |
|---|---|---|---|---|---|
| perplexity | 8.3 | **5.0** | 4.9 | 9.9 | **5.8** |
| next identifier top-1 / top-5 | 0.333 / 0.522 | **0.471 / 0.707** | 0.490 / 0.717 | 0.303 / 0.487 | **0.443 / 0.670** |
| ranker MRR / top-1 | 0.715 / 0.606 | **0.773 / 0.673** | 0.768 / 0.667 | 0.741 / 0.641 | **0.779 / 0.687** |
| LM-only baseline MRR | 0.631 | 0.719 | 0.714 | 0.585 | 0.677 |

Largest gain of all experiments so far, at zero model size and ~1 µs per token of bookkeeping. λ=0.5 is better for the raw LM,
λ=0.3 for the ranker; keep 0.3 as default and tune λ per project later (plan §6, EM on held-out files). Decay of old cache
entries and a project-level cache (Hellendoorn's nested cache) are the obvious next steps (see docs/RESEARCH.md §1, §7).

## e05 — corpus deduplication and repository-level split (honest evaluation)

`Dedup`: exact duplicates by token-sequence hash, near-duplicates by MinHash/LSH over the identifier set (Jaccard ≥ 0.8,
Allamanis 2019); offered train-first, so test files that duplicate training files are dropped. `--test-repos` holds out
explicit repositories; `--split repo` uses the hash split. All runs with cache λ=0.3.

| setting | Go files train/test | Go ppl | Go top-1/top-5 | Go OOV | Go ranker MRR | C# ppl | C# top-1/top-5 | C# OOV | C# ranker MRR |
|---|---|---|---|---|---|---|---|---|---|
| base13, file split, no dedup (e04) | 6031 / 696 | 5.0 | 0.471 / 0.707 | 8.8 % | 0.773 | 5.8 | 0.443 / 0.670 | 17.5 % | 0.779 |
| base13, file split, dedup 0.8 | 5720 / 664 | 5.0 | 0.464 / 0.703 | 8.9 % | 0.770 | 6.0 | 0.436 / 0.661 | 18.4 % | 0.772 |
| all repos (33 Go / 15 C#), **repo split**, dedup | 16553 / 1719 | 5.5 | 0.440 / 0.682 | **29.4 %** | **0.712** | 6.9 | 0.371 / 0.639 | **26.5 %** | **0.720** |

Held-out repos: Go — caddy, etcd, esbuild, v2ray-core (hash split); C# — Flow.Launcher, Newtonsoft.Json (explicit).

Findings: (1) near-duplicates are rare in these curated repos (5 % of files) and barely move the metrics; (2) the split
matters a lot — unseen repositories triple the identifier OOV rate and cost ~0.06 MRR; the file split numbers were
optimistic, as docs/RESEARCH.md predicted. From here on the standard measurement is **repo split + dedup**; file-split
numbers of e01–e04 remain valid for relative comparisons only. The LM-only baseline drops much more (0.72 → 0.55) than the
ranker (0.77 → 0.71): in-file features carry over to new repositories, the global LM does not — the argument for the cache
and for project-level counts.
