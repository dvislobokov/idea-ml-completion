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

## e06 — smoothing: Jelinek–Mercer vs modified Kneser–Ney, order 4 vs 5

docs/RESEARCH.md reports JM-6 beating KN-3 on Java (Hellendoorn & Devanbu 2017). Checked on our data with the standard
measurement (repo split, dedup, cache λ=0.3). `--smoothing jm --lambda 0.5`, `--order N`; `tools/bench/std.sh` wraps the
standard setting.

| variant | Go size | Go ppl | Go top-1/top-5 | Go ranker MRR | C# size | C# ppl | C# top-1/top-5 | C# ranker MRR |
|---|---|---|---|---|---|---|---|---|
| MKN order 4 (e05) | 30 MB | **5.5** | 0.440 / 0.682 | 0.712 | 29 MB | **6.9** | 0.371 / 0.639 | 0.720 |
| JM order 4, λ=0.5 | 29 MB | 6.1 | **0.450** / 0.685 | 0.710 | 28 MB | 7.4 | **0.399** / 0.641 | 0.718 |
| JM order 5 | 59 MB | 5.5 | 0.446 / 0.688 | 0.713 | 54 MB | 6.9 | 0.380 / 0.644 | 0.718 |
| MKN order 5 | 61 MB | **5.1** | 0.448 / **0.689** | **0.716** | 56 MB | **6.5** | 0.373 / **0.651** | 0.717 |

Conclusion: with a properly estimated MKN the literature result does not transfer — MKN is 0.5–0.6 perplexity better than
JM at equal order; JM wins only on identifier top-1 (+1–3 p.p., it spreads less mass to unseen words), which the ranker does
not convert into MRR. Order 5 buys ~0.4 perplexity for 2× size; worth it only after pruning on the full corpus. Keep MKN-4
as default; JM stays available (`--smoothing jm`) for the project-level model where counts are small.

## e07 — list-relative ranker features and cross-fitted training (LM and ranker on disjoint repositories)

New base features: `lm_rank_log`, `lm_delta_best` (rank/gap by LM score within the list), `lm_global_logprob` (global LM
without cache, so the model can weigh cache vs global), `freq_rank_log`, `is_most_recent`; `--features N` keeps the first N
base features for ablations. 8 → 13 base features, 56 → 91 weights.

First attempt (LM trained on the same repositories the ranker trains on) made things **worse**: Go MRR 0.712 → 0.683,
C# 0.720 → 0.680, while the training loss converged normally. Cause: the LM has memorised the ranker's training files, so
`lm_global_logprob` is inflated there, the ranker over-trusts it and fails on unseen repositories — a leakage that the
single mixed LM feature had been masking partially. Fix: **cross-fitting** — LM trained on `sets/lm.txt` (2/3 of the
training repos), ranker on `sets/rank.txt` (the other 1/3), both evaluated on the held-out repos (`sets/test.txt`).

| ranker (standard split, cache λ=0.3) | Go MRR / top-1 / top-5 | C# MRR / top-1 / top-5 |
|---|---|---|
| 8 features, LM on all train repos (e05, leaky) | 0.712 / 0.601 / 0.848 | 0.720 / 0.608 / 0.858 |
| 13 features, LM on all train repos (leaky) | 0.683 / 0.567 / 0.824 | 0.680 / 0.558 / 0.830 |
| 8 features, cross-fitted | 0.715 / 0.601 / 0.857 | 0.716 / 0.598 / 0.864 |
| **13 features, cross-fitted** | **0.718 / 0.604 / 0.861** | **0.723 / 0.606 / 0.871** |
| LM-only baseline | 0.551 | 0.530 |

The ranker trained on 1/3 of the repositories with a smaller LM matches or beats the leaky full-data one — data volume is
not the bottleneck for L1 (as docs/RESEARCH.md says: features are). Cross-fitting is now mandatory for the server recipe;
the real PSI features (expected type, kind, scope) are the next lever, and they need the plugin adapters.

## e08 — parallel LM evaluation, `--max-test-files`, server runbook

Evaluation of the LM now runs per file in parallel (`parallelStream`; the cache LM is per file, so files are independent) —
C# test set 190 s → 21 s on 8 cores, perplexity identical. The top-k sample is now taken per file (every 20th in-vocabulary
identifier of each file) instead of globally, so the sampled positions differ from e01–e07: C# e05 LM reads top-1 0.357 /
top-5 0.643 under the new sampling (was 0.371 / 0.639); compare top-1/top-5 only within one sampling scheme from now on.
`--max-test-files N` stride-samples the test files, keeping every held-out repository represented, to bound evaluation time
on the full corpus.

`tools/server/` — runbook for a 24-core / 64 GB Linux box: `setup.sh` (JDK 21, build, environment report), `fetch.sh`
(both corpora, resumable), `stats.sh`, `sets.sh` (deterministic test / lm / rank repository folds), `train.sh <lang> <tag>
[l2 args]` (standard measurement, logs and `summary.txt` per experiment), `report.sh`. See `tools/server/README.md`.

## e09 — adapter contract: language feature block, shared feature extractor, example shards

Preparation for the PSI-based generators in the plugins (docs/ADAPTER.md):

- `MlLanguage.rankFeatures` — names of the PSI-derived per-candidate features an adapter supplies; they form the language
  block of the schema (`FeatureSchema.common(languageFeatures = ...)`), conjoined with the context kind like the common block.
- `FeatureExtractor` + `FileState` — the common features (LM with cache, frequency, recency, prefix, list-relative) now live in
  one class used by `ProxyExampleGenerator` and, later, by the IDE weigher — the training/serving parity point.
- `ExampleShards` (`*.cmlx`, gzip, self-describing with schema and source) — exchange format between the plugin generators
  and `ml-train`. `l1 --shards <dir> [--test-shards <dir>]` trains on shards, `l1 ... --dump-shards <dir>` writes proxy
  examples as shards, `eval-rank --shards <dir>` evaluates a ranker on shards.

Regression: proxy ranker on the C# standard folds reproduces e07 exactly (MRR 0.723, top-1 0.606); training from the dumped
shards gives the same numbers. 17 unit tests green.

## e10 — first ranker on real Go completion lists (PSI candidates from the plugin)

The Go plugin now embeds this repository as a subtree and exports examples with its real completion
(`idea-golang-support`: `go-psi-ide/.../ml/GoMlFeatures.kt`, `GoMlDatasetExport`, `./gradlew :go-psi-ide:mlDataset`; parity test
`GoMlFeatureParityTest`). Language block: 17 features (candidate kind groups, scope level, needs import, expected-type match,
declared in file + distance, the plugin's deterministic order as `rule_rank_log`). Export cost 30–160 ms per position
(bigger repositories are slower: more package-level declarations to resolve). Candidate recall of the plugin, i.e. the share of
positions whose real identifier is in the list at all (positions where the plugin shows no list — declaration names etc. — excluded):
88–92 % per repository.

Local run: ranker fold = 7 repositories, 200 files each, 20 positions per file → 11 943 lists; held-out = caddy, etcd, esbuild,
v2ray-core → 7 811 lists, 49 candidates on average. LM from the lm fold (e07), cache λ=0.3, cross-fitted.

| order of the list | top-1 | top-5 | MRR |
|---|---|---|---|
| plugin rules (expected type → scope level → name) | 0.394 | 0.699 | 0.534 |
| n-gram LM only | 0.342 | 0.532 | 0.439 |
| most frequent in file | 0.353 | 0.727 | 0.516 |
| **ML ranker (linear, 210 weights)** | **0.675** | **0.919** | **0.783** |

Per context kind the ranker is 0.735 (after `.`) … 0.812 (statement start) MRR; the rules are weakest after `.` (0.378: same-level members
are ordered by name) and in type positions (0.407). Heaviest weights: exact-case prefix match, declared in this file, scope level,
needs import, keyword kind (negative). Caveat: `rule_rank_log` reproduces the plugin's weigher but not the platform's prefix/proximity
weighers that follow it, so the "rules" row is slightly pessimistic. Known gaps: one repository exported 0 lists (headscale: no
`go.mod` at the root, to check); near-duplicate repositories across folds (v2fly/v2ray-core vs v2ray/v2ray-core — excluded from
training) call for dedup at the shard level too.

## e11 — full Go corpus on the server (580 repositories), streaming loader, pruning series

Server: AMD EPYC 7443P 24 cores / 64 GB. The provider breaks the git smart protocol towards GitHub ("expected flush after ref
listing"), so `tools/corpus/fetch.sh` falls back to codeload tarballs (`.commit` records the revision) and
`tools/server/update.sh` updates the checkout by commit tarball. Corpus: 900 Go repositories (1.07 M files, 11.1 GB), 737 C#
(0.8 M files, 6.2 GB); folds `sets.sh`: 30 test, 580 lm, 290 rank for Go.

Two engine fixes were needed for corpus-sized counts: (1) `ml-train` no longer keeps token lists of the whole corpus — every
pass re-tokenises in parallel batches of 256 files (`forEachTokenised`), local regression exact (Go std: ppl 5.5, MRR 0.683);
(2) `NgramTable` capped at 2^28 slots (the `ids` array overflowed `Int` at 349 M tokens: `NegativeArraySizeException`).
Counting 349 M tokens takes 5 min, 23 GB peak; LM evaluation 80 s on 3664 files (`--max-test-files 4000`).

| LM (MKN-4, cache λ=0.3, 30 held-out repos) | n-grams | size | ppl | OOV | top-1 / top-5 | proxy ranker MRR / top-1 / top-5 |
|---|---|---|---|---|---|---|
| local, 19 repos (e07) | 2.5 M | 30 MB | 5.5 | 29.4 % | 0.440 / 0.682 | 0.718 / 0.604 / 0.861 (4 held-out repos) |
| server, 100 repos | 5.3 M | 53 MB | 5.1 | 24.1 % | 0.453 / 0.713 | — |
| server, 580 repos, no pruning | 27.1 M | 163 MB | 5.0 | 21.5 % | 0.468 / 0.720 | 0.739 / 0.629 / 0.876 |
| `--min-count 1,1,2,2` | 12.8 M | 78 MB | 5.0 | 21.5 % | 0.470 / 0.722 | 0.739 / 0.629 / 0.876 |
| `--min-count 1,2,3,3` | 8.1 M | 50 MB | 5.1 | 21.5 % | 0.470 / 0.722 | 0.738 / 0.629 / 0.875 |
| `--min-count 1,1,2,2 --min-repos 1,1,1,3` | 5.9 M | 35 MB | 5.0 | 21.5 % | 0.480 / 0.727 | 0.738 / 0.628 / 0.874 |
| **`--min-count 1,1,3,3 --min-repos 1,1,2,5`** | 3.4 M | **20 MB** | 5.1 | 21.5 % | **0.481 / 0.727** | 0.737 / 0.627 / 0.873 |

Ranker on the full corpus: 731 k training lists (`PER_FILE=10`), LM-only baseline 0.575. Repository-count pruning, which hurt
on 19 repositories (e03), now improves identifier top-1 by 1 p.p. while cutting the model 8×: project-specific n-grams only
add noise on unseen code. Recommended plugin model: the 20 MB variant. Report in Russian: `docs/REPORT-GO-RU.md`.
