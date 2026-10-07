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

## e12 — full C# corpus on the server (472 repositories), Go order 5 with repository pruning

Same recipe as e11 (`tools/server/train.sh`, `PER_FILE=10`, 30 held-out repositories per language, cross-fitted folds).

| C# LM (MKN-4, cache λ=0.3) | n-grams | size | ppl | OOV | top-1 / top-5 | proxy ranker MRR / top-1 / top-5 | LM-only MRR |
|---|---|---|---|---|---|---|---|
| local, 10 repos (e07) | — | 29 MB | 6.9 | 26.5 % | 0.371 / 0.639 | 0.723 / 0.606 / 0.871 (2 held-out repos) | 0.530 |
| server, 472 repos, no pruning | 27.4 M | 162 MB | 5.9 | 32.9 % | 0.380 / 0.620 | 0.718 / 0.599 / 0.868 | 0.501 |
| `--min-count 1,1,2,2 --min-repos 1,1,1,3` | 5.9 M | 35 MB | 6.2 | 32.9 % | 0.387 / 0.618 | 0.713 / 0.593 / 0.864 | 0.497 |
| `--min-count 1,1,3,3 --min-repos 1,1,2,5` | 3.5 M | 20 MB | 6.4 | 32.9 % | 0.388 / 0.615 | 0.712 / 0.593 / 0.862 | 0.496 |

C# corpus: 310 M tokens in 383 k files after dedup (470 k before: 7 % duplicates, more generated code than in Go). The 30 held-out
C# repositories are harder than the two local ones (OOV 33 % vs 27 %), so the absolute numbers are not comparable with e07; the
server LM-only baseline is 0.50 against 0.53 locally for the same reason. Unlike Go, repository pruning costs C# 0.3–0.5 perplexity
and 0.006 MRR; still, 20 MB at −0.006 MRR is the plugin candidate. 111 k test lists.

| Go LM | size | ppl | top-1 / top-5 | proxy ranker MRR / top-1 / top-5 |
|---|---|---|---|---|
| order 4, `--min-count 1,1,3,3 --min-repos 1,1,2,5` (e11) | 20 MB | 5.1 | 0.481 / 0.727 | 0.737 / 0.627 / 0.873 |
| **order 5, `--min-count 1,1,3,3,3 --min-repos 1,1,2,5,5`** | 27 MB | **4.7** | **0.497 / 0.738** | **0.742 / 0.636 / 0.874** |

Order 5 with the same pruning is the best Go model so far: −0.4 perplexity, +1.6 p.p. top-1, +0.005 MRR for 7 MB more. On the
19-repository local corpus order 5 was not worth its size (e06); on 580 repositories it is. Recommended Go plugin LM: order 5, 27 MB.

## e13 — inline continuation ("grey text") with the n-gram LM: how much of a line it gets right

`ml-train eval-inline`: at every 50th token position of the held-out files the mixed model (per-file cache λ=0.3 + global) greedily
generates up to 8 tokens; a predicted `<ID>` stops it, an actual out-of-vocabulary identifier never matches. Confidence = geometric mean
probability of the first three generated tokens — what a feature would gate the suggestion on. Go, server, order-5 LM (e12, 27 MB),
30 held-out repositories, 1858 files, 45 133 positions (5461 at line starts).

| first k tokens right | k=1 | k=2 | k=3 | k=4 | k=5 | k=8 |
|---|---|---|---|---|---|---|
| all positions | 0.631 | 0.403 | 0.287 | 0.205 | 0.157 | 0.083 |
| line starts | 0.398 | 0.294 | 0.158 | 0.120 | 0.082 | 0.037 |

Rest of the line (≤ 8 tokens) generated exactly: 30.6 % of 28 199 eligible positions; 1.99 correct tokens per position on average.

| show when confidence ≥ | shown | of those, ≥ 3 tokens right | token precision |
|---|---|---|---|
| 0 (always) | 100 % | 28.7 % | 0.440 |
| 0.5 | 37.0 % | 56.2 % | 0.692 |
| 0.7 | 14.3 % | 80.2 % | 0.868 |
| 0.8 | 9.0 % | 89.5 % | 0.929 |
| 0.9 | 2.5 % | 94.9 % | 0.967 |

Reading: mid-line, the LM continues idioms (`err != nil { return err }`, `"error", err)`, closing brackets) — a third of the lines it
could finish exactly. At line starts it is weak (what statement comes next is not an n-gram question). A gated feature at 0.7–0.8 shows a
suggestion at 9–14 % of positions and is right in 80–90 % of them (token precision 0.87–0.93), which is the regime of a usable inline
completion; a neural model is needed for the other 86 %. Cost: greedy argmax over the 50 k vocabulary per token, ~1 ms per step — fine
for an inline provider that runs after a pause in typing, and reducible with a candidate shortlist from the n-gram table.

## e14 — n-gram LM on the full Go corpus (22 610 repositories, 3.9 G tokens): token shards and partitioned counting

Pipeline: `prepare` (filters, global dedup 0.8, folds lm/rank/test = 22 610 / 11 305 / 300 repos; the generated-code filter now
requires the `Code generated … DO NOT EDIT` / `<auto-generated` marker to sit in a comment line, which un-drops generator sources such as
`cmd/cgo/godefs.go`: 2162 of 7.0 M files changed status, 1296 became `ok`; kept 4 697 459 files, 32.7 GB, 5.79 G tokens), then
`shard`, then `l2 --token-shards`.

- **Token shards** (`ml-train shard`, 14.4 min on 28 threads, 46 GB RSS): the manifest's `ok` files are lexed once (pass A counts token
  texts, pass B writes ids). 26.3 M vocabulary ids (identifiers seen once in the whole corpus, 9.6 M tokens = 0.17 %, read as `<ID>`),
  4 bytes per value because the vocabulary exceeds 2 M ids: 23.2 GB for 5.79 G tokens (4.01 B/token), 304 part files. A value is
  `id << 2 | atLineEnd << 1 | lineStart`, so `eval-inline` needs no source text. Lex rate ≈ 2 × 6.7 M tokens/s.
- **`PartitionedNgramTrainer`** (ml-core): same MKN model with SRILM-style pruning as `NgramTrainer` (unit test: identical probabilities and
  entry counts on random corpora, pruned by count and by repositories). Vocabulary ids are split into 64 balanced bins by first token; one
  scan of the shards per bin collects the raw tables (first token in bin) and the distinct-n-gram sets (second token in bin, which give the
  continuation counts), computes per-context totals/N1/N2/N3+ from the full counts, prunes, spills only survivors, frees the tables. Discounts
  and probabilities come afterwards from the survivors. 24 threads at once, peak heap ≈ 85 GB of 110 GB (RSS 104–110 GB, mostly uncollected
  garbage); 223 M distinct 5-grams, 105 M 4-grams, 34 M 3-grams, 2.6 M bigrams. Counting takes ~8–10 min, whole run 11–12 min.
- Evaluation on the 300-repository test fold, evenly thinned: eval-lm 2847 files / 3.45 M tokens (39 818 sampled positions), cache λ=0.3;
  eval-inline 1941 files, 46 271 positions.

| LM (order 5, MKN) | n-grams kept | size | ppl | OOV | top-1 / top-5 |
|---|---|---|---|---|---|
| e12: 580 repos, `--min-count 1,1,3,3,3 --min-repos 1,1,2,5,5` (30 held-out repos) | — | 27 MB | 4.7 | 21.5 % | 0.497 / — |
| e14-a: 22 610 repos, same thresholds | 33.1 M | 207 MB | 3.8 | 20.9 % | 0.525 / 0.760 |
| **e14-b**: `--min-count 1,2,5,8,8 --min-repos 1,1,10,40,60` | 5.5 M | **32 MB** | 4.1 | 20.9 % | **0.521 / 0.753** |
| e14-d: `--min-count 1,2,5,10,10 --min-repos 1,1,20,60,100` | 4.4 M | 23 MB | 4.3 | 20.9 % | 0.520 / 0.750 |

(Test sets differ from e12's 30 repositories, so the comparison is indicative; the OOV rate is the vocabulary's 50 k identifier cap, unchanged.)
Repository thresholds have to grow with the corpus: the e12 thresholds keep 7× more entries and give 207 MB.

Inline continuation, e14-b (e13 in brackets): confidence ≥ 0.8 shown at 10.0 % of positions (9.0 %), ≥ 3 tokens right in 92.9 % (89.5 %),
token precision 0.956 (0.929); ≥ 0.7: 16.5 % shown, 83.3 % right (14.3 / 80.2); rest of line generated exactly 35.1 % (30.6 %);
first token right 0.652 (0.631).

Proxy ranker (lists from the rank fold, 560 k training lists, `--per-file 10 --max-files 100000`; 15 448 test lists): MRR 0.759 / top-1 0.652 /
top-5 0.891 with the e14-b LM (e12: 0.742); LM-only baseline MRR 0.602. Real plugin completion lists (e10) are not on this server, so the
real-list ranker was not retrained.

## e15 — n-gram LM and proxy ranker on the full C# corpus (24 945 repositories, 2.95 G tokens)

Same pipeline as e14 for C#: `prepare` → `shard` → `l2 --token-shards` → `eval-inline` → `l1 --token-shards`. The C# `prepare` run used the
stricter generated-code filters (case-insensitive `*.designer.cs`, `*ModelSnapshot.cs`, `*AssemblyAttributes.cs`, Verify snapshots, EF Core
`Migrations/<timestamp>_*.cs`, unambiguous header phrases such as "Code generated by" / "lost if the code is regenerated", Unity `PackageCache`
excluded) and decodes windows-1252 headers instead of dropping the file. Manifest v2 vs the first run (v1):

| prepare (38 150 repos on disk, 26 min) | v1 | v2 |
|---|---|---|
| kept files / bytes / tokens | 6 303 915 / 36.1 GB / 4.38 G | **6 131 384 / 35.0 GB / 4.28 G** |
| dropped: generated marker / generated name | 611 457 (6.3 GB) / 505 292 (5.5 GB) | 895 491 (8.6 GB) / 833 603 (7.6 GB) |
| excluded dirs (incl. `PackageCache`) | 118 145 files (0.6 GB) | 405 369 files (2.7 GB) |
| non-UTF-8 dropped | 42 161 (510 MB) | **268** (4 MB; windows-1252 fallback) |
| duplicates removed (exact / near) | 1.91 M / 1.65 M | 1.57 M / 1.31 M (generated SDKs were the biggest duplicate source) |
| folds lm / rank / test (repos with data) | 24 940 / 12 334 / 290 | 24 945 / 12 333 / 290 |
| top kept repos | dotnet 535 MB, **azure-powershell 413 MB, aws-sdk-net 364 MB**, mono, roslyn, **AutoSDK 207 MB** | dotnet 483 MB, mono 248 MB, roslyn 213 MB (the three generated SDKs are gone) |

- **Shards**: 8.0 min on 20 threads (CPUs 0–23 shared with other jobs), 46.6 GB RSS; 4.28 G tokens → 17.2 GB in 319 part files (4.01 B/token,
  4-byte values: 29.66 M vocabulary ids, 10.1 M rare identifiers = 0.24 % of tokens read as `<ID>`). 18 759 kind conflicts merged to IDENT.
- **LM** (`PartitionedNgramTrainer`, order 5, MKN, `--partitions 64 --threads 20`, `-Xmx100g`): 9.0–9.4 min each, peak RSS 98–102 GB.
  Fold lm: 4 216 152 files, 2.95 G tokens; distinct n-grams 3.1 M bigrams / 32.0 M 3-grams / 96.8 M 4-grams / 202 M 5-grams (Go: 2.6 / 34 / 105 / 224 M).
  Evaluation on the 300-repository test fold, evenly thinned: eval-lm 2860 files (33 815 sampled positions), cache λ=0.3; eval-inline 1980 files, 27 408 positions.

| LM (order 5, MKN) | n-grams kept | size | ppl | OOV | top-1 / top-5 | inline ≥ 0.8: shown / right | ≥ 0.7 | rest of line | first token |
|---|---|---|---|---|---|---|---|---|---|
| e12 C#: 472 repos, `--min-count 1,1,3,3 --min-repos 1,1,2,5` (order 4, other test set) | 3.5 M | 20 MB | 6.4 | 32.9 % | 0.388 / 0.615 | — | — | — | — |
| e14-b Go (reference) | 5.5 M | 32 MB | 4.1 | 20.9 % | 0.521 / 0.753 | 10.0 % / 92.9 % | 16.5 % / 83.3 % | 35.1 % | 0.652 |
| **e15-a**: `--min-count 1,2,5,8,8 --min-repos 1,1,10,40,60` (= e14-b) | 5.6 M | **32 MB** | 5.9 | 22.3 % | **0.484 / 0.729** | 3.1 % / 87.9 % | 6.6 % / 76.6 % | 31.5 % | 0.570 |
| e15-b: `--min-count 1,2,5,10,10 --min-repos 1,1,20,60,100` (= e14-d) | 4.3 M | 24 MB | 6.1 | 22.3 % | 0.483 / 0.726 | 3.1 % / 87.2 % | 6.4 % / 76.0 % | 31.3 % | 0.566 |
| e15-c: `--min-count 1,2,4,6,6 --min-repos 1,1,5,20,30` | 8.9 M | 52 MB | 5.6 | 22.3 % | 0.490 / 0.736 | 3.3 % / 88.0 % | 7.0 % / 76.6 % | 32.0 % | 0.574 |

(e12 numbers come from a different, smaller test set and order 4, so they are indicative only; OOV is again the 50 k identifier cap.)
Pruning barely matters between 24 and 52 MB (top-1 0.483–0.490, ranker MRR 0.754–0.758); **e15-a** (32 MB, same thresholds as the Go
model) is the recommended C# model: `../ml-data/csharp/models/e15-a.cml`.

C# is harder than Go for the n-gram model at the same corpus size: perplexity 5.9 vs 4.1, first-token accuracy 0.570 vs 0.652, and the
confident inline continuation covers 3× fewer positions (3.1 % vs 10.0 % at ≥ 0.8) with lower precision (87.9 % vs 92.9 %). Reasons seen in
the dumps: C# lines are longer and more nominal (generic types, member chains, lambdas), and the C# lexer sets no `atLineEnd` flag.

Qualitative (e15-a, 40 + 40 random confident positions at ≥ 0.8): among the *right* ones about a third are literal tables (`<NUM> , <NUM> , …`,
`<STR> , <STR>`, `{ <STR> , <STR> } ,`, `<STR> + <STR> +`, `Environment.NewLine + <STR> +`) that the plugin cannot render anyway (placeholders),
the rest are real idioms: property/const boilerplate (`{ get; set; } public …`, `public const string X = "…";`), xUnit `[Fact] public void X() {`,
`for (var y = 0; y < …`, `stream.Seek(0, SeekOrigin.Begin);`, `IValueConverter.Convert(object value, Type targetType, object parameter`,
`Should.Throw<T>(() `, `StringComparison.Ordinal))`, `using System.Collections.Generic;`, `GUILayout.Button("…")) {`, `MessageBoxIcon.Error);`.
Among the *wrong* ones more than half are literal lists continued past their real end (`Color.FromArgb(r, g, b` → `, <NUM>` instead of `)`,
`new DateTime(y, m, d`, `InlineData(…)`, `Utc(…)`): a 5-gram does not know the arity. The rest: `using` blocks (`using System.Collections.Generic;`
where a project namespace or `namespace` follows), loop headers with a different bound or direction (`i >= firstIndex; i--`), `switch` expression
default arm (`_ =>`), attribute lists (`, DefaultValue(…)]`), `OnPropertyChanged(nameof(…))` vs `(string.Empty)`. A useful gate for the plugin:
do not show a continuation that consists only of literal placeholders.

Proxy ranker (rank fold, `--per-file 10 --max-files 100000`: 430 380 training lists, 12 933 test lists, 32.7 candidates): e15-a LM MRR **0.756** /
top-1 0.657 / top-5 0.880 (e15-b 0.754, e15-c 0.758); LM-only baseline MRR 0.613, most-recent-in-file 0.397. Old C# state (e12, 472 repos,
other test set): 0.712–0.718 / LM-only 0.496. 1.2–1.9 min, 10.9 GB RSS. → `../ml-data/csharp/models/e15-a-rank.cml`.

Generated code still slipping through v2 (manifest scan + header scan of a 24.7 k-file sample, ≈ 1–1.5 % of kept files): `*.gen.cs` (9268 files,
e.g. Silk.NET `Sdl.gen.cs`), `*.Generated.cs` with a capital G (1177 files — the name rule is case-sensitive), ApprovalTests `*.approved.cs` (139),
`/Generated/` directories (38 760 files, 0.24 GB: google-api-dotnet-client "Generated code. DO NOT EDIT!" is not matched because the rule looks for
"Code generated"), decompiler output ("Decompiled with JetBrains decompiler", ILSpy, dnSpy, IL2CPP dumps / `DummyScripts`: ~38 k files,
0.14 GB), swagger-codegen / openapi-generator clients (`/* … Generated by: https://openapi-generator.tech */`), opensearch-net `_Generated`
("do not edit" in lower case), and hand-written numeric tables (`ConversionLookupTable.cs`, VSOP87 planetary series, `DefinitionHashes.*.cs`),
which explain the `<NUM> , <NUM>` dominance in the inline dumps. Not fixed in this experiment.

## e16 — own transformer for whole-line completion, Go (go31m: 31 M parameters, BPE 16k, FIM), Kotlin + native inference

First neural model of the project. Everything below runs on the user's machine as Kotlin (`ml-core/nn`) with optional native kernels;
training is PyTorch on the server (`tools/nn/`). Design and the full measurement tables: `docs/NEURAL-RU.md`.

**Tokenizer.** Byte-level BPE, our own trainer and encoder (`tools/nn/tokenizer`, `ml-core/bpe`), code-aware pre-tokenizer
(`go-code-1`: newline + indentation is one pre-token, letters/digits/punctuation never merge across class boundaries, strings and
comments kept). Go vocabulary 16 384 (3.16 bytes/token on the test fold; 32k would save only 3.8 % tokens at twice the embedding),
C# vocabulary trained separately (4.05 B/token; the Go vocabulary on C# costs +7 % tokens). Kotlin ↔ Python parity: 10 571 cases
(571 synthetic + 10 000 test files), 21.7 M ids identical, exact byte round trip; Kotlin encoder ~84 MB/s.

**Corpus.** `go/prepared` manifest (e14) encoded to uint16 shards: lm 6.84 G tokens, rank 3.20 G, test 91 M. Repository down-weighting
(file kept with probability `min(1, sqrt(2M / repo tokens))`: SDK wrappers fall from 0.9 % to 0.2 % of the stream), documents
`<|file_sep|> path\n body <|endoftext|>`, windows of 2048. A secret/PII audit of the lm folds (`tools/nn/clean`) found real leaks
(cloud keys, JWTs, DSN passwords, private keys — mostly in test fixtures); the scrubbing policy drops 0.39 % of Go bytes / 0.16 % of C#
and replaces values in 0.013 % / 0.017 %. go31m was trained before the filter existed (internal comparison only); C# and all later runs
are encoded through it.

**Model.** Llama-style decoder: d512 × 8 layers, 8 heads / 2 KV heads, SwiGLU 1408, RMSNorm, RoPE, tied embeddings — 30.9 M parameters.
AdamW(0.9, 0.95), wd 0.1, lr 1e-3 cosine to 1e-4, warmup 500, bf16, `torch.compile`, 1 048 576 tokens/step, one epoch = 6.8 G tokens,
2.5 h on one RTX PRO 6000 (771k tok/s, 17 GB). Test ppl 2.13 (plain). Export to CMLN (int8 per-output-channel scales, f32 norms, 31 MB)
is lossless: ppl 2.131 → 2.131.

**FIM bug found and fixed mid-run.** At step 2500 the PSM layout was useless (2 % lines right, SPM 62 %): documents were packed and cut into
fixed windows, and with 74.5 % of tokens in files longer than 2048 tokens the PSM middle usually landed in a window without its prefix.
Fix (`data.py`): FIM documents bounded (middle ≤ 256, suffix ≤ 512, prefix tail so that the document ≤ 2040), never split across a window,
half of the cuts line-aligned; tests assert prefix and middle share a window and `pre+mid+suf` rebuilds the file. Resumed at step 4000
(62 % of the epoch on the old data). PSM recovered to 56 % lines right at ≥0.8 but SPM (70 %) remains the inference layout; the next run
starts clean with `fim_rate 0.7` (the bounded documents lowered the FIM token share to 27 %).

**Whole-line evaluation** (`tools/nn/eval/eval_inline.py`, 3 000 positions in 2 522 test-fold files, stride 50 as `eval-inline`; metrics at
the character/lexical level so the n-gram numbers are comparable; confidence = product of token probabilities):

| | n-gram e14-b | go31m plain | go31m SPM |
|---|---|---|---|
| shown at ≈95 % whole-line precision | 10.0 % (≥3 tokens 92.9 %) | **32.1 %** (prod ≥0.8: ≥3 tokens 96.7 %, line 95.4 %) | ≈ same |
| shown at ≈90 % | 16.5 % (83.3 %) | 45.7 % (93.1 %) | — |
| rest of line exact, ≤ 8 tokens left | 35.1 % | 63.2 % | **65.8 %** |
| rest of line exact, all positions | — | 53.2 % | 55.7 % |
| first lexical token right | 0.652 | 0.783 | **0.798** |
| inside string literals (67 positions) | ≈ 0 | shown 36 %, 75 % right | |

Calibration is monotone; the geometric mean of the first three tokens over-trusts whole lines, the product does not. Error classes:
right structure with the wrong field name, repetition on literal tables (needs a repetition guard), early newline (2–4 %). Hardest
positions: right after `(` / `{` / operators.

**Inference (`ml-core/nn`).** CMLN v1 container (uncompressed, memory-mapped, weights stay off-heap), KV cache with longest-common-prefix
reuse, greedy/top-k with per-token log-probs. Scalar Kotlin kernels shaped for C2 auto-vectorisation (axpy loops on offset-0 arrays, kept
above the inline size; ~18 GMAC/s per core). JetBrains Runtime ships no `jdk.incubator.vector` (verified: the flag breaks IDE start-up),
so the Vector API path is a benchmark reference only. Parity on the trained model: scalar and native-f32 reproduce PyTorch bit-for-bit up
to summation order (argmax 32/32, greedy-48 32/32, 1000/1000 identical lines on 500 positions × 2 modes).

**Native kernels (`native/`, `ml-core/nn/native`).** ~1 100 lines of C11, runtime dispatch (AVX2, AVX2+VNNI, AVX-512, AVX-512-VNNI, NEON,
NEON+DotProd), f32 and q8 (int8 activations, VNNI `vpdpbusd` / NEON `sdot`) modes, self-test at load, JNI with no weight copies, scalar
fallback on any failure; four binaries (linux-x64, windows-x64, macos-x64, macos-arm64; 85–103 KB) cross-compiled with zig. Kernels reach
the FMA roofline (50 GMAC/s/core f32) and 180–210 GMAC/s/core with VNNI. On the trained model, 1 500-token prompt, 8 threads:

| | scalar Kotlin | native f32 | native q8 |
|---|---|---|---|
| prefill | 492 ms | 211 ms | **147 ms** |
| 20-token line | 530 ms | 240 ms | **167 ms** |
| line while typing (KV reuse) | 47 ms | 33 ms | **24 ms** |
| decode per token | 1.98 ms | 1.50 ms | 1.06 ms |

q8 does not change accuracy on the real model (line exact 54.7 % vs 54.6 %, argmax 32/32) and is the default native mode; f32/scalar
are bit-exact and used for ranker log-prob features. Untested: NEON (compiled, self-test-guarded), Windows/macOS loading and signing,
AVX-VNNI. Laptop estimate (8-core AVX2, scaled): 30 M ≈ 65 ms per cold 512-token line with native kernels (≈150 ms scalar), 48 M ≈ 105 ms,
100 M ≈ 200 ms — so native kernels make 48 M shippable; 100 M needs PSI context compression or the next native refactor.

**C# (cs31m-e1).** Same architecture and recipe on the C# corpus (manifest v3, `cs-16384.bpe`, secret-scrubbed shards: lm 5.65 G tokens),
`fim_rate 0.7` and the FIM fix from step 1; 5 388 steps / 123 min; ppl 3.97 (n-gram e15-a: 5.9); export lossless (3.966 → 3.967).
Whole-line evaluation (3 000 positions, C# lexical splitter, CRLF-aware; `eval_inline.py --lang csharp`):

| | n-gram e15-a | cs31m plain | cs31m SPM |
|---|---|---|---|
| shown at ≈96 % whole-line precision | — | prod ≥0.9: 11.4 % | 9.8 % |
| shown at ≈92 % | 3.1 % (≥3 tokens 87.9 %) | **17.6 %** (line 91.9 %) | 16.5 % (93.5 %) |
| rest of line exact, ≤ 8 tokens left | 31.5 % | 46.8 % | **49.0 %** |
| first lexical token right | 0.570 | 0.700 | **0.718** |

C# is harder than Go for both model families (longer, more nominal lines: mean rest 8.1 vs 6.5 lexical tokens). The transformer removes the
n-gram's arity failure (opens with `)` after the last argument in 65 % of such positions, continues with `,` in 2 %); most confident shows are
closers (`);`, `)]`, `}`) — the plugin should not show one-token closers. Interpolated strings and `_logger.Log…` messages are rarely byte-exact.
PSM is still far behind SPM (line exact 20.7 % vs 39.8 %) even with correct data: teacher-forced NLL of the first middle token is 2.12 (PSM)
vs 0.81 (SPM) — a capacity/format effect of a 31 M model, so inference uses SPM. Model: `csharp/models/cs-nn-31m-e1.cml` (31.2 MB).

**Token healing and show policy (2026-10-07, `NnCompletion`).** The eval protocol (and a live caret) often sits inside a BPE pre-token
(`foo(⟨⟩)`, `"x"⟨⟩)`, after a typed space, mid-identifier) — 10–11 % of positions, where the model got 1.5–4.8 % of lines right because
`()`, `");` etc. are single pre-tokens never split in training. Healing (cut the prompt back to the last pre-token boundary, constrain the
first generated tokens to start with the typed remainder, strip it from the output) lifts those positions to 67 % (Go) / 75 % (C#):
rest of line exact all positions Go 55.7 → **61.7 %**, C# 39.8 → **47.9 %**; ≤ 8 tokens Go 65.8 → 73.3 %, C# 49.0 → 60.2 %; first token
Go 0.80 → 0.88, C# 0.72 → 0.81. `--heal none` reproduces the old numbers exactly. Show policy measured: gate on the product of token
probabilities; a repetition guard removes 2.5–2.8 % of lines, all wrong (C# precision at ≥0.8: 94.5 → 96.6 %); punctuation-only closers
(`);`, `}`) are 22–25 % of positions and 98 % right at ≥0.8 — showing them is a UX choice. Kotlin: `BpeTokenizer.lastPreTokenBoundary`,
`VocabPrefixIndex`, `NnCompletion.complete(path, before, after)` → text, conf_prod/conf_min, show decision; parity with Python on 99/99
healed records (scalar, native f32) and on the C# model (`models/parity-cs`, argmax 32/32, 40/40 lines). API: `docs/NN-COMPLETION-API.md`.

## e17 — Go ranker on real plugin completion lists, 300 + 295 repositories (headless PSI export on the server)

The e10 export (`idea-golang-support`: `go-psi-ide/.../ml/GoMlDatasetExport`, `./gradlew :go-psi-ide:mlDataset`) now runs on the
server: IntelliJ IDEA Community 2026.1.4 unpacked to `/root/work/idea` (`-PlocalIdePath`), the IDE's JBR as `JAVA_HOME`, Go 1.27.1
as GOROOT, fontconfig + DejaVu installed (the headless test editor still asks for a font), delve submodule initialised. The export is
single-threaded, so 10 workers run as plain JVMs with the command line of Gradle's test worker and separate sandbox config/system
dirs (`~/work/nn/psi/run-export.sh`, `launch-dataset.sh`; setup notes in `~/work/nn/psi/README.md`). Two robustness fixes in the export
class: a unique temp directory per repository (the reused path `gopsi-ml-/...` after a bulk VFS deletion raised `Incorrect CachedValue use`
in `GoImportPaths`, which the test logger turns into an exception that ended the run after 3–17 repositories) and a per-repository
try/catch; the worker JVMs also run with `-Dintellij.testFramework.rethrow.logged.errors=false`.

Data: 300 `rank`-fold repositories sampled from the 7 165 with 20–3000 kept files (seed 17) and all 295 `test`-fold repositories of the
e14 manifest; ≤ 120 files per repository (alphabetical, non-test, 200 B–400 KB, generated files skipped), 10 positions per file, prefix 0–2
characters, ≤ 100 candidates (answer kept), LM e14-b (lm fold, cross-fitted), cache λ = 0.3, candidate names stored. Export: 27 ms per
position; 595 repositories in ~26 min on 10 JVMs (6 GB heap each).

| fold | repos | files | positions | plugin shows a list | answer in list (recall) | lists | candidates / list |
|---|---|---|---|---|---|---|---|
| rank | 300 | 20 970 | 204 497 | 66.1 % (28.6 % no list: declaration names etc., 5.3 % single candidate inserted) | 88.6 % | 119 723 | 48.1 |
| test | 295 | 14 794 | 144 507 | 65.3 % (28.9 % / 5.7 %) | 88.9 % | 83 904 | 46.3 |

External modules are not in a module cache here, so members of third-party types are unresolved (the list then has no such candidate):
part of the 11 % recall gap. Three repositories gave 0 lists (`cockroachdb__cockroach-gen`, `OpenCSGs__csghub-server`,
`made-in-bangladesh__made-in-bangladesh`): their first 120 files alphabetically are generated mocks / code.

Ranker: `ml-train l1 --lang go --shards psi/rank --test-shards psi/test` (schema of e10: 13 common + 17 language features × 7 = 210 weights),
55 s, → `../ml-data/go/models/e17-rank.cml`. Evaluation on the 83 904 test-fold lists:

| order of the list | top-1 | top-5 | MRR |
|---|---|---|---|
| plugin rules (expected type → scope level → name, `rule_rank_log`) | 0.388 | 0.685 | 0.527 |
| n-gram LM e14-b only | 0.396 | 0.608 | 0.499 |
| most frequent in file | 0.357 | 0.710 | 0.515 |
| proxy-trained ranker e14-b-rank (13 common features, projected onto the real lists) | 0.390 | 0.666 | 0.518 |
| **e17 ranker (real lists, 210 weights)** | **0.710** | **0.934** | **0.808** |

Per context kind (e17 / rules MRR): after `.` 0.776 / 0.342, statement start 0.832 / 0.642, argument 0.837 / 0.653, type position
0.825 / 0.420, assignment rhs 0.780 / 0.490, other 0.786 / 0.500. Heaviest standardised weights: exact-case prefix match +1.51, scope level
−0.95, declared in file +0.86, in vocabulary +0.73, needs import −0.73, keyword kind −0.69, `after_dot:lm_global_logprob` +0.67.
Compared with e10 (7 training repositories, 4 held-out): MRR 0.783 → 0.808, top-1 0.675 → 0.710 on a test set 10× larger; the rules
baseline is unchanged (0.534 → 0.527). The proxy ranker transfers badly to real lists (0.518, below the rules): proxy lists are
vocabulary-sampled and lack the PSI candidate structure (kinds, scope, expected type), so a plugin ranker must be trained on real lists.

Side result for the neural model (`docs/NEURAL-RU.md` §7 item 2): a PSI-context dump at 2 436 sampled positions of the test fold
(`GoMlContextExport`, `./gradlew :go-psi-ide:mlContext`, `../ml-data/go/psi/context-test.jsonl`). Resolution cost per position
(enclosing function, qualifier type + members, expected type, scope, imports, cross-file signatures): median 13.5 ms, p90 77 ms, mean 35 ms
(first position of a file pays stub/AST loading; the 4 s maximum is a package with 300+ declarations). Of the identifiers in the true
rest of line, 80.7 % already occur in the file prefix, 4.8 % come only from the PSI context (first identifier: 7.5 %; after `.`: 12.7 %),
14.5 % from neither (58 % of those are members of a qualifier typed later in the line, 12 % names declared on that line, 9 % calls of
functions outside the context, 18 % signature parameters / external modules). Expected type is known at 21 % of positions, the
qualifier's type at 41 % of `.`-positions (+23 % packages; the rest are external modules without a module cache).

## e18 — C# recipe ablations (SPM share, lr 2e-3 / 0.5 M batch), decode-time studies, teacher ceiling with Hugging Face models, DDP

Server reinstalled by the provider on 2026-10-07 (2 × H200): both corpora re-prepared (`tools/server/rebuild-data.sh`; Go 34 229 repos → 4.69 M kept
files / 5.81 G lexer tokens, lm fold 22 432 repos / 3.93 G; C# 38 150 → 6.06 M / 4.22 G, lm 24 940 / 2.92 G), the Go BPE vocabulary retrained
(`go-16384.bpe`, sha256 770a316b…; the published C# vocabulary reused), secret-scrubbed BPE shards: Go lm 6.85 G tokens, C# lm 5.65 G.

**CPU models rebuilt** (same recipes; the test folds changed with the stricter generated-code filters, so the numbers are a re-measurement, not a regression):

| model | ppl | top-1 / top-5 | inline ≥ 0.8: shown / right | rest of line ≤ 8 | proxy ranker MRR (LM-only) |
|---|---|---|---|---|---|
| Go e14-b (`go/models/e14-b.cml`, 32 MB; e14: 4.1 / 0.521 / 10.0 % / 92.9 % / 35.1 %) | 4.6 | 0.510 / 0.747 | 6.6 % / 90.9 % | 31.7 % | 0.685 (0.496), 221 722 lists |
| C# e15-a (`csharp/models/e15-a.cml`, 32 MB; e15: 5.9 / 0.484 / 3.1 % / 87.9 % / 31.5 %) | 5.3 | 0.487 / 0.740 | 3.9 % / 89.4 % | 32.7 % | 0.708 (0.527), 9 013 lists; 0.702 on all 235 684 |

**Two C# runs** (go31m preset, scrubbed corpus, fim 0.7, 1 epoch = 5.65 G tokens, one H200 each, ~77–81 min; baseline cs31m-e1 from e16: ppl 3.97,
rest of line exact 47.9 % healed): `cs31m-e2-spm10` — spm-rate 1.0, lr 1e-3, 1 M tokens/step (5 388 steps); `cs31m-e2-lr2e3` — spm 0.5, **lr 2e-3,
0.5 M tokens/step** (10 776 steps). Standard whole-line evaluation (`eval_inline.py --lang csharp --positions 3000 --seed 1`, SPM, token healing,
repetition guard, 2 518 files) plus the free-of-training studies on the same positions:

| run / study | eval ppl plain / FIM | rest of line exact, all | ≤ 8 tokens | first token | prod ≥ 0.8: shown / line exact | prod ≥ 0.9 |
|---|---|---|---|---|---|---|
| cs31m-e2-spm10 | 3.98 / 4.39 | 47.7 % | 60.3 % | 0.819 | 17.6 % / 94.3 % | 9.7 % / 97.3 % |
| … prefix ≤ 1024 tokens (default 1450) | | 48.1 % | 60.5 % | 0.819 | 19.9 % / 94.6 % | 12.1 % / 97.5 % |
| … prefix ≤ 512 | | 47.7 % | 60.6 % | 0.813 | 20.6 % / 95.0 % | 13.0 % / 97.7 % |
| … beam 4 | | 49.3 % | 62.2 % | 0.804 | 17.4 % / 96.7 % | 9.6 % / 98.6 % |
| **cs31m-e2-lr2e3** | **3.80 / 3.98** | **50.2 %** | **62.6 %** | 0.821 | 20.0 % / 97.0 % | 11.2 % / 98.8 % |
| … prefix ≤ 1024 | | 50.2 % | 62.5 % | 0.822 | 21.7 % / 96.9 % | 13.2 % / 99.0 % |
| … prefix ≤ 512 | | 49.4 % | 62.2 % | 0.814 | 22.2 % / 97.0 % | 14.2 % / 98.8 % |
| … beam 4 | | 51.1 % | 63.8 % | 0.807 | 19.8 % / 97.3 % | 11.1 % / 98.8 % |

- SPM-only training changes nothing at inference (47.7 % vs 47.9 %) and loses PSM (FIM ppl 4.39): keep spm 0.5.
- lr 2e-3 with the 0.5 M batch is the first recipe change that moves the needle: ppl −4 %, +2.5 p.p. line exact, all metrics in the same direction
  (standard error on 3 000 positions ≈ 0.9 p.p.). Grad norm occasionally hit the 1.0 clip; lr 3e-3 / longer warmup are the next knobs.
  Exported: `csharp/models/cs31m-e2-lr2e3.cml` (int8 check 3.798 → 3.801), the model to ship for C# until cs50m.
- Prefix length: 1450 → 1024 tokens costs nothing, → 512 costs < 1 p.p. — the plugin can prefill 30–65 % less.
- Beam 4 (`eval_inline.py --beam 4`: sum of log-probs incl. the stop token, repeating branches pruned, healing constraints per beam; never returns a
  finished line with lower probability than greedy): +0.9–1.6 p.p. and +1–2 p.p. precision at ≥ 0.8 for ×3.6 generation time — not for the plugin.

**Teacher ceiling** (`tools/nn/eval/eval_hf.py`: a Hugging Face model on the same sampled positions, lexer and metrics; Qwen PSM prompt
`<|fim_prefix|><|file_sep|>path\n…<|fim_suffix|>…<|fim_middle|>`, prefix ≤ 6 000 chars, suffix ≤ 2 000, plain healing, confidence = product of the
greedy probabilities; every added token (`<|fim_pad|>` is not flagged special) ends the middle). Nothing of this ships — it measures the
distillation headroom.

| 500 standard test positions (seed 1), C# | rest of line exact | ≤ 8 | first token | prod ≥ 0.7: shown / exact |
|---|---|---|---|---|
| cs31m-e2-spm10 / cs31m-e2-lr2e3 (ours, 31 M) | 48.0 % / 49.4 % | 62.2 % / 64.0 % | 0.812 / 0.808 | 23.6 % / 93.2 %, 26.6 % / 94.7 % |
| Qwen2.5-Coder-1.5B | 63.8 % | 74.3 % | 0.874 | 26.2 % / 95.4 % |
| Qwen2.5-Coder-7B | 68.4 % | 77.3 % | 0.890 | 35.6 % / 94.4 % |

Paired: 7B and lr2e3 both right at 229 positions, 7B only 113, ours only 18. The teachers' errors are the same kind as ours — names declared in other
files, constants, literals — not syntax. Contamination is real (7B reproduced a `buymeacoffee.com/<user>` URL): split by repository creation date,
before 2024 ours / 1.5B / 7B = 50.3 / 65.7 / 71.2 % (344 positions), created after 2025-07 = 49.1 / 56.2 / 59.8 % (112 positions).

**Fresh-repository evaluation set** (`<lang>/prepared/manifest-fresh.jsonl`, `tools/nn/eval/fresh_manifest.py <lang> 2026-05-01 1000`): repositories created on/after
2026-05-01 (after every candidate teacher's release), not in the lm fold, ≥ 1 000 kept lines — C# 264 repos / 61 596 files / 12.2 M lines,
Go 481 / 129 918 / 31.4 M. 2 000 positions (seed 1) in 1 828 C# files:

| fresh C# positions | rest of line exact | ≤ 8 | first token | rest 1–3 tokens | 4–8 | 9+ |
|---|---|---|---|---|---|---|
| cs31m-e2-lr2e3 (ours) | 40.9 % | 52.7 % | 0.797 | 72.7 % | 34.5 % | 14.2 % |
| Qwen2.5-Coder-1.5B | 56.1 % | 65.6 % | 0.846 | 78.9 % | 53.5 % | 34.5 % |
| Qwen2.5-Coder-7B | 64.1 % | 73.6 % | 0.887 | 85.7 % | 62.5 % | 42.8 % |
| Qwen2.5-Coder-14B | 66.7 % | 75.4 % | 0.894 | 84.5 % | 67.2 % | 46.9 % |
| Qwen2.5-Coder-32B | 66.3 % | 74.5 % | 0.887 | 83.9 % | 65.9 % | 47.9 % |
| Qwen3.6-35B-A3B (instruct, FIM tokens in the vocabulary) | 11.8 % | 12.0 % | 0.236 | — | — | — |

Fresh code is harder for the small model (50.2 → 40.9 %) than for the teacher (68.4 → 64.1 %); paired 7B / ours: both 760, 7B only 523, ours only
59 (1.5B: 381 / 78; 14B: 570 / 55; 32B: 576 / 68). Above 7B the teachers saturate (14B 66.7 %, 32B 66.3 %, 0.5 p.p. apart) at 1.5–2.7× the
generation time, so 7B is the teacher for distillation data and 14B the ceiling reference. The gap grows with the length of the rest (14 vs 43 % at 9+ tokens): multi-token "knowledge" lines. Qwen3.6 (the general model, not a coder
base) does not do fill-in-the-middle — mostly empty middles — so the Qwen2.5-Coder family stays the teacher candidate (Apache-2.0; outputs usable).

**Fix (Windows, reported by the C# plugin build):** `NnFormat.read(File)` memory-mapped the model, and a mapped file cannot be replaced or
deleted on Windows while the mapping lives (Java has no explicit unmap) — `NnModelTest.writeReadRoundTrip` failed on the second write and a
model update in the IDE would too. Now `NnFormat.mapFiles` (default `false` on Windows, `-Dcompletionml.nn.mmap=false` elsewhere) reads the
file into a direct buffer instead (31 MB copy, ~20 ms); `write` replaces the target with an atomic `Files.move`. The test covers both paths.

**cs50m** (`cs50m-e3-lr2e3`: go50m preset d640 × 10, 49.8 M params / 39.3 M non-embedding, the lr2e3 recipe, 5.65 G tokens, 62.5 min on both
GPUs at 1.5 M tok/s): eval ppl 3.60 / FIM 3.77 (31 M: 3.80 / 3.98, −5 %). Standard 3 000 positions: rest of line exact **51.6 %** (31 M 50.2 %),
≤ 8 tokens 63.5 % (62.6 %), first token 0.832 (0.821), prod ≥ 0.7: 27.4 % shown / 93.2 % exact (27.2 % / 94.4 %); fresh repositories 43.5 %
(40.9 %). +1.4 / +2.6 p.p. for 1.7× the compute and ~19 MB more in the jar: capacity is not the C# bottleneck, knowledge of the project
is. Published as `models/cs-nn-50m-e3-lr2e3.cml` (the optional big model); the 31 M stays the default.

**go50m** (`go50m-e3-lr2e3`: go50m preset, the lr2e3 recipe, 6.85 G tokens, 89 min on both GPUs while the C# list export loaded the CPU):
eval ppl 2.83 / FIM 3.12 (31 M: 2.99 / 3.30). Standard 3 000 positions (SPM, healing): rest of line exact **65.7 %** (go31m-e2 63.6 %),
≤ 8 tokens 76.3 % (74.8 %), first token 0.890 (0.881), prod ≥ 0.7: 45.3 % shown / 95.9 % exact (43.8 / 95.7), plain mode 61.8 % (60.0 %);
fresh repositories 59.2 % (56.6 %), ≤ 8 tokens 71.7 % (69.6 %). +2.1 / +2.6 p.p. — Go gains more from capacity than C# (+1.4 / +2.6).
Published as `models/go-nn-50m-e3-lr2e3.cml` (optional big model; the 31 M stays the default for weaker laptops).

**Two epochs for C# 31 M** (`cs31m-e4-2ep`: the lr2e3 recipe over 11.3 G tokens, 21 553 steps, 83 min on both GPUs): eval ppl 3.65 / FIM 3.82
(1 epoch: 3.80 / 3.98) but rest of line exact 50.0 % vs 50.2 %, ≤ 8 tokens 62.3 % vs 62.6 %, first token 0.825 vs 0.821; fresh repositories
42.8 % vs 40.9 %. The lower perplexity does not turn into more exact lines on the standard test (+1.9 p.p. on fresh code only): C# is not
data-starved at 31 M; the model is not published. (go50m / cs50m gained more from capacity than from a second pass.)

**Line-FIM** (`--line-rate 0.9 --single-line 0.8`: 90 % of FIM middles end at a line end, 80 % of those span one line — the inference
situation; `cs31m-e5-line`, 1 epoch): ppl 3.83 / 4.03 (base 3.80 / 3.98), rest of line exact 48.8 % vs 50.2 %, fresh repos 41.5 % vs 40.9 %.
No gain (slightly negative on the standard test): the default mix (50 % line-aligned, half of them one line) already covers the case.
**Proxy calibration**: the same pair at 8.6 M (`go5m` preset, 1.5 G tokens, 20 min per run): 40.2 % vs 40.3 % — the proxy predicted
"no effect" correctly. Working rule from here: data/format hypotheses on the 8.6 M proxy (minutes), winners straight to the 50 M models; the
31 M models are not retrained further (user decision 2026-10-07).

**Go ranker re-exported and retrained (e17b).** The Go export now sees dependencies: the corpus snapshots had no `go.mod` (`fetch.sh` kept only
`*.go`), so `tools/psi/prepare-mods.sh` fetches `go.mod`/`go.sum` from GitHub into an overlay and runs `go mod download` outside the IDE; per
repository +20–31 % lists, recall 0.888 → 0.920, receiver type known at 92 % of `.`-positions (was 79 %), ~190 ms per position
(`tools/psi/REPORT.md`). 150 rank + 100 test repositories, 5 positions per file, 8 direct JVMs, 60 min: 21 767 / 11 845 lists.

| order of the list (11 845 test lists, 100 repos) | top-1 | top-5 | MRR |
|---|---|---|---|
| plugin rules | 0.380 | 0.662 | 0.513 |
| n-gram e14-b only | 0.376 | 0.551 | 0.466 |
| **e17b ranker (210 weights)** | **0.700** | **0.926** | **0.799** |

Per context (MRR): after `.` 0.738, statement start 0.824, argument 0.855, type position 0.843, assignment rhs 0.792, other 0.782 — e17's
0.808 reproduced with half the repositories. Model: `models/go-rank-e17b.cml` (also `idea-golang-support/ml-models/go/`); shards `data/go-psi/`.

**C# ranker on real plugin lists (the e18 goal).** The .NET plugin agent built the headless export (`mlDataset`: the plugin's own completion at
sampled positions, 19 language features in `CSharpMlFeatures` + 13 common, candidate names kept); on the server 16 workers in git worktrees
(`-Pml.maxFiles=60 -Pml.maxCopy=400 -Pml.names=true`; CodeVision/daemon off and `--no-daemon` were needed, see the task file) exported
**127 rank-fold repositories, 22 081 lists**, 57 candidates per list, in ~6 h; stopped there by the user (enough for 224 weights).
Split by repository 4:1 (the plugin's test-fold export did not run): `ml-train l1` 17 411 / 4 670 lists, 10 epochs, seconds.

| order of the list (4 670 test lists) | top-1 | top-5 | MRR |
|---|---|---|---|
| plugin rules (`rule_rank_log`) | 0.367 | 0.718 | 0.526 |
| n-gram e15-a only | 0.176 | 0.313 | 0.253 |
| most frequent in file | 0.279 | 0.639 | 0.442 |
| **e18 ranker (224 weights)** | **0.588** | **0.870** | **0.711** |

Per context (MRR): after `.` 0.647, statement start 0.657, argument 0.736, type position 0.723, assignment rhs 0.779, other 0.721. Heaviest
standardised weights: exact-case prefix match +1.31, `needs_using` (argument) −0.87, keyword at statement start −0.85, upper-case initial in a
type position +0.78, scope level after `.` −0.70, in vocabulary +0.66 — the same picture as Go e17. Growth with data: 83 repos → 0.743 on a
smaller test split, 127 → 0.711 on a harder one; the Go ranker reached 0.808 with 300 + 295 repositories. Model: `models/cs-rank-e18.cml`,
also in `idea-dotnet-support/ml-models/csharp/`; the IDE weigher on the same features is the plugin's next step. Shards: `data/csharp-psi/`.

**First live run of go-nn-31m-e2 in the Go plugin (0.2.199–0.2.202, reported by the plugin agent) → three engine changes** (`NnCompletion`,
mirrored in the harness): (1) **word-start healing** — `return le⟨⟩` is a pre-token boundary (the word ends at the caret), so the model
continued a finished ` le` with `(` (confProd 0.003–0.05) while `len` typed gave `(o.items)` at 0.73; `Options.healMode = WORD_EOL`
heals from the start of the word when only closers/whitespace follow the caret → ` len(o.items)` for `l`, `le` and `len` alike
(P(` len`) = 0.996; the 3 000-position evals are unchanged within 0.1 p.p., they rarely sample that situation). (2) **closers after the
caret**: the editor pairs `)`; `trimClosersAfterCaret` drops the suggestion tail that repeats them. (3) **default gate 0.7** instead of
0.8 (table in docs/NN-COMPLETION-API.md: Go 25.8 % shown / 93.5 % exact, C# 14.2 % / 94.4 %); dropping the newline probability from
the product does not improve the trade-off. Heal-parity fixture regenerated with go31m-e2 (`make_parity_heal.py --heal word-eol`).

**Project bigram cache at decode time — negative result** (`eval_inline.py --project-cache λ`, `ProjectCache`: bigram counts over the BPE
tokens of the other files of the same repository, the completed file subtracted, mixed as p = (1−λ)·p_model + λ·p_cache(prev)):
Go go31m-e2 63.6 → 63.4 % (λ 0.1) / 63.5 % (0.2), C# cs31m-e2-lr2e3 50.2 → 50.3 % (0.05) / 50.2 % (0.1); paired flips Go +3 / −8,
C# +6 / −5 of 3 000 — the model already has the local statistics of the file, and the missing facts (member and argument names) are not
bigram-shaped. The mixture also caps every token probability at 1−λ+λ·p_cache, which collapses the confidence gate (Go at ≥ 0.7: 43.8 → 20.3 %
shown). Not implemented in `ml-core`; the project/dependency *name* information is better used as structure in the prompt (signatures,
member lists) and as a filter after `.` (PSI candidates) — the next experiments.

**DDP** (`train.py` under `torchrun --nproc_per_node 2`): data groups striped by rank (`PackedStream(rank, world_size)`, unit test
`test_ddp_striping`), `--tokens-per-step` stays the global batch, rank 0 builds the path-token cache and writes checkpoints carrying every rank's stream
state (resume only with the same world size), loss all-reduced for logging, eval/metrics on rank 0, `require_backward_grad_sync` once per step.
**Go baseline on the scrubbed corpus** (`go31m-e2`: go31m preset, new `go-16384.bpe`, fim 0.7 / spm 0.5, the lr 2e-3 / 0.5 M recipe, 6.85 G tokens,
13 065 steps in 49 min on both GPUs): eval ppl 2.99 plain / 3.30 FIM (int8 export 2.988 → 2.989). Standard whole-line evaluation (3 000 positions,
2 560 files, SPM, healing): rest of line exact **63.6 %** (plain 60.0 %), ≤ 8 tokens 74.8 %, first token 0.881; prod ≥ 0.8 shown 36.1 % with
97.0 % exact lines, ≥ 0.9: 26.7 % / 98.1 %. go31m-e1 (e16, unscrubbed corpus, broken PSM, other vocabulary and test fold) had 61.7 % / 73.3 % / 0.88:
not worse, slightly better, and publishable — `models/go-nn-31m-e2.cml`.

| fresh Go positions (2 000, repos created ≥ 2026-05-01) | rest of line exact | ≤ 8 | first token | rest 1–3 / 4–8 / 9+ | only teacher / only ours |
|---|---|---|---|---|---|
| go31m-e2 (ours) | 56.6 % | 69.6 % | 0.860 | 83.9 / 51.7 / 24.2 % | — |
| Qwen2.5-Coder-1.5B | 63.9 % | 74.7 % | 0.871 | 84.7 / 62.2 / 36.8 % | 248 / 103 |
| Qwen2.5-Coder-7B | 70.6 % | 80.7 % | 0.901 | 87.4 / 72.2 / 45.4 % | 357 / 78 |

On Go the gap to the 7B teacher is 14 p.p. (C#: 23), on short rests (1–3 tokens) 3.5 p.p.; our model shows 33 % of positions at
prod ≥ 0.8 with 97 % exact lines — the same share the 7B shows at its 0.7 gate with 94 %.

Smoke on the C# shards (6 min + 3 min resume): resume continues exactly (step 399 → 591, loss continuous); throughput while both GPUs were shared with the
teacher evaluations 1.3 M tok/s; clean measurement on idle GPUs **2.45 M tok/s** (1.96× one GPU at 1.25 M; 428 ms per 1 M-token step), so a
31 M epoch takes 47 min instead of 77. The Go baseline `go31m-e2` (scrubbed corpus, fim 0.7, lr 2e-3 / 0.5 M) runs on both GPUs with it.
