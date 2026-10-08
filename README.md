# idea-ml-completion

Local ML engine for code completion in the `idea-golang-support` (Go) and `idea-dotnet-support` (C#) IntelliJ plugins:
an n-gram language model with a listwise ranker for the completion list, and our own small transformer for whole-line
("ghost text") suggestions. Everything that runs on the user's machine is Kotlin/JVM inside the plugin — no Python,
no third-party model runtimes, no network. Optional native SIMD kernels (our own C, ~100 KB per platform) speed the
transformer up 3–4× and fall back to Kotlin when absent.

Design notes and all measurements: [`docs/NEURAL-RU.md`](docs/NEURAL-RU.md) (Russian). Experiment history: `CHANGELOG.md`.

## Results (2026-10-06)

Whole-line suggestions, Go, 300 held-out repositories, 3 000 positions, same protocol for both models:

| | n-gram LM (e14) | neural go31m (31 M params) |
|---|---|---|
| positions where a suggestion is shown, at ≈95 % whole-line precision | 10 % | **32 %** |
| rest of the line completed exactly (≤ 8 tokens left) | 35 % | **73 %** (66 % before token healing) |
| first token right | 0.65 | **0.88** (0.80 before token healing) |
| inside string literals (log messages, format strings) | ≈ 0 | shown 36 %, 75 % right |
| model file | 32 MB | 31 MB (int8) |
| 20-token line, 8 threads, this server — Kotlin / native kernels | — | 530 ms / **167 ms** (while typing, KV cache reused: 47 / 24 ms) |

Completion-list ranking (Go, real plugin lists exported headless from the plugin's PSI, 83 904 lists in 295 held-out repos): MRR **0.808** / top-1 **0.710** vs plugin rules 0.527 / 0.388 (e17). A ranker trained on proxy lists does not transfer (0.518) — real lists are required; the C# ranker (e15) is still proxy-trained.
C#: n-gram LM 32 MB, proxy-list ranker MRR 0.756 (e15). C# transformer cs31m (same size, secret-scrubbed corpus): whole-line suggestions shown in 18 % of positions at 92 % precision (n-gram: 3 % at 88 %), rest of line exact 49 % vs 32 %, first token 0.72 vs 0.57 — C# is harder for both models, and the transformer fixes the n-gram's arity failures (`)` after the last argument: 65 % vs the n-gram's literal continuation).

## Layout

```
ml-core/        pure Kotlin, embedded into the plugins (stdlib only, JDK 21, no preview APIs):
                lex/        C# and Go lexers (tokens for the n-gram model and ranker features)
                ngram/      modified Kneser–Ney n-gram LM, per-file cache LM, partitioned trainer
                rank/       listwise linear ranker, feature schema, example shards
                bpe/        byte-level BPE tokenizer (parity-tested against the Python trainer)
                nn/         transformer inference: CMLN model format (int8, mmap), KV cache, scalar kernels,
                            nn/native: JNI loader for the SIMD kernels with scalar fallback
                format/     .cml container
native/         C11 SIMD kernels (AVX2 / AVX-512 (VNNI) / NEON), self-test, benchmark, cross-compiled with zig
ml-train/       Kotlin CLI: prepare, shard, l2 (LM), l1 (ranker), eval-lm, eval-rank, eval-inline, tokens, bench-nn
tools/nn/       Python (PyTorch) — training side only: tokenizer/ (BPE trainer, corpus encoder), train/ (model, data,
                FIM, training loop, .cml export), eval/ (inline-completion harness, parity fixtures), clean/ (secret scrubbing)
tools/corpus/   GitHub catalogue (enumerate.py) and parallel source-only download (fetch-catalog.sh)
tools/server/   server setup, disk preparation, training recipes
docs/           NEURAL-RU.md (design + results), NN-FORMAT.md, NATIVE-KERNELS.md, NN-PARITY.md, ADAPTER.md (plugin
                contract), RESEARCH.md (literature), REPORT-*-RU.md, SERVER-RU.md, EARLY-RESULTS.md
```

## Build and test

JDK 21, Gradle wrapper, Kotlin 2.3.

```sh
./gradlew :ml-core:test :ml-train:test :ml-train:installDist -q   # CLI at ml-train/build/install/ml-train/bin/ml-train
make -C native test                                                # C self-test + benchmark; `make cross` builds all 4 platforms (zig)
```

`ml-core` tests include the BPE parity fixture (small, committed) and the neural inference tests (scalar, native, incremental
decode vs. naive reference). The large parity fixtures (`CML_BPE_FIXTURE`, `~/work/ml-data/.../parity`) are optional and skipped
when absent.

## Embedding into the plugins

The plugins take this repository as a `git subtree` under `ml/` and include `ml/ml-core` in their Gradle build; the adapter
contract (language feature block, offline generator, weigher, parity test) is in `docs/ADAPTER.md`. Models are plain files in
the plugin resources: `<lang>-lm.cml` (n-gram), `<lang>-rank.cml` (ranker), `<lang>-nn.cml` (transformer + BPE vocabulary).
The native kernels are loaded from `ml-core` resources (`native/libcmlkernels-<os>-<arch>.*`) through `NativeLib`; any load or
self-test failure selects the Kotlin kernels (`-Dcompletionml.nn.native=false` forces it; `.native.mode=q8|f32` picks the
activation mode, q8 by default).

## Data pipeline

```sh
# 0. corpus: all non-fork GitHub repositories with ≥ 20 stars (catalogue), source-only snapshots (*.go / *.cs + LICENSE/README)
python3 tools/corpus/enumerate.py go  > ../ml-data/catalog/go-20.jsonl
tools/corpus/fetch-catalog.sh go ../ml-data/catalog/go-20.jsonl --jobs 32          # resumable; ~25 min for 34 k Go repos

# 1. manifest: filters (generated/vendored code, size, encoding), near-dedup 0.8, deterministic folds lm / rank / test
ML=ml-train/build/install/ml-train/bin/ml-train      # copy the install dir before long runs: rebuilding replaces jars under a running JVM
JAVA_OPTS=-Xmx64g $ML prepare --lang go --data ../ml-data/go --catalog ../ml-data/catalog/go-20.jsonl --test 300
#    -> go/prepared/manifest.jsonl (one line per file: repo, path, bytes, lines, tokens, fold, status) + stats.json

# 2a. n-gram path: lexer token shards, LM, ranker, evaluation
$ML shard --lang go --data ../ml-data/go                                              # ~15 min, 23 GB for 5.8 G tokens
JAVA_OPTS=-Xmx110g $ML l2 --lang go --token-shards ../ml-data/go/shards --out ../ml-data/go/models/lm.cml --order 5 \
    --min-count 1,2,5,8,8 --min-repos 1,1,10,40,60 --partitions 64 --threads 24 --cache 0.3 --max-test-files 3000
$ML eval-inline --lang go --token-shards ../ml-data/go/shards --lm ../ml-data/go/models/lm.cml --max-test-files 2000
$ML l1 --lang go --token-shards ../ml-data/go/shards --lm ../ml-data/go/models/lm.cml --out ../ml-data/go/models/rank.cml

# 2b. neural path (server with a GPU; venv with torch): BPE vocabulary, secret-scrubbed uint16 shards, training, export, eval
cd tools/nn/tokenizer && python -I train_bpe.py --lang go --vocab 16384                 # ~5 min on a 1.6 GB stratified sample
cd ../clean      && python -I encode_corpus_clean.py --vocab ../../../../ml-data/tokenizer/go-16384.bpe --lang go --fold lm --workers 24
cd ../train      && python -I train.py --preset go31m --run go31m-e1 --max-tokens 6.8e9 --compile ...   # see train/README.md
                    python -I export.py --ckpt .../ckpt-latest.pt --out ../../../../ml-data/go/models/go-nn-31m.cml --check 128
cd ../eval       && python -I eval_inline.py --ckpt .../ckpt-latest.pt --positions 3000 --modes plain,fim,spm --dump 60
                    # studies without retraining: --max-prefix 512|1024, --beam 4 (sum of log-probs incl. the stop token)
```

Folds are assigned by the md5 of the repository name (first `--test` repositories → test, every third of the rest → rank,
others → lm), so LM, ranker and evaluation never share a repository and every run is reproducible from the manifest.
`--token-shards` replaces `--data` for `l2`, `eval-lm`, `eval-inline`, `l1`, `eval-rank`; the old tree-walking path still works
for small corpora. Long jobs on the server run under `systemd-run` (see `tools/server/` and `CLAUDE.md`).

## Experiment log

`CHANGELOG.md` records one experiment per commit with measurements. Standard measurement: repository split, near-dedup 0.8,
per-file cache LM (λ=0.3), LM / ranker / test on disjoint repository folds, 300 held-out repositories for the full corpora.

| what changed | effect (Go / C#) |
|---|---|
| e01 fixed-context scorer | LM scoring 2–3× faster, identical output |
| e02 24-bit fingerprints + 8-bit quantised values | model 45 → 19 MB, no measurable loss |
| e03 SRILM-style pruning (`--min-count 1,1,2,2`) | 19 → 8 MB for +0.5–0.9 perplexity, −1 p.p. top-1; `--min-repos` added |
| e04 per-file cache LM | perplexity 8.3 → 5.0 / 9.9 → 5.8; identifier top-1 +14 p.p.; ranker MRR +0.04–0.06 |
| e05 dedup + repo split | honest numbers: OOV 9 → 29 % / 17 → 27 %, ranker MRR ≈ 0.71–0.72 (file split was optimistic) |
| e06 JM vs MKN, order 5 | MKN stays; order 5 = −0.4 perplexity for 2× size |
| e07 list features + cross-fitting | leakage found and fixed; 13 features: MRR 0.718 / 0.723 |
| e10 ranker on real Go completion lists (PSI) | held-out repos: MRR 0.783 vs plugin rules 0.534 (top-1 0.675 vs 0.394) |
| e11 full corpus on the server (580 Go repos) | LM ppl 5.5 → 5.0, OOV 29 → 21.5 %; proxy ranker MRR 0.739; repo pruning: 163 → 20 MB at −0.002 MRR |
| e12 full C# corpus (472 repos), Go order 5 | C#: ppl 5.9, proxy ranker MRR 0.718, 20 MB at −0.006; Go order 5 + repo pruning: ppl 4.7, MRR 0.742, 27 MB |
| e13 inline continuation with the n-gram LM | Go: 30 % of lines finished exactly; gated at confidence 0.8: shown 9 % of positions, 90 % right |
| e14 full Go corpus (22 610 repos, 3.9 G tokens): token shards, partitioned counting | order 5, 32 MB: ppl 4.1, top-1 0.521 (207 MB unpruned-ish: 3.8 / 0.525); inline at 0.8: 10 % shown, 93 % right; proxy ranker MRR 0.759 |
| e15 full C# corpus (24 945 repos, 2.95 G tokens), stricter generated-code filters | order 5, 32 MB: ppl 5.9, top-1 0.484 (52 MB: 5.6 / 0.490; 24 MB: 6.1 / 0.483); inline at 0.8: 3 % shown, 88 % right; proxy ranker MRR 0.756 (e12: 0.712) |
| **e16 own transformer go31m** (d512 × 8, 31 M, BPE 16k, FIM, 6.8 G tokens, 2.5 h on one GPU) | Go: ppl 2.13 (BPE); whole-line suggestions at 95 % precision shown in 32 % of positions (n-gram: 10 %), rest of line exact 66 % vs 35 %; int8 export lossless; Kotlin inference reproduces PyTorch 1000/1000 lines; native kernels ×3.2; with token healing at the cursor (`NnCompletion`) rest of line exact 62 % / 73 % (≤ 8 tokens). **cs31m** (C#, 5.65 G tokens, secret-scrubbed): ppl 3.97; shown 18 % at 92 % (n-gram 3 % at 88 %), rest of line exact 49 % vs 32 % (60 % with token healing) |
| e17 ranker on real Go completion lists, corpus scale (300 rank + 295 test repos, 120 k + 84 k lists) | test fold: MRR 0.808 vs plugin rules 0.527 vs proxy ranker 0.518 (top-1 0.710 / 0.388 / 0.390, top-5 0.934 / 0.685 / 0.666); PSI context adds only 4.8 % of rest-of-line identifiers beyond the file prefix |
| e18 C# recipe ablations (spm 1.0, **lr 2e-3 / 0.5 M batch**), prefix/beam studies, teacher ceiling (Qwen2.5-Coder via `eval_hf.py`), fresh-repository eval set, DDP | C#: lr 2e-3 → ppl 3.80 (3.97), rest of line exact 50.2 % (47.9 %); prefix 1024 free, 512 −0.8 p.p.; beam 4 +1 p.p. for ×3.6 time; teachers on the same positions: Qwen2.5-Coder-1.5B 63.8 %, 7B 68.4 % (ours 49.4 %), on repos created after 2026-05: 7B 64.1 % vs ours 40.9 %; Qwen3.6 does not do FIM (11.8 %); CPU models rebuilt (Go e14-b ppl 4.6, C# e15-a 5.3 on the new test folds) |

| e19 GBDT ranker (LightGBM lambdarank → own `tree-ranker` `.cml`, `TreeRanker` in `ml-core`) on the real plugin lists, paired with the linear ranker on the same held-out repositories | Go: MRR 0.834 / top-1 0.750 vs linear 0.799 / 0.700 (rules 0.513); C#: 0.759 / 0.651 vs 0.713 / 0.589 (rules 0.530); 161 / 100 KB, 0.41 / 0.27 ms per list of 50 candidates; Kotlin = LightGBM to 1e-5 |

Earlier prototype and scaling tables: `docs/EARLY-RESULTS.md`.

## Constraints

- No third-party pretrained models and no third-party inference runtimes in the plugins; the transformer is trained from
  scratch on our corpus and run by our Kotlin (and optionally our C) code.
- Nothing is required from the user: no `vmoptions` edits (JetBrains Runtime has no `jdk.incubator.vector`; the scalar Kotlin
  path is the production path), no downloads at runtime.
- Training data: public GitHub repositories (≥ 20 stars, no forks), generated and vendored code removed, near-duplicates
  removed, secrets and personal data scrubbed before encoding (`tools/nn/clean`). Corpora and models are not part of this
  repository.
