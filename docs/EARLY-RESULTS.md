# Early results (prototype and 13-repo scaling check, 2026-10-05)

Kept for reference; the current numbers are in the README experiment log and in `REPORT-GO-RU.md` / `REPORT-CSHARP-RU.md`.

## Scaling check: 3 → 13 repos per language (split by file 90/10, same test files for both model versions)

Added C#: PowerShell, calculator, semantic-kernel, Bulk-Crap-Uninstaller, MaterialDesignInXaml, Flow.Launcher, Playnite,
BenchmarkDotNet, Newtonsoft.Json, eShop. Added Go: ollama, frp, fzf, dive, lazydocker, etcd, LocalAI, v2ray-core, cli, bubbletea.

| metric (on the 13-repo test split) | Go, 3-repo model | Go, 13-repo model | C#, 3-repo model | C#, 13-repo model |
|---|---|---|---|---|
| train tokens / vocabulary | 0.66 M / 10.7 k | 8.0 M / 50 k | 1.6 M / 18 k | 7.9 M / 50 k |
| perplexity per token | 38.6 | **8.3** | 92.3 | **9.9** |
| identifier OOV rate | 38 % | 8.8 % | 50 % | 17.5 % |
| next identifier, LM alone, top-1 / top-5 | 0.27 / 0.42 | 0.33 / 0.52 | 0.22 / 0.38 | 0.30 / 0.49 |
| ranker MRR / top-1 (proxy lists) | 0.55 / 0.43 | **0.72 / 0.61** | 0.56 / 0.44 | **0.74 / 0.64** |
| LM-only baseline MRR | 0.47 | 0.63 | 0.44 | 0.59 |
| ranker gain over LM-only, MRR | +0.08 | +0.08 | +0.12 | +0.15 |
| lists / avg candidates | 22.8 k / 32 | 22.8 k / 32 | 23.7 k / 29 | 23.7 k / 29 |
| LM size, gzip, no pruning | 5.9 MB | 45 MB | 8.7 MB | 46 MB |

The 3-repo models look fine on their own test split but collapse on unseen repositories: half of the C# identifiers are
out of vocabulary. More repositories fix generalisation first of all; the ranker's gain over the LM-only baseline also grows
with data (C#: +0.12 → +0.15 MRR).

### Model size: what pruning buys (13-repo models)

| `--min-count` (1-gram..4-gram) | Go size | Go perplexity | Go top-1/top-5 | C# size | C# perplexity | C# top-1/top-5 |
|---|---|---|---|---|---|---|
| 1,1,1,1 (none) | 45 MB | 8.3 | 0.33 / 0.52 | 46 MB | 9.9 | 0.30 / 0.49 |
| 1,1,2,2 | 18 MB | 9.5 | 0.32 / 0.50 | 20 MB | 11.9 | 0.28 / 0.46 |
| 1,2,2,3 | 12 MB | 10.0 | 0.32 / 0.49 | | | |
| 1,2,3,3 | 11 MB | 10.3 | 0.32 / 0.49 | 12 MB | 13.1 | 0.28 / 0.45 |
| order 3, no pruning | | | | 20 MB | 13.0 | 0.24 / 0.42 |

Dropping singletons of orders 3–4 cuts the size 2.5× for ~1 point of perplexity and ~1 p.p. of top-1; pruning beats
lowering the order at equal size. Later implemented: 8-bit quantised values and 24-bit fingerprints (e02), SRILM-style
pruning and `--min-repos` (e03, e11).

## Prototype results (3 repos per language, split by file 90/10)

| | Go (cobra, gin, caddy) | C# (spectre.console, Polly, Humanizer) |
|---|---|---|
| tokens / vocabulary | 0.66 M / 10.7 k | 1.6 M / 18.3 k |
| LM size (`lm.cml`, gzip) | 5.9 MB | 8.7 MB |
| perplexity per token | 11.5 | 9.6 |
| next identifier top-1 / top-5 (LM alone, in-vocab) | 0.31 / 0.52 | 0.42 / 0.61 |
| ranker top-1 / top-5 / MRR (proxy lists, avg 38 and 27 candidates) | 0.59 / 0.83 / 0.70 | 0.68 / 0.88 / 0.77 |
| best single-feature baseline (LM log-prob) MRR | 0.64 | 0.73 |
| frequency / recency baselines MRR | 0.42 / 0.43 | 0.41 / 0.41 |
| inference per candidate | 0.03 µs | 0.05 µs |

The ranker was trained on *proxy* lists (candidates = identifiers seen earlier in the same file), not on real completion
lists from PSI/Roslyn; real Go lists came later (e10).

## Memory notes for the tree-walking trainer (pre-e14)

Counting keeps all orders in RAM (8 + 4 + 4·order bytes per distinct n-gram, ×1.4 for the hash table); ~1 G tokens →
~300 M n-grams → ~15–20 GB heap for order 4. Ranker examples are held in RAM compactly (8 base floats per candidate):
~32 bytes × candidates; 300 repos → ~10 GB at the default `--per-file 60`. Both limits are gone with token shards and the
partitioned trainer (e14).
