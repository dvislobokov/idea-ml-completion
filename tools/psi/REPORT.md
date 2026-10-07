# Go headless export: module cache and reused IDE system directory (2026-10-07, evening)

Question: does caching the standard library / module indexes make the real-list export (e17/e18) faster and more complete?
Setup: plugin 0.2.206 + commit 7b41f6d (`migration`), IDEA 2026.1.4 headless test IDE, Go 1.27.1, server 161.104.59.19 under load
(16 C# export workers + GPU jobs, load ~21/32 — absolute times are ~6× slower than e17's 27 ms/position and noisy ±10 %).
Sampling as e17: ≤120 files, 10 positions/file, prefix 0–2 chars, ≤100 candidates, seed 7. Repository: `gatewayd-io__gatewayd`
(rank fold, 118 files, 43 direct dependencies; 65 non-test files sampled, 640 positions, 191 of them after `.`).

Finding first: the corpus snapshots contain NO `go.mod`/`go.sum` (`tools/corpus/fetch.sh` kept only `*.go` + LICENSE/README; fixed for
future downloads), so in e17 every external import was unresolved. The module files are now fetched from GitHub into an overlay
(`prepare-mods.sh`) and the dependencies downloaded into `/root/go/pkg/mod` outside the IDE.

## gatewayd-io__gatewayd — one configuration at a time

| configuration | wall s | ms/position | positions | lists | recall | `.` receiver resolved | index ms |
|---|---|---|---|---|---|---|---|
| baseline, Gradle `mlDataset` (fresh sandbox) | 137 (+38 s Gradle) | 185 | 640 | 366 | 0.888 | 79.1 % | 1.3 k |
| baseline, direct JVM (`run-export.sh`, fresh system dir) — same shard bit for bit | 134 | 178 | 640 | 366 | 0.888 | 79.1 % | 1.2 k |
| (a) shared system dir, 2nd run, no library roots | 138 | 181 | 640 | 366 | 0.888 | 79.1 % | 3.0 k |
| (b) go.mod overlay + GOMODCACHE, fresh system dir | 163 | 219 | 640 | **480** | **0.920** | **92.1 %** | 1.3 k |
| (a+b) + `libraryRoots=stdlib`, 1st run (GOROOT indexed: 5 373 files) | 173 | 201 | 640 | 480 | 0.920 | 92.1 % | 21.7 k |
| (a+b) + `libraryRoots=stdlib`, 2nd run (index reused, 0 files to index) | 159 | 201 | 640 | 480 | 0.920 | 92.1 % | 8.7 k |
| (a+b) + `libraryRoots=all`, 1st run (GOROOT + 26 385 module files indexed, 405 MB) | 220 | 176 | 640 | 480 | 0.920 | 92.1 % | 86.9 k |
| (a+b) + `libraryRoots=all`, 2nd run (index reused) | 168 | 192 | 640 | 480 | 0.920 | 92.1 % | 24.8 k |
| **(b+c+d) final: overlay + GOMODCACHE, fresh system dir, no library roots, incremental edits** | **144** | **190** | 640 | 480 | 0.920 | 92.1 % | 1.2 k |

- (a) A shared `idea.system.path` by itself changes nothing: each repository is a new temporary content root, so its index is built
  anyway (1–3 s). GOROOT is not indexed at all unless the plugin's `GoRootsProvider` is enabled (off in unit-test mode) — with
  `-Dml.libraryRoots=stdlib|all` and the roots-changed event fired per repository it is, and the warm index IS reused by the next JVM
  (22 s → 9 s for GOROOT, 87 s → 25 s for all modules; the remaining time is the up-to-date scan). But completion recall and speed do
  not depend on the stub index (resolution and member lists come from the package model / PSI, `GoProjectMemberCandidates` uses the
  project scope only), so library roots stay off: `ML_SYSTEM=fresh`, `libraryRoots=none`.
- (b) Module files + module cache is the whole gain: +31 % lists (no-list positions 203 → 87: lists appear where the receiver was
  unknown), recall 0.888 → 0.920, receiver type known at 92 % of `.`-positions instead of 79 %. Costs ~15–20 % per position
  (resolution into the module cache) and ~0.5 GB / 1–2 min of `go mod download` per 40-dependency repository (cached modules are free).
- (c) Already true in the exporter: content root = the repository copy only; `vendor/`, `testdata/`, hidden dirs are not copied,
  `_test.go` are copied (resolution) but not sampled, nothing from the module cache is copied or indexed.
- (d) Replacing only the identifier per position (incremental reparse) instead of `setText` of the whole file twice: 219 → 190 ms.
- Where the rest goes (stack sampling, `profile.sh`): the plugin's completion itself — `GoScopeCandidates.declarationCandidate` →
  `declarationType` → type inference of every candidate (35 % of samples), document edits + reparse (30 %). Every document change
  invalidates the plugin's caches, so each position infers types from scratch; a cheaper lever would be to keep the file untouched
  and complete at a prefix of the identifier (needs a change of the sampling, not done).
- Shards of configurations with the module cache differ from run to run by md5 (candidate order of lists >100 before the
  sampling shuffle); counts and recall are identical.

## Confirmation: final configuration vs baseline on three more rank repositories

| repository | files / positions | lists | recall | `.` receiver resolved | ms/position | wall s |
|---|---|---|---|---|---|---|
| imposter-project__imposter-cli (23 deps) | 115 / 1 146 | 684 → **834** (+22 %) | 0.861 → **0.896** | 77.6 → **97.4 %** | 163 → 178 | 220 → 243 |
| StevenWeathers__thunderdome-planning-poker (36 deps) | 120 / 1 199 | 754 → **903** (+20 %) | 0.905 → **0.919** | 82.7 → **94.7 %** | 164 → 170 | 234 → 251 |
| lucky7xz__drako (8 deps) | 80 / 800 | 565 → **605** (+7 %) | 0.908 → **0.915** | 91.3 → **95.0 %** | 189 → 185 | 179 → 179 |
| gatewayd-io__gatewayd (43 deps) | 65 / 640 | 366 → **480** (+31 %) | 0.888 → **0.920** | 79.1 → **92.1 %** | 178 → 190 | 134 → 144 |

(baseline here = same exporter without overlay / module cache; 3 JVMs in parallel.) The gain scales with the number of external
dependencies; per-position time changes within noise (+0–10 %). Over the e17 set (595 repos) this means ~+20 % more lists and recall
~0.90 instead of ~0.88 for the same wall time, provided `prepare-mods.sh` has run (network: GitHub raw + proxy.golang.org).

## What changed where

- Plugin (`idea-golang-support`, branch `migration`): `go-psi-ide/src/test/kotlin/io/github/golangsupport/ml/GoMlDatasetExport.kt` —
  `ml.overlay`, `ml.libraryRoots=none|stdlib|all` (policy service replaced, roots-changed event fired synchronously), counters
  `dot`/`dot-resolved`/`index-ms`/`wall-ms`, incremental document edits; the e17 robustness fixes, `GoMlContextExport.kt` and the
  `mlContext` Gradle task from the patch. Commits: cc53203 (the patch parts, swept into another agent's ml-core sync commit from the
  same working tree), 7b41f6d (the rest). Feature schema unchanged; `ml-core/` untouched.
- Engine (`tools/psi/`): `run-export.sh` (system-dir modes, `ML_GOMODCACHE`), `prepare-mods.sh` (overlay + `go mod download`),
  `launch-dataset.sh` (overlay + module cache), `measure.sh`, `profile.sh`, `GO-HEADLESS-EXPORT.md` (recipe), this report,
  `go-plugin-headless-export.patch` regenerated against 0.2.206; `tools/corpus/fetch.sh` keeps `go.mod`/`go.sum`/`go.work`.
  Server copies: `/root/work/go-psi/` (scripts, `overlay/`, `system/`, `logs/`, `out/`, `worker-cmdline.txt`, `test-classpath.txt`).

## Did not work / caveats

- `GoRootsProvider.scheduleRootsUpdate` is asynchronous and needs a pumped EDT; in the export the event is fired directly.
  `LocalFileSystem.findFileByNioFile` returns null for GOROOT in a fresh VFS — refreshed explicitly before computing the roots.
- `:go-psi-ide:testClasses` does not refresh `build/instrumented/instrumentTestCode`, which is what the direct JVM runs; use
  `:go-psi-ide:instrumentTestCode` after source changes (two measurements were silently done with stale classes and redone).
- `go mod download` in a systemd unit needs `HOME`/`GOPATH`/`GOMODCACHE` set explicitly.
- Shared (`ML_SYSTEM=shared`) system directories must not be used by two JVMs at once; `copy` mode copies the warmed base per worker.
