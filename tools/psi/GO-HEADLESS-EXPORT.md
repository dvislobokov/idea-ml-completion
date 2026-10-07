# Headless Go PSI exports on this server (idea-golang-support → ml-data)

What runs here: the Go plugin's own PSI (go-psi-core / go-psi-semantic / go-psi-ide) inside a headless IntelliJ test IDE,
driven by two JUnit "export" classes of the plugin repository (`go-psi-ide/src/test/kotlin/io/github/golangsupport/ml/`):

| class | Gradle task | output |
|---|---|---|
| `GoMlDatasetExport` (existing, e10) | `:go-psi-ide:mlDataset` | real completion lists → `.cmlx` shards for `ml-train l1 --shards` |
| `GoMlContextExport` (new) | `:go-psi-ide:mlContext` | PSI context per sampled cursor position → JSONL (neural prompt design) |

## Environment that works

- IDE: IntelliJ IDEA Community 2026.1.4 (`IC-261.26222.65`) unpacked at `/root/work/idea` (GitHub release tarball of
  `JetBrains/intellij-community`; the unified `download.jetbrains.com` tarball was not a gzip here). Passed with
  `-PlocalIdePath=/root/work/idea` on every Gradle call — the tracked `gradle.properties` points at a Windows path and stays untouched.
- JDK for Gradle: the IDE's JBR, `JAVA_HOME=/root/work/idea/jbr` (system OpenJDK 21 also exists; the plugin's CLAUDE.md asks for the JBR).
- Go: `/usr/local/go` (1.27.1): `PATH=/usr/local/go/bin:...`, `GOROOT=/usr/local/go`. `go-psi-ide/build.gradle.kts` resolves GOROOT by
  `go env GOROOT` and passes `-Dgopsi.goroot`; the test base pins the toolchain to linux/amd64 + cgo, no `go` binary needed at run time.
  The module cache (`-Dgopsi.gomodcache`) does not exist here: imports of external modules stay unresolved (`GoUnknownType`, empty
  resolve), the standard library resolves from `$GOROOT/src`, project packages resolve through the copied `go.mod`.
- Fonts: the test editor creates an `EditorImpl` even headless → `Fontconfig head is null` without fonts. Fixed with
  `apt-get install fontconfig fonts-dejavu-core`.
- Network: Maven Central, plugins.gradle.org, JetBrains repositories are reachable directly (no proxy configured for Gradle).
  Gradle 9.7.1 wrapper was already in `~/.gradle`. `third_party/delve` submodule had to be initialised (`git submodule update --init`):
  `prepareSandbox_mlDataset` copies delve sources and fails on an empty submodule.
- Configuration cache: `testIde` tasks need `--no-configuration-cache` (as the plugin's CLAUDE.md says).
- First build (`:go-psi-ide:testClasses`): 1 m 46 s. Each `mlDataset`/`mlContext` Gradle invocation: ~25–45 s overhead
  (instrumentation, sandbox) + the export itself.

Gradle form (one JVM, the reference path):

```sh
cd /root/work/idea-golang-support
export JAVA_HOME=/root/work/idea/jbr PATH=/usr/local/go/bin:$PATH GOROOT=/usr/local/go
./gradlew -PlocalIdePath=/root/work/idea --no-configuration-cache :go-psi-ide:mlDataset \
  -Pml.repos=/root/work/ml-data/go/psi/sets/smoke.txt -Pml.lm=/root/work/ml-data/go/models/e14-b.cml \
  -Pml.data=/root/work/ml-data/go -Pml.out=/root/work/ml-data/go/psi/smoke -Pml.perFile=10 -Pml.maxFiles=120 -Pml.names=true
./gradlew -PlocalIdePath=/root/work/idea --no-configuration-cache :go-psi-ide:mlContext \
  -Pml.repos=/root/work/ml-data/go/psi/sets/test.txt -Pml.data=/root/work/ml-data/go \
  -Pml.out=/root/work/ml-data/go/psi/context-test.jsonl -Pml.positions=2400
```

Long runs: `systemd-run --unit=<name> --collect -p WorkingDirectory=$PWD -E JAVA_HOME=... -E PATH=... bash -c "... > log 2>&1"`.

## Parallel export without Gradle (`run-export.sh`)

The export is single-threaded (one fixture, one completion at a time). Several Gradle builds in one checkout would share the
sandbox (`.intellijPlatform/sandbox/go-psi-ide/IC-2026.1.4/{config,system,log}_mlDataset`) and the build directory, so the
workers are started as plain JVMs instead, with exactly the command line Gradle's test worker uses:

- `worker-cmdline.txt`: `/proc/<pid>/cmdline` of the `Gradle Test Executor` of one `mlDataset` run (JBR java, ~120 `-D`/`--add-opens`
  options: `idea.home.path`, `idea.platform.prefix=Idea`, `java.system.class.loader=com.intellij.util.lang.PathClassLoader`, the
  module-descriptors path, `-Xbootclasspath/a:/root/work/idea/lib/nio-fs.jar`, `-Dgopsi.goroot`, ...).
- `test-classpath.txt`: `java.class.path` printed by `GoMlContextExport` from inside the worker (663 entries: instrumented test
  classes, the prepared plugin sandbox jars, `/root/work/idea/lib/*`, test framework, junit 4.13.2).
- `run-export.sh <Class> <worker-id> <log> -Dml.*=...` rewrites `idea.config.path` (copy of `config_mlDataset`), `idea.system.path`,
  `idea.log.path` per worker, adds `-Djava.awt.headless=true`, `-Xmx6g` (`ML_HEAP`), and runs
  `org.junit.runner.JUnitCore io.github.golangsupport.ml.<Class>` from `go-psi-ide/` (cwd of the Gradle worker).
- `launch-dataset.sh <N> [perFile] [maxFiles]` splits `sets/rank.txt` and `sets/test.txt` into N parts and starts
  `ml-export-w<i>` units; logs in `logs/`. Scripts live in `/root/work/go-psi/` on the server (copies here); run `prepare-mods.sh` first.
- Rule: no Gradle build of the plugin while workers run (it rewrites the sandbox jars and `build/instrumented/instrumentTestCode`).
  After a source change: one Gradle smoke run (`sets/smoke.txt`), then re-capture `test-classpath.txt` if the classpath changed.

Trial (3 small repos, 50 files each, 10 positions/file): identical shards from Gradle and from `run-export.sh`; 19–29 ms per
position, 24 s per JVM including start-up (~10 s), indexing of a small repository 1–3 s.

## Module cache and reused IDE system directory (2026-10-07 evening, `REPORT.md`)

The corpus snapshots hold only `*.go` + LICENSE/README (no `go.mod`/`go.sum`; `tools/corpus/fetch.sh` now keeps them for new
downloads), so the plugin saw every external import as unresolved (`GoUnknownType`) in e17. The fix is outside the IDE:

1. `prepare-mods.sh <repo-list> [jobs]` fetches the root `go.mod` + `go.sum` of every repository from GitHub (HEAD) into
   `/root/work/go-psi/overlay/<repo>/` and runs `go mod download` there (`GOMODCACHE=/root/go/pkg/mod`, `GOFLAGS=-mod=mod`,
   `GOTOOLCHAIN=local`; ~0.5 GB and 1–2 min for a 40-dependency repository, cached modules are free). Repositories without a root
   `go.mod` are listed in `overlay/no-go-mod.txt` (they export as before, with unresolved imports).
2. The exporter copies `overlay/<repo>/go.{mod,sum,work}` over the repository copy (`-Dml.overlay=<dir>`) and the worker JVM gets
   `-Dgopsi.gomodcache=/root/go/pkg/mod` (`run-export.sh`, env `ML_GOMODCACHE`; the Gradle default on Linux is the Windows path
   `$HOME\go\pkg\mod` — pass `-Pgopsi.gomodcache` when running through Gradle). The plugin resolves imports from the extracted
   module directories through its own module graph (MVS over the `cache/download/**/@v/*.mod` files); nothing is copied or indexed.
3. The IDE system directory (`idea.system.path`) is under `/root/work/go-psi/system/`: `ML_SYSTEM=fresh` (default, an empty directory
   per start like Gradle's sandbox), `shared` (one directory reused between runs, one JVM at a time), `copy` (each worker starts from a
   copy of the warmed `system/base`). Reuse pays only together with `-Dml.libraryRoots=stdlib|all` (GOROOT / module directories as
   library roots, indexed as in the IDE): the warmed stub index is picked up (GOROOT 5.4 k files: 22 s cold, 7 s warm scan; all modules of
   gatewayd 26 k files: 87 s cold, 25 s warm) — but completion recall and speed do not change with it, so the default stays `none`/`fresh`.
4. Per position the exporter now replaces only the identifier in the document (incremental reparse) instead of `setText` of the whole
   file twice; the shards are the same lists (−10–15 % per position).

`measure.sh <name> <fresh|shared|copy> <gomodcache> [-Dml.*]` runs one measured configuration on `sets/one.txt` (unit `go-psi-<name>`,
log `logs/<name>.log`, shards `out/<name>/`); `profile.sh` samples the stacks of the running worker. Summary line fields added by this
round: `dot=<.-positions> dot-resolved=<receiver type or package known> (<share>) index-ms=<copy+refresh+index> wall-ms=<repository>`.

Direct JVM runs need the instrumented test classes: after a source change run `:go-psi-ide:instrumentTestCode` (not just `testClasses` —
the worker's classpath has `build/instrumented/instrumentTestCode`, and `testClasses` alone leaves stale classes there).

## Data layout (`/root/work/ml-data/go/psi/`)

- `sets/rank.txt` — 300 repositories sampled (seed 17) from the 7 165 `rank`-fold repositories with 20–3000 kept files
  (manifest `prepared/manifest.jsonl`); `sets/test.txt` — all 295 `test`-fold repositories; `sets/smoke.txt` — 3 small test repos.
- `rank/<repo>.cmlx`, `test/<repo>.cmlx` — real completion lists (`ExampleShards`, with candidate names), header
  `source = idea-golang-support GoMlDatasetExport lm=e14-b.cml`; features: 13 common (from the e14-b LM, cache λ=0.3) + 17 language.
- `context-test.jsonl` — PSI context dump (format in `GoMlContextExport.kt`), `context-stats.md` — output of `context_stats.py`.

## Tools in this directory

- `context_stats.py <jsonl> [--md out]` — timings, context coverage, the (a) prefix / (b) PSI-only / (c) nowhere table.
- `tools/EvalProxy.java` — evaluates a ranker of a smaller schema (the proxy ranker `e14-b-rank.cml`, 13 common features) on the
  real-list shards by projecting the base features; prints rules / LM-only / ranker rows per context kind.
  `javac -cp ml-core.jar:kotlin-stdlib.jar -d classes EvalProxy.java; java -cp classes:... EvalProxy <rank.cml> <shards dir> <name>`.
- `export_summary.py <logdir>... [--md out]` — totals of the dataset export logs (`export-summary.md`).
- `REPORT.md` — module cache / shared system directory measurements (2026-10-07 evening).
- `E17.md` — experiment e17 (CHANGELOG entry + README row); `context-stats.md` — PSI-context statistics; the final report was returned to the coordinator.

## Measured on 2026-10-07

- Lists: rank 300 repos / 119 723 lists, test 295 repos / 83 904 lists (27 ms per position, 595 repos in ~26 min on 10 JVMs; the first
  run died on `Incorrect CachedValue use` after 3–17 repos per worker — fixed by unique temp dirs + `rethrow.logged.errors=false`).
- Context dump: 2 436 positions in 237 s (one JVM); PSI resolution median 13.5 ms, p90 77 ms per position.
- Training e17: 55 s; MRR 0.808 vs rules 0.527 vs proxy 0.518.
