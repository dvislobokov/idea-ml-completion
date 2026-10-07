# ML completion engine — working notes for Claude Code

Shared ML engine for the Go (`idea-golang-support`) and C# (`idea-dotnet-support`) IntelliJ plugins: n-gram LM + listwise
ranker today, own neural model next. Everything here is pure Kotlin/JVM (`ml-core` ships inside the plugins); training
tooling is `ml-train` (Kotlin CLI) plus, for the neural model, PyTorch under `~/work/nn` (training only, never in plugins).

## Communication and conventions
- Answer the user in Russian; code, identifiers, comments and commit messages in English.
- One commit per experiment: CHANGELOG entry (`eNN`) + metrics row in README. Standard measurement: repo split, dedup 0.8,
  cache λ=0.3, cross-fitted lm/rank/test folds. Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Do not commit or push without being asked. Never commit secrets or data.
- Keep shell calls short and single-purpose (the user dislikes long chained commands). Long jobs: `systemd-run --unit=<name>
  --collect -p WorkingDirectory=$PWD bash -c "... > log 2>&1"` so they survive the SSH session; `nohup ... &` inside ssh did not.
- Subagents: sonnet/haiku unless strong reasoning is needed; opus for the plugin-side work.

## Hard constraints from the user
- No third-party pretrained models (Qwen, llama-family, …) and no native runtimes (llama.cpp, ONNX) in the plugins.
  A neural model must be our own, trained offline, exported to our `.cml` format and run by Kotlin code in `ml-core`.
- Nothing may be required from the user (no vmoptions edits). JBR ships NO `jdk.incubator.vector` (verified on JBR 25.0.3; the flag
  breaks IDE startup), so the scalar Kotlin path is the production path; no preview APIs in `ml-core`. Design doc: `docs/NEURAL-RU.md`.
- Nothing leaves the machine except GitHub (this repo is public) — no corpus or model uploads elsewhere.

## This server (161.104.59.19) — REINSTALLED by the provider on 2026-10-07 (everything on `/` was lost)
- Now: VM with 32 vCPU, 235 GB RAM, 2 × H200 NVL 143 GB (driver 615.71, CUDA 13.4), Ubuntu 26.04, JDK 21, Node 22, uv, Go 1.27.1
  (`/usr/local/go`), .NET SDK 10, gh, zig, tmux, nvtop. GPUs: no NVLink bridge, no P2P (VM chipset) — NCCL all_reduce goes through
  host memory at ~28 GB/s (62 MB of bf16 grads for go31m ≈ 2 ms/step, fine); bf16 matmul 718 TFLOPS per GPU; H2D 53 GB/s.
  `/dev/sda1` 400 GB system (empty); `/dev/sdb` 2 TB ext4 at `/mnt/corpus` (fstab by UUID, `noatime`) — SURVIVED with the C# corpus
  (`/mnt/corpus/csharp/repos`, 38 150 repos, 113 GB). Both disks mounted `noatime`.
- LOST with the system disk (see `docs/MIGRATION-SERVER-RU.md` §1 for what each was): `~/work/ml-data/{catalog,tokenizer,go,csharp/{prepared,bpe16k,models,nn,psi}}`,
  `~/work/nn/{psi,psi-cs,research,clean/out}`, the Go corpus (`go/repos`, 190 GB), IDEA at `/root/work/idea`, plugin clones.
  To rebuild: `gh auth login` → `tools/corpus/enumerate.py` (catalogues) → `fetch-catalog.sh go` (~25 min) → `prepare`/`shard`/`train_bpe`/`encode`
  for both languages; models e14/e15/e17 and the transformers must be retrained; real Go completion lists need IDEA + plugin again.
- PyTorch: `~/work/nn/.venv` (Python 3.12, torch 2.14.1+cu130, NCCL 2.30, 2-GPU torchrun verified with `~/work/nn/nccltest.py`).
  `tools/nn/` in the repo is the only copy of the training code now (`~/work/nn` has just the venv and test scripts).
- GitHub: repo cloned over https; the server's new deploy key `~/.ssh/id_ed25519.pub` must be added to GitHub before pushing
  (then `git remote set-url origin git@github.com:dvislobokov/idea-ml-completion.git`).
- Claude Code 2.1.292 via npm; `~/.claude/settings.json` has the proxy env (HTTPS_PROXY to the user's Caddy forward proxy, NO_PROXY,
  autoupdater off); the OAuth token still has to be added (`claude setup-token`).
- Run Claude Code inside `tmux` (`tmux new -s ml`, `claude --resume`): without it an SSH drop kills the session and its subagents.
  Subagents: `fable` by default in this project (its quota is separate from sonnet/opus, which hit the weekly limit on 2026-10-06);
  parallel agents in git worktrees. Rebuilding `ml-train` while a run is in flight crashes it (`NoClassDefFoundError`) — copy
  `ml-train/build/install/ml-train` elsewhere for long runs.
- Repo: `~/work/idea-ml-completion` (git works directly here). CLI: `./gradlew :ml-train:installDist` → `ml-train/build/install/ml-train/bin/ml-train`.
- Data: `~/work/ml-data/`
  - `catalog/go-20.jsonl`, `catalog/csharp-20.jsonl` — all non-fork GitHub repos ≥20 stars, size>300 KB (34 217 Go / 38 156 C# non-archived)
    with stars, size, licence, pushed_at (`tools/corpus/enumerate.py`).
  - `go/repos/` — ALL 34 215 Go repos downloaded (source-only snapshots `*.go` + LICENSE/README, ~190 GB, 4 fetch errors).
    `csharp/repos/` → symlink to `/mnt/corpus/csharp/repos`: all 38 149 non-archived repos (old sda copy in `csharp/repos.sda-old`, delete once verified).
    `go/prepared/manifest.jsonl` + `stats.json` from `ml-train prepare` (fold lm/rank/test by md5 of repo name, `--test 300`); `go/shards/` lexer
    token shards; `tokenizer/go-16384.bpe` (+32k) BPE vocabularies and parity fixtures; `go/bpe16k/` BPE-encoded folds for the neural model.
    Download tool: `tools/corpus/fetch-catalog.sh <lang> <catalog> --jobs 32 [--limit N]` (resumable, skips existing);
    GitHub tarballs arrive at ~170 MB/s here, the whole Go catalogue took ~25 min. Logs: `fetch-go.log`, `fetch-csharp.log`.
    A background disk-guard loop (`pgrep -f "df --output"`) kills downloads under 40 GB free — remaining C# repos go to the 2 TB disk.
  - Old server (157.22.134.66, unreachable since 2026-10-06) held the previous 900-Go/737-C# corpus and experiments e11–e13; nothing needed from it.
- PyTorch: `~/work/nn/.venv` (Python 3.12, torch 2.14 cu130, GPU verified).
- Claude Code 2.1.291 installed via npm (`claude.ai/install.sh` and `api.anthropic.com` are blocked from RU without proxy);
  settings in `~/.claude/settings.json` (`env`: OAuth token from `claude setup-token`, HTTP(S)_PROXY, autoupdater off).

## State of the art (2026-10-06; details in docs/NEURAL-RU.md §7a/7b, CHANGELOG e14–e16, README)
- N-grams on the full corpora: Go e14-b (`go/models/e14-b.cml`, 32 MB, ppl 4.1, inline ≥0.8: 10 % shown / 93 % right),
  C# e15-a (`csharp/models/e15-a.cml`, 32 MB, ppl 5.9, inline 3 % / 88 %); proxy rankers MRR 0.759 / 0.756.
  Go ranker on real plugin lists (e10): MRR 0.783 / top-1 0.675 vs plugin rules 0.534 / 0.394 — real C# lists do not exist yet.
- Own transformers (31 M, d512×8, BPE 16k, FIM, 1 epoch, 2–2.5 h each): `go/models/go-nn-31m-e1.cml` (ppl 2.13; whole line shown
  in 32 % of positions at 95 % precision vs n-gram 10 %; rest of line exact 66 % vs 35 %; trained BEFORE the secret filter — not for
  release) and `csharp/models/cs-nn-31m-e1.cml` (ppl 3.97; 18 % at 92 % vs n-gram 3 %; 49 % vs 32 %; secret-scrubbed corpus).
  Inference layout: SPM (PSM stays weak at 31 M); gate on the product of token probabilities (prod ≥0.8), not on the first-3 mean.
- Kotlin inference (`ml-core/nn`) reproduces PyTorch (1000/1000 lines); native kernels (`native/`, q8 default) ×3.2: 1 500-token prompt
  + 20 tokens = 167 ms vs 530 ms scalar, 24 ms vs 47 ms while typing (8 threads, this server). NEON / Windows / macOS loading untested.
- Eval harness for the neural models: `tools/nn/eval/eval_inline.py --lang go|csharp --ckpt … --positions 3000 --modes plain,fim,spm`
  (~5 min on the GPU); results in `~/work/ml-data/<lang>/nn/eval-inline-step*.md`; parity fixtures in `go/nn/parity/`, `csharp/nn/parity/`.
- Token healing (2026-10-07, `ml-core/nn/NnCompletion.kt`, `docs/NN-COMPLETION-API.md`): honest whole-line numbers are Go 61.7 % / C# 47.9 %
  (rest of line exact, all positions); gate on conf_prod ≥0.8; the plugin provider should call `NnCompletion.complete(path, before, after)`.
- e17 (2026-10-07): Go ranker on REAL plugin lists exported headless on the server (595 repos, 204 k positions; `tools/psi/`):
  MRR 0.808 / top-1 0.710 vs rules 0.527 — `go/models/e17-rank.cml` is the one to ship. Proxy-trained rankers do NOT transfer
  (Go 0.518, C# 0.337 < LM-only) — the C# ranker e15 must be retrained on real lists (e18, exporter prototype in `~/work/nn/psi-cs/`).
- PSI-context headroom measured (Go, 2 436 positions): 81 % of rest-of-line identifiers are already in the file prefix, PSI adds 4.8 %
  (12.7 % right after `.`), 14.5 % are nowhere; C# (Roslyn, 3 000 positions): 75 % / 9 % / 16 %. PSI compression is mostly a speed lever;
  recall needs member lists of resolved third-party types (module cache / NuGet index on the server — 35 % of C# `.`-positions unresolved).
- Headless plugin builds work on this server: IDEA Community 2026.1.4 at `/root/work/idea` (`-PlocalIdePath`, JBR as JAVA_HOME), Go 1.27.1
  at `/usr/local/go`, .NET SDK 10, fontconfig. Recipes: `tools/psi/GO-HEADLESS-EXPORT.md`; plugin-side changes are kept as
  `tools/psi/go-plugin-headless-export.patch` (not committed to the plugin repo).
- Go plugin (branch `migration`) already runs the ranker behind `-PmlEnabled=true` (models from `../ml-data/go/models`), ML items marked " ML".
- GigaCode context providers for both plugins are done on branches `gigacode` (Go 0.2.186 in worktree `idea-golang-support-gigacode`,
  C# 0.1.104 in `idea-dotnet-support-gigacode`) on the user's Windows machine only: not merged, not pushed, not verified live with GigaCode.
  Design: compile-only stub module `gigacode-api` mirroring GigaCode 26.9.3 interfaces, optional `<depends>` + `*-gigacode.xml`.

## Server migration (decided 2026-10-07): the user moves to 2 × H200 — follow `docs/MIGRATION-SERVER-RU.md`
What to copy (~45 GB without corpora), what to re-download, first task on arrival: DDP in `tools/nn/train/train.py`. Stopped before the move:
agent S (C# real-list exporter → e18) — resume from `~/work/ml-data/csharp/psi/REPORT.md` + `~/work/nn/psi-cs/` (copied over).

## Decisions 2026-10-07 (after the reinstall)
- Native kernels: SHIP. Verified on a MacBook Pro M1 Pro: dylib loads from the jar, NEON+DotProd self-test passes, scalar/f32 are
  bit-exact with PyTorch, q8 behaves as VNNI on x86; cs31m line 68 ms native q8 vs 252 ms scalar (docs/MAC-CHECK-RU.md).
- The repository will carry BOTH a ~30 M and a ~50 M model per language (`models/`); the plugin gets a switch between them later.
  So the 50 M runs (preset go50m) are part of the queue, not an option.
- C# experiments e2 started 2026-10-07 10:35 on the two H200s (1.25 M tok/s each, ~75 min per epoch): `cs31m-e2-spm10`
  (spm-rate 1.0) and `cs31m-e2-lr2e3` (lr 2e-3, 0.5 M tokens/step), both fim 0.7, 1 epoch, logs/ckpts in `~/work/ml-data/csharp/nn/`.
  Baseline numbers for cs31m-e1 come from CHANGELOG e16 (its .pt checkpoint was lost; only the exported .cml survives in `models/`).
  Research notes with the ablation order: the user's machine, `~/work/ml-research/architecture-review-2026-10-07.md`.
- Both runs building `lm.pathtok.u16` at the same time race (os.replace on the .tmp): start the second run a minute later or
  pre-build the cache once.

## Plan (agreed with the user, in order) — items 1, 2 (n-gram part), 3 (first models) and 5 (download) are DONE as of 2026-10-06
Open decisions for the user: (a) ship our own native kernels (spike done, ×3–4; needs a test on the user's Mac: NEON + dylib loading);
(b) PSI context compression in the training format (decide before the next big run); (c) hardware — 2×B300 would turn 20-hour teacher
runs into 2-hour ones; current server is enough until distillation.
Next steps, in order:
1. Retrain Go with the secret-scrubbed corpus, clean FIM from step 1, `fim_rate 0.7` (`tools/nn/train/README.md` has the command);
   consider go50m / 2 epochs; abliations one at a time (FIM share, context, vocab) — ~2.5 h per 31 M run.
2. Plugin side (repos are NOT on this server): `inline.completion.provider` with incremental KV-cache reuse between keystrokes,
   background prefill on file open, JIT warm-up, SPM prompt, gate on prod ≥0.8, no one-token closers (`)`, `;`), repetition guard,
   PSI validation of generated identifiers; C# ranker adapter + collection of real C# completion lists.
3. Quality levers (see docs/NEURAL-RU.md §7, "what beats FLCC"): PSI context in the prompt (types/signatures from other files),
   mixture with the project n-gram cache, constrained decoding over PSI candidates, distillation from a 300 M–1 B teacher.
4. Ranker features + GBDT (plan item 2, second half); "mapping" suggestions (`member.Name = dto.Name;`) as PSI candidates + ranker.
5. Housekeeping: remove the agent worktrees under `.claude/worktrees/`; old C# manifests `csharp/prepared-v1,-v2` and
   `go/prepared-prev,-crashed,-300` can be deleted; `go31m-e1/ckpt-before-fimfix.pt` is the pre-fix checkpoint.
