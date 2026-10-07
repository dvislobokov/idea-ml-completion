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

## State 2026-10-07 afternoon (e18, details in CHANGELOG e18)
- Data rebuilt for both languages (`tools/server/rebuild-data.sh`); CPU models rebuilt: `go/models/e14-b.cml` (ppl 4.6), `csharp/models/e15-a.cml`
  (ppl 5.3), proxy rankers `*-rank.cml` (do not transfer to real lists — e17). Go BPE vocabulary retrained (`tokenizer/go-16384.bpe`); the old
  `go-nn-31m-e1.cml` is incompatible with it and gone anyway.
- C# recipe: **lr 2e-3, 0.5 M tokens/step** wins (`csharp/models/cs31m-e2-lr2e3.cml`, ppl 3.80, rest of line exact 50.2 %); spm-rate 1.0 = nothing;
  prefix ≤1024 free, ≤512 −0.8 p.p.; beam 4 (+1 p.p., ×3.6 time, harness only: `eval_inline.py --beam`).
- DDP works: `torchrun --nproc_per_node 2 train.py …` = 2.45 M tok/s (1.96×), 31 M epoch 47 min. `go31m-e2` (Go baseline, lr2e3 recipe) started
  13:45 on both GPUs (`go-queue` unit); `go-eval` unit exports + evaluates it afterwards (`go/nn/eval-go31m-e2*`, `fresh-go-*`).
- Teachers (`tools/nn/eval/eval_hf.py`, HF cache `~/.cache/huggingface/hub`, 180 GB: Qwen2.5-Coder 1.5B/7B/14B/32B, Qwen3.6-35B-A3B): on the
  fresh-repo C# set (`csharp/prepared/manifest-fresh.jsonl`, repos created ≥2026-05-01, not lm, ≥1000 lines; `tools/nn/eval/fresh_manifest.py`)
  ours 40.9 %, Qwen2.5-Coder 1.5B 56.1 / 7B 64.1 / 14B 66.7 / 32B 66.3 %; Qwen3.6 cannot FIM (11.8 %). Teacher of choice: 7B (Apache-2.0).
  Contamination of the standard test fold by the teachers is real; compare teachers only on the fresh sets.
- Next in the queue (user-approved direction, not yet started): cs50m/go50m with the lr2e3 recipe on DDP; line-aligned FIM middle distribution;
  PSI as validator / constrained decoding measured on re-exported real Go lists (e17 export has to be redone, ~2 h); sequence-level distillation
  from Qwen2.5-Coder-7B with Roslyn/gopls-verified outputs; GBDT ranker + nn feature. 4 × H200: only worth it for the distillation-generation day.
- Everything is committed and pushed as it lands (user's rule since 16:00: GitHub is the only durable store; other agents build the plugin
  repos). Engine `models/`: cs-nn-31m-e2-lr2e3 (default C#), cs-nn-50m-e3-lr2e3 (ppl 3.60, 51.6 %: +1.4 p.p. for 1.7× compute — optional),
  go-nn-31m-e2 (ppl 2.99, 63.6 %), go-16384.bpe, n-grams, proxy rankers; the same files in `idea-dotnet-support/ml-models/csharp` and
  `idea-golang-support/ml-models/go`. go50m-e3-lr2e3 trains 16:40–18:00 (`go50m-queue`), publish it the same way.
- Live-run feedback from the Go plugin agent (0.2.199–0.2.202) → `NnCompletion`: `healMode WORD_EOL` (heal from the word start when only
  closers follow the caret), `trimClosersAfterCaret`, default gate 0.7; harness `--heal word-eol` default; heal-parity fixture at
  `~/work/ml-data/go/nn/parity-heal` (go31m-e2; test needs `CML_NN_MODEL=…/go/models/go31m-e2.cml`, run with `:ml-core:cleanTest`).
- .NET plugin agent delivered the e18 export (`CSharpMlFeatures` 19 language features, `mlDataset` Gradle task, test-scope exporter;
  tests pass on the server). Export running in 8 git worktrees `/root/work/dotnet-wt/<n>` (units `cs-export-0..7`, script
  `~/work/cs-export-worker.sh`, lists `csharp/psi/lists/`, shards `csharp/psi/{rank,test}/*.cmlx` with candidate names): ~0.6 repos/min in
  total (the exporter copies + indexes the whole repository per repo), rank 300 repos ≈ 8 h, test cut to 100 repos. Then
  `ml-train l1 --lang csharp --shards csharp/psi/rank --test-shards csharp/psi/test` → e18 ranker; the plugin side (weigher) follows.
- Plugin tasks written for other agents: `idea-dotnet-support/ML_INLINE_TASK.md`, `ML_RANKER_EXPORT_TASK.md` (done), `idea-golang-support/ML_INLINE_TASK.md`.
- Known gotchas: two evals must not share `--scratch` (checkpoint copy race); Gradle test env/fixture changes need `cleanTest`; 8 parallel
  Kotlin compilations run out of memory — compile worktrees sequentially (`~/work/cs-export-launch.sh`).

## 2026-10-07 evening — the server will be deleted soon (user's warning at ~20:00)
- Everything that matters is pushed: engine main (models incl. cs50m/go50m, `data/csharp-psi/*.cmlx` real C# lists + `models/cs-rank-e18-pre.cml`),
  Go plugin `migration`, .NET plugin `master`. A unit `shard-sync` pushes new C# shards every 20 min; the checkpoints/evals/fixtures that are
  not on GitHub are packed into `~/work/backup/ckpts-evals-2026-10-07.tar` (user downloads it to their machine).
- Export (`cs-export-0..15`): workers run `gradlew --no-daemon` each (a shared Gradle daemon in the launcher's cgroup was killed by systemd when
  the launcher exited → "daemon disappeared"); CodeVision + daemon timer disabled in the fixture (0db9501), `-Pml.maxCopy=400`. Rate ≈ 0.7 repos/min
  on 16 workers — the remaining cost is the plugin's completion itself (`solutionTypes` scan per position).
- Preliminary e18 ranker on 83 repos / 14 k lists: MRR 0.743 / top-1 0.625 vs plugin rules 0.567 / 0.409 (`ml-train l1 --shards`). Retrain on
  everything exported before the server goes; the final numbers go to CHANGELOG e18 and `cs-rank-e18.cml` to both repos; then the plugin weigher.
- Queue on the GPUs: `cs31m-e4-2ep` (2 epochs, ends ≈20:00) → `cs31m-e5-line` (line-FIM, `--line-rate 0.9 --single-line 0.8`, ≈21:00); each
  exports + evaluates itself (`*-queue.out`). Project bigram cache: measured, useless (CHANGELOG). Next levers: structured context in the prompt,
  PSI filter after `.`, distillation from Qwen2.5-Coder-7B; go50m 2 epochs / line-FIM after the C# results.

## 2026-10-08 night (МСК = UTC+3; report times to the user in МСК)
- Rankers on real lists are DONE and published: Go `models/go-rank-e17b.cml` (MRR 0.799 vs rules 0.513; 150+100 repos, dependencies via
  `tools/psi/prepare-mods.sh` overlay + GOMODCACHE — the corpus has no go.mod) and C# `models/cs-rank-e18.cml` (0.711 vs 0.526; 127 repos,
  export stopped by the user). Same files in the plugins' `ml-models/`; shards in `data/{go,csharp}-psi/`. Plugin side next: weighers.
- 31 M models are frozen (user decision); hypotheses go to the 8.6 M proxy (`go5m`, `tools/server/proxy-run.sh`, gpu `0,1` = DDP), winners to
  50 M. Measured negatives: line-FIM, 2 epochs, project bigram cache, PSI resolve filter (tools/roslyn/REPORT.md: 1 of 46), context block (2 %).
- Distillation (tools/nn/distill): C# 1.5B teacher lines generated (data/csharp-distill, 246 k) but the C# proxy comparison never completed
  (the first run was killed by my own edit of a running script; then the user stopped all C# jobs). Go track running: 7B teacher on 300 k Go
  positions (units distill-go7b-0/1, ~04:15 МСК) → `distill-go-proxy` encodes and trains `go5m-teacher` vs `go5m-base` (running on GPU 0).
  If the proxy gains ≥ 1 p.p.: go50m with `--teacher`. vLLM lives in `~/work/nn/.venv-vllm` (needs CUDA_HOME=<venv>/nvidia/cu13, ninja,
  VLLM_ATTENTION_BACKEND=FLASH_ATTN, VLLM_USE_FLASHINFER_SAMPLER=0); the training venv keeps torch 2.14.1.
- .NET export speed-up by the (stopped) C# agent is on master as 41ca187 (restore + assembly index, type snapshot, debug log off: ×7).
- Deck: docs/talk/slides.md (82 slides, Marp, charts from real numbers), HTML built; PDF needs a browser on another machine.
- Never edit a script while a unit runs it (bash reads incrementally — it killed the C# proxy run). Backup archive for the user:
  ~/work/backup/ckpts-evals-2026-10-07.tar (1.9 GB).

## 2026-10-08 02:30 МСК — the user stopped all GPU work (money); state for whoever continues
- GPUs idle; nothing trains. Distillation is UNDECIDED: no proxy comparison completed (C# killed by my script edit, Go stopped at the teacher
  stage). Saved: C# 1.5B teacher lines (data/csharp-distill, 246 k) and Go 7B lines (data/go-distill, 75 k of 300 k). To finish on any
  GPU box: `proxy-run.sh <lang> <run> 0,1 1.5e9 [--teacher <npz> --teacher-rate 1.0]` for base vs teacher (20 min each), then 50 M if ≥ +1 p.p.
- 03:10 МСК: the .NET plugin agent is DONE (0.1.132, 8a13d1b: inline provider, e18 weigher, settings, preload/prefill, ML zip in
  `build/distributions`); engine d6af9dc adds the stable line-aligned prefix cut (`InlinePrompt.stableTail`, Go 63.7 %, parity 99/99) and
  ml-core is synced into both plugins (Go e3f7f95 on `migration`, .NET 52acaa8 on `master`). Everything is pushed; nothing runs.
- Not on GitHub: checkpoints (.pt) — ~/work/backup/ckpts-evals-2026-10-07.tar (1.9 GB; the four shipped models + evals + parity fixture),
  the module cache /root/go/pkg/mod (50 GB, re-downloadable), HF models (re-downloadable), corpora (re-downloadable in ~1 h).

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
