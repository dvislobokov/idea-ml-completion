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
- Nothing leaves the machine except GitHub (this repo is public) — no corpus or model uploads elsewhere.

## This server (161.104.59.19)
- 32× EPYC 9554, 176 GB RAM, RTX PRO 6000 Blackwell 96 GB (CUDA 13.4), Ubuntu 26.04, JDK 21, Node 22, uv.
  `/dev/sda1` 400 GB system (corpus lives here now); `/dev/sdb` 2 TB unformatted, reserved for the rest of the corpus — ask before touching.
- Repo: `~/work/idea-ml-completion` (git works directly here). CLI: `./gradlew :ml-train:installDist` → `ml-train/build/install/ml-train/bin/ml-train`.
- Data: `~/work/ml-data/`
  - `catalog/go-20.jsonl`, `catalog/csharp-20.jsonl` — all non-fork GitHub repos ≥20 stars, size>300 KB (34 217 Go / 38 156 C# non-archived)
    with stars, size, licence, pushed_at (`tools/corpus/enumerate.py`).
  - `go/repos/` — ALL 34 215 Go repos downloaded (source-only snapshots, ~190 GB). `csharp/repos/` — top-10 000 by stars (~45 GB).
    Download tool: `tools/corpus/fetch-catalog.sh <lang> <catalog> --jobs 32 [--limit N]` (resumable, skips existing).
    Logs: `fetch-go.log`, `fetch-csharp.log`. A disk guard loop stops downloads under 40 GB free.
  - Old server (157.22.134.66, unreachable since 2026-10-06) held the previous 900-Go/737-C# corpus and experiments e11–e13; nothing needed from it.
- PyTorch: `~/work/nn/.venv` (Python 3.12, torch 2.14 cu130, GPU verified).
- Claude Code: settings in `~/.claude/settings.json` (OAuth token + proxy in `env`).

## State of the art (see docs/REPORT-GO-RU.md, docs/REPORT-CSHARP-RU.md, README experiment table)
- Go LM: order 5 MKN, repo pruning `--min-count 1,1,3,3,3 --min-repos 1,1,2,5,5`: 27 MB, ppl 4.7, OOV 21.5 %, next-identifier top-1 0.497.
- Go ranker on real plugin completion lists: MRR 0.783 / top-1 0.675 vs plugin rules 0.534 / 0.394. C#: MRR 0.712 (proxy lists), 20 MB LM.
- e13 inline (n-gram greedy continuation): confidence ≥0.8 shown at 9 % of positions, 90 % right.
- Go plugin (branch `migration`) already runs the ranker behind `-PmlEnabled=true` (models from `../ml-data/go/models`), ML items marked " ML".
- GigaCode context providers for both plugins are done on branches `gigacode` (Go 0.2.186, C# 0.1.104), not merged/pushed, not yet verified live.

## Plan (agreed with the user, in order)
1. Speed up training: tokenise once into binary shards, parallel n-gram counting and feature extraction across 32 cores,
   ranker feature search on a 100k-list subsample. Training must accept a size cap (repos by stars / files per repo / token budget).
2. Rerun n-gram LM + ranker on the big corpus (Go full, C# 10k) with the standard measurement; experiment rows e14+.
   Then ranker features (declaration distance, already-used-in-function, expected-type match, project frequency) and a small
   gradient-boosted ranker with JVM inference.
3. Own neural model: BPE tokenizer (Kotlin and Python must tokenise identically, parity test), decoder transformer 30→100 M params
   with fill-in-the-middle, trained on the GPU; export int8 weights to `.cml`; Kotlin inference in `ml-core` (KV cache, multi-thread);
   compare with n-grams by `ml-train eval-inline` on 30 held-out repos; wire into plugins via `inline.completion.provider`.
4. "Mapping" suggestions (`member.Name = dto.Name;`) as PSI candidates + ranker; training pairs mined from `a.X = b.Y` in the corpus.
5. Later: download the remaining C# repos onto the 2 TB disk; retrain; C# plugin ranker adapter.
