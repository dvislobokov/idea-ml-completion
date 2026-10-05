#!/usr/bin/env bash
# Standard measurement since e05: all cloned repos, repo-level split, dedup 0.8, cache λ=0.3.
# usage: tools/bench/std.sh <csharp|go> <tag> [extra l2 args...]
set -euo pipefail
lang="$1"; tag="$2"; shift 2
root="$(cd "$(dirname "$0")/../.." && pwd)"
case "$lang" in
  csharp) split="--test-repos Flow-Launcher__Flow.Launcher,JamesNK__Newtonsoft.Json";;
  go) split="--split repo";;
esac
common="--cache 0.3 --dedup 0.8 $split"
REPOS="$root/../ml-data/$lang/sets/all.txt" L1_ARGS="$common" "$root/tools/bench/run.sh" "$lang" "$tag" $common "$@"
