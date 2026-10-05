#!/usr/bin/env bash
# Train + evaluate L2 and L1 for one language on a fixed repo set and print the summary lines.
# usage: tools/bench/run.sh <csharp|go> <tag> [extra l2 args...]   (set REPOS=path to override the include list)
set -euo pipefail
lang="$1"; tag="$2"; shift 2
root="$(cd "$(dirname "$0")/../.." && pwd)"
data="$root/../ml-data/$lang"; repos="${REPOS:-$data/sets/base13.txt}"
out="$data/exp/$tag"; mkdir -p "$out"
ML="$root/ml-train/build/install/ml-train/bin/ml-train"
{
  echo "### $lang / $tag  ($(date +%F' '%T))  repos=$(basename "$repos") l2-args: $*"
  "$ML" l2 --lang "$lang" --data "$data" --repos "$repos" --out "$out/lm.cml" --order 4 "$@" 2>&1 | grep -aE "repos,|vocabulary|model:|written|perplexity|next identifier|done in"
  "$ML" l1 --lang "$lang" --data "$data" --repos "$repos" --lm "$out/lm.cml" --out "$out/rank.cml" --epochs 6 2>&1 | grep -aE "examples:|^ranker|^baseline: n-gram|inference|done in"
} | tee "$out/summary.txt"
