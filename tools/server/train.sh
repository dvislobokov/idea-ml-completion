#!/usr/bin/env bash
# Full training + evaluation for one language with the standard measurement (cross-fitted folds, dedup, cache λ=0.3).
# usage: tools/server/train.sh <csharp|go> <tag> [extra l2 args...]
#   env: XMX_LM (default 40g), XMX_RANK (default 20g), PER_FILE (ranker lists per file, default 20),
#        MAX_TEST (test files for the LM eval, default 4000), L1_ARGS (extra ranker args)
# Output: ../ml-data/<lang>/exp/<tag>/{lm.cml,rank.cml,l2.log,l1.log,summary.txt}; paste summary.txt back.
set -euo pipefail
lang="$1"; tag="$2"; shift 2
root="$(cd "$(dirname "$0")/../.." && pwd)"
data="$root/../ml-data/$lang"; out="$data/exp/$tag"; mkdir -p "$out"
ML="$root/ml-train/build/install/ml-train/bin/ml-train"
test_repos="$(paste -sd, "$data/sets/test.txt")"
common="--dedup 0.8 --test-repos $test_repos"
timer() { if command -v /usr/bin/time >/dev/null; then /usr/bin/time -f "$1: %e s wall, %M KB max RSS" "${@:2}"; else "${@:2}"; fi; }
{
  echo "### $lang / $tag  ($(date +%F' '%T))  host=$(hostname) l2-args: $*"
  if ! JAVA_OPTS="-Xmx${XMX_LM:-40g}" timer l2 "$ML" l2 --lang "$lang" --data "$data" --repos "$data/sets/lm.txt" $common --cache 0.3 \
       --max-test-files "${MAX_TEST:-4000}" --out "$out/lm.cml" "$@" > "$out/l2.log" 2>&1; then
    echo "l2 FAILED, see $out/l2.log"; tail -5 "$out/l2.log"; exit 1
  fi
  grep -aE "repos,|Dedup|vocabulary|order [0-9]+:|model:|written|perplexity|next identifier|^l2:" "$out/l2.log"
  if ! JAVA_OPTS="-Xmx${XMX_RANK:-20g}" timer l1 "$ML" l1 --lang "$lang" --data "$data" --repos "$data/sets/rank.txt" $common --cache 0.3 \
       --per-file "${PER_FILE:-20}" --lm "$out/lm.cml" --out "$out/rank.cml" --epochs 6 ${L1_ARGS:-} > "$out/l1.log" 2>&1; then
    echo "l1 FAILED, see $out/l1.log"; tail -5 "$out/l1.log"; exit 1
  fi
  grep -aE "examples:|^ranker|^baseline|^  [a-z_:]+ +[+-]|inference|^l1:" "$out/l1.log" | head -30
  echo "sizes: lm $(du -h "$out/lm.cml" | cut -f1), rank $(du -h "$out/rank.cml" | cut -f1)"
} 2>&1 | tee "$out/summary.txt"
