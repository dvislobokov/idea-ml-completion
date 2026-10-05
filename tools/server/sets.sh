#!/usr/bin/env bash
# Build the repository folds for one language from what is on disk: test (held out), lm, rank (disjoint, 2:1).
# Deterministic: repositories are ordered by the hash of their name.
# usage: tools/server/sets.sh <csharp|go> [--test N]      (default: 30 held-out repositories)
set -euo pipefail
lang="$1"; shift; ntest=30
while [ $# -gt 0 ]; do
  case "$1" in
    --test) ntest="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
root="$(cd "$(dirname "$0")/../.." && pwd)"; d="$root/../ml-data/$lang"; mkdir -p "$d/sets"
ls "$d/repos" | while read -r r; do echo "$(printf '%s' "$r" | md5sum | cut -c1-8) $r"; done | sort | cut -d' ' -f2 > "$d/sets/all.txt"
head -n "$ntest" "$d/sets/all.txt" > "$d/sets/test.txt"
tail -n +"$((ntest + 1))" "$d/sets/all.txt" > "$d/sets/train.txt"
awk 'NR%3!=0' "$d/sets/train.txt" > "$d/sets/lm.txt"
awk 'NR%3==0' "$d/sets/train.txt" > "$d/sets/rank.txt"
# the held-out repositories are listed in both folds and excluded from training via --test-repos
cat "$d/sets/test.txt" >> "$d/sets/lm.txt"
cat "$d/sets/test.txt" >> "$d/sets/rank.txt"
echo "$lang: all $(wc -l < "$d/sets/all.txt"), test $ntest, lm $(($(wc -l < "$d/sets/lm.txt") - ntest)), rank $(($(wc -l < "$d/sets/rank.txt") - ntest))"
