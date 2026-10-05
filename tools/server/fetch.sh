#!/usr/bin/env bash
# Clone the corpora for both languages (sparse, no history; resumable) and print size statistics.
# usage: tools/server/fetch.sh [--limit N] [--jobs J]      (defaults: all repositories, 12 parallel clones)
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
limit=""; jobs=12
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) limit="--limit $2"; shift 2;;
    --jobs) jobs="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$root/../ml-data"
for lang in go csharp; do
  echo "== fetching $lang ($(date +%T))"
  "$root/tools/corpus/fetch-all.sh" "$lang" $limit --jobs "$jobs" 2> "$root/../ml-data/fetch-$lang.log" || true
  tail -1 "$root/../ml-data/fetch-$lang.log"
done
"$root/tools/server/stats.sh"
