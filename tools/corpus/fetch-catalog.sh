#!/usr/bin/env bash
# Download every repository of a catalogue (tools/corpus/enumerate.py output) as a source-only snapshot with fetch.sh, in parallel.
#
# usage: fetch-catalog.sh <csharp|go> <catalog.jsonl> [--jobs J] [--limit N] [--min-stars S] [--dry-run]
#   non-archived repositories only, most starred first; already downloaded ones are skipped by fetch.sh, so the command is resumable.
#   Progress: ../ml-data/fetch-<lang>.log (one line per repository); failures are listed there as "error:".
set -euo pipefail
lang="$1"; catalog="$2"; shift 2
jobs=16; limit=0; minstars=0; dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) jobs="$2"; shift 2;;
    --limit) limit="$2"; shift 2;;
    --min-stars) minstars="$2"; shift 2;;
    --dry-run) dry=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
here="$(cd "$(dirname "$0")" && pwd)"
log="$here/../../../ml-data/fetch-$lang.log"
selected=$(jq -r --argjson s "$minstars" 'select(.archived==false and .stars>=$s) | "\(.stars) \(.full_name)"' "$catalog" | sort -rn | awk '{print $2}')
[ "$limit" -gt 0 ] && selected=$(echo "$selected" | head -n "$limit")
echo "$(echo "$selected" | grep -c .) repositories selected" >&2
[ "$dry" = 1 ] && { echo "$selected" | head -20; exit 0; }
echo "$selected" | FETCH_MODE=tar xargs -P "$jobs" -n 1 "$here/fetch.sh" "$lang" 2>> "$log"
echo "done: $(grep -c ' files (tar)' "$log") downloaded, $(grep -c '^error' "$log") failed" >&2
