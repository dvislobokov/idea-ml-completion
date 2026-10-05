#!/usr/bin/env bash
# Clone repositories from a reviewed list (data/<lang>-repos.csv, sorted by stars) with fetch.sh.
#
# usage: fetch-all.sh <csharp|go> [--limit N] [--max-gb G] [--jobs J] [--csv path] [--dry-run]
#   --limit N    clone at most N repositories (first N rows of the list)
#   --max-gb G   stop when the estimated source size (sum of the list's size_mb column, git-packed size) exceeds G gigabytes
#   --jobs J     parallel clones (default 4)
#   --dry-run    print what would be cloned and the total, clone nothing
# Already cloned repositories are skipped by fetch.sh, so the command is resumable.
set -euo pipefail
lang="$1"; shift
limit=0; maxgb=0; jobs=4; dry=0; csv="$(dirname "$0")/data/${lang}-repos.csv"
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) limit="$2"; shift 2;;
    --max-gb) maxgb="$2"; shift 2;;
    --jobs) jobs="$2"; shift 2;;
    --csv) csv="$2"; shift 2;;
    --dry-run) dry=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
selected=$(tail -n +2 "$csv" | awk -F'","|^"' -v limit="$limit" -v maxmb="$(awk -v g="$maxgb" 'BEGIN{printf "%d", g*1024}')" '
  { repo=$2; sub(/".*/, "", repo); n++;
    split($0, cols, ","); size=cols[4]
    if (limit > 0 && n > limit) exit
    total += size
    if (maxmb > 0 && total > maxmb) exit
    print repo }')
count=$(echo "$selected" | grep -c . || true)
echo "selected $count repos from $csv (limit=$limit max-gb=$maxgb)" >&2
if [ "$dry" = 1 ]; then echo "$selected"; exit 0; fi
echo "$selected" | xargs -P "$jobs" -n 1 "$(dirname "$0")/fetch.sh" "$lang"
echo "fetched: $(ls -d "$(dirname "$0")/../../../ml-data/$lang/repos"/*/ 2>/dev/null | wc -l) repos on disk" >&2
