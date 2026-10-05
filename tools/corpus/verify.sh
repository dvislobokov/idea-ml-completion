#!/usr/bin/env bash
# Fetch repo details for share-filtered candidates (resumable). Output: data/<lang>-verified.jsonl
set -euo pipefail
lang="$1"; jobs="${2:-8}"
in="data/${lang}-enriched.jsonl"; out="data/${lang}-verified.jsonl"; touch "$out"
jq -R -s -c 'split("\n") | map(fromjson? // empty) | unique_by(.full_name)[] | select(.share >= 0.6)' "$in" > "data/${lang}-cand.jsonl"
jq -R -r 'fromjson? // empty | .full_name' "$out" | sort -u > "data/${lang}-vdone.txt"
grep -v -F -f <(sed 's/.*/"full_name":"&"/' "data/${lang}-vdone.txt") "data/${lang}-cand.jsonl" > "data/${lang}-vtodo.jsonl" || true
echo "verify todo: $(wc -l < data/${lang}-vtodo.jsonl)" >&2
tr '\n' '\0' < "data/${lang}-vtodo.jsonl" | xargs -0 -P "$jobs" -n 1 ./verify_one.sh >> "$out"
echo "verified: $(wc -l < "$out")" >&2
