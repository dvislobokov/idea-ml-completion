#!/usr/bin/env bash
# Parallel enrichment; resumable (skips repos already in output).
set -euo pipefail
lang="$1"; key="$2"; jobs="${3:-8}"
in="data/${lang}-prefilter.jsonl"; out="data/${lang}-enriched.jsonl"
touch "$out"
jq -R -r 'fromjson? // empty | .full_name' "$out" | sort -u > "data/${lang}-done.txt"
jq -R -c 'fromjson? // empty' "$in" | grep -v -F -f <(sed 's/.*/"full_name":"&"/' "data/${lang}-done.txt") > "data/${lang}-todo.jsonl" || true
echo "todo: $(wc -l < data/${lang}-todo.jsonl)" >&2
tr '\n' '\0' < "data/${lang}-todo.jsonl" | xargs -0 -P "$jobs" -n 1 ./enrich_one.sh "$key" >> "$out"
echo "done: $(wc -l < "$out") in $out" >&2
