#!/usr/bin/env bash
# Collect candidate repos for a language via GitHub Search API, sliced by star ranges
# to work around the 1000-results-per-query cap. Output: data/<lang>-search.jsonl
set -euo pipefail
lang="$1"            # csharp | go
ghlang="$2"          # "C#" | Go
out="$(dirname "$0")/data/${lang}-search.jsonl"
touch "$out"
ranges=("300..450" "450..700" "700..1000" "1000..1500" "1500..2200" "2200..3200" "3200..5000" "5000..8000" "8000..15000" "15000..40000" ">40000")
for r in "${ranges[@]}"; do
  q="language:\"$ghlang\" fork:false archived:false stars:$r pushed:>2025-06-01"
  for page in 1 2 3 4 5 6 7 8 9 10; do
    res=$(gh api -X GET search/repositories -f q="$q" -f sort=stars -f order=desc -f per_page=100 -f page=$page 2>/dev/null || echo '{"items":[]}')
    n=$(echo "$res" | jq '.items | length')
    echo "$res" | jq -c '.items[] | {full_name, stars: .stargazers_count, pushed_at, size, license: .license.spdx_id, topics, description, owner: .owner.login, owner_type: .owner.type, default_branch}' >> "$out"
    echo "range=$r page=$page got=$n" >&2
    [ "$n" -lt 100 ] && break
    sleep 2.2
  done
done
sort -u "$out" -o "$out"
wc -l "$out" >&2
