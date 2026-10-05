#!/usr/bin/env bash
# Apply share/topic filters and per-owner quotas; emit data/<lang>-repos.csv (top N by stars).
set -euo pipefail
lang="$1"; limit="${2:-300}"
in="data/${lang}-verified.jsonl"; out="data/${lang}-repos.csv"
jq -R -s -r --arg lang "$lang" 'split("
") | map(fromjson? // empty) | unique_by(.full_name)[]
  | 
  select(.share >= 0.6)
  | select(.created_at != null and .created_at < "2024-07-01")          # at least ~2 years of history
  | select((.forks / .stars) >= 0.03 and .forks <= .stars)                 # star-inflated repos have almost no forks; fork farms have more forks than stars
  | select(.watchers >= 20)
  | select((.full_name + " " + (.topics|join(" "))) | ascii_downcase
      | test("awesome|interview|tutorial|roadmap|cheat.?sheet|-book|course|learning|study|exercise|leetcode|algorithms|design-pattern|^[^/]+/docs|examples?($|[^a-z])|samples?($|[^a-z])|templates?($|[^a-z])|boilerplate|starter|quickstart") | not)
  | select((.description // "") | ascii_downcase
      | test("awesome list|curated list|interview|tutorial|roadmap|cheat.?sheet|learning (path|resources)|^samples? (for|of)|^examples? (for|of)|collection of (samples|examples)") | not)
  | [.full_name, .stars, .license, (.size/1024|floor), (.share*100|floor), .pushed_at[0:10], .owner, (.topics|join("|")), ((.description // "")|gsub("[,\n\r]";" ")|.[0:100])]
  | @csv' "$in" \
| sort -t, -k2,2nr \
| awk -F, -v limit="$limit" -v lang="$lang" '
  BEGIN{OFS=","; owner_quota=(lang=="csharp")?8:5; print "repo,stars,license,size_mb,share_pct,pushed,owner,topics,description"}
  {
    owner=$7; gsub(/"/,"",owner)
    quota=owner_quota
    if (lang=="csharp" && (owner=="dotnet" || owner=="microsoft")) quota=15
    if (lang=="go" && (owner=="golang" || owner=="kubernetes" || owner=="hashicorp")) quota=12
    unity = (lang=="csharp" && tolower($0) ~ /unity/)
    if (unity && unity_n>=10) next
    if (cnt[owner]>=quota) next
    cnt[owner]++; if (unity) unity_n++
    print; n++
    if (n>=limit) exit
  }' > "$out"
echo "$out: $(($(wc -l < "$out")-1)) repos" >&2
