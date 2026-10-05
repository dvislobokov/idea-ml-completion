#!/usr/bin/env bash
line="$1"; fn=$(echo "$line" | jq -r .full_name)
d=$(gh api "repos/$fn" 2>/dev/null) || exit 0
echo "$line" | jq -c --argjson d "$d" '. + {created_at: $d.created_at, forks: $d.forks_count, watchers: $d.subscribers_count, open_issues: $d.open_issues_count}'
