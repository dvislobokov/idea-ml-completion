#!/usr/bin/env bash
# usage: enrich_one.sh <key> <jsonl-line>  -> prints enriched line
key="$1"; line="$2"
fn=$(echo "$line" | jq -r .full_name)
langs=$(gh api "repos/$fn/languages" 2>/dev/null || echo '{}')
echo "$line" | jq -c --argjson L "$langs" --arg k "$key" \
  '. + {lang_bytes: ($L[$k] // 0), total_bytes: ([$L[]] | add // 0)} | . + {share: (if .total_bytes > 0 then (.lang_bytes / .total_bytes) else 0 end)}'
