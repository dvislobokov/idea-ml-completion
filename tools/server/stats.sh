#!/usr/bin/env bash
# Corpus statistics: repositories, source files, bytes per language (paste the output back).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
for lang in go csharp; do
  d="$root/../ml-data/$lang/repos"; [ -d "$d" ] || continue
  case "$lang" in go) ext=go;; csharp) ext=cs;; esac
  repos=$(ls -d "$d"/*/ 2>/dev/null | wc -l)
  files=$(find "$d" -name "*.$ext" -type f | wc -l)
  gb=$(find "$d" -name "*.$ext" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.1f", s/1024/1024/1024}')
  failed=$(grep -c -E "^(fatal|error)" "$root/../ml-data/fetch-$lang.log" 2>/dev/null || true)
  echo "$lang: $repos repos, $files files, $gb GB of .$ext, clone errors: ${failed:-0}"
done
echo "disk: $(df -h "$root" | awk 'NR==2 {print $4 " free"}')"
