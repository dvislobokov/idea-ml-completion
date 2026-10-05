#!/usr/bin/env bash
# Concatenate the corpus statistics and all experiment summaries into ../ml-data/report.txt (paste it back).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/../ml-data/report.txt"
{
  "$root/tools/server/stats.sh"
  for f in "$root"/../ml-data/*/exp/*/summary.txt; do [ -f "$f" ] && { echo; cat "$f"; }; done
} > "$out"
echo "written $out ($(wc -l < "$out") lines)"
