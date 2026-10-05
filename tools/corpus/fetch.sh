#!/usr/bin/env bash
# Sparse, history-less clone of source files only: fetch.sh <lang> <owner/repo> [<owner/repo>...]
# lang: csharp -> *.cs ; go -> *.go ; destination ../ml-data/<lang>/repos/<owner>__<repo>
set -euo pipefail
lang="$1"; shift
case "$lang" in csharp) pat='*.cs';; go) pat='*.go';; *) echo "unknown lang" >&2; exit 1;; esac
root="$(cd "$(dirname "$0")/../.." && pwd)/../ml-data/$lang/repos"
mkdir -p "$root"
for r in "$@"; do
  dst="$root/${r/\//__}"
  if [ -d "$dst/.git" ]; then echo "skip $r" >&2; continue; fi
  git clone -q --depth 1 --filter=blob:none --no-checkout "https://github.com/$r.git" "$dst"
  git -C "$dst" sparse-checkout set --no-cone "$pat" '/LICENSE*' '/NOTICE*' '/README*' >/dev/null
  git -C "$dst" checkout -q
  echo "$r $(git -C "$dst" rev-parse HEAD) $(find "$dst" -name "$pat" | wc -l) files" >&2
done
