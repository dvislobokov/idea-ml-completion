#!/usr/bin/env bash
# Source-only snapshot of repositories: fetch.sh <lang> <owner/repo> [<owner/repo>...]
# lang: csharp -> *.cs ; go -> *.go ; destination ../ml-data/<lang>/repos/<owner>__<repo>
#
# Two transports: a sparse, history-less git clone (keeps `.git` so the commit is known), and — when the git smart protocol
# is unusable from the machine ("expected flush after ref listing" through some providers/proxies) — the HEAD tarball from
# codeload.github.com, extracted to the same layout with the commit recorded in `.commit`. FETCH_MODE=tar forces the tarball.
set -euo pipefail
lang="$1"; shift
case "$lang" in csharp) pat='*.cs';; go) pat='*.go';; *) echo "unknown lang" >&2; exit 1;; esac
root="$(cd "$(dirname "$0")/../.." && pwd)/../ml-data/$lang/repos"
mkdir -p "$root"

clone_git() {   # $1 repo, $2 dst
  git clone -q --depth 1 --filter=blob:none --no-checkout "https://github.com/$1.git" "$2" 2>/dev/null &&
  git -C "$2" sparse-checkout set --no-cone "$pat" '/LICENSE*' '/NOTICE*' '/README*' >/dev/null 2>&1 &&
  git -C "$2" checkout -q 2>/dev/null &&
  echo "$1 $(git -C "$2" rev-parse HEAD) $(find "$2" -name "$pat" | wc -l) files (git)" >&2
}

clone_tar() {   # $1 repo, $2 dst
  local tgz; tgz="$(mktemp --suffix=.tgz)"
  if ! curl -sfL --retry 3 --max-time 900 -o "$tgz" "https://codeload.github.com/$1/tar.gz/HEAD"; then rm -f "$tgz"; return 1; fi
  local top; top="$(tar -tzf "$tgz" 2>/dev/null | head -1 | cut -d/ -f1)"
  [ -n "$top" ] || { rm -f "$tgz"; return 1; }
  mkdir -p "$2"
  tar -xzf "$tgz" -C "$2" --strip-components=1 --wildcards "$pat" '*/LICENSE*' '*/NOTICE*' '*/README*' 2>/dev/null || true
  rm -f "$tgz"
  echo "${top##*-}" > "$2/.commit"      # codeload names the top directory <repo>-<sha>
  echo "$1 $(cat "$2/.commit") $(find "$2" -name "$pat" | wc -l) files (tar)" >&2
}

for r in "$@"; do
  dst="$root/${r/\//__}"
  if [ -d "$dst" ] && [ -n "$(find "$dst" -name "$pat" -print -quit)" ]; then echo "skip $r" >&2; continue; fi
  rm -rf "$dst"
  if [ "${FETCH_MODE:-git}" != "tar" ] && clone_git "$r" "$dst"; then continue; fi
  rm -rf "$dst"
  clone_tar "$r" "$dst" || { rm -rf "$dst"; echo "error: $r: both git and tarball failed" >&2; }
done
