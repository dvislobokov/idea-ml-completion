#!/usr/bin/env bash
# Module files + module cache for the headless export: the corpus snapshots carry only *.go (no go.mod/go.sum), so for every
# repository of <list> the root go.mod and go.sum are fetched from GitHub (HEAD) into $OVERLAY/<repo>/ and their dependencies
# are downloaded into $GOMODCACHE with `go mod download` (outside the IDE; GOPROXY default). Resumable: repositories with an
# overlay directory are only re-downloaded (fast when cached). Repositories without a root go.mod are listed in $OVERLAY/no-go-mod.txt.
# usage: prepare-mods.sh <repo-list> [jobs=4]
set -uo pipefail
LIST=$1; JOBS=${2:-4}
OVERLAY=${OVERLAY:-/root/work/go-psi/overlay}
export HOME=/root GOPATH=/root/go GOMODCACHE=${GOMODCACHE:-/root/go/pkg/mod} GOCACHE=/root/.cache/go-build GOFLAGS=-mod=mod GOTOOLCHAIN=local PATH=/usr/local/go/bin:$PATH
mkdir -p "$OVERLAY"
one() {
  n=$1; o=${n%%__*}; r=${n#*__}; d=$OVERLAY/$n
  if [ ! -f "$d/go.mod" ]; then
    mkdir -p "$d"
    if ! curl -s -f -m 20 -o "$d/go.mod" "https://raw.githubusercontent.com/$o/$r/HEAD/go.mod"; then rmdir "$d"; echo "$n" >> "$OVERLAY/no-go-mod.txt"; echo "no go.mod: $n"; return; fi
    curl -s -f -m 20 -o "$d/go.sum" "https://raw.githubusercontent.com/$o/$r/HEAD/go.sum" || rm -f "$d/go.sum"
  fi
  ( cd "$d" && timeout 900 go mod download > download.log 2>&1 && echo "ok: $n" || echo "download failed: $n ($(tail -1 download.log))" )
}
export -f one; export OVERLAY
grep -v '^#' "$LIST" | grep -v '^$' | xargs -P "$JOBS" -I{} bash -c 'one {}'
