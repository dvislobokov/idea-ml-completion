#!/usr/bin/env bash
# Runs one headless export class of the Go plugin directly in the IDE test JVM (the same JVM Gradle's `:go-psi-ide:mlDataset`
# starts), so several exports can run in parallel on disjoint repository lists and the IDE system directory (indices, VFS)
# can be reused between runs instead of being rebuilt in a fresh sandbox every time.
# usage: run-export.sh <GoMlDatasetExport|GoMlContextExport> <worker-id> <log> -Dml.repos=... -Dml.out=... [-Dml.*=...]
# env:  ML_HEAP        JVM heap (6g)
#       ML_SYSTEM      "fresh" (default): empty system dir per start, like Gradle's sandbox;
#                      "shared": use $HERE/system/base directly (ONE worker at a time: warm-up run, builds the GOROOT/module index);
#                      "copy": copy $HERE/system/base to $HERE/system/w<id> before the start (parallel workers reuse the warm index)
#       ML_GOMODCACHE  module cache passed as -Dgopsi.gomodcache (/root/go/pkg/mod; fill it with `go mod download`, see prepare-mods.sh)
# Prerequisites: one Gradle run of the task (prepared sandbox + instrumented test classes), worker-cmdline.txt and
# test-classpath.txt captured from it (GO-HEADLESS-EXPORT.md). No Gradle build may run while workers are alive.
set -euo pipefail
CLASS=$1; W=$2; LOG=$3; shift 3
PLUGIN=/root/work/idea-golang-support
SB=$PLUGIN/.intellijPlatform/sandbox/go-psi-ide/IC-2026.1.4
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${ML_SYSTEM:-fresh}
case $MODE in
  fresh) SYS=$HERE/system/w$W; rm -rf "$SYS" ;;
  shared) SYS=$HERE/system/base ;;
  copy) SYS=$HERE/system/w$W; rm -rf "$SYS"; cp -r "$HERE/system/base" "$SYS" ;;
  *) echo "ML_SYSTEM must be fresh|shared|copy" >&2; exit 2 ;;
esac
mkdir -p "$SYS" "$HERE/log/w$W" "$HERE/config"
[ -d "$HERE/config/w$W" ] || cp -r "$SB/config_mlDataset" "$HERE/config/w$W"
ARGS=$(mktemp "$HERE/.jvmargs-w$W-XXXX")
# JVM options of the captured Gradle test worker minus the worker-specific ones (sandbox paths, ml.*, heap, Gradle worker main)
grep -E '^(-D|-X|--add-opens|--enable-native-access|-ea)' "$HERE/worker-cmdline.txt" \
  | grep -v -E '^-D(ml\.|gopsi\.gomodcache|idea\.config\.path|idea\.system\.path|idea\.log\.path|org\.gradle\.internal\.worker\.tmpdir|intellij\.testFramework\.rethrow\.logged\.errors)|^-Xmx|^-Xms' > "$ARGS"
{
  echo "-Didea.config.path=$HERE/config/w$W"
  echo "-Didea.system.path=$SYS"
  echo "-Didea.log.path=$HERE/log/w$W"
  echo "-Dgopsi.gomodcache=${ML_GOMODCACHE:-/root/go/pkg/mod}"
  echo "-Djava.awt.headless=true"
  # a logged error (LOG.error) must not abort the export; the test logger still reports them at the end
  echo "-Dintellij.testFramework.rethrow.logged.errors=false"
  echo "-Xms256m"
  echo "-Xmx${ML_HEAP:-6g}"
  for a in "$@"; do echo "$a"; done
  echo "-cp"
  sed 's|^[^:]*gradle-worker.jar:||' "$HERE/test-classpath.txt"
} >> "$ARGS"
cd "$PLUGIN/go-psi-ide"
echo "ml: run-export $CLASS w$W system=$MODE:$SYS gomodcache=${ML_GOMODCACHE:-/root/go/pkg/mod} $*" >> "$LOG"
/root/work/idea/jbr/bin/java "@$ARGS" org.junit.runner.JUnitCore "io.github.golangsupport.ml.$CLASS" >> "$LOG" 2>&1
rc=$?; rm -f "$ARGS"; exit $rc
