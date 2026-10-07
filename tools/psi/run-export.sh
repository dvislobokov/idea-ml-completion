#!/usr/bin/env bash
# Runs one headless export class of the Go plugin directly in the IDE test JVM (the same JVM the Gradle `testIde` task
# `:go-psi-ide:mlDataset` / `mlContext` starts), so that several exports can run in parallel on disjoint repository lists.
# usage: run-export.sh <GoMlDatasetExport|GoMlContextExport> <worker-id> <log> -Dml.repos=... -Dml.out=... [-Dml.*=...]
# Prerequisites: one Gradle run of the task (prepared sandbox + instrumented test classes), worker-cmdline.txt and
# test-classpath.txt captured from it (see README.md). No Gradle build may run while the workers are alive.
set -euo pipefail
CLASS=$1; W=$2; LOG=$3; shift 3
PLUGIN=/root/work/idea-golang-support
SB=$PLUGIN/.intellijPlatform/sandbox/go-psi-ide/IC-2026.1.4
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$SB/system_w$W" "$SB/log_w$W"
[ -d "$SB/config_w$W" ] || cp -r "$SB/config_mlDataset" "$SB/config_w$W"
ARGS=$(mktemp "$HERE/.jvmargs-w$W-XXXX")
# JVM options of the captured Gradle test worker minus the worker-specific ones (sandbox paths, ml.*, heap, Gradle worker main)
grep -E '^(-D|-X|--add-opens|--enable-native-access|-ea)' "$HERE/worker-cmdline.txt" \
  | grep -v -E '^-D(ml\.|idea\.config\.path|idea\.system\.path|idea\.log\.path|org\.gradle\.internal\.worker\.tmpdir|intellij\.testFramework\.rethrow\.logged\.errors)|^-Xmx|^-Xms' > "$ARGS"
{
  echo "-Didea.config.path=$SB/config_w$W"
  echo "-Didea.system.path=$SB/system_w$W"
  echo "-Didea.log.path=$SB/log_w$W"
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
exec /root/work/idea/jbr/bin/java "@$ARGS" org.junit.runner.JUnitCore "io.github.golangsupport.ml.$CLASS" >> "$LOG" 2>&1
