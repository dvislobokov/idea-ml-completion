#!/usr/bin/env bash
# Parallel real-list export: splits the rank and test repository lists into N parts and starts N direct-JVM workers
# (run-export.sh GoMlDatasetExport), each processing its rank part then its test part. Output: ~/work/ml-data/go/psi/{rank,test}/<repo>.cmlx
# usage: launch-dataset.sh <workers> [perFile=10] [maxFiles=120]
set -euo pipefail
N=${1:-8}; PER_FILE=${2:-10}; MAX_FILES=${3:-120}
D=/root/work/ml-data/go/psi; HERE=/root/work/nn/psi
LM=/root/work/ml-data/go/models/e14-b.cml
mkdir -p "$D/rank" "$D/test" "$D/sets/parts" "$HERE/logs"
for fold in rank test; do split -n l/$N -d -a 1 "$D/sets/$fold.txt" "$D/sets/parts/$fold-"; done
for i in $(seq 0 $((N - 1))); do
  cat "$D/sets/parts/rank-$i" > "$D/sets/parts/w$i-rank.txt"
  cat "$D/sets/parts/test-$i" > "$D/sets/parts/w$i-test.txt"
  systemd-run --unit="ml-export-w$i" --collect -p WorkingDirectory=$HERE bash -c "
    $HERE/run-export.sh GoMlDatasetExport $i $HERE/logs/export-w$i-rank.log -Dml.repos=$D/sets/parts/w$i-rank.txt -Dml.lm=$LM -Dml.data=/root/work/ml-data/go -Dml.out=$D/rank -Dml.perFile=$PER_FILE -Dml.maxFiles=$MAX_FILES -Dml.names=true;
    $HERE/run-export.sh GoMlDatasetExport $i $HERE/logs/export-w$i-test.log -Dml.repos=$D/sets/parts/w$i-test.txt -Dml.lm=$LM -Dml.data=/root/work/ml-data/go -Dml.out=$D/test -Dml.perFile=$PER_FILE -Dml.maxFiles=$MAX_FILES -Dml.names=true"
done
