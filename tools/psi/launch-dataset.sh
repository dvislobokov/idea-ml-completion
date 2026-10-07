#!/usr/bin/env bash
# Parallel real-list export (e17 layout): splits the rank and test repository lists into N parts and starts N direct-JVM workers
# (run-export.sh GoMlDatasetExport), each processing its rank part then its test part.
# Output: $D/{rank,test}/<repo>.cmlx. Before this: ./prepare-mods.sh $D/sets/rank.txt (and test.txt) fills the overlay and the module cache.
# usage: launch-dataset.sh <workers> [perFile=10] [maxFiles=120]
set -euo pipefail
N=${1:-8}; PER_FILE=${2:-10}; MAX_FILES=${3:-120}
D=${D:-/root/work/ml-data/go/psi}; HERE=/root/work/go-psi
LM=/root/work/ml-data/go/models/e14-b.cml
OVERLAY=$HERE/overlay
mkdir -p "$D/rank" "$D/test" "$D/sets/parts" "$HERE/logs"
for fold in rank test; do split -n l/$N -d -a 1 "$D/sets/$fold.txt" "$D/sets/parts/$fold-"; done
COMMON="-Dml.lm=$LM -Dml.data=/root/work/ml-data/go -Dml.perFile=$PER_FILE -Dml.maxFiles=$MAX_FILES -Dml.names=true -Dml.overlay=$OVERLAY"
for i in $(seq 0 $((N - 1))); do
  systemd-run --unit="ml-export-w$i" --collect -p WorkingDirectory=$HERE -E ML_SYSTEM=fresh -E ML_GOMODCACHE=/root/go/pkg/mod bash -c "
    $HERE/run-export.sh GoMlDatasetExport $i $HERE/logs/export-w$i-rank.log -Dml.repos=$D/sets/parts/rank-$i -Dml.out=$D/rank $COMMON;
    $HERE/run-export.sh GoMlDatasetExport $i $HERE/logs/export-w$i-test.log -Dml.repos=$D/sets/parts/test-$i -Dml.out=$D/test $COMMON"
done
