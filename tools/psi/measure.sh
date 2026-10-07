#!/usr/bin/env bash
# One measured export run (direct JVM) of the repository list in $SET: measure.sh <name> <fresh|shared|copy> <gomodcache> [-Dml.*=...]
# Output shards: out/<name>/, log: logs/<name>.log; started as the systemd unit go-psi-<name>.
set -euo pipefail
NAME=$1; MODE=$2; MC=$3; shift 3
HERE=/root/work/go-psi; SET=${SET:-$HERE/sets/one.txt}
rm -rf "$HERE/out/$NAME"
systemd-run --unit="go-psi-$NAME" --collect -p WorkingDirectory=$HERE -E ML_SYSTEM=$MODE -E ML_GOMODCACHE=$MC bash -c "$HERE/run-export.sh GoMlDatasetExport ${W:-0} $HERE/logs/$NAME.log -Dml.repos=$SET -Dml.lm=/root/work/ml-data/go/models/e14-b.cml -Dml.data=/root/work/ml-data/go -Dml.out=$HERE/out/$NAME -Dml.perFile=10 -Dml.maxFiles=120 -Dml.names=true $*"
