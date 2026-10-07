#!/usr/bin/env bash
# e17: Go ranker on the real plugin completion lists (rank fold) evaluated on the test-fold lists.
set -euo pipefail
D=/root/work/ml-data/go/psi; H=/root/work/nn/psi
export JAVA_OPTS="-Xmx40g"
$H/ml-train-bin/bin/ml-train l1 --lang go --shards $D/rank --test-shards $D/test --out /root/work/ml-data/go/models/e17-rank.cml
