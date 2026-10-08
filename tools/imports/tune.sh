#!/bin/bash
# e20 helper: evaluate several packed variants of one language in parallel on a held-out fold (rank fold = tuning).
#   tune.sh csharp rank 300 "decl excess both" [extra eval args]
# Reads ~/work/ml-data/<lang>/imports/<lang-short>-<variant>.cml, writes tune-<variant>.log next to it.
set -e
lang=$1; fold=$2; repos=$3; variants=$4; shift 4
P=${PYTHON:-$HOME/work/nn/.venv/bin/python}
D=$HOME/work/ml-data/$lang/imports
here=$(cd "$(dirname "$0")" && pwd)
short=go; decl=""
if [ "$lang" = csharp ]; then short=cs; decl="--decl-counts $D/counts-lm.pkl,$D/counts-rank.pkl"; fi
for v in $variants; do
  $P -I "$here/eval_imports.py" --lang "$lang" --model "$D/$short-$v.cml" --fold "$fold" --max-repos "$repos" $decl "$@" > "$D/tune-$v.log" 2>&1 &
done
wait
for v in $variants; do echo "== $v"; grep -h "repos\|lambda\|co-imports" "$D/tune-$v.log"; done
