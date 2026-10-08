#!/bin/bash
# e20 helper: pack several pruning variants in parallel.  pack-variants.sh <lang> "<name>:<packer args>" ...
set -e
lang=$1; shift
P=${PYTHON:-$HOME/work/nn/.venv/bin/python}
D=$HOME/work/ml-data/$lang/imports
here=$(cd "$(dirname "$0")" && pwd)
short=go; [ "$lang" = csharp ] && short=cs
for spec in "$@"; do
  name=${spec%%:*}; args=${spec#*:}
  ( $P -I "$here/pack_imports.py" --lang "$lang" --counts "$D/counts-lm.pkl" --out "$D/$short-$name.cml" $args | sed "s/^/$name /" ) &
done
wait
