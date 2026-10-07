#!/usr/bin/env bash
# Rebuild all data for the neural + n-gram pipelines. Usage: rebuild-data.sh go|csharp|all [step ...]
# Resumable: every step writes <lang>/rebuild-<step>.done after success and is skipped when it exists
# (delete the .done file to redo a step). Logs: /root/work/ml-data/<lang>/rebuild-<step>.log
# Run under systemd-run / tmux, e.g.:
#   systemd-run --unit=rebuild-go --collect -p WorkingDirectory=/root/work bash -c "/root/work/rebuild-data.sh go > /root/work/rebuild-go.out 2>&1"
set -euo pipefail

REPO=/root/work/idea-ml-completion
ML=/root/work/ml-train-run/bin/ml-train
DATA=/root/work/ml-data
PY=/root/work/nn/.venv/bin/python
TOK=$REPO/tools/nn/tokenizer
CLEAN=$REPO/tools/nn/clean
export JAVA_OPTS="${JAVA_OPTS:--Xmx64g}"
WORKERS="${WORKERS:-24}"            # encode_corpus caps at 24
FOLDS="${FOLDS:-lm test}"           # train.py needs lm (train) + test (eval); add rank only if a rank-fold model is wanted

# step <lang> <name> <check-output> <cmd...>  -- skip if .done exists; log to <lang>/rebuild-<name>.log
step() {
  local lang=$1 name=$2; shift 2
  local dir=$DATA/$lang done_f=$DATA/$lang/rebuild-$name.done
  if [[ -e $done_f ]]; then echo "[$lang] $name: done, skipping"; return 0; fi
  echo "[$lang] $name: start $(date +%T)"
  ( "$@" ) > "$dir/rebuild-$name.log" 2>&1 || { echo "[$lang] $name FAILED, see $dir/rebuild-$name.log" >&2; return 1; }
  touch "$done_f"
  echo "[$lang] $name: ok $(date +%T)"
}

preflight() {
  local lang=$1 cat=$2
  [[ -x $ML ]] || { echo "missing $ML" >&2; exit 1; }
  [[ -e $DATA/$lang/repos ]] || { echo "missing $DATA/$lang/repos" >&2; exit 1; }
  [[ -s $DATA/catalog/$cat ]] || { echo "missing catalogue $DATA/catalog/$cat" >&2; exit 1; }
  [[ -n "$(ls -A "$DATA/$lang/repos/" | head -1)" ]] || { echo "$lang/repos is empty" >&2; exit 1; }
  mkdir -p "$DATA/$lang" "$DATA/tokenizer"
}

# --- common steps -----------------------------------------------------------------------------
do_prepare() {  # $1 lang, $2 catalogue file name   (Go ~12 min, C# ~26 min on 32 cores)
  step "$1" prepare "$ML" prepare --lang "$1" --data "$DATA/$1" --catalog "$DATA/catalog/$2" --test 300
}
do_shard() {    # lexer token shards for the n-gram path (Go ~15 min / 23 GB, C# ~8 min / 17 GB)
  step "$1" shard "$ML" shard --lang "$1" --data "$DATA/$1"
}
do_bpe() {      # $1 lang, $2 prefix (go|cs)  -- ~5 min, 1.5 GB stratified sample from the lm fold of the manifest
  local lang=$1 prefix=$2
  if [[ -s $DATA/tokenizer/$prefix-16384.bpe ]]; then echo "[$lang] bpe: vocab exists"; touch "$DATA/$lang/rebuild-bpe.done"; return 0; fi
  step "$lang" bpe bash -c "cd $TOK && $PY train_bpe.py --lang $lang --prefix $prefix --vocab-sizes 16384 --workers 24 --out-dir $DATA/tokenizer"
}
do_encode() {   # secret-scrubbed uint16 shards <lang>/bpe16k/<fold>.* ; lm ~25 min, test ~1 min (path-token caches are built by train.py itself)
  local lang=$1 prefix=$2 fold
  for fold in $FOLDS; do
    if [[ -s $DATA/$lang/bpe16k/$fold.meta.json ]]; then echo "[$lang] encode-$fold: done, skipping"; continue; fi
    step "$lang" "encode-$fold" bash -c "cd $CLEAN && $PY encode_corpus_clean.py --vocab $DATA/tokenizer/$prefix-16384.bpe --lang $lang --fold $fold --out-dir $DATA/$lang/bpe16k --workers $WORKERS"
  done
}

go() {
  preflight go go-20.jsonl
  do_prepare go go-20.jsonl
  do_shard go
  do_bpe go go
  do_encode go go
}

csharp() {
  preflight csharp csharp-20.jsonl
  # the published C# vocabulary (sha256 bc9b48da... = tokenizerSha256 of cs-nn-31m-e1.cml) is kept in the repo; reuse it so the
  # existing model stays compatible. Delete tokenizer/cs-16384.bpe to retrain instead (do_bpe: --lang csharp --prefix cs).
  [[ -s $DATA/tokenizer/cs-16384.bpe ]] || cp "$REPO/models/cs-16384.bpe" "$DATA/tokenizer/cs-16384.bpe"
  do_prepare csharp csharp-20.jsonl
  do_shard csharp
  do_bpe csharp cs
  do_encode csharp cs
}

targets=${1:-all}
case $targets in
  go) go ;;
  csharp) csharp ;;
  all) go; csharp ;;
  *) echo "usage: $0 go|csharp|all" >&2; exit 2 ;;
esac
echo "all requested steps finished"
