#!/usr/bin/env bash
# Small proxy model (go5m) for data/format hypotheses: train on a token budget on one GPU, export, standard 3000-position eval.
# usage: proxy-run.sh <lang go|csharp> <run> <gpu|0,1> <max-tokens> [extra train args...]   (0,1 = both GPUs via DDP)
set -euo pipefail
lang=$1; run=$2; gpu=$3; toks=$4; shift 4
PY=/root/work/nn/.venv/bin/python; T=/root/work/idea-ml-completion/tools/nn; D=/root/work/ml-data/$lang; S=$D/nn
case $lang in go) V=/root/work/ml-data/tokenizer/go-16384.bpe;; csharp) V=/root/work/ml-data/tokenizer/cs-16384.bpe;; esac
export CUDA_VISIBLE_DEVICES=$gpu
cd $T/train
# gpu "0,1" -> both cards with DDP (torchrun), a single index -> one card
if [[ $gpu == *,* ]]; then LAUNCH="$PY -m torch.distributed.run --nproc_per_node 2 train.py"; else LAUNCH="$PY train.py"; fi
$LAUNCH --run $run --preset go5m --data $D/bpe16k --vocab $V --out $S --fim-rate 0.7 --spm-rate 0.5 --max-tokens $toks \
  --micro-batch 32 --tokens-per-step 524288 --lr 2e-3 --warmup 300 --compile --eval-every 250 --eval-windows 128 --ckpt-every 500 \
  --seed 1 --no-resume "$@" > $S/$run.log 2>&1
grep -q 'final eval' $S/$run.log || { echo "no final eval"; exit 1; }
$PY -I export.py --ckpt $S/$run/ckpt-latest.pt --out $D/models/$run.cml --lang $lang --vocab $V --data $D/bpe16k --check 64 > $S/$run.export.log 2>&1
cd $T/eval
$PY -I eval_inline.py --lang $lang --ckpt $S/$run/ckpt-latest.pt --positions 3000 --seed 1 --modes spm --scratch $S/eval-tmp-$run --out $S/eval-$run > $S/eval-$run.log 2>&1
grep -E 'final eval' $S/$run.log; grep -E '^spm:' $S/eval-$run.log
