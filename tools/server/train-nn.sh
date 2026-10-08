#!/usr/bin/env bash
# Train one of our transformers end to end on this server: DDP on both GPUs with the lr2e3 recipe (the one every published
# model used), then export to .cml and run the two standard evaluations (3000 positions of the test fold, 2000 positions of
# the fresh repositories). Everything lands in ~/work/ml-data/<lang>/nn/<run>/ and the model in ~/work/ml-data/<lang>/models/.
#
# usage: tools/server/train-nn.sh <go|csharp> <preset> [run-name] [extra train.py args...]
#   presets (tools/nn/train/model.py): go31m (30.9 M), go50m (49.8 M), go102m (102.3 M); the same presets serve C#.
#   e.g.  tools/server/train-nn.sh go go102m go102m-e6
#         tools/server/train-nn.sh csharp go102m cs102m-e6 --max-tokens 5.65e9
# env: GPUS (default "0,1"; one index = single GPU without DDP), MAX_TOKENS (default = one epoch: Go 6.85e9, C# 5.65e9),
#      MICRO_BATCH (default 32; 102 M fits on an H200 at 32), SKIP_EVAL=1 to stop after the export.
# Run it detached so it survives the SSH session:
#   systemd-run --unit=go102m --collect -p WorkingDirectory=$PWD bash -c "tools/server/train-nn.sh go go102m go102m-e6 > ~/work/ml-data/go/nn/go102m-e6.queue.out 2>&1"
#   journalctl -u go102m -f   /   tail -f ~/work/ml-data/go/nn/go102m-e6.log
# Expected on 2 x H200 (DDP): 31 M ~50 min / epoch, 50 M ~65-90 min, 102 M ~2.5-3 h (bf16, torch.compile); export + evals ~15 min.
set -euo pipefail
lang="${1:?go|csharp}"; preset="${2:?preset}"; run="${3:-$preset-$(date +%m%d)}"; shift 3 2>/dev/null || shift $#
root="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/root/work/nn/.venv/bin/python; T="$root/tools/nn"; D=/root/work/ml-data/$lang; S=$D/nn
mkdir -p "$S" "$D/models"
case "$lang" in
  go)     vocab=/root/work/ml-data/tokenizer/go-16384.bpe; epoch=6.85e9 ;;
  csharp) vocab=/root/work/ml-data/tokenizer/cs-16384.bpe; epoch=5.65e9 ;;
  *) echo "lang must be go or csharp"; exit 2 ;;
esac
gpus="${GPUS:-0,1}"; tokens="${MAX_TOKENS:-$epoch}"
mb="${MICRO_BATCH:-}"; [ -n "$mb" ] || { mb=32; }
echo "start $(date +%F' '%T) lang=$lang preset=$preset run=$run gpus=$gpus tokens=$tokens micro-batch=$mb extra: $*"
cd "$T/train"
if [[ "$gpus" == *,* ]]; then
  launcher="$PY -m torch.distributed.run --nproc_per_node $(tr ',' '\n' <<<"$gpus" | wc -l) train.py"
else
  launcher="$PY train.py"
fi
CUDA_VISIBLE_DEVICES=$gpus $launcher --run "$run" --preset "$preset" --data "$D/bpe16k" --vocab "$vocab" --out "$S" \
  --fim-rate 0.7 --spm-rate 0.5 --max-tokens "$tokens" --micro-batch "$mb" --tokens-per-step 524288 --lr 2e-3 --warmup 500 \
  --compile --eval-every 250 --eval-windows 128 --ckpt-every 500 --keep-every 2000 --seed 1 "$@" > "$S/$run.log" 2>&1
grep -q 'final eval' "$S/$run.log" || { echo "no final eval in $S/$run.log"; tail -5 "$S/$run.log"; exit 1; }
echo "export $(date +%T)"
g0="${gpus%%,*}"
CUDA_VISIBLE_DEVICES=$g0 $PY -I export.py --ckpt "$S/$run/ckpt-latest.pt" --out "$D/models/$run.cml" --lang "$lang" \
  --vocab "$vocab" --data "$D/bpe16k" --check 128
ls -la "$D/models/$run.cml"
[ "${SKIP_EVAL:-0}" = 1 ] && { echo "done (no eval) $(date +%T)"; exit 0; }
cd "$T/eval"
E="$PY -I eval_inline.py --lang $lang --ckpt $S/$run/ckpt-latest.pt --seed 1"
CUDA_VISIBLE_DEVICES=$g0 $E --positions 3000 --modes plain,spm --dump 60 --scratch "$S/eval-tmp-$run-std" --out "$S/eval-$run" &
if [ -f "$D/prepared/manifest-fresh.jsonl" ]; then
  g1="${gpus##*,}"
  CUDA_VISIBLE_DEVICES=$g1 $E --positions 2000 --modes spm --scratch "$S/eval-tmp-$run-fresh" --manifest "$D/prepared/manifest-fresh.jsonl" --out "$S/fresh-$lang-$run"
fi
wait
echo "done $(date +%F' '%T)"; echo "results: $S/eval-$run.md  $S/fresh-$lang-$run.md  model: $D/models/$run.cml"
