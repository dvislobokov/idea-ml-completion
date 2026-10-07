# Sequence-level distillation data, C# (tools/nn/distill/distill.py)

- positions (not stored: 182 MB; reproducible with `distill.py sample --seed 1`) — 300 000 FIM positions sampled in token space from the lm fold shards (file index `fi`, token cursor `a`, line end `b`, decoded prefix tail / suffix head, the author's rest of the line).
- `teacher-1.5b.jsonl.gz` — 245 844 rest-of-line completions by Qwen2.5-Coder-1.5B (vLLM, greedy, PSM prompt with path header, plain healing; special tokens may appear as text in this first run — `encode` cuts at `<|`). Teacher == author's line in ~54 % of the kept samples.
Use: `distill.py encode --vocab cs-16384.bpe --in teacher-1.5b.jsonl --out teacher-1.5b.npz`, then `train.py --teacher teacher-1.5b.npz --teacher-rate 1.0`.
