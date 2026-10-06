# Neural code model — training pipeline (PyTorch, GPU only)

Our own small decoder-only LM for Go completion. Everything here is training-side tooling; the plugins run the exported
`.cml` with Kotlin code in `ml-core` (`io.github.completionml.core.nn`). No third-party pretrained weights anywhere.

Files

| file | purpose |
|---|---|
| `model.py` | `ModelConfig` (+ presets, exact parameter counts) and `CodeLM` (LLaMA-style: RMSNorm pre-norm, RoPE rotate-half, GQA, SwiGLU, tied embeddings, no biases). `python model.py` prints the preset table. |
| `data.py` | memory-mapped shard reader, repo down-weighting, repo-grouped packing to `seq_len+1` windows, document-level FIM (PSM/SPM), fixed held-out eval windows, exact resume state, background prefetcher. `python data.py --show 1` dumps a window and the repo-share statistics. |
| `train.py` | AdamW (0.9, 0.95), wd 0.1 on matrices only, cosine LR with warmup, bf16 autocast, grad clip 1.0, grad accumulation, optional `torch.compile`, periodic eval (plain + FIM ppl on fixed test windows), checkpoints with resume, JSONL metrics. |
| `export.py` | checkpoint -> neural `.cml` ("CMLN" v1, docs/NN-FORMAT.md in the repo): int8 per-output-row quantisation of all matrices incl. `tok_emb`, f32 norms, config in the meta block. `--check N` measures the int8 loss delta. |
| `sample.py` | greedy continuations (plain and FIM) from a checkpoint on held-out test files. |

Inputs: shards from `../tokenizer/encode_corpus.py` in `~/work/ml-data/go/bpe16k/` (`lm`, `test`, `rank` folds),
vocabulary `~/work/ml-data/tokenizer/go-16384.bpe`. Python: `~/work/nn/.venv/bin/python`.

## Presets (vocab 16 384, head_dim 64)

| preset | d | layers | heads/kv | ffn | params | non-embedding | int8 file |
|---|---|---|---|---|---|---|---|
| go19m | 384 | 8 | 6/2 | 1024 | 18.9 M | 12.6 M | 19 MB |
| go31m | 512 | 8 | 8/2 | 1408 | 30.9 M | 22.6 M | 31 MB |
| go50m | 640 | 10 | 10/2 | 1536 | 49.8 M | 39.3 M | 50 MB |
| go102m | 768 | 12 | 12/4 | 2560 | 102.3 M | 89.7 M | 103 MB |

## Data pipeline decisions

* **Repo down-weighting.** File of repo *r* with *T_r* tokens is kept in an epoch with probability
  `p_r = min(1, sqrt(T_min / T_r))`, `T_min` = 2 M tokens (`--t-min`). Expected tokens per epoch: `T_r` for normal repos,
  `sqrt(T_min * T_r)` for big ones, so a 200 M-token SDK repo (100x the knee) contributes 10x a 2 M repo, not 100x.
  Different epochs sample different subsets of the big repos. `python data.py` prints the resulting shares.
* **Repo-grouped packing.** Kept files are grouped by repo and cut into groups of <= `2*seq_len` tokens, so a 2048 window
  mostly contains a few files of one repository (cross-file context like the plugin will provide). Groups are shuffled
  globally; the document queue is packed into windows of 2049 tokens, no padding, no cross-document masking, loss on
  every token. Plain documents may be split across windows; **a FIM document never is** (document-aware packer, since
  the step-4000 data fix): if it does not fit into the rest of the window it is placed with a shortened prefix when
  >= 384 tokens are left (and >= 32 prefix tokens survive), otherwise it is deferred to the start of the next window and
  the gap is filled with the following (plain) documents. Before the fix, windows were cut at arbitrary points of a flat
  stream and 74 % of tokens are in files > 2048 tokens, so most PSM middles had lost their prefix — the model learned
  to ignore PSM prompts (inline eval at step 2500: PSM line-exact 2 %, SPM/plain 50 %).
* **Document format.** `<|file_sep|> path\n <tokens> <|endoftext|>`. The relative path (`pkg/foo/bar_test.go`) is cheap
  context the plugin always has (`--no-path` disables it). The GitHub repo name is deliberately *not* included.
* **FIM** (rate 0.5, per document; the path header stays with the prefix):
  * PSM: `<|fim_prefix|> <|file_sep|> path\n pre <|fim_suffix|> suf <|fim_middle|> mid <|endoftext|>`
  * SPM (Code Llama variant, 50 % of FIM docs): `<|fim_prefix|> <|fim_suffix|> suf <|fim_middle|> <|file_sep|> path\n pre mid <|endoftext|>`
  Bounds (so the whole document fits one window, total <= `seq_len - 8`): middle <= 256 tokens, suffix = first <= 512
  tokens after the middle, prefix = the last tokens before the middle that still fit (up to ~1 250–1 500). Cut placement:
  50 % token-level (start uniform, middle length U[0,32] or U[0,256] half/half); 50 % line-aligned — the middle ends at a
  line end (before the newline token / EOF, as the inference suffix does) and spans k lines (k = 1 half of the time, else
  1 + Geom(0.35), <= 8); it starts at the line start (after indentation) or at a uniform mid-line token, half/half.
  Inference uses PSM with the same bounds (eval: prefix <= 1450, suffix <= 512, prompt + 48 new tokens <= 2048); the model
  stops the middle with `<|endoftext|>`. Token share of FIM documents is ~27 % (FIM documents are bounded, plain files are
  not). `test_data.py` checks no-straddle, PSM reassembly, line alignment, exact resume and the FIM share.
* **Eval.** `--eval-windows N` fixed windows from the `test` fold (300 held-out repos), once plain (FIM rate 0) and once
  with every document FIM-transformed; reported as `eval_ppl` / `eval_fim_ppl`.
* **I/O.** Tokens are read with `os.pread` (a page fault on the 13.7 GB memmap cost ~20 ms on this kernel, pread 0.2 ms);
  the BPE ids of all file paths are cached once in `<fold>.pathtok.u16/.pathoff.u64` (encoding paths on the fly was the
  bottleneck). The loader delivers ~16 M tokens/s single-threaded; `train.py` also pre-reads the token shard into the page
  cache (`--no-warm-cache` to skip).
* **Determinism / resume.** Epoch composition and FIM draws come from `numpy.default_rng([seed, epoch, ...])`; the stream
  state (epoch, group cursor, pending document queue as plain numpy tuples) is saved in every checkpoint and restored
  exactly; checkpoints of the old flat packer (`carry` array) still load.

## Usage

```bash
cd ~/work/nn/train
PY=~/work/nn/.venv/bin/python

# 1. shards (once): lm, test, rank folds, 20 workers (~25 min for lm on a shared CPU)
cd ~/work/nn/tokenizer
$PY -I -c 'import sys; sys.path.insert(0, "/root/work/nn/tokenizer"); sys.argv = sys.argv[1:]; import runpy; runpy.run_path(sys.argv[0], run_name="__main__")' \
   encode_corpus.py --vocab ~/work/ml-data/tokenizer/go-16384.bpe --lang go --fold lm --out-dir ~/work/ml-data/go/bpe16k --workers 20

# 2. smoke run (15 minutes, go31m)
systemd-run --unit=nn-smoke-go30 --collect -p WorkingDirectory=$HOME/work/nn/train bash -c \
  "$PY train.py --preset go31m --run smoke-go30 --micro-batch 32 --tokens-per-step 524288 --warmup 100 \
   --max-minutes 15 --eval-every 200 --ckpt-every 200 --eval-windows 64 --compile > ~/work/ml-data/go/nn/smoke.log 2>&1"

# 3. full run (one epoch ~ see report), resumes automatically if ckpt-latest.pt exists
systemd-run --unit=nn-go31m --collect -p WorkingDirectory=$HOME/work/nn/train bash -c \
  "$PY train.py --preset go31m --run go31m-e1 --micro-batch 32 --tokens-per-step 1048576 --lr 1e-3 --warmup 500 \
   --max-tokens 6.8e9 --eval-every 500 --ckpt-every 500 --keep-every 5000 --compile > ~/work/ml-data/go/nn/go31m-e1.log 2>&1"

# 4. export + int8 check, then sanity samples
$PY export.py --ckpt ~/work/ml-data/go/nn/go31m-e1/ckpt-latest.pt --out ~/work/ml-data/go/models/go-nn-31m.cml --check 64
$PY sample.py --ckpt ~/work/ml-data/go/nn/go31m-e1/ckpt-latest.pt --snippets 3
```

`metrics.jsonl` rows: `step, tokens, loss, lr, grad_norm, step_time, tok_s, gpu_mem_gb, gpu_reserved_gb, epoch, elapsed`
and eval rows `eval_loss, eval_ppl, eval_fim_loss, eval_fim_ppl`.

## Export format notes (keep in sync with ml-core `NnFormat`)

2-D tensors are written `[in, out]` (= PyTorch `weight.T`) int8 with one f32 scale per output column
(`scale = max|row| / 127`, symmetric, no zero point); `tok_emb` is `[dModel, vocab]`, so each token has its own scale.
Norms are f32 vectors. Meta keys: `kind=nn`, the `NnConfig` keys (`vocabSize, dModel, nLayers, nHeads, nKvHeads, ffnDim,
maxContext, ropeTheta, normEps, tiedEmbeddings, activation, norm`) plus `language, tokenizer, tokenizerSha256, preset,
trainStep, trainTokens, docFormat, pathHeader`. All byte packing lives in `export.write_cmln()`.
