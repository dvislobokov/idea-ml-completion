# Neural model file format (`.cml`, kind `nn`, version 1)

Written/read by `ml-core` `io.github.completionml.core.nn.NnFormat`. This document is the contract for the Python
exporter (`~/work/nn`). Unlike the n-gram/ranker `.cml` files (gzip, `ModelFormat`), the neural file is
**uncompressed** so the runtime can memory-map it and keep the int8 weights off the Java heap.

## Byte layout

All integers and floats are **little-endian**. Offsets are absolute from the start of the file.

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `CMLN` (ASCII) |
| 4 | 4 | `u32` format version = `1` |
| 8 | 4 | `u32` M = meta length in bytes |
| 12 | M | meta: UTF-8 text, `key=value\n` lines (no `=` in keys, no newlines in values) |
| 12+M | 4 | `u32` T = tensor count |
| … | … | T directory entries (below), packed, no padding |
| … | … | zero padding up to the first 64-byte boundary, then tensor data |

Directory entry:

| size | field |
|---|---|
| 2 | `u16` name length N |
| N | name, UTF-8 |
| 1 | `u8` dtype: `0` = float32, `1` = int8 with per-column float32 scales |
| 1 | `u8` rank R: `1` for float32 vectors, `2` for int8 matrices |
| 4·R | `u32` dims |
| 8 | `u64` data offset (multiple of 64) |
| 8 | `u64` scales offset (multiple of 64; `0` for float32 tensors) |

Data: every array (tensor data, scales) starts at a 64-byte aligned offset; gaps are zero-filled; the file length is a
multiple of 64. float32 vector: `dims[0]` floats. int8 matrix `[rows, cols]`: `rows*cols` signed bytes, row-major,
followed (at its own aligned offset) by `cols` float32 scales. The reader tolerates any order of tensors and any
extra tensors; the writer emits them in the order of `NnFormat.expectedShapes`.

## Meta keys

Required: `kind=nn`, and the config: `vocabSize`, `dModel`, `nLayers`, `nHeads`, `nKvHeads`, `ffnDim`, `maxContext`,
`ropeTheta` (float), `normEps` (float), `tiedEmbeddings` (`true`/`false`), `activation=swiglu`, `norm=rmsnorm`.
Free-form (recommended): `language` (`go`/`csharp`), `corpusId`, `createdAt` (epoch ms), `tokenizer` (BPE vocabulary
id/hash — the runtime must refuse a tokenizer mismatch), `trainStep`, `valLoss`.

## Quantisation

Every 2-D weight is stored as **`[in, out]` = PyTorch `weight.T`** (nn.Linear keeps `[out, in]`), so that `y = x · W`.
Symmetric int8, one scale per **output column** (= per row of the PyTorch weight = per output channel):

```
scale[o] = max_i |W[i, o]| / 127          (0 for an all-zero column)
q[i, o]  = clamp(round(W[i, o] / scale[o]), -127, 127)      # -128 is never used
W[i, o] ≈ q[i, o] * scale[o]
```

Python sketch:

```python
w = linear.weight.detach().float().T.contiguous()          # [in, out]
scale = w.abs().amax(dim=0) / 127
q = torch.where(scale > 0, (w / scale.clamp_min(1e-30)).round(), 0).clamp(-127, 127).to(torch.int8)
```

Norm weights are float32 vectors.

## Tensors

`hd = dModel / nHeads`, `kvDim = nKvHeads * hd`. Layer tensors are prefixed `layers.<i>.` (i from 0).

| name | dtype | shape | PyTorch origin (LLaMA-style naming) |
|---|---|---|---|
| `tok_emb` | int8 | `[dModel, vocabSize]` | `tok_embeddings.weight.T` (embedding of token v = column v; scale per token) |
| `layers.i.attn_norm` | f32 | `[dModel]` | `attention_norm.weight` |
| `layers.i.wq` | int8 | `[dModel, dModel]` | `attention.wq.weight.T` |
| `layers.i.wk` | int8 | `[dModel, kvDim]` | `attention.wk.weight.T` |
| `layers.i.wv` | int8 | `[dModel, kvDim]` | `attention.wv.weight.T` |
| `layers.i.wo` | int8 | `[dModel, dModel]` | `attention.wo.weight.T` |
| `layers.i.ffn_norm` | f32 | `[dModel]` | `ffn_norm.weight` |
| `layers.i.w1` | int8 | `[dModel, ffnDim]` | gate: `feed_forward.w1.weight.T` |
| `layers.i.w3` | int8 | `[dModel, ffnDim]` | up: `feed_forward.w3.weight.T` |
| `layers.i.w2` | int8 | `[ffnDim, dModel]` | down: `feed_forward.w2.weight.T` |
| `final_norm` | f32 | `[dModel]` | `norm.weight` |
| `lm_head` | int8 | `[dModel, vocabSize]` | `output.weight.T` — only when `tiedEmbeddings=false` |

With tied embeddings the logits are `final_norm(x) · tok_emb` (the same int8 tensor serves lookup and output).

## Model semantics the trainer must match

- Pre-norm blocks, no biases anywhere:
  `h = x + Wo·Attn(RoPE(Wq·n), RoPE(Wk·n), Wv·n)` with `n = RMSNorm(x)`, then `x' = h + W2·(SiLU(W1·m) ⊙ W3·m)`
  with `m = RMSNorm(h)`; logits from `RMSNorm_final(x)`.
- RMSNorm: `y = x / sqrt(mean(x²) + normEps) * weight`.
- RoPE: **"rotate half"** pairing (GPT-NeoX / HF LLaMA): for each head, dims `i` and `i + hd/2` are rotated by angle
  `pos * ropeTheta^(-2i/hd)`, `i < hd/2`, positions from 0. (Meta's original LLaMA code pairs interleaved dims
  `2i, 2i+1` instead — if the trainer uses that, permute wq/wk columns at export.)
- Attention: causal, scale `1/sqrt(hd)`, GQA: query head `h` uses KV head `h / (nHeads / nKvHeads)`.
- No dropout/bias at inference; no attention sinks; no sliding window.
