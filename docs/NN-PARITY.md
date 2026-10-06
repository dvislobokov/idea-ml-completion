# Kotlin inference vs PyTorch on the trained Go model (parity, 2026-10-06)

Closes the last unverified link of the neural pipeline: `ml-core`'s Kotlin inference (`core.nn`) + the Kotlin BPE
tokenizer (`core.bpe`) reproduce the PyTorch model on the **real trained weights** (`go-nn-31m-e1.cml`, go31m, step 6484,
6.8 G tokens) through all three kernel paths — scalar Kotlin, native f32, native q8 — with the exact prompt assembly
and stop rules of the inline eval (`~/work/nn/eval/eval_inline.py`).

## Procedure

1. **Fixture** — `~/work/nn/eval/make_parity.py` (Python, GPU, ~2 min) re-derives the positions and prompts of the
   eval run `eval-inline-step6484.json` (same seed/stride, asserted against the JSON records) and writes
   `~/work/ml-data/go/nn/parity/` (not in the repo; env `CML_NN_PARITY` overrides the location):
   - `prompts.bin` — 32 prompts (16 plain, 16 SPM, prompt lengths spread over 189…2000 tokens): path / prefix /
     suffix bytes, the Python ids, float32 logits of the last position and the greedy 48-token continuation with
     per-token log-probs from **two** references: the float model and the int8 *fake-quantised* model
     (`export.fake_quantize_`, i.e. exactly the weights in the `.cml`). The model runs in true fp32 (no autocast, TF32
     off, batch 1, KV cache).
   - `behav.bin` — 500 positions × {plain, spm}: prompt ids, true rest of the line, the eval's generated line (the
     bf16 batched run, from the JSON) and a fresh fp32 fake-quantised greedy run with the eval's stop rule.
   - `meta.json` — provenance (checkpoint, vocab sha256, position indices).
2. **Kotlin** — `ml-core/src/test/.../nn/NnParity.kt` (test scope). `InlinePrompt` rebuilds the prompts
   (`<|file_sep|> path\n prefix` and SPM `<|fim_prefix|><|fim_suffix|> suffix <|fim_middle|> header prefix`, budgets
   ctx 2000 / prefix ≤1450 / suffix ≤512) and the stop set (every token starting with LF/CR + all specials).
   Per kernel path: prefill → logits at the last prompt position (max |Δlogit|, argmax); teacher-forced pass over the
   reference continuation (first divergence index = first step where Kotlin's argmax differs, log-prob RMSE over the
   48 steps); then free-running greedy generation of the 1000 behavioural prompts with `maxNew = 48` and the stop rule,
   written to `kotlin-<kernels>-t<threads>.tsv`. `NnParityTest` runs the same on a 40-position subset (skipped without
   the fixture); the full run is `NnParity.main` (`--threads 8 --kernels scalar,native-f32,native-q8`).
3. **Scoring** — `~/work/nn/eval/score_parity.py` applies the eval's own `code_part` line-exact metric to the Kotlin
   generations next to the Python numbers on the same subset.

Checks done on the way: the vocab file `go-16384.bpe` sha256 equals the model's `tokenizerSha256` meta
(`d5f7a21c…`); `NnConfig` read from the file equals the training config.

## Tokenizer parity in the real flow

32/32 prompts: Kotlin `InlinePrompt` ids are identical to the Python prompts (header, 40 KB prefix cut, suffix
truncation to 512 tokens, SPM special-token order). No divergence in special-token assembly.

## Logit parity (32 prompts, 8 threads, EPYC 9554 AVX-512 VNNI)

| kernels | reference | max \|Δlogit\| (mean over prompts) | argmax at prompt end | greedy-48 identical | mean first divergence | log-prob RMSE / max |
|---|---|---|---|---|---|---|
| scalar Kotlin | fp32 fake-quantised (same int8 weights) | **0.0001** (0.0000) | 32/32 | 32/32 | 48.0 | 0.00000 / 0.0000 |
| native f32 | fp32 fake-quantised | **0.0001** (0.0000) | 32/32 | 32/32 | 48.0 | 0.00000 / 0.0000 |
| native q8 | fp32 fake-quantised | 0.36 (0.28) | 32/32 | 26/32 | 42.6 (min 3) | 0.0215 / 0.27 |
| scalar / native f32 | float model | 0.39 (0.26) | 32/32 | 29/32 | 45.5 (min 4) | 0.0119 / 0.13 |
| native q8 | float model | 0.62 (0.40) | 32/32 | 26/32 | 43.3 (min 3) | 0.0237 / 0.29 |

Reading: scalar Kotlin and native f32 are *bit-for-bit equivalent to PyTorch* up to summation order (1e-4 on logits of
magnitude ~10–20). The int8 weight quantisation itself costs 0.26 logits on average (float row vs fq row; identical
argmax, 3/32 greedy runs diverge late — at steps 4, 25 and 27 — on near-ties). Q8 activation quantisation adds about
as much again (0.28 vs the fq model, log-prob RMSE 0.02); 6/32 free-running continuations of 48 tokens diverge.
Divergences are not errors (no systematic direction, argmax always agrees at the prompt end); they are near-ties that
any ±0.3-logit perturbation flips.

## Behavioural parity (500 positions × plain/spm, greedy to end of line, eval stop rules)

| run | same line as fp32-fq reference | same line as the eval (bf16, batched) | line exact vs truth | plain / spm |
|---|---|---|---|---|
| Python eval (bf16, batched, from the JSON) | 96.0 % | 100 % | 54.4 % | 53.4 / 55.4 % |
| Python fp32 fake-quantised (reference) | 100 % | 96.0 % | 54.6 % | 54.0 / 55.2 % |
| **Kotlin scalar** | **100.0 %** (1000/1000) | 96.0 % | **54.6 %** | 54.0 / 55.2 % |
| **Kotlin native f32** | **100.0 %** (1000/1000) | 96.0 % | **54.6 %** | 54.0 / 55.2 % |
| **Kotlin native q8** | 95.7 % (957/1000) | 93.4 % | **54.7 %** | 53.8 / 55.6 % |

(Full eval on 3000 positions: plain 53.2 %, spm 55.7 %; the 500-position subset is ±2 pp of that.)

- Scalar and native f32 reproduce the fp32 reference line **exactly in every one of the 1000 positions**, including the
  stop kind (newline / special / 48-token limit) and the first token.
- The 4 % difference between the reference and the eval's own numbers is bf16 autocast + left-padded batching on the
  GPU side, not Kotlin: Kotlin agrees with the eval exactly as often as fp32 PyTorch does (96.0 %).
- Q8: 43/1000 lines differ from the reference (5 on the first token, 36 later in the line, 2 on the stop kind) — all
  near-ties on long literal/identifier continuations; line-exact vs truth is unchanged (547 vs 546 of 1000: among the
  differing lines Kotlin-q8 is right 4 times, the reference 3). No systematic divergence found: newline-with-indent
  tokens, `<|endoftext|>` stops and the 48-token limit behave identically; token healing is not involved (the eval puts
  the cursor at a pre-token boundary, and the Kotlin prompt is the same ids).
- Kotlin's simplified line-exact (trailing `//` comment stripped, right-trimmed) gives the same 546/1000 as the Python
  `code_part` scoring on this subset.

## Latency on the real weights

`NnBench --file go-nn-31m-e1.cml --prompt-file parity/prompts.bin --prompt 1500 --gen 20`, `taskset -c 24-31`,
JDK 21, medians of 7 after warm-up; the box was shared (load ≈ 12: a C# BPE encode job on 24 workers), so absolute
numbers are pessimistic by maybe 10–20 %, relative ones hold. Prompt = the first 1500 tokens of a real eval prompt.

| kernels | threads | prefill 1500 tok, ms | decode, ms/token | 20-token line, ms | reuse8 line (8 new prompt tokens + 20), ms | first line (JIT cold), ms |
|---|---|---|---|---|---|---|
| scalar Kotlin | 1 | 3007 | 6.78 | 3136 | 155 | 3507 |
| scalar Kotlin | 4 | 860 | 2.64 | 910 | 64 | 1459 |
| scalar Kotlin | 8 | 492 | 1.98 | 530 | 47 | 1443 |
| native f32 (AVX-512) | 1 | 1229 | 4.82 | 1321 | 106 | 1411 |
| native f32 | 4 | 372 | 2.21 | 415 | 49 | 572 |
| native f32 | 8 | 211 | 1.50 | 240 | 33 | 441 |
| native q8 (AVX-512 VNNI) | 1 | 782 | 3.29 | 845 | 71 | 952 |
| native q8 | 4 | 265 | 1.47 | 293 | 32 | 433 |
| native q8 | 8 | **147** | **1.06** | **167** | **24** | 371 |

Memory and load (31.2 MB file): `NnFormat.read(File)` mmap parse 18–23 ms (page-cache warm; touching every weight page
3–4 ms), stream read into a direct buffer 40–76 ms, q8 weight repacking (`prepare`) 39–45 ms once per model. RSS of the
bench JVM 314–385 MB (base JVM 60–66 MB, Java heap in use 173–249 MB with `-Xmx2g`: KV cache + activation buffers +
the bench's own copies; the mapped weights are 31 MB of page cache, the q8 packed copy another ~31 MB native).
JIT warm-up: the first line costs 1.4 s scalar at 4–8 threads (0.37–0.57 s native) — a background warm-up prefill at
plugin start is worth it; steady state is reached by the second line.

## Recommendation: default kernel mode

- **q8 by default where the native library loads** (`completionml.nn.native.mode=q8`, the current default): on the real
  model its activation quantisation changes 4 % of generated lines but **not the accuracy** (line exact 54.7 % vs 54.6 %,
  argmax 32/32, log-prob RMSE 0.02 — far below the 0.26-logit noise the int8 weights already carry), and it is 1.4× faster
  than native f32 on prefill and decode at 8 threads (1.6× at 1 thread), 3.3× faster than scalar Kotlin. For a 1500-token
  cold prompt that is 147 vs 211 ms, and 24 vs 33 ms for an incremental line — both well inside the budget
  (line ≤150 ms, cold prompt ≤1 s); scalar Kotlin at 8 threads also meets the budget on this CPU (492 / 47 ms), so the
  fallback is usable, not just safe.
- **f32 when exactness matters**: ranker features (candidate log-probs) and any A/B against PyTorch numbers should use
  native f32 or scalar — they are bit-equivalent to the reference. Keep `mode=f32` as the documented switch.
- Caveats before shipping q8: the AVX2 q8 kernel (`vpmaddubsw` sign trick) and NEON `sdot` were not exercised here (this
  box selects AVX-512 VNNI); the parity run should be repeated with `-Dcompletionml.nn.native.isa=1` (AVX2) and on an
  Apple Silicon machine — the fixture and `NnParity` make that a 10-minute job. The user-visible metric to watch is the
  "same line as reference" rate on 1000 positions (expect ≥ 95 %) and line-exact within ±0.5 pp.

## Bugs found

None in the inference or tokenizer: the first full run of every path matched the reference. Two cosmetic issues in the
new test code itself (per-mode label order in the `NnParity` report, fixture availability check before `meta.json` is
written) were fixed during the run.

## Files

- `ml-core/src/test/kotlin/io/github/completionml/core/nn/NnParity.kt` — `InlinePrompt`, fixture reader, parity run (main).
- `ml-core/src/test/kotlin/io/github/completionml/core/nn/NnParityTest.kt` — JUnit test (skipped without the fixture).
- `ml-core/src/test/kotlin/io/github/completionml/core/nn/NnBench.kt` — `--file`, `--prompt-file`, load/prepare timings.
- `~/work/nn/eval/make_parity.py`, `~/work/nn/eval/score_parity.py` — fixture writer and scorer (outside the repo).
- `~/work/ml-data/go/nn/parity/` — fixture (8.4 MB) and the Kotlin `kotlin-*.tsv` outputs.
