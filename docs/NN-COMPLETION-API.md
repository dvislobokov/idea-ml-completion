# Neural inline completion — plugin-facing API (`ml-core`, package `io.github.completionml.core.nn`)

What a plugin calls to get a whole-line suggestion from our own transformer (`.cml` neural model, docs/NN-FORMAT.md),
with the behaviour the eval harness measures (`tools/nn/eval/eval_inline.py --heal boundary`): token healing at the
caret, constrained first tokens, SPM prompt, greedy decode to the end of the line, repetition guard, show policy.
Measured effect and numbers: `~/work/ml-data/go/nn/eval-heal.md`, docs/NEURAL-RU.md §7c.

## Objects

| class | role | lifetime |
|---|---|---|
| `BpeTokenizer` (`core.bpe`) | byte-level BPE, twin of `cmlbpe.py`; `encodeBytes`, `decodeBytes`, **`preTokenBoundaries` / `lastPreTokenBoundary`** (pre-token scanner exposed for healing) | one per vocabulary, shared, thread-safe |
| `NnFormat.read(file)` → `NnFormat.Model` | mmap of the `.cml` weights (18–23 ms for 31 MB) | one per model file |
| `NnModel(weights, nThreads, kernels)` | the network; `NnKernels.best()` picks native q8 → scalar | one per decoding thread (its executor is not reentrant) |
| `NnSession` (`model.newSession(capacity)`) | KV cache with prefix reuse: `prefill` recomputes only after the longest common prefix with the previous prompt | one per editor/thread; `close()` frees native memory |
| `VocabPrefixIndex(tok)` | sorted byte-prefix table over the vocabulary (`allowed(remainder)`, `consume`) — built once by `NnCompletion` | one per vocabulary |
| **`NnCompletion(model, tok, options)`** | the entry point: `complete(path, before, after, session): Result` | one per (model, options); immutable |
| `InlinePrompt` | prompt assembly identical to the Python harness (`cutPrefix`, `plain`, `spm`, `stopIds`) | object |

## `NnCompletion.complete(path, before, after, session)`

Inputs (all `ByteArray`, UTF-8):
- `path` — file path relative to the project root (goes into the `<|file_sep|> path\n` header exactly as in training);
- `before` — text before the caret; a tail of ≥ 40 KB is enough (the prompt uses the last 40 KB cut forward to a line start, then ≤ `maxPrefix` tokens);
- `after` — text after the caret: the rest of the current line (needed for healing — `foo(⟨⟩)` is one pre-token `()`) and the following lines (the SPM suffix starts at the end of the current line, ≤ 16 KB / ≤ `suffixTokens` tokens). May be empty.

Steps:
1. **Healing.** `healedBoundary(before, after)` = the last pre-token boundary ≤ caret, scanning the current line from the LF before it (the newline token owns the indentation) to the end of the line. The bytes `before[boundary, end)` are the *typed remainder* (`""` in ~90 % of positions, `(`, `)`, `"`, `()`, `")`, a space, `Hi` of `High`…). The prompt is built from `before[0, boundary)`.
2. **Prompt.** `Options.mode` SPM (default): `<|fim_prefix|><|fim_suffix|> suffix <|fim_middle|> <|file_sep|> path\n prefix` with `ctx` 2000, `maxPrefix` 1450, `suffixTokens` 512; or PLAIN: header + prefix tail. `session.prefill(prompt)` reuses the cached common prefix.
3. **Decode.** Greedy, ≤ `maxNew` (48) tokens. While the typed remainder is pending, only `VocabPrefixIndex.allowed(remainder)` ids may be chosen (tokens starting with the remainder, or tokens that are a proper prefix of it — then the rest of the remainder constrains the next step); their log-probabilities are from the masked softmax. Stops at any token starting with LF/CR or a special token (`Stop.NEWLINE` / `SPECIAL`), at the token limit (`LIMIT`), or when the **repetition guard** fires (`REPEAT`: a BPE n-gram, n ≤ 4, ≥ 4 bytes, three times in a row).
4. **Result.**
   - `text` — the suggestion to insert at the caret (typed remainder stripped), `textString` for convenience;
   - `typed`, `prompt`, `tokens`, `logProbs`, `stop`, `stopLogProb`;
   - `confProd` = Π p(token) × p(newline) (the gate to use), `confMin` = min p;
   - `punctOnly` — no letter/digit/underscore/non-ASCII byte and no quote (`);`, `}`, `)]`);
   - `repeated` — stopped by the guard or the token limit;
   - **`show`** = `confProd ≥ options.showThreshold` ∧ ¬(`suppressPunctOnly` ∧ `punctOnly`) ∧ ¬`repeated` ∧ `text` non-empty;
   - `healMiss` — the model did not reproduce the typed remainder (only possible when the limit cut it; treat as "do not show").

`Options` defaults: `mode = SPM, ctx = 2000, maxPrefix = 1450, suffixTokens = 512, maxNew = 48, prefixBytes = 40 000,
suffixBytes = 16 000, heal = true, repGuard = true, showThreshold = 0.8, suppressPunctOnly = true`. Requires
`ctx + maxNew ≤ model.config.maxContext` and `tok.vocabSize == model.config.vocabSize`.

## Threshold guidance (3 000 positions, SPM, healed run; `eval-heal.md`)

| | Go go31m-e1 | C# cs31m-e1 |
|---|---|---|
| `showThreshold` 0.8, punct-only suppressed, rep guard: shown / line-exact precision | 19.7 % / 98.1 % | 8.7 % / 95.0 % |
| 0.7: shown / precision | 24.7 % / 94.6 % | 12.4 % / 92.2 % |
| 0.8 without the punct-only filter (closers `);`, `}` shown too) | 37.9 % / 98.5 % | 19.7 % / 96.6 % |

Dropping punctuation-only suggestions halves the show rate at the same precision: closers are 2–3 characters the IDE
inserts anyway. Whether to show them is a UX decision (`suppressPunctOnly = false` keeps them); the repetition guard
is free (removes 2.5–2.8 % of lines, all wrong, 13 of them above 0.8 in C#).

## Usage sketch

```kotlin
val weights = NnFormat.read(File(modelsDir, "go-nn-31m-e1.cml"))
val tok = BpeTokenizer.load(Path.of(modelsDir, "go-16384.bpe"))
val model = NnModel(weights, nThreads = 4)                     // NnKernels.best(): native q8 if the library loads
val completion = NnCompletion(model, tok, NnCompletion.Options(showThreshold = 0.8))
val session = model.newSession(2048)                           // per editor; keep between keystrokes

val r = completion.complete(path.toByteArray(), before.toByteArray(), after.toByteArray(), session)
if (r.show) ghostText(r.textString, r.confProd)
```

Latency (docs/NN-PARITY.md, 8 threads, native q8): 1 500-token cold prefill ≈ 150 ms, 24 ms for an incremental line
(8 new prompt tokens + 20 generated); scalar Kotlin ≈ 3.3× slower. The healing itself costs one pre-token scan of the
current line and a binary search in the prefix table (cached per remainder) — microseconds.

## Parity with the Python harness

- `NnParityTest` — plain/SPM prompts and greedy lines vs PyTorch on the real weights (docs/NN-PARITY.md).
- `NnHealParityTest` — `NnCompletion` end to end (typed remainder, prompt ids, generated text, `confProd`, `show`)
  against `tools/nn/eval/make_parity_heal.py` (fp32 fake-quantised model, 254 healed positions × SPM + 42 plain);
  skipped without the fixture (`~/work/ml-data/go/nn/parity-heal`, env `CML_NN_HEAL_PARITY`).
- `NnCompletionTest` — unit tests on the committed BPE fixture + a random tiny model: healing boundaries
  (`foo(⟨⟩)`, `foo()⟨⟩;`, `"x"⟨⟩)`, typed space, `e.Hi⟨⟩gh`, line start, 128-byte chunks), prefix table vs brute force,
  constrained first tokens always start with the typed remainder, masked log-softmax, repetition guard, punct-only rule.
