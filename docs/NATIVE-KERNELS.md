# Native SIMD kernels for the neural inference (`native/`, spike 2026-10-06)

Status: **research spike** — measured, tested, not yet a shipping decision. Our own C11 library (`libcmlkernels`,
~900 lines of C) replaces the hot loops of `ml-core`'s int8 transformer inference (`nn.NnModel`) through JNI, with a
mandatory fallback to the Kotlin kernels. No third-party runtime, no BLAS; the only dependency is the C library of the
OS (and pthreads for the C benchmark driver only — the shipped library itself spawns no threads).

## What is native, what is not

| piece | native | Kotlin |
|---|---|---|
| matmul / matvec (int8 weights `[in,out]`, per-column scale) | `cml_gemm_f32`, `cml_gemm_q8`, `cml_gemv_f32_partial` | fallback `ScalarNnKernels` |
| activation quantisation (Q8 mode), weight repacking | `cml_qact`, `cml_pack` | — |
| attention (scores over transposed keys, softmax, values) + KV cache memory | `cml_attn_head`, `NativeKv` | fallback `Attn` + heap arrays |
| SiLU·up | `cml_silu_mul` | fallback loop |
| RMSNorm, RoPE, residual adds, embedding lookup, sampling, orchestration | — | Kotlin (small; RoPE uses cos/sin tables) |

Two activation modes (`NativeNnKernels.Mode`, property `completionml.nn.native.mode=q8|f32`, default `q8`):

- **f32** — float activations, weights dequantised on the fly. Same arithmetic as the Kotlin path (results differ only by
  summation order; the model tests pass with the same tolerance). Weights are read in place from the mapped `.cml`.
- **q8** — activations quantised to int8 per (token, block of 128 inputs) with a float scale; weights repacked once per
  matrix into panels of 16 columns, each `[Kp/4][16][4]` (groups of 4 inputs interleaved per column; K padded to 128, N
  to 16; a panel is one contiguous stream over K, which matters for decode — the first `[Kp/4][Np][4]` layout made the
  vocabulary head's GEMV jump 128 KB per step and halved decode speed), plus per-block column sums. Integer dot products: AVX-512 VNNI `vpdpbusd` (u8 activations = s8 + 128, undone with the column sums),
  AVX2 `vpmaddubsw`/`vpmaddwd` with the sign trick (|w| × sign(w)·x, no saturation since |q| ≤ 127), NEON `sdot`
  (s8 × s8). Adds ~0.4 % per-element noise to activations — the usual W8A8 scheme; **validate on the trained model**
  (ppl / eval-inline) before choosing it as default. The packed copy doubles the weight memory in native heap (the mapped
  original stays untouched, so it does not count twice in RSS once paged out).

## ABI (`native/cml_kernels.h`)

State-free: pointers + sizes; the only global is the selected ISA level. Weights: int8 `q[K][N]` row-major, `scale[N]`.
Activations `x[t*ldx + k]`, outputs `y[t*ldy + c]`, column range `[c0, c1)`, token range `[t0, t1)`.

```
int   cml_abi_version();                  // 1
int   cml_isa_detect(); cml_isa(); cml_set_isa(int); const char* cml_isa_name(int);
int   cml_selftest();                     // 0 = every kernel of the selected level matches the scalar reference
void  cml_gemm_f32(q, scale, K, N, x, ldx, n, y, ldy, c0, c1);
void  cml_gemv_f32_partial(q, K, N, x, k0, k1, acc);        // acc[c] += Σ_{k∈[k0,k1)} x[k] q[k][c] (unscaled)
size_t cml_pack_size(K, N); void cml_pack(q, K, N, packed);
size_t cml_qact_size(n, K); cml_qact_stride(K); void cml_qact(x, ldx, n, K, xq); int cml_qact_unsigned();
void  cml_gemm_q8(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1);
void  cml_rmsnorm(x, w, d, eps, out); cml_silu_mul(g, u, n); cml_softmax(s, n);
void  cml_attn_head(q, hd, scale, kT, ldk, v, ldv, len, sc, out);
```

ISA levels: `0` scalar (SSE2 baseline, auto-vectorised), `1` AVX2+FMA, `2` AVX2+AVX-VNNI (**untested**: no such CPU here),
`3` AVX-512 F/BW/DQ/VL, `4` AVX-512 VNNI, `16` NEON, `17` NEON+DotProd (**untested**). Detection is our own `cpuid` +
`xgetbv` (OS-enabled ZMM state), not `__builtin_cpu_supports` (compiler-rt's version is not PIC under zig). One binary
per OS/arch; every level lives in functions with `__attribute__((target(...)))`.

Kernel shapes (register tiles, MR tokens × NR columns): AVX-512 f32 12×32 (24 zmm accumulators, int8→f32 via
`vpmovsxbd`+`vcvtdq2ps`, activations as embedded broadcasts), AVX2 f32 6×16; AVX-512 VNNI 6×32 (int32 accumulators per
128-block + float accumulators in registers), AVX2 q8 4×16 (float accumulation in `y` between blocks), NEON f32 8×8
(`fmla` by lane), NEON dot 4×16. Drivers block K in 512 (f32) and columns in 256 so the weight panel stays in L2.

## Kernel throughput (C benchmark `native/bench`, EPYC 9554 @ ~3.1 GHz, random data, n = 512 tokens, GMAC/s)

Roofline per core: f32 FMA 16 MAC/cycle (Zen 4: 2×256-bit FMA/cycle, AVX-512 double-pumped) ≈ 50 GMAC/s;
AVX-512 VNNI `vpdpbusd` 64 MAC/cycle ≈ 200 GMAC/s; AVX2 `vpmaddubsw`+`vpmaddwd` ≈ 32 MAC/cycle ≈ 100 GMAC/s.

| path | 1 thread | 8 threads | 16 threads |
|---|---|---|---|
| AVX-512 f32 (dequant on the fly), 12×32 tile | 50–53 (≈ roofline) | 270–400 | 525–765 |
| AVX-512 VNNI q8, 6×32 tile | 178–211 (90–100 %) | 680–1070 | 1040–2180 |
| AVX2 f32, 6×16 tile (forced level 1) | 45–50 (≈ roofline) | 245–370 | — |
| AVX2 q8 sign trick, 4×16 tile | 61–66 (~60 %) | 270–390 | — |
| GEMV (decode), GB/s of weights per thread: f32 / q8 | 25 / 52–97 | 180 / 320 total (L3-resident) | 340 / 430 total |

The 8/16-thread numbers were taken on a shared box (load 7–11) and with ~30 tasks of 10–50 µs each; the larger matrices
reach 70–75 % of linear scaling, the small ones less (dispatch + barrier of the benchmark pool).
The f32 AVX2 tile needed `#pragma GCC unroll` on the token loop: without it gcc kept the accumulators in memory (23 GMAC/s).

## End to end (`NnBench`, random weights, 512-token prompt + 20 tokens, taskset 8 cores, this server, ms)

| cfg | threads | scalar Kotlin: prefill / decode / line / reuse8 | native f32 | native q8 (VNNI) |
|---|---|---|---|---|
| S 30M | 1 | 611 / 3.71 / 681 / 88 | 230 / 2.86 / 283 / 62 | 108 / 1.32 / 132 / 28 |
| S | 4 | 235 / 2.03 / 273 / 48 | 86 / 1.36 / 111 / 30 | 57 / 1.07 / 77 / 23 |
| S | 8 | 118 / 1.62 / 148 / 39 | 49 / 0.99 / 67 / 22 | 33 / 0.70 / 46 / 16 |
| S | 16 | 68 / 1.80 / 102 / 42 | — | 21 / 0.64 / 33 / 14 |
| M 48M | 1 | 1021 / 6.1 / 1138 / 144 | 391 / 5.6 / 499 / 121 | 179 / 3.5 / 245 / 73 |
| M | 8 | 166 / 2.20 / 208 / 53 | 81 / 1.46 / 108 / 33 | 49 / 0.94 / 67 / 21 |
| M | 16 | 107 / 2.02 / 145 / 49 | — | 34 / 0.95 / 51 / 22 |
| L 99M | 1 | 2421 / 12.4 / 2659 / 306 | 857 / 7.9 / 1007 / 171 | 348 / 4.4 / 432 / 94 |
| L | 4 | 797 / 6.1 / 912 / 151 | 283 / 4.5 / 369 / 96 | 161 / 4.7 / 252 / 99 |
| L | 8 | 457 / 4.0 / 534 / 100 | 161 / 2.7 / 212 / 59 | 89 / 1.7 / 122 / 37 |
| L | 16 | 251 / 3.3 / 313 / 80 | — | 65 / 2.0 / 104 / 45 |
| S, AVX2 level forced | 8 | — | 50 / 1.18 / 73 / 27 | 45 / 1.07 / 66 / 23 |
| L, AVX2 level forced | 8 | — | 171 / 3.3 / 233 / 73 | 151 / 3.1 / 211 / 67 |

(q8 rows with 1/8 threads re-measured after the panel packing; the 4/16-thread q8 rows and the AVX2-level rows are
from the earlier layout and are 10–40 % pessimistic for decode.) Profile of the S / L q8 prefill at 8 threads:
matmul 14.5 / 48 ms (of which heap↔native copies ≈ 5 / 12 ms wall), attention 9.5 / 23, rope+kv 4.3 / 7, norms 2 / 5,
SiLU 1 / 2 — the kernels themselves are now a third of the prefill; the rest is the JVM boundary and Kotlin glue.

## Build

```
cd native
make            # host gcc: build/libcmlkernels-linux-x64.so, build/bench, build/selftest
make test       # self-test at every ISA level the host supports
make cross      # zig cc: the four shipping binaries (linux-x64 .so, windows-x64 .dll, macos-arm64/x64 .dylib)
make install    # copies them to ml-core/src/main/resources/native/
```

Toolchain: `zig` 0.14 (apt `zig` on this server; one package cross-compiles all targets, bundles libc/mingw/libSystem
stubs). JNI headers: `$JAVA_HOME/include/jni.h` plus our own minimal `native/jni-md/jni_md.h` (all targets are 64-bit).
Binary sizes (stripped): linux 84 KB, windows 101 KB, macos-arm64 103 KB, macos-x64 91 KB — ~380 KB added to the jar.
The Windows DLL imports only `KERNEL32` and the UCRT (`api-ms-win-crt-*`, present on Windows 10+).

The gcc and zig (clang 19) builds of the Linux library measure the same (f32 51 GMAC/s, q8 ≈ 190 GMAC/s per core).

## Loading and fallback (`nn/native/NativeLib.kt`)

1. `-Dcompletionml.nn.native=false` or env `CML_NATIVE=0` → never loaded (Kotlin kernels).
2. Resource `native/libcmlkernels-<os>-<arch>.<ext>` is extracted to `<java.io.tmpdir>/cmlkernels/<sha1>/` (reused when
   present; written via temp file + atomic rename, so concurrent JVMs and a locked DLL on Windows are fine),
   `System.load`, `cml_abi_version() == 1`, optional `-Dcompletionml.nn.native.isa=N` (force a lower level, e.g. AVX2 on an
   AVX-512 box), then `cml_selftest()` **on the user's CPU** — every kernel of the selected level against the scalar C
   reference on random shapes incl. odd sizes. Any failure (unsupported platform, missing resource, link error,
   self-test mismatch) → `NativeLib.status` holds the reason and `NnKernels.best()` returns the Kotlin kernels.
3. `-Dcompletionml.nn.native.path=/abs/lib.so` loads a given file instead (benchmarks, debugging).

`NnKernels.best()` order: native → Vector API (only outside JBR) → scalar. One `NativeNnKernels` instance per model
(owns scratch and the prepared/packed weights; freed by `NnModel.close()`). `NnSession` is `AutoCloseable` now (native KV
cache; a `Cleaner` frees it if forgotten).

Safety: a crash in native code kills the IDE. Mitigations in place: the self-test at load, fuzz test vs the Kotlin
kernels (`NativeNnKernelsTest`), bounds come from the Kotlin side (`NnTensor.Q8` validates shapes), heap arrays are
pinned with `GetPrimitiveArrayCritical` only for the duration of a kernel call. Not done yet: ASan/UBSan run of the JNI
glue under the JVM tests, Windows/macOS loading tests.

## Threading and the JVM boundary

The native library spawns no threads; the model's `NnExecutor` dispatches tasks that each make one JNI call:
- prefill (n ≥ 48 tokens): token-block tasks (~3 per thread) — each copies its activation rows into native scratch
  (`FloatBuffer.put`, ~85 GB/s), quantises them (Q8), runs every matrix over all columns, copies the output rows back;
- 2–47 tokens: (matrix × 256-column chunk × token block) tasks;
- decode (1 token): ~2 column chunks per thread across the matrices of the region.
Attention: one JNI call per (head, query block) task over the native KV cache; `kvStore` one call per layer.

Measured JNI costs (JDK 21, EPYC 9554): empty JNI call 5 ns, kernel call with 11 args 84 ns, `GetPrimitiveArrayCritical`
copy of 384 floats 21 ns, `FloatBuffer.put/get` of 384 floats 18 ns (4096: 170 ns), `NnExecutor.run(8 tasks)` round trip
3.9 µs. A decode step does ~45 parallel regions and ~400 kernel calls: ~0.2 ms of dispatch per token (S: 0.7 ms/token
total), i.e. the executor, not JNI, is the decode floor. For prefill the remaining non-kernel costs are the heap↔native
copies and the Kotlin glue (profile with `-Dcompletionml.nn.profile=true`, see the report).

## Platform notes

- **Linux x64**: tested (this server, gcc and zig builds).
- **Windows x64**: compiles (zig, mingw-w64 CRT → UCRT imports). Untested: loading from the JBR, antivirus false
  positives on an unsigned DLL extracted to `%TEMP%` (common for JNI libs; mitigations: sign the DLL with the same
  code-signing certificate as the plugin, extract under the plugin's own directory instead of `%TEMP%`).
- **macOS arm64 / x64**: compiles (zig, no SDK needed). Untested. Gatekeeper: libraries loaded with `dlopen` from a
  signed, notarised plugin zip are fine if the dylib is itself code-signed (ad-hoc signing `codesign -s -` is enough for
  `dlopen` of a library the user's app writes to disk, but the Marketplace build should sign with the Developer ID and
  include the dylibs in the notarisation of the plugin archive). Hardened runtime of the JBR allows JNI libraries.
  **NEON verified 2026-10-07 on a MacBook Pro M1 Pro (32 GB)**: `NativeNnKernelsTest` reports
  `loaded libcmlkernels-macos-arm64.dylib, ISA neon-dotprod (detected neon-dotprod)` — the dylib loads from the jar without
  Gatekeeper prompts and the NEON + DotProd kernels pass the self-test against the scalar reference.
  `NnBench` on the real cs31m model (`models/cs-nn-31m-e1.cml`, 512-token prompt + 20 tokens, 4 threads, ms; prefill / decode per
  token / line / line while typing with 8 new tokens): scalar Kotlin 219 / 1.71 / 252 / 40; native f32 120 / 2.28 / 164 / 49;
  native q8 (`sdot`) 55 / 0.75 / 68 / 17 — ×3.7 on a cold line, ×2.4 while typing; `NnKernels.best()` picks q8 by itself.
  The NEON f32 path decodes slower than scalar Kotlin (2.28 vs 1.71 ms/token): not the default, but worth a look if f32 is ever needed.
  Scalar Kotlin on the M1 Pro is on par with the EPYC server at 4 threads.
  `NnParityTest` on the same Mac: scalar and native f32 reproduce the PyTorch int8 reference bit-for-bit (argmax 32/32,
  greedy 32/32, logprob rmse 0); native q8 argmax 30/32, same line on 39/40 positions, identical line accuracy vs truth —
  the usual W8A8 noise, same as VNNI on x86.
- **AVX-VNNI (VEX) level** untested (no Alder Lake/Zen 5 here); guarded by the self-test.

## Known gaps / next steps if we ship

1. Native activations end to end (one JNI call per layer or per forward, activations in native buffers): removes the
   copies and most dispatch; the biggest remaining win for prefill and the only way to make decode bandwidth-bound.
2. Native thread pool (or `NnExecutor` with futex-style parking) for decode; one region per layer.
3. int4 weights (groups of 32) for decode on laptops: decode is DRAM-bound there (S = 30 MB per token).
4. Export packed Q8 weights in `.cml` to avoid the repack at load (~0.1 s per 100 MB) and the second copy.
5. Attention kernel: blocked over queries × keys (currently one query at a time, auto-vectorised C).
6. CI: build the four binaries reproducibly (zig version pinned), run the JVM tests on Windows and macOS runners,
   `-Dcompletionml.nn.expectNative=true` so a silently skipped native path fails the build.
