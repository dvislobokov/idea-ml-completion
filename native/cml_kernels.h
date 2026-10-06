/*
 * cmlkernels — our own SIMD kernels for the int8 transformer inference in ml-core.
 *
 * Plain C11, no dependencies. State-free ABI: every function takes pointers and sizes; the only global state is the
 * selected ISA level (cml_set_isa), which exists for the self-test and benchmarks.
 *
 * Weight layout (as in the .cml file): int8 matrix [K = in][N = out], row-major, one float scale per column:
 *   W[k][c] = q[k][c] * scale[c],   y[t][c] = scale[c] * sum_k x[t][k] * q[k][c]
 *
 * Two activation paths:
 *   f32  — float activations, weights dequantised on the fly (same numbers as the Kotlin kernels up to summation order);
 *   q8   — activations quantised to int8 per (token, block of CML_KB inputs) with a float scale; weights repacked once
 *          into the dot-product layout of cml_pack(); integer dot products (VNNI / pmaddubsw / sdot), float scaling.
 */
#ifndef CML_KERNELS_H
#define CML_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define CML_API __declspec(dllexport)
#else
#define CML_API __attribute__((visibility("default")))
#endif

#define CML_ABI_VERSION 1

/* Inputs per quantised activation block (the per-token activation scale has this granularity). Multiple of 4. */
#define CML_KB 128
/* Column padding of the packed weight layout. */
#define CML_NP 16

enum cml_isa {
    CML_ISA_SCALAR = 0,
    CML_ISA_AVX2 = 1,          /* AVX2 + FMA */
    CML_ISA_AVX2_VNNI = 2,     /* + AVX-VNNI (VEX vpdpbusd), Alder Lake+/Zen 5+ — UNTESTED (no such CPU here) */
    CML_ISA_AVX512 = 3,        /* AVX-512 F/BW/DQ/VL */
    CML_ISA_AVX512_VNNI = 4,   /* + AVX-512 VNNI */
    CML_ISA_NEON = 16,         /* aarch64 base — UNTESTED */
    CML_ISA_NEON_DOTPROD = 17  /* + FEAT_DotProd (all Apple Silicon) — UNTESTED */
};

CML_API int cml_abi_version(void);
/* Best ISA level supported by this CPU (and by this build). */
CML_API int cml_isa_detect(void);
/* Currently selected level (defaults to cml_isa_detect() on first use). */
CML_API int cml_isa(void);
CML_API const char* cml_isa_name(int isa);
/* Selects a level (clamped to what the CPU supports); returns the level in effect. */
CML_API int cml_set_isa(int isa);

/* Runs every kernel of the selected level against the scalar reference on random data. 0 = OK, else the id of the
 * first failing check (see selftest.c). Takes a few milliseconds. */
CML_API int cml_selftest(void);

/* ---------------------------------------------------------------- f32 activations ---------------------------- */

/* y[t][c] = scale[c] * sum_k x[t][k] * q[k][c]  for t in [0, n), c in [c0, c1). Overwrites y. x rows are ldx floats
 * apart, y rows ldy floats apart (y points at column 0 of row 0, i.e. y[t*ldy + c] is written for c in [c0, c1)). */
CML_API void cml_gemm_f32(const int8_t* q, const float* scale, int K, int N,
                          const float* x, int ldx, int n, float* y, int ldy, int c0, int c1);

/* acc[c] += sum_{k in [k0, k1)} x[k] * q[k][c] for all c in [0, N) — unscaled partial sums (K-split decoding). */
CML_API void cml_gemv_f32_partial(const int8_t* q, int K, int N, const float* x, int k0, int k1, float* acc);

/* ---------------------------------------------------------------- int8 activations --------------------------- */

/* Byte size of the packed form of a K x N matrix (64-byte aligned buffer expected). */
CML_API size_t cml_pack_size(int K, int N);
/* Repacks q[K][N] into the dot-product layout: K padded to a multiple of CML_KB, N to a multiple of CML_NP; panels of
 * CML_NP columns, each [Kp/4][CML_NP][4] (4 consecutive inputs interleaved per column, a panel = one contiguous stream
 * over K), followed by per-block column sums ([Kp/KB][Np] int32, used to undo the unsigned offset of the x86 kernels). */
CML_API void cml_pack(const int8_t* q, int K, int N, void* packed);

/* Byte size of the quantised form of n activation rows of K floats; cml_qact_stride is the per-row stride. */
CML_API size_t cml_qact_size(int n, int K);
CML_API size_t cml_qact_stride(int K);
/* Quantises rows [0, n) of x (ldx floats apart) into xq. Per (row, block of CML_KB): scale = max|x| / 127,
 * q = round(x / scale) in [-127, 127]; stored as signed bytes, or offset by +128 (unsigned) when the selected ISA
 * uses u8*s8 products (x86 VNNI). Row layout: Kp bytes, then Kp/KB float scales, padded to 64 bytes. */
CML_API void cml_qact(const float* x, int ldx, int n, int K, void* xq);
/* Whether cml_qact currently writes unsigned (offset) bytes — for tests. */
CML_API int cml_qact_unsigned(void);

/* y[(t - t0)*ldy + c] = scale[c] * sum_k x[t][k] * q[k][c] for t in [t0, t1), c in [c0, c1), reading quantised rows
 * t0.. of xq (produced by cml_qact for the whole batch). y points at row t0. */
CML_API void cml_gemm_q8(const void* packed, const float* scale, int K, int N,
                         const void* xq, int t0, int t1, float* y, int ldy, int c0, int c1);

/* ---------------------------------------------------------------- small ops ---------------------------------- */

/* out[j] = x[j] / sqrt(mean(x^2) + eps) * w[j]; tiny results flushed to zero (|v| < 1e-30) like the Kotlin path. */
CML_API void cml_rmsnorm(const float* x, const float* w, int d, float eps, float* out);
/* g[j] = silu(g[j]) * u[j] (flush-to-zero like the Kotlin path). */
CML_API void cml_silu_mul(float* g, const float* u, int n);
/* In-place softmax over s[0..n). */
CML_API void cml_softmax(float* s, int n);

/* Attention for one query head over the transposed key cache: sc[s] = scale * sum_d q[d] * kT[d*ldk + s] for s < len,
 * softmax, out[d] = sum_s sc[s] * v[s*ldv + d]. kT rows are ldk floats apart (one row per dimension), v rows ldv apart
 * (one row per position). sc is scratch of >= len floats. */
CML_API void cml_attn_head(const float* q, int hd, float scale, const float* kT, int ldk, const float* v, int ldv,
                           int len, float* sc, float* out);

#ifdef __cplusplus
}
#endif
#endif
