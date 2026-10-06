/*
 * cmlkernels: dispatch, scalar reference kernels, packing/quantisation, self-test. SIMD paths: kernels_x86.inc
 * (AVX2 / AVX-VNNI / AVX-512 / AVX-512 VNNI, selected at runtime by cpuid) and kernels_neon.inc (aarch64).
 * Build: see Makefile. Plain C11; the SIMD code lives in functions with per-function target attributes, so one
 * translation unit compiled for the baseline ISA (x86-64 / armv8-a) contains every level.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "cml_kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define CML_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CML_ARM64 1
#include <arm_neon.h>
#endif

#if defined(__clang__)
#define CML_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define CML_UNROLL _Pragma("GCC unroll 16")
#else
#define CML_UNROLL
#endif
#if defined(__GNUC__) || defined(__clang__)
#define CML_INLINE static inline __attribute__((always_inline))
#define CML_NOINLINE __attribute__((noinline))
#define CML_RESTRICT __restrict__
#else
#define CML_INLINE static inline
#define CML_NOINLINE
#define CML_RESTRICT
#endif

/* ----------------------------------------------------------------------------------------------- layout helpers */

static inline int round_up(int v, int m) { return (v + m - 1) / m * m; }
static inline size_t align64(size_t v) { return (v + 63) & ~(size_t)63; }

static inline int kp_of(int K) { return round_up(K, CML_KB); }
static inline int np_of(int N) { return round_up(N, CML_NP); }

/* packed weights: panels of CML_NP columns, each [Kp/4][CML_NP][4] int8 (so a panel is one contiguous stream over K),
 * then [Kp/KB][Np] int32 holding -128 * column sums of each block */
static inline size_t pack_index(int k, int c, int kp) {
    return ((size_t)(c / CML_NP) * (size_t)(kp / 4) + (size_t)(k / 4)) * (CML_NP * 4) + (size_t)(c % CML_NP) * 4 + (size_t)(k & 3);
}
static inline const int32_t* pack_negsums(const void* packed, int K, int N) {
    return (const int32_t*)((const char*)packed + (size_t)kp_of(K) * np_of(N));
}

CML_API size_t cml_pack_size(int K, int N) {
    size_t kp = (size_t)kp_of(K), np = (size_t)np_of(N);
    return align64(kp * np + (kp / CML_KB) * np * sizeof(int32_t));
}

CML_API void cml_pack(const int8_t* q, int K, int N, void* packed) {
    const int kp = kp_of(K), np = np_of(N);
    int8_t* w = (int8_t*)packed;
    memset(packed, 0, cml_pack_size(K, N));
    for (int k = 0; k < K; k++) {
        const int8_t* row = q + (size_t)k * N;
        for (int c = 0; c < N; c++) w[pack_index(k, c, kp)] = row[c];
    }
    int32_t* sums = (int32_t*)(w + (size_t)kp * np);
    for (int b = 0; b < kp / CML_KB; b++) {
        int32_t* s = sums + (size_t)b * np;
        int kend = (b + 1) * CML_KB; if (kend > K) kend = K;
        for (int k = b * CML_KB; k < kend; k++) {
            const int8_t* row = q + (size_t)k * N;
            for (int c = 0; c < N; c++) s[c] -= 128 * (int32_t)row[c];
        }
    }
}

CML_API size_t cml_qact_stride(int K) {
    size_t kp = (size_t)kp_of(K);
    return align64(kp + (kp / CML_KB) * sizeof(float));
}

CML_API size_t cml_qact_size(int n, int K) { return (size_t)n * cml_qact_stride(K); }

static inline float* qact_scales(void* row, int K) { return (float*)((char*)row + kp_of(K)); }
static inline const float* qact_scales_c(const void* row, int K) { return (const float*)((const char*)row + kp_of(K)); }

/* ----------------------------------------------------------------------------------------------- ISA selection */

static int g_isa = -1;
static int g_detected = -1;

/* Bitmask of the ISA levels this CPU (and this build) can run. */
static unsigned g_supported = 0;

#if CML_X86 && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
static void cpuid_count(unsigned leaf, unsigned sub, unsigned r[4]) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(leaf, sub, a, b, c, d);
    r[0] = a; r[1] = b; r[2] = c; r[3] = d;
}
static unsigned long long xgetbv0(void) {
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}
#endif

/* cpuid-based detection (own code: compiler-rt's __builtin_cpu_supports is not position-independent in every
 * toolchain), including the OS-enabled register state (XCR0). */
static unsigned detect_supported(void) {
    unsigned m = 1u << CML_ISA_SCALAR;
#if CML_X86 && (defined(__GNUC__) || defined(__clang__))
    unsigned r[4];
    cpuid_count(0, 0, r);
    unsigned maxleaf = r[0];
    if (maxleaf < 7) return m;
    cpuid_count(1, 0, r);
    int osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1;
    if (!(osxsave && avx && fma)) return m;
    unsigned long long xcr0 = xgetbv0();
    if ((xcr0 & 0x6) != 0x6) return m;                       /* XMM + YMM state enabled by the OS */
    unsigned l7[4], l7s1[4];
    cpuid_count(7, 0, l7);
    cpuid_count(7, 1, l7s1);
    int avx2 = (l7[1] >> 5) & 1;
    if (!avx2) return m;
    m |= 1u << CML_ISA_AVX2;
#if defined(CML_HAVE_AVXVNNI)
    if ((l7s1[0] >> 4) & 1) m |= 1u << CML_ISA_AVX2_VNNI;
#endif
    int f = (l7[1] >> 16) & 1, dq = (l7[1] >> 17) & 1, bw = (l7[1] >> 30) & 1, vl = (l7[1] >> 31) & 1;
    if (f && dq && bw && vl && (xcr0 & 0xE0) == 0xE0) {     /* opmask + ZMM state enabled */
        m |= 1u << CML_ISA_AVX512;
        if ((l7[2] >> 11) & 1) m |= 1u << CML_ISA_AVX512_VNNI;
    }
#elif CML_ARM64
    /* FEAT_DotProd is present on every Apple Silicon / armv8.4+ core; we only ship aarch64 for macOS. */
    m |= (1u << CML_ISA_NEON) | (1u << CML_ISA_NEON_DOTPROD);
#endif
    return m;
}

static unsigned supported(void) {
    if (!g_supported) g_supported = detect_supported();
    return g_supported;
}

CML_API int cml_abi_version(void) { return CML_ABI_VERSION; }

CML_API int cml_isa_detect(void) {
    if (g_detected < 0) {
        unsigned m = supported();
        int best = CML_ISA_SCALAR;
        for (int l = 0; l < 32; l++) if (m & (1u << l)) best = l;
        g_detected = best;
    }
    return g_detected;
}

CML_API int cml_isa(void) {
    if (g_isa < 0) g_isa = cml_isa_detect();
    return g_isa;
}

CML_API int cml_set_isa(int isa) {
    if (isa < 0 || isa >= 32 || !(supported() & (1u << isa))) {
        /* unsupported: the highest supported level below it */
        int pick = CML_ISA_SCALAR;
        for (int l = 0; l < 32 && l < isa; l++) if (supported() & (1u << l)) pick = l;
        isa = pick;
    }
    g_isa = isa;
    return g_isa;
}

CML_API const char* cml_isa_name(int isa) {
    switch (isa) {
        case CML_ISA_SCALAR: return "scalar";
        case CML_ISA_AVX2: return "avx2";
        case CML_ISA_AVX2_VNNI: return "avx2-vnni";
        case CML_ISA_AVX512: return "avx512";
        case CML_ISA_AVX512_VNNI: return "avx512-vnni";
        case CML_ISA_NEON: return "neon";
        case CML_ISA_NEON_DOTPROD: return "neon-dotprod";
        default: return "?";
    }
}

static inline int isa_unsigned_act(int isa) { return isa == CML_ISA_AVX2_VNNI || isa == CML_ISA_AVX512_VNNI; }

CML_API int cml_qact_unsigned(void) { return isa_unsigned_act(cml_isa()); }

/* ----------------------------------------------------------------------------------------------- generic pieces */

/* Fast exp for softmax (relative error ~2e-7 over [-87, 88]); written to auto-vectorise. */
CML_INLINE float fast_exp(float x) {
    if (x < -87.0f) x = -87.0f;
    if (x > 88.0f) x = 88.0f;
    float n = floorf(x * 1.44269504f + 0.5f);
    float r = x - n * 0.693145751953125f - n * 1.428606765330187e-06f;
    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    union { float f; int32_t i; } u;
    u.i = ((int32_t)n + 127) << 23;
    return p * u.f;
}

CML_INLINE float ftz(float v) { return fabsf(v) < 1e-30f ? 0.0f : v; }

/* The generic (auto-vectorisable) small ops, instantiated once per ISA level via GEN_SMALL_OPS(suffix, attr). */
#define GEN_SMALL_OPS(SUF, ATTR)                                                                                     \
    ATTR static void rmsnorm_##SUF(const float* CML_RESTRICT x, const float* CML_RESTRICT w, int d, float eps,        \
                                   float* CML_RESTRICT out) {                                                        \
        float ss = 0.0f;                                                                                             \
        for (int j = 0; j < d; j++) ss += x[j] * x[j];                                                               \
        float inv = (float)(1.0 / sqrt((double)ss / d + (double)eps));                                               \
        for (int j = 0; j < d; j++) out[j] = ftz(x[j] * inv * w[j]);                                                 \
    }                                                                                                                \
    ATTR static void silu_mul_##SUF(float* CML_RESTRICT g, const float* CML_RESTRICT u, int n) {                     \
        for (int j = 0; j < n; j++) {                                                                                \
            float v = g[j];                                                                                          \
            float s = v < -80.0f ? 0.0f : v / (1.0f + fast_exp(-v));                                                 \
            g[j] = ftz(s * u[j]);                                                                                    \
        }                                                                                                            \
    }                                                                                                                \
    ATTR static void softmax_##SUF(float* CML_RESTRICT s, int n) {                                                   \
        float mx = -INFINITY;                                                                                        \
        for (int i = 0; i < n; i++) mx = s[i] > mx ? s[i] : mx;                                                      \
        float sum = 0.0f;                                                                                            \
        for (int i = 0; i < n; i++) { float z = s[i] - mx; float e = z < -60.0f ? 0.0f : fast_exp(z); s[i] = e; sum += e; } \
        float inv = 1.0f / sum;                                                                                      \
        for (int i = 0; i < n; i++) s[i] = ftz(s[i] * inv);                                                          \
    }                                                                                                                \
    ATTR static void attn_head_##SUF(const float* CML_RESTRICT q, int hd, float scale, const float* CML_RESTRICT kT, \
                                     int ldk, const float* CML_RESTRICT v, int ldv, int len, float* CML_RESTRICT sc, \
                                     float* CML_RESTRICT out) {                                                      \
        for (int s = 0; s < len; s++) sc[s] = 0.0f;                                                                  \
        int d = 0;                                                                                                   \
        for (; d + 4 <= hd; d += 4) {                                                                                \
            float a0 = q[d] * scale, a1 = q[d + 1] * scale, a2 = q[d + 2] * scale, a3 = q[d + 3] * scale;           \
            const float* k0 = kT + (size_t)d * ldk; const float* k1 = k0 + ldk; const float* k2 = k1 + ldk;          \
            const float* k3 = k2 + ldk;                                                                              \
            for (int s = 0; s < len; s++) sc[s] += k0[s] * a0 + k1[s] * a1 + k2[s] * a2 + k3[s] * a3;               \
        }                                                                                                            \
        for (; d < hd; d++) {                                                                                        \
            float a = q[d] * scale; const float* k0 = kT + (size_t)d * ldk;                                          \
            for (int s = 0; s < len; s++) sc[s] += k0[s] * a;                                                        \
        }                                                                                                            \
        softmax_##SUF(sc, len);                                                                                      \
        for (int j = 0; j < hd; j++) out[j] = 0.0f;                                                                  \
        int s = 0;                                                                                                   \
        for (; s + 4 <= len; s += 4) {                                                                               \
            float a0 = sc[s], a1 = sc[s + 1], a2 = sc[s + 2], a3 = sc[s + 3];                                        \
            const float* v0 = v + (size_t)s * ldv; const float* v1 = v0 + ldv; const float* v2 = v1 + ldv;           \
            const float* v3 = v2 + ldv;                                                                              \
            for (int j = 0; j < hd; j++) out[j] += v0[j] * a0 + v1[j] * a1 + v2[j] * a2 + v3[j] * a3;               \
        }                                                                                                            \
        for (; s < len; s++) {                                                                                       \
            float a = sc[s]; const float* v0 = v + (size_t)s * ldv;                                                  \
            for (int j = 0; j < hd; j++) out[j] += v0[j] * a;                                                        \
        }                                                                                                            \
    }                                                                                                                \
    /* quantise one row: per block scale, signed or offset bytes */                                                  \
    ATTR static void qact_row_generic_##SUF(const float* CML_RESTRICT x, int K, uint8_t* CML_RESTRICT out,           \
                                    float* CML_RESTRICT scales, int unsigned_out) {                                  \
        const int kp = kp_of(K);                                                                                     \
        for (int b = 0; b < kp / CML_KB; b++) {                                                                      \
            int k0 = b * CML_KB, k1 = k0 + CML_KB; if (k1 > K) k1 = K;                                               \
            float mx = 0.0f;                                                                                         \
            for (int k = k0; k < k1; k++) { float a = fabsf(x[k]); mx = a > mx ? a : mx; }                           \
            float s = mx / 127.0f, inv = s > 0.0f ? 1.0f / s : 0.0f;                                                 \
            scales[b] = s;                                                                                           \
            int off = unsigned_out ? 128 : 0;                                                                        \
            for (int k = k0; k < k1; k++) {                                                                          \
                int qv = (int)lrintf(x[k] * inv);                                                                    \
                qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);                                                       \
                out[k] = (uint8_t)(qv + off);                                                                        \
            }                                                                                                        \
            for (int k = k1; k < k0 + CML_KB; k++) out[k] = (uint8_t)off;                                            \
        }                                                                                                            \
    }

/* ----------------------------------------------------------------------------------------------- scalar kernels */

GEN_SMALL_OPS(scalar, )

/* y[t][c] (c in [c0,c1)) = scale * sum_k x[t][k] q[k][c]; plain loops (auto-vectorised at the baseline ISA). */
static void gemm_f32_scalar(const int8_t* CML_RESTRICT q, const float* CML_RESTRICT scale, int K, int N,
                            const float* CML_RESTRICT x, int ldx, int n, float* CML_RESTRICT y, int ldy, int c0, int c1) {
    const int cw = c1 - c0;
    for (int t = 0; t < n; t++) {
        float* yt = y + (size_t)t * ldy + c0;
        const float* xt = x + (size_t)t * ldx;
        for (int c = 0; c < cw; c++) yt[c] = 0.0f;
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            const int8_t* r0 = q + (size_t)k * N + c0; const int8_t* r1 = r0 + N; const int8_t* r2 = r1 + N; const int8_t* r3 = r2 + N;
            float a0 = xt[k], a1 = xt[k + 1], a2 = xt[k + 2], a3 = xt[k + 3];
            for (int c = 0; c < cw; c++) yt[c] += r0[c] * a0 + r1[c] * a1 + r2[c] * a2 + r3[c] * a3;
        }
        for (; k < K; k++) {
            const int8_t* r0 = q + (size_t)k * N + c0; float a0 = xt[k];
            for (int c = 0; c < cw; c++) yt[c] += r0[c] * a0;
        }
        for (int c = 0; c < cw; c++) yt[c] *= scale[c0 + c];
    }
}

static void gemv_f32_partial_scalar(const int8_t* CML_RESTRICT q, int K, int N, const float* CML_RESTRICT x, int k0, int k1,
                                    float* CML_RESTRICT acc) {
    (void)K;
    int k = k0;
    for (; k + 4 <= k1; k += 4) {
        const int8_t* r0 = q + (size_t)k * N; const int8_t* r1 = r0 + N; const int8_t* r2 = r1 + N; const int8_t* r3 = r2 + N;
        float a0 = x[k], a1 = x[k + 1], a2 = x[k + 2], a3 = x[k + 3];
        for (int c = 0; c < N; c++) acc[c] += r0[c] * a0 + r1[c] * a1 + r2[c] * a2 + r3[c] * a3;
    }
    for (; k < k1; k++) {
        const int8_t* r0 = q + (size_t)k * N; float a0 = x[k];
        for (int c = 0; c < N; c++) acc[c] += r0[c] * a0;
    }
}

/* Reference for the packed/quantised path: integer dot products per block, float scaling. */
static void gemm_q8_scalar(const void* packed, const float* scale, int K, int N, const void* xq, int t0, int t1,
                           float* y, int ldy, int c0, int c1, int unsigned_act) {
    const int kp = kp_of(K), np = np_of(N), nb = kp / CML_KB;
    const int8_t* w = (const int8_t*)packed;
    const int32_t* negsums = pack_negsums(packed, K, N);
    const size_t xs = cml_qact_stride(K);
    for (int t = t0; t < t1; t++) {
        const uint8_t* xrow = (const uint8_t*)xq + (size_t)t * xs;
        const float* xsc = qact_scales_c(xrow, K);
        float* yt = y + (size_t)(t - t0) * ldy;
        for (int c = c0; c < c1; c++) {
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                int32_t s = unsigned_act ? negsums[(size_t)b * np + c] : 0;
                for (int k = b * CML_KB; k < (b + 1) * CML_KB; k++) {
                    int xv = unsigned_act ? (int)xrow[k] : (int)(int8_t)xrow[k];
                    s += xv * (int32_t)w[pack_index(k, c, kp)];
                }
                acc += (float)s * xsc[b];
            }
            yt[c] = acc * scale[c];
        }
    }
}

/* ----------------------------------------------------------------------------------------------- SIMD kernels */

#if CML_X86
#include "kernels_x86.inc"
#elif CML_ARM64
#include "kernels_neon.inc"
#endif

/* ----------------------------------------------------------------------------------------------- public dispatch */

CML_API void cml_gemm_f32(const int8_t* q, const float* scale, int K, int N, const float* x, int ldx, int n, float* y,
                          int ldy, int c0, int c1) {
    if (n <= 0 || c1 <= c0) return;
    switch (cml_isa()) {
#if CML_X86
        case CML_ISA_AVX512: case CML_ISA_AVX512_VNNI: gemm_f32_avx512(q, scale, K, N, x, ldx, n, y, ldy, c0, c1); return;
        case CML_ISA_AVX2: case CML_ISA_AVX2_VNNI: gemm_f32_avx2(q, scale, K, N, x, ldx, n, y, ldy, c0, c1); return;
#elif CML_ARM64
        case CML_ISA_NEON: case CML_ISA_NEON_DOTPROD: gemm_f32_neon(q, scale, K, N, x, ldx, n, y, ldy, c0, c1); return;
#endif
        default: gemm_f32_scalar(q, scale, K, N, x, ldx, n, y, ldy, c0, c1); return;
    }
}

CML_API void cml_gemv_f32_partial(const int8_t* q, int K, int N, const float* x, int k0, int k1, float* acc) {
    if (k1 <= k0) return;
    switch (cml_isa()) {
#if CML_X86
        case CML_ISA_AVX512: case CML_ISA_AVX512_VNNI: gemv_f32_partial_avx512(q, K, N, x, k0, k1, acc); return;
        case CML_ISA_AVX2: case CML_ISA_AVX2_VNNI: gemv_f32_partial_avx2(q, K, N, x, k0, k1, acc); return;
#elif CML_ARM64
        case CML_ISA_NEON: case CML_ISA_NEON_DOTPROD: gemv_f32_partial_neon(q, K, N, x, k0, k1, acc); return;
#endif
        default: gemv_f32_partial_scalar(q, K, N, x, k0, k1, acc); return;
    }
}

CML_API void cml_qact(const float* x, int ldx, int n, int K, void* xq) {
    const size_t stride = cml_qact_stride(K);
    const int uns = cml_qact_unsigned();
    for (int t = 0; t < n; t++) {
        uint8_t* row = (uint8_t*)xq + (size_t)t * stride;
        const float* xt = x + (size_t)t * ldx;
        switch (cml_isa()) {
#if CML_X86
            case CML_ISA_AVX512: case CML_ISA_AVX512_VNNI: qact_row_avx512(xt, K, row, qact_scales(row, K), uns); break;
            case CML_ISA_AVX2: case CML_ISA_AVX2_VNNI: qact_row_avx2(xt, K, row, qact_scales(row, K), uns); break;
#elif CML_ARM64
            case CML_ISA_NEON: case CML_ISA_NEON_DOTPROD: qact_row_generic_neon(xt, K, row, qact_scales(row, K), uns); break;
#endif
            default: qact_row_generic_scalar(xt, K, row, qact_scales(row, K), uns); break;
        }
    }
}

CML_API void cml_gemm_q8(const void* packed, const float* scale, int K, int N, const void* xq, int t0, int t1, float* y,
                         int ldy, int c0, int c1) {
    if (t1 <= t0 || c1 <= c0) return;
    switch (cml_isa()) {
#if CML_X86
        case CML_ISA_AVX512_VNNI: gemm_q8_avx512vnni(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1); return;
        case CML_ISA_AVX512: gemm_q8_avx512(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1); return;
#if defined(CML_HAVE_AVXVNNI)
        case CML_ISA_AVX2_VNNI: gemm_q8_avx2vnni(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1); return;
#endif
        case CML_ISA_AVX2: gemm_q8_avx2(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1); return;
#elif CML_ARM64
        case CML_ISA_NEON_DOTPROD: gemm_q8_neon_dot(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1); return;
        case CML_ISA_NEON: gemm_q8_neon(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1); return;
#endif
        default: gemm_q8_scalar(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1, cml_qact_unsigned()); return;
    }
}

#define SMALL_DISPATCH(CALL_SUFFIXED)                                                                                \
    switch (cml_isa()) {                                                                                             \
        SMALL_CASES(CALL_SUFFIXED)                                                                                   \
        default: CALL_SUFFIXED(scalar); return;                                                                      \
    }
#if CML_X86
#define SMALL_CASES(C) case CML_ISA_AVX512: case CML_ISA_AVX512_VNNI: C(avx512); return; \
                       case CML_ISA_AVX2: case CML_ISA_AVX2_VNNI: C(avx2); return;
#elif CML_ARM64
#define SMALL_CASES(C) case CML_ISA_NEON: case CML_ISA_NEON_DOTPROD: C(neon); return;
#else
#define SMALL_CASES(C)
#endif

CML_API void cml_rmsnorm(const float* x, const float* w, int d, float eps, float* out) {
#define C_(S) rmsnorm_##S(x, w, d, eps, out)
    SMALL_DISPATCH(C_)
#undef C_
}

CML_API void cml_silu_mul(float* g, const float* u, int n) {
#define C_(S) silu_mul_##S(g, u, n)
    SMALL_DISPATCH(C_)
#undef C_
}

CML_API void cml_softmax(float* s, int n) {
#define C_(S) softmax_##S(s, n)
    SMALL_DISPATCH(C_)
#undef C_
}

CML_API void cml_attn_head(const float* q, int hd, float scale, const float* kT, int ldk, const float* v, int ldv, int len,
                           float* sc, float* out) {
#define C_(S) attn_head_##S(q, hd, scale, kT, ldk, v, ldv, len, sc, out)
    SMALL_DISPATCH(C_)
#undef C_
}

/* ----------------------------------------------------------------------------------------------- self-test */

static void* amalloc(size_t n) {
    n = (n + 63) & ~(size_t)63;
#if defined(_WIN32)
    return _aligned_malloc(n, 64);
#else
    void* p = NULL;
    return posix_memalign(&p, 64, n) == 0 ? p : NULL;
#endif
}
static void afree(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}
static float rndf(void) { return (float)(rnd() & 0xFFFF) / 32768.0f - 1.0f; }

static int close_enough(const float* a, const float* b, int n, float tol) {
    float maxerr = 0.0f, maxabs = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]); if (e > maxerr) maxerr = e;
        float m = fabsf(a[i]); if (m > maxabs) maxabs = m;
        if (a[i] != a[i] || b[i] != b[i]) return 0;
    }
    return maxerr <= tol * (maxabs > 1.0f ? maxabs : 1.0f);
}

/* One shape: compares the selected level with the scalar reference. Returns 0 or a check id. */
static int selftest_shape(int K, int N, int n, int level) {
    int rc = 0;
    int8_t* q = (int8_t*)malloc((size_t)K * N);
    float* scale = (float*)malloc(sizeof(float) * N);
    size_t xsz = (size_t)n * K > 4 * (size_t)K ? (size_t)n * K : 4 * (size_t)K;      /* small-op tests read x[0 .. 3K) */
    size_t ysz = (size_t)n * N > 2 * (size_t)K + 2048 ? (size_t)n * N : 2 * (size_t)K + 2048;
    float* x = (float*)malloc(sizeof(float) * xsz);
    float* y1 = (float*)malloc(sizeof(float) * ysz);
    float* y2 = (float*)malloc(sizeof(float) * ysz);
    void* packed = amalloc(cml_pack_size(K, N));
    void* xq = amalloc(cml_qact_size(n, K));
    if (!q || !scale || !x || !y1 || !y2 || !packed || !xq) { rc = 99; goto done; }
    for (size_t i = 0; i < (size_t)K * N; i++) { int v = (int)(rnd() % 255) - 127; q[i] = (int8_t)v; }
    for (int c = 0; c < N; c++) scale[c] = 0.5f + 0.5f * fabsf(rndf());
    for (size_t i = 0; i < xsz; i++) x[i] = rndf() * 2.0f;
    const int c0 = (N > 40) ? 7 : 0, c1 = N - (N > 40 ? 5 : 0);

    /* gemm f32, two column ranges */
    cml_set_isa(CML_ISA_SCALAR);
    gemm_f32_scalar(q, scale, K, N, x, K, n, y1, N, 0, N);
    cml_set_isa(level);
    memset(y2, 0, sizeof(float) * (size_t)n * N);
    cml_gemm_f32(q, scale, K, N, x, K, n, y2, N, 0, c0);
    cml_gemm_f32(q, scale, K, N, x, K, n, y2, N, c0, c1);
    cml_gemm_f32(q, scale, K, N, x, K, n, y2, N, c1, N);
    if (!close_enough(y1, y2, n * N, 1e-4f)) { rc = 1; goto done; }

    /* gemv partial: two K ranges, unscaled */
    for (int c = 0; c < N; c++) { y1[c] = 0.0f; y2[c] = 0.0f; }
    gemv_f32_partial_scalar(q, K, N, x, 0, K, y1);
    cml_gemv_f32_partial(q, K, N, x, 0, K / 3, y2);
    cml_gemv_f32_partial(q, K, N, x, K / 3, K, y2);
    if (!close_enough(y1, y2, N, 1e-4f)) { rc = 2; goto done; }

    /* quantised path: pack + qact + gemm_q8 vs the scalar reference on the same quantised data */
    cml_pack(q, K, N, packed);
    cml_qact(x, K, n, K, xq);
    gemm_q8_scalar(packed, scale, K, N, xq, 0, n, y1, N, 0, N, cml_qact_unsigned());
    memset(y2, 0, sizeof(float) * (size_t)n * N);
    cml_gemm_q8(packed, scale, K, N, xq, 0, n, y2, N, 0, c0);
    cml_gemm_q8(packed, scale, K, N, xq, 0, n, y2, N, c0, c1);
    cml_gemm_q8(packed, scale, K, N, xq, 0, n, y2, N, c1, N);
    if (!close_enough(y1, y2, n * N, 1e-4f)) { rc = 3; goto done; }
    /* a token sub-range */
    if (n > 3) {
        cml_gemm_q8(packed, scale, K, N, xq, 1, n - 1, y2 + N, N, 0, N);
        if (!close_enough(y1 + N, y2 + N, (n - 2) * N, 1e-4f)) { rc = 4; goto done; }
    }
    /* the quantised path must be close to the float one (quantisation noise) */
    gemm_f32_scalar(q, scale, K, N, x, K, n, y2, N, 0, N);
    if (!close_enough(y1, y2, n * N, 3e-2f)) { rc = 5; goto done; }

    /* small ops */
    {
        int d = K;
        cml_set_isa(CML_ISA_SCALAR);
        rmsnorm_scalar(x, scale, d < N ? d : N, 1e-5f, y1);
        cml_set_isa(level);
        cml_rmsnorm(x, scale, d < N ? d : N, 1e-5f, y2);
        if (!close_enough(y1, y2, d < N ? d : N, 1e-5f)) { rc = 6; goto done; }
        memcpy(y1, x, sizeof(float) * K); memcpy(y2, x, sizeof(float) * K);
        silu_mul_scalar(y1, x + K, K);
        cml_silu_mul(y2, x + K, K);
        if (!close_enough(y1, y2, K, 1e-5f)) { rc = 7; goto done; }
        memcpy(y1, x, sizeof(float) * K); memcpy(y2, x, sizeof(float) * K);
        softmax_scalar(y1, K);
        cml_softmax(y2, K);
        if (!close_enough(y1, y2, K, 1e-5f)) { rc = 8; goto done; }
        /* attention: hd = 32, len = n, kT [hd][len] from x, v [len][hd] from x */
        int hd = 32, len = n;
        if (K >= hd * len && K >= len * hd) {
            attn_head_scalar(x, hd, 0.17f, x + K, len, x + 2 * K, hd, len, y1 + 1000, y1);
            cml_attn_head(x, hd, 0.17f, x + K, len, x + 2 * K, hd, len, y2 + 1000, y2);
            if (!close_enough(y1, y2, hd, 1e-5f)) { rc = 9; goto done; }
        }
    }
done:
    free(q); free(scale); free(x); free(y1); free(y2); afree(packed); afree(xq);
    return rc;
}

CML_API int cml_selftest(void) {
    int level = cml_isa();
    rng_state = 0x9E3779B97F4A7C15ull;
    static const int shapes[][3] = { {384, 1536, 29}, {200, 301, 7}, {1536, 384, 13}, {96, 128, 1}, {32, 50, 2}, {512, 64, 3} };
    int rc = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]) && rc == 0; i++) {
        rc = selftest_shape(shapes[i][0], shapes[i][1], shapes[i][2], level);
        if (rc) rc += 10 * (int)(i + 1);
    }
    cml_set_isa(level);
    return rc;
}
