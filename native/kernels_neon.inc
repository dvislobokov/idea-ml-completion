/*
 * aarch64 NEON kernels (macOS arm64 target). *** UNTESTED ***: no ARM machine was available; this compiles with
 * `zig cc -target aarch64-macos` and is guarded by cml_selftest() at load time (a failing self-test makes the Kotlin
 * side fall back to its scalar kernels). Included from kernels.c.
 *
 * f32 tile: 8 tokens x 8 columns; activations of 4 consecutive inputs are loaded as one float32x4 per token and fed
 * to vfmaq_laneq_f32 (one load per token per 4 inputs); weights int8 -> f32 via sxtl/scvtf.
 * q8 tile (FEAT_DotProd): 4 tokens x 16 columns, vdotq_laneq_s32 on the [Kp/4][Np][4] packing (s8 x s8, so no
 * offset correction); int32 accumulators per block, float accumulation in y.
 */

#define ATTR_NEON
#if defined(__clang__)
#define ATTR_DOTPROD __attribute__((target("dotprod")))
#else
#define ATTR_DOTPROD __attribute__((target("+dotprod")))
#endif

#define KC_F32 512
/* column block of the drivers: a weight panel of Kp x CML_NC bytes stays in L2 across the token tiles */
#define CML_NC 256
#define MRN 8

CML_INLINE void neon_w8(const int8_t* row, float32x4_t* w0, float32x4_t* w1) {
    int16x8_t s16 = vmovl_s8(vld1_s8(row));
    *w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16)));
    *w1 = vcvtq_f32_s32(vmovl_high_s16(s16));
}

CML_INLINE void tile_f32_neon(const int8_t* CML_RESTRICT q, int N, int k0, int k1, const float* CML_RESTRICT x, int ldx,
                              float* CML_RESTRICT y, int ldy, int c, const int MR, int first) {
    float32x4_t acc[MRN][2];
    CML_UNROLL for (int t = 0; t < MR; t++) {
        if (first) { acc[t][0] = vdupq_n_f32(0.0f); acc[t][1] = vdupq_n_f32(0.0f); }
        else { acc[t][0] = vld1q_f32(y + (size_t)t * ldy + c); acc[t][1] = vld1q_f32(y + (size_t)t * ldy + c + 4); }
    }
    const int8_t* row = q + (size_t)k0 * N + c;
    int k = k0;
    for (; k + 4 <= k1; k += 4, row += 4 * (size_t)N) {
        float32x4_t w00, w01, w10, w11, w20, w21, w30, w31;
        neon_w8(row, &w00, &w01); neon_w8(row + N, &w10, &w11); neon_w8(row + 2 * (size_t)N, &w20, &w21); neon_w8(row + 3 * (size_t)N, &w30, &w31);
        CML_UNROLL for (int t = 0; t < MR; t++) {
            float32x4_t xv = vld1q_f32(x + (size_t)t * ldx + k);
            acc[t][0] = vfmaq_laneq_f32(acc[t][0], w00, xv, 0); acc[t][1] = vfmaq_laneq_f32(acc[t][1], w01, xv, 0);
            acc[t][0] = vfmaq_laneq_f32(acc[t][0], w10, xv, 1); acc[t][1] = vfmaq_laneq_f32(acc[t][1], w11, xv, 1);
            acc[t][0] = vfmaq_laneq_f32(acc[t][0], w20, xv, 2); acc[t][1] = vfmaq_laneq_f32(acc[t][1], w21, xv, 2);
            acc[t][0] = vfmaq_laneq_f32(acc[t][0], w30, xv, 3); acc[t][1] = vfmaq_laneq_f32(acc[t][1], w31, xv, 3);
        }
    }
    for (; k < k1; k++, row += N) {
        float32x4_t w0, w1;
        neon_w8(row, &w0, &w1);
        CML_UNROLL for (int t = 0; t < MR; t++) {
            float a = x[(size_t)t * ldx + k];
            acc[t][0] = vfmaq_n_f32(acc[t][0], w0, a); acc[t][1] = vfmaq_n_f32(acc[t][1], w1, a);
        }
    }
    CML_UNROLL for (int t = 0; t < MR; t++) {
        vst1q_f32(y + (size_t)t * ldy + c, acc[t][0]);
        vst1q_f32(y + (size_t)t * ldy + c + 4, acc[t][1]);
    }
}

CML_NOINLINE static void tile_f32_neon_mr(const int8_t* q, int N, int k0, int k1, const float* x, int ldx, float* y, int ldy,
                                          int c, int mr, int first) {
    switch (mr) {
        case 8: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 8, first); break;
        case 7: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 7, first); break;
        case 6: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 6, first); break;
        case 5: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 5, first); break;
        case 4: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 4, first); break;
        case 3: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 3, first); break;
        case 2: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 2, first); break;
        default: tile_f32_neon(q, N, k0, k1, x, ldx, y, ldy, c, 1, first); break;
    }
}

#define GEN_GEMM_F32_DRIVER(NAME, ATTR, NR, MRMAX, TILE)                                                              \
    ATTR static void NAME(const int8_t* q, const float* scale, int K, int N, const float* x, int ldx, int n, float* y,  \
                          int ldy, int c0, int c1) {                                                                 \
        int cfull = c0 + (c1 - c0) / NR * NR;                                                                         \
        if (cfull > c0) {                                                                                             \
            for (int cb = c0; cb < cfull; cb += CML_NC) {                                                             \
                int ce = cb + CML_NC < cfull ? cb + CML_NC : cfull;                                                   \
                for (int k0 = 0; k0 < K; k0 += KC_F32) {                                                              \
                    int k1 = k0 + KC_F32 < K ? k0 + KC_F32 : K;                                                       \
                    for (int t = 0; t < n; t += MRMAX) {                                                              \
                        int mr = n - t < MRMAX ? n - t : MRMAX;                                                       \
                        for (int c = cb; c < ce; c += NR)                                                             \
                            TILE(q, N, k0, k1, x + (size_t)t * ldx, ldx, y + (size_t)t * ldy, ldy, c, mr, k0 == 0);   \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
            for (int t = 0; t < n; t++) {                                                                             \
                float* yt = y + (size_t)t * ldy;                                                                      \
                for (int c = c0; c < cfull; c++) yt[c] *= scale[c];                                                   \
            }                                                                                                         \
        }                                                                                                             \
        if (cfull < c1) gemm_f32_scalar(q, scale, K, N, x, ldx, n, y, ldy, cfull, c1);                               \
    }

GEN_GEMM_F32_DRIVER(gemm_f32_neon, ATTR_NEON, 8, MRN, tile_f32_neon_mr)

static void gemv_f32_partial_neon(const int8_t* CML_RESTRICT q, int K, int N, const float* CML_RESTRICT x, int k0, int k1,
                                  float* CML_RESTRICT acc) {
    (void)K;
    int k = k0;
    for (; k + 4 <= k1; k += 4) {
        const int8_t* r0 = q + (size_t)k * N; const int8_t* r1 = r0 + N; const int8_t* r2 = r1 + N; const int8_t* r3 = r2 + N;
        float a0 = x[k], a1 = x[k + 1], a2 = x[k + 2], a3 = x[k + 3];
        int c = 0;
        for (; c + 8 <= N; c += 8) {
            float32x4_t s0 = vld1q_f32(acc + c), s1 = vld1q_f32(acc + c + 4);
            float32x4_t w0, w1;
            neon_w8(r0 + c, &w0, &w1); s0 = vfmaq_n_f32(s0, w0, a0); s1 = vfmaq_n_f32(s1, w1, a0);
            neon_w8(r1 + c, &w0, &w1); s0 = vfmaq_n_f32(s0, w0, a1); s1 = vfmaq_n_f32(s1, w1, a1);
            neon_w8(r2 + c, &w0, &w1); s0 = vfmaq_n_f32(s0, w0, a2); s1 = vfmaq_n_f32(s1, w1, a2);
            neon_w8(r3 + c, &w0, &w1); s0 = vfmaq_n_f32(s0, w0, a3); s1 = vfmaq_n_f32(s1, w1, a3);
            vst1q_f32(acc + c, s0); vst1q_f32(acc + c + 4, s1);
        }
        for (; c < N; c++) acc[c] += r0[c] * a0 + r1[c] * a1 + r2[c] * a2 + r3[c] * a3;
    }
    for (; k < k1; k++) {
        const int8_t* r0 = q + (size_t)k * N; float a0 = x[k];
        for (int c = 0; c < N; c++) acc[c] += r0[c] * a0;
    }
}

/* ---- dot-product path */
#define MRD 4

ATTR_DOTPROD CML_INLINE void tile_q8_dot(const int8_t* CML_RESTRICT w, int np, int nb, int K, const uint8_t* CML_RESTRICT xrow,
                                         size_t xs, float* CML_RESTRICT y, int ldy, int c, const int MR) {
    const int kp = kp_of(K);
    for (int b = 0; b < nb; b++) {
        int32x4_t acc[MRD][4];
        CML_UNROLL for (int t = 0; t < MR; t++) for (int v = 0; v < 4; v++) acc[t][v] = vdupq_n_s32(0);
        const size_t panel = (size_t)(kp / 4) * 64;
        const int8_t* wp = w + (size_t)(c / 16) * panel + (size_t)(b * CML_KB / 4) * 64;
        const int8_t* xp = (const int8_t*)xrow + b * CML_KB;
        (void)np;
        for (int j = 0; j < CML_KB / 4; j += 4, wp += 4 * 64, xp += 16) {
            int8x16_t xv[MRD];
            CML_UNROLL for (int t = 0; t < MR; t++) xv[t] = vld1q_s8(xp + (size_t)t * xs);
#define STEP(L)                                                                                                       \
            {                                                                                                         \
                const int8_t* wl = wp + (size_t)(L) * 64;                                                             \
                int8x16_t w0 = vld1q_s8(wl), w1 = vld1q_s8(wl + 16), w2 = vld1q_s8(wl + 32), w3 = vld1q_s8(wl + 48);  \
                CML_UNROLL for (int t = 0; t < MR; t++) {                                                                        \
                    acc[t][0] = vdotq_laneq_s32(acc[t][0], w0, xv[t], L);                                             \
                    acc[t][1] = vdotq_laneq_s32(acc[t][1], w1, xv[t], L);                                             \
                    acc[t][2] = vdotq_laneq_s32(acc[t][2], w2, xv[t], L);                                             \
                    acc[t][3] = vdotq_laneq_s32(acc[t][3], w3, xv[t], L);                                             \
                }                                                                                                     \
            }
            STEP(0) STEP(1) STEP(2) STEP(3)
#undef STEP
        }
        CML_UNROLL for (int t = 0; t < MR; t++) {
            float s = ((const float*)(xrow + (size_t)t * xs + kp))[b];
            float* yt = y + (size_t)t * ldy + c;
            for (int v = 0; v < 4; v++) {
                float32x4_t f = vcvtq_f32_s32(acc[t][v]);
                if (b == 0) vst1q_f32(yt + 4 * v, vmulq_n_f32(f, s));
                else vst1q_f32(yt + 4 * v, vfmaq_n_f32(vld1q_f32(yt + 4 * v), f, s));
            }
        }
    }
}

ATTR_DOTPROD CML_NOINLINE static void tile_q8_dot_mr(const int8_t* w, const int32_t* negsums, int np, int nb, int K,
                                                     const uint8_t* xrow, size_t xs, float* y, int ldy, int c, int mr) {
    (void)negsums;
    switch (mr) {
        case 4: tile_q8_dot(w, np, nb, K, xrow, xs, y, ldy, c, 4); break;
        case 3: tile_q8_dot(w, np, nb, K, xrow, xs, y, ldy, c, 3); break;
        case 2: tile_q8_dot(w, np, nb, K, xrow, xs, y, ldy, c, 2); break;
        default: tile_q8_dot(w, np, nb, K, xrow, xs, y, ldy, c, 1); break;
    }
}

#define GEN_GEMM_Q8_DRIVER(NAME, ATTR, NR, MRMAX, TILE)                                                               \
    ATTR static void NAME(const void* packed, const float* scale, int K, int N, const void* xq, int t0, int t1, float* y, \
                          int ldy, int c0, int c1) {                                                                 \
        const int kp = kp_of(K), np = np_of(N), nb = kp / CML_KB;                                                     \
        const size_t xs = cml_qact_stride(K);                                                                        \
        const int8_t* w = (const int8_t*)packed;                                                                     \
        const int32_t* negsums = pack_negsums(packed, K, N);                                                          \
        int cs = (c0 + 15) & ~15; if (cs > c1) cs = c1;                                                               \
        int cfull = cs + (c1 - cs) / NR * NR;                                                                         \
        (void)kp;                                                                                                     \
        if (cs > c0) gemm_q8_scalar(packed, scale, K, N, xq, t0, t1, y, ldy, c0, cs, cml_qact_unsigned());            \
        for (int cb = cs; cb < cfull; cb += CML_NC) {                                                                 \
            int ce = cb + CML_NC < cfull ? cb + CML_NC : cfull;                                                       \
            for (int t = t0; t < t1; t += MRMAX) {                                                                    \
                int mr = t1 - t < MRMAX ? t1 - t : MRMAX;                                                             \
                const uint8_t* xrow = (const uint8_t*)xq + (size_t)t * xs;                                           \
                float* yt = y + (size_t)(t - t0) * ldy;                                                              \
                for (int c = cb; c < ce; c += NR) TILE(w, negsums, np, nb, K, xrow, xs, yt, ldy, c, mr);             \
                for (int tt = 0; tt < mr; tt++) for (int c = cb; c < ce; c++) yt[(size_t)tt * ldy + c] *= scale[c];  \
            }                                                                                                         \
        }                                                                                                             \
        if (cfull < c1) gemm_q8_scalar(packed, scale, K, N, xq, t0, t1, y, ldy, cfull, c1, cml_qact_unsigned());     \
    }

GEN_GEMM_Q8_DRIVER(gemm_q8_neon_dot, ATTR_DOTPROD, 16, MRD, tile_q8_dot_mr)

/* Without FEAT_DotProd an int8 NEON path is no faster than the f32 one; use the reference. */
static void gemm_q8_neon(const void* packed, const float* scale, int K, int N, const void* xq, int t0, int t1, float* y,
                         int ldy, int c0, int c1) {
    gemm_q8_scalar(packed, scale, K, N, xq, t0, t1, y, ldy, c0, c1, cml_qact_unsigned());
}

GEN_SMALL_OPS(neon, ATTR_NEON)
