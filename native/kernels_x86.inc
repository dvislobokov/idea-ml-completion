/*
 * x86-64 kernels: AVX2+FMA, AVX-VNNI (VEX), AVX-512 (F/BW/DQ/VL), AVX-512 VNNI. Included from kernels.c.
 *
 * GEMM micro-kernels: register tile of MR tokens x (NRV vectors of columns); the weight row slice is loaded once per
 * input k (int8 -> f32 on the fly for the f32 path) and broadcast activations feed MR*NRV FMAs. K is processed in
 * blocks of KC so that the weight panel of a column tile (KC x NR bytes) and the activation tile stay in L1/L2.
 * The scale is applied in a final pass over the output tile. Tails: tokens via smaller MR instantiations, columns
 * (< NR) via the scalar kernel.
 *
 * Quantised path: packed weights [Kp/4][Np][4]; one vpdpbusd (u8 activations x s8 weights, 4 inputs per lane) does
 * 64 MACs per zmm. The +128 offset of the activations is undone with the per-block column sums stored by cml_pack.
 * Without VNNI (plain AVX2) the classic sign trick: |w| (u8) x sign(w)*x (s8) through vpmaddubsw + vpmaddwd — the
 * pair sums are <= 2*127*127 so vpmaddubsw cannot saturate.
 */

#define ATTR_AVX2 __attribute__((target("avx2,fma")))
#define ATTR_AVX2VNNI __attribute__((target("avx2,fma,avxvnni")))
#if defined(__clang__)   /* clang >= 18 wants the 512-bit vector width feature spelled out */
#define ATTR_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,evex512")))
#define ATTR_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,evex512")))
#else
#define ATTR_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
#define ATTR_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
#endif

#define KC_F32 512
/* column block of the drivers: a weight panel of Kp x CML_NC bytes stays in L2 across the token tiles */
#define CML_NC 256

/* ================================================================================================ AVX-512 f32 */

#define MR512 12

ATTR_AVX512 CML_INLINE void tile_f32_512(const int8_t* CML_RESTRICT q, int N, int k0, int k1, const float* CML_RESTRICT x,
                                         int ldx, float* CML_RESTRICT y, int ldy, int c, const int MR, int first) {
    __m512 acc[MR512][2];
    CML_UNROLL for (int t = 0; t < MR; t++) {
        if (first) { acc[t][0] = _mm512_setzero_ps(); acc[t][1] = _mm512_setzero_ps(); }
        else { acc[t][0] = _mm512_loadu_ps(y + (size_t)t * ldy + c); acc[t][1] = _mm512_loadu_ps(y + (size_t)t * ldy + c + 16); }
    }
    const int8_t* row = q + (size_t)k0 * N + c;
    const float* xk = x + k0;
    for (int k = k0; k < k1; k++, row += N, xk++) {
        __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)row)));
        __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(row + 16))));
        CML_UNROLL for (int t = 0; t < MR; t++) {
            __m512 a = _mm512_set1_ps(xk[(size_t)t * ldx]);
            acc[t][0] = _mm512_fmadd_ps(w0, a, acc[t][0]);
            acc[t][1] = _mm512_fmadd_ps(w1, a, acc[t][1]);
        }
    }
    CML_UNROLL for (int t = 0; t < MR; t++) {
        _mm512_storeu_ps(y + (size_t)t * ldy + c, acc[t][0]);
        _mm512_storeu_ps(y + (size_t)t * ldy + c + 16, acc[t][1]);
    }
}

ATTR_AVX512 CML_NOINLINE static void tile_f32_512_mr(const int8_t* q, int N, int k0, int k1, const float* x, int ldx,
                                                     float* y, int ldy, int c, int mr, int first) {
    switch (mr) {
        case 12: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 12, first); break;
        case 11: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 11, first); break;
        case 10: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 10, first); break;
        case 9: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 9, first); break;
        case 8: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 8, first); break;
        case 7: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 7, first); break;
        case 6: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 6, first); break;
        case 5: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 5, first); break;
        case 4: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 4, first); break;
        case 3: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 3, first); break;
        case 2: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 2, first); break;
        default: tile_f32_512(q, N, k0, k1, x, ldx, y, ldy, c, 1, first); break;
    }
}

/* Generic driver shared by the f32 tiles: NR-column tiles, scalar column tail, scale pass. */
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

GEN_GEMM_F32_DRIVER(gemm_f32_avx512, ATTR_AVX512, 32, MR512, tile_f32_512_mr)

/* K-split GEMV: for every row k the whole width of acc is updated; 4 rows per pass so that acc is loaded/stored
 * once per 4 MACs per element (memory-bound anyway: this streams the weights once). */
ATTR_AVX512 static void gemv_f32_partial_avx512(const int8_t* CML_RESTRICT q, int K, int N, const float* CML_RESTRICT x,
                                                int k0, int k1, float* CML_RESTRICT acc) {
    (void)K;
    int k = k0;
    for (; k + 4 <= k1; k += 4) {
        const int8_t* r0 = q + (size_t)k * N; const int8_t* r1 = r0 + N; const int8_t* r2 = r1 + N; const int8_t* r3 = r2 + N;
        __m512 a0 = _mm512_set1_ps(x[k]), a1 = _mm512_set1_ps(x[k + 1]), a2 = _mm512_set1_ps(x[k + 2]), a3 = _mm512_set1_ps(x[k + 3]);
        int c = 0;
        for (; c + 16 <= N; c += 16) {
            __m512 s = _mm512_loadu_ps(acc + c);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(r0 + c)))), a0, s);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(r1 + c)))), a1, s);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(r2 + c)))), a2, s);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(r3 + c)))), a3, s);
            _mm512_storeu_ps(acc + c, s);
        }
        if (c < N) {
            __mmask16 m = (__mmask16)((1u << (N - c)) - 1);
            __m512 s = _mm512_maskz_loadu_ps(m, acc + c);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, r0 + c))), a0, s);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, r1 + c))), a1, s);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, r2 + c))), a2, s);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, r3 + c))), a3, s);
            _mm512_mask_storeu_ps(acc + c, m, s);
        }
    }
    for (; k < k1; k++) {
        const int8_t* r0 = q + (size_t)k * N; __m512 a0 = _mm512_set1_ps(x[k]);
        int c = 0;
        for (; c + 16 <= N; c += 16) {
            __m512 s = _mm512_loadu_ps(acc + c);
            s = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(r0 + c)))), a0, s);
            _mm512_storeu_ps(acc + c, s);
        }
        for (; c < N; c++) acc[c] += r0[c] * x[k];
    }
}

/* ================================================================================================ AVX2 f32 */

#define MR256 6

ATTR_AVX2 CML_INLINE void tile_f32_256(const int8_t* CML_RESTRICT q, int N, int k0, int k1, const float* CML_RESTRICT x,
                                       int ldx, float* CML_RESTRICT y, int ldy, int c, const int MR, int first) {
    __m256 acc[6][2];
    CML_UNROLL for (int t = 0; t < MR; t++) {
        if (first) { acc[t][0] = _mm256_setzero_ps(); acc[t][1] = _mm256_setzero_ps(); }
        else { acc[t][0] = _mm256_loadu_ps(y + (size_t)t * ldy + c); acc[t][1] = _mm256_loadu_ps(y + (size_t)t * ldy + c + 8); }
    }
    const int8_t* row = q + (size_t)k0 * N + c;
    const float* xk = x + k0;
    for (int k = k0; k < k1; k++, row += N, xk++) {
        __m128i raw = _mm_loadu_si128((const __m128i*)row);
        __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw));
        __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(raw, 8)));
        CML_UNROLL for (int t = 0; t < MR; t++) {
            __m256 a = _mm256_broadcast_ss(xk + (size_t)t * ldx);
            acc[t][0] = _mm256_fmadd_ps(w0, a, acc[t][0]);
            acc[t][1] = _mm256_fmadd_ps(w1, a, acc[t][1]);
        }
    }
    CML_UNROLL for (int t = 0; t < MR; t++) {
        _mm256_storeu_ps(y + (size_t)t * ldy + c, acc[t][0]);
        _mm256_storeu_ps(y + (size_t)t * ldy + c + 8, acc[t][1]);
    }
}

ATTR_AVX2 CML_NOINLINE static void tile_f32_256_mr(const int8_t* q, int N, int k0, int k1, const float* x, int ldx, float* y,
                                                   int ldy, int c, int mr, int first) {
    switch (mr) {
        case 6: tile_f32_256(q, N, k0, k1, x, ldx, y, ldy, c, 6, first); break;
        case 5: tile_f32_256(q, N, k0, k1, x, ldx, y, ldy, c, 5, first); break;
        case 4: tile_f32_256(q, N, k0, k1, x, ldx, y, ldy, c, 4, first); break;
        case 3: tile_f32_256(q, N, k0, k1, x, ldx, y, ldy, c, 3, first); break;
        case 2: tile_f32_256(q, N, k0, k1, x, ldx, y, ldy, c, 2, first); break;
        default: tile_f32_256(q, N, k0, k1, x, ldx, y, ldy, c, 1, first); break;
    }
}

GEN_GEMM_F32_DRIVER(gemm_f32_avx2, ATTR_AVX2, 16, MR256, tile_f32_256_mr)

ATTR_AVX2 static void gemv_f32_partial_avx2(const int8_t* CML_RESTRICT q, int K, int N, const float* CML_RESTRICT x, int k0,
                                            int k1, float* CML_RESTRICT acc) {
    (void)K;
    int k = k0;
    for (; k + 4 <= k1; k += 4) {
        const int8_t* r0 = q + (size_t)k * N; const int8_t* r1 = r0 + N; const int8_t* r2 = r1 + N; const int8_t* r3 = r2 + N;
        __m256 a0 = _mm256_set1_ps(x[k]), a1 = _mm256_set1_ps(x[k + 1]), a2 = _mm256_set1_ps(x[k + 2]), a3 = _mm256_set1_ps(x[k + 3]);
        int c = 0;
        for (; c + 8 <= N; c += 8) {
            __m256 s = _mm256_loadu_ps(acc + c);
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r0 + c)))), a0, s);
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r1 + c)))), a1, s);
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r2 + c)))), a2, s);
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r3 + c)))), a3, s);
            _mm256_storeu_ps(acc + c, s);
        }
        for (; c < N; c++) acc[c] += r0[c] * x[k] + r1[c] * x[k + 1] + r2[c] * x[k + 2] + r3[c] * x[k + 3];
    }
    for (; k < k1; k++) {
        const int8_t* r0 = q + (size_t)k * N; __m256 a0 = _mm256_set1_ps(x[k]);
        int c = 0;
        for (; c + 8 <= N; c += 8) {
            __m256 s = _mm256_loadu_ps(acc + c);
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(r0 + c)))), a0, s);
            _mm256_storeu_ps(acc + c, s);
        }
        for (; c < N; c++) acc[c] += r0[c] * x[k];
    }
}

/* ================================================================================================ quantised path */

/* Common driver: tiles of MR tokens x NR columns over all K blocks; the tile accumulates in float registers (or in
 * y between blocks, see the tile), then the scale pass; column tail via the scalar reference. */
#define GEN_GEMM_Q8_DRIVER(NAME, ATTR, NR, MRMAX, TILE)                                                               \
    ATTR static void NAME(const void* packed, const float* scale, int K, int N, const void* xq, int t0, int t1, float* y, \
                          int ldy, int c0, int c1) {                                                                 \
        const int kp = kp_of(K), np = np_of(N), nb = kp / CML_KB;                                                     \
        const size_t xs = cml_qact_stride(K);                                                                        \
        const int8_t* w = (const int8_t*)packed;                                                                     \
        const int32_t* negsums = pack_negsums(packed, K, N);                                                          \
        int cs = (c0 + 15) & ~15; if (cs > c1) cs = c1;                                                               \
        int cfull = cs + (c1 - cs) / NR * NR;                                                                         \
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
        if (cfull < c1) gemm_q8_scalar(packed, scale, K, N, xq, t0, t1, y, ldy, cfull, c1, cml_qact_unsigned()); \
    }

/* ---- AVX-512 VNNI: MR x 32 columns, int32 accumulators per block, float accumulators across blocks (registers) */
#define MRV512 6

ATTR_AVX512VNNI CML_INLINE void tile_q8_512vnni(const int8_t* CML_RESTRICT w, const int32_t* CML_RESTRICT negsums, int np, int nb,
                                                int K, const uint8_t* CML_RESTRICT xrow, size_t xs, float* CML_RESTRICT y,
                                                int ldy, int c, const int MR) {
    (void)np;
    __m512 facc[MRV512][2];
    CML_UNROLL for (int t = 0; t < MR; t++) { facc[t][0] = _mm512_setzero_ps(); facc[t][1] = _mm512_setzero_ps(); }
    const int kp = kp_of(K);
    for (int b = 0; b < nb; b++) {
        __m512i acc[MRV512][2];
        __m512i ns0 = _mm512_loadu_si512((const void*)(negsums + (size_t)b * np + c));
        __m512i ns1 = _mm512_loadu_si512((const void*)(negsums + (size_t)b * np + c + 16));
        CML_UNROLL for (int t = 0; t < MR; t++) { acc[t][0] = ns0; acc[t][1] = ns1; }
        const size_t panel = (size_t)(kp / 4) * 64;              /* bytes per 16-column panel */
        const int8_t* wp = w + (size_t)(c / 16) * panel + (size_t)(b * CML_KB / 4) * 64;
        const uint8_t* xp = xrow + b * CML_KB;
        for (int j = 0; j < CML_KB / 4; j++, wp += 64, xp += 4) {
            __m512i w0 = _mm512_loadu_si512((const void*)wp);
            __m512i w1 = _mm512_loadu_si512((const void*)(wp + panel));
            CML_UNROLL for (int t = 0; t < MR; t++) {
                __m512i a = _mm512_set1_epi32(*(const int32_t*)(xp + (size_t)t * xs));
                acc[t][0] = _mm512_dpbusd_epi32(acc[t][0], a, w0);
                acc[t][1] = _mm512_dpbusd_epi32(acc[t][1], a, w1);
            }
        }
        CML_UNROLL for (int t = 0; t < MR; t++) {
            __m512 s = _mm512_set1_ps(((const float*)(xrow + (size_t)t * xs + kp))[b]);
            facc[t][0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[t][0]), s, facc[t][0]);
            facc[t][1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[t][1]), s, facc[t][1]);
        }
    }
    CML_UNROLL for (int t = 0; t < MR; t++) {
        _mm512_storeu_ps(y + (size_t)t * ldy + c, facc[t][0]);
        _mm512_storeu_ps(y + (size_t)t * ldy + c + 16, facc[t][1]);
    }
}

ATTR_AVX512VNNI CML_NOINLINE static void tile_q8_512vnni_mr(const int8_t* w, const int32_t* negsums, int np, int nb, int K,
                                                            const uint8_t* xrow, size_t xs, float* y, int ldy, int c, int mr) {
    switch (mr) {
        case 6: tile_q8_512vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 6); break;
        case 5: tile_q8_512vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 5); break;
        case 4: tile_q8_512vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 4); break;
        case 3: tile_q8_512vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 3); break;
        case 2: tile_q8_512vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 2); break;
        default: tile_q8_512vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 1); break;
    }
}

GEN_GEMM_Q8_DRIVER(gemm_q8_avx512vnni, ATTR_AVX512VNNI, 32, MRV512, tile_q8_512vnni_mr)

/* ---- AVX-512 without VNNI: emulate vpdpbusd with vpmaddubsw + vpmaddwd (u8 x s8; pairs can saturate at
 * 2*255*127, so the activations are signed here (sign trick like AVX2): |w| as u8, sign(w)*x as s8). The
 * selected-ISA signedness says this level quantises to signed bytes, hence negsums are not used. */
#define MRQ512 6

ATTR_AVX512 CML_INLINE void tile_q8_512(const int8_t* CML_RESTRICT w, const int32_t* CML_RESTRICT negsums, int np, int nb, int K,
                                        const uint8_t* CML_RESTRICT xrow, size_t xs, float* CML_RESTRICT y, int ldy, int c,
                                        const int MR) {
    (void)negsums; (void)np;
    __m512 facc[MRQ512][2];
    CML_UNROLL for (int t = 0; t < MR; t++) { facc[t][0] = _mm512_setzero_ps(); facc[t][1] = _mm512_setzero_ps(); }
    const int kp = kp_of(K);
    const __m512i ones = _mm512_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        __m512i acc[MRQ512][2];
        CML_UNROLL for (int t = 0; t < MR; t++) { acc[t][0] = _mm512_setzero_si512(); acc[t][1] = _mm512_setzero_si512(); }
        const size_t panel = (size_t)(kp / 4) * 64;              /* bytes per 16-column panel */
        const int8_t* wp = w + (size_t)(c / 16) * panel + (size_t)(b * CML_KB / 4) * 64;
        const uint8_t* xp = xrow + b * CML_KB;
        for (int j = 0; j < CML_KB / 4; j++, wp += 64, xp += 4) {
            __m512i w0 = _mm512_loadu_si512((const void*)wp);
            __m512i w1 = _mm512_loadu_si512((const void*)(wp + panel));
            __m512i a0 = _mm512_abs_epi8(w0), a1 = _mm512_abs_epi8(w1);
            /* sign(w) * x: x xor (w >> 7) minus (w >> 7)  ==  (x ^ m) - m with m = w < 0 ? -1 : 0 */
            __m512i m0 = _mm512_movm_epi8(_mm512_movepi8_mask(w0)), m1 = _mm512_movm_epi8(_mm512_movepi8_mask(w1));
            CML_UNROLL for (int t = 0; t < MR; t++) {
                __m512i xv = _mm512_set1_epi32(*(const int32_t*)(xp + (size_t)t * xs));
                __m512i x0 = _mm512_sub_epi8(_mm512_xor_si512(xv, m0), m0);
                __m512i x1 = _mm512_sub_epi8(_mm512_xor_si512(xv, m1), m1);
                acc[t][0] = _mm512_add_epi32(acc[t][0], _mm512_madd_epi16(_mm512_maddubs_epi16(a0, x0), ones));
                acc[t][1] = _mm512_add_epi32(acc[t][1], _mm512_madd_epi16(_mm512_maddubs_epi16(a1, x1), ones));
            }
        }
        CML_UNROLL for (int t = 0; t < MR; t++) {
            __m512 s = _mm512_set1_ps(((const float*)(xrow + (size_t)t * xs + kp))[b]);
            facc[t][0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[t][0]), s, facc[t][0]);
            facc[t][1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[t][1]), s, facc[t][1]);
        }
    }
    CML_UNROLL for (int t = 0; t < MR; t++) {
        _mm512_storeu_ps(y + (size_t)t * ldy + c, facc[t][0]);
        _mm512_storeu_ps(y + (size_t)t * ldy + c + 16, facc[t][1]);
    }
}

ATTR_AVX512 CML_NOINLINE static void tile_q8_512_mr(const int8_t* w, const int32_t* negsums, int np, int nb, int K,
                                                    const uint8_t* xrow, size_t xs, float* y, int ldy, int c, int mr) {
    switch (mr) {
        case 6: tile_q8_512(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 6); break;
        case 5: tile_q8_512(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 5); break;
        case 4: tile_q8_512(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 4); break;
        case 3: tile_q8_512(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 3); break;
        case 2: tile_q8_512(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 2); break;
        default: tile_q8_512(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 1); break;
    }
}

GEN_GEMM_Q8_DRIVER(gemm_q8_avx512, ATTR_AVX512, 32, MRQ512, tile_q8_512_mr)

/* ---- AVX2 (no VNNI): sign trick, MR x 16 columns, int32 accumulators per block, float accumulation in y. */
#define MRQ256 4

ATTR_AVX2 CML_INLINE void tile_q8_256(const int8_t* CML_RESTRICT w, const int32_t* CML_RESTRICT negsums, int np, int nb, int K,
                                      const uint8_t* CML_RESTRICT xrow, size_t xs, float* CML_RESTRICT y, int ldy, int c,
                                      const int MR) {
    (void)negsums; (void)np;
    const int kp = kp_of(K);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        __m256i acc[MRQ256][2];
        CML_UNROLL for (int t = 0; t < MR; t++) { acc[t][0] = _mm256_setzero_si256(); acc[t][1] = _mm256_setzero_si256(); }
        const size_t panel = (size_t)(kp / 4) * 64;
        const int8_t* wp = w + (size_t)(c / 16) * panel + (size_t)(b * CML_KB / 4) * 64;
        const uint8_t* xp = xrow + b * CML_KB;
        for (int j = 0; j < CML_KB / 4; j++, wp += 64, xp += 4) {
            __m256i w0 = _mm256_loadu_si256((const __m256i*)wp);
            __m256i w1 = _mm256_loadu_si256((const __m256i*)(wp + 32));
            __m256i a0 = _mm256_abs_epi8(w0), a1 = _mm256_abs_epi8(w1);
            CML_UNROLL for (int t = 0; t < MR; t++) {
                __m256i xv = _mm256_set1_epi32(*(const int32_t*)(xp + (size_t)t * xs));
                __m256i x0 = _mm256_sign_epi8(xv, w0), x1 = _mm256_sign_epi8(xv, w1);
                acc[t][0] = _mm256_add_epi32(acc[t][0], _mm256_madd_epi16(_mm256_maddubs_epi16(a0, x0), ones));
                acc[t][1] = _mm256_add_epi32(acc[t][1], _mm256_madd_epi16(_mm256_maddubs_epi16(a1, x1), ones));
            }
        }
        CML_UNROLL for (int t = 0; t < MR; t++) {
            __m256 s = _mm256_set1_ps(((const float*)(xrow + (size_t)t * xs + kp))[b]);
            float* yt = y + (size_t)t * ldy + c;
            __m256 f0 = _mm256_cvtepi32_ps(acc[t][0]), f1 = _mm256_cvtepi32_ps(acc[t][1]);
            if (b == 0) { _mm256_storeu_ps(yt, _mm256_mul_ps(f0, s)); _mm256_storeu_ps(yt + 8, _mm256_mul_ps(f1, s)); }
            else {
                _mm256_storeu_ps(yt, _mm256_fmadd_ps(f0, s, _mm256_loadu_ps(yt)));
                _mm256_storeu_ps(yt + 8, _mm256_fmadd_ps(f1, s, _mm256_loadu_ps(yt + 8)));
            }
        }
    }
}

ATTR_AVX2 CML_NOINLINE static void tile_q8_256_mr(const int8_t* w, const int32_t* negsums, int np, int nb, int K,
                                                  const uint8_t* xrow, size_t xs, float* y, int ldy, int c, int mr) {
    switch (mr) {
        case 4: tile_q8_256(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 4); break;
        case 3: tile_q8_256(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 3); break;
        case 2: tile_q8_256(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 2); break;
        default: tile_q8_256(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 1); break;
    }
}

GEN_GEMM_Q8_DRIVER(gemm_q8_avx2, ATTR_AVX2, 16, MRQ256, tile_q8_256_mr)

/* ---- AVX-VNNI (VEX vpdpbusd ymm): u8 activations with the negsum correction. UNTESTED here (no such CPU);
 * guarded by the self-test at load time. */
#if defined(CML_HAVE_AVXVNNI)
#define MRV256 6

ATTR_AVX2VNNI CML_INLINE void tile_q8_256vnni(const int8_t* CML_RESTRICT w, const int32_t* CML_RESTRICT negsums, int np, int nb,
                                              int K, const uint8_t* CML_RESTRICT xrow, size_t xs, float* CML_RESTRICT y, int ldy,
                                              int c, const int MR) {
    const int kp = kp_of(K);
    for (int b = 0; b < nb; b++) {
        __m256i acc[MRV256][2];
        __m256i ns0 = _mm256_loadu_si256((const __m256i*)(negsums + (size_t)b * np + c));
        __m256i ns1 = _mm256_loadu_si256((const __m256i*)(negsums + (size_t)b * np + c + 8));
        CML_UNROLL for (int t = 0; t < MR; t++) { acc[t][0] = ns0; acc[t][1] = ns1; }
        const size_t panel = (size_t)(kp / 4) * 64;
        const int8_t* wp = w + (size_t)(c / 16) * panel + (size_t)(b * CML_KB / 4) * 64;
        const uint8_t* xp = xrow + b * CML_KB;
        for (int j = 0; j < CML_KB / 4; j++, wp += 64, xp += 4) {
            __m256i w0 = _mm256_loadu_si256((const __m256i*)wp);
            __m256i w1 = _mm256_loadu_si256((const __m256i*)(wp + 32));
            CML_UNROLL for (int t = 0; t < MR; t++) {
                __m256i a = _mm256_set1_epi32(*(const int32_t*)(xp + (size_t)t * xs));
                acc[t][0] = _mm256_dpbusd_avx_epi32(acc[t][0], a, w0);
                acc[t][1] = _mm256_dpbusd_avx_epi32(acc[t][1], a, w1);
            }
        }
        CML_UNROLL for (int t = 0; t < MR; t++) {
            __m256 s = _mm256_set1_ps(((const float*)(xrow + (size_t)t * xs + kp))[b]);
            float* yt = y + (size_t)t * ldy + c;
            __m256 f0 = _mm256_cvtepi32_ps(acc[t][0]), f1 = _mm256_cvtepi32_ps(acc[t][1]);
            if (b == 0) { _mm256_storeu_ps(yt, _mm256_mul_ps(f0, s)); _mm256_storeu_ps(yt + 8, _mm256_mul_ps(f1, s)); }
            else {
                _mm256_storeu_ps(yt, _mm256_fmadd_ps(f0, s, _mm256_loadu_ps(yt)));
                _mm256_storeu_ps(yt + 8, _mm256_fmadd_ps(f1, s, _mm256_loadu_ps(yt + 8)));
            }
        }
    }
}

ATTR_AVX2VNNI CML_NOINLINE static void tile_q8_256vnni_mr(const int8_t* w, const int32_t* negsums, int np, int nb, int K,
                                                          const uint8_t* xrow, size_t xs, float* y, int ldy, int c, int mr) {
    switch (mr) {
        case 6: tile_q8_256vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 6); break;
        case 5: tile_q8_256vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 5); break;
        case 4: tile_q8_256vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 4); break;
        case 3: tile_q8_256vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 3); break;
        case 2: tile_q8_256vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 2); break;
        default: tile_q8_256vnni(w, negsums, np, nb, K, xrow, xs, y, ldy, c, 1); break;
    }
}

GEN_GEMM_Q8_DRIVER(gemm_q8_avx2vnni, ATTR_AVX2VNNI, 16, MRV256, tile_q8_256vnni_mr)
#endif

/* ================================================================================================ qact */

/* Per-block activation quantisation with explicit SIMD (the generic loop's lrintf does not vectorise). */
ATTR_AVX512 static void qact_row_avx512(const float* CML_RESTRICT x, int K, uint8_t* CML_RESTRICT out, float* CML_RESTRICT scales,
                                        int unsigned_out) {
    const int kp = kp_of(K);
    const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127), off = _mm512_set1_epi32(unsigned_out ? 128 : 0);
    for (int b = 0; b < kp / CML_KB; b++) {
        int k0 = b * CML_KB, k1 = k0 + CML_KB; if (k1 > K) k1 = K;
        __m512 mx = _mm512_setzero_ps();
        int k = k0;
        for (; k + 16 <= k1; k += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(x + k)));
        if (k < k1) { __mmask16 m = (__mmask16)((1u << (k1 - k)) - 1); mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_maskz_loadu_ps(m, x + k))); }
        float amax = _mm512_reduce_max_ps(mx);
        float s = amax / 127.0f, inv = s > 0.0f ? 1.0f / s : 0.0f;
        scales[b] = s;
        __m512 vinv = _mm512_set1_ps(inv);
        for (k = k0; k < k0 + CML_KB; k += 16) {
            __m512 v;
            if (k + 16 <= k1) v = _mm512_loadu_ps(x + k);
            else if (k < k1) v = _mm512_maskz_loadu_ps((__mmask16)((1u << (k1 - k)) - 1), x + k);
            else v = _mm512_setzero_ps();
            __m512i qv = _mm512_cvtps_epi32(_mm512_mul_ps(v, vinv));
            qv = _mm512_add_epi32(_mm512_min_epi32(_mm512_max_epi32(qv, lo), hi), off);
            _mm_storeu_si128((__m128i*)(out + k), _mm512_cvtepi32_epi8(qv));
        }
    }
}

ATTR_AVX2 static void qact_row_avx2(const float* CML_RESTRICT x, int K, uint8_t* CML_RESTRICT out, float* CML_RESTRICT scales,
                                    int unsigned_out) {
    const int kp = kp_of(K);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i xorv = _mm256_set1_epi8(unsigned_out ? (char)0x80 : 0);
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    float tmp[CML_KB];
    for (int b = 0; b < kp / CML_KB; b++) {
        int k0 = b * CML_KB, k1 = k0 + CML_KB; if (k1 > K) k1 = K;
        const float* xb;
        if (k1 - k0 == CML_KB) xb = x + k0;
        else { memcpy(tmp, x + k0, sizeof(float) * (size_t)(k1 - k0)); memset(tmp + (k1 - k0), 0, sizeof(float) * (size_t)(k0 + CML_KB - k1)); xb = tmp; }
        __m256 mx = _mm256_setzero_ps();
        for (int k = 0; k < CML_KB; k += 8) mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(xb + k), signmask));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float amax = _mm_cvtss_f32(m4);
        float s = amax / 127.0f, inv = s > 0.0f ? 1.0f / s : 0.0f;
        scales[b] = s;
        __m256 vinv = _mm256_set1_ps(inv);
        for (int k = 0; k < CML_KB; k += 32) {
            __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k), vinv));
            __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k + 8), vinv));
            __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k + 16), vinv));
            __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k + 24), vinv));
            q0 = _mm256_min_epi32(_mm256_max_epi32(q0, lo), hi); q1 = _mm256_min_epi32(_mm256_max_epi32(q1, lo), hi);
            q2 = _mm256_min_epi32(_mm256_max_epi32(q2, lo), hi); q3 = _mm256_min_epi32(_mm256_max_epi32(q3, lo), hi);
            __m256i p01 = _mm256_packs_epi32(q0, q1), p23 = _mm256_packs_epi32(q2, q3);   /* per-lane packing */
            __m256i p = _mm256_packs_epi16(p01, p23);                                       /* lanes: 0,8,16,24 | 4,12,20,28 ... */
            p = _mm256_permutevar8x32_epi32(p, perm);
            _mm256_storeu_si256((__m256i*)(out + k0 + k), _mm256_xor_si256(p, xorv));
        }
    }
}

/* ================================================================================================ small ops */

GEN_SMALL_OPS(avx2, ATTR_AVX2)
GEN_SMALL_OPS(avx512, ATTR_AVX512)
