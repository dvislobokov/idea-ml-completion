package io.github.completionml.core.nn

/**
 * The int8-weight × float-activation hot loops. Weights are `[in, out]` int8 with per-column scales (see [NnTensor.Q8]);
 * they live off-heap, so kernels copy row tiles into a per-worker heap `tile` (bulk ByteBuffer.get, ~memcpy speed)
 * and compute from there.
 *
 * Two implementations: [ScalarNnKernels] (plain loops shaped for C2 auto-vectorisation — the fallback that runs in any
 * JVM) and the native SIMD library (`nn.native.NativeNnKernels`, our own C kernels shipped in the jar, see [NnKernels.best]).
 * No jdk.incubator.vector: the JetBrains Runtime does not ship it and nothing may be required from the user.
 */
interface NnKernels {
    val name: String

    /**
     * Column-split matmul: `y[t][c] = scales[c] * Σ_i x[t][i] * q[i, c]` for t in 0 until n, c in [c0, c1).
     * Overwrites that column range of each `y[t]`.
     */
    fun matmulCols(w: NnTensor.Q8, x: Array<FloatArray>, n: Int, y: Array<FloatArray>, c0: Int, c1: Int, s: KernelScratch)

    /**
     * Row-split (K-split) matvec for decoding: `acc[c] += Σ_{i in [r0, r1)} x[i] * q[i, c]` for all columns, unscaled.
     * [acc] has at least `w.cols` elements and offset 0; the caller reduces the per-worker partials and applies scales.
     */
    fun matvecRowsPartial(w: NnTensor.Q8, x: FloatArray, r0: Int, r1: Int, acc: FloatArray, s: KernelScratch)

    /**
     * Optional whole-matmul path (the native kernels): computes `ys[m][t] = x[t] · ws[m]` for all m and t < n itself,
     * parallelising on [executor], and returns true; false means "not supported — split into [matmulCols] /
     * [matvecRowsPartial] tasks". All `ws` share the input width `x[t].size`.
     */
    fun matmul(ws: Array<NnTensor.Q8>, x: Array<FloatArray>, n: Int, ys: Array<Array<FloatArray>>, executor: NnExecutor): Boolean = false

    companion object {
        /** Byte tile size used by the kernels. */
        const val TILE_BYTES = 32 * 1024
        /** Column block of the prefill kernels. */
        const val CB = 256
        /** Float tile rows (CB columns each) of the scalar prefill kernel. */
        const val FROWS = 32
        /** Tokens per accumulator group in the scalar prefill kernel. */
        const val TOK_GROUP = 64
        /**
         * Preference: the native SIMD library (`nn.native.NativeNnKernels`, if it loads and passes its self-test;
         * `-Dcompletionml.nn.native=false` disables it), then the plain kernels. A native instance owns scratch memory:
         * one per model, closed with it.
         */
        fun best(): NnKernels = nativeOrNull() ?: ScalarNnKernels

        fun nativeOrNull(): NnKernels? = try {
            io.github.completionml.core.nn.native.NativeNnKernels.loadOrNull()
        } catch (_: Throwable) { null }
    }
}

/** Per-worker scratch buffers of the kernels (one instance per worker thread). */
class KernelScratch(maxCols: Int) {
    val tile = ByteArray(maxOf(NnKernels.TILE_BYTES, maxCols))
    /** dequantised weight rows (offset-0 arrays: C2 only vectorises float loops whose arrays are all at offset 0) */
    val wf = Array(NnKernels.FROWS) { FloatArray(NnKernels.CB) }
    /** per-token accumulators for one column block */
    val acc = Array(NnKernels.TOK_GROUP) { FloatArray(NnKernels.CB) }
}

/**
 * Plain loops. C2 (JDK 21) vectorises `y[yo + j] += q[qo + j] * a` with a byte source (different array types can't
 * alias), but NOT float→float loops with non-zero offsets, nor float reductions — hence the `[in, out]` weight layout
 * and the axpy formulation everywhere (no dot products in the hot loops).
 */
object ScalarNnKernels : NnKernels {
    override val name = "scalar"

    /**
     * Prefill: per column block (256) and token group (64): the weight rows are dequantised once into offset-0 float
     * rows `wf` (in FROWS-row tiles) and every token accumulates `acc[t][j] += Σ x[t][i] * wf[i][j]` in offset-0 float
     * arrays — the fastest form C2 vectorises (~18 GMAC/s/core on AVX-512 vs ~11 for byte→float on the fly).
     */
    override fun matmulCols(w: NnTensor.Q8, x: Array<FloatArray>, n: Int, y: Array<FloatArray>, c0: Int, c1: Int, s: KernelScratch) {
        val cols = w.cols; val rows = w.rows; val data = w.data
        val tile = s.tile; val wf = s.wf; val acc = s.acc; val sc = w.scales
        var cb = c0
        while (cb < c1) {
            val cw = minOf(NnKernels.CB, c1 - cb)
            var t0 = 0
            while (t0 < n) {
                val tn = minOf(NnKernels.TOK_GROUP, n - t0)
                for (t in 0 until tn) acc[t].fill(0f, 0, cw)
                var r = 0
                while (r < rows) {
                    val ih = minOf(NnKernels.FROWS, rows - r)
                    for (ii in 0 until ih) {
                        data.get((r + ii) * cols + cb, tile, 0, cw)
                        dequantRow(tile, wf[ii], cw)
                    }
                    for (t in 0 until tn) accumulateF(x[t0 + t], r, ih, wf, acc[t], cw)
                    r += ih
                }
                for (t in 0 until tn) { val yt = y[t0 + t]; val at = acc[t]; for (j in 0 until cw) yt[cb + j] = at[j] * sc[cb + j] }
                t0 += tn
            }
            cb += cw
        }
    }

    override fun matvecRowsPartial(w: NnTensor.Q8, x: FloatArray, r0: Int, r1: Int, acc: FloatArray, s: KernelScratch) {
        val cols = w.cols; val data = w.data; val tile = s.tile
        val ib = maxOf(1, tile.size / cols)
        var r = r0
        while (r < r1) {
            val ih = minOf(ib, r1 - r)
            data.get(r * cols, tile, 0, ih * cols)
            accumulateD(x, r, ih, tile, cols, acc)
            r += ih
        }
    }

    @JvmStatic fun dequantRow(q: ByteArray, out: FloatArray, n: Int) {
        for (j in 0 until n) out[j] = q[j].toFloat()
    }

    /** `acc[j] += Σ_ii x[r + ii] * wf[ii][j]`, 4 rows per pass. Big on purpose, see below. */
    @JvmStatic fun accumulateF(x: FloatArray, r: Int, ih: Int, wf: Array<FloatArray>, acc: FloatArray, cw: Int) {
        var ii = 0
        while (ii + 4 <= ih) {
            val ri = r + ii
            val a0 = x[ri]; val a1 = x[ri + 1]; val a2 = x[ri + 2]; val a3 = x[ri + 3]
            val w0 = wf[ii]; val w1 = wf[ii + 1]; val w2 = wf[ii + 2]; val w3 = wf[ii + 3]
            for (j in 0 until cw) acc[j] += w0[j] * a0 + w1[j] * a1 + w2[j] * a2 + w3[j] * a3
            ii += 4
        }
        if (ii + 3 == ih) {
            val a0 = x[r + ii]; val a1 = x[r + ii + 1]; val a2 = x[r + ii + 2]
            val w0 = wf[ii]; val w1 = wf[ii + 1]; val w2 = wf[ii + 2]
            for (j in 0 until cw) acc[j] += w0[j] * a0 + w1[j] * a1 + w2[j] * a2
            ii += 3
        } else if (ii + 2 <= ih) {
            val a0 = x[r + ii]; val a1 = x[r + ii + 1]
            val w0 = wf[ii]; val w1 = wf[ii + 1]
            for (j in 0 until cw) acc[j] += w0[j] * a0 + w1[j] * a1
            ii += 2
        }
        if (ii < ih) {
            val a = x[r + ii]; val w0 = wf[ii]
            for (j in 0 until cw) acc[j] += w0[j] * a
        }
    }

    // Decode inner loop: `y[j] += Σ_ii x[r + ii] * tile[ii * cw + j]`, 4 weight rows per pass (one load/store of y
    // per 4 multiply-adds; 8 rows are NOT vectorised by C2), byte source so C2 can vectorise despite the offsets.
    //
    // These methods are deliberately larger than C2's FreqInlineSize (325 bytecodes) so they are compiled as
    // standalone units: when inlined into the caller's loop nest, C2 (JDK 21) intermittently failed to vectorise the
    // j-loop (~1.6 vs ~12 GMAC/s per core, varying from run to run). Keep them big and called from one place each.

    @JvmStatic fun accumulateD(x: FloatArray, r: Int, ih: Int, q: ByteArray, cw: Int, y: FloatArray) {
        var ii = 0
        while (ii + 4 <= ih) {
            val ri = r + ii
            val a0 = x[ri]; val a1 = x[ri + 1]; val a2 = x[ri + 2]; val a3 = x[ri + 3]
            val o0 = ii * cw; val o1 = o0 + cw; val o2 = o1 + cw; val o3 = o2 + cw
            for (j in 0 until cw) y[j] += q[o0 + j] * a0 + q[o1 + j] * a1 + q[o2 + j] * a2 + q[o3 + j] * a3
            ii += 4
        }
        if (ii + 2 <= ih) {
            val a0 = x[r + ii]; val a1 = x[r + ii + 1]
            val o0 = ii * cw; val o1 = o0 + cw
            for (j in 0 until cw) y[j] += q[o0 + j] * a0 + q[o1 + j] * a1
            ii += 2
        }
        if (ii < ih) {
            val a = x[r + ii]; val o = ii * cw
            for (j in 0 until cw) y[j] += q[o + j] * a
        }
    }

    /** Generic tail helper for other kernels. */
    @JvmStatic fun axpyB(a: Float, q: ByteArray, qo: Int, y: FloatArray, yo: Int, n: Int) {
        for (j in 0 until n) y[yo + j] += q[qo + j] * a
    }
}

/** Float helpers with offset-0 arrays only, which C2 auto-vectorises. */
internal object Vec {
    /** `y[j] += a * x[j]` for j < n. */
    @JvmStatic fun axpy(a: Float, x: FloatArray, y: FloatArray, n: Int) {
        for (j in 0 until n) y[j] += a * x[j]
    }

    /** `y[j] += x[j]`. */
    @JvmStatic fun add(x: FloatArray, y: FloatArray, n: Int) {
        for (j in 0 until n) y[j] += x[j]
    }

    /** `y[j] *= s[j]`. */
    @JvmStatic fun mul(s: FloatArray, y: FloatArray, n: Int) {
        for (j in 0 until n) y[j] *= s[j]
    }
}
