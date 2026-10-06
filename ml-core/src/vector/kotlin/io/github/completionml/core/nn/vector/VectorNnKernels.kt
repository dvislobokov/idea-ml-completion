package io.github.completionml.core.nn.vector

import io.github.completionml.core.nn.KernelScratch
import io.github.completionml.core.nn.NnKernels
import io.github.completionml.core.nn.NnTensor
import io.github.completionml.core.nn.ScalarNnKernels
import jdk.incubator.vector.ByteVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorShape
import jdk.incubator.vector.VectorSpecies

// Static finals so that C2 constant-folds the species (essential for intrinsification).
private val FS: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED
private val L: Int = FS.length()
private val BS: VectorSpecies<Byte> = VectorSpecies.of(java.lang.Byte.TYPE, VectorShape.forBitSize(L * 8))

/**
 * jdk.incubator.vector kernels: int8 → float conversion + FMA with register-blocked accumulators (4 tokens × 2
 * vectors for prefill, 4 vectors for decode), so partial sums stay in registers across the whole row tile instead of
 * going through memory as in the auto-vectorised axpy loops. Compiled in the `vector` source set with
 * `-Xadd-modules=jdk.incubator.vector`; instantiated reflectively by [NnKernels.best] only when the module is present.
 */
class VectorNnKernels : NnKernels {
    init {
        // The Java fallback of the Vector API (no SIMD) is far slower than plain loops: refuse below AVX2 width.
        require(L >= 8) { "preferred float vector has $L lanes" }
    }

    override val name = "vector${L * 32}"

    override fun matmulCols(w: NnTensor.Q8, x: Array<FloatArray>, n: Int, y: Array<FloatArray>, c0: Int, c1: Int, sc: KernelScratch) {
        val cols = w.cols; val rows = w.rows; val data = w.data; val tile = sc.tile
        for (t in 0 until n) y[t].fill(0f, c0, c1)
        var cb = c0
        while (cb < c1) {
            val cw = minOf(CB, c1 - cb)
            val ib = maxOf(1, NnKernels.TILE_BYTES / cw)
            var r = 0
            while (r < rows) {
                val ih = minOf(ib, rows - r)
                for (ii in 0 until ih) data.get((r + ii) * cols + cb, tile, ii * cw, cw)
                var t = 0
                while (t + 4 <= n) { block4(x, y, t, r, ih, tile, cb, cw); t += 4 }
                while (t < n) { block1(x[t], y[t], r, ih, tile, cb, cw); t++ }
                r += ih
            }
            cb += cw
        }
        val s = w.scales
        for (t in 0 until n) {
            val yt = y[t]
            var c = c0
            while (c + L <= c1) { FloatVector.fromArray(FS, yt, c).mul(FloatVector.fromArray(FS, s, c)).intoArray(yt, c); c += L }
            while (c < c1) { yt[c] *= s[c]; c++ }
        }
    }

    /** 4 tokens × 2 vectors of columns, accumulators in registers across the tile rows. */
    private fun block4(x: Array<FloatArray>, y: Array<FloatArray>, t: Int, r: Int, ih: Int, tile: ByteArray, cb: Int, cw: Int) {
        val x0 = x[t]; val x1 = x[t + 1]; val x2 = x[t + 2]; val x3 = x[t + 3]
        val y0 = y[t]; val y1 = y[t + 1]; val y2 = y[t + 2]; val y3 = y[t + 3]
        var c = 0
        while (c + 2 * L <= cw) {
            val yc = cb + c
            var a00 = FloatVector.fromArray(FS, y0, yc); var a01 = FloatVector.fromArray(FS, y0, yc + L)
            var a10 = FloatVector.fromArray(FS, y1, yc); var a11 = FloatVector.fromArray(FS, y1, yc + L)
            var a20 = FloatVector.fromArray(FS, y2, yc); var a21 = FloatVector.fromArray(FS, y2, yc + L)
            var a30 = FloatVector.fromArray(FS, y3, yc); var a31 = FloatVector.fromArray(FS, y3, yc + L)
            var off = c
            for (ii in 0 until ih) {
                val w0 = b2f(tile, off); val w1 = b2f(tile, off + L)
                val ri = r + ii
                val b0 = FloatVector.broadcast(FS, x0[ri]); a00 = w0.fma(b0, a00); a01 = w1.fma(b0, a01)
                val b1 = FloatVector.broadcast(FS, x1[ri]); a10 = w0.fma(b1, a10); a11 = w1.fma(b1, a11)
                val b2 = FloatVector.broadcast(FS, x2[ri]); a20 = w0.fma(b2, a20); a21 = w1.fma(b2, a21)
                val b3 = FloatVector.broadcast(FS, x3[ri]); a30 = w0.fma(b3, a30); a31 = w1.fma(b3, a31)
                off += cw
            }
            a00.intoArray(y0, yc); a01.intoArray(y0, yc + L)
            a10.intoArray(y1, yc); a11.intoArray(y1, yc + L)
            a20.intoArray(y2, yc); a21.intoArray(y2, yc + L)
            a30.intoArray(y3, yc); a31.intoArray(y3, yc + L)
            c += 2 * L
        }
        if (c < cw) for (k in 0 until 4) tail(x[t + k], y[t + k], r, ih, tile, cb, cw, c)
    }

    /** 1 token × 4 vectors of columns. */
    private fun block1(xt: FloatArray, yt: FloatArray, r: Int, ih: Int, tile: ByteArray, cb: Int, cw: Int) {
        var c = 0
        while (c + 4 * L <= cw) {
            val yc = cb + c
            var a0 = FloatVector.fromArray(FS, yt, yc); var a1 = FloatVector.fromArray(FS, yt, yc + L)
            var a2 = FloatVector.fromArray(FS, yt, yc + 2 * L); var a3 = FloatVector.fromArray(FS, yt, yc + 3 * L)
            var off = c
            for (ii in 0 until ih) {
                val b = FloatVector.broadcast(FS, xt[r + ii])
                a0 = b2f(tile, off).fma(b, a0); a1 = b2f(tile, off + L).fma(b, a1)
                a2 = b2f(tile, off + 2 * L).fma(b, a2); a3 = b2f(tile, off + 3 * L).fma(b, a3)
                off += cw
            }
            a0.intoArray(yt, yc); a1.intoArray(yt, yc + L); a2.intoArray(yt, yc + 2 * L); a3.intoArray(yt, yc + 3 * L)
            c += 4 * L
        }
        while (c + L <= cw) {
            val yc = cb + c
            var a0 = FloatVector.fromArray(FS, yt, yc)
            var off = c
            for (ii in 0 until ih) { a0 = b2f(tile, off).fma(FloatVector.broadcast(FS, xt[r + ii]), a0); off += cw }
            a0.intoArray(yt, yc)
            c += L
        }
        if (c < cw) tail(xt, yt, r, ih, tile, cb, cw, c)
    }

    private fun tail(xt: FloatArray, yt: FloatArray, r: Int, ih: Int, tile: ByteArray, cb: Int, cw: Int, from: Int) {
        for (ii in 0 until ih) ScalarNnKernels.axpyB(xt[r + ii], tile, ii * cw + from, yt, cb + from, cw - from)
    }

    override fun matvecRowsPartial(w: NnTensor.Q8, x: FloatArray, r0: Int, r1: Int, acc: FloatArray, sc: KernelScratch) {
        val cols = w.cols; val data = w.data; val tile = sc.tile
        val ib = maxOf(1, tile.size / cols)
        var r = r0
        while (r < r1) {
            val ih = minOf(ib, r1 - r)
            data.get(r * cols, tile, 0, ih * cols)
            block1(x, acc, r, ih, tile, 0, cols)
            r += ih
        }
    }

    private companion object {
        const val CB = NnKernels.CB

        @JvmStatic fun b2f(a: ByteArray, off: Int): FloatVector =
            ByteVector.fromArray(BS, a, off).convertShape(VectorOperators.B2F, FS, 0) as FloatVector
    }
}
