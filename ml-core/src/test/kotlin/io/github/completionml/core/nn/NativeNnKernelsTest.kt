package io.github.completionml.core.nn

import io.github.completionml.core.nn.native.CmlNative
import io.github.completionml.core.nn.native.NativeLib
import io.github.completionml.core.nn.native.NativeNnKernels
import java.nio.ByteBuffer
import java.util.Random
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Native kernels vs the Kotlin ones. Skipped (with a message) when the library is not available for this platform;
 * `-Dcompletionml.nn.expectNative=true` turns that into a failure (CI on supported platforms).
 */
class NativeNnKernelsTest {
    private fun native(mode: NativeNnKernels.Mode): NativeNnKernels? {
        val k = NativeNnKernels.loadOrNull(mode)
        if (k == null) {
            val msg = "native kernels unavailable: ${NativeLib.status}"
            if (System.getProperty("completionml.nn.expectNative") == "true") throw AssertionError(msg)
            println("SKIP $msg")
        }
        return k
    }

    private fun maxRelErr(expected: FloatArray, actual: FloatArray, n: Int, off: Int = 0): Float {
        var maxErr = 0f; var maxAbs = 0f
        for (i in off until off + n) { maxErr = maxOf(maxErr, abs(expected[i] - actual[i])); maxAbs = maxOf(maxAbs, abs(expected[i])) }
        return maxErr / maxOf(1f, maxAbs)
    }

    /**
     * Bound for the Q8 path: activations are rounded to multiples of s_x = max|x_block| / 127 (error uniform in
     * ±s_x/2, σ = s_x/√12), so the output error has σ ≈ 0.29 · s_x · ‖q[:, c]‖ · scale[c]; we allow 5σ.
     */
    private fun q8Tolerance(w: NnTensor.Q8, q: ByteArray, x: FloatArray): FloatArray {
        var sx = 0f
        for (b in 0 until (w.rows + 127) / 128) {
            var mx = 0f
            for (k in b * 128 until minOf(w.rows, b * 128 + 128)) mx = maxOf(mx, abs(x[k]))
            sx = maxOf(sx, mx / 127f)
        }
        return FloatArray(w.cols) { c ->
            var ss = 0.0
            for (k in 0 until w.rows) { val v = q[k * w.cols + c].toDouble(); ss += v * v }
            (5 * 0.29 * sx * Math.sqrt(ss) * w.scales[c]).toFloat() + 1e-6f
        }
    }

    private fun assertWithinQ8(ref: FloatArray, actual: FloatArray, tol: FloatArray, c0: Int, c1: Int, what: String) {
        for (c in c0 until c1) assertTrue(abs(ref[c] - actual[c]) <= tol[c], "$what c=$c ref=${ref[c]} got=${actual[c]} tol=${tol[c]}")
    }

    @Test fun libraryLoadsAndSelfTests() {
        if (native(NativeNnKernels.Mode.F32) == null) return
        assertEquals(1, CmlNative.abiVersion())
        assertEquals(0, CmlNative.selftest())
        println("native: ${NativeLib.status}")
    }

    /** Random shapes (odd sizes, column sub-ranges, 1..70 tokens) against ScalarNnKernels, both modes. */
    @Test fun fuzzAgainstScalarKernels() {
        val f32 = native(NativeNnKernels.Mode.F32) ?: return
        val q8 = native(NativeNnKernels.Mode.Q8)!!
        val rnd = Random(2024)
        repeat(60) { iter ->
            val rows = 1 + rnd.nextInt(if (iter % 3 == 0) 700 else 90)
            val cols = 1 + rnd.nextInt(if (iter % 4 == 0) 1100 else 100)
            val n = 1 + rnd.nextInt(if (iter % 5 == 0) 70 else 9)
            val q = ByteArray(rows * cols).also { rnd.nextBytes(it) }
            val direct = rnd.nextBoolean()
            val buf = if (direct) ByteBuffer.allocateDirect(q.size).put(q).flip() else ByteBuffer.wrap(q).asReadOnlyBuffer()
            val w = NnTensor.Q8("w", rows, cols, buf, FloatArray(cols) { 0.001f + rnd.nextFloat() * 0.01f })
            val x = Array(n) { FloatArray(rows) { (rnd.nextGaussian() * (1 + rnd.nextInt(3))).toFloat() } }
            val c0 = rnd.nextInt(cols); val c1 = c0 + 1 + rnd.nextInt(cols - c0)
            val ref = Array(n) { FloatArray(cols) }
            val scratch = KernelScratch(cols)
            ScalarNnKernels.matmulCols(w, x, n, ref, c0, c1, scratch)
            val what = "iter $iter rows=$rows cols=$cols n=$n range=[$c0,$c1) direct=$direct"
            // f32: same arithmetic, summation order differs
            val y = Array(n) { FloatArray(cols) }
            f32.matmulCols(w, x, n, y, c0, c1, scratch)
            for (t in 0 until n) assertTrue(maxRelErr(ref[t], y[t], c1 - c0, c0) < 1e-4f, "f32 $what t=$t err=${maxRelErr(ref[t], y[t], c1 - c0, c0)}")
            // q8: activation quantisation noise (per 128-block absmax/127) — relative to the row's max output
            val y8 = Array(n) { FloatArray(cols) }
            q8.matmulCols(w, x, n, y8, c0, c1, scratch)
            val tol = Array(n) { q8Tolerance(w, q, x[it]) }
            for (t in 0 until n) assertWithinQ8(ref[t], y8[t], tol[t], c0, c1, "q8 $what t=$t")
            // whole-matmul path with several matrices and the model's executor
            val cols2 = 1 + rnd.nextInt(300)
            val q2 = ByteArray(rows * cols2).also { rnd.nextBytes(it) }
            val w2 = NnTensor.Q8("w2", rows, cols2, ByteBuffer.wrap(q2), FloatArray(cols2) { 0.01f })
            val refFull = Array(n) { FloatArray(cols) }; val ref2 = Array(n) { FloatArray(cols2) }
            ScalarNnKernels.matmulCols(w, x, n, refFull, 0, cols, scratch)
            ScalarNnKernels.matmulCols(w2, x, n, ref2, 0, cols2, KernelScratch(cols2))
            NnExecutor(1 + rnd.nextInt(4)).use { ex ->
                val ys = arrayOf(Array(n) { FloatArray(cols) }, Array(n) { FloatArray(cols2) })
                assertTrue(f32.matmul(arrayOf(w, w2), x, n, ys, ex))
                for (t in 0 until n) {
                    assertTrue(maxRelErr(refFull[t], ys[0][t], cols) < 1e-4f, "f32 matmul $what t=$t")
                    assertTrue(maxRelErr(ref2[t], ys[1][t], cols2) < 1e-4f, "f32 matmul(w2) $what t=$t")
                }
                val ys8 = arrayOf(Array(n) { FloatArray(cols) }, Array(n) { FloatArray(cols2) })
                assertTrue(q8.matmul(arrayOf(w, w2), x, n, ys8, ex))
                for (t in 0 until n) {
                    assertWithinQ8(refFull[t], ys8[0][t], tol[t], 0, cols, "q8 matmul $what t=$t")
                    assertWithinQ8(ref2[t], ys8[1][t], q8Tolerance(w2, q2, x[t]), 0, cols2, "q8 matmul(w2) $what t=$t")
                }
            }
            // K-split partial (f32 path)
            val acc = FloatArray(cols); val accRef = FloatArray(cols)
            val mid = rnd.nextInt(rows + 1)
            f32.matvecRowsPartial(w, x[0], 0, mid, acc, scratch); f32.matvecRowsPartial(w, x[0], mid, rows, acc, scratch)
            ScalarNnKernels.matvecRowsPartial(w, x[0], 0, rows, accRef, scratch)
            assertTrue(maxRelErr(accRef, acc, cols) < 1e-4f, "partial $what")
        }
        f32.close(); q8.close()
    }

    /** End to end in Q8 mode: logits close to the float reference and the same greedy choices on a tiny model. */
    @Test fun q8ModelMatchesReferenceApproximately() {
        val k = native(NativeNnKernels.Mode.Q8) ?: return
        val cfg = NnConfig(vocabSize = 301, dModel = 96, nLayers = 2, nHeads = 3, nKvHeads = 1, ffnDim = 200, maxContext = 64, tiedEmbeddings = false)
        val weights = NnTestModels.inMemory(cfg, 11)
        val rnd = Random(5)
        val tokens = IntArray(40) { rnd.nextInt(cfg.vocabSize) }
        val ref = NnTestModels.referenceLogits(weights, tokens)
        NnModel(weights, 3, k).use { model ->
            val s = model.newSession()
            val logits = s.prefill(tokens)
            assertNotNull(logits)
            val err = maxRelErr(ref.last(), logits, cfg.vocabSize)
            assertTrue(err < 5e-2f, "q8 prefill logits rel. error $err")
            var agree = 0
            val s2 = model.newSession(capacity = 4)
            s2.prefill(tokens.copyOf(5))
            for (t in 5 until tokens.size) {
                val l = s2.decode(tokens[t])
                if (Sampler.argmax(l) == Sampler.argmax(ref[t])) agree++
                assertTrue(maxRelErr(ref[t], l, cfg.vocabSize) < 5e-2f, "q8 decode@$t")
            }
            assertTrue(agree >= 30, "greedy agreement $agree / 35")
        }
    }
}
