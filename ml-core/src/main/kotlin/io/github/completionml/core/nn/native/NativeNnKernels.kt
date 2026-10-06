package io.github.completionml.core.nn.native

import io.github.completionml.core.nn.KernelScratch
import io.github.completionml.core.nn.NnExecutor
import io.github.completionml.core.nn.NnKernels
import io.github.completionml.core.nn.NnTensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.IdentityHashMap

/**
 * [NnKernels] backed by `libcmlkernels` (native/). Two modes:
 *  - [Mode.F32]: float activations, int8 weights dequantised on the fly — the same arithmetic as [io.github.completionml.core.nn.ScalarNnKernels]
 *    (results differ only by summation order), weights are read in place from the mapped model;
 *  - [Mode.Q8]: activations quantised to int8 per (token, 128-input block); weights repacked once per matrix into the
 *    integer dot-product layout (a second copy of the weights in native memory), 3-4x more MACs per cycle. Adds
 *    quantisation noise to the activations (~0.4 % per element) — to be validated on the trained model (ppl, eval-inline).
 *
 * The whole-matmul path ([matmul]) is what the model uses: activations are copied once per matmul into native scratch,
 * the work is split into (matrix, column chunk, token block) tasks on the model's [NnExecutor], each task runs one
 * native GEMM and copies its output rows back. The per-task [matmulCols]/[matvecRowsPartial] methods exist for the
 * kernel-level tests and as a drop-in for the scalar interface.
 *
 * One instance per model (it owns scratch buffers and the per-tensor prepared weights); not thread-safe across
 * concurrent matmuls, like the rest of NnModel.
 */
class NativeNnKernels private constructor(val mode: Mode) : NnKernels, AutoCloseable {
    enum class Mode { F32, Q8 }

    override val name: String get() = "native-${NativeLib.isaName}-${mode.name.lowercase()}"

    private class Prepared(val q: Long, val scale: Long, val packed: Long, val ownsQ: Boolean)

    private val prepared = IdentityHashMap<NnTensor.Q8, Prepared>()
    private val owned = ArrayList<Long>()

    // scratch: activations (floats or quantised) and outputs, grown on demand
    private var xBuf: ByteBuffer = ByteBuffer.allocateDirect(0)
    private var xAddr = 0L
    private var xFloats: FloatBuffer = xBuf.asFloatBuffer()
    private var xqAddr = 0L; private var xqCap = 0L
    private var yBuf: ByteBuffer = ByteBuffer.allocateDirect(0)
    private var yAddr = 0L
    private var yFloats: FloatBuffer = yBuf.asFloatBuffer()

    private fun ensureX(floats: Int) {
        if (xBuf.capacity() >= floats * 4) return
        xBuf = ByteBuffer.allocateDirect(maxOf(floats * 4, xBuf.capacity() * 2, 1 shl 16)).order(ByteOrder.nativeOrder())
        xFloats = xBuf.asFloatBuffer(); xAddr = CmlNative.address(xBuf)
    }
    private fun ensureY(floats: Int) {
        if (yBuf.capacity() >= floats * 4) return
        yBuf = ByteBuffer.allocateDirect(maxOf(floats * 4, yBuf.capacity() * 2, 1 shl 16)).order(ByteOrder.nativeOrder())
        yFloats = yBuf.asFloatBuffer(); yAddr = CmlNative.address(yBuf)
    }
    private fun ensureXq(bytes: Long) {
        if (xqCap >= bytes) return
        if (xqAddr != 0L) { CmlNative.free(xqAddr); owned.remove(xqAddr) }
        xqCap = maxOf(bytes, xqCap * 2, 1L shl 16)
        xqAddr = CmlNative.malloc(xqCap); owned += xqAddr
    }

    /** Native view of a weight matrix: data address (copied if the buffer is on the heap), native scales, packed form. */
    @Synchronized
    private fun prep(w: NnTensor.Q8): Prepared {
        prepared[w]?.let { return it }
        val bytes = w.rows.toLong() * w.cols
        var q = if (w.data.isDirect) CmlNative.address(w.data) + w.data.position() else 0L
        var ownsQ = false
        if (q == 0L) {
            q = CmlNative.malloc(bytes); owned += q; ownsQ = true
            val src = w.data.duplicate()
            val chunk = ByteArray(minOf(bytes, 1L shl 20).toInt())
            var off = 0L
            while (src.hasRemaining()) {
                val n = minOf(chunk.size, src.remaining()); src.get(chunk, 0, n)
                CmlNative.copyInBytes(chunk, 0, q + off, n); off += n
            }
        }
        val scale = CmlNative.malloc(4L * w.cols); owned += scale
        CmlNative.copyIn(w.scales, 0, scale, w.cols)
        var packed = 0L
        if (mode == Mode.Q8) {
            packed = CmlNative.malloc(CmlNative.packSize(w.rows, w.cols)); owned += packed
            CmlNative.pack(q, w.rows, w.cols, packed)
            if (ownsQ) { CmlNative.free(q); owned.remove(q); q = 0L }   // the packed copy is all the Q8 path needs
        }
        return Prepared(q, scale, packed, ownsQ).also { prepared[w] = it }
    }

    /** Prepares all matrices up front (packing for Q8 takes ~1 s per 100 MB otherwise paid at first use). */
    fun prepare(ws: Iterable<NnTensor.Q8>) { for (w in ws) prep(w) }

    /** Releases native memory (prepared weights, scratch); the instance stays usable and re-prepares on demand. */
    override fun close() {
        synchronized(this) {
            for (p in owned) CmlNative.free(p)
            owned.clear(); prepared.clear(); xqAddr = 0; xqCap = 0
        }
    }

    // ---------------------------------------------------------------------------------------- whole-matmul path

    override fun matmul(ws: Array<NnTensor.Q8>, x: Array<FloatArray>, n: Int, ys: Array<Array<FloatArray>>, executor: NnExecutor): Boolean {
        val k = ws[0].rows
        val threads = executor.nThreads
        val preps = Array(ws.size) { prep(ws[it]) }
        // activations: one contiguous [n][k] copy (+ quantised form)
        ensureX(n * k)
        val xf = xFloats
        val q8 = mode == Mode.Q8
        val xqStride = if (q8) CmlNative.qactStride(k) else 0L
        if (q8) ensureXq(xqStride * n)
        val xq = xqAddr
        val prof = io.github.completionml.core.nn.NnSession.Profile.enabled
        // outputs: per matrix an [n][cols] block in one native buffer
        val yOff = IntArray(ws.size + 1)
        for (m in ws.indices) yOff[m + 1] = yOff[m] + n * ws[m].cols
        ensureY(yOff[ws.size])
        val yf = yFloats; val yBase = yAddr

        if (n >= 48) {
            // Prefill: token-block tasks, each copies in / quantises its rows, runs every matrix over all columns
            // (the native drivers block columns internally so the weight panel stays in L2) and copies whole rows out.
            val tb = maxOf(12, ((n + threads * 3 - 1) / (threads * 3) + 11) / 12 * 12)
            val nTok = (n + tb - 1) / tb
            executor.run(nTok) { task, _ ->
                val t0 = task * tb; val t1 = minOf(n, t0 + tb)
                val tC = if (prof) System.nanoTime() else 0L
                for (t in t0 until t1) xf.put(t * k, x[t], 0, k)
                if (q8) CmlNative.qact(xAddr + 4L * t0 * k, k, t1 - t0, k, xq + t0 * xqStride)
                if (prof) io.github.completionml.core.nn.NnSession.Profile.copyIn.add(System.nanoTime() - tC)
                for (m in ws.indices) {
                    val w = ws[m]; val p = preps[m]; val cols = w.cols
                    val yAddrM = yBase + 4L * (yOff[m] + t0 * cols)
                    if (q8) CmlNative.gemmQ8(p.packed, p.scale, k, cols, xq, t0, t1, yAddrM, cols, 0, cols)
                    else CmlNative.gemmF32(p.q, p.scale, k, cols, xAddr + 4L * t0 * k, k, t1 - t0, yAddrM, cols, 0, cols)
                    val tO = if (prof) System.nanoTime() else 0L
                    val ym = ys[m]
                    for (t in t0 until t1) yf.get(yOff[m] + t * cols, ym[t], 0, cols)
                    if (prof) io.github.completionml.core.nn.NnSession.Profile.copyOut.add(System.nanoTime() - tO)
                }
            }
            return true
        }

        val tC = if (prof) System.nanoTime() else 0L
        for (t in 0 until n) xf.put(t * k, x[t], 0, k)
        if (q8) CmlNative.qact(xAddr, k, n, k, xq)
        if (prof) io.github.completionml.core.nn.NnSession.Profile.copyIn.add(System.nanoTime() - tC)

        if (n == 1) {
            // decode: split the columns of all matrices into ~2 chunks per thread (multiples of 32)
            val total = ws.sumOf { it.cols }
            val chunk = maxOf(64, ((total + threads * 2 - 1) / (threads * 2) + 31) / 32 * 32)
            val starts = IntArray(ws.size + 1)
            for (m in ws.indices) starts[m + 1] = starts[m] + (ws[m].cols + chunk - 1) / chunk
            executor.run(starts[ws.size]) { task, _ ->
                var m = 0
                while (task >= starts[m + 1]) m++
                val w = ws[m]; val p = preps[m]
                val c0 = (task - starts[m]) * chunk; val c1 = minOf(w.cols, c0 + chunk)
                val yAddrM = yBase + 4L * yOff[m]
                if (q8) CmlNative.gemmQ8(p.packed, p.scale, k, w.cols, xq, 0, 1, yAddrM, w.cols, c0, c1)
                else CmlNative.gemmF32(p.q, p.scale, k, w.cols, xAddr, k, 1, yAddrM, w.cols, c0, c1)
                yf.get(yOff[m] + c0, ys[m][0], c0, c1 - c0)
            }
            return true
        }
        // a few tokens (incremental prefill): (matrix, column chunk, token block) tasks
        val chunk = 256
        val colTasks = ws.sumOf { (it.cols + chunk - 1) / chunk }
        val tokBlocks = maxOf((threads * 4 + colTasks - 1) / colTasks, (n + 127) / 128).coerceIn(1, maxOf(1, n / 12))
        val tokBlock = ((n + tokBlocks - 1) / tokBlocks + 11) / 12 * 12
        val nTok = (n + tokBlock - 1) / tokBlock
        val starts = IntArray(ws.size + 1)
        for (m in ws.indices) starts[m + 1] = starts[m] + (ws[m].cols + chunk - 1) / chunk
        executor.run(colTasks * nTok) { task, _ ->
            val ct = task / nTok; val tb = task % nTok
            var m = 0
            while (ct >= starts[m + 1]) m++
            val w = ws[m]; val p = preps[m]; val cols = w.cols
            val c0 = (ct - starts[m]) * chunk; val c1 = minOf(cols, c0 + chunk)
            val t0 = tb * tokBlock; val t1 = minOf(n, t0 + tokBlock)
            val yAddrM = yBase + 4L * (yOff[m] + t0 * cols)
            if (q8) CmlNative.gemmQ8(p.packed, p.scale, k, cols, xq, t0, t1, yAddrM, cols, c0, c1)
            else CmlNative.gemmF32(p.q, p.scale, k, cols, xAddr + 4L * t0 * k, k, t1 - t0, yAddrM, cols, c0, c1)
            val ym = ys[m]
            val tO = if (prof) System.nanoTime() else 0L
            for (t in t0 until t1) yf.get(yOff[m] + t * cols + c0, ym[t], c0, c1 - c0)
            if (prof) io.github.completionml.core.nn.NnSession.Profile.copyOut.add(System.nanoTime() - tO)
        }
        return true
    }

    // ---------------------------------------------------------------------------------------- per-task interface

    override fun matmulCols(w: NnTensor.Q8, x: Array<FloatArray>, n: Int, y: Array<FloatArray>, c0: Int, c1: Int, s: KernelScratch) {
        // Not used by the model (see matmul); kept for the kernel tests. Serialised: shares the scratch buffers.
        synchronized(this) {
            val p = prep(w); val k = w.rows
            ensureX(n * k); ensureY(n * w.cols)
            for (t in 0 until n) xFloats.put(t * k, x[t], 0, k)
            if (mode == Mode.Q8) {
                ensureXq(CmlNative.qactStride(k) * n)
                CmlNative.qact(xAddr, k, n, k, xqAddr)
                CmlNative.gemmQ8(p.packed, p.scale, k, w.cols, xqAddr, 0, n, yAddr, w.cols, c0, c1)
            } else CmlNative.gemmF32(p.q, p.scale, k, w.cols, xAddr, k, n, yAddr, w.cols, c0, c1)
            for (t in 0 until n) yFloats.get(t * w.cols + c0, y[t], c0, c1 - c0)
        }
    }

    override fun matvecRowsPartial(w: NnTensor.Q8, x: FloatArray, r0: Int, r1: Int, acc: FloatArray, s: KernelScratch) {
        val p = prep(w)
        if (p.q != 0L) { CmlNative.gemvF32Partial(p.q, w.rows, w.cols, x, r0, r1, acc); return }
        // Q8 mode dropped the plain copy of a heap-backed matrix: fall back to the scalar loop for this rare path
        io.github.completionml.core.nn.ScalarNnKernels.matvecRowsPartial(w, x, r0, r1, acc, s)
    }

    companion object {
        /** Native kernels if the library loads and passes its self-test, else null. Mode from
         * `completionml.nn.native.mode` (`q8` default, `f32`) unless given. */
        fun loadOrNull(mode: Mode? = null): NativeNnKernels? {
            if (!NativeLib.ensureLoaded()) return null
            val m = mode ?: when (System.getProperty("completionml.nn.native.mode", "q8").lowercase()) { "f32" -> Mode.F32; else -> Mode.Q8 }
            return NativeNnKernels(m)
        }
    }
}
