package io.github.completionml.core.nn

import io.github.completionml.core.nn.native.CmlNative
import io.github.completionml.core.nn.native.NativeLib
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * JNI overhead micro-benchmark (test-scope main): cost of an empty JNI call, of a kernel call with 11 arguments on a
 * tiny matrix, of heap<->native copies via GetPrimitiveArrayCritical and via FloatBuffer bulk put/get.
 */
object NativeMicroBench {
    @JvmStatic fun main(args: Array<String>) {
        check(NativeLib.ensureLoaded()) { NativeLib.status }
        println(NativeLib.status)
        val reps = 2_000_000
        fun time(name: String, perCall: Int = 1, body: () -> Unit) {
            repeat(3) { repeat(reps / 10) { body() } }
            val t0 = System.nanoTime()
            repeat(reps) { body() }
            val ns = (System.nanoTime() - t0).toDouble() / reps / perCall
            println("%-48s %8.1f ns".format(name, ns))
        }
        time("empty JNI call (abiVersion)") { CmlNative.abiVersion() }
        val k = 32; val n = 32
        val q = CmlNative.malloc((k * n).toLong()); val sc = CmlNative.malloc(4L * n)
        val xb = ByteBuffer.allocateDirect(4 * k).order(ByteOrder.nativeOrder()); val x = CmlNative.address(xb)
        val yb = ByteBuffer.allocateDirect(4 * n).order(ByteOrder.nativeOrder()); val y = CmlNative.address(yb)
        time("gemmF32 32x32, n=1 (JNI + kernel, 11 args)") { CmlNative.gemmF32(q, sc, k, n, x, k, 1, y, n, 0, n) }
        val arr = FloatArray(384); val big = FloatArray(4096)
        time("copyIn 384 floats (critical region)") { CmlNative.copyIn(arr, 0, x, 8) }
        val fb = ByteBuffer.allocateDirect(4 * 4096).order(ByteOrder.nativeOrder()).asFloatBuffer()
        time("FloatBuffer.put 384 floats") { fb.put(0, arr, 0, 384) }
        time("FloatBuffer.get 384 floats") { fb.get(0, arr, 0, 384) }
        time("FloatBuffer.put 4096 floats") { fb.put(0, big, 0, 4096) }
        val bigDirect = ByteBuffer.allocateDirect(4 * 4096).order(ByteOrder.nativeOrder()); val bigAddr = CmlNative.address(bigDirect)
        time("copyIn 4096 floats (critical region)") { CmlNative.copyIn(big, 0, bigAddr, 4096) }
        time("gemvF32Partial 32 rows (2 critical arrays)") { CmlNative.gemvF32Partial(q, k, n, arr, 0, 32, arr) }
        // dispatch cost of the executor for comparison
        NnExecutor(8).use { ex -> time("NnExecutor.run(8 tasks) x8 threads") { ex.run(8) { _, _ -> } } }
        CmlNative.free(q); CmlNative.free(sc)
    }
}
