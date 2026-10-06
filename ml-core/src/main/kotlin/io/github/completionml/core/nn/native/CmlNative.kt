package io.github.completionml.core.nn.native

import java.nio.ByteBuffer

/**
 * JNI bindings of `libcmlkernels` (native/cml_kernels.h). Every method is a direct call into one C function; raw
 * addresses (`Long`) refer to off-heap memory: mapped/direct ByteBuffers of the model weights and scratch buffers
 * from [malloc]. Only [NativeLib] loads the library; callers go through [NativeNnKernels].
 */
internal object CmlNative {
    const val ISA_SCALAR = 0
    const val ISA_AVX2 = 1
    const val ISA_AVX2_VNNI = 2
    const val ISA_AVX512 = 3
    const val ISA_AVX512_VNNI = 4
    const val ISA_NEON = 16
    const val ISA_NEON_DOTPROD = 17

    @JvmStatic external fun abiVersion(): Int
    @JvmStatic external fun isaDetect(): Int
    @JvmStatic external fun isa(): Int
    @JvmStatic external fun setIsa(isa: Int): Int
    @JvmStatic external fun isaName(isa: Int): String
    @JvmStatic external fun selftest(): Int
    @JvmStatic external fun qactUnsigned(): Int

    /** Start address of a direct buffer's memory (position not included), 0 for heap buffers. */
    @JvmStatic external fun address(buf: ByteBuffer): Long
    @JvmStatic external fun malloc(bytes: Long): Long
    @JvmStatic external fun free(p: Long)

    @JvmStatic external fun gemmF32(q: Long, scale: Long, k: Int, n: Int, x: Long, ldx: Int, tokens: Int, y: Long, ldy: Int, c0: Int, c1: Int)
    @JvmStatic external fun gemmQ8(packed: Long, scale: Long, k: Int, n: Int, xq: Long, t0: Int, t1: Int, y: Long, ldy: Int, c0: Int, c1: Int)
    @JvmStatic external fun gemvF32Partial(q: Long, k: Int, n: Int, x: FloatArray, k0: Int, k1: Int, acc: FloatArray)

    @JvmStatic external fun packSize(k: Int, n: Int): Long
    @JvmStatic external fun pack(q: Long, k: Int, n: Int, packed: Long)
    @JvmStatic external fun qactSize(tokens: Int, k: Int): Long
    @JvmStatic external fun qactStride(k: Int): Long
    @JvmStatic external fun qact(x: Long, ldx: Int, tokens: Int, k: Int, xq: Long)

    @JvmStatic external fun copyIn(src: FloatArray, off: Int, dst: Long, n: Int)
    @JvmStatic external fun copyOut(src: Long, dst: FloatArray, off: Int, n: Int)
    @JvmStatic external fun copyInBytes(src: ByteArray, off: Int, dst: Long, n: Int)

    @JvmStatic external fun rmsnorm(x: FloatArray, w: FloatArray, d: Int, eps: Float, out: FloatArray)
    @JvmStatic external fun siluMul(g: FloatArray, u: FloatArray, n: Int)
    @JvmStatic external fun memcpy(dst: Long, src: Long, bytes: Long)
    @JvmStatic external fun kvStore(kAddrs: LongArray, vAddrs: LongArray, base: Int, nKv: Int, ldk: Int, hd: Int, p0: Int, k: Array<FloatArray>, v: Array<FloatArray>, n: Int)
    @JvmStatic external fun attnBlock(q: Array<FloatArray>, out: Array<FloatArray>, t0: Int, t1: Int, qo: Int, hd: Int, scale: Float, kT: Long, ldk: Int, v: Long, p0: Int, sc: Long)
    @JvmStatic external fun attnHead(q: FloatArray, qo: Int, hd: Int, scale: Float, kT: Long, ldk: Int, v: Long, ldv: Int, len: Int, sc: Long, out: FloatArray, oo: Int)
}
