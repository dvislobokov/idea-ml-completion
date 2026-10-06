package io.github.completionml.core.nn.native

import java.lang.ref.Cleaner

/**
 * KV cache in native memory for the native attention kernel: per (layer, kv head) keys transposed `[hd][capacity]`
 * and values `[capacity][hd]`, both float32. Grown by reallocation; freed by [close] or, failing that, by a Cleaner.
 */
internal class NativeKv(private val layers: Int, private val kvHeads: Int, private val hd: Int, capacity: Int) : AutoCloseable {
    var capacity = capacity; private set
    val kAddrs = LongArray(layers * kvHeads)
    val vAddrs = LongArray(layers * kvHeads)
    private val state = State(kAddrs, vAddrs)
    private val cleanable = cleaner.register(this, state)

    private class State(val k: LongArray, val v: LongArray) : Runnable {
        var scratch = LongArray(0); var scratchFloats = 0
        override fun run() {
            for (i in k.indices) { if (k[i] != 0L) CmlNative.free(k[i]); if (v[i] != 0L) CmlNative.free(v[i]); k[i] = 0; v[i] = 0 }
            for (p in scratch) if (p != 0L) CmlNative.free(p)
            scratch = LongArray(0)
        }
    }

    /** Per-worker native scratch of at least [floats] floats (attention scores). */
    fun scratch(worker: Int, workers: Int, floats: Int): Long {
        val st = state
        if (st.scratch.size < workers || st.scratchFloats < floats) {
            for (p in st.scratch) if (p != 0L) CmlNative.free(p)
            val n = maxOf(floats, 256)
            st.scratch = LongArray(workers) { CmlNative.malloc(4L * n) }; st.scratchFloats = n
        }
        return st.scratch[worker]
    }

    init {
        for (i in kAddrs.indices) { kAddrs[i] = CmlNative.malloc(4L * hd * capacity); vAddrs[i] = CmlNative.malloc(4L * hd * capacity) }
    }

    fun index(layer: Int, kvHead: Int) = layer * kvHeads + kvHead

    /** Reallocates to [nc] positions keeping the first [used] ones. */
    fun grow(nc: Int, used: Int) {
        if (nc <= capacity) return
        for (i in kAddrs.indices) {
            val nk = CmlNative.malloc(4L * hd * nc)
            for (d in 0 until hd) CmlNative.memcpy(nk + 4L * d * nc, kAddrs[i] + 4L * d * capacity, 4L * used)
            val nv = CmlNative.malloc(4L * hd * nc)
            CmlNative.memcpy(nv, vAddrs[i], 4L * used * hd)
            CmlNative.free(kAddrs[i]); CmlNative.free(vAddrs[i])
            kAddrs[i] = nk; vAddrs[i] = nv
        }
        capacity = nc
    }

    /** Stores k/v rows (all kv heads) of tokens [0, n) at positions p0.. of [layer]. */
    fun store(layer: Int, p0: Int, k: Array<FloatArray>, v: Array<FloatArray>, n: Int) =
        CmlNative.kvStore(kAddrs, vAddrs, layer * kvHeads, kvHeads, capacity, hd, p0, k, v, n)

    /** Causal attention of one head for query rows [t0, t1) (positions p0 + t), writing the head slice of out. */
    fun attention(layer: Int, kvHead: Int, q: Array<FloatArray>, out: Array<FloatArray>, t0: Int, t1: Int, qo: Int, scale: Float, p0: Int, scratch: Long) {
        val i = index(layer, kvHead)
        CmlNative.attnBlock(q, out, t0, t1, qo, hd, scale, kAddrs[i], capacity, vAddrs[i], p0, scratch)
    }

    override fun close() = cleanable.clean()

    companion object { private val cleaner: Cleaner = Cleaner.create() }
}
