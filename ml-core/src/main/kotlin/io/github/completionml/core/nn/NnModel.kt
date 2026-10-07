package io.github.completionml.core.nn

import io.github.completionml.core.nn.NnFormat.Names
import io.github.completionml.core.nn.native.CmlNative
import io.github.completionml.core.nn.native.NativeKv
import io.github.completionml.core.nn.native.NativeNnKernels
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Inference for our decoder-only transformer: pre-norm blocks `x += Attn(RMSNorm(x)); x += SwiGLU(RMSNorm(x))`,
 * RoPE ("rotate half" / GPT-NeoX convention, as in HF LLaMA), causal attention with GQA, tied or separate output head.
 *
 * Thread-safety: the model is immutable and may be shared; a [NnSession] (KV cache) and its scratch buffers are not,
 * and the model's [NnExecutor] runs one region at a time — use one model instance per decoding thread, or synchronise.
 */
class NnModel(
    val weights: NnFormat.Model,
    /** Worker threads for prefill/decode; the default uses the cores (≤ 8: beyond that decode is memory-bound) so a plugin cannot forget it. */
    nThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8),
    val kernels: NnKernels = NnKernels.best(),
) : AutoCloseable {
    val config: NnConfig = weights.config
    val executor = NnExecutor(nThreads)
    val nThreads: Int get() = executor.nThreads

    internal class Layer(
        val attnNorm: FloatArray, val wq: NnTensor.Q8, val wk: NnTensor.Q8, val wv: NnTensor.Q8, val wo: NnTensor.Q8,
        val ffnNorm: FloatArray, val w1: NnTensor.Q8, val w3: NnTensor.Q8, val w2: NnTensor.Q8,
    )

    internal val tokEmb = weights.q8(Names.TOK_EMB)
    internal val lmHead = if (config.tiedEmbeddings) tokEmb else weights.q8(Names.LM_HEAD)
    internal val finalNorm = weights.f32(Names.FINAL_NORM).data
    internal val layers = Array(config.nLayers) { l ->
        fun q(n: String) = weights.q8(Names.layer(l, n))
        Layer(
            weights.f32(Names.layer(l, Names.ATTN_NORM)).data, q(Names.WQ), q(Names.WK), q(Names.WV), q(Names.WO),
            weights.f32(Names.layer(l, Names.FFN_NORM)).data, q(Names.W1), q(Names.W3), q(Names.W2),
        )
    }

    // widest layer matrix; wider ones (the vocabulary head) are always column-split
    private val maxCols = maxOf(config.ffnDim, config.dModel, config.vocabSize)
    internal val scratch = Array(nThreads) { KernelScratch(maxCols) }
    // per-worker partial sums for K-split decode matvecs, one per concurrently computed matrix (≤ 3)
    internal val partials = Array(nThreads) { Array(3) { FloatArray(maxCols) } }
    internal val partialUsed = Array(nThreads) { BooleanArray(3) }

    // RoPE inverse frequencies (float32 like PyTorch: 1 / theta^(2i/hd))
    internal val invFreq = FloatArray(config.headDim / 2) { i ->
        (1.0 / Math.pow(config.ropeTheta.toDouble(), (2.0 * i) / config.headDim)).toFloat()
    }

    // cos/sin per (position, pair) — float32 like the reference: cos(pos * invFreq[i])
    private val ropeCos = FloatArray(config.maxContext * (config.headDim / 2))
    private val ropeSin = FloatArray(config.maxContext * (config.headDim / 2))
    init {
        val half = config.headDim / 2
        for (pos in 0 until config.maxContext) for (i in 0 until half) {
            val a = pos * invFreq[i]
            ropeCos[pos * half + i] = kotlin.math.cos(a); ropeSin[pos * half + i] = kotlin.math.sin(a)
        }
    }

    fun newSession(capacity: Int = minOf(config.maxContext, 1024)): NnSession = NnSession(this, capacity)

    override fun close() {
        executor.close()
        (kernels as? AutoCloseable)?.close()
    }

    // ------------------------------------------------------------------------------------------- parallel matmuls

    /**
     * `ys[m][t] = x[t] · ws[m]` for every matrix m (all sharing input x of n tokens). One parallel region.
     * A single token (decode, logits) uses row-split partial sums (contiguous weight reads, full-width vector loops);
     * several tokens (prefill) split into (column chunk × token block) tasks.
     */
    internal fun matmul(ws: Array<NnTensor.Q8>, x: Array<FloatArray>, n: Int, ys: Array<Array<FloatArray>>) {
        if (kernels.matmul(ws, x, n, ys, executor)) return
        val threads = executor.nThreads
        if (n == 1 && ws.size <= 3) {
            matvecRowSplit(ws, x[0], Array(ws.size) { ys[it][0] }); return
        }
        // Tasks = (matrix, column chunk, token block). Column chunks stay wide (long vector loops, kernel tiles of
        // 256 columns); with several tokens, token blocks supply the remaining parallelism.
        val chunk = if (n == 1) maxOf(256, (ws.sumOf { it.cols } / (threads * 4) + 63) / 64 * 64).coerceAtMost(2048) else 256
        val colTasks = ws.sumOf { (it.cols + chunk - 1) / chunk }
        val tokBlocks = if (n == 1) 1 else maxOf((threads * 4 + colTasks - 1) / colTasks, (n + 127) / 128).coerceIn(1, maxOf(1, n / 16))
        val tokBlock = ((n + tokBlocks - 1) / tokBlocks + 3) / 4 * 4
        val nTok = (n + tokBlock - 1) / tokBlock
        val starts = IntArray(ws.size + 1)
        for (m in ws.indices) starts[m + 1] = starts[m] + (ws[m].cols + chunk - 1) / chunk
        executor.run(colTasks * nTok) { task, worker ->
            val ct = task / nTok; val tb = task % nTok
            var m = 0
            while (ct >= starts[m + 1]) m++
            val c0 = (ct - starts[m]) * chunk
            val t0 = tb * tokBlock; val t1 = minOf(n, t0 + tokBlock)
            val xs = if (nTok == 1) x else x.copyOfRange(t0, t1)
            val yy = if (nTok == 1) ys[m] else ys[m].copyOfRange(t0, t1)
            kernels.matmulCols(ws[m], xs, t1 - t0, yy, c0, minOf(ws[m].cols, c0 + chunk), scratch[worker])
        }
    }

    private fun matvecRowSplit(ws: Array<NnTensor.Q8>, x: FloatArray, ys: Array<FloatArray>) {
        val threads = executor.nThreads
        for (w in 0 until threads) partialUsed[w].fill(false)
        // tasks: per matrix `threads` row ranges
        val tasks = ws.size * threads
        executor.run(tasks) { task, worker ->
            val m = task / threads; val part = task % threads
            val rows = ws[m].rows
            val r0 = (rows.toLong() * part / threads).toInt(); val r1 = (rows.toLong() * (part + 1) / threads).toInt()
            val acc = partials[worker][m]
            if (!partialUsed[worker][m]) { acc.fill(0f, 0, ws[m].cols); partialUsed[worker][m] = true }
            kernels.matvecRowsPartial(ws[m], x, r0, r1, acc, scratch[worker])
        }
        for (m in ws.indices) {
            val y = ys[m]; val cols = ws[m].cols
            y.fill(0f, 0, cols)
            for (w in 0 until threads) if (partialUsed[w][m]) Vec.add(partials[w][m], y, cols)
            Vec.mul(ws[m].scales, y, cols)
        }
    }

    // ---------------------------------------------------------------------------------------------- small ops

    internal fun rmsNorm(x: FloatArray, w: FloatArray, out: FloatArray) {
        val d = config.dModel
        var ss = 0.0
        for (j in 0 until d) ss += x[j] * x[j]
        val inv = (1.0 / sqrt(ss / d + config.normEps)).toFloat()
        for (j in 0 until d) out[j] = ftz(x[j] * inv * w[j])
    }

    /** Rotates every head (size headDim) of [v] for position [pos], "rotate half" pairing (i, i + headDim/2). */
    internal fun rope(v: FloatArray, heads: Int, pos: Int) {
        val hd = config.headDim; val half = hd / 2
        val base = pos * half
        for (i in 0 until half) {
            val c = ropeCos[base + i]; val s = ropeSin[base + i]
            for (h in 0 until heads) {
                val o = h * hd
                val x0 = v[o + i]; val x1 = v[o + i + half]
                v[o + i] = x0 * c - x1 * s
                v[o + i + half] = x0 * s + x1 * c
            }
        }
    }

    internal fun embed(token: Int, out: FloatArray) {
        require(token in 0 until config.vocabSize) { "token $token out of vocabulary" }
        val v = config.vocabSize; val data = tokEmb.data; val s = tokEmb.scales[token]
        for (j in 0 until config.dModel) out[j] = data.get(j * v + token) * s
    }

    internal fun silu(x: Float): Float = if (x < -80f) 0f else x / (1f + exp(-x))
}

/**
 * Decoding state: the KV cache of one sequence. Keys are stored transposed per KV head and dimension
 * (`kT[layer][kvHead][dim][pos]`) and values per position (`v[layer][kvHead][pos][dim]`), so that both attention
 * products are offset-0 axpy loops that C2 vectorises. Supports prefix reuse: [prefill] keeps the longest common
 * prefix with the previously processed tokens.
 */
class NnSession internal constructor(private val model: NnModel, capacity: Int) : AutoCloseable {
    private val c = model.config
    private val hd = c.headDim
    var capacity = capacity.coerceIn(1, c.maxContext); private set
    /** With the native kernels the KV cache lives in native memory and attention/SiLU run natively. */
    private val native: NativeNnKernels? = model.kernels as? NativeNnKernels
    private val nkv: NativeKv? = native?.let { NativeKv(c.nLayers, c.nKvHeads, hd, this.capacity) }
    private var kT = if (nkv == null) Array(c.nLayers) { Array(c.nKvHeads) { Array(hd) { FloatArray(this.capacity) } } } else emptyArray()
    private var vC = if (nkv == null) Array(c.nLayers) { Array(c.nKvHeads) { arrayOfNulls<FloatArray>(this.capacity) } } else emptyArray()
    private var history = IntArray(this.capacity)

    /** Frees the native KV cache (no-op for the Kotlin cache; a Cleaner also frees it eventually). */
    override fun close() { nkv?.close() }

    /** Number of tokens in the cache. */
    var length = 0; private set

    /** Tokens currently in the cache. */
    fun tokens(): IntArray = history.copyOf(length)

    /** Drops cached positions ≥ [len]. */
    fun truncate(len: Int) { require(len in 0..length); length = len }

    /**
     * Feeds [prompt] and returns the logits after its last token. Reuses the cached common prefix (at least the last
     * prompt token is recomputed, since logits are not cached).
     */
    fun prefill(prompt: IntArray): FloatArray {
        require(prompt.isNotEmpty())
        var lcp = 0
        val max = minOf(length, prompt.size - 1)
        while (lcp < max && history[lcp] == prompt[lcp]) lcp++
        length = lcp
        return forward(prompt, lcp, prompt.size - lcp)
    }

    /** Appends one token and returns the next-token logits. */
    fun decode(token: Int): FloatArray = forward(intArrayOf(token), 0, 1)

    /**
     * Greedy (or [sampler]) continuation of [prompt]: stops after [maxNew] tokens or right after emitting a token in
     * [stopIds] (included in the result). Log-probabilities are of the full softmax at temperature 1.
     */
    fun generate(prompt: IntArray, maxNew: Int, stopIds: IntArray = IntArray(0), sampler: Sampler = Sampler.Greedy): Generation {
        val out = IntArray(maxNew); val lp = FloatArray(maxNew)
        var n = 0
        var stopped = false
        if (maxNew > 0) {
            var logits = prefill(prompt)
            while (true) {
                val tok = sampler.sample(logits)
                out[n] = tok; lp[n] = Sampler.logProb(logits, tok); n++
                if (tok in stopIds) { stopped = true; break }
                if (n == maxNew || length >= c.maxContext) break
                logits = decode(tok)
            }
        }
        return Generation(out.copyOf(n), lp.copyOf(n), stopped)
    }

    private fun ensureCapacity(need: Int) {
        if (need <= capacity) return
        require(need <= c.maxContext) { "context $need exceeds maxContext ${c.maxContext}" }
        val nc = minOf(c.maxContext, maxOf(need, capacity * 2))
        if (nkv != null) nkv.grow(nc, length)
        else {
            kT = Array(c.nLayers) { l -> Array(c.nKvHeads) { g -> Array(hd) { d -> kT[l][g][d].copyOf(nc) } } }
            vC = Array(c.nLayers) { l -> Array(c.nKvHeads) { g -> vC[l][g].copyOf(nc) } }
        }
        history = history.copyOf(nc)
        capacity = nc
    }

    /** Opt-in phase timers (ns): `-Dcompletionml.nn.profile=true`; read by the benchmark. */
    object Profile {
        @JvmField val enabled = System.getProperty("completionml.nn.profile") == "true"
        val names = arrayOf("matmul", "attention", "other", "alloc", "embed+norm", "rope+kv", "silu", "residual+norm", "copy-in/qact(sum)", "copy-out(sum)")
        @JvmField val ns = LongArray(names.size)
        /** thread-safe slot for the parallel copy-out inside the native matmul tasks */
        @JvmField val copyOut = java.util.concurrent.atomic.LongAdder()
        @JvmField val copyIn = java.util.concurrent.atomic.LongAdder()
        val matmulNs get() = ns[0]; val attentionNs get() = ns[1]
        fun reset() { ns.fill(0); copyOut.reset(); copyIn.reset() }
        override fun toString(): String {
            ns[9] = copyOut.sum(); ns[8] = copyIn.sum()
            return names.indices.filter { ns[it] != 0L }.joinToString(", ") { "%s %.1f ms".format(names[it], ns[it] / 1e6) }
        }
    }

    private inline fun <T> timed(what: Int, body: () -> T): T {
        if (!Profile.enabled) return body()
        val t0 = System.nanoTime()
        val r = body()
        Profile.ns[what] += System.nanoTime() - t0
        return r
    }

    private fun forward(tokens: IntArray, from: Int, n: Int): FloatArray {
        if (!Profile.enabled) return forward0(tokens, from, n)
        val tStart = System.nanoTime(); val before = Profile.ns.copyOf()
        val r = forward0(tokens, from, n)
        var accounted = 0L
        for (i in Profile.ns.indices) if (i != 2) accounted += Profile.ns[i] - before[i]
        Profile.ns[2] += (System.nanoTime() - tStart) - accounted
        return r
    }

    private fun forward0(tokens: IntArray, from: Int, n: Int): FloatArray {
        val m = model
        val p0 = length
        ensureCapacity(p0 + n)
        val d = c.dModel
        val tA = if (Profile.enabled) System.nanoTime() else 0L
        val b = buffers(n)
        val x = b[0]; val h = b[1]; val q = b[2]; val k = b[3]; val v = b[4]; val att = b[5]; val o = b[6]; val g = b[7]; val u = b[8]
        if (Profile.enabled) Profile.ns[3] += System.nanoTime() - tA
        for (t in 0 until n) history[p0 + t] = tokens[from + t]
        timed(4) { par(n) { t -> m.embed(tokens[from + t], x[t]); m.rmsNorm(x[t], m.layers[0].attnNorm, h[t]) } }
        val last = n - 1
        val ffnDim = c.ffnDim

        for (l in 0 until c.nLayers) {
            val layer = m.layers[l]
            // In the last layer only the last token's output matters: K/V for all tokens, everything else for one.
            val lastLayer = l == c.nLayers - 1 && n > 1
            timed(0) {
                if (!lastLayer) m.matmul(arrayOf(layer.wq, layer.wk, layer.wv), h, n, arrayOf(q, k, v))
                else {
                    m.matmul(arrayOf(layer.wk, layer.wv), h, n, arrayOf(k, v))
                    m.matmul(arrayOf(layer.wq), arrayOf(h[last]), 1, arrayOf(arrayOf(q[last])))
                }
            }
            timed(5) { par(n) { t ->
                val pos = p0 + t
                if (!lastLayer || t == last) m.rope(q[t], c.nHeads, pos)
                m.rope(k[t], c.nKvHeads, pos)
                if (nkv == null) for (kh in 0 until c.nKvHeads) {
                    val kt = kT[l][kh]
                    for (dd in 0 until hd) kt[dd][pos] = k[t][kh * hd + dd]
                    val vv = vC[l][kh][pos] ?: FloatArray(hd).also { vC[l][kh][pos] = it }
                    System.arraycopy(v[t], kh * hd, vv, 0, hd)
                }
            }
            nkv?.store(l, p0, k, v, n) }
            // the rest runs on tokens [t0, n)
            val t0 = if (lastLayer) last else 0
            val nn = n - t0
            fun <T> sub(a: Array<T>): Array<T> = if (t0 == 0) a else a.copyOfRange(t0, n)
            val xs = sub(x); val hs = sub(h); val atts = sub(att); val os = sub(o); val gs = sub(g); val us = sub(u)
            timed(1) { attention(l, sub(q), atts, p0 + t0, nn) }
            timed(0) { m.matmul(arrayOf(layer.wo), atts, nn, arrayOf(os)) }
            timed(7) { par(nn) { t -> Vec.add(os[t], xs[t], d); m.rmsNorm(xs[t], layer.ffnNorm, hs[t]) } }
            timed(0) { m.matmul(arrayOf(layer.w1, layer.w3), hs, nn, arrayOf(gs, us)) }
            timed(6) {
                if (native != null) par(nn) { t -> CmlNative.siluMul(gs[t], us[t], ffnDim) }
                else par(nn) { t -> val gt = gs[t]; val ut = us[t]; for (j in 0 until ffnDim) gt[j] = ftz(m.silu(gt[j]) * ut[j]) }
            }
            timed(0) { m.matmul(arrayOf(layer.w2), gs, nn, arrayOf(os)) }
            val next = if (l + 1 < c.nLayers) m.layers[l + 1].attnNorm else null
            timed(7) { par(nn) { t -> Vec.add(os[t], xs[t], d); if (next != null) m.rmsNorm(xs[t], next, hs[t]) } }
        }
        length = p0 + n
        val fin = FloatArray(d)
        m.rmsNorm(x[last], m.finalNorm, fin)
        val logits = FloatArray(c.vocabSize)
        timed(0) { m.matmul(arrayOf(m.lmHead), arrayOf(fin), 1, arrayOf(arrayOf(logits))) }
        return logits
    }

    // activation rows reused across forward passes (nothing relies on them being zeroed); grown to the largest n seen
    private var bufRows = 0
    private var bufs: Array<Array<FloatArray>> = emptyArray()
    private var bufViews: Array<Array<FloatArray>> = emptyArray()
    private fun buffers(n: Int): Array<Array<FloatArray>> {
        if (n > bufRows) {
            val widths = intArrayOf(c.dModel, c.dModel, c.dModel, c.kvDim, c.kvDim, c.dModel, c.dModel, c.ffnDim, c.ffnDim)
            val old = bufs
            bufs = Array(widths.size) { i -> Array(n) { t -> if (t < bufRows) old[i][t] else FloatArray(widths[i]) } }
            bufRows = n; bufViews = bufs
        }
        if (bufViews[0].size != n) bufViews = Array(bufs.size) { bufs[it].copyOfRange(0, n) }
        return bufViews
    }

    // per-worker attention scratch
    private val scores = Array(model.nThreads) { FloatArray(0) }
    private val headOut = Array(model.nThreads) { FloatArray(hd) }

    /** Runs [body] for t in 0 until n, in parallel blocks of 8 tokens when there are enough tokens. */
    private inline fun par(n: Int, crossinline body: (Int) -> Unit) {
        if (n < 16 || model.nThreads == 1) { for (t in 0 until n) body(t); return }
        val blocks = (n + 7) / 8
        model.executor.run(blocks) { b, _ -> for (t in b * 8 until minOf(n, b * 8 + 8)) body(t) }
    }

    private fun attention(l: Int, q: Array<FloatArray>, att: Array<FloatArray>, p0: Int, n: Int) {
        val group = c.nHeads / c.nKvHeads
        val scale = (1.0 / sqrt(hd.toDouble())).toFloat()
        val qBlock = if (n == 1) 1 else maxOf(1, minOf(32, n * c.nHeads / (model.nThreads * 4)))
        val blocks = (n + qBlock - 1) / qBlock
        if (nkv != null) {
            val workers = model.nThreads
            nkv.scratch(0, workers, p0 + n)   // sized once, outside the parallel region
            model.executor.run(c.nHeads * blocks) { task, worker ->
                val head = task / blocks; val b = task % blocks
                nkv.attention(l, head / group, q, att, b * qBlock, minOf(n, (b + 1) * qBlock), head * hd, scale, p0, nkv.scratch(worker, workers, p0 + n))
            }
            return
        }
        model.executor.run(c.nHeads * blocks) { task, worker ->
            val head = task / blocks; val b = task % blocks
            val kh = head / group
            val kt = kT[l][kh]; val vs = vC[l][kh]
            if (scores[worker].size < p0 + n) scores[worker] = FloatArray(capacity)
            val sc = scores[worker]; val ho = headOut[worker]
            for (t in b * qBlock until minOf(n, (b + 1) * qBlock)) {
                val len = p0 + t + 1
                val qt = q[t]; val qo = head * hd
                sc.fill(0f, 0, len)
                Attn.scores(qt, qo, scale, kt, hd, sc, len)
                var mx = Float.NEGATIVE_INFINITY
                for (s in 0 until len) if (sc[s] > mx) mx = sc[s]
                var sum = 0f
                for (s in 0 until len) { val z = sc[s] - mx; val e = if (z < -60f) 0f else exp(z); sc[s] = e; sum += e }
                val inv = 1f / sum
                ho.fill(0f)
                for (s in 0 until len) sc[s] *= inv
                Attn.values(sc, vs, len, ho, hd)
                System.arraycopy(ho, 0, att[t], qo, hd)
            }
        }
    }
}

/**
 * Flush-to-zero for tiny values. The JVM has no FTZ/DAZ mode and subnormal floats make SIMD multiplies ~100× slower,
 * so activations feeding the matmuls (and softmax weights) are flushed explicitly.
 */
internal fun ftz(v: Float): Float = if (kotlin.math.abs(v) < 1e-30f) 0f else v

/**
 * Attention inner loops in the 4-rows-per-pass, offset-0 form that C2 vectorises; methods kept above the inlining
 * size limit for stable compilation (see ScalarNnKernels).
 */
internal object Attn {
    /** `sc[s] += Σ_d q[qo + d] * scale * kT[d][s]` for s < len. */
    @JvmStatic fun scores(q: FloatArray, qo: Int, scale: Float, kT: Array<FloatArray>, hd: Int, sc: FloatArray, len: Int) {
        var d = 0
        while (d + 4 <= hd) {
            val a0 = q[qo + d] * scale; val a1 = q[qo + d + 1] * scale; val a2 = q[qo + d + 2] * scale; val a3 = q[qo + d + 3] * scale
            val k0 = kT[d]; val k1 = kT[d + 1]; val k2 = kT[d + 2]; val k3 = kT[d + 3]
            for (s in 0 until len) sc[s] += k0[s] * a0 + k1[s] * a1 + k2[s] * a2 + k3[s] * a3
            d += 4
        }
        if (d + 3 == hd) {
            val a0 = q[qo + d] * scale; val a1 = q[qo + d + 1] * scale; val a2 = q[qo + d + 2] * scale
            val k0 = kT[d]; val k1 = kT[d + 1]; val k2 = kT[d + 2]
            for (s in 0 until len) sc[s] += k0[s] * a0 + k1[s] * a1 + k2[s] * a2
            d += 3
        }
        if (d + 2 <= hd) {
            val a0 = q[qo + d] * scale; val a1 = q[qo + d + 1] * scale
            val k0 = kT[d]; val k1 = kT[d + 1]
            for (s in 0 until len) sc[s] += k0[s] * a0 + k1[s] * a1
            d += 2
        }
        while (d < hd) {
            val a = q[qo + d] * scale; val k0 = kT[d]
            for (s in 0 until len) sc[s] += k0[s] * a
            d++
        }
    }

    /** `out[j] += Σ_s p[s] * v[s][j]` for j < hd. */
    @JvmStatic fun values(p: FloatArray, v: Array<FloatArray?>, len: Int, out: FloatArray, hd: Int) {
        var s = 0
        while (s + 4 <= len) {
            val a0 = p[s]; val a1 = p[s + 1]; val a2 = p[s + 2]; val a3 = p[s + 3]
            val v0 = v[s]!!; val v1 = v[s + 1]!!; val v2 = v[s + 2]!!; val v3 = v[s + 3]!!
            for (j in 0 until hd) out[j] += v0[j] * a0 + v1[j] * a1 + v2[j] * a2 + v3[j] * a3
            s += 4
        }
        if (s + 2 <= len) {
            val a0 = p[s]; val a1 = p[s + 1]
            val v0 = v[s]!!; val v1 = v[s + 1]!!
            for (j in 0 until hd) out[j] += v0[j] * a0 + v1[j] * a1
            s += 2
        }
        while (s < len) {
            val a = p[s]; val v0 = v[s]!!
            for (j in 0 until hd) out[j] += v0[j] * a
            s++
        }
    }
}

class Generation(val tokens: IntArray, val logProbs: FloatArray, val stopped: Boolean) {
    /** Sum of log-probabilities (natural log); `exp(sumLogProb)` is the sequence probability under the model. */
    val sumLogProb: Double get() = logProbs.sumOf { it.toDouble() }
}

fun interface Sampler {
    fun sample(logits: FloatArray): Int

    companion object {
        val Greedy = Sampler { argmax(it) }

        /** Top-k sampling with temperature; [random] drives the draw (pass a seeded one for reproducibility). */
        fun topK(k: Int, temperature: Float = 1f, random: java.util.Random = java.util.Random()): Sampler = Sampler { logits ->
            val kk = minOf(k, logits.size)
            val idx = topKIndices(logits, kk)
            val mx = logits[idx[0]]
            val p = DoubleArray(kk) { Math.exp(((logits[idx[it]] - mx) / temperature).toDouble()) }
            var r = random.nextDouble() * p.sum()
            var pick = idx[kk - 1]
            for (i in 0 until kk) { r -= p[i]; if (r <= 0) { pick = idx[i]; break } }
            pick
        }

        fun argmax(a: FloatArray): Int {
            var best = 0
            for (i in 1 until a.size) if (a[i] > a[best]) best = i
            return best
        }

        /** Indices of the [k] largest logits, descending. */
        fun topKIndices(a: FloatArray, k: Int): IntArray {
            val idx = IntArray(k) { -1 }
            for (i in a.indices) {
                if (idx[k - 1] >= 0 && a[i] <= a[idx[k - 1]]) continue
                var j = k - 1
                while (j > 0 && (idx[j - 1] < 0 || a[idx[j - 1]] < a[i])) { idx[j] = idx[j - 1]; j-- }
                idx[j] = i
            }
            return idx
        }

        /** log softmax(logits)[token]. */
        fun logProb(logits: FloatArray, token: Int): Float {
            var mx = Float.NEGATIVE_INFINITY
            for (x in logits) if (x > mx) mx = x
            var s = 0.0
            for (x in logits) s += Math.exp((x - mx).toDouble())
            return (logits[token] - mx - Math.log(s)).toFloat()
        }
    }
}
