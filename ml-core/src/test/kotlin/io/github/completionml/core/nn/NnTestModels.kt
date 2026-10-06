package io.github.completionml.core.nn

import io.github.completionml.core.nn.NnFormat.Names
import java.nio.ByteBuffer
import java.util.Random
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/** Random models for tests and the benchmark, plus a naive full-recompute reference forward pass. */
object NnTestModels {
    /** Random int8 weights (uniform bytes) with scales ~ 1/sqrt(in), norms around 1. */
    fun random(config: NnConfig, seed: Long = 1): List<NnTensor> {
        val rnd = Random(seed)
        return NnFormat.expectedShapes(config).map { (name, dims) ->
            if (dims.size == 1) NnTensor.F32(name, FloatArray(dims[0]) { 0.8f + 0.4f * rnd.nextFloat() })
            else {
                val (rows, cols) = dims[0] to dims[1]
                val q = ByteArray(rows * cols)
                rnd.nextBytes(q)
                for (i in q.indices) if (q[i] == Byte.MIN_VALUE) q[i] = -127
                val base = (1.0 / sqrt(rows.toDouble()) / 64.0).toFloat()  // |w| ≲ 2/sqrt(in)
                // the embedding gets a larger scale so that the inputs are not tiny
                val mul = if (name == Names.TOK_EMB) 8f else 1f
                NnTensor.Q8(name, rows, cols, ByteBuffer.wrap(q), FloatArray(cols) { base * mul * (0.5f + rnd.nextFloat()) })
            }
        }
    }

    fun inMemory(config: NnConfig, seed: Long = 1): NnFormat.Model =
        NnFormat.Model(mapOf("kind" to "nn"), config, random(config, seed).associateBy { it.name })

    /** Naive reference: all logits for every position of [tokens], full recompute, plain dequantised float math. */
    fun referenceLogits(m: NnFormat.Model, tokens: IntArray): Array<FloatArray> {
        val c = m.config
        val n = tokens.size; val d = c.dModel; val hd = c.headDim
        fun deq(name: String): Array<FloatArray> {
            val t = m.q8(name)
            return Array(t.rows) { i -> FloatArray(t.cols) { o -> t.get(i, o) } }
        }
        fun mm(x: FloatArray, w: Array<FloatArray>): FloatArray {
            val out = FloatArray(w[0].size)
            for (o in out.indices) { var s = 0.0; for (i in x.indices) s += x[i].toDouble() * w[i][o]; out[o] = s.toFloat() }
            return out
        }
        fun norm(x: FloatArray, w: FloatArray): FloatArray {
            var ss = 0.0; for (v in x) ss += v.toDouble() * v
            val inv = 1.0 / sqrt(ss / x.size + c.normEps)
            return FloatArray(x.size) { (x[it] * inv * w[it]).toFloat() }
        }
        fun rope(v: FloatArray, heads: Int, pos: Int) {
            val half = hd / 2
            for (h in 0 until heads) for (i in 0 until half) {
                val f = (1.0 / Math.pow(c.ropeTheta.toDouble(), 2.0 * i / hd)).toFloat()
                val a = (pos * f).toDouble()
                val x0 = v[h * hd + i]; val x1 = v[h * hd + i + half]
                v[h * hd + i] = (x0 * cos(a) - x1 * sin(a)).toFloat()
                v[h * hd + i + half] = (x0 * sin(a) + x1 * cos(a)).toFloat()
            }
        }
        val emb = deq(Names.TOK_EMB)
        val x = Array(n) { t -> FloatArray(d) { j -> emb[j][tokens[t]] } }
        for (l in 0 until c.nLayers) {
            fun f(nm: String) = m.f32(Names.layer(l, nm)).data
            val wq = deq(Names.layer(l, Names.WQ)); val wk = deq(Names.layer(l, Names.WK)); val wv = deq(Names.layer(l, Names.WV))
            val wo = deq(Names.layer(l, Names.WO)); val w1 = deq(Names.layer(l, Names.W1)); val w2 = deq(Names.layer(l, Names.W2)); val w3 = deq(Names.layer(l, Names.W3))
            val h = Array(n) { norm(x[it], f(Names.ATTN_NORM)) }
            val q = Array(n) { mm(h[it], wq).also { v -> rope(v, c.nHeads, it) } }
            val k = Array(n) { mm(h[it], wk).also { v -> rope(v, c.nKvHeads, it) } }
            val v = Array(n) { mm(h[it], wv) }
            val group = c.nHeads / c.nKvHeads
            for (t in 0 until n) {
                val att = FloatArray(d)
                for (head in 0 until c.nHeads) {
                    val kh = head / group
                    val s = DoubleArray(t + 1) { p -> var a = 0.0; for (j in 0 until hd) a += q[t][head * hd + j].toDouble() * k[p][kh * hd + j]; a / sqrt(hd.toDouble()) }
                    val mx = s.max(); val e = s.map { exp(it - mx) }; val z = e.sum()
                    for (j in 0 until hd) { var a = 0.0; for (p in 0..t) a += e[p] / z * v[p][kh * hd + j]; att[head * hd + j] = a.toFloat() }
                }
                val o = mm(att, wo)
                for (j in 0 until d) x[t][j] += o[j]
            }
            for (t in 0 until n) {
                val hh = norm(x[t], f(Names.FFN_NORM))
                val g = mm(hh, w1); val u = mm(hh, w3)
                val a = FloatArray(c.ffnDim) { (g[it] / (1 + exp(-g[it].toDouble())) * u[it]).toFloat() }
                val o = mm(a, w2)
                for (j in 0 until d) x[t][j] += o[j]
            }
        }
        val head = deq(if (c.tiedEmbeddings) Names.TOK_EMB else Names.LM_HEAD)
        return Array(n) { mm(norm(x[it], m.f32(Names.FINAL_NORM).data), head) }
    }
}
