package io.github.completionml.core.ngram

import io.github.completionml.core.format.ModelFormat
import io.github.completionml.core.util.LongFloatMap
import io.github.completionml.core.util.NgramHash
import io.github.completionml.core.vocab.Vocabulary
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import kotlin.math.ln

/**
 * Queryable interpolated n-gram language model (natural-log probabilities).
 * Storage is KenLM-like: `probs[hash(w1..wn)]` holds the fully interpolated log P(wn | w1..wn-1); `backoffs[hash(w1..wk)]`
 * holds log γ for contexts of length 1..order-1. Query falls back to shorter contexts, multiplying backoff weights.
 */
class NgramModel(
    val order: Int,
    val vocab: Vocabulary,
    private val probs: LongFloatMap,
    private val backoffs: LongFloatMap,
) {
    /** log P(word | context), where [context] holds the preceding ids and the last `order-1` of them are used. */
    fun logProb(context: IntArray, contextEnd: Int, word: Int): Float {
        val maxCtx = minOf(order - 1, contextEnd)
        var backoff = 0f
        for (k in maxCtx downTo 0) {
            val h = NgramHash.extend(NgramHash.of(context, contextEnd - k, contextEnd), word)
            val p = probs.get(h)
            if (!p.isNaN()) return p + backoff
            if (k > 0) {
                val b = backoffs.get(NgramHash.of(context, contextEnd - k, contextEnd))
                if (!b.isNaN()) backoff += b
            }
        }
        return UNSEEN + backoff   // unigram missing: word not in vocabulary at training time
    }

    fun logProb(context: IntArray, word: Int): Float = logProb(context, context.size, word)

    /** Scores every candidate id; returns log-probabilities parallel to [candidates]. */
    fun score(context: IntArray, contextEnd: Int, candidates: IntArray): FloatArray =
        FloatArray(candidates.size) { logProb(context, contextEnd, candidates[it]) }

    /** Top-[k] next tokens over the whole vocabulary (slow: O(|V|·order); evaluation and debugging only). */
    fun topK(context: IntArray, contextEnd: Int, k: Int, filter: (Int) -> Boolean = { true }): List<Pair<Int, Float>> {
        val best = ArrayList<Pair<Int, Float>>()
        for (w in 0 until vocab.size) {
            if (!filter(w)) continue
            val p = logProb(context, contextEnd, w)
            if (best.size < k) { best.add(w to p); if (best.size == k) best.sortByDescending { it.second } }
            else if (p > best.last().second) {
                best[best.size - 1] = w to p
                var i = best.size - 1
                while (i > 0 && best[i].second > best[i - 1].second) { val t = best[i]; best[i] = best[i - 1]; best[i - 1] = t; i-- }
            }
        }
        return best
    }

    val entryCount get() = probs.size + backoffs.size

    fun write(file: File, language: String, corpusId: String) {
        ModelFormat.write(file, ModelFormat.Header("ngram", language, signature(), System.currentTimeMillis(), corpusId)) { out ->
            out.writeInt(order)
            vocab.write(out)
            writeMap(out, probs)
            writeMap(out, backoffs)
        }
    }

    private fun signature(): Long = order.toLong() * 1_000_003L + vocab.size

    companion object {
        /** Log-probability for a token whose unigram was never seen (outside vocabulary and no `<ID>` fallback). */
        const val UNSEEN = -20f

        fun read(file: File): NgramModel = ModelFormat.read(file, "ngram") { _, inp ->
            val order = inp.readInt()
            val vocab = Vocabulary.read(inp)
            val probs = readMap(inp)
            val backoffs = readMap(inp)
            NgramModel(order, vocab, probs, backoffs)
        }

        private fun writeMap(out: DataOutputStream, m: LongFloatMap) {
            out.writeInt(m.size)
            m.forEach { k, v -> out.writeLong(k); out.writeFloat(v) }
        }

        private fun readMap(inp: DataInputStream): LongFloatMap {
            val n = inp.readInt()
            val m = LongFloatMap(n)
            for (i in 0 until n) m.put(inp.readLong(), inp.readFloat())
            return m
        }

        fun ln(x: Double): Float = kotlin.math.ln(x).toFloat()
    }
}

/** Perplexity helper: accumulates log-probabilities of a token sequence. */
class PerplexityAccumulator {
    var tokens = 0L; private set
    var sumLogProb = 0.0; private set
    fun add(logProb: Float) { tokens++; sumLogProb += logProb }
    val perplexity: Double get() = if (tokens == 0L) Double.NaN else kotlin.math.exp(-sumLogProb / tokens)
    val entropyBits: Double get() = if (tokens == 0L) Double.NaN else -sumLogProb / tokens / ln(2.0)
}
