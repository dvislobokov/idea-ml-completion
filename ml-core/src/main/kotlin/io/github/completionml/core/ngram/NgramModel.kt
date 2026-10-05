package io.github.completionml.core.ngram

import io.github.completionml.core.format.ModelFormat
import io.github.completionml.core.util.CompactFloatMap
import io.github.completionml.core.util.FloatLookup
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
    private val probs: FloatLookup,
    private val backoffs: FloatLookup,
) : TokenLm {
    /** log P(word | context), where [context] holds the preceding ids and the last `order-1` of them are used. */
    override fun logProb(context: IntArray, contextEnd: Int, word: Int): Float {
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

    /**
     * Fixed-context scorer: hashes of the `order-1` context suffixes and the cumulative backoff weights are computed once,
     * so scoring a candidate costs at most `order` hash extensions + lookups. Use it whenever several candidates share a context
     * (a completion list, top-k enumeration).
     */
    inner class Scorer(context: IntArray, contextEnd: Int) {
        private val maxCtx = minOf(order - 1, contextEnd)
        private val ctxHash = LongArray(maxCtx + 1)           // ctxHash[k] = hash of the last k context tokens
        private val backoffSum = FloatArray(maxCtx + 1)       // backoffSum[k] = sum of log γ for contexts of length k..1 that exist

        init {
            var h = NgramHash.EMPTY
            ctxHash[0] = h
            // build suffix hashes: last k tokens for k = 1..maxCtx; each needs its own chain, so rebuild per k (order ≤ 5: trivial)
            for (k in 1..maxCtx) ctxHash[k] = NgramHash.of(context, contextEnd - k, contextEnd)
            // cumulative backoff: falling from k to k-1 adds log γ(ctx_k) if that context exists
            var acc = 0f
            backoffSum[maxCtx] = 0f
            for (k in maxCtx downTo 1) {
                val b = backoffs.get(ctxHash[k])
                if (!b.isNaN()) acc += b
                backoffSum[k - 1] = acc
            }
            @Suppress("UNUSED_VALUE") h = 0L
        }

        fun logProb(word: Int): Float {
            for (k in maxCtx downTo 0) {
                val p = probs.get(NgramHash.extend(ctxHash[k], word))
                if (!p.isNaN()) return p + backoffSum[k]
            }
            return UNSEEN + backoffSum[0]
        }

        fun score(candidates: IntArray): FloatArray = FloatArray(candidates.size) { logProb(candidates[it]) }
    }

    fun scorer(context: IntArray, contextEnd: Int) = Scorer(context, contextEnd)

    /** Scores every candidate id; returns log-probabilities parallel to [candidates]. */
    fun score(context: IntArray, contextEnd: Int, candidates: IntArray): FloatArray = Scorer(context, contextEnd).score(candidates)

    /** Top-[k] next tokens over the whole vocabulary (O(|V|·order) lookups; evaluation and debugging only). */
    fun topK(context: IntArray, contextEnd: Int, k: Int, filter: (Int) -> Boolean = { true }): List<Pair<Int, Float>> {
        val best = ArrayList<Pair<Int, Float>>()
        val scorer = Scorer(context, contextEnd)
        for (w in 0 until vocab.size) {
            if (!filter(w)) continue
            val p = scorer.logProb(w)
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

    /** Writes the model; exact tables are converted to the compact quantised representation ([fingerprintBits] 16..32). */
    fun write(file: File, language: String, corpusId: String, fingerprintBits: Int = 24) {
        ModelFormat.write(file, ModelFormat.Header("ngram", language, signature(), System.currentTimeMillis(), corpusId)) { out ->
            out.writeInt(order)
            vocab.write(out)
            compact(probs, fingerprintBits).write(out)
            compact(backoffs, fingerprintBits).write(out)
        }
    }

    /** The same model with compact storage, as a plugin would load it (for measuring quantisation loss without a file). */
    fun compacted(fingerprintBits: Int = 24) = NgramModel(order, vocab, compact(probs, fingerprintBits), compact(backoffs, fingerprintBits))

    private fun compact(m: FloatLookup, bits: Int): CompactFloatMap = when (m) {
        is CompactFloatMap -> m
        is LongFloatMap -> CompactFloatMap.of(m, bits)
        else -> error("unknown map type")
    }

    private fun signature(): Long = order.toLong() * 1_000_003L + vocab.size

    companion object {
        /** Log-probability for a token whose unigram was never seen (outside vocabulary and no `<ID>` fallback). */
        const val UNSEEN = -20f

        fun read(file: File): NgramModel = read(file.inputStream(), file.toString())

        fun read(stream: java.io.InputStream, name: String = "ngram model"): NgramModel = ModelFormat.read(stream, "ngram", { _, inp ->
            val order = inp.readInt()
            val vocab = Vocabulary.read(inp)
            val probs = CompactFloatMap.read(inp)
            val backoffs = CompactFloatMap.read(inp)
            NgramModel(order, vocab, probs, backoffs)
        }, name)

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
