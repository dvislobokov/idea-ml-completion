package io.github.completionml.core.ngram

import io.github.completionml.core.util.LongIntMap
import io.github.completionml.core.util.NgramHash
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/** Anything that gives log P(word | preceding tokens). */
interface TokenLm {
    fun logProb(context: IntArray, contextEnd: Int, word: Int): Float
}

/**
 * Dynamic n-gram model over the tokens seen so far in the current file ("cache"/"localness" component, Tu, Su & Devanbu 2014).
 * Interpolated absolute discounting within the cache (orders 1..[order]); the unigram backs off to a uniform distribution.
 * Feed tokens with [add] as the cursor moves; [logProb] uses the preceding `order-1` tokens of the context you pass.
 */
class CacheLm(val order: Int, private val vocabSize: Int, private val discount: Double = 0.5) : TokenLm {
    private val counts = LongIntMap(4096)      // hash(n-gram) -> count, all orders
    private val totals = LongIntMap(4096)      // hash(context) -> Σ counts of its continuations
    private val types = LongIntMap(4096)       // hash(context) -> number of distinct continuations
    private val history = IntArray(order - 1)
    private var seen = 0

    val tokens get() = seen

    /** Appends the next token of the file. */
    fun add(word: Int) {
        val n = minOf(seen, order - 1)
        var ctx = NgramHash.EMPTY
        // contexts of length 0..n ending at the previous token: history holds the last order-1 tokens (oldest first)
        for (k in 0..n) {
            if (k > 0) ctx = NgramHash.of(history, (order - 1) - k, order - 1)
            val h = NgramHash.extend(ctx, word)
            val before = counts.get(h)
            counts.addTo(h, 1)
            totals.addTo(ctx, 1)
            if (before == 0) types.addTo(ctx, 1)
        }
        if (order > 1) { System.arraycopy(history, 1, history, 0, order - 2); history[order - 2] = word }
        seen++
    }

    fun reset() { /* cheap enough to recreate; kept for API symmetry */ }

    /** P(word | context) within the cache; 0 < p ≤ 1 (uniform floor), as a probability. */
    fun prob(context: IntArray, contextEnd: Int, word: Int): Double {
        var p = 1.0 / vocabSize
        val maxCtx = minOf(order - 1, contextEnd)
        for (k in 0..maxCtx) {
            val ctx = if (k == 0) NgramHash.EMPTY else NgramHash.of(context, contextEnd - k, contextEnd)
            val t = totals.get(ctx)
            if (t == 0) break
            val c = counts.get(NgramHash.extend(ctx, word))
            val ty = types.get(ctx)
            p = max(c - discount, 0.0) / t + (discount * ty / t) * p
        }
        return p
    }

    override fun logProb(context: IntArray, contextEnd: Int, word: Int): Float = ln(prob(context, contextEnd, word)).toFloat()
}

/** Linear mixture P = λ·P_cache + (1−λ)·P_global, in log space. */
class MixedLm(private val global: TokenLm, private val cache: CacheLm, private val lambda: Double) : TokenLm {
    override fun logProb(context: IntArray, contextEnd: Int, word: Int): Float {
        val pg = exp(global.logProb(context, contextEnd, word).toDouble())
        if (cache.tokens == 0) return ln(pg).toFloat()
        val pc = cache.prob(context, contextEnd, word)
        return ln(lambda * pc + (1 - lambda) * pg).toFloat()
    }
}
