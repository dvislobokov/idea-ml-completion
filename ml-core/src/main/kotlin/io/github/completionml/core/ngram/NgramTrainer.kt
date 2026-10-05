package io.github.completionml.core.ngram

import io.github.completionml.core.util.LongFloatMap
import io.github.completionml.core.util.LongIntMap
import io.github.completionml.core.util.NgramHash
import io.github.completionml.core.vocab.Vocabulary
import kotlin.math.ln
import kotlin.math.max

/**
 * Counts n-grams and estimates an interpolated modified Kneser–Ney model (Chen & Goodman 1998; same formulas as SRILM/KenLM).
 *
 * Usage: `addFile(ids)` for every training file (ids from the same [Vocabulary]), then `estimate()`.
 * Each file is padded with `order-1` `<BOS>` and one `<EOS>`.
 */
class NgramTrainer(val order: Int, val vocab: Vocabulary, expectedTokens: Int = 1 shl 20) {
    private val tables = Array(order) { n -> NgramTable(n + 1, if (n == 0) vocab.size * 2 else expectedTokens / (order - n)) }
    var tokens = 0L; private set
    var files = 0; private set

    fun addFile(ids: IntArray) {
        files++
        tokens += ids.size + 1
        val padded = IntArray(ids.size + order)
        for (i in 0 until order - 1) padded[i] = Vocabulary.BOS_ID
        System.arraycopy(ids, 0, padded, order - 1, ids.size)
        padded[padded.size - 1] = Vocabulary.EOS_ID
        // every position i (>= order-1) ends an n-gram of every order 1..order; BOS-only n-grams are not counted
        for (end in order - 1 until padded.size) {
            for (n in 1..order) tables[n - 1].add(padded, end - n + 1)
        }
    }

    /**
     * @param minCounts per order (index = order-1): n-grams with raw count below the threshold are dropped before estimation
     *        (pruning of the highest orders is what keeps the model small; default keeps everything).
     */
    fun estimate(minCounts: IntArray = IntArray(order) { 1 }, log: (String) -> Unit = {}): NgramModel {
        for (n in order downTo 1) {
            val dropped = tables[n - 1].prune(minCounts[n - 1])
            log("order $n: ${tables[n - 1].size} n-grams kept, $dropped pruned (min count ${minCounts[n - 1]})")
        }
        // Adjusted counts: raw for the highest order, continuation counts N1+(• w2..wn) for lower orders.
        // Continuation count of an (n-1)-gram = number of distinct left extensions among kept n-grams.
        val adjusted = Array(order) { n -> if (n == order - 1) tables[n] else NgramTable(n + 1, tables[n].size) }
        for (n in order - 1 downTo 1) {
            val higher = tables[n]   // order n+1
            val lower = adjusted[n - 1]
            higher.forEach { _, _, ids, off -> lower.add(ids, off + 1) }   // suffix of length n
            log("order $n: ${lower.size} continuation n-grams")
        }
        val probs = LongFloatMap((adjusted.sumOf { it.size } * 11L / 10).toInt().coerceAtLeast(1024))
        val backoffs = LongFloatMap(adjusted.take(order - 1).sumOf { it.size }.coerceAtLeast(1024))
        val vocabSize = vocab.size.toDouble()
        for (n in 1..order) {
            val table = adjusted[n - 1]
            val d = discounts(table)
            log("order $n discounts: D1=%.3f D2=%.3f D3+=%.3f".format(d[0], d[1], d[2]))
            // per-context totals and N1/N2/N3+ counts
            val total = LongIntMap(table.size / 2 + 16)
            val n1 = LongIntMap(table.size / 2 + 16)
            val n2 = LongIntMap(table.size / 4 + 16)
            val n3 = LongIntMap(table.size / 4 + 16)
            table.forEach { _, c, ids, off ->
                val ctx = NgramHash.of(ids, off, off + n - 1)
                total.addTo(ctx, c)
                when { c == 1 -> n1.addTo(ctx, 1); c == 2 -> n2.addTo(ctx, 1); else -> n3.addTo(ctx, 1) }
            }
            // backoff weights
            total.forEach { ctx, t ->
                val gamma = (d[0] * n1.get(ctx) + d[1] * n2.get(ctx) + d[2] * n3.get(ctx)) / t
                if (n > 1) backoffs.put(ctx, ln(gamma).toFloat())
                else backoffs.put(ctx, ln(gamma).toFloat())  // unigram context (empty) kept for uniform interpolation below
            }
            val emptyCtx = NgramHash.EMPTY
            table.forEach { h, c, ids, off ->
                val ctx = NgramHash.of(ids, off, off + n - 1)
                val t = total.get(ctx).toDouble()
                val disc = when { c == 1 -> d[0]; c == 2 -> d[1]; else -> d[2] }
                val gamma = kotlin.math.exp(backoffs.get(ctx).toDouble())
                val lower = if (n == 1) 1.0 / vocabSize else kotlin.math.exp(probs.get(NgramHash.of(ids, off + 1, off + n)).toDouble().let { if (it.isNaN()) NgramModel.UNSEEN.toDouble() else it })
                val p = max(c - disc, 0.0) / t + gamma * lower
                probs.put(h, ln(p).toFloat())
            }
            if (n == 1) {
                // words never seen get the pure interpolation mass; <ID>/<BOS>/<EOS> included above if seen
                val gamma = kotlin.math.exp(backoffs.get(emptyCtx).toDouble())
                val unseenLog = ln(gamma / vocabSize).toFloat()
                for (w in 0 until vocab.size) {
                    val h = NgramHash.extend(emptyCtx, w)
                    if (probs.get(h).isNaN()) probs.put(h, unseenLog)
                }
            }
        }
        log("model: ${probs.size} probabilities, ${backoffs.size} backoff weights")
        return NgramModel(order, vocab, probs, backoffs)
    }

    /** Modified KN discounts from counts-of-counts: D1 = 1 − 2Y·n2/n1, D2 = 2 − 3Y·n3/n2, D3+ = 3 − 4Y·n4/n3, Y = n1/(n1+2n2). */
    private fun discounts(table: NgramTable): DoubleArray {
        val cc = LongArray(5)
        table.forEach { _, c, _, _ -> if (c in 1..4) cc[c]++ }
        val n1 = cc[1].toDouble().coerceAtLeast(1.0); val n2 = cc[2].toDouble().coerceAtLeast(1.0)
        val n3 = cc[3].toDouble().coerceAtLeast(1.0); val n4 = cc[4].toDouble().coerceAtLeast(1.0)
        val y = n1 / (n1 + 2 * n2)
        fun clamp(x: Double, lo: Double, hi: Double) = x.coerceIn(lo, hi)
        return doubleArrayOf(
            clamp(1 - 2 * y * n2 / n1, 0.1, 0.999),
            clamp(2 - 3 * y * n3 / n2, 0.1, 1.999),
            clamp(3 - 4 * y * n4 / n3, 0.1, 2.999),
        )
    }
}
