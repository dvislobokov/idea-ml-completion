package io.github.completionml.core.ngram

import io.github.completionml.core.util.LongFloatMap
import io.github.completionml.core.util.LongIntMap
import io.github.completionml.core.util.NgramHash
import io.github.completionml.core.vocab.Vocabulary
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Counts n-grams and estimates an interpolated modified Kneser–Ney model (Chen & Goodman 1998; same formulas as SRILM/KenLM).
 *
 * Usage: `addFile(ids, repo)` for every training file (ids from the same [Vocabulary]), then `estimate()`.
 * Each file is padded with `order-1` `<BOS>` and one `<EOS>`.
 *
 * Pruning is done the SRILM way: probabilities are estimated from the full counts, then selected n-grams are dropped and the
 * backoff weight of their context is recomputed so that the distribution still sums to one — the kept probabilities do not
 * change, the mass of the dropped n-grams flows to the lower order.
 */
class NgramTrainer(val order: Int, val vocab: Vocabulary, expectedTokens: Int = 1 shl 20, trackRepos: Boolean = false) {
    enum class Smoothing { MKN, JM }

    private val tables = Array(order) { n -> NgramTable(n + 1, if (n == 0) vocab.size * 2 else minOf(expectedTokens / (order - n), NgramTable.MAX_EXPECTED), trackRepos) }
    var tokens = 0L; private set
    var files = 0; private set

    /** @param repo index of the source repository; files must arrive grouped by repository for repo counts to be exact. */
    fun addFile(ids: IntArray, repo: Int = 0) {
        files++
        tokens += ids.size + 1
        val padded = IntArray(ids.size + order)
        for (i in 0 until order - 1) padded[i] = Vocabulary.BOS_ID
        System.arraycopy(ids, 0, padded, order - 1, ids.size)
        padded[padded.size - 1] = Vocabulary.EOS_ID
        // every position i (>= order-1) ends an n-gram of every order 1..order; BOS-only n-grams are not counted
        for (end in order - 1 until padded.size) {
            for (n in 1..order) tables[n - 1].add(padded, end - n + 1, 1, repo)
        }
    }

    /**
     * @param minCounts per order (index = order-1): n-grams with (adjusted) count below the threshold are pruned.
     * @param minRepos per order: n-grams seen in fewer distinct repositories are pruned (needs `trackRepos`).
     */
    /**
     * @param smoothing MKN — interpolated modified Kneser–Ney (continuation counts for lower orders);
     *                  JM — Jelinek–Mercer interpolation of raw relative frequencies with a fixed weight [jmLambda] per order
     *                  (Hellendoorn & Devanbu 2017 found JM better than MKN on source code).
     */
    fun estimate(minCounts: IntArray = IntArray(order) { 1 }, minRepos: IntArray = IntArray(order) { 1 },
                 smoothing: Smoothing = Smoothing.MKN, jmLambda: Double = 0.5, log: (String) -> Unit = {}): NgramModel {
        // MKN: raw counts for the highest order, continuation counts N1+(• w2..wn) for lower orders (from the full,
        // unpruned higher-order tables). JM: raw counts everywhere.
        val adjusted = Array(order) { n -> if (n == order - 1 || smoothing == Smoothing.JM) tables[n] else NgramTable(n + 1, tables[n].size) }
        if (smoothing == Smoothing.MKN) for (n in order - 1 downTo 1) {
            tables[n].forEach { _, _, ids, off -> adjusted[n - 1].add(ids, off + 1) }   // suffix of length n
            log("order $n: ${adjusted[n - 1].size} continuation n-grams (order ${n + 1}: ${tables[n].size})")
        }
        val probs = LongFloatMap((adjusted.sumOf { it.size } * 11L / 10).toInt().coerceAtLeast(1024))
        val backoffs = LongFloatMap(adjusted.take(order - 1).sumOf { it.size }.coerceAtLeast(1024))
        val vocabSize = vocab.size.toDouble()
        for (n in 1..order) {
            val table = adjusted[n - 1]
            val raw = tables[n - 1]
            val d = if (smoothing == Smoothing.MKN) discounts(table) else doubleArrayOf(0.0, 0.0, 0.0)
            if (smoothing == Smoothing.MKN) log("order $n discounts: D1=%.3f D2=%.3f D3+=%.3f".format(d[0], d[1], d[2]))
            // per-context totals and N1/N2/N3+ counts over the full adjusted counts
            val total = LongIntMap(table.size / 2 + 16)
            val n1 = LongIntMap(table.size / 2 + 16)
            val n2 = LongIntMap(table.size / 4 + 16)
            val n3 = LongIntMap(table.size / 4 + 16)
            table.forEach { _, c, ids, off ->
                val ctx = NgramHash.of(ids, off, off + n - 1)
                total.addTo(ctx, c)
                when { c == 1 -> n1.addTo(ctx, 1); c == 2 -> n2.addTo(ctx, 1); else -> n3.addTo(ctx, 1) }
            }
            // full-count backoff weights γ(h) = (D1·N1 + D2·N2 + D3·N3+) / C(h)
            val gammaFull = LongFloatMap(total.size)
            total.forEach { ctx, t -> gammaFull.put(ctx, if (smoothing == Smoothing.JM) jmLambda.toFloat() else ((d[0] * n1.get(ctx) + d[1] * n2.get(ctx) + d[2] * n3.get(ctx)) / t).toFloat()) }
            // lower-order model view (orders < n are final, including their pruning); its order limits the context it uses
            val lower = if (n == 1) null else NgramModel(n - 1, vocab, probs, backoffs)
            val minCount = minCounts[n - 1]; val minRepo = minRepos[n - 1]
            val pruning = minCount > 1 || minRepo > 1
            val keptSum = LongFloatMap(if (pruning) total.size else 16)
            val keptLower = LongFloatMap(if (pruning) total.size else 16)
            var kept = 0; var dropped = 0
            table.forEach { h, c, ids, off ->
                val ctx = NgramHash.of(ids, off, off + n - 1)
                val t = total.get(ctx).toDouble()
                val disc = when { c == 1 -> d[0]; c == 2 -> d[1]; else -> d[2] }
                val word = ids[off + n - 1]
                val pl = if (lower == null) 1.0 / vocabSize else exp(lower.logProb(ids, off + n - 1, word).toDouble())
                val p = if (smoothing == Smoothing.JM) (1 - jmLambda) * c / t + jmLambda * pl else max(c - disc, 0.0) / t + gammaFull.get(ctx) * pl
                val keep = !pruning || (c >= minCount && (minRepo <= 1 || raw.reposOf(h) >= minRepo))
                if (keep) {
                    probs.put(h, ln(p).toFloat()); kept++
                    if (pruning) {
                        keptSum.put(ctx, (orZero(keptSum.get(ctx)) + p).toFloat())
                        keptLower.put(ctx, (orZero(keptLower.get(ctx)) + pl).toFloat())
                    }
                } else dropped++
            }
            if (n > 1) {
                total.forEach { ctx, _ ->
                    if (!pruning) backoffs.put(ctx, ln(gammaFull.get(ctx).toDouble()).toFloat())
                    else {
                        val ks = keptSum.get(ctx)
                        if (!ks.isNaN()) {   // some entries kept: renormalise so the context still sums to one
                            val gamma = ((1.0 - ks) / max(1.0 - keptLower.get(ctx), 1e-6)).coerceIn(1e-6, 10.0)
                            backoffs.put(ctx, ln(gamma).toFloat())
                        }                    // nothing kept: no entry, γ = 1 (pure lower-order)
                    }
                }
            } else {
                // unigrams: words never seen get the interpolation mass γ/|V|
                val gamma = gammaFull.get(NgramHash.EMPTY).toDouble()
                val unseenLog = ln(gamma / vocabSize).toFloat()
                for (w in 0 until vocab.size) {
                    val h = NgramHash.extend(NgramHash.EMPTY, w)
                    if (probs.get(h).isNaN()) probs.put(h, unseenLog)
                }
            }
            log("order $n: $kept n-grams kept, $dropped pruned (min count $minCount, min repos $minRepo)")
        }
        log("model: ${probs.size} probabilities, ${backoffs.size} backoff weights")
        return NgramModel(order, vocab, probs, backoffs)
    }

    private fun orZero(x: Float): Double = if (x.isNaN()) 0.0 else x.toDouble()

    /** Modified KN discounts from counts-of-counts: D1 = 1 − 2Y·n2/n1, D2 = 2 − 3Y·n3/n2, D3+ = 3 − 4Y·n4/n3, Y = n1/(n1+2n2). */
    private fun discounts(table: NgramTable): DoubleArray {
        val cc = LongArray(5)
        table.forEach { _, c, _, _ -> if (c in 1..4) cc[c]++ }
        return discountsFromCounts(cc)
    }

    companion object {
        /** [cc] holds the counts of counts n1..n4 at indexes 1..4. */
        fun discountsFromCounts(cc: LongArray): DoubleArray {
            val n1 = cc[1].toDouble().coerceAtLeast(1.0); val n2 = cc[2].toDouble().coerceAtLeast(1.0)
            val n3 = cc[3].toDouble().coerceAtLeast(1.0); val n4 = cc[4].toDouble().coerceAtLeast(1.0)
            val y = n1 / (n1 + 2 * n2)
            return doubleArrayOf(
                (1 - 2 * y * n2 / n1).coerceIn(0.1, 0.999),
                (2 - 3 * y * n3 / n2).coerceIn(0.1, 1.999),
                (3 - 4 * y * n4 / n3).coerceIn(0.1, 2.999),
            )
        }
    }
}
