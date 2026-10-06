package io.github.completionml.core.ngram

import io.github.completionml.core.util.LongFloatMap
import io.github.completionml.core.util.LongIntMap
import io.github.completionml.core.util.NgramHash
import io.github.completionml.core.vocab.Vocabulary
import java.util.PriorityQueue
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Modified Kneser–Ney estimation for corpora whose n-gram tables do not fit in memory. Same model as [NgramTrainer.estimate]
 * (MKN, SRILM-style pruning with recomputed backoff weights; verified equal by `PartitionedNgramTrainerTest`), different bookkeeping:
 *
 * Vocabulary ids are split into [partitions] bins (balanced by [weights], heavy tokens alone). A partition owns every n-gram whose
 * FIRST token is in the bin. That keeps all children of a context together (needed for totals and backoff weights) and, because the
 * continuation count of `g = w2..wn` is the number of distinct `x·g`, those `(n+1)`-grams are collected by their SECOND token. So per
 * partition one scan over the corpus fills raw tables (`first ∈ bin`: top order and repository counts) and distinct-n-gram sets
 * (`second ∈ bin`), from which the adjusted counts are derived. Each partition then computes per-context statistics from the full
 * counts, drops the pruned n-grams and keeps only `(ids, count)` of the survivors plus `(total, N1, N2, N3+)` of their contexts; the
 * tables are freed. Discounts need the global counts-of-counts, so the probabilities are computed afterwards from the (small) survivors.
 *
 * Peak memory is roughly `threads` partitions of tables; the corpus is scanned once per partition, in parallel.
 */
class PartitionedNgramTrainer(
    val order: Int, val vocab: Vocabulary,
    /** relative frequency of each vocabulary id as the first token of an n-gram (unigram counts; BOS = files × (order-1)) */
    weights: LongArray,
    val partitions: Int, val threads: Int,
    val minCounts: IntArray = IntArray(order) { 1 }, val minRepos: IntArray = IntArray(order) { 1 },
) {
    /** Calls the consumer for each training file: (ids in [vocab], number of ids, repository index); repositories must be contiguous. */
    fun interface Scan { fun run(consumer: (IntArray, Int, Int) -> Unit) }

    private class OrderOut(val n: Int) {
        var counts = IntArray(1 shl 10); var ids = IntArray((1 shl 10) * n); var size = 0
        fun add(src: IntArray, off: Int, c: Int) {
            if (size == counts.size) { counts = counts.copyOf(size * 2); ids = ids.copyOf(size * 2 * n) }
            System.arraycopy(src, off, ids, size * n, n); counts[size++] = c
        }
    }
    private class CtxOut {
        var h = LongArray(1 shl 10); var t = IntArray(1 shl 10); var n1 = IntArray(1 shl 10); var n2 = IntArray(1 shl 10); var n3 = IntArray(1 shl 10); var size = 0
        fun add(hash: Long, tt: Int, a: Int, b: Int, c: Int) {
            if (size == h.size) { val m = size * 2; h = h.copyOf(m); t = t.copyOf(m); n1 = n1.copyOf(m); n2 = n2.copyOf(m); n3 = n3.copyOf(m) }
            h[size] = hash; t[size] = tt; n1[size] = a; n2[size] = b; n3[size] = c; size++
        }
    }
    private class PartOut(order: Int) {
        val orders = Array(order) { OrderOut(it + 1) }
        val ctx = Array(order) { CtxOut() }
        val cc = Array(order) { LongArray(5) }
        var distinct = LongArray(order)      // distinct adjusted n-grams before pruning
    }

    /** bin of every vocabulary id */
    val partOf = IntArray(vocab.size)
    private val binWeight = LongArray(partitions)

    init {
        require(weights.size == vocab.size)
        val heap = PriorityQueue<LongArray>(compareBy<LongArray> { it[0] }.thenBy { it[1] })   // [load, bin]
        for (b in 0 until partitions) heap.add(longArrayOf(0, b.toLong()))
        for (w in vocab.let { (0 until it.size).sortedByDescending { id -> weights[id] } }) {
            val e = heap.poll(); partOf[w] = e[1].toInt(); e[0] += weights[w] + 1; binWeight[e[1].toInt()] = e[0]; heap.add(e)
        }
    }

    private fun processPartition(p: Int, scan: Scan, log: (String) -> Unit): PartOut {
        val trackRepos = BooleanArray(order) { minRepos[it] > 1 }
        val a = Array(order) { n -> if (n == order - 1 || trackRepos[n]) NgramTable(n + 1, 1 shl 16, trackRepos[n]) else null }
        val b = Array(order) { n -> if (n >= 1) NgramTable(n + 1, 1 shl 16) else null }
        var padded = IntArray(1 shl 12)
        val part = partOf
        scan.run { ids, len, repo ->
            val total = len + order
            if (padded.size < total) padded = IntArray(total + total / 2)
            for (i in 0 until order - 1) padded[i] = Vocabulary.BOS_ID
            System.arraycopy(ids, 0, padded, order - 1, len)
            padded[total - 1] = Vocabulary.EOS_ID
            for (end in order - 1 until total) {
                for (n in 1..order) {
                    val start = end - n + 1
                    if (part[padded[start]] == p) a[n - 1]?.add(padded, start, 1, repo)
                    if (n >= 2 && part[padded[start + 1]] == p) b[n - 1]!!.add(padded, start)
                }
            }
        }
        val out = PartOut(order)
        for (n in order downTo 1) {
            val adj: NgramTable
            if (n == order) adj = a[n - 1]!!
            else {
                val src = b[n]!!
                adj = NgramTable(n, src.size + 16)
                src.forEach { _, _, ids, off -> adj.add(ids, off + 1) }
                b[n] = null
            }
            val raw = a[n - 1]
            val total = LongIntMap(adj.size / 2 + 16); val n1 = LongIntMap(adj.size / 2 + 16); val n2 = LongIntMap(adj.size / 4 + 16); val n3 = LongIntMap(adj.size / 4 + 16)
            val cc = out.cc[n - 1]
            adj.forEach { _, c, ids, off ->
                if (c in 1..4) cc[c]++
                val ctx = NgramHash.of(ids, off, off + n - 1)
                total.addTo(ctx, c)
                when { c == 1 -> n1.addTo(ctx, 1); c == 2 -> n2.addTo(ctx, 1); else -> n3.addTo(ctx, 1) }
            }
            out.distinct[n - 1] = adj.size.toLong()
            val minCount = minCounts[n - 1]; val minRepo = minRepos[n - 1]
            val pruning = minCount > 1 || minRepo > 1
            val needed = if (pruning) LongIntMap(1024) else null
            val oo = out.orders[n - 1]
            adj.forEach { h, c, ids, off ->
                val keep = !pruning || (c >= minCount && (minRepo <= 1 || raw!!.reposOf(h) >= minRepo))
                if (keep) {
                    oo.add(ids, off, c)
                    needed?.put(NgramHash.of(ids, off, off + n - 1), 1)
                }
            }
            val co = out.ctx[n - 1]
            if (needed == null) total.forEach { ctx, t -> co.add(ctx, t, n1.get(ctx), n2.get(ctx), n3.get(ctx)) }
            else needed.forEach { ctx, _ -> co.add(ctx, total.get(ctx), n1.get(ctx), n2.get(ctx), n3.get(ctx)) }
            a[n - 1] = null
        }
        return out
    }

    fun train(scan: Scan, log: (String) -> Unit = {}): NgramModel {
        val t0 = System.currentTimeMillis()
        val results = arrayOfNulls<PartOut>(partitions)
        val pool = Executors.newFixedThreadPool(threads) { r -> Thread(r, "ngram-part").also { it.isDaemon = true } }
        val done = AtomicInteger()
        val order0 = (0 until partitions).sortedByDescending { binWeight[it] }
        try {
            pool.invokeAll(order0.map { p -> Callable {
                results[p] = processPartition(p, scan, log)
                val d = done.incrementAndGet()
                val kept = results[p]!!.orders.sumOf { it.size.toLong() }
                log("partition $p done ($d/$partitions): kept $kept n-grams, %.0f s, heap %.1f GB".format((System.currentTimeMillis() - t0) / 1000.0, (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1e9))
            } }).forEach { it.get() }
        } finally { pool.shutdown() }
        val parts = results.map { it!! }
        for (n in 1..order) log("order $n: ${parts.sumOf { it.distinct[n - 1] }} distinct (adjusted) n-grams, ${parts.sumOf { it.orders[n - 1].size.toLong() }} kept (min count ${minCounts[n - 1]}, min repos ${minRepos[n - 1]})")
        return estimate(parts, log)
    }

    private fun estimate(parts: List<PartOut>, log: (String) -> Unit): NgramModel {
        val keptTotal = (1..order).sumOf { n -> parts.sumOf { it.orders[n - 1].size.toLong() } }
        val probs = LongFloatMap((keptTotal * 11L / 10 + vocab.size).toInt().coerceAtLeast(1024))
        val backoffs = LongFloatMap(parts.sumOf { p -> (2..order).sumOf { p.ctx[it - 1].size.toLong() } }.toInt().coerceAtLeast(1024))
        val vocabSize = vocab.size.toDouble()
        for (n in 1..order) {
            val cc = LongArray(5); for (p in parts) for (i in 1..4) cc[i] += p.cc[n - 1][i]
            val d = NgramTrainer.discountsFromCounts(cc)
            log("order $n discounts: D1=%.3f D2=%.3f D3+=%.3f".format(d[0], d[1], d[2]))
            val ctxCount = parts.sumOf { it.ctx[n - 1].size }
            val total = LongIntMap(ctxCount + 16); val n1 = LongIntMap(ctxCount + 16); val n2 = LongIntMap(ctxCount + 16); val n3 = LongIntMap(ctxCount + 16)
            for (p in parts) { val c = p.ctx[n - 1]; for (i in 0 until c.size) { total.addTo(c.h[i], c.t[i]); n1.addTo(c.h[i], c.n1[i]); n2.addTo(c.h[i], c.n2[i]); n3.addTo(c.h[i], c.n3[i]) } }
            fun gamma(ctx: Long) = ((d[0] * n1.get(ctx) + d[1] * n2.get(ctx) + d[2] * n3.get(ctx)) / total.get(ctx)).toFloat()
            val lower = if (n == 1) null else NgramModel(n - 1, vocab, probs, backoffs)
            val pruning = minCounts[n - 1] > 1 || minRepos[n - 1] > 1
            val keptSum = LongFloatMap(if (pruning) total.size else 16)
            val keptLower = LongFloatMap(if (pruning) total.size else 16)
            var kept = 0
            for (part in parts) {
                val o = part.orders[n - 1]
                for (e in 0 until o.size) {
                    val off = e * n; val c = o.counts[e]
                    val h = NgramHash.of(o.ids, off, off + n)
                    val ctx = NgramHash.of(o.ids, off, off + n - 1)
                    val t = total.get(ctx).toDouble()
                    val disc = when { c == 1 -> d[0]; c == 2 -> d[1]; else -> d[2] }
                    val word = o.ids[off + n - 1]
                    val pl = if (lower == null) 1.0 / vocabSize else exp(lower.logProb(o.ids, off + n - 1, word).toDouble())
                    val p = max(c - disc, 0.0) / t + gamma(ctx) * pl
                    probs.put(h, ln(p).toFloat()); kept++
                    if (pruning) {
                        keptSum.put(ctx, (orZero(keptSum.get(ctx)) + p).toFloat())
                        keptLower.put(ctx, (orZero(keptLower.get(ctx)) + pl).toFloat())
                    }
                }
            }
            if (n > 1) {
                total.forEach { ctx, _ ->
                    if (!pruning) backoffs.put(ctx, ln(gamma(ctx).toDouble()).toFloat())
                    else {
                        val ks = keptSum.get(ctx)
                        if (!ks.isNaN()) backoffs.put(ctx, ln(((1.0 - ks) / max(1.0 - keptLower.get(ctx), 1e-6)).coerceIn(1e-6, 10.0)).toFloat())
                    }
                }
            } else {
                val g = gamma(NgramHash.EMPTY).toDouble()
                val unseenLog = ln(g / vocabSize).toFloat()
                for (w in 0 until vocab.size) { val h = NgramHash.extend(NgramHash.EMPTY, w); if (probs.get(h).isNaN()) probs.put(h, unseenLog) }
            }
            log("order $n: $kept n-grams in the model")
        }
        log("model: ${probs.size} probabilities, ${backoffs.size} backoff weights")
        return NgramModel(order, vocab, probs, backoffs)
    }

    private fun orZero(x: Float): Double = if (x.isNaN()) 0.0 else x.toDouble()
}
