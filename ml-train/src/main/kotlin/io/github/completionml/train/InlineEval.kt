package io.github.completionml.train

import io.github.completionml.core.lex.Languages
import io.github.completionml.core.ngram.CacheLm
import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.vocab.Vocabulary
import java.io.File
import kotlin.math.exp
import kotlin.math.max

/**
 * Inline ("grey text") continuation with the n-gram LM: at sampled token positions of the held-out files the mixed model
 * (per-file cache + global) greedily generates up to `--max-tokens` tokens; we count how long a prefix of them equals the
 * real next tokens, whether the rest of the line is covered, and — since a real feature shows a suggestion only when it is
 * confident — precision and coverage at thresholds on the mean per-token probability of the first three tokens.
 */
private class InlineStats(maxTokens: Int) {
    var positions = 0L
    var lineStarts = 0L
    /** [k] = positions whose first k generated tokens are all right (k = 1..maxTokens). */
    val prefixOk = LongArray(maxTokens + 1)
    val prefixOkLineStart = LongArray(maxTokens + 1)
    /** rest of the line (≤ maxTokens tokens) generated exactly. */
    var lineCovered = 0L; var lineEligible = 0L
    /** per threshold: positions shown, positions shown with ≥3 tokens right, total correct tokens shown, total tokens shown */
    val shown = LongArray(THRESHOLDS.size); val shownOk3 = LongArray(THRESHOLDS.size)
    val shownCorrect = LongArray(THRESHOLDS.size); val shownTokens = LongArray(THRESHOLDS.size)
    var correctTokens = 0L

    fun add(o: InlineStats) {
        positions += o.positions; lineStarts += o.lineStarts; lineCovered += o.lineCovered; lineEligible += o.lineEligible; correctTokens += o.correctTokens
        for (i in prefixOk.indices) { prefixOk[i] += o.prefixOk[i]; prefixOkLineStart[i] += o.prefixOkLineStart[i] }
        for (i in THRESHOLDS.indices) { shown[i] += o.shown[i]; shownOk3[i] += o.shownOk3[i]; shownCorrect[i] += o.shownCorrect[i]; shownTokens[i] += o.shownTokens[i] }
    }

    companion object { val THRESHOLDS = doubleArrayOf(0.0, 0.5, 0.7, 0.8, 0.9) }
}

/** Uniform reservoir sample of [k] lines, shared by the file threads (`--dump`). */
private class Reservoir(private val k: Int) {
    val items = ArrayList<String>(); private var seen = 0L
    @Synchronized fun add(line: String) {
        seen++
        if (items.size < k) items.add(line) else { val j = java.util.concurrent.ThreadLocalRandom.current().nextLong(seen); if (j < k) items[j.toInt()] = line }
    }
    @Synchronized fun snapshot(): Pair<Long, List<String>> = seen to items.toList()
}

internal fun evalInline(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    val model = NgramModel.read(File(args.getValue("lm")))
    val cacheLambda = args["cache"]?.toDouble() ?: 0.3
    val maxTokens = args["max-tokens"]?.toInt() ?: 8
    val stride = args["stride"]?.toInt() ?: 50
    // --dump <file>: random samples of confident (≥ --dump-conf, default 0.8) positions, --dump-n (default 30) right (≥3 tokens) and as many wrong
    val dumpFile = args["dump"]?.let { File(it) }
    val dumpConf = args["dump-conf"]?.toDouble() ?: 0.8
    val dumpN = args["dump-n"]?.toInt() ?: 30
    val dumpOk = Reservoir(dumpN); val dumpBad = Reservoir(dumpN)
    val test = testSet(args, lang)
    val vocab = model.vocab
    val t0 = System.currentTimeMillis()
    log("inline continuation: ${test.size} files, up to $maxTokens tokens, every ${stride}th position, cache λ=$cacheLambda")

    val stats = test.indices().parallelStream().map { fileIndex ->
        val file = test.file(fileIndex, vocab)
        val st = InlineStats(maxTokens)
        if (file.ids.size < 2) return@map st
        val ids = file.ids
        val order = model.order
        val cache = if (cacheLambda > 0) CacheLm(3, vocab.size) else null
        // window: the last (order-1) real ids before the position, then the generated ones
        val window = IntArray(order - 1 + maxTokens)
        val generated = IntArray(maxTokens); val stepProb = DoubleArray(maxTokens)
        for (i in ids.indices) {
            if (i > 0 && i % stride == 0) {
                st.positions++
                val lineStart = file.lineStart[i]
                if (lineStart) st.lineStarts++
                for (k in 0 until order - 1) { val j = i - (order - 1) + k; window[k] = if (j < 0) Vocabulary.BOS_ID else ids[j] }
                // greedy generation; a predicted <ID> (identifier outside the vocabulary) stops it
                var n = 0
                while (n < maxTokens) {
                    val pos = order - 1 + n
                    val top = if (cache == null) model.topK(window, pos, 1) { it != Vocabulary.UNK_ID && it > Vocabulary.EOS_ID }
                              else topKMixed(model, cache, cacheLambda, window, pos, 1) { it != Vocabulary.UNK_ID && it > Vocabulary.EOS_ID }
                    if (top.isEmpty()) break
                    generated[n] = top[0].first; stepProb[n] = exp(top[0].second.toDouble())
                    window[pos] = generated[n]
                    n++
                }
                // longest correct prefix (an actual <ID> token never matches: the suggestion would be a wrong identifier)
                var match = 0
                while (match < n && i + match < ids.size && ids[i + match] != Vocabulary.UNK_ID && generated[match] == ids[i + match]) match++
                st.correctTokens += match
                for (k in 1..match) { st.prefixOk[k]++; if (lineStart) st.prefixOkLineStart[k]++ }
                val rest = file.restOfLine(i)
                if (rest in 1..maxTokens) { st.lineEligible++; if (match >= rest) st.lineCovered++ }
                // confidence: geometric mean probability of the first three generated tokens
                val head = minOf(3, n)
                if (head > 0) {
                    var lp = 0.0
                    for (k in 0 until head) lp += kotlin.math.ln(max(stepProb[k], 1e-9))
                    val conf = exp(lp / head)
                    for (t in InlineStats.THRESHOLDS.indices) if (conf >= InlineStats.THRESHOLDS[t]) {
                        st.shown[t]++; if (match >= 3) st.shownOk3[t]++
                        st.shownTokens[t] += head; st.shownCorrect[t] += minOf(match, head)
                    }
                    if (dumpFile != null && conf >= dumpConf) {
                        val ctx = (maxOf(0, i - 12) until i).joinToString(" ") { vocab.word(ids[it]) }
                        val gen = (0 until n).joinToString(" ") { vocab.word(generated[it]) }
                        val real = (i until minOf(ids.size, i + maxTokens)).joinToString(" ") { vocab.word(ids[it]) }
                        val line = "conf=%.2f match=%d%s | %s ||| gen: %s ||| real: %s".format(conf, match, if (lineStart) " line-start" else "", ctx, gen, real)
                        (if (match >= 3) dumpOk else dumpBad).add(line)
                    }
                }
            }
            cache?.add(ids[i])
        }
        st
    }.reduce(InlineStats(maxTokens)) { x, y -> InlineStats(maxTokens).also { it.add(x); it.add(y) } }

    val n = stats.positions.coerceAtLeast(1).toDouble()
    log("positions %d (line starts %d); correct tokens per position %.2f".format(stats.positions, stats.lineStarts, stats.correctTokens / n))
    log("P(first k tokens right): " + (1..maxTokens).joinToString("  ") { k -> "k=$k %.3f".format(stats.prefixOk[k] / n) })
    log("  at line starts:        " + (1..maxTokens).joinToString("  ") { k -> "k=$k %.3f".format(stats.prefixOkLineStart[k] / stats.lineStarts.coerceAtLeast(1).toDouble()) })
    log("rest of line (≤%d tokens) generated exactly: %.3f of %d eligible positions".format(maxTokens, stats.lineCovered / stats.lineEligible.coerceAtLeast(1).toDouble(), stats.lineEligible))
    for (t in InlineStats.THRESHOLDS.indices) {
        val shown = stats.shown[t].coerceAtLeast(1).toDouble()
        log("confidence ≥ %.1f: shown %.1f%% of positions, ≥3 tokens right in %.1f%% of them, token precision %.3f".format(
            InlineStats.THRESHOLDS[t], 100.0 * stats.shown[t] / n, 100.0 * stats.shownOk3[t] / shown, stats.shownCorrect[t] / stats.shownTokens[t].coerceAtLeast(1).toDouble()))
    }
    if (dumpFile != null) {
        val (nOk, ok) = dumpOk.snapshot(); val (nBad, bad) = dumpBad.snapshot()
        dumpFile.bufferedWriter().use { w ->
            w.write("# confidence ≥ $dumpConf: $nOk positions with ≥3 tokens right, $nBad with fewer; ${ok.size} + ${bad.size} random samples\n")
            w.write("## RIGHT\n"); for (l in ok) w.write(l + "\n")
            w.write("## WRONG\n"); for (l in bad) w.write(l + "\n")
        }
        log("dump: $nOk right / $nBad wrong confident positions, samples in $dumpFile")
    }
    log("%.1f s".format((System.currentTimeMillis() - t0) / 1000.0))
}
