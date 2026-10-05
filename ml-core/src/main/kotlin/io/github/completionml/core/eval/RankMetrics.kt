package io.github.completionml.core.eval

import io.github.completionml.core.rank.TrainingExample
import io.github.completionml.core.spi.ContextKind

/** Top-1 / top-5 / MRR accumulated overall and per [ContextKind]. */
class RankMetrics {
    class Bucket { var n = 0L; var top1 = 0L; var top5 = 0L; var rr = 0.0; var listSize = 0L
        val top1Rate get() = if (n == 0L) 0.0 else top1.toDouble() / n
        val top5Rate get() = if (n == 0L) 0.0 else top5.toDouble() / n
        val mrr get() = if (n == 0L) 0.0 else rr / n
        val avgList get() = if (n == 0L) 0.0 else listSize.toDouble() / n
    }
    val all = Bucket()
    val byKind = ContextKind.values().associateWith { Bucket() }

    /** [scores] parallel to the example's candidates; higher is better. Ties are broken pessimistically (answer last). */
    fun add(example: TrainingExample, scores: FloatArray) {
        val s = scores[example.chosen]
        var rank = 1
        for (i in scores.indices) if (i != example.chosen && scores[i] >= s) rank++
        for (b in listOf(all, byKind.getValue(example.kind))) {
            b.n++; b.listSize += example.size
            if (rank == 1) b.top1++
            if (rank <= 5) b.top5++
            b.rr += 1.0 / rank
        }
    }

    fun report(title: String): String = buildString {
        appendLine("$title: n=${all.n}  top1=%.3f  top5=%.3f  MRR=%.3f  avg list=%.1f".format(all.top1Rate, all.top5Rate, all.mrr, all.avgList))
        for ((k, b) in byKind) if (b.n > 0)
            appendLine("  %-16s n=%-7d top1=%.3f  top5=%.3f  MRR=%.3f".format(k.name, b.n, b.top1Rate, b.top5Rate, b.mrr))
    }
}
