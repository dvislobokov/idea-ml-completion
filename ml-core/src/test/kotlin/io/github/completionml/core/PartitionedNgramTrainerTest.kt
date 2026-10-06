package io.github.completionml.core

import io.github.completionml.core.ngram.NgramTrainer
import io.github.completionml.core.ngram.PartitionedNgramTrainer
import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.vocab.Vocabulary
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class PartitionedNgramTrainerTest {
    private class Corpus(val vocab: Vocabulary, val files: List<IntArray>, val repos: IntArray)

    private fun corpus(seed: Long): Corpus {
        val rnd = Random(seed)
        val words = List(40) { "w$it" } + listOf("(", ")", "{", "}", ".", ",")
        fun tok(w: String) = MlToken(if (w.length > 1) TokenKind.IDENT else TokenKind.PUNCT, w, 0)
        val texts = List(300) { List(20 + rnd.nextInt(80)) { tok(words[(rnd.nextGaussian().let { g -> kotlin.math.abs(g) * 12 }).toInt().coerceAtMost(words.size - 1)]) } }
        val vb = Vocabulary.Builder(); texts.forEach { vb.addFile(it) }
        val vocab = vb.build(1000, minDocFreq = 1)
        return Corpus(vocab, texts.map { vocab.encode(it) }, IntArray(texts.size) { it / 25 })
    }

    private fun check(order: Int, minCounts: IntArray, minRepos: IntArray, partitions: Int) {
        val c = corpus(42L + order)
        val ref = NgramTrainer(order, c.vocab, trackRepos = true).also { t -> c.files.forEachIndexed { i, f -> t.addFile(f, c.repos[i]) } }
            .estimate(minCounts, minRepos)
        val weights = LongArray(c.vocab.size).also { w -> c.files.forEach { f -> f.forEach { id -> w[id]++ } } }
        val part = PartitionedNgramTrainer(order, c.vocab, weights, partitions, threads = 3, minCounts = minCounts, minRepos = minRepos)
            .train(PartitionedNgramTrainer.Scan { consumer -> c.files.forEachIndexed { i, f -> consumer(f, f.size, c.repos[i]) } })
        val rnd = Random(7)
        val ctx = IntArray(order - 1)
        repeat(300) {
            for (k in ctx.indices) ctx[k] = if (rnd.nextInt(10) == 0) Vocabulary.BOS_ID else rnd.nextInt(c.vocab.size)
            for (w in 0 until c.vocab.size) assertEquals(ref.logProb(ctx, w), part.logProb(ctx, w), 1e-5f, "order $order ctx ${ctx.toList()} word $w")
        }
        assertEquals(ref.entryCount, part.entryCount)
    }

    @Test fun equalsInMemoryTrainerUnpruned() = check(3, intArrayOf(1, 1, 1), intArrayOf(1, 1, 1), 4)
    @Test fun equalsInMemoryTrainerCountPruned() = check(4, intArrayOf(1, 1, 2, 3), intArrayOf(1, 1, 1, 1), 5)
    @Test fun equalsInMemoryTrainerRepoPruned() = check(4, intArrayOf(1, 1, 2, 2), intArrayOf(1, 1, 2, 3), 3)
    @Test fun singlePartition() = check(5, intArrayOf(1, 1, 2, 2, 2), intArrayOf(1, 1, 1, 2, 2), 1)
}
