package io.github.completionml.core

import io.github.completionml.core.lex.GoLanguage
import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.ngram.NgramTrainer
import io.github.completionml.core.vocab.Vocabulary
import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NgramModelTest {
    private val files = listOf(
        "func f(a int) int { if a > 0 { return a } ; return 0 }",
        "func g(b int) int { if b > 1 { return b } ; return 1 }",
        "func h(c int) int { for c > 0 { c-- } ; return c }",
        "x := f(1) ; y := g(x) ; z := h(y)",
    )

    private fun train(order: Int): NgramModel {
        val vb = Vocabulary.Builder()
        val toks = files.map { GoLanguage.tokenizer.tokens(it) }
        toks.forEach { vb.addFile(it) }
        val vocab = vb.build(1000, minDocFreq = 1)
        val tr = NgramTrainer(order, vocab)
        toks.forEach { tr.addFile(vocab.encode(it)) }
        return tr.estimate()
    }

    @Test fun distributionsSumToOne() {
        val m = train(3)
        val v = m.vocab
        val contexts = listOf(intArrayOf(), intArrayOf(v.id("return")), intArrayOf(v.id("{"), v.id("return")), intArrayOf(v.id("zzz_unseen"), v.id("if")), intArrayOf(v.id("func"), v.id("f")))
        for (ctx in contexts) {
            var sum = 0.0
            for (w in 0 until v.size) sum += exp(m.logProb(ctx, w).toDouble())
            assertTrue(abs(sum - 1.0) < 1e-3, "context ${ctx.map { v.word(it) }}: sum = $sum")
        }
    }

    @Test fun seenContinuationsBeatUnseen() {
        val m = train(3)
        val v = m.vocab
        val ctx = intArrayOf(v.id("{"), v.id("return"))
        assertTrue(m.logProb(ctx, v.id("a")) > m.logProb(ctx, v.id("func")))
        assertTrue(m.logProb(intArrayOf(v.id("func")), v.id("f")) > m.logProb(intArrayOf(v.id("func")), v.id("return")))
    }

    @Test fun scorerMatchesLogProb() {
        val m = train(4)
        val v = m.vocab
        for (ctx in listOf(intArrayOf(), intArrayOf(v.id("return")), intArrayOf(v.id("zzz"), v.id("{"), v.id("return")), intArrayOf(v.id("func"), v.id("f"), v.id("("), v.id("a")))) {
            val s = m.scorer(ctx, ctx.size)
            for (w in 0 until v.size) assertEquals(m.logProb(ctx, w), s.logProb(w), "ctx ${ctx.toList()} word $w")
        }
    }

    @Test fun roundTripThroughFile() {
        val m = train(4)
        val f = File.createTempFile("lm-test", ".cml").also { it.deleteOnExit() }
        m.write(f, "go", "test")
        val r = NgramModel.read(f)
        assertEquals(m.order, r.order)
        assertEquals(m.vocab.size, r.vocab.size)
        val v = r.vocab
        val ctx = intArrayOf(v.id(";"), v.id("return"))
        for (w in 0 until v.size) assertEquals(m.logProb(ctx, w), r.logProb(ctx, w))
    }
}
