package io.github.completionml.core.nn

import io.github.completionml.core.bpe.BpeTokenizer
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Token healing, constrained decoding and the show policy on the small committed BPE fixture and a random tiny model. */
class NnCompletionTest {
    private val tok = BpeTokenizer.load(checkNotNull(javaClass.classLoader.getResourceAsStream("bpe/vocab.bpe")))
    private val config = NnConfig(vocabSize = tok.vocabSize, dModel = 32, nLayers = 2, nHeads = 4, nKvHeads = 2, ffnDim = 64, maxContext = 512)
    private fun model() = NnModel(NnTestModels.inMemory(config, 11), 1, ScalarNnKernels)
    private fun b(s: String) = s.toByteArray(Charsets.UTF_8)

    // ------------------------------------------------------------------------------------------------ boundaries

    private fun typed(before: String, after: String): String {
        val c = NnCompletion(model(), tok, NnCompletion.Options(ctx = 400, maxNew = 8))
        val bb = b(before)
        val boundary = c.healedBoundary(bb, b(after))
        return String(bb, boundary, bb.size - boundary, Charsets.UTF_8)
    }

    @Test fun healingBoundaries() {
        assertEquals("(", typed("x := foo(", ")\n"))                 // `foo(⟨⟩)`: the run `()` is one pre-token
        assertEquals("()", typed("\tfoo()", ";\n"))                  // `foo()⟨⟩;`
        assertEquals("\"", typed("f(\"x\"", ")\n"))                  // `"x"⟨⟩)`: the run `")`
        assertEquals(" ", typed("x = ", "foo\n"))                    // typed space before a word: ` foo` is one pre-token
        assertEquals("Hi", typed("e.Hi", "gh\n"))                    // partial identifier
        assertEquals(" Hi", typed("x := e, Hi", "gh\n"))             // space + partial identifier
        assertEquals("", typed("x", " = 1\n"))                       // caret at the end of a word followed by a space
        assertEquals("", typed("func main() {\n    ", "foo()\n"))    // line start after the indentation
        assertEquals("", typed("x = 1", "\n"))                       // caret at the end of the line
        assertEquals("", typed("x = 1", ""))                         // no text after the caret
        assertEquals("", typed("x =  ", "\n"))                       // trailing whitespace before the newline
        assertEquals(" ", typed("a  ", "b\n"))                       // the run `  ` splits as ` ` + ` b`: the caret is inside ` b`
        assertEquals("", typed("", "x\n"))                           // empty file
    }

    @Test fun boundariesMatchEncoding() {
        // every boundary is a place where encode(prefix) + encode(rest) == encode(all)
        val text = b("package main\n\nfunc f(a int) {\n\tx := foo(a, \"s\")\n\treturn x\r\n}\n")
        val bounds = tok.preTokenBoundaries(text)
        assertEquals(0, bounds.first()); assertEquals(text.size, bounds.last())
        val all = tok.encodeBytes(text)
        for (p in bounds) {
            assertContentEquals(all, tok.encodeBytes(text, 0, p) + tok.encodeBytes(text, p, text.size - p), "boundary $p")
        }
        // `(` followed by `a` is a boundary; the closing quote followed by `)` is inside the run `")`
        val i = text.indexOf('('.code.toByte())
        assertTrue(bounds.contains(i + 1))
        val q = '"'.code.toByte()
        val j = (i + 1 until text.size).filter { text[it] == q }[1]   // closing quote of "s"
        assertFalse(bounds.contains(j + 1), "between `\"` and `)` of `\")`")
    }

    @Test fun longPreTokensAreChunkedAt128Bytes() {
        val text = ByteArray(300) { 'a'.code.toByte() }
        assertContentEquals(intArrayOf(0, 128, 256, 300), tok.preTokenBoundaries(text))
        assertEquals(256, tok.lastPreTokenBoundary(text, 0, 270, 300))
    }

    // ------------------------------------------------------------------------------------------------ prefix table

    @Test fun prefixTableMatchesBruteForce() {
        val idx = VocabPrefixIndex(tok)
        val rnd = Random(3)
        val rems = ArrayList<ByteArray>()
        rems += listOf(b(" "), b("("), b("()"), b("\""), b("\")"), b(" Hi"), b("re"), b("));"), b("é"), b(" \t"))
        repeat(60) { rems += tok.tokenBytes(rnd.nextInt(tok.specialBase)).let { t -> t.copyOf(rnd.nextInt(1, t.size + 1)) } }
        repeat(20) { rems += ByteArray(rnd.nextInt(1, 5)) { " a(z)_;9\"".random(rnd).code.toByte() } }
        for (rem in rems) {
            val allowed = idx.allowed(rem)
            val expect = (0 until tok.specialBase).filter { id ->
                val t = tok.tokenBytes(id)
                VocabPrefixIndex.startsWith(t, rem) || (t.size < rem.size && VocabPrefixIndex.startsWith(rem, t))
            }.toIntArray()
            assertContentEquals(expect, allowed, "remainder ${String(rem, Charsets.ISO_8859_1)}")
            assertTrue(allowed.isNotEmpty())
            for (id in allowed) {
                val left = idx.consume(rem, id)
                val t = tok.tokenBytes(id)
                if (VocabPrefixIndex.startsWith(t, rem)) assertEquals(0, left.size) else assertContentEquals(rem.copyOfRange(t.size, rem.size), left)
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ decoding

    @Test fun constrainedGenerationStartsWithTheTypedRemainder() {
        model().use { m ->
            val c = NnCompletion(m, tok, NnCompletion.Options(ctx = 400, maxNew = 12, repGuard = false))
            m.newSession(512).use { s ->
                val cases = listOf("x := foo(" to ")\n", "\tfoo()" to ";\n", "f(\"x\"" to ")\n", "x = " to "foo\n", "e.Hi" to "gh\n",
                    "if err != nil {\n\t\treturn" to " err\n", "x" to " = 1\n", "a.B(c, \"d\"" to ", e)\n")
                for ((before, after) in cases) {
                    val r = c.complete(b("main.go"), b(before), b(after), s)
                    val raw = tok.decodeBytes(r.tokens)
                    assertTrue(VocabPrefixIndex.startsWith(raw, r.typed), "'$before': generated ${String(raw)} does not start with typed '${String(r.typed)}'")
                    assertFalse(r.healMiss)
                    assertContentEquals(raw.copyOfRange(r.typed.size, raw.size), r.text)
                    // the healed prompt ends at the boundary: it equals the prompt of the text cut there
                    val cut = b(before).let { it.copyOf(it.size - r.typed.size) }
                    assertContentEquals(c.buildPrompt(b("main.go"), cut, cut.size, b(after)), r.prompt)
                    // probabilities: every log-prob <= 0, confidences consistent
                    assertTrue(r.logProbs.all { it <= 1e-6f })
                    assertTrue(r.confProd in 0.0..1.0 + 1e-9 && r.confMin >= r.confProd - 1e-9, "confProd ${r.confProd} confMin ${r.confMin}")
                    if (r.stop == NnCompletion.Stop.NEWLINE || r.stop == NnCompletion.Stop.SPECIAL) assertFalse(r.stopLogProb.isNaN()) else assertTrue(r.stopLogProb.isNaN())
                }
            }
        }
    }

    @Test fun maskedLogProbIsTheMaskedSoftmax() {
        val logits = FloatArray(10) { it * 0.5f }
        val ids = intArrayOf(1, 4, 7)
        val lp = NnCompletion.logProbOver(logits, ids, 4)
        val z = ids.sumOf { Math.exp(logits[it].toDouble()) }
        assertEquals(Math.log(Math.exp(2.0) / z), lp.toDouble(), 1e-5)
        assertEquals(7, NnCompletion.argmaxOver(logits, ids))
        assertEquals(9, NnCompletion.argmaxOver(logits, IntArray(10) { it }))
    }

    @Test fun withoutHealingThePromptEndsAtTheCaret() {
        model().use { m ->
            val c = NnCompletion(m, tok, NnCompletion.Options(ctx = 400, maxNew = 4, heal = false))
            m.newSession(512).use { s ->
                val r = c.complete(b("a.go"), b("x := foo("), b(")\n"), s)
                assertEquals(0, r.typed.size)
                assertContentEquals(c.buildPrompt(b("a.go"), b("x := foo("), 9, b(")\n")), r.prompt)
            }
        }
    }

    @Test fun repetitionGuard() {
        model().use { m ->
            val c = NnCompletion(m, tok, NnCompletion.Options(ctx = 400, maxNew = 48))
            val long = (0 until tok.specialBase).first { tok.tokenBytes(it).size >= 4 }
            // a >= 4-byte token repeated three times
            assertTrue(c.repeated(intArrayOf(long, long, long), 3))
            assertFalse(c.repeated(intArrayOf(long, long), 2))
            assertTrue(c.repeated(intArrayOf(5, long, long, long), 4))
            // single bytes: `000` (3 bytes) exempt, a 4-gram of bytes repeated 3x (4 bytes) caught
            assertFalse(c.repeated(intArrayOf('1'.code, '0'.code, '0'.code, '0'.code), 4))
            val g = IntArray(12) { "abcd"[it % 4].code }
            assertTrue(c.repeated(g, 12))
            assertFalse(c.repeated(g, 11))
            // 2-gram of two bytes repeated: 2 bytes, exempt
            assertFalse(c.repeated(intArrayOf(93, 91, 93, 91, 93, 91), 6))
        }
    }

    @Test fun showPolicy() {
        assertTrue(NnCompletion.punctOnly(b(");")))
        assertTrue(NnCompletion.punctOnly(b(" )]")))
        assertTrue(NnCompletion.punctOnly(b("}")))
        assertTrue(NnCompletion.punctOnly(b("")))
        assertFalse(NnCompletion.punctOnly(b("\"\"")))
        assertFalse(NnCompletion.punctOnly(b(" err)")))
        assertFalse(NnCompletion.punctOnly(b("1")))
        assertFalse(NnCompletion.punctOnly(b("é")))
        model().use { m ->
            m.newSession(512).use { s ->
                // with a threshold of 0 everything that is not punctuation-only / repeated is shown, with 1.01 nothing
                val lo = NnCompletion(m, tok, NnCompletion.Options(ctx = 400, maxNew = 8, showThreshold = 0.0))
                val r = lo.complete(b("a.go"), b("x := "), b("\n"), s)
                assertEquals(!r.punctOnly && !r.repeated && r.text.isNotEmpty(), r.show)
                val hi = NnCompletion(m, tok, NnCompletion.Options(ctx = 400, maxNew = 8, showThreshold = 1.01))
                assertFalse(hi.complete(b("a.go"), b("x := "), b("\n"), s).show)
            }
        }
    }

    @Test fun prefixCutIsTheEvalCut() {
        val text = b("line1\nline2\nline3\n")
        assertEquals(0, InlinePrompt.cutPrefix(text, text.size, 100))
        assertEquals(6, InlinePrompt.cutPrefix(text, text.size, 14))   // 18 - 14 = 4 -> forward to the start of line2
        assertEquals(12, InlinePrompt.cutPrefix(text, text.size, 7))   // 11 -> start of line3
        assertEquals(14, InlinePrompt.cutPrefix(text, 17, 3))          // no newline in [14, 17): the cut stays
    }
}
