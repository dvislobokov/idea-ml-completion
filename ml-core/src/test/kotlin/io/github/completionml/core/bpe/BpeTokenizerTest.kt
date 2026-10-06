package io.github.completionml.core.bpe

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.random.Random
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class BpeTokenizerTest {
    private fun tok() = BpeTokenizer.load(checkNotNull(javaClass.classLoader.getResourceAsStream("bpe/vocab.bpe")))

    @Test
    fun layoutAndSpecials() {
        val t = tok()
        assertEquals(256 + t.mergeCount + t.specials.size, t.vocabSize)
        assertEquals(t.specialBase, t.endOfText)
        assertEquals(t.specialBase + 1, t.fimPrefix)
        assertEquals(t.specialBase + 2, t.fimMiddle)
        assertEquals(t.specialBase + 3, t.fimSuffix)
        assertEquals(t.specialBase + 4, t.fileSep)
        assertNull(t.specialId("<|nope|>"))
        assertTrue(t.isSpecial(t.pad) && !t.isSpecial(255))
    }

    @Test
    fun specialsAreNeverProducedFromText() {
        val t = tok()
        val ids = t.encode("<|endoftext|>")
        assertTrue(ids.none { t.isSpecial(it) })
        assertEquals("<|endoftext|>", t.decode(ids))
        // but ids inserted by the caller decode to their literal text
        assertEquals("a<|fim_middle|>b", t.decode(t.encode("a") + t.fimMiddle + t.encode("b")))
    }

    @Test
    fun stringApiRoundTrip() {
        val t = tok()
        for (s in listOf("", "x := 1\r\n", "func привет() {}", "log(\"🚀 ok\")", " \t \n\n\t")) {
            assertEquals(s, t.decode(t.encode(s)))
        }
    }

    @Test
    fun randomBytesRoundTrip() {
        val t = tok()
        val r = Random(5)
        repeat(300) {
            val b = ByteArray(r.nextInt(0, 700)) { if (r.nextInt(4) == 0) r.nextInt(256).toByte() else " \t\n\r_a(Z9".random(r).code.toByte() }
            assertContentEquals(b, t.decodeBytes(t.encodeBytes(b)))
        }
    }

    @Test
    fun offsetsAndLengths() {
        val t = tok()
        val b = "xxfunc main() {}yy".toByteArray()
        assertContentEquals(t.encode("func main() {}"), t.encodeBytes(b, 2, b.size - 4))
    }

    /** Opt-in benchmark: `CML_BPE_BENCH=<vocab.bpe>` and optionally `CML_BPE_BENCH_DIR` (a tree of .go files). */
    @Test
    fun throughput() {
        val vocab = System.getenv("CML_BPE_BENCH")
        assumeTrue(vocab != null, "set CML_BPE_BENCH=<vocab.bpe> to run the throughput benchmark")
        val dir = File(System.getenv("CML_BPE_BENCH_DIR") ?: (System.getProperty("user.home") + "/work/ml-data/go/repos/golang__go/src"))
        val files = dir.walkTopDown().filter { it.isFile && it.name.endsWith(".go") }.take(3000).map { it.readBytes() }.toList()
        val total = files.sumOf { it.size.toLong() }
        val t = BpeTokenizer.load(File(vocab!!).toPath())
        var ids = 0L
        repeat(3) { for (f in files) ids += t.encodeBytes(f).size }
        var best = Double.MAX_VALUE
        repeat(5) {
            val t0 = System.nanoTime()
            ids = 0
            for (f in files) ids += t.encodeBytes(f).size
            best = minOf(best, (System.nanoTime() - t0) / 1e9)
        }
        println("BPE encode: ${files.size} files, ${total / 1e6} MB, $ids ids, best ${"%.3f".format(best)} s = ${"%.1f".format(total / 1e6 / best)} MB/s single-threaded")
    }
}
