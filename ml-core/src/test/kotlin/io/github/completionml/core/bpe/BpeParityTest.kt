package io.github.completionml.core.bpe

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.DataInputStream
import java.io.File
import java.io.InputStream
import java.util.zip.GZIPInputStream
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Parity of the Kotlin encoder with the Python reference (`~/work/nn/tokenizer/cmlbpe.py`).
 *
 * Fixtures are written by `make_parity.py` (format: "CMLPAR1\n" + gzip of big-endian records
 * `int32 n; n x { int32 nbytes, bytes, int32 nids, nids x uint16 }`):
 *  - small, committed: `src/test/resources/bpe/` (1k-id vocabulary, synthetic edge cases + fuzz + 120 real files), always runs;
 *  - big, not committed: 10 000 test-fold files with the production vocabulary, generated into
 *    `~/work/ml-data/tokenizer/parity-go` (override with env `CML_BPE_FIXTURE`); skipped when the directory is absent.
 */
class BpeParityTest {
    @Test
    fun smallCommittedFixture() {
        val vocab = resource("bpe/vocab.bpe")
        val data = resource("bpe/parity.bin.gz")
        val n = check(BpeTokenizer.load(vocab), data)
        assertTrue(n > 500, "fixture too small: $n")
    }

    @Test
    fun bigLocalFixture() {
        val dir = File(System.getenv("CML_BPE_FIXTURE") ?: (System.getProperty("user.home") + "/work/ml-data/tokenizer/parity-go"))
        val vocab = File(dir, "vocab.bpe")
        val data = File(dir, "parity.bin.gz")
        assumeTrue(vocab.isFile && data.isFile, "big BPE parity fixture not found in $dir (run make_parity.py); skipped")
        val n = check(BpeTokenizer.load(vocab.inputStream()), data.inputStream())
        println("BPE parity: $n cases identical")
    }

    private fun resource(name: String): InputStream =
        checkNotNull(javaClass.classLoader.getResourceAsStream(name)) { "missing test resource $name" }

    private fun check(tok: BpeTokenizer, gz: InputStream): Int {
        val din = DataInputStream(GZIPInputStream(gz.buffered(1 shl 16), 1 shl 16).buffered(1 shl 16))
        val magic = ByteArray(8).also { din.readFully(it) }
        assertEquals("CMLPAR1\n", String(magic, Charsets.US_ASCII))
        val n = din.readInt()
        var ids = 0L
        var bytes = 0L
        for (case in 0 until n) {
            val input = ByteArray(din.readInt()).also { din.readFully(it) }
            val expected = IntArray(din.readInt()) { din.readUnsignedShort() }
            val actual = tok.encodeBytes(input)
            if (!expected.contentEquals(actual)) {
                var i = 0
                while (i < minOf(expected.size, actual.size) && expected[i] == actual[i]) i++
                error("case $case (${input.size} bytes): first difference at id #$i, python=${expected.getOrNull(i)} kotlin=${actual.getOrNull(i)}; " +
                    "input around: ${String(tok.decodeBytes(actual, 0, minOf(i, actual.size)).takeLast(40).toByteArray(), Charsets.ISO_8859_1)}")
            }
            assertTrue(input.contentEquals(tok.decodeBytes(actual)), "case $case does not round-trip")
            ids += actual.size; bytes += input.size
        }
        println("BPE parity: $n cases, $bytes bytes, $ids ids")
        return n
    }
}
