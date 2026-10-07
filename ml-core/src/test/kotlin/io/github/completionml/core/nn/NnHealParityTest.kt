package io.github.completionml.core.nn

import io.github.completionml.core.bpe.BpeTokenizer
import io.github.completionml.core.nn.native.NativeNnKernels
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Parity of [NnCompletion] (token healing, constrained first tokens, repetition guard, confidences, show decision)
 * with the Python harness on the real trained Go model. Fixture: `~/work/nn/eval/make_parity_heal.py` →
 * `~/work/ml-data/go/nn/parity-heal/heal.bin` (env `CML_NN_HEAL_PARITY`); model and vocabulary as in [NnParity].
 * Skipped when any of them is absent.
 */
class NnHealParityTest {
    class Record(
        val name: String, val mode: String, val kind: String, val path: ByteArray, val before: ByteArray, val after: ByteArray,
        val ctx: Int, val maxPrefix: Int, val suffixTokens: Int, val maxNew: Int, val typed: ByteArray, val prompt: IntArray,
        val gen: IntArray, val lps: FloatArray, val stop: String, val stopLp: Float, val confProd: Float, val confMin: Float,
        val punctOnly: Boolean, val repeated: Boolean, val show: Boolean, val miss: Boolean, val exact: Boolean,
        val text: ByteArray, val trueCode: ByteArray,
    )

    companion object {
        val defaultDir: File get() = File(System.getenv("CML_NN_HEAL_PARITY") ?: (System.getProperty("user.home") + "/work/ml-data/go/nn/parity-heal"))
        fun available() = File(defaultDir, "heal.bin").isFile && NnParity.defaultModel.isFile && NnParity.defaultVocab.isFile

        private fun DataInputStream.bytes(): ByteArray = ByteArray(readInt()).also { readFully(it) }
        private fun DataInputStream.u16s(n: Int): IntArray = IntArray(n) { readUnsignedShort() }
        private fun DataInputStream.f32s(n: Int): FloatArray = FloatArray(n) { readFloat() }

        fun load(dir: File = defaultDir): List<Record> {
            val out = ArrayList<Record>()
            DataInputStream(BufferedInputStream(FileInputStream(File(dir, "heal.bin")), 1 shl 20)).use { d ->
                require(d.readUTF() == "nn-heal-1")
                d.readInt() // vocab size
                repeat(d.readInt()) {
                    val name = d.readUTF(); val mode = d.readUTF(); val kind = d.readUTF()
                    val path = d.bytes(); val before = d.bytes(); val after = d.bytes()
                    val ctx = d.readInt(); val maxPrefix = d.readInt(); val sufTok = d.readInt(); val maxNew = d.readInt()
                    val typed = d.bytes(); val prompt = d.u16s(d.readInt())
                    val gen = d.u16s(d.readInt()); val lps = d.f32s(gen.size)
                    val stop = d.readUTF(); val stopLp = d.readFloat()
                    val confProd = d.readFloat(); val confMin = d.readFloat()
                    val punct = d.readInt() != 0; val rep = d.readInt() != 0; val show = d.readInt() != 0; val miss = d.readInt() != 0; val exact = d.readInt() != 0
                    val text = d.bytes(); val trueCode = d.bytes()
                    out += Record(name, mode, kind, path, before, after, ctx, maxPrefix, sufTok, maxNew, typed, prompt, gen, lps, stop, stopLp, confProd, confMin, punct, rep, show, miss, exact, text, trueCode)
                }
            }
            return out
        }
    }

    class Stats(val name: String) {
        var n = 0; var sameTyped = 0; var samePrompt = 0; var sameText = 0; var sameStop = 0; var sameShow = 0; var showUndecided = 0
        var maxConfDiff = 0f; var exactKotlin = 0; var exactPython = 0; var healed = 0
        val perKind = LinkedHashMap<String, IntArray>()
        val diffs = ArrayList<String>()
        override fun toString() = "%-10s n=%d typed %d prompt %d text %d (%.1f %%) stop %d show %d (+%d within 0.01 of the threshold) max|dconf| %.4f; line exact kotlin %d / python %d; healed %d\n%s".format(
            name, n, sameTyped, samePrompt, sameText, 100.0 * sameText / n, sameStop, sameShow, showUndecided, maxConfDiff, exactKotlin, exactPython, healed,
            perKind.entries.joinToString("\n") { (k, v) -> "    %-12s n=%d same text %d exact kotlin/python %d/%d".format(k, v[0], v[1], v[2], v[3]) })
    }

    fun run(weights: NnFormat.Model, kernels: NnKernels, threads: Int, tok: BpeTokenizer, records: List<Record>): Stats {
        val st = Stats(kernels.name)
        NnModel(weights, threads, kernels).use { model ->
            if (kernels is NativeNnKernels) kernels.prepare(weights.tensors.values.filterIsInstance<NnTensor.Q8>())
            val byOpts = HashMap<String, NnCompletion>()
            model.newSession(model.config.maxContext).use { s ->
                for (r in records) {
                    val key = "${r.mode}/${r.ctx}/${r.maxPrefix}/${r.suffixTokens}/${r.maxNew}"
                    val c = byOpts.getOrPut(key) {
                        NnCompletion(model, tok, NnCompletion.Options(
                            mode = if (r.mode == "plain") NnCompletion.Mode.PLAIN else NnCompletion.Mode.SPM,
                            ctx = r.ctx, maxPrefix = r.maxPrefix, suffixTokens = r.suffixTokens, maxNew = r.maxNew, showThreshold = 0.8,
                            trimClosersAfterCaret = false))   // the fixture is the untrimmed generation; healMode WORD_EOL = make_parity_heal.py --heal word-eol
                    }
                    s.truncate(0)
                    val res = c.complete(r.path, r.before, r.after, s)
                    st.n++
                    if (r.typed.isNotEmpty()) st.healed++
                    val typedOk = res.typed.contentEquals(r.typed); if (typedOk) st.sameTyped++
                    val promptOk = res.prompt.contentEquals(r.prompt); if (promptOk) st.samePrompt++
                    val textOk = res.text.contentEquals(r.text); if (textOk) st.sameText++
                    val stopName = res.stop.name.lowercase()
                    if (stopName == r.stop) st.sameStop++
                    if (textOk) st.maxConfDiff = maxOf(st.maxConfDiff, abs(res.confProd.toFloat() - r.confProd))
                    if (res.show == r.show) st.sameShow++ else if (abs(r.confProd - 0.8f) < 0.01f || abs(res.confProd - 0.8) < 0.01) st.showUndecided++
                    val ex = NnParity.lineExact(res.text, r.trueCode); if (ex) st.exactKotlin++; if (r.exact) st.exactPython++
                    val pk = st.perKind.getOrPut(r.kind) { IntArray(4) }
                    pk[0]++; if (textOk) pk[1]++; if (ex) pk[2]++; if (r.exact) pk[3]++
                    if ((!textOk || !typedOk || !promptOk) && st.diffs.size < 10)
                        st.diffs += "  ${r.name}: typed '${String(res.typed)}'/'${String(r.typed)}' prompt ${res.prompt.size}/${r.prompt.size} kotlin='${String(res.text).take(60)}' ($stopName %.3f) python='${String(r.text).take(60)}' (${r.stop} %.3f)".format(res.confProd, r.confProd)
                }
            }
        }
        return st
    }

    @Test fun healedCompletionMatchesPython() {
        assumeTrue(available(), "heal parity fixture / model / vocab not present: $defaultDir (${File(defaultDir, "heal.bin").isFile}), ${NnParity.defaultModel} (${NnParity.defaultModel.isFile}), ${NnParity.defaultVocab} (${NnParity.defaultVocab.isFile})")
        val all = load()
        val limit = (System.getProperty("completionml.nn.parity.heal") ?: "120").toInt()
        val records = if (all.size > limit) all.filterIndexed { i, _ -> i % ((all.size + limit - 1) / limit) == 0 } else all
        val weights = NnFormat.read(NnParity.defaultModel)
        val tok = BpeTokenizer.load(NnParity.defaultVocab.toPath())
        assertEquals(weights.meta["tokenizerSha256"], NnParity.sha256(NnParity.defaultVocab))
        val kernels = listOfNotNull(ScalarNnKernels, NativeNnKernels.loadOrNull(NativeNnKernels.Mode.F32), NativeNnKernels.loadOrNull(NativeNnKernels.Mode.Q8))
        for (k in kernels) {
            val st = run(weights, k, 4, tok, records)
            println("== heal parity, ${k.name}, ${records.size} of ${all.size} records\n$st")
            for (d in st.diffs) println(d)
            assertEquals(st.n, st.sameTyped, "${k.name}: typed remainder differs")
            assertEquals(st.n, st.samePrompt, "${k.name}: prompt ids differ")
            val q8 = k.name.endsWith("q8")
            assertTrue(st.sameText >= st.n * (if (q8) 0.85 else 0.97), "${k.name}: ${st.sameText}/${st.n} identical suggestions")
            assertTrue(st.maxConfDiff < (if (q8) 0.2f else 0.02f), "${k.name}: conf_prod differs by ${st.maxConfDiff} on identical lines")
            assertTrue(st.sameShow + st.showUndecided >= st.sameText, "${k.name}: show decision differs on identical lines")
        }
    }

    @Test fun fixtureContainsAllHealingKinds() {
        assumeTrue(available(), "heal parity fixture not present: $defaultDir")
        val kinds = load().map { it.kind }.toSet()
        assertTrue(kinds.containsAll(listOf("typed-space", "mid-ident")), "kinds $kinds")
    }
}
