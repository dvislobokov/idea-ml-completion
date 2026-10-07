package io.github.completionml.core.nn

import io.github.completionml.core.bpe.BpeTokenizer
import io.github.completionml.core.nn.native.NativeLib
import io.github.completionml.core.nn.native.NativeNnKernels
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.Base64
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * End-to-end parity of the Kotlin inference with the PyTorch model on real trained weights. Fixture written by
 * `~/work/nn/eval/make_parity.py` (see its docstring for the layout), default location `~/work/ml-data/go/nn/parity`
 * (env `CML_NN_PARITY`), model `~/work/ml-data/go/models/go-nn-31m-e1.cml` (env `CML_NN_MODEL`), vocabulary
 * `~/work/ml-data/tokenizer/go-16384.bpe` (env `CML_BPE_VOCAB`). Documented in docs/NN-PARITY.md.
 *
 * Run as a main (test classpath: `./gradlew :ml-core:printBenchClasspath`):
 *   NnParityKt --threads 8 --kernels scalar,native-f32,native-q8 [--behav 500] [--out <dir for kotlin-*.tsv>]
 */
object NnParity {
    class LogitPrompt(
        val name: String, val mode: String, val pos: Int, val path: ByteArray, val prefix: ByteArray, val suffix: ByteArray,
        val ctx: Int, val maxPrefix: Int, val suffixTokens: Int, val ids: IntArray,
        val logits: FloatArray, val logitsFq: FloatArray,
        val gen: IntArray, val genLp: FloatArray, val genFq: IntArray, val genFqLp: FloatArray,
    )

    class BehavRecord(
        val pos: Int, val mode: String, val kind: String, val ids: IntArray, val trueCode: ByteArray,
        val genEval: ByteArray, val stopEval: String, val genFq: ByteArray, val genFqIds: IntArray, val stopFq: String, val lpFq: FloatArray,
    )

    class Fixture(val dir: File, val vocabSize: Int, val prompts: List<LogitPrompt>, val behav: List<BehavRecord>)

    val defaultDir: File get() = File(System.getenv("CML_NN_PARITY") ?: (System.getProperty("user.home") + "/work/ml-data/go/nn/parity"))
    val defaultModel: File get() = File(System.getenv("CML_NN_MODEL") ?: (System.getProperty("user.home") + "/work/ml-data/go/models/go-nn-31m-e1.cml"))
    val defaultVocab: File get() = File(System.getenv("CML_BPE_VOCAB") ?: (System.getProperty("user.home") + "/work/ml-data/tokenizer/go-16384.bpe"))

    fun available(): Boolean = File(defaultDir, "meta.json").isFile && File(defaultDir, "prompts.bin").isFile && File(defaultDir, "behav.bin").isFile && defaultModel.isFile && defaultVocab.isFile

    // ------------------------------------------------------------------------------------------------ fixture reading

    private fun DataInputStream.bytes(): ByteArray = ByteArray(readInt()).also { readFully(it) }
    private fun DataInputStream.u16s(n: Int): IntArray = IntArray(n) { readUnsignedShort() }
    private fun DataInputStream.i32s(n: Int): IntArray = IntArray(n) { readInt() }
    private fun DataInputStream.f32s(n: Int): FloatArray = FloatArray(n) { readFloat() }

    fun load(dir: File = defaultDir, behavLimit: Int = Int.MAX_VALUE): Fixture {
        val prompts = ArrayList<LogitPrompt>()
        var vocab = 0
        DataInputStream(BufferedInputStream(FileInputStream(File(dir, "prompts.bin")), 1 shl 20)).use { d ->
            require(d.readUTF() == "nn-parity-1")
            vocab = d.readInt()
            repeat(d.readInt()) {
                val name = d.readUTF(); val mode = d.readUTF(); val pos = d.readInt()
                val path = d.bytes(); val pre = d.bytes(); val suf = d.bytes()
                val ctx = d.readInt(); val maxPrefix = d.readInt(); val sufTok = d.readInt()
                val ids = d.u16s(d.readInt())
                val v = d.readInt(); val lg = d.f32s(v); val lgq = d.f32s(v)
                val n1 = d.readInt(); val g1 = d.i32s(n1); val lp1 = d.f32s(n1)
                val n2 = d.readInt(); val g2 = d.i32s(n2); val lp2 = d.f32s(n2)
                prompts += LogitPrompt(name, mode, pos, path, pre, suf, ctx, maxPrefix, sufTok, ids, lg, lgq, g1, lp1, g2, lp2)
            }
        }
        val behav = ArrayList<BehavRecord>()
        DataInputStream(BufferedInputStream(FileInputStream(File(dir, "behav.bin")), 1 shl 20)).use { d ->
            require(d.readUTF() == "nn-behav-1")
            require(d.readInt() == vocab)
            val n = d.readInt()
            for (i in 0 until n) {
                val pos = d.readInt(); val mode = d.readUTF(); val kind = d.readUTF()
                val ids = d.u16s(d.readInt())
                val trueCode = d.bytes(); val genEval = d.bytes(); val stopEval = d.readUTF()
                val genFq = d.bytes(); val genFqIds = d.u16s(d.readInt()); val stopFq = d.readUTF(); val lp = d.f32s(d.readInt())
                if (behav.size < behavLimit) behav += BehavRecord(pos, mode, kind, ids, trueCode, genEval, stopEval, genFq, genFqIds, stopFq, lp)
            }
        }
        return Fixture(dir, vocab, prompts, behav)
    }

    // ------------------------------------------------------------------------------------------------ measurements

    class LogitStats(val name: String) {
        var n = 0
        var maxAbs = 0f; var sumMaxAbs = 0.0
        var argmaxAgree = 0
        var fullAgree = 0; var sumFirstDiv = 0.0; var minFirstDiv = Int.MAX_VALUE
        var lpSq = 0.0; var lpN = 0; var lpMaxAbs = 0f
        val lines = ArrayList<String>()
        fun add(maxAbsLogit: Float, argmaxOk: Boolean, firstDiv: Int, genLen: Int, lpErr: FloatArray) {
            n++
            maxAbs = maxOf(maxAbs, maxAbsLogit); sumMaxAbs += maxAbsLogit
            if (argmaxOk) argmaxAgree++
            if (firstDiv >= genLen) fullAgree++
            sumFirstDiv += minOf(firstDiv, genLen); minFirstDiv = minOf(minFirstDiv, minOf(firstDiv, genLen))
            for (e in lpErr) { lpSq += e.toDouble() * e; lpN++; lpMaxAbs = maxOf(lpMaxAbs, abs(e)) }
        }
        val lpRmse: Double get() = if (lpN == 0) 0.0 else sqrt(lpSq / lpN)
        override fun toString() = "%-12s n=%d  max|dlogit| %.4f (mean %.4f)  argmax %d/%d  greedy48 identical %d/%d (mean first divergence %.1f, min %d)  logprob rmse %.5f max %.4f"
            .format(name, n, maxAbs, sumMaxAbs / maxOf(1, n), argmaxAgree, n, fullAgree, n, sumFirstDiv / maxOf(1, n), if (minFirstDiv == Int.MAX_VALUE) 0 else minFirstDiv, lpRmse, lpMaxAbs)
    }

    class BehavStats(val name: String) {
        var n = 0; var sameAsEval = 0; var sameAsFq = 0; var exact = 0; var exactEval = 0; var exactFq = 0
        var sameStopFq = 0; var firstTokSameFq = 0
        val perMode = LinkedHashMap<String, IntArray>()  // mode -> [n, sameEval, sameFq, exact, exactEval, exactFq]
        var totalMs = 0.0
        val diffs = ArrayList<String>()
        fun add(r: BehavRecord, gen: ByteArray, genIds: IntArray, stop: String, ms: Double) {
            n++; totalMs += ms
            val sEval = gen.contentEquals(r.genEval); val sFq = gen.contentEquals(r.genFq)
            val ex = lineExact(gen, r.trueCode); val exE = lineExact(r.genEval, r.trueCode); val exF = lineExact(r.genFq, r.trueCode)
            if (sEval) sameAsEval++; if (sFq) sameAsFq++; if (ex) exact++; if (exE) exactEval++; if (exF) exactFq++
            if (stop == r.stopFq) sameStopFq++
            if ((genIds.isEmpty() && r.genFqIds.isEmpty()) || (genIds.isNotEmpty() && r.genFqIds.isNotEmpty() && genIds[0] == r.genFqIds[0])) firstTokSameFq++
            val pm = perMode.getOrPut(r.mode) { IntArray(6) }
            pm[0]++; if (sEval) pm[1]++; if (sFq) pm[2]++; if (ex) pm[3]++; if (exE) pm[4]++; if (exF) pm[5]++
            if (!sFq && diffs.size < 12) diffs += "  pos ${r.pos} ${r.mode} ${r.kind}: kotlin=${show(gen)} ($stop) fq32=${show(r.genFq)} (${r.stopFq}) eval=${show(r.genEval)} true=${show(r.trueCode)}"
        }
        override fun toString(): String {
            val sb = StringBuilder()
            sb.append("%-12s n=%d  same line as fp32-fq %d (%.1f %%), as eval(bf16) %d (%.1f %%); first token = fq %d; stop kind = fq %d; line exact vs truth: kotlin %.1f %% / fp32-fq %.1f %% / eval %.1f %%; %.0f ms/prompt\n"
                .format(name, n, sameAsFq, 100.0 * sameAsFq / n, sameAsEval, 100.0 * sameAsEval / n, firstTokSameFq, sameStopFq,
                    100.0 * exact / n, 100.0 * exactFq / n, 100.0 * exactEval / n, totalMs / n))
            for ((m, v) in perMode) sb.append("    %-6s n=%d same-fq %d same-eval %d exact kotlin/fq/eval %d/%d/%d\n".format(m, v[0], v[2], v[1], v[3], v[5], v[4]))
            return sb.toString()
        }
    }

    /** Simplified `exact`: generated line (right-trimmed) equals the true rest of the line; a trailing `// comment`
     * outside strings is dropped like `code_part` does. The authoritative scoring is `score_parity.py`. */
    fun lineExact(gen: ByteArray, trueCode: ByteArray): Boolean {
        val g = stripTrailingComment(String(gen, Charsets.ISO_8859_1)).trimEnd()
        val t = String(trueCode, Charsets.ISO_8859_1).trimEnd()
        return g.isNotEmpty() && g == t
    }

    private fun stripTrailingComment(s: String): String {
        var inStr = false; var q = ' '
        var i = 0
        while (i < s.length) {
            val c = s[i]
            if (inStr) { if (c == '\\' && q != '`') i++ else if (c == q) inStr = false }
            else if (c == '"' || c == '`' || c == '\'') { inStr = true; q = c }
            else if (c == '/' && i + 1 < s.length && s[i + 1] == '/') return s.substring(0, i)
            i++
        }
        return s
    }

    private fun show(b: ByteArray) = "'" + String(b, Charsets.UTF_8).replace("\n", "\\n").replace("\t", "\\t").take(60) + "'"

    fun kernelsByName(name: String): NnKernels = when (name) {
        "scalar" -> ScalarNnKernels
        "native-f32" -> NativeNnKernels.loadOrNull(NativeNnKernels.Mode.F32) ?: error("native: ${NativeLib.status}")
        "native-q8" -> NativeNnKernels.loadOrNull(NativeNnKernels.Mode.Q8) ?: error("native: ${NativeLib.status}")
        else -> error("unknown kernels '$name'")
    }

    fun sha256(f: File): String = MessageDigest.getInstance("SHA-256").digest(f.readBytes()).joinToString("") { "%02x".format(it) }

    /** Tokenizer parity: rebuilt prompt ids vs the Python ids. Returns the number of mismatching prompts. */
    fun tokenizerParity(tok: BpeTokenizer, fx: Fixture, out: Appendable): Int {
        var bad = 0
        for (p in fx.prompts) {
            val ids = when (p.mode) {
                "plain" -> InlinePrompt.plain(tok, p.path, p.prefix, p.ctx)
                "spm" -> InlinePrompt.spm(tok, p.path, p.prefix, p.suffix, p.ctx, p.maxPrefix, p.suffixTokens)
                else -> error("mode ${p.mode}")
            }
            if (!ids.contentEquals(p.ids)) {
                bad++
                var k = 0
                while (k < minOf(ids.size, p.ids.size) && ids[k] == p.ids[k]) k++
                out.append("  TOKENIZER MISMATCH ${p.name}: kotlin ${ids.size} ids, python ${p.ids.size}, first difference at $k\n")
            }
        }
        out.append("tokenizer: ${fx.prompts.size - bad}/${fx.prompts.size} prompts identical to Python (header + prefix/suffix assembly)\n")
        return bad
    }

    class Result(val kernels: String, val vsFloat: LogitStats, val vsFq: LogitStats, val behav: BehavStats)

    /** Logit + greedy parity and the behavioural run of one kernel implementation. */
    fun runKernel(weights: NnFormat.Model, kernels: NnKernels, threads: Int, tok: BpeTokenizer, fx: Fixture, outDir: File?, out: Appendable): Result {
        val vsFloat = LogitStats("vs float32"); val vsFq = LogitStats("vs fq-int8")
        val behav = BehavStats(kernels.name)
        NnModel(weights, threads, kernels).use { model ->
            if (kernels is NativeNnKernels) kernels.prepare(weights.tensors.values.filterIsInstance<NnTensor.Q8>())
            model.newSession(model.config.maxContext).use { s ->
                for (p in fx.prompts) {
                    s.truncate(0)
                    val logits = s.prefill(p.ids)
                    val am = Sampler.argmax(logits)
                    fun cmp(ref: FloatArray, gen: IntArray, lp: FloatArray, st: LogitStats) {
                        var mx = 0f
                        for (i in ref.indices) mx = maxOf(mx, abs(ref[i] - logits[i]))
                        // teacher-forced pass over the reference continuation: argmax agreement and log-prob error per step
                        s.truncate(p.ids.size)
                        var lg = logits
                        var firstDiv = gen.size
                        val err = FloatArray(gen.size)
                        for (i in gen.indices) {
                            if (firstDiv == gen.size && Sampler.argmax(lg) != gen[i]) firstDiv = i
                            err[i] = Sampler.logProb(lg, gen[i]) - lp[i]
                            lg = s.decode(gen[i])
                        }
                        st.add(mx, am == Sampler.argmax(ref), firstDiv, gen.size, err)
                        st.lines += "  %-10s len %4d  max|dlogit| %.4f  argmax %s  first divergence %2d/%d  lp rmse %.5f".format(
                            p.name, p.ids.size, mx, if (am == Sampler.argmax(ref)) "ok " else "DIFF", firstDiv, gen.size, sqrt(err.sumOf { it.toDouble() * it } / maxOf(1, err.size)))
                    }
                    cmp(p.logits, p.gen, p.genLp, vsFloat)
                    cmp(p.logitsFq, p.genFq, p.genFqLp, vsFq)
                }
                val stops = InlinePrompt.stopIds(tok)
                val tsv = outDir?.let { File(it, "kotlin-${kernels.name}-t$threads.tsv").printWriter() }
                tsv?.println("pos\tmode\tstop\tgen_b64\tgen_ids\tms")
                for (r in fx.behav) {
                    s.truncate(0)
                    val t0 = System.nanoTime()
                    val g = s.generate(r.ids, 49, stops)
                    val ms = (System.nanoTime() - t0) / 1e6
                    val genIds = if (g.stopped) g.tokens.copyOf(g.tokens.size - 1) else g.tokens.copyOf(minOf(48, g.tokens.size))
                    val stop = if (!g.stopped) "limit" else if (tok.isSpecial(g.tokens.last())) "special" else "newline"
                    val gen = tok.decodeBytes(genIds)
                    behav.add(r, gen, genIds, stop, ms)
                    tsv?.println("${r.pos}\t${r.mode}\t$stop\t${Base64.getEncoder().encodeToString(gen)}\t${genIds.joinToString(",")}\t%.1f".format(ms))
                }
                tsv?.close()
            }
        }
        out.append("== ${kernels.name}, $threads threads (native lib: ${NativeLib.status})\n")
        for (l in vsFloat.lines) out.append(l).append('\n')
        out.append(vsFloat.toString()).append('\n')
        out.append(vsFq.toString()).append('\n')
        out.append(behav.toString())
        for (d in behav.diffs) out.append(d).append('\n')
        return Result(kernels.name, vsFloat, vsFq, behav)
    }

    @JvmStatic fun main(args: Array<String>) {
        val opt = args.toList().chunked(2).associate { (k, v) -> k.removePrefix("--") to v }
        val threads = (opt["threads"] ?: "8").toInt()
        val names = (opt["kernels"] ?: "scalar,native-f32,native-q8").split(",")
        val behavLimit = opt["behav"]?.toInt() ?: Int.MAX_VALUE
        val dir = opt["dir"]?.let(::File) ?: defaultDir
        val outDir = opt["out"]?.let(::File) ?: dir
        val model = opt["model"]?.let(::File) ?: defaultModel
        val vocab = opt["vocab"]?.let(::File) ?: defaultVocab
        val out = System.out
        val fx = load(dir, behavLimit)
        println("fixture $dir: ${fx.prompts.size} logit prompts, ${fx.behav.size} behavioural records; model $model")
        val weights = NnFormat.read(model)
        val tok = BpeTokenizer.load(vocab.toPath())
        require(tok.vocabSize == weights.config.vocabSize && tok.vocabSize == fx.vocabSize)
        val sha = sha256(vocab)
        println("tokenizer sha256 $sha, model meta tokenizerSha256=${weights.meta["tokenizerSha256"]} -> ${if (sha == weights.meta["tokenizerSha256"]) "match" else "MISMATCH"}")
        tokenizerParity(tok, fx, out)
        for (n in names) runKernel(weights, kernelsByName(n), threads, tok, fx, outDir, out)
    }
}
