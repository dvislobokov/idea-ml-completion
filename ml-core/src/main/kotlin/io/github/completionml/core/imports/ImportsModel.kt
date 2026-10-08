package io.github.completionml.core.imports

import io.github.completionml.core.format.ModelFormat
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.InputStream
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Corpus import statistics (experiment e20): which import path / namespace supplies an identifier, and which imports usually
 * go together. Mined offline by `tools/imports/mine_imports.py`, packed by `pack_imports.py` into a `.cml` of kind `imports`
 * (`models/go-imports-e20.cml`, `models/cs-imports-e20.cml`); the Python reference of the queries is `tools/imports/imports_model.py`
 * and `ImportsParityTest` checks both agree on the quantised scores.
 *
 * Scores are natural logs quantised to 1/16 nat: `names[name]` holds `-ln p(path | name)`, `co[path]` the pointwise mutual
 * information `ln(c(a,b) D / (c(a) c(b)))` with the most frequent partners (a pair missing from both lists counts as 0).
 * Heap-resident (a few MB); immutable and thread-safe after loading.
 */
class ImportsModel private constructor(
    val language: String,
    val params: Map<String, String>,
    val totalDocs: Int,
    val paths: Array<String>,
    val pathDocs: IntArray,
    private val nameIndex: HashMap<String, Int>,
    private val nameDocs: IntArray,
    private val nameStart: IntArray,       // names[i] owns entries nameStart[i] until nameStart[i+1]
    private val namePath: IntArray,
    private val nameQ: ByteArray,          // unsigned: -ln p * 16
    private val coStart: IntArray,         // per path id, -1-based layout like nameStart (coStart.size = paths.size + 1)
    private val coOther: IntArray,
    private val coQ: ByteArray,            // signed: pmi * 16
) {
    /** Weight of the context association (sum of PMI over the present imports) against the prior; packed by the tool. */
    val lambda: Float = params["lambda"]?.toFloatOrNull() ?: 1f

    /** How the PMI sum over the present imports is normalised: `sum` (default), `sqrt` (÷ √n) or `mean` (÷ n). */
    val contextNorm: String = params["ctx_norm"] ?: "sum"

    private val pathIndex = HashMap<String, Int>(paths.size * 2).also { m -> paths.forEachIndexed { i, p -> m[p] = i } }

    class Suggestion(val path: String, val score: Float) {
        override fun toString() = "$path=$score"
    }

    val nameCount: Int get() = nameDocs.size

    fun pathId(path: String): Int = pathIndex[path] ?: -1

    /** Quantised PMI (×16) between two path ids, 0 when the pair is stored in neither list. */
    fun pmiQ(a: Int, b: Int): Int {
        if (a < 0 || b < 0) return 0
        val q = lookupCo(a, b)
        return if (q != NONE) q else lookupCo(b, a).let { if (it != NONE) it else 0 }
    }

    private fun lookupCo(a: Int, b: Int): Int {
        for (k in coStart[a] until coStart[a + 1]) if (coOther[k] == b) return coQ[k].toInt()
        return NONE
    }

    /**
     * Import paths that may supply the unresolved [name] (Go: an exported identifier such as `Client`, or a package
     * qualifier such as `http`; C#: a type name such as `JsonSerializer`), best first, given the imports already present in
     * the file. Score = ln p(path | name) + λ · Σ PMI(path, present import) in nats (comparable across calls, not a probability).
     * Empty when the name was never seen in the corpus. Unknown present imports are ignored.
     */
    fun rankImports(name: String, currentImports: Collection<String>, lambda: Float = this.lambda): List<Suggestion> {
        val id = nameIndex[name] ?: return emptyList()
        val ctx = IntArray(currentImports.size); var n = 0
        for (c in currentImports) { val p = pathIndex[c] ?: continue; ctx[n++] = p }
        val scale = contextScale(n)
        val out = ArrayList<Suggestion>(nameStart[id + 1] - nameStart[id])
        for (k in nameStart[id] until nameStart[id + 1]) {
            val pid = namePath[k]
            var s = -(nameQ[k].toInt() and 0xFF).toFloat()
            if (lambda != 0f && n > 0) {
                var acc = 0
                for (i in 0 until n) if (ctx[i] != pid) acc += pmiQ(pid, ctx[i])
                s += lambda * acc * scale
            }
            out.add(Suggestion(paths[pid], s / SCALE))
        }
        out.sortWith(compareByDescending<Suggestion> { it.score }.thenBy { it.path })
        return out
    }

    /** `ln p(path | name)` alone (the "most frequent path for the name" baseline). */
    fun prior(name: String): List<Suggestion> = rankImports(name, emptyList(), 0f)

    /**
     * Imports that usually accompany [currentImports] ("files with these imports also import X"), best first, at most
     * [limit]; present imports are never returned. Score = ln p(path) + Σ PMI(path, present import), nats.
     */
    fun rankCoImports(currentImports: Collection<String>, limit: Int = 20): List<Suggestion> {
        val ctx = HashSet<Int>()
        for (c in currentImports) pathIndex[c]?.let { ctx.add(it) }
        if (ctx.isEmpty()) return emptyList()
        val acc = HashMap<Int, Int>()
        for (c in ctx) for (k in coStart[c] until coStart[c + 1]) {
            val o = coOther[k]
            if (o !in ctx) acc[o] = (acc[o] ?: 0) + coQ[k]
        }
        val out = ArrayList<Suggestion>(acc.size)
        for ((o, s) in acc) {
            val prior = qLogP(pathDocs[o].toDouble() / totalDocs)
            out.add(Suggestion(paths[o], (s - prior) / SCALE))
        }
        out.sortWith(compareByDescending<Suggestion> { it.score }.thenBy { it.path })
        return if (out.size > limit) out.subList(0, limit) else out
    }

    private fun contextScale(n: Int): Float = when (contextNorm) {
        "sqrt" -> if (n > 0) 1f / sqrt(n.toFloat()) else 0f
        "mean" -> if (n > 0) 1f / n else 0f
        else -> 1f
    }

    companion object {
        const val KIND = "imports"
        const val SCALE = 16f
        private const val NONE = Int.MIN_VALUE

        /** `round(-ln p * 16)` clipped to a byte, as the packer does (`imports_model.q_logp`). */
        fun qLogP(p: Double): Int = (-ln(p) * SCALE).roundToInt().coerceIn(0, 255)

        fun read(file: File): ImportsModel = read(file.inputStream(), file.toString())

        /** Reads the artifact from a gzip stream (a bundled resource, for instance); the stream is closed. */
        fun read(stream: InputStream, name: String = "imports model"): ImportsModel = ModelFormat.read(stream, KIND, { header, inp ->
            val params = LinkedHashMap<String, String>()
            repeat(inp.readInt()) { params[inp.readUTF()] = inp.readUTF() }
            val totalDocs = inp.readInt()
            val nPaths = inp.readInt()
            val paths = Array(nPaths) { "" }
            val pathDocs = IntArray(nPaths)
            for (i in 0 until nPaths) { paths[i] = inp.readUTF(); pathDocs[i] = inp.readInt() }
            val nNames = inp.readInt()
            val nameIndex = HashMap<String, Int>(nNames * 2)
            val nameDocs = IntArray(nNames)
            val nameStart = IntArray(nNames + 1)
            var namePath = IntArray(nNames * 2); var nameQ = ByteArray(nNames * 2); var used = 0
            for (i in 0 until nNames) {
                nameIndex[inp.readUTF()] = i
                nameDocs[i] = inp.readInt()
                val k = inp.readUnsignedByte()
                if (used + k > namePath.size) {
                    val cap = maxOf(namePath.size * 2, used + k)
                    namePath = namePath.copyOf(cap); nameQ = nameQ.copyOf(cap)
                }
                nameStart[i] = used
                repeat(k) { namePath[used] = inp.readInt(); nameQ[used] = inp.readByte(); used++ }
            }
            nameStart[nNames] = used
            namePath = namePath.copyOf(used); nameQ = nameQ.copyOf(used)
            val nCo = inp.readInt()
            val lists = arrayOfNulls<Pair<IntArray, ByteArray>>(nPaths)
            var coTotal = 0
            repeat(nCo) {
                val pid = inp.readInt()
                val k = inp.readUnsignedByte()
                val others = IntArray(k); val qs = ByteArray(k)
                for (j in 0 until k) { others[j] = inp.readInt(); qs[j] = inp.readByte() }
                require(pid in 0 until nPaths) { "$name: co-import list for unknown path $pid" }
                lists[pid] = others to qs
                coTotal += k
            }
            val coStart = IntArray(nPaths + 1)
            val coOther = IntArray(coTotal); val coQ = ByteArray(coTotal)
            var pos = 0
            for (i in 0 until nPaths) {
                coStart[i] = pos
                val l = lists[i] ?: continue
                System.arraycopy(l.first, 0, coOther, pos, l.first.size)
                System.arraycopy(l.second, 0, coQ, pos, l.second.size)
                pos += l.first.size
            }
            coStart[nPaths] = pos
            ImportsModel(header.language, params, totalDocs, paths, pathDocs, nameIndex, nameDocs, nameStart, namePath, nameQ, coStart, coOther, coQ)
        }, name)

        /**
         * Writes an artifact in the packer's layout — for tests and tooling (the production files come from
         * `tools/imports/pack_imports.py`). [names]: name → (doc count, list of (path, q)); [co]: path → list of (other path, pmi q).
         */
        fun write(
            file: File, language: String, params: Map<String, String>, totalDocs: Int,
            paths: List<String>, pathDocs: List<Int>,
            names: Map<String, Pair<Int, List<Pair<String, Int>>>>, co: Map<String, List<Pair<String, Int>>>,
            corpusId: String = "test",
        ) {
            val pid = paths.withIndex().associate { (i, p) -> p to i }
            ModelFormat.write(file, ModelFormat.Header(KIND, language, 0L, System.currentTimeMillis(), corpusId)) { out: DataOutputStream ->
                out.writeInt(params.size)
                for ((k, v) in params) { out.writeUTF(k); out.writeUTF(v) }
                out.writeInt(totalDocs)
                out.writeInt(paths.size)
                for (i in paths.indices) { out.writeUTF(paths[i]); out.writeInt(pathDocs[i]) }
                out.writeInt(names.size)
                for ((n, e) in names.entries.sortedBy { it.key }) {
                    out.writeUTF(n); out.writeInt(e.first); out.writeByte(e.second.size)
                    for ((p, q) in e.second) { out.writeInt(pid.getValue(p)); out.writeByte(q) }
                }
                out.writeInt(co.size)
                for ((p, lst) in co.entries.sortedBy { pid.getValue(it.key) }) {
                    out.writeInt(pid.getValue(p)); out.writeByte(lst.size)
                    for ((o, q) in lst) { out.writeInt(pid.getValue(o)); out.writeByte(q) }
                }
            }
        }
    }
}
