package io.github.completionml.core.rank

import io.github.completionml.core.format.ModelFormat
import java.io.BufferedReader
import java.io.DataInputStream
import java.io.File
import kotlin.math.abs

/**
 * Gradient-boosted decision trees (LightGBM `lambdarank`, exported by `tools/gbdt/train_gbdt.py` → `ml-train import-gbdt`).
 * Score = Σ leaf values over all trees on the expanded feature vector; the decision rule reproduces LightGBM's
 * `NumericalDecision` including its three missing-value modes, so the Kotlin scores match the Python `predict` exactly.
 *
 * `.cml` body (kind `tree-ranker`):
 * ```
 * i32 n, UTF×n schema names | i32 trees | per tree: i32 nodes, per node: i32 feature (−1 = leaf) f64 threshold u8 flags i32 left i32 right f64 value
 * ```
 * flags: bit 0 = default left, bits 1–2 = missing type (0 none, 1 zero, 2 NaN). Node 0 is the root; leaves carry the value.
 */
class TreeRanker(override val schema: FeatureSchema, val trees: Array<Tree>) : Ranker {

    class Tree(val feature: IntArray, val threshold: DoubleArray, val flags: ByteArray, val left: IntArray, val right: IntArray, val value: DoubleArray) {
        val size get() = feature.size
        val leaves get() = feature.count { it < 0 }
        init {
            require(threshold.size == size && flags.size == size && left.size == size && right.size == size && value.size == size)
            for (i in 0 until size) if (feature[i] >= 0) require(left[i] in 0 until size && right[i] in 0 until size) { "tree node $i: bad children" }
        }

        fun predict(x: FloatArray): Double {
            var n = 0
            while (true) {
                val f = feature[n]
                if (f < 0) return value[n]
                var v = x[f].toDouble()
                val fl = flags[n].toInt()
                val missing = (fl shr 1) and 3
                val isNan = v.isNaN()
                if (isNan && missing != MISSING_NAN) v = 0.0
                n = if ((missing == MISSING_ZERO && abs(v) <= ZERO_THRESHOLD) || (missing == MISSING_NAN && isNan)) {
                    if (fl and 1 != 0) left[n] else right[n]
                } else if (v <= threshold[n]) left[n] else right[n]
            }
        }
    }

    val leaves: Int get() = trees.sumOf { it.leaves }
    override val description: String get() = "${trees.size} trees, $leaves leaves"

    override fun score(features: FloatArray): Float {
        var s = 0.0
        for (t in trees) s += t.predict(features)
        return s.toFloat()
    }

    fun write(file: File, language: String, corpusId: String) {
        ModelFormat.write(file, ModelFormat.Header(Rankers.KIND_TREE, language, schema.hash, System.currentTimeMillis(), corpusId)) { out ->
            out.writeInt(schema.size)
            for (n in schema.names) out.writeUTF(n)
            out.writeInt(trees.size)
            for (t in trees) {
                out.writeInt(t.size)
                for (i in 0 until t.size) {
                    out.writeInt(t.feature[i]); out.writeDouble(t.threshold[i]); out.writeByte(t.flags[i].toInt())
                    out.writeInt(t.left[i]); out.writeInt(t.right[i]); out.writeDouble(t.value[i])
                }
            }
        }
    }

    companion object {
        const val MISSING_NONE = 0
        const val MISSING_ZERO = 1
        const val MISSING_NAN = 2
        /** LightGBM `kZeroThreshold`. */
        const val ZERO_THRESHOLD = 1e-35

        fun read(file: File): TreeRanker = read(file.inputStream(), file.toString())

        fun read(stream: java.io.InputStream, name: String = "tree ranker model"): TreeRanker =
            ModelFormat.read(stream, Rankers.KIND_TREE, { header, inp -> readBody(header, inp, name) }, name)

        internal fun readBody(header: ModelFormat.Header, inp: DataInputStream, name: String): TreeRanker {
            val n = inp.readInt()
            val schema = FeatureSchema(List(n) { inp.readUTF() })
            require(schema.hash == header.schemaHash) { "$name: schema hash mismatch" }
            val trees = Array(inp.readInt()) {
                val size = inp.readInt()
                val feature = IntArray(size); val threshold = DoubleArray(size); val flags = ByteArray(size)
                val left = IntArray(size); val right = IntArray(size); val value = DoubleArray(size)
                for (i in 0 until size) {
                    feature[i] = inp.readInt(); threshold[i] = inp.readDouble(); flags[i] = inp.readByte()
                    left[i] = inp.readInt(); right[i] = inp.readInt(); value[i] = inp.readDouble()
                }
                Tree(feature, threshold, flags, left, right, value)
            }
            return TreeRanker(schema, trees)
        }

        /**
         * Text export of `tools/gbdt/train_gbdt.py` (`--export-trees`):
         * ```
         * tree-ranker 1
         * features <n>            then n lines with feature names (must equal the schema the trees were trained on)
         * trees <T>
         * tree <nodes>            then one line per node: feature threshold default_left missing_type left right value
         * ```
         */
        fun parseText(reader: BufferedReader): TreeRanker {
            fun next(): List<String> { while (true) { val l = reader.readLine() ?: error("unexpected end of tree text"); if (l.isNotBlank()) return l.trim().split(' ') } }
            val head = next(); require(head[0] == "tree-ranker" && head[1] == "1") { "not a tree-ranker text export: $head" }
            val nf = next().also { require(it[0] == "features") }[1].toInt()
            val names = List(nf) { reader.readLine()!!.trim() }
            val nt = next().also { require(it[0] == "trees") }[1].toInt()
            val trees = Array(nt) {
                val size = next().also { require(it[0] == "tree") }[1].toInt()
                val feature = IntArray(size); val threshold = DoubleArray(size); val flags = ByteArray(size)
                val left = IntArray(size); val right = IntArray(size); val value = DoubleArray(size)
                for (i in 0 until size) {
                    val p = next()
                    feature[i] = p[0].toInt(); threshold[i] = p[1].toDouble()
                    flags[i] = ((p[2].toInt() and 1) or (p[3].toInt() shl 1)).toByte()
                    left[i] = p[4].toInt(); right[i] = p[5].toInt(); value[i] = p[6].toDouble()
                    require(feature[i] < nf) { "tree $it node $i: feature ${feature[i]} out of range" }
                }
                Tree(feature, threshold, flags, left, right, value)
            }
            return TreeRanker(FeatureSchema(names), trees)
        }
    }
}
