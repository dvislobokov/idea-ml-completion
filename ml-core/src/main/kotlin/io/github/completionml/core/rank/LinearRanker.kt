package io.github.completionml.core.rank

import io.github.completionml.core.format.ModelFormat
import java.io.File
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Listwise logistic regression (softmax over the candidates of one completion list). Inference is a dot product per
 * candidate on standardised features; weights are plain floats so IDE plugins need nothing but this class.
 */
class LinearRanker(
    override val schema: FeatureSchema,
    val weights: FloatArray,
    val mean: FloatArray,
    val std: FloatArray,
) : Ranker {
    init { require(weights.size == schema.size && mean.size == schema.size && std.size == schema.size) }

    override val description: String get() = "${weights.size} weights"

    override fun score(features: FloatArray): Float {
        var s = 0f
        for (i in weights.indices) s += weights[i] * ((features[i] - mean[i]) / std[i])
        return s
    }

    fun write(file: File, language: String, corpusId: String) {
        ModelFormat.write(file, ModelFormat.Header(Rankers.KIND_LINEAR, language, schema.hash, System.currentTimeMillis(), corpusId)) { out ->
            out.writeInt(schema.size)
            for (n in schema.names) out.writeUTF(n)
            for (w in weights) out.writeFloat(w)
            for (m in mean) out.writeFloat(m)
            for (s in std) out.writeFloat(s)
        }
    }

    companion object {
        fun read(file: File): LinearRanker = read(file.inputStream(), file.toString())

        fun read(stream: java.io.InputStream, name: String = "ranker model"): LinearRanker =
            ModelFormat.read(stream, Rankers.KIND_LINEAR, { header, inp -> readBody(header, inp, name) }, name)

        internal fun readBody(header: ModelFormat.Header, inp: java.io.DataInputStream, name: String): LinearRanker {
            val n = inp.readInt()
            val schema = FeatureSchema(List(n) { inp.readUTF() })
            require(schema.hash == header.schemaHash) { "$name: schema hash mismatch" }
            return LinearRanker(schema, FloatArray(n) { inp.readFloat() }, FloatArray(n) { inp.readFloat() }, FloatArray(n) { inp.readFloat() })
        }
    }
}

/**
 * Trainer: minimises listwise cross-entropy  −log softmax(w·x)[chosen]  with L2 regularisation towards [anchor]
 * (zero for pretraining; the global weights when fine-tuning on a user's project). Adagrad step sizes, in-memory examples.
 */
class LinearRankerTrainer(
    private val schema: FeatureSchema,
    private val l2: Double = 1e-4,
    private val learningRate: Double = 0.1,
    private val epochs: Int = 10,
    private val anchor: FloatArray? = null,
    private val seed: Long = 42,
) {
    class Report(val epochLoss: List<Double>)

    fun train(examples: List<TrainingExample>, log: (String) -> Unit = {}): Pair<LinearRanker, Report> {
        require(examples.isNotEmpty())
        val dim = schema.size
        val (mean, std) = standardisation(examples, dim)
        val w = DoubleArray(dim)
        anchor?.let { for (i in 0 until dim) w[i] = it[i].toDouble() }
        val g2 = DoubleArray(dim) { 1e-8 }
        val grad = DoubleArray(dim)
        val rnd = java.util.Random(seed)
        val order = examples.indices.toMutableList()
        val losses = ArrayList<Double>()
        val x = FloatArray(dim)
        for (epoch in 1..epochs) {
            order.shuffle(rnd)
            var loss = 0.0
            for (idx in order) {
                val ex = examples[idx]
                val s = DoubleArray(ex.size)
                var maxS = Double.NEGATIVE_INFINITY
                for (c in 0 until ex.size) {
                    ex.expand(c, x); val f = x
                    var dot = 0.0
                    for (i in 0 until dim) dot += w[i] * ((f[i] - mean[i]) / std[i])
                    s[c] = dot; if (dot > maxS) maxS = dot
                }
                var z = 0.0
                for (c in 0 until ex.size) { s[c] = exp(s[c] - maxS); z += s[c] }
                loss += -ln(s[ex.chosen] / z)
                grad.fill(0.0)
                for (c in 0 until ex.size) {
                    val p = s[c] / z - (if (c == ex.chosen) 1.0 else 0.0)
                    ex.expand(c, x); val f = x
                    for (i in 0 until dim) grad[i] += p * ((f[i] - mean[i]) / std[i])
                }
                for (i in 0 until dim) {
                    val a = anchor?.get(i)?.toDouble() ?: 0.0
                    val gi = grad[i] + l2 * (w[i] - a)
                    g2[i] += gi * gi
                    w[i] -= learningRate * gi / sqrt(g2[i])
                }
            }
            losses.add(loss / examples.size)
            log("epoch $epoch: loss %.4f".format(loss / examples.size))
        }
        return LinearRanker(schema, FloatArray(dim) { w[it].toFloat() }, mean, std) to Report(losses)
    }

    private fun standardisation(examples: List<TrainingExample>, dim: Int): Pair<FloatArray, FloatArray> {
        val sum = DoubleArray(dim); val sq = DoubleArray(dim); var n = 0L
        val f = FloatArray(dim)
        for (ex in examples) for (c in 0 until ex.size) { ex.expand(c, f); for (i in 0 until dim) { sum[i] += f[i]; sq[i] += f[i].toDouble() * f[i] }; n++ }
        val mean = FloatArray(dim) { (sum[it] / n).toFloat() }
        val std = FloatArray(dim) { val v = sq[it] / n - (sum[it] / n) * (sum[it] / n); sqrt(v.coerceAtLeast(1e-6)).toFloat() }
        return mean to std
    }
}
