package io.github.completionml.core.rank

import io.github.completionml.core.format.ModelFormat
import java.io.File
import java.io.InputStream

/**
 * A scoring model over expanded feature vectors ([FeatureSchema.expand]): higher score = better candidate. Implemented by
 * [LinearRanker] (`.cml` kind `ranker`) and [TreeRanker] (kind `tree-ranker`); plugins load either through [Rankers.read]
 * and only touch this interface.
 */
interface Ranker {
    val schema: FeatureSchema

    /** Score of one candidate; [features] has `schema.size` entries. */
    fun score(features: FloatArray): Float

    /** Scores of all candidates of a list (expands each candidate into a scratch buffer). */
    fun scores(example: TrainingExample): FloatArray {
        val buf = FloatArray(schema.size)
        return FloatArray(example.size) { example.expand(it, buf); score(buf) }
    }

    /** Short description for logs / settings UI ("210 weights", "300 trees, 8 937 leaves"). */
    val description: String
}

/** Loads a ranker `.cml` of either kind. */
object Rankers {
    const val KIND_LINEAR = "ranker"
    const val KIND_TREE = "tree-ranker"

    fun read(file: File): Ranker = read(file.inputStream(), file.toString())

    /** [stream] is a gzip `.cml` (a bundled plugin resource, for instance); it is closed. */
    fun read(stream: InputStream, name: String = "ranker model"): Ranker = ModelFormat.read(stream, setOf(KIND_LINEAR, KIND_TREE), { header, inp ->
        when (header.kind) {
            KIND_LINEAR -> LinearRanker.readBody(header, inp, name)
            KIND_TREE -> TreeRanker.readBody(header, inp, name)
            else -> error("$name: unknown ranker kind ${header.kind}")
        }
    }, name)
}
