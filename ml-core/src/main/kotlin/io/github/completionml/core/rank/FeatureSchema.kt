package io.github.completionml.core.rank

import io.github.completionml.core.spi.ContextKind

/**
 * Ordered list of feature names of a ranking model. The hash goes into the model header: a model trained on one schema
 * refuses to score vectors from another (the training/serving parity guard).
 */
class FeatureSchema(val names: List<String>) {
    val size get() = names.size
    val hash: Long = names.fold(1469598103934665603L) { h, n -> (h xor n.hashCode().toLong()) * 1099511628211L }
    fun index(name: String): Int = names.indexOf(name).also { require(it >= 0) { "no feature '$name'" } }

    companion object {
        /**
         * Features that every adapter can compute from text + token statistics alone, plus their conjunction with the
         * context kind (so the model learns kind-specific weights: recency matters after `.` differently than at statement start).
         */
        val BASE: List<String> = listOf(
            "lm_logprob",        // n-gram log P(candidate | previous tokens)
            "file_freq_log",     // ln(1 + occurrences earlier in the file)
            "recency_log",       // ln(1 + tokens since the last occurrence in the file); 0 if never seen
            "name_len",          // length / 10
            "prefix_case_match", // typed prefix matches with exact case (0 if no prefix)
            "upper_initial",     // candidate starts with an upper-case letter
            "in_vocab",          // candidate is in the global vocabulary (not <ID>)
            "list_size_log",     // ln(number of candidates)
        )

        fun common(): FeatureSchema {
            val names = ArrayList(BASE)
            for (k in ContextKind.values()) for (b in BASE) names.add("${k.name.lowercase()}:$b")
            return FeatureSchema(names)
        }

        /** Expands base features into the full vector (base ++ base·onehot(kind)). */
        fun expand(base: FloatArray, kind: ContextKind, out: FloatArray) {
            val b = BASE.size
            require(out.size == b * (1 + ContextKind.values().size))
            out.fill(0f)
            System.arraycopy(base, 0, out, 0, b)
            System.arraycopy(base, 0, out, b * (1 + kind.ordinal), b)
        }
    }
}
