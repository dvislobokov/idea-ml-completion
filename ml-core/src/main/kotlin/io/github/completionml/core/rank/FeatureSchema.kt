package io.github.completionml.core.rank

import io.github.completionml.core.spi.ContextKind
import kotlin.math.ln

/**
 * Ordered list of feature names of a ranking model. The hash goes into the model header: a model trained on one schema
 * refuses to score vectors from another (the training/serving parity guard).
 *
 * Layout: `base ++ base·onehot(kind)` where `base = BASE[0 until baseCount] ++ languageFeatures`. The language block holds
 * the PSI-derived per-candidate features an adapter declares in `MlLanguage.rankFeatures`; its values are supplied by the
 * adapter, the common block is computed by [FeatureExtractor] from tokens and the LM.
 */
class FeatureSchema(val names: List<String>) {
    val size get() = names.size
    val hash: Long = names.fold(1469598103934665603L) { h, n -> (h xor n.hashCode().toLong()) * 1099511628211L }
    fun index(name: String): Int = names.indexOf(name).also { require(it >= 0) { "no feature '$name'" } }

    /** Number of per-candidate values an adapter must supply (common block + language block). */
    val baseSize: Int get() = names.size / (1 + KINDS)
    val baseNames: List<String> get() = names.take(baseSize)
    /** Index of a base feature, or -1. */
    fun baseIndex(name: String): Int = baseNames.indexOf(name)
    /** Names of the language block (base features that are not in [BASE]). */
    val languageFeatures: List<String> get() = baseNames.filter { it !in BASE }

    init { require(names.size % (1 + KINDS) == 0) { "schema size ${names.size} is not a multiple of ${1 + KINDS}" } }

    companion object {
        val KINDS = ContextKind.values().size

        /**
         * Features that every adapter can compute from text + token statistics alone, plus their conjunction with the
         * context kind (so the model learns kind-specific weights: recency matters after `.` differently than at statement start).
         */
        val BASE: List<String> = listOf(
            "lm_logprob",        // n-gram log P(candidate | previous tokens), with the per-file cache mixed in when enabled
            "file_freq_log",     // ln(1 + occurrences earlier in the file)
            "recency_log",       // ln(1 + tokens since the last occurrence in the file); 0 if never seen
            "name_len",          // length / 10
            "prefix_case_match", // typed prefix matches with exact case (0 if no prefix)
            "upper_initial",     // candidate starts with an upper-case letter
            "in_vocab",          // candidate is in the global vocabulary (not <ID>)
            "list_size_log",     // ln(number of candidates)
            // list-relative features (computed after all candidates of a list are scored)
            "lm_rank_log",       // ln(1 + rank of the candidate by lm_logprob within the list, best = 0)
            "lm_delta_best",     // lm_logprob of the best candidate minus this one's (0 for the best)
            "lm_global_logprob", // global n-gram log-prob without the cache (so the model can weigh cache vs global)
            "freq_rank_log",     // ln(1 + rank by in-file frequency within the list)
            "is_most_recent",    // candidate is the most recently used identifier in the list
        )

        /** Fills the list-relative features (indices 8..12) from the per-candidate values already in [base]. */
        fun fillListFeatures(base: Array<FloatArray>) {
            val n = base.size
            if (n == 0 || base[0].size < BASE.size) return
            val byLm = (0 until n).sortedByDescending { base[it][0] }
            val byFreq = (0 until n).sortedByDescending { base[it][1] }
            val best = base[byLm[0]][0]
            for ((rank, c) in byLm.withIndex()) { base[c][8] = ln(1.0 + rank).toFloat(); base[c][9] = best - base[c][0] }
            for ((rank, c) in byFreq.withIndex()) base[c][11] = ln(1.0 + rank).toFloat()
            var recent = -1; var recentDist = Float.MAX_VALUE
            for (c in 0 until n) { val d = base[c][2]; if (d > 0f && d < recentDist) { recentDist = d; recent = c } }
            for (c in 0 until n) base[c][12] = if (c == recent) 1f else 0f
        }

        /**
         * Schema over the first [baseCount] common features (ablations) plus the adapter's [languageFeatures], each block
         * conjoined with the context kind.
         */
        fun common(baseCount: Int = BASE.size, languageFeatures: List<String> = emptyList()): FeatureSchema {
            require(baseCount in 1..BASE.size)
            require(languageFeatures.none { it in BASE } && languageFeatures.toSet().size == languageFeatures.size) { "language features must be unique and distinct from BASE" }
            val base = BASE.take(baseCount) + languageFeatures
            val names = ArrayList(base)
            for (k in ContextKind.values()) for (b in base) names.add("${k.name.lowercase()}:$b")
            return FeatureSchema(names)
        }

        /** Expands base features into the full vector (base ++ base·onehot(kind)); `base.size` must equal `out.size / (1+kinds)`. */
        fun expand(base: FloatArray, kind: ContextKind, out: FloatArray) {
            val b = out.size / (1 + KINDS)
            require(b * (1 + KINDS) == out.size && b == base.size) { "base ${base.size} does not match schema base $b" }
            out.fill(0f)
            System.arraycopy(base, 0, out, 0, b)
            System.arraycopy(base, 0, out, b * (1 + kind.ordinal), b)
        }
    }
}
