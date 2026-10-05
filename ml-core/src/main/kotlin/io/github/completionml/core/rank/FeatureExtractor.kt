package io.github.completionml.core.rank

import io.github.completionml.core.ngram.CacheLm
import io.github.completionml.core.ngram.MixedLm
import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.ngram.TokenLm
import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.vocab.Vocabulary
import kotlin.math.ln

/**
 * Incremental per-file state behind the common features: token ids for the LM context, identifier frequencies and recency,
 * and the per-file cache LM. Feed every token that precedes the caret, in order; the state at the caret is what the
 * features are computed from. The IDE path tokenises the text before the caret and feeds it; the offline generators feed
 * the file token by token and take snapshots at the sampled positions.
 */
class FileState(private val vocab: Vocabulary, withCache: Boolean = false, cacheOrder: Int = 3) {
    var ids = IntArray(256); private set
    var size = 0; private set
    val cache: CacheLm? = if (withCache) CacheLm(cacheOrder, vocab.size) else null
    val freq = HashMap<String, Int>()
    val lastSeen = HashMap<String, Int>()      // identifier -> index of its last occurrence

    fun add(t: MlToken) {
        if (size == ids.size) ids = ids.copyOf(size * 2)
        val id = vocab.id(t.text)
        ids[size++] = id
        if (t.kind == TokenKind.IDENT) { lastSeen[t.text] = size - 1; freq.merge(t.text, 1, Int::plus) }
        cache?.add(id)
    }

    fun addAll(tokens: List<MlToken>) { for (t in tokens) add(t) }
}

/**
 * Computes the common block of per-candidate features ([FeatureSchema.BASE]) and appends the adapter-supplied language
 * block. Shared by the offline example generators and the IDE weigher — the parity point of the ranker.
 *
 * @param cacheLambda > 0 mixes the file cache LM into `lm_logprob` with this weight (`lm_global_logprob` stays global).
 */
class FeatureExtractor(val schema: FeatureSchema, private val vocab: Vocabulary, private val lm: NgramModel?, private val cacheLambda: Double = 0.0) {
    private val commonCount = schema.baseSize - schema.languageFeatures.size
    private val languageCount = schema.languageFeatures.size

    /**
     * @param state file state at the caret (tokens before the caret already fed)
     * @param prefix text typed so far at the caret (candidates are assumed to match it already)
     * @param names candidate insert texts
     * @param language per-candidate language-block values (`schema.languageFeatures.size` each); null when the schema has none
     * @return per-candidate base vectors of length `schema.baseSize`, list-relative features filled
     */
    fun features(state: FileState, prefix: String, names: Array<String>, language: Array<FloatArray>? = null): Array<FloatArray> {
        require((language == null && languageCount == 0) || (language != null && language.size == names.size)) { "language block mismatch" }
        val ids = state.ids; val pos = state.size
        val cache = state.cache
        val mixed: TokenLm? = if (lm != null && cache != null && cacheLambda > 0) MixedLm(lm, cache, cacheLambda) else lm
        val global = lm?.scorer(ids, pos)
        val listSizeLog = ln(1.0 + names.size).toFloat()
        val full = Array(names.size) { c ->
            val name = names[c]
            val id = vocab.id(name)
            val b = FloatArray(FeatureSchema.BASE.size)
            b[0] = mixed?.logProb(ids, pos, id) ?: 0f
            b[1] = ln(1.0 + (state.freq[name] ?: 0)).toFloat()
            b[2] = state.lastSeen[name]?.let { ln(1.0 + (pos - it)).toFloat() } ?: 0f
            b[3] = name.length / 10f
            b[4] = if (prefix.isNotEmpty() && name.startsWith(prefix)) 1f else 0f
            b[5] = if (name.isNotEmpty() && name[0].isUpperCase()) 1f else 0f
            b[6] = if (id != Vocabulary.UNK_ID) 1f else 0f
            b[7] = listSizeLog
            b[10] = global?.logProb(id) ?: 0f
            b
        }
        FeatureSchema.fillListFeatures(full)
        if (commonCount == FeatureSchema.BASE.size && languageCount == 0) return full
        return Array(names.size) { c ->
            val out = FloatArray(schema.baseSize)
            System.arraycopy(full[c], 0, out, 0, commonCount)
            if (language != null) {
                require(language[c].size == languageCount) { "candidate $c: ${language[c].size} language features, schema has $languageCount" }
                System.arraycopy(language[c], 0, out, commonCount, languageCount)
            }
            out
        }
    }
}
