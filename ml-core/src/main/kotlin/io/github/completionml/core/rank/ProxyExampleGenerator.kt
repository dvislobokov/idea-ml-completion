package io.github.completionml.core.rank

import io.github.completionml.core.ngram.CacheLm
import io.github.completionml.core.ngram.MixedLm
import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.ngram.TokenLm
import io.github.completionml.core.spi.ContextKind
import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.vocab.Vocabulary
import kotlin.math.ln

/**
 * Language-agnostic stand-in for the PSI-based example generators of the plugins (`CSharpExampleGenerator`, `GoExampleGenerator`).
 *
 * Completion position = every identifier token; candidates = distinct identifiers that already occurred earlier in the same file
 * ("scope proxy"); the answer is the identifier actually written. Positions whose answer has no earlier occurrence are skipped
 * (a real adapter would get such candidates from imports/types). A typed prefix of 0–2 characters is sampled and candidates are
 * filtered by it, as the IDE would.
 *
 * Feature computation lives here so that the same code can later serve the IDE path (training/serving parity).
 */
class ProxyExampleGenerator(
    private val vocab: Vocabulary,
    private val lm: NgramModel?,
    private val maxCandidates: Int = 100,
    private val maxExamplesPerFile: Int = 60,
    private val seed: Long = 7,
    /** > 0: mix a per-file cache LM into the LM feature with this weight (what the IDE will do). */
    private val cacheLambda: Double = 0.0,
    private val cacheOrder: Int = 3,
    /** number of base features to emit (ablations; default all) */
    private val baseCount: Int = FeatureSchema.BASE.size,
) {
    private val schema = FeatureSchema.common(baseCount)

    fun schema() = schema

    fun generate(tokens: List<MlToken>, sink: (TrainingExample) -> Unit) {
        val rnd = java.util.Random(seed xor tokens.size.toLong())
        val ids = vocab.encode(tokens)
        val cache = if (lm != null && cacheLambda > 0) CacheLm(cacheOrder, vocab.size) else null
        val scorer: TokenLm? = if (cache != null) MixedLm(lm!!, cache, cacheLambda) else lm
        val lastSeen = HashMap<String, Int>()      // identifier -> last token index
        val freq = HashMap<String, Int>()
        val positions = ArrayList<Int>()
        for (i in 1 until tokens.size) if (tokens[i].kind == TokenKind.IDENT) positions.add(i)
        // subsample positions deterministically
        val chosenPositions = if (positions.size <= maxExamplesPerFile) positions.toHashSet()
                              else positions.shuffled(rnd).take(maxExamplesPerFile).toHashSet()
        for (i in tokens.indices) {
            val t = tokens[i]
            if (t.kind == TokenKind.IDENT && i in chosenPositions && lastSeen.containsKey(t.text)) {
                val kind = contextKind(tokens, i)
                val prefixLen = when (rnd.nextInt(10)) { in 0..4 -> 0; in 5..7 -> 1; else -> 2 }.coerceAtMost(t.text.length)
                val prefix = t.text.substring(0, prefixLen)
                var cands = lastSeen.keys.filter { it.length >= prefixLen && it.regionMatches(0, prefix, 0, prefixLen, ignoreCase = true) }
                if (cands.size > 1) {
                    if (cands.size > maxCandidates) {
                        val others = cands.filter { it != t.text }.shuffled(rnd).take(maxCandidates - 1)
                        cands = others + t.text
                    }
                    val names = cands.shuffled(rnd).toTypedArray()
                    val chosen = names.indexOf(t.text)
                    val listSizeLog = ln(1.0 + names.size).toFloat()
                    val features = Array(names.size) { c ->
                        val name = names[c]
                        val id = vocab.id(name)
                        val base = FloatArray(FeatureSchema.BASE.size)
                        base[0] = scorer?.logProb(ids, i, id) ?: 0f
                        base[1] = ln(1.0 + (freq[name] ?: 0)).toFloat()
                        base[2] = lastSeen[name]?.let { ln(1.0 + (i - it)).toFloat() } ?: 0f
                        base[3] = name.length / 10f
                        base[4] = if (prefixLen > 0 && name.startsWith(prefix)) 1f else 0f
                        base[5] = if (name[0].isUpperCase()) 1f else 0f
                        base[6] = if (id != Vocabulary.UNK_ID) 1f else 0f
                        base[7] = listSizeLog
                        base[10] = lm?.logProb(ids, i, id) ?: 0f
                        base
                    }
                    FeatureSchema.fillListFeatures(features)
                    sink(TrainingExample(kind, features, chosen, names))
                }
            }
            if (t.kind == TokenKind.IDENT) { lastSeen[t.text] = i; freq.merge(t.text, 1, Int::plus) }
            cache?.add(ids[i])
        }
    }

    companion object {
        fun contextKind(tokens: List<MlToken>, i: Int): ContextKind {
            val prev = tokens[i - 1]
            if (prev.kind == TokenKind.PUNCT) return when (prev.text) {
                ".", "?.", "->", "::" -> ContextKind.AFTER_DOT
                ";", "{", "}" -> ContextKind.STATEMENT_START
                "(", "," -> ContextKind.ARGUMENT
                "=", ":=", "??=" -> ContextKind.ASSIGN_RHS
                "<", "[", "*", "&" -> ContextKind.TYPE_POSITION   // generic argument / array element / Go pointer: rough
                else -> ContextKind.OTHER
            }
            if (prev.kind == TokenKind.KEYWORD) return when (prev.text) {
                "return", "await", "yield", "case" -> ContextKind.ASSIGN_RHS
                "new", "is", "as", "typeof", "chan", "map", "struct", "func", "var", "const", "type" -> ContextKind.TYPE_POSITION
                "if", "while", "for", "foreach", "switch", "else", "defer", "go", "throw" -> ContextKind.STATEMENT_START
                else -> ContextKind.OTHER
            }
            return ContextKind.OTHER
        }
    }
}
