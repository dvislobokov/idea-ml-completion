package io.github.completionml.core.rank

import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.spi.ContextKind
import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.vocab.Vocabulary

/**
 * Language-agnostic stand-in for the PSI-based example generators of the plugins (`CSharpExampleGenerator`, `GoExampleGenerator`).
 *
 * Completion position = every identifier token; candidates = distinct identifiers that already occurred earlier in the same file
 * ("scope proxy"); the answer is the identifier actually written. Positions whose answer has no earlier occurrence are skipped
 * (a real adapter would get such candidates from imports/types). A typed prefix of 0–2 characters is sampled and candidates are
 * filtered by it, as the IDE would.
 *
 * Features come from [FeatureExtractor], the same code the IDE path uses (training/serving parity).
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
    /** number of common features to emit (ablations; default all) */
    private val baseCount: Int = FeatureSchema.BASE.size,
) {
    private val schema = FeatureSchema.common(baseCount)
    private val extractor = FeatureExtractor(schema, vocab, lm, cacheLambda)

    fun schema() = schema

    fun generate(tokens: List<MlToken>, sink: (TrainingExample) -> Unit) {
        val rnd = java.util.Random(seed xor tokens.size.toLong())
        val state = FileState(vocab, withCache = lm != null && cacheLambda > 0, cacheOrder = cacheOrder)
        val positions = ArrayList<Int>()
        for (i in 1 until tokens.size) if (tokens[i].kind == TokenKind.IDENT) positions.add(i)
        // subsample positions deterministically
        val chosenPositions = if (positions.size <= maxExamplesPerFile) positions.toHashSet()
                              else positions.shuffled(rnd).take(maxExamplesPerFile).toHashSet()
        for (i in tokens.indices) {
            val t = tokens[i]
            if (t.kind == TokenKind.IDENT && i in chosenPositions && state.lastSeen.containsKey(t.text)) {
                val kind = contextKind(tokens, i)
                val prefixLen = when (rnd.nextInt(10)) { in 0..4 -> 0; in 5..7 -> 1; else -> 2 }.coerceAtMost(t.text.length)
                val prefix = t.text.substring(0, prefixLen)
                var cands = state.lastSeen.keys.filter { it.length >= prefixLen && it.regionMatches(0, prefix, 0, prefixLen, ignoreCase = true) }
                if (cands.size > 1) {
                    if (cands.size > maxCandidates) {
                        val others = cands.filter { it != t.text }.shuffled(rnd).take(maxCandidates - 1)
                        cands = others + t.text
                    }
                    val names = cands.shuffled(rnd).toTypedArray()
                    val chosen = names.indexOf(t.text)
                    sink(TrainingExample(kind, extractor.features(state, prefix, names), chosen, names))
                }
            }
            state.add(t)
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
