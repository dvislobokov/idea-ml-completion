package io.github.completionml.core.vocab

import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import java.io.DataInput
import java.io.DataOutput

/**
 * Token -> id mapping shared by the n-gram model and the ranker features.
 * Ids 0..2 are reserved; keywords, punctuation and literal placeholders are always kept; identifiers are kept by
 * document frequency (number of files they appear in), everything else maps to [UNK_ID] (`<ID>`).
 */
class Vocabulary private constructor(private val words: Array<String>, private val isIdent: BooleanArray) {
    private val index = HashMap<String, Int>(words.size * 2).apply { words.forEachIndexed { i, w -> put(w, i) } }

    val size get() = words.size

    fun id(token: MlToken): Int = if (token.kind == TokenKind.IDENT) (index[token.text] ?: UNK_ID) else (index[token.text] ?: UNK_ID)
    fun id(text: String): Int = index[text] ?: UNK_ID
    fun word(id: Int): String = words[id]
    fun isIdentifier(id: Int): Boolean = isIdent[id]

    fun encode(tokens: List<MlToken>): IntArray {
        val ids = IntArray(tokens.size)
        for (i in tokens.indices) ids[i] = id(tokens[i])
        return ids
    }

    fun write(out: DataOutput) {
        out.writeInt(words.size)
        for (i in words.indices) { out.writeUTF(words[i]); out.writeBoolean(isIdent[i]) }
    }

    /** Collects statistics file by file; call [build] at the end. */
    class Builder {
        private val docFreq = HashMap<String, Int>()
        private val termFreq = HashMap<String, Long>()
        private val identSet = HashSet<String>()
        var files = 0; private set
        var tokens = 0L; private set

        fun addFile(tokens: List<MlToken>) {
            files++
            this.tokens += tokens.size
            val seen = HashSet<String>()
            for (t in tokens) {
                termFreq.merge(t.text, 1L, Long::plus)
                if (t.kind == TokenKind.IDENT) identSet.add(t.text)
                if (seen.add(t.text)) docFreq.merge(t.text, 1, Int::plus)
            }
        }

        /** Adds pre-aggregated statistics of one token text (document frequency = number of files containing it); see [addTotals]. */
        fun addCounts(text: String, ident: Boolean, docFreq: Int, termFreq: Long) {
            this.docFreq.merge(text, docFreq, Int::plus); this.termFreq.merge(text, termFreq, Long::plus)
            if (ident) identSet.add(text)
        }

        fun addTotals(files: Int, tokens: Long) { this.files += files; this.tokens += tokens }

        /** @param maxIdentifiers how many identifiers to keep (by document frequency, ties by term frequency); others become `<ID>`. */
        fun build(maxIdentifiers: Int, minDocFreq: Int = 2): Vocabulary {
            val fixed = ArrayList<String>()
            val idents = ArrayList<String>()
            for (w in termFreq.keys) if (w in identSet) idents.add(w) else fixed.add(w)
            fixed.sort()
            idents.sortWith(compareByDescending<String> { docFreq[it] ?: 0 }.thenByDescending { termFreq[it] ?: 0 }.thenBy { it })
            val kept = idents.asSequence().filter { (docFreq[it] ?: 0) >= minDocFreq }.take(maxIdentifiers).toList()
            val words = ArrayList<String>(RESERVED.size + fixed.size + kept.size)
            words.addAll(RESERVED); words.addAll(fixed); words.addAll(kept)
            val isIdent = BooleanArray(words.size) { i -> i == UNK_ID || i >= RESERVED.size + fixed.size }
            return Vocabulary(words.toTypedArray(), isIdent)
        }
    }

    companion object {
        const val UNK_ID = 0   // <ID>: identifier outside the vocabulary
        const val BOS_ID = 1
        const val EOS_ID = 2
        private val RESERVED = listOf("<ID>", "<BOS>", "<EOS>")

        fun read(inp: DataInput): Vocabulary {
            val n = inp.readInt()
            val words = Array(n) { "" }
            val isIdent = BooleanArray(n)
            for (i in 0 until n) { words[i] = inp.readUTF(); isIdent[i] = inp.readBoolean() }
            return Vocabulary(words, isIdent)
        }
    }
}
