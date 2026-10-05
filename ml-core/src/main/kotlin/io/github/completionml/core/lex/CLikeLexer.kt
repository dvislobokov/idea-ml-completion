package io.github.completionml.core.lex

import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.MlTokenizer
import io.github.completionml.core.spi.TokenKind

/**
 * Minimal hand-written lexer for C-family syntax (C#, Go). Good enough for n-gram statistics and for the
 * language-agnostic example generator; the IDE adapters replace it with the real lexer of their PSI.
 *
 * Literals are normalised: strings -> `<STR>`, chars/runes -> `<CHAR>`, numbers -> `<NUM>`. Comments and whitespace are dropped.
 * Newlines are not emitted; in Go the lexer inserts `;` where the Go spec would.
 */
open class CLikeLexer(
    private val keywords: Set<String>,
    private val verbatimPrefix: Boolean = false,   // C#: @"..." and $@"..."
    private val backtickStrings: Boolean = false,  // Go: `raw`
    private val goAutoSemicolon: Boolean = false,
    private val preprocessor: Boolean = false,     // C#: drop #-lines
    private val punctuation: List<String> = DEFAULT_PUNCT,
) : MlTokenizer {

    override fun tokens(text: CharSequence): List<MlToken> {
        val out = ArrayList<MlToken>(text.length / 4)
        var i = 0
        val n = text.length
        fun goNeedsSemicolon(): Boolean {
            val t = out.lastOrNull() ?: return false
            return when (t.kind) {
                TokenKind.IDENT, TokenKind.NUMBER, TokenKind.STRING, TokenKind.CHAR -> true
                TokenKind.KEYWORD -> t.text in GO_SEMI_KEYWORDS
                TokenKind.PUNCT -> t.text in GO_SEMI_PUNCT
            }
        }
        while (i < n) {
            val c = text[i]
            if (c == '\n') {
                if (goAutoSemicolon && goNeedsSemicolon()) out.add(MlToken(TokenKind.PUNCT, ";", i))
                i++; continue
            }
            if (c.isWhitespace()) { i++; continue }
            // comments
            if (c == '/' && i + 1 < n) {
                val d = text[i + 1]
                if (d == '/') { while (i < n && text[i] != '\n') i++; continue }
                if (d == '*') { val e = indexOf(text, "*/", i + 2); i = if (e < 0) n else e + 2; continue }
            }
            if (preprocessor && c == '#' && atLineStart(text, i)) { while (i < n && text[i] != '\n') i++; continue }
            // strings: "…", @"…", $"…", $@"…", @$"…", """…"""
            val strStart = stringPrefixEnd(text, i)
            if (strStart >= 0) {
                val start = i
                val verbatim = verbatimPrefix && text.subSequence(i, strStart).contains('@')
                i = skipString(text, strStart, verbatim)
                out.add(MlToken(TokenKind.STRING, "<STR>", start)); continue
            }
            if (backtickStrings && c == '`') {
                val start = i; val e = indexOf(text, "`", i + 1); i = if (e < 0) n else e + 1
                out.add(MlToken(TokenKind.STRING, "<STR>", start)); continue
            }
            if (c == '\'') {
                val start = i; i++
                while (i < n && text[i] != '\'' && text[i] != '\n') { if (text[i] == '\\') i++; i++ }
                if (i < n && text[i] == '\'') i++
                out.add(MlToken(TokenKind.CHAR, "<CHAR>", start)); continue
            }
            // numbers
            if (c.isDigit() || (c == '.' && i + 1 < n && text[i + 1].isDigit())) {
                val start = i; i++
                while (i < n) {
                    val ch = text[i]
                    val expSign = (ch == '+' || ch == '-') && (text[i - 1] == 'e' || text[i - 1] == 'E' || text[i - 1] == 'p' || text[i - 1] == 'P')
                    if (ch.isLetterOrDigit() || ch == '_' || ch == '.' || expSign) i++ else break
                }
                out.add(MlToken(TokenKind.NUMBER, "<NUM>", start)); continue
            }
            // identifiers / keywords (C#: @identifier escapes a keyword)
            if (c.isLetter() || c == '_' || (c == '@' && i + 1 < n && (text[i + 1].isLetter() || text[i + 1] == '_'))) {
                val start = i
                val escaped = c == '@'
                if (escaped) i++
                while (i < n && (text[i].isLetterOrDigit() || text[i] == '_')) i++
                val word = text.subSequence(if (escaped) start + 1 else start, i).toString()
                val kind = if (!escaped && word in keywords) TokenKind.KEYWORD else TokenKind.IDENT
                out.add(MlToken(kind, word, start)); continue
            }
            // punctuation: longest match
            var matched: String? = null
            for (p in punctuation) {
                if (p.length <= n - i && regionMatches(text, i, p)) { matched = p; break }
            }
            if (matched != null) { out.add(MlToken(TokenKind.PUNCT, matched, i)); i += matched.length; continue }
            i++ // unknown char: skip
        }
        if (goAutoSemicolon && goNeedsSemicolon()) out.add(MlToken(TokenKind.PUNCT, ";", n))
        return out
    }

    /** If a string literal starts at [i] (possibly after `@`/`$` prefixes), returns the index of its opening quote, else -1. */
    private fun stringPrefixEnd(text: CharSequence, i: Int): Int {
        var j = i
        if (verbatimPrefix) {
            while (j < text.length && j - i < 2 && (text[j] == '@' || text[j] == '$')) j++
        }
        return if (j < text.length && text[j] == '"') j else -1
    }

    private fun skipString(text: CharSequence, quoteIndex: Int, verbatim: Boolean): Int {
        val n = text.length
        var i = quoteIndex
        // raw string literal """..."""
        if (i + 2 < n && text[i + 1] == '"' && text[i + 2] == '"') {
            var q = 0; while (i < n && text[i] == '"') { q++; i++ }
            var run = 0
            while (i < n) { if (text[i] == '"') { run++; if (run == q) return i + 1 } else run = 0; i++ }
            return n
        }
        i++
        while (i < n) {
            val ch = text[i]
            if (verbatim) {
                if (ch == '"') { if (i + 1 < n && text[i + 1] == '"') { i += 2; continue }; return i + 1 }
            } else {
                if (ch == '\\') { i += 2; continue }
                if (ch == '"') return i + 1
                if (ch == '\n') return i
            }
            i++
        }
        return n
    }

    private fun atLineStart(text: CharSequence, i: Int): Boolean {
        var j = i - 1
        while (j >= 0 && (text[j] == ' ' || text[j] == '\t')) j--
        return j < 0 || text[j] == '\n'
    }

    private fun indexOf(text: CharSequence, s: String, from: Int): Int {
        if (text is String) return text.indexOf(s, from)
        var i = from
        while (i <= text.length - s.length) { if (regionMatches(text, i, s)) return i; i++ }
        return -1
    }

    private fun regionMatches(text: CharSequence, at: Int, s: String): Boolean {
        for (k in s.indices) if (text[at + k] != s[k]) return false
        return true
    }

    companion object {
        val DEFAULT_PUNCT: List<String> = listOf(
            ">>>=", "<<=", ">>=", "...", "?.", "??=", "->", "=>", "::", "++", "--", "&&", "||", "<-", ":=",
            "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "&^", "??",
            "+", "-", "*", "/", "%", "&", "|", "^", "!", "~", "<", ">", "=", "?", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}",
        ).sortedByDescending { it.length }
        private val GO_SEMI_KEYWORDS = setOf("break", "continue", "fallthrough", "return", "true", "false", "nil", "iota")
        private val GO_SEMI_PUNCT = setOf("++", "--", ")", "]", "}")
    }
}
