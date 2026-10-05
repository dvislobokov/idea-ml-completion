package io.github.completionml.core.spi

/** Lexical class of a token as seen by the models. Literal values are normalised by the tokenizer ([MlToken.text] is `<STR>` etc.). */
enum class TokenKind { KEYWORD, IDENT, PUNCT, STRING, NUMBER, CHAR }

class MlToken(val kind: TokenKind, val text: String, val offset: Int) {
    override fun toString() = "$kind($text)@$offset"
}

/** Lexer-only tokenizer: no PSI, no semantics. Comments and whitespace are dropped. */
interface MlTokenizer {
    fun tokens(text: CharSequence): List<MlToken>
}

/** Kinds of completion positions. Adapters map their own PSI positions onto these; the ranker gets them as one-hot features. */
enum class ContextKind { AFTER_DOT, STATEMENT_START, ARGUMENT, TYPE_POSITION, ASSIGN_RHS, OTHER }

/** Everything the engine needs to know about a language. One implementation per plugin adapter. */
interface MlLanguage {
    /** "csharp" | "go": model folder name and storage key. */
    val id: String
    val tokenizer: MlTokenizer
    val keywords: Set<String>
    /** File extensions (with dot) that belong to this language. */
    val extensions: Set<String>
}
