package io.github.completionml.core.lex

import io.github.completionml.core.spi.MlLanguage
import io.github.completionml.core.spi.MlTokenizer

object CSharpLanguage : MlLanguage {
    override val id = "csharp"
    override val extensions = setOf(".cs")
    override val keywords: Set<String> = setOf(
        "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked", "class", "const", "continue", "decimal",
        "default", "delegate", "do", "double", "else", "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
        "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock", "long", "namespace", "new", "null",
        "object", "operator", "out", "override", "params", "private", "protected", "public", "readonly", "ref", "return", "sbyte",
        "sealed", "short", "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true", "try", "typeof",
        "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "virtual", "void", "volatile", "while",
        // contextual keywords that behave like keywords for statistics
        "var", "async", "await", "yield", "get", "set", "init", "record", "where", "select", "from", "nameof", "when", "with", "required",
    )
    override val tokenizer: MlTokenizer = CLikeLexer(keywords, verbatimPrefix = true, preprocessor = true)
}

object GoLanguage : MlLanguage {
    override val id = "go"
    override val extensions = setOf(".go")
    override val keywords: Set<String> = setOf(
        "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func", "go", "goto", "if",
        "import", "interface", "map", "package", "range", "return", "select", "struct", "switch", "type", "var",
        // predeclared identifiers: treated as keywords for statistics
        "nil", "true", "false", "iota", "error", "string", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32",
        "uint64", "uintptr", "byte", "rune", "float32", "float64", "complex64", "complex128", "bool", "any", "append", "cap", "close",
        "copy", "delete", "len", "make", "new", "panic", "print", "println", "recover", "min", "max", "clear",
    )
    override val tokenizer: MlTokenizer = CLikeLexer(keywords, backtickStrings = true, goAutoSemicolon = true)
}

object Languages {
    val all: List<MlLanguage> = listOf(CSharpLanguage, GoLanguage)
    fun byId(id: String): MlLanguage = all.firstOrNull { it.id == id } ?: error("unknown language '$id'; known: ${all.map { it.id }}")
}
