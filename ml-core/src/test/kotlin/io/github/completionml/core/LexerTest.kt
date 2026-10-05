package io.github.completionml.core

import io.github.completionml.core.lex.CSharpLanguage
import io.github.completionml.core.lex.GoLanguage
import io.github.completionml.core.spi.TokenKind
import kotlin.test.Test
import kotlin.test.assertEquals

class LexerTest {
    private fun cs(src: String) = CSharpLanguage.tokenizer.tokens(src).map { it.text }
    private fun go(src: String) = GoLanguage.tokenizer.tokens(src).map { it.text }

    @Test fun csharpBasics() {
        assertEquals(
            listOf("var", "x", "=", "await", "foo", ".", "Bar", "(", "<STR>", ",", "<NUM>", ")", ";"),
            cs("var x = await foo.Bar(\"a\\\"b\", 1.5e-3); // comment"),
        )
    }

    @Test fun csharpStringsAndDirectives() {
        assertEquals(listOf("<STR>", "<STR>", "<STR>", "<STR>", "<CHAR>", "x"),
            cs("#if DEBUG\n@\"c:\\\"\"q\" \$\"a{b}\" \$@\"x\" \"\"\"raw \" text\"\"\" 'c' x\n#endif"))
        assertEquals(TokenKind.IDENT, CSharpLanguage.tokenizer.tokens("@class").single().kind)
        assertEquals(TokenKind.KEYWORD, CSharpLanguage.tokenizer.tokens("class").single().kind)
    }

    @Test fun goAutoSemicolons() {
        assertEquals(
            listOf("x", ":=", "a", ".", "B", "(", ")", ";", "return", "x", ";"),
            go("x := a.B()\nreturn x\n"),
        )
        // no semicolon after an operator or opening brace
        assertEquals(listOf("if", "x", "&&", "y", "{", "z", "++", ";", "}", ";"), go("if x &&\n y {\n z++\n}\n"))
        assertEquals(listOf("s", ":=", "<STR>", ";"), go("s := `multi\nline`"))
    }

    @Test fun offsetsAreSourcePositions() {
        val t = CSharpLanguage.tokenizer.tokens("  foo.Bar")
        assertEquals(listOf(2, 5, 6), t.map { it.offset })
    }
}
