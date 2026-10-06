package io.github.completionml.train

import io.github.completionml.core.lex.GoLanguage
import java.io.File
import java.nio.file.Files
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ShardTest {
    private fun tmp(): File = Files.createTempDirectory("shardtest").toFile().also { it.deleteOnExit() }

    @Test fun partRoundTripForEveryWidth() {
        val rnd = Random(1)
        for (width in 2..4) {
            val maxId = when (width) { 2 -> (1 shl 14) - 1; 3 -> (1 shl 21) - 1; else -> (1 shl 29) - 1 }
            val files = List(20) { IntArray(rnd.nextInt(300)) { (rnd.nextInt(maxId + 1) shl 2) or rnd.nextInt(4) } }
            val f = File(tmp(), "p.tok")
            PartWriter(f, width).use { w -> files.forEachIndexed { i, raw -> w.add(i / 4, raw) } }
            assertEquals((ShardFormat.HEADER_BYTES + files.sumOf { 8 + it.size * width }).toLong(), f.length())
            PartReader(f).use { r ->
                assertEquals(20L, r.files); assertEquals(files.sumOf { it.size }.toLong(), r.tokens)
                for ((i, raw) in files.withIndex()) {
                    val t = r.next()!!
                    assertEquals(i / 4, t.repo); assertEquals(raw.size, t.size)
                    assertTrue(raw.indices.all { raw[it] == t.raw[it] })
                }
                assertEquals(null, r.next())
            }
        }
    }

    @Test fun vocabularyRoundTrip() {
        val v = ShardVocab(arrayOf("<ID>", "func", "héllo", "<STR>"), byteArrayOf(1, 0, 1, 3), longArrayOf(5, 100, 7, 3))
        val f = File(tmp(), "vocab.bin"); v.write(f)
        val r = ShardVocab.read(f)
        assertEquals(v.texts.toList(), r.texts.toList()); assertEquals(v.counts.toList(), r.counts.toList())
        assertTrue(r.isIdent(2) && !r.isIdent(1))
    }

    @Test fun manifestFieldsWithEscapes() {
        val l = "{\"repo\":\"a__b\",\"path\":\"dir/we\\\"ird\\\\x.go\",\"bytes\":123,\"lines\":3,\"tokens\":9,\"fold\":\"rank\",\"status\":\"ok\"}"
        assertEquals("dir/we\"ird\\x.go", ManifestLine.string(l, "path")); assertEquals("rank", ManifestLine.string(l, "fold")); assertEquals(123L, ManifestLine.long(l, "bytes"))
    }

    @Test fun lineFlagsMatchTheOffsetFormulas() {
        val text = "package a\n\nfunc F(x int) int {\n\ts := `raw\nstring` + \"a\"\n\tif x > 0 { return x }\n\treturn len(s)\n}\n"
        val tokens = GoLanguage.tokenizer.tokens(text)
        val raw = encodeFile(tokens, text, emptyMap())
        val flags = raw.map { it and 1 != 0 }
        val file = EvalFile(IntArray(raw.size), BooleanArray(raw.size), BooleanArray(raw.size) { raw[it] and 1 != 0 }, BooleanArray(raw.size) { raw[it] and 2 != 0 })
        for (i in 1 until tokens.size) {
            assertEquals(text.lastIndexOf('\n', tokens[i].offset - 1) >= tokens[i - 1].offset, flags[i], "line start of token $i")
        }
        for (i in tokens.indices) {
            val lineEnd = text.indexOf('\n', tokens[i].offset).let { if (it < 0) text.length else it }
            var old = 0; while (i + old < tokens.size && tokens[i + old].offset < lineEnd) old++
            assertEquals(old, file.restOfLine(i), "rest of line from token $i")
        }
    }

    @Test fun buildThenReadReproducesTheLexerOutput() {
        val data = tmp()
        val sources = mapOf(
            "o1__r1/a.go" to "package a\n\nfunc Add(x int, y int) int {\n\treturn x + y\n}\n",
            "o1__r1/sub/b.go" to "package a\n\nfunc Sub(x, y int) int {\n\ts := `raw\nstring`\n\t_ = s\n\treturn x - y\n}\n",
            "o2__r2/c.go" to "package c\n\nfunc Add(a, b int) int { return a + b }\n\nvar rareName1234 = 1\n",
            "o3__r3/d.go" to "package d\n\nfunc Mul(x int, y int) int { return x * y }\n",
        )
        val folds = mapOf("o1__r1" to "lm", "o2__r2" to "lm", "o3__r3" to "test")
        val manifest = File(data, "manifest.jsonl")
        manifest.bufferedWriter().use { w ->
            for ((p, text) in sources) {
                val (repo, path) = p.split("/", limit = 2)
                File(data, "repos/$p").also { it.parentFile.mkdirs() }.writeText(text)
                w.write("{\"repo\":\"$repo\",\"path\":\"$path\",\"bytes\":${text.length},\"lines\":1,\"tokens\":1,\"fold\":\"${folds[repo]}\",\"status\":\"ok\"}\n")
            }
            w.write("{\"repo\":\"o3__r3\",\"path\":\"skipped.go\",\"bytes\":1,\"lines\":1,\"tokens\":1,\"fold\":\"test\",\"status\":\"dup_exact\"}\n")
        }
        buildShards(ShardOptions(GoLanguage, data, manifest, File(data, "shards"), minCount = 2, threads = 2, partBytes = 1))
        val shards = TokenShards(File(data, "shards"))
        assertEquals(3, shards.repos.size); assertEquals(3L, shards.files("lm")); assertEquals(2, shards.parts("lm").size)   // partBytes = 1: one part per repository
        val got = ArrayList<List<String>>(); val newlines = ArrayList<Int>()
        shards.forEachFile("lm") { f -> got.add(shards.mlTokens(f).map { it.text }); newlines.add((0 until f.size).count { f.lineStart(it) }) }
        val lexed = sources.mapValues { GoLanguage.tokenizer.tokens(it.value) }
        val freq = lexed.values.flatten().groupingBy { it.text }.eachCount()
        // identifiers seen once in the whole corpus (all folds) read as <ID>
        val expected = listOf("o1__r1/a.go", "o1__r1/sub/b.go", "o2__r2/c.go").map { p ->
            lexed.getValue(p).map { if (it.kind == io.github.completionml.core.spi.TokenKind.IDENT && freq.getValue(it.text) < 2) "<ID>" else it.text }
        }
        assertTrue(expected.flatten().count { it == "<ID>" } >= 3)
        assertEquals(expected, got)
        // a line break between two tokens sets the flag: in a.go func, return and the closing brace start lines (the auto semicolon sits at the line end)
        assertEquals(3, newlines[0])
        assertEquals(1, shards.files("test").toInt())
    }
}
