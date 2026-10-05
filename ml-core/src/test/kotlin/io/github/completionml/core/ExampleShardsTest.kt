package io.github.completionml.core

import io.github.completionml.core.lex.Languages
import io.github.completionml.core.rank.ExampleShards
import io.github.completionml.core.rank.FeatureExtractor
import io.github.completionml.core.rank.FeatureSchema
import io.github.completionml.core.rank.FileState
import io.github.completionml.core.rank.TrainingExample
import io.github.completionml.core.spi.ContextKind
import io.github.completionml.core.vocab.Vocabulary
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ExampleShardsTest {
    private val go = Languages.byId("go")
    private val tokens = go.tokenizer.tokens("package p\nfunc f(ctx Context, err error) error {\n\tif err != nil { return err }\n\treturn ctx.Err()\n}\n")
    private val vocab = Vocabulary.Builder().also { it.addFile(tokens) }.build(1000, 1)

    @Test fun `schema with a language block keeps the common block first and hashes differently`() {
        val plain = FeatureSchema.common()
        val lang = FeatureSchema.common(languageFeatures = listOf("expected_type_match", "scope_level"))
        assertEquals(FeatureSchema.BASE.size + 2, lang.baseSize)
        assertEquals(listOf("expected_type_match", "scope_level"), lang.languageFeatures)
        assertEquals(lang.baseSize * (1 + FeatureSchema.KINDS), lang.size)
        assertTrue(plain.hash != lang.hash)
        assertFailsWith<IllegalArgumentException> { FeatureSchema.common(languageFeatures = listOf("lm_logprob")) }
    }

    @Test fun `extractor appends the language block after the common features`() {
        val schema = FeatureSchema.common(languageFeatures = listOf("expected_type_match", "scope_level"))
        val ex = FeatureExtractor(schema, vocab, null)
        val state = FileState(vocab).also { it.addAll(tokens.take(12)) }
        val names = arrayOf("err", "ctx", "nil")
        val lang = arrayOf(floatArrayOf(1f, 2f), floatArrayOf(0f, 1f), floatArrayOf(0f, 0f))
        val f = ex.features(state, "", names, lang)
        assertEquals(schema.baseSize, f[0].size)
        assertEquals(1f, f[0][FeatureSchema.BASE.size]); assertEquals(2f, f[0][FeatureSchema.BASE.size + 1])
        assertEquals(1f, f[0][schema.baseIndex("in_vocab")])          // err is in the vocabulary
        assertTrue(f[0][schema.baseIndex("file_freq_log")] > 0f)      // err occurred before position 12
        assertFailsWith<IllegalArgumentException> { ex.features(state, "", names, null) }
        // expansion into the full vector puts the language block into the kind-specific slot too
        val full = FloatArray(schema.size)
        TrainingExample(ContextKind.ARGUMENT, f, 0).expand(0, full)
        assertEquals(2f, full[schema.index("argument:scope_level")])
    }

    @Test fun `shards round-trip examples, names and schema`() {
        val schema = FeatureSchema.common(languageFeatures = listOf("kind_local"))
        val ex = FeatureExtractor(schema, vocab, null)
        val state = FileState(vocab).also { it.addAll(tokens.take(12)) }
        val names = arrayOf("err", "ctx")
        val examples = List(5) { i -> TrainingExample(ContextKind.values()[i % ContextKind.values().size], ex.features(state, "", names, arrayOf(floatArrayOf(1f), floatArrayOf(0f))), i % 2, names) }
        val dir = File.createTempFile("shards", "").also { it.delete(); it.mkdirs() }
        try {
            ExampleShards.Writer(File(dir, "a.cmlx"), "go", "test", schema, withNames = true).use { w -> examples.take(3).forEach(w::add) }
            ExampleShards.Writer(File(dir, "b.cmlx"), "go", "test", schema).use { w -> examples.drop(3).forEach(w::add) }
            val (header, read) = ExampleShards.readAll(dir)
            assertEquals("go", header.language); assertEquals(schema.hash, header.schema.hash)
            assertEquals(5, read.size)
            for (i in read.indices) {
                assertEquals(examples[i].kind, read[i].kind); assertEquals(examples[i].chosen, read[i].chosen)
                for (c in names.indices) assertContentEquals(examples[i].base[c], read[i].base[c])
            }
            assertContentEquals(names, read[0].candidateNames!!); assertEquals(null, read[4].candidateNames)
            // a shard with another schema is rejected
            ExampleShards.Writer(File(dir, "c.cmlx"), "go", "test", FeatureSchema.common()).use { }
            assertFailsWith<IllegalArgumentException> { ExampleShards.readAll(dir) }
        } finally { dir.deleteRecursively() }
    }
}
