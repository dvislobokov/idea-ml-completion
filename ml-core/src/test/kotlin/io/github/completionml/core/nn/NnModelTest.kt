package io.github.completionml.core.nn

import io.github.completionml.core.nn.native.NativeNnKernels
import java.io.File
import java.nio.ByteBuffer
import java.nio.file.Files
import java.util.Random
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NnModelTest {
    private val tiny = NnConfig(vocabSize = 50, dModel = 32, nLayers = 2, nHeads = 4, nKvHeads = 2, ffnDim = 64, maxContext = 64)
    // odd sizes: column tails, tile boundaries, MHA with 1 kv head (MQA), untied head
    private val odd = NnConfig(vocabSize = 301, dModel = 96, nLayers = 2, nHeads = 3, nKvHeads = 1, ffnDim = 200, maxContext = 64, tiedEmbeddings = false)

    // native f32 computes the same numbers as the scalar kernels (summation order aside); native q8 is tested separately
    private fun kernelsUnderTest(): List<NnKernels> =
        listOfNotNull(ScalarNnKernels, NativeNnKernels.loadOrNull(NativeNnKernels.Mode.F32))

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String, tol: Float = 2e-3f) {
        assertEquals(expected.size, actual.size)
        var maxErr = 0f; var maxAbs = 0f
        for (i in expected.indices) { maxErr = maxOf(maxErr, abs(expected[i] - actual[i])); maxAbs = maxOf(maxAbs, abs(expected[i])) }
        assertTrue(maxErr <= tol * maxOf(1f, maxAbs), "$what: max error $maxErr (max |logit| $maxAbs)")
    }

    @Test fun int8RoundTrip() {
        val rnd = Random(3)
        val rows = 37; val cols = 23
        val w = FloatArray(rows * cols) { (rnd.nextGaussian() * 0.1).toFloat() }
        for (i in 0 until rows) w[i * cols + 5] = 0f   // an all-zero column
        val q = Int8Quant.quantize("w", w, rows, cols)
        assertEquals(0f, q.scales[5])
        val back = Int8Quant.dequantize(q)
        for (i in 0 until rows) for (o in 0 until cols) {
            assertTrue(abs(back[i * cols + o] - w[i * cols + o]) <= q.scales[o] / 2 + 1e-7f, "($i,$o)")
        }
        val again = Int8Quant.quantize("w", back, rows, cols)
        assertContentEquals(q.data.array(), again.data.array())
        // column max maps to ±127
        for (o in 0 until cols) if (q.scales[o] > 0) assertEquals(127, (0 until rows).maxOf { abs(q.data.get(it * cols + o).toInt()) })
    }

    @Test fun writeReadRoundTrip() {
        val dir = Files.createTempDirectory("nnfmt").toFile()
        try {
            for (cfg in listOf(tiny, odd)) {
                val tensors = NnTestModels.random(cfg, 7)
                val f = File(dir, "m.cml")
                NnFormat.write(f, cfg, mapOf("language" to "go", "corpusId" to "test"), tensors)
                assertEquals(0L, f.length() % 64)
                for (m in listOf(NnFormat.read(f), NnFormat.read(f.inputStream(), "stream"))) {
                    assertEquals(cfg, m.config)
                    assertEquals("go", m.meta["language"]); assertEquals("nn", m.meta["kind"])
                    assertEquals(tensors.map { it.name }, m.tensors.keys.toList())
                    for (t in tensors) {
                        val r = m.tensors.getValue(t.name)
                        when (t) {
                            is NnTensor.F32 -> assertContentEquals(t.data, (r as NnTensor.F32).data)
                            is NnTensor.Q8 -> {
                                r as NnTensor.Q8
                                assertEquals(t.rows, r.rows); assertEquals(t.cols, r.cols)
                                assertContentEquals(t.scales, r.scales)
                                val bytes = ByteArray(r.rows * r.cols).also { r.data.get(0, it) }
                                assertContentEquals(t.data.array(), bytes)
                            }
                        }
                    }
                }
                assertTrue(NnFormat.read(f).q8(NnFormat.Names.TOK_EMB).data.isDirect, "mapped weights are off-heap")
            }
        } finally { dir.deleteRecursively() }
    }

    @Test fun incrementalDecodeMatchesReference() {
        for (cfg in listOf(tiny, odd)) {
            val weights = NnTestModels.inMemory(cfg, 11)
            val rnd = Random(5)
            val tokens = IntArray(40) { rnd.nextInt(cfg.vocabSize) }  // ≥ 16: parallel per-token ops
            val ref = NnTestModels.referenceLogits(weights, tokens)
            for (k in kernelsUnderTest()) for (threads in listOf(1, 3)) {
                NnModel(weights, threads, k).use { model ->
                    val what = "${k.name} t=$threads d=${cfg.dModel}"
                    // full prefill
                    assertClose(ref.last(), model.newSession().prefill(tokens), "$what prefill")
                    // prefill 5 then decode one by one (small capacity to exercise cache growth)
                    val s = model.newSession(capacity = 4)
                    assertClose(ref[4], s.prefill(tokens.copyOf(5)), "$what prefill5")
                    for (t in 5 until tokens.size) assertClose(ref[t], s.decode(tokens[t]), "$what decode@$t")
                    assertEquals(tokens.size, s.length)
                    // prefix reuse: a prompt sharing 8 tokens with the cache
                    val alt = tokens.copyOf(10).also { it[8] = (it[8] + 1) % cfg.vocabSize; it[9] = 0 }
                    val altRef = NnTestModels.referenceLogits(weights, alt)
                    assertClose(altRef.last(), s.prefill(alt), "$what prefix reuse")
                    assertEquals(10, s.length)
                }
            }
        }
    }

    @Test fun generateGreedyWithLogProbs() {
        val weights = NnTestModels.inMemory(tiny, 2)
        val prompt = intArrayOf(1, 2, 3, 4)
        for (k in kernelsUnderTest()) NnModel(weights, 2, k).use { model ->
            val gen = model.newSession().generate(prompt, maxNew = 6)
            assertEquals(6, gen.tokens.size)
            var seq = prompt
            for (i in gen.tokens.indices) {
                val logits = NnTestModels.referenceLogits(weights, seq).last()
                assertEquals(Sampler.argmax(logits), gen.tokens[i], "${k.name} step $i")
                assertEquals(Sampler.logProb(logits, gen.tokens[i]), gen.logProbs[i], 1e-3f)
                seq += gen.tokens[i]
            }
            // stop token
            val stop = gen.tokens[2]
            val g2 = model.newSession().generate(prompt, maxNew = 6, stopIds = intArrayOf(stop))
            assertTrue(g2.stopped); assertEquals(stop, g2.tokens.last()); assertTrue(g2.tokens.size <= 3)
            // top-k sampling only returns top-k tokens
            val logits = NnTestModels.referenceLogits(weights, prompt).last()
            val top3 = Sampler.topKIndices(logits, 3).toSet()
            val sampler = Sampler.topK(3, 1f, Random(1))
            repeat(50) { assertTrue(sampler.sample(logits) in top3) }
        }
    }

    @Test fun kernelsMatchOnLargerShapes() {
        // matmul paths against a plain loop, incl. row-split decode and many tokens (4-token blocks + remainder)
        val rnd = Random(9)
        val rows = 300; val cols = 1000
        val q = ByteArray(rows * cols).also { rnd.nextBytes(it) }
        val w = NnTensor.Q8("w", rows, cols, ByteBuffer.allocateDirect(q.size).put(q).flip(), FloatArray(cols) { rnd.nextFloat() })
        val n = 70  // > TOK_GROUP, not a multiple of 4
        val x = Array(n) { FloatArray(rows) { rnd.nextFloat() - 0.5f } }
        val ref = Array(n) { t -> FloatArray(cols) { o -> var s = 0.0; for (i in 0 until rows) s += x[t][i] * q[i * cols + o].toDouble(); (s * w.scales[o]).toFloat() } }
        for (k in kernelsUnderTest()) {
            val y = Array(n) { FloatArray(cols) }
            val tile = KernelScratch(cols)
            k.matmulCols(w, x, n, y, 0, 500, tile); k.matmulCols(w, x, n, y, 500, cols, tile)
            for (t in 0 until n) assertClose(ref[t], y[t], "${k.name} matmul t=$t", 1e-4f)
            val acc = FloatArray(cols)
            k.matvecRowsPartial(w, x[0], 0, 123, acc, tile); k.matvecRowsPartial(w, x[0], 123, rows, acc, tile)
            for (o in 0 until cols) acc[o] *= w.scales[o]
            assertClose(ref[0], acc, "${k.name} matvec", 1e-4f)
        }
    }
}
