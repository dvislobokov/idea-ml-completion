package io.github.completionml.core.imports

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The Kotlin queries reproduce the Python reference (`tools/imports/imports_model.py`) on the published artifacts:
 * `src/test/resources/imports/<lang>-parity.txt` holds 50 `rankImports` and 50 `rankCoImports` queries from the test fold with
 * the expected (path, score) lists, written by `tools/imports/eval_imports.py --fixture`. The models are looked up in
 * `$CML_MODELS_DIR`, else `../models` (the engine checkout); skipped when absent (plugin subtrees).
 */
class ImportsParityTest {
    companion object {
        val modelsDir: File get() = File(System.getenv("CML_MODELS_DIR") ?: "../models")
        fun model(lang: String) = File(modelsDir, if (lang == "go") "go-imports-e20.cml" else "cs-imports-e20.cml")

        /** Lines of the fixture as (kind, name-or-null, context, expected list). */
        fun fixture(lang: String): List<Query> {
            val text = ImportsParityTest::class.java.getResource("/imports/$lang-parity.txt")?.readText() ?: return emptyList()
            return text.lineSequence().filter { it.isNotBlank() && !it.startsWith("#") }.map { line ->
                val f = line.split('\t')
                val expected = f.last().split(';').filter { it.isNotEmpty() }.map {
                    val eq = it.lastIndexOf('='); it.substring(0, eq) to it.substring(eq + 1).toFloat()
                }
                if (f[0] == "R") Query(f[1], f[2].split(',').filter { it.isNotEmpty() }, expected)
                else Query(null, f[1].split(',').filter { it.isNotEmpty() }, expected)
            }.toList()
        }

        class Query(val name: String?, val context: List<String>, val expected: List<Pair<String, Float>>)

        fun check(lang: String, out: Appendable?): Int {
            val tLoad = System.nanoTime()
            val m = ImportsModel.read(model(lang))
            val loadMs = (System.nanoTime() - tLoad) / 1e6
            val qs = fixture(lang)
            var checked = 0
            for (q in qs) {
                val got = if (q.name != null) m.rankImports(q.name, q.context) else m.rankCoImports(q.context, 10)
                assertEquals(q.expected.size, got.size, "${q.name ?: "co"} ${q.context}: size")
                for (i in q.expected.indices) {
                    val (ep, es) = q.expected[i]
                    assertTrue(abs(es - got[i].score) < 2e-3f, "${q.name ?: "co"} #$i: expected $ep=$es got ${got[i]}")
                    // order: the same path unless the scores tie within the float tolerance
                    if (got[i].path != ep) {
                        val other = got.indexOfFirst { it.path == ep }
                        assertTrue(other >= 0 && abs(got[other].score - got[i].score) < 2e-3f, "${q.name ?: "co"} #$i: expected $ep got ${got[i].path}")
                    }
                }
                checked++
            }
            // latency, warmed up: the rankImports queries of the fixture
            val rank = qs.filter { it.name != null }
            if (out != null && rank.isNotEmpty()) {
                repeat(20) { for (q in rank) m.rankImports(q.name!!, q.context) }
                val reps = 200
                val t0 = System.nanoTime()
                var sink = 0
                repeat(reps) { for (q in rank) sink += m.rankImports(q.name!!, q.context).size }
                val us = (System.nanoTime() - t0) / 1e3 / (reps * rank.size)
                val co = qs.filter { it.name == null }
                val t1 = System.nanoTime()
                repeat(reps) { for (q in co) sink += m.rankCoImports(q.context, 10).size }
                val usCo = (System.nanoTime() - t1) / 1e3 / (reps * co.size)
                out.append("$lang: ${model(lang).length() / 1024} KB, ${m.paths.size} paths, ${m.nameCount} names, load ${"%.0f".format(loadMs)} ms; rankImports ${"%.2f".format(us)} µs, rankCoImports ${"%.2f".format(usCo)} µs ($sink)\n")
            }
            return checked
        }
    }

    @Test fun goMatchesPython() {
        assumeTrue(model("go").isFile && fixture("go").isNotEmpty(), "go-imports-e20.cml / fixture not present")
        assertTrue(check("go", System.out) >= 50)
    }

    @Test fun csharpMatchesPython() {
        assumeTrue(model("csharp").isFile && fixture("csharp").isNotEmpty(), "cs-imports-e20.cml / fixture not present")
        assertTrue(check("csharp", System.out) >= 50)
    }
}
