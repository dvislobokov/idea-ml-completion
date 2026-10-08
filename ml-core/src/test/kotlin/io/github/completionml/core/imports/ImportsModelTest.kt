package io.github.completionml.core.imports

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Round trip through the `imports` container and the two queries on a tiny hand-made artifact (numbers worked out by hand). */
class ImportsModelTest {
    @TempDir lateinit var dir: File

    private fun q(p: Double) = (-ln(p) * 16).roundToInt()

    /**
     * Toy corpus: 100 files. `Client` comes from net/http 80 % and from redis 20 %; `Logger` only from log.
     * Co-imports: net/http ↔ encoding/json pmi +1.0, redis ↔ context pmi +2.0, net/http ↔ context pmi −0.5.
     */
    private fun write(file: File, params: Map<String, String> = mapOf("lambda" to "1.0", "ctx_norm" to "sum")) {
        val paths = listOf("net/http", "github.com/redis/go-redis/v9", "context", "encoding/json", "log")
        ImportsModel.write(
            file, "go", params, 100, paths, listOf(50, 10, 40, 30, 20),
            names = mapOf(
                "Client" to (50 to listOf("net/http" to q(0.8), "github.com/redis/go-redis/v9" to q(0.2))),
                "Logger" to (20 to listOf("log" to q(1.0))),
            ),
            co = mapOf(
                "net/http" to listOf("encoding/json" to 16, "context" to -8),
                "github.com/redis/go-redis/v9" to listOf("context" to 32),
            ),
        )
    }

    @Test fun roundTripAndRankImports() {
        val f = File(dir, "toy.cml"); write(f)
        val m = ImportsModel.read(f)
        assertEquals("go", m.language); assertEquals(100, m.totalDocs); assertEquals(5, m.paths.size); assertEquals(2, m.nameCount)
        assertEquals(1f, m.lambda); assertEquals("sum", m.contextNorm)
        // prior only
        val prior = m.prior("Client")
        assertEquals(listOf("net/http", "github.com/redis/go-redis/v9"), prior.map { it.path })
        assertEquals(-q(0.8) / 16f, prior[0].score, 1e-6f)
        assertEquals(-q(0.2) / 16f, prior[1].score, 1e-6f)
        // context pulls redis up: ln 0.2 + 2.0 = −1.61 + 2 > ln 0.8 − 0.5
        val withCtx = m.rankImports("Client", listOf("context", "unknown/path"))
        assertEquals("github.com/redis/go-redis/v9", withCtx[0].path)
        assertEquals((-q(0.2) + 32) / 16f, withCtx[0].score, 1e-6f)
        assertEquals((-q(0.8) - 8) / 16f, withCtx[1].score, 1e-6f)
        // pmi is symmetric: the pair stored under net/http is found when context is encoding/json
        assertEquals(16, m.pmiQ(m.pathId("encoding/json"), m.pathId("net/http")))
        assertEquals(0, m.pmiQ(m.pathId("log"), m.pathId("context")))
        // unknown name, explicit lambda
        assertTrue(m.rankImports("Nope", listOf("context")).isEmpty())
        assertEquals("net/http", m.rankImports("Client", listOf("context"), lambda = 0f)[0].path)
        // the candidate itself in the context does not count
        assertEquals(-q(0.8) / 16f, m.rankImports("Client", listOf("net/http"))[0].score, 1e-6f)
    }

    @Test fun contextNormalisation() {
        val f = File(dir, "sqrt.cml"); write(f, mapOf("lambda" to "0.5", "ctx_norm" to "sqrt"))
        val m = ImportsModel.read(f)
        assertEquals(0.5f, m.lambda)
        val r = m.rankImports("Client", listOf("context", "encoding/json"))
        // net/http: −q(0.8) + 0.5·(−8 + 16)/√2 ; redis: −q(0.2) + 0.5·32/√2
        val http = r.first { it.path == "net/http" }; val redis = r.first { it.path != "net/http" }
        assertEquals((-q(0.8) + 0.5f * 8 / sqrt(2f)) / 16f, http.score, 1e-5f)
        assertEquals((-q(0.2) + 0.5f * 32 / sqrt(2f)) / 16f, redis.score, 1e-5f)
    }

    @Test fun rankCoImports() {
        val f = File(dir, "co.cml"); write(f)
        val m = ImportsModel.read(f)
        val r = m.rankCoImports(listOf("net/http", "github.com/redis/go-redis/v9"))
        // candidates: encoding/json (pmi 16, prior ln 0.3), context (−8 + 32 = 24, prior ln 0.4); present imports excluded
        assertEquals(listOf("context", "encoding/json"), r.map { it.path })
        assertEquals((24 - ImportsModel.qLogP(0.4)) / 16f, r[0].score, 1e-6f)
        assertEquals((16 - ImportsModel.qLogP(0.3)) / 16f, r[1].score, 1e-6f)
        assertEquals(1, m.rankCoImports(listOf("net/http"), limit = 1).size)
        assertTrue(m.rankCoImports(listOf("nothing")).isEmpty())
        // the list stored under redis is also reachable from context (symmetry is not required for co-imports: only stored
        // lists are enumerated), so context alone proposes nothing here
        assertTrue(m.rankCoImports(listOf("context")).isEmpty())
    }

    @Test fun refusesOtherKinds() {
        val f = File(dir, "toy.cml"); write(f)
        val ex = runCatching { io.github.completionml.core.ngram.NgramModel.read(f) }.exceptionOrNull()
        assertTrue(ex is IllegalArgumentException && "imports" in ex.message!!, "$ex")
    }
}
