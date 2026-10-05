package io.github.completionml.core

import io.github.completionml.core.util.CompactFloatMap
import io.github.completionml.core.util.LongFloatMap
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.util.Random
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CompactFloatMapTest {
    @Test fun quantisedValuesAreCloseAndMissingKeysStayMissing() {
        val rnd = Random(5)
        val exact = LongFloatMap(100_000)
        val keys = LongArray(100_000) { var k = rnd.nextLong(); if (k == 0L) k = 1; k }
        for (k in keys) exact.put(k, (-12.0 * rnd.nextDouble()).toFloat())
        val c = CompactFloatMap.of(exact, 24)
        var err = 0.0
        for (k in keys) { val d = abs(c.get(k) - exact.get(k)); assertTrue(d < 0.2, "error $d"); err += d }
        assertTrue(err / keys.size < 0.03, "mean error ${err / keys.size}")
        var falsePositives = 0
        for (i in 0 until 100_000) if (!c.get(rnd.nextLong()).isNaN()) falsePositives++
        assertTrue(falsePositives <= 3, "false positives $falsePositives")
    }

    @Test fun roundTrip() {
        val exact = LongFloatMap(1000)
        val rnd = Random(6)
        for (i in 0 until 1000) exact.put(rnd.nextLong() or 1L, rnd.nextFloat() * -10)
        val c = CompactFloatMap.of(exact, 24)
        val bytes = ByteArrayOutputStream().also { c.write(DataOutputStream(it)) }.toByteArray()
        val back = CompactFloatMap.read(DataInputStream(ByteArrayInputStream(bytes)))
        assertEquals(c.size, back.size)
        exact.forEach { k, _ -> assertEquals(c.get(k), back.get(k)) }
        assertTrue(bytes.size < 1000 * 7, "bytes ${bytes.size}")
    }
}
