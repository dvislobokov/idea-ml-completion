package io.github.completionml.core

import io.github.completionml.core.rank.FeatureSchema
import io.github.completionml.core.rank.LinearRanker
import io.github.completionml.core.rank.Ranker
import io.github.completionml.core.rank.Rankers
import io.github.completionml.core.rank.TreeRanker
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TreeRankerTest {
    private val schema = FeatureSchema.common()

    /** One stump per missing mode on feature 0: left leaf 1.0, right leaf 2.0, threshold 0.5. */
    private fun stump(missingType: Int, defaultLeft: Boolean) = TreeRanker.Tree(
        intArrayOf(0, -1, -1), doubleArrayOf(0.5, 0.0, 0.0), byteArrayOf(((if (defaultLeft) 1 else 0) or (missingType shl 1)).toByte(), 0, 0),
        intArrayOf(1, -1, -1), intArrayOf(2, -1, -1), doubleArrayOf(0.0, 1.0, 2.0))

    private fun score(tree: TreeRanker.Tree, x0: Float): Double = tree.predict(FloatArray(schema.size).also { it[0] = x0 })

    @Test fun `missing values follow LightGBM's three modes`() {
        // None: NaN becomes 0 and is compared normally
        val none = stump(TreeRanker.MISSING_NONE, defaultLeft = false)
        assertEquals(1.0, score(none, 0.2f)); assertEquals(2.0, score(none, 0.7f)); assertEquals(1.0, score(none, Float.NaN))
        // Zero: zeros (and NaN, which becomes 0) take the default direction
        val zeroRight = stump(TreeRanker.MISSING_ZERO, defaultLeft = false)
        assertEquals(2.0, score(zeroRight, 0f)); assertEquals(2.0, score(zeroRight, Float.NaN)); assertEquals(1.0, score(zeroRight, 0.2f))
        val zeroLeft = stump(TreeRanker.MISSING_ZERO, defaultLeft = true)
        assertEquals(1.0, score(zeroLeft, 0f)); assertEquals(2.0, score(zeroLeft, 0.7f))
        // NaN: only NaN takes the default direction, zero is an ordinary value
        val nanRight = stump(TreeRanker.MISSING_NAN, defaultLeft = false)
        assertEquals(2.0, score(nanRight, Float.NaN)); assertEquals(1.0, score(nanRight, 0f)); assertEquals(1.0, score(nanRight, 0.5f))
        val nanLeft = stump(TreeRanker.MISSING_NAN, defaultLeft = true)
        assertEquals(1.0, score(nanLeft, Float.NaN)); assertEquals(2.0, score(nanLeft, 0.7f))
    }

    @Test fun `text import, cml round trip and Rankers dispatch`() {
        val text = buildString {
            appendLine("tree-ranker 1"); appendLine("features ${schema.size}"); schema.names.forEach { appendLine(it) }
            appendLine("trees 2")
            appendLine("tree 3"); appendLine("0 0.5 0 0 1 2 0.0"); appendLine("-1 0 0 0 -1 -1 1.0"); appendLine("-1 0 0 0 -1 -1 2.0")
            appendLine("tree 3"); appendLine("3 1.25 1 2 1 2 0.0"); appendLine("-1 0 0 0 -1 -1 -0.5"); appendLine("-1 0 0 0 -1 -1 0.25")
        }
        val r = TreeRanker.parseText(text.reader().buffered())
        assertEquals(2, r.trees.size); assertEquals(4, r.leaves); assertEquals(schema.hash, r.schema.hash)
        val x = FloatArray(schema.size).also { it[0] = 0.7f; it[3] = Float.NaN }
        assertEquals(2.0f + -0.5f, r.score(x))
        val f = File.createTempFile("tree", ".cml").also { it.deleteOnExit() }
        r.write(f, "go", "test")
        val back = Rankers.read(f)
        assertTrue(back is TreeRanker); assertEquals(r.score(x), back.score(x)); assertEquals(r.schema.names, back.schema.names)
        assertEquals("2 trees, 4 leaves", back.description)
        // the linear reader still works through the same loader
        val lin = LinearRanker(schema, FloatArray(schema.size) { 1f }, FloatArray(schema.size), FloatArray(schema.size) { 1f })
        val lf = File.createTempFile("lin", ".cml").also { it.deleteOnExit() }
        lin.write(lf, "go", "test")
        assertTrue(Rankers.read(lf) is LinearRanker)
    }

    /**
     * Parity with LightGBM: `tools/gbdt/train_gbdt.py --fixture` wrote whole test lists (expanded features exactly as the
     * plugins build them) with `booster.predict` scores; the shipped `.cml` must reproduce them and the top-1 of every list.
     */
    @Test fun `shipped models reproduce LightGBM scores`() {
        for (lang in listOf("go", "cs")) {
            val model = javaClass.getResourceAsStream("/rank/$lang-rank-gbdt-e19.cml") ?: error("missing resource $lang-rank-gbdt-e19.cml")
            val ranker: Ranker = Rankers.read(model, "$lang-rank-gbdt-e19.cml")
            assertTrue(ranker is TreeRanker)
            val lines = GZIPInputStream(javaClass.getResourceAsStream("/rank/$lang-gbdt-parity.tsv.gz")!!).bufferedReader().readLines().filter { !it.startsWith("#") }
            val groups = LinkedHashMap<Int, MutableList<Pair<Float, Float>>>()   // group -> (kotlin, reference)
            var maxDiff = 0.0
            for (l in lines) {
                val p = l.split('\t')
                val g = p[0].toInt(); val ref = p.last().toFloat()
                val x = FloatArray(ranker.schema.size) { p[2 + it].toFloat() }
                assertEquals(ranker.schema.size, p.size - 3)
                val s = ranker.score(x)
                maxDiff = maxOf(maxDiff, abs(s - ref).toDouble())
                groups.getOrPut(g) { ArrayList() }.add(s to ref)
            }
            assertTrue(maxDiff < 1e-5, "$lang: max |kotlin − lightgbm| = $maxDiff")
            assertTrue(lines.size >= 200, "$lang: only ${lines.size} fixture rows")
            for ((g, rows) in groups) {
                val top = rows.indices.maxByOrNull { rows[it].first }!!; val refTop = rows.indices.maxByOrNull { rows[it].second }!!
                assertEquals(refTop, top, "$lang: list $g top-1 differs")
            }
        }
    }
}
