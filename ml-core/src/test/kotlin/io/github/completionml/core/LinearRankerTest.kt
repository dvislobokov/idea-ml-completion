package io.github.completionml.core

import io.github.completionml.core.eval.RankMetrics
import io.github.completionml.core.rank.FeatureSchema
import io.github.completionml.core.rank.LinearRanker
import io.github.completionml.core.rank.LinearRankerTrainer
import io.github.completionml.core.rank.TrainingExample
import io.github.completionml.core.spi.ContextKind
import java.io.File
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LinearRankerTest {
    private val schema = FeatureSchema.common()
    private val sig = FeatureSchema.BASE.indexOf("lm_logprob"); private val noise = FeatureSchema.BASE.indexOf("name_len"); private val anti = FeatureSchema.BASE.indexOf("recency_log")

    /** The chosen candidate has a higher "signal", a lower "anti", and random "noise". */
    private fun synthetic(n: Int, rnd: Random): List<TrainingExample> = List(n) {
        val size = 2 + rnd.nextInt(8)
        val chosen = rnd.nextInt(size)
        val f = Array(size) { c -> FloatArray(FeatureSchema.BASE.size).also { b -> b[sig] = rnd.nextFloat() + (if (c == chosen) 0.6f else 0f); b[noise] = rnd.nextFloat(); b[anti] = rnd.nextFloat() - (if (c == chosen) 0.4f else 0f) } }
        FeatureSchema.fillListFeatures(f)
        TrainingExample(ContextKind.values()[it % ContextKind.values().size], f, chosen)
    }

    @Test fun learnsSeparableSignal() {
        val rnd = Random(1)
        val train = synthetic(2000, rnd); val test = synthetic(500, rnd)
        val (ranker, report) = LinearRankerTrainer(schema, epochs = 5).train(train)
        assertTrue(report.epochLoss.last() < report.epochLoss.first())
        assertTrue(ranker.weights[sig] > 0 && ranker.weights[anti] < 0, "weights ${ranker.weights.toList()}")
        assertTrue(kotlin.math.abs(ranker.weights[noise]) < kotlin.math.abs(ranker.weights[sig]) / 3)
        val m = RankMetrics()
        for (ex in test) m.add(ex, ranker.scores(ex))
        assertTrue(m.all.top1Rate > 0.8, "top1 ${m.all.top1Rate}")
    }

    @Test fun anchorKeepsWeightsClose() {
        val rnd = Random(2)
        val train = synthetic(300, rnd)
        val anchor = FloatArray(schema.size) { 5f }
        val (r, _) = LinearRankerTrainer(schema, l2 = 10.0, learningRate = 0.01, epochs = 2, anchor = anchor).train(train)
        for (i in 0 until schema.size) assertTrue(kotlin.math.abs(r.weights[i] - 5f) < 1f, r.weights.toList().toString())
    }

    @Test fun roundTrip() {
        val (r, _) = LinearRankerTrainer(schema, epochs = 1).train(synthetic(50, Random(3)))
        val f = File.createTempFile("rank", ".cml").also { it.deleteOnExit() }
        r.write(f, "go", "test")
        val back = LinearRanker.read(f)
        assertEquals(r.weights.toList(), back.weights.toList())
        assertEquals(r.schema.names, back.schema.names)
    }
}
