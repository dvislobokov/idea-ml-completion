package io.github.completionml.train

import io.github.completionml.core.eval.RankMetrics
import io.github.completionml.core.lex.Languages
import io.github.completionml.core.rank.ExampleShards
import io.github.completionml.core.rank.FeatureSchema
import io.github.completionml.core.rank.Ranker
import io.github.completionml.core.rank.Rankers
import io.github.completionml.core.rank.TrainingExample
import io.github.completionml.core.rank.TreeRanker
import java.io.DataOutputStream
import java.io.File
import java.util.zip.GZIPOutputStream

/**
 * `dump-features`: the expanded feature matrix of example shards for an external trainer (LightGBM, `tools/gbdt/train_gbdt.py`).
 * The vectors are exactly what [Ranker.score] receives ([FeatureSchema.expand]), so the feature semantics stay in `ml-core`.
 *
 * Format (big-endian, gzip):
 * ```
 * "CMLF" i32 1 | i32 n, UTF×n names | i32 lists, per list: UTF repo, u8 kind, i32 chosen, i32 candidates | i32 rows, f32 × rows × n
 * ```
 * `repo` is the shard file name without extension (one shard per repository), so the trainer can split by repository.
 */
internal fun dumpFeatures(args: Map<String, String>) {
    val dir = File(args.getValue("shards"))
    val out = File(args.getValue("out"))
    val files = (if (dir.isDirectory) dir.listFiles { f -> f.name.endsWith(ExampleShards.EXTENSION) }!!.sortedBy { it.name } else listOf(dir))
    require(files.isNotEmpty()) { "no shards in $dir" }
    val schema = ExampleShards.header(files[0]).schema
    out.parentFile?.mkdirs()
    // two passes: index first (needs the counts), then the matrix
    val index = ArrayList<Triple<String, TrainingExample, Int>>()
    var rows = 0
    DataOutputStream(GZIPOutputStream(out.outputStream().buffered(1 shl 16), 1 shl 16)).use { o ->
        o.writeBytes("CMLF"); o.writeInt(1)
        o.writeInt(schema.size); for (n in schema.names) o.writeUTF(n)
        for (f in files) {
            val repo = f.name.removeSuffix(ExampleShards.EXTENSION)
            val h = ExampleShards.forEach(f) { ex -> index.add(Triple(repo, ex, ex.size)); rows += ex.size }
            require(h.schema.hash == schema.hash) { "$f: schema differs" }
        }
        o.writeInt(index.size)
        for ((repo, ex, _) in index) { o.writeUTF(repo); o.writeByte(ex.kind.ordinal); o.writeInt(ex.chosen); o.writeInt(ex.size) }
        o.writeInt(rows)
        val buf = FloatArray(schema.size)
        for ((_, ex, _) in index) for (c in 0 until ex.size) { ex.expand(c, buf); for (x in buf) o.writeFloat(x) }
    }
    log("dumped ${index.size} lists / $rows rows × ${schema.size} features from ${files.size} shards to $out (${out.length() shr 10} KB)")
}

/** `import-gbdt`: text trees from `train_gbdt.py --export-trees` → `.cml` kind `tree-ranker`; the schema is checked against the shards. */
internal fun importGbdt(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    val ranker = File(args.getValue("trees")).bufferedReader().use { TreeRanker.parseText(it) }
    args["shards"]?.let {
        val d = File(it)
        val first = (if (d.isDirectory) d.listFiles { f -> f.name.endsWith(ExampleShards.EXTENSION) }!!.sortedBy { f -> f.name }.first() else d)
        val schema = ExampleShards.header(first).schema
        require(schema.names == ranker.schema.names) { "tree feature names differ from the shard schema" }
    }
    val out = File(args.getValue("out"))
    ranker.write(out, lang.id, args["source"] ?: "trees=${args.getValue("trees")}")
    log("written $out (${out.length() shr 10} KB): ${ranker.description}")
}

/** Scores every test list with [ranker] and reports the metrics plus the scoring cost (warm JIT, lists of [benchSize] candidates). */
internal fun evalRankerOnShards(ranker: Ranker, shardsDir: File, benchSize: Int = 50) {
    val (header, examples) = ExampleShards.readAll(shardsDir)
    require(header.schema.hash == ranker.schema.hash) { "ranker schema differs from the shards'" }
    val m = RankMetrics(); val lm = RankMetrics(); val rules = RankMetrics()
    val lmIdx = ranker.schema.baseIndex("lm_logprob"); val ruleIdx = ranker.schema.baseIndex("rule_rank_log")
    for (ex in examples) {
        m.add(ex, ranker.scores(ex)); lm.add(ex, FloatArray(ex.size) { ex.baseFeature(it, lmIdx) })
        if (ruleIdx >= 0) rules.add(ex, FloatArray(ex.size) { -ex.baseFeature(it, ruleIdx) })
    }
    System.err.print(m.report("ranker (${ranker.description})"))
    System.err.print(lm.report("baseline: n-gram log-prob only"))
    if (ruleIdx >= 0) System.err.print(rules.report("baseline: plugin rules (rule_rank_log)"))
    log("scoring cost: " + benchScoring(ranker, examples, benchSize))
}

/** Microseconds per list of [size] candidates (expand + score), after warm-up; rows are taken from the real examples. */
internal fun benchScoring(ranker: Ranker, examples: List<TrainingExample>, size: Int): String {
    val dim = ranker.schema.size
    val lists = ArrayList<Array<FloatArray>>()
    val buf = FloatArray(dim)
    val pool = ArrayList<FloatArray>()
    for (ex in examples) { for (c in 0 until ex.size) { ex.expand(c, buf); pool.add(buf.copyOf()) }; if (pool.size >= 200 * size) break }
    for (i in 0 until pool.size / size) lists.add(Array(size) { pool[i * size + it] })
    var sink = 0f
    fun run(reps: Int): Long {
        val t0 = System.nanoTime()
        repeat(reps) { for (l in lists) for (x in l) sink += ranker.score(x) }
        return System.nanoTime() - t0
    }
    run(20)                                    // JIT warm-up
    val reps = 50
    val ns = run(reps)
    val perList = ns / 1000.0 / (reps * lists.size)
    return "%.1f µs per list of %d candidates (%.2f µs per candidate; %d lists × %d reps, sink %.0f)".format(perList, size, perList / size, lists.size, reps, sink)
}
