package io.github.completionml.train

import io.github.completionml.core.eval.RankMetrics
import io.github.completionml.core.lex.Languages
import io.github.completionml.core.ngram.CacheLm
import io.github.completionml.core.ngram.MixedLm
import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.ngram.NgramTable
import io.github.completionml.core.ngram.TokenLm
import io.github.completionml.core.ngram.NgramTrainer
import io.github.completionml.core.ngram.PerplexityAccumulator
import io.github.completionml.core.rank.ExampleShards
import io.github.completionml.core.rank.FeatureSchema
import io.github.completionml.core.rank.LinearRanker
import io.github.completionml.core.rank.LinearRankerTrainer
import io.github.completionml.core.rank.ProxyExampleGenerator
import io.github.completionml.core.rank.TrainingExample
import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.vocab.Vocabulary
import java.io.File
import kotlin.math.ln
import kotlin.system.exitProcess

private const val USAGE = """
usage:
  l2  --lang csharp|go --data <dir> --out <lm.cml> [--order 4] [--vocab 50000] [--min-count 1,1,2,2] [--split auto|repo|file] [--fp-bits 24] [--eval-exact true] [--min-repos 1,1,1,2] [--cache 0.3] [--smoothing mkn|jm] [--lambda 0.5]
      count n-grams on the train split, estimate modified Kneser-Ney, evaluate on the test split, write the model
  l1  --lang csharp|go --data <dir> --lm <lm.cml> --out <rank.cml> [--epochs 10] [--l2 1e-4] [--lr 0.1] [--max-files N] [--per-file 60] [--cache 0.3] [--features 8]
                                      [--dump-shards <dir>] | --shards <dir> [--test-shards <dir>]  (train on plugin-generated shards)
      NOTE: train the LM on repositories disjoint from the ranker's training repositories (--repos lists), otherwise the
      LM feature is inflated on the ranker's training data and the ranker over-trusts it on unseen projects.
      generate proxy ranking examples, train the listwise logistic regression, report metrics vs. baselines
  eval-lm --lang .. --data <dir> --lm <lm.cml>      re-evaluate an n-gram model on the test split
  common: --repos <file>   restrict the corpus to the repo directory names listed in the file (one per line)
          --dedup 0.8      drop exact duplicates and near-duplicates (MinHash Jaccard over identifiers >= value), train first then test
          --split repo|file  force the split (auto: by repo when >= 20 repos, else by file)
          --test-repos a__b,c__d   hold out exactly these repositories (directory names)
  eval-rank --lang .. --rank <rank.cml> (--data <dir> --lm <lm.cml> [--max-files N] | --shards <dir>)
      re-evaluate a ranker (with the LM it was trained with) on the test split of another corpus
  eval-inline --lang .. --data <dir> --lm <lm.cml> [--cache 0.3] [--max-tokens 8] [--stride 50] [--max-test-files N]
      greedy multi-token continuation (inline "grey text") on the test split: how many of the next tokens the LM gets right
  tokens  --lang .. --file <path>                   dump tokens (lexer debugging)
"""

fun main(argv: Array<String>) {
    if (argv.isEmpty()) { System.err.println(USAGE); exitProcess(2) }
    val args = parse(argv.drop(1))
    val t0 = System.currentTimeMillis()
    when (argv[0]) {
        "l2" -> trainLm(args)
        "l1" -> trainRanker(args)
        "eval-lm" -> evalLm(args)
        "eval-rank" -> evalRanker(args)
        "eval-inline" -> evalInline(args)
        "tokens" -> dumpTokens(args)
        else -> { System.err.println(USAGE); exitProcess(2) }
    }
    log("done in %.1f s".format((System.currentTimeMillis() - t0) / 1000.0))
}

private fun parse(a: List<String>): Map<String, String> {
    val m = HashMap<String, String>()
    var i = 0
    while (i < a.size) {
        require(a[i].startsWith("--")) { "unexpected argument ${a[i]}" }
        m[a[i].removePrefix("--")] = a.getOrNull(i + 1) ?: ""
        i += 2
    }
    return m
}

internal fun log(s: String) = System.err.println("[%tT] %s".format(System.currentTimeMillis(), s))

internal fun corpus(args: Map<String, String>, lang: io.github.completionml.core.spi.MlLanguage) =
    Corpus(File(args.getValue("data")), lang, args["split"] ?: "auto", includeList = args["repos"]?.let { File(it) },
           testRepos = args["test-repos"]?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }?.toSet() ?: emptySet())

internal fun readTokens(c: Corpus, s: Corpus.Source): List<MlToken> = c.language.tokenizer.tokens(s.file.readText())

/**
 * Train/test files after the split and `--dedup`; tokens are not kept — every pass re-tokenises the files in parallel batches
 * ([forEachTokenised]), so memory does not grow with the corpus (a million files would not fit as token lists).
 */
class Loaded(val train: List<Corpus.Source>, val test: List<Corpus.Source>)

/** Tokenises [sources] in parallel batches and hands the files to [consumer] one by one, in order, on the calling thread. */
internal fun forEachTokenised(corpus: Corpus, sources: List<Corpus.Source>, batch: Int = 256, consumer: (Corpus.Source, List<MlToken>) -> Unit) {
    for (from in sources.indices step batch) {
        val chunk = sources.subList(from, minOf(from + batch, sources.size))
        val tokens = chunk.parallelStream().map { readTokens(corpus, it) }.toList()
        for (i in chunk.indices) consumer(chunk[i], tokens[i])
    }
}

internal fun load(corpus: Corpus, args: Map<String, String>, maxFiles: Int? = null): Loaded {
    val files = corpus.files().let { f -> maxFiles?.let { f.take(it) } ?: f }
    val (testAll, trainSrc) = files.partition { corpus.isTest(it) }
    val maxTest = args["max-test-files"]?.toInt()
    val testSrc = if (maxTest == null || testAll.size <= maxTest) testAll
                  else { val step = testAll.size.toDouble() / maxTest; List(maxTest) { testAll[(it * step).toInt()] } }
    val dedup = args["dedup"]?.toDouble()?.let { Dedup(it) }
    fun take(list: List<Corpus.Source>): List<Corpus.Source> {
        if (dedup == null) return list
        val kept = ArrayList<Corpus.Source>(list.size)
        forEachTokenised(corpus, list) { s, t -> if (dedup.offer(t)) kept.add(s) }
        return kept
    }
    val train = take(trainSrc); val test = take(testSrc)
    log("${corpus.language.id}: ${corpus.repos.size} repos, ${files.size} files (train ${trainSrc.size}, test ${testSrc.size}), split by ${if (corpus.repos.size >= 20 && corpus.splitBy == "auto") "repo" else corpus.splitBy}")
    if (dedup != null) log("$dedup -> train ${train.size}, test ${test.size}")
    log("test repos: ${testSrc.map { it.repo }.distinct().sorted()}")
    return Loaded(train, test)
}

private fun trainLm(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    val corpus = corpus(args, lang)
    val order = args["order"]?.toInt() ?: 4
    val maxVocab = args["vocab"]?.toInt() ?: 50_000
    fun perOrder(key: String) = (args[key] ?: "1").split(",").map { it.trim().toInt() }.let { l -> IntArray(order) { l.getOrElse(it) { l.last() } } }
    val minCounts = perOrder("min-count"); val minRepos = perOrder("min-repos")
    val loaded = load(corpus, args)
    val train = loaded.train; val test = loaded.test
    log("order $order")

    // pass 1: vocabulary
    val vb = Vocabulary.Builder()
    forEachTokenised(corpus, train) { _, t -> vb.addFile(t) }
    val vocab = vb.build(maxVocab, minDocFreq = if (corpus.repos.size >= 20) 2 else 1)
    log("vocabulary: ${vocab.size} entries from ${vb.tokens} tokens in ${vb.files} files")

    // pass 2: counts (files re-tokenised; grouped by repository so that repo counts are exact)
    // initial table sizes: distinct n-grams are a fraction of the tokens; the tables grow (doubling) up to NgramTable.MAX_SLOTS
    val trainer = NgramTrainer(order, vocab, expectedTokens = (vb.tokens / 4).coerceIn(1L shl 16, NgramTable.MAX_EXPECTED.toLong()).toInt(), trackRepos = minRepos.any { it > 1 })
    val repoIndex = corpus.repos.withIndex().associate { it.value.name to it.index }
    forEachTokenised(corpus, train) { s, t -> trainer.addFile(vocab.encode(t), repoIndex.getValue(s.repo)) }
    log("counted ${trainer.tokens} tokens in ${trainer.files} files")
    val smoothing = NgramTrainer.Smoothing.valueOf((args["smoothing"] ?: "mkn").uppercase())
    val exact = trainer.estimate(minCounts, minRepos, smoothing, args["lambda"]?.toDouble() ?: 0.5) { log(it) }
    log("smoothing: $smoothing" + if (smoothing == NgramTrainer.Smoothing.JM) " λ=${args["lambda"] ?: "0.5"}" else "")
    val bits = args["fp-bits"]?.toInt() ?: 24
    val out = File(args.getValue("out"))
    exact.write(out, lang.id, "repos=${corpus.repos.joinToString(",") { it.name }}", bits)
    log("written $out (${out.length() / 1024} KB, $bits-bit fingerprints, 8-bit values)")

    val cacheLambda = args["cache"]?.toDouble() ?: 0.0
    if (args["eval-exact"] == "true") { log("eval of the exact (unquantised) model:"); evaluateLm(exact, corpus, test, cacheLambda) }
    if (args["no-eval"] != "true") evaluateLm(NgramModel.read(out), corpus, test, cacheLambda)
}

private fun evalLm(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    val corpus = corpus(args, lang)
    val model = NgramModel.read(File(args.getValue("lm")))
    evaluateLm(model, corpus, load(corpus, args).test, args["cache"]?.toDouble() ?: 0.0)
}

/** Per-file evaluation result; files are independent (the cache is per file), so they are evaluated in parallel. */
private class LmFileStats {
    var tokens = 0L; var sumAll = 0.0; var identTokens = 0L; var sumIdent = 0.0
    var identPositions = 0L; var top1 = 0L; var top5 = 0L; var oov = 0L; var scored = 0L
    fun add(o: LmFileStats) {
        tokens += o.tokens; sumAll += o.sumAll; identTokens += o.identTokens; sumIdent += o.sumIdent
        identPositions += o.identPositions; top1 += o.top1; top5 += o.top5; oov += o.oov; scored += o.scored
    }
}

/** Perplexity over all tokens; next-token top-1/top-5 over identifiers that are in the vocabulary (what gray text could predict). */
private fun evaluateLm(model: NgramModel, corpus: Corpus, test: List<Corpus.Source>, cacheLambda: Double = 0.0) {
    val t0 = System.currentTimeMillis()
    val vocab = model.vocab
    val sampleEvery = 20   // top-k over the whole vocabulary is O(|V|): sample positions
    val stats = test.parallelStream().map { source ->
        val tokens = readTokens(corpus, source)
        val st = LmFileStats()
        val ids = vocab.encode(tokens)
        val padded = IntArray(ids.size + model.order - 1) { if (it < model.order - 1) Vocabulary.BOS_ID else ids[it - model.order + 1] }
        val cache = if (cacheLambda > 0) CacheLm(3, vocab.size) else null
        val lm: TokenLm = if (cache != null) MixedLm(model, cache, cacheLambda) else model
        for (i in ids.indices) {
            val pos = i + model.order - 1
            val lp = lm.logProb(padded, pos, ids[i])
            st.tokens++; st.sumAll += lp
            if (tokens[i].kind == TokenKind.IDENT) {
                st.identPositions++
                if (ids[i] == Vocabulary.UNK_ID) { st.oov++; cache?.add(ids[i]); continue }
                st.identTokens++; st.sumIdent += lp
                if (st.identPositions % sampleEvery == 0L) {
                    st.scored++
                    val top = if (cache == null) model.topK(padded, pos, 5) { vocab.isIdentifier(it) && it != Vocabulary.UNK_ID }
                              else topKMixed(model, cache, cacheLambda, padded, pos, 5) { vocab.isIdentifier(it) && it != Vocabulary.UNK_ID }
                    if (top.isNotEmpty() && top[0].first == ids[i]) st.top1++
                    if (top.any { it.first == ids[i] }) st.top5++
                }
            }
            cache?.add(ids[i])
        }
        st
    }.reduce(LmFileStats()) { x, y -> LmFileStats().also { it.add(x); it.add(y) } }
    val ppAll = kotlin.math.exp(-stats.sumAll / stats.tokens.coerceAtLeast(1))
    val bits = -stats.sumAll / stats.tokens.coerceAtLeast(1) / ln(2.0)
    val ppIdent = kotlin.math.exp(-stats.sumIdent / stats.identTokens.coerceAtLeast(1))
    log("LM eval on ${test.size} files%s: perplexity all=%.1f (%.2f bits), identifiers in vocab=%.1f; identifier OOV rate=%.1f%%".format(
        if (cacheLambda > 0) " (cache λ=$cacheLambda)" else "", ppAll, bits, ppIdent, 100.0 * stats.oov / stats.identPositions.coerceAtLeast(1)))
    log("next identifier (sampled %d positions, in-vocab): top1=%.3f top5=%.3f; %.1f s".format(stats.scored, stats.top1.toDouble() / stats.scored.coerceAtLeast(1), stats.top5.toDouble() / stats.scored.coerceAtLeast(1), (System.currentTimeMillis() - t0) / 1000.0))
}

/** Top-k over the vocabulary for the cache mixture: global scores via the fixed-context Scorer, cache probabilities on top. */
internal fun topKMixed(model: NgramModel, cache: CacheLm, lambda: Double, ctx: IntArray, pos: Int, k: Int, filter: (Int) -> Boolean): List<Pair<Int, Float>> {
    val scorer = model.scorer(ctx, pos)
    val best = ArrayList<Pair<Int, Float>>()
    for (w in 0 until model.vocab.size) {
        if (!filter(w)) continue
        val p = (kotlin.math.ln(lambda * cache.prob(ctx, pos, w) + (1 - lambda) * kotlin.math.exp(scorer.logProb(w).toDouble()))).toFloat()
        if (best.size < k) { best.add(w to p); if (best.size == k) best.sortByDescending { it.second } }
        else if (p > best.last().second) {
            best[best.size - 1] = w to p
            var i = best.size - 1
            while (i > 0 && best[i].second > best[i - 1].second) { val t = best[i]; best[i] = best[i - 1]; best[i - 1] = t; i-- }
        }
    }
    return best
}

private fun trainRanker(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    val trainEx = ArrayList<TrainingExample>(); val testEx = ArrayList<TrainingExample>()
    val schema: FeatureSchema
    val source: String
    if (args["shards"] != null) {
        // examples produced by a plugin-side generator (real PSI candidates and features)
        val (header, _) = ExampleShards.readAll(File(args.getValue("shards")), trainEx) { log("train $it") }
        require(header.language == lang.id) { "shards are for ${header.language}, not ${lang.id}" }
        schema = header.schema
        args["test-shards"]?.let { val (th, _) = ExampleShards.readAll(File(it), testEx) { m -> log("test $m") }; require(th.schema.hash == schema.hash) { "test shards have another schema" } }
        source = "shards=${args["shards"]}"
        log("${lang.id}: ${schema.size} features, language block ${schema.languageFeatures}")
    } else {
        val corpus = corpus(args, lang)
        val lm = NgramModel.read(File(args.getValue("lm")))
        val loaded = load(corpus, args, args["max-files"]?.toInt())
        val gen = ProxyExampleGenerator(lm.vocab, lm, maxExamplesPerFile = args["per-file"]?.toInt() ?: 60, cacheLambda = args["cache"]?.toDouble() ?: 0.0,
                                        baseCount = args["features"]?.toInt() ?: FeatureSchema.BASE.size)
        schema = gen.schema()
        log("${lang.id}: generating examples from ${loaded.train.size} train / ${loaded.test.size} test files, ${schema.size} features")
        forEachTokenised(corpus, loaded.train) { _, t -> gen.generate(t) { trainEx.add(it) } }
        forEachTokenised(corpus, loaded.test) { _, t -> gen.generate(t) { testEx.add(it) } }
        source = "proxy examples; lm=${args.getValue("lm")}"
        args["dump-shards"]?.let { dir ->
            File(dir).mkdirs()
            ExampleShards.Writer(File(dir, "train.cmlx"), lang.id, source, schema).use { w -> trainEx.forEach(w::add) }
            ExampleShards.Writer(File(dir, "test.cmlx"), lang.id, source, schema, withNames = true).use { w -> testEx.forEach(w::add) }
            log("shards written to $dir")
        }
    }
    log("examples: train ${trainEx.size}, test ${testEx.size}, avg candidates %.1f".format(trainEx.sumOf { it.size }.toDouble() / trainEx.size.coerceAtLeast(1)))

    val trainer = LinearRankerTrainer(schema, l2 = args["l2"]?.toDouble() ?: 1e-4, learningRate = args["lr"]?.toDouble() ?: 0.1, epochs = args["epochs"]?.toInt() ?: 10)
    val (ranker, _) = trainer.train(trainEx) { log(it) }
    val out = File(args.getValue("out"))
    ranker.write(out, lang.id, source)
    log("written $out")

    // baselines on the same test lists: single features
    fun single(name: String) = schema.baseIndex(name)
    val baselines = linkedMapOf(
        "ranker" to { ex: TrainingExample -> ranker.scores(ex) },
        "baseline: alphabetical (ties -> last)" to { ex: TrainingExample -> FloatArray(ex.size) { 0f } },
        "baseline: n-gram log-prob only" to { ex: TrainingExample -> FloatArray(ex.size) { ex.baseFeature(it, single("lm_logprob")) } },
        "baseline: most frequent in file" to { ex: TrainingExample -> FloatArray(ex.size) { ex.baseFeature(it, single("file_freq_log")) } },
        "baseline: most recent in file" to { ex: TrainingExample -> FloatArray(ex.size) { -ex.baseFeature(it, single("recency_log")) } },
    )
    // plugin adapters record their own deterministic order as `rule_rank_log`: the "rules only" baseline the ML has to beat
    if (schema.baseIndex("rule_rank_log") >= 0) baselines["baseline: plugin rules (rule_rank_log)"] = { ex: TrainingExample -> FloatArray(ex.size) { -ex.baseFeature(it, single("rule_rank_log")) } }
    for ((title, scorer) in baselines) {
        if (testEx.isEmpty()) break
        val m = RankMetrics()
        for (ex in testEx) m.add(ex, scorer(ex))
        System.err.print(m.report(title))
    }
    val t0 = System.nanoTime(); var n = 0
    for (ex in testEx) { ranker.scores(ex); n += ex.size }
    log("inference: %.2f µs per candidate".format((System.nanoTime() - t0) / 1000.0 / n.coerceAtLeast(1)))
    System.err.println("weights (standardised features):")
    ranker.schema.names.zip(ranker.weights.toList()).filter { kotlin.math.abs(it.second) > 0.05 }.sortedByDescending { kotlin.math.abs(it.second) }.take(20)
        .forEach { System.err.println("  %-28s %+.3f".format(it.first, it.second)) }
}

private fun evalRanker(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    val ranker = LinearRanker.read(File(args.getValue("rank")))
    val m = RankMetrics(); val base = RankMetrics()
    val lmIdx = ranker.schema.baseIndex("lm_logprob"); val ruleIdx = ranker.schema.baseIndex("rule_rank_log")
    val rules = RankMetrics()
    var n = 0
    val add = { ex: TrainingExample ->
        n++; m.add(ex, ranker.scores(ex)); base.add(ex, FloatArray(ex.size) { ex.baseFeature(it, lmIdx) })
        if (ruleIdx >= 0) rules.add(ex, FloatArray(ex.size) { -ex.baseFeature(it, ruleIdx) })
    }
    if (args["shards"] != null) {
        val (header, _) = ExampleShards.readAll(File(args.getValue("shards")), object : AbstractMutableList<TrainingExample>() {
            override val size get() = 0
            override fun get(index: Int) = throw IndexOutOfBoundsException()
            override fun add(index: Int, element: TrainingExample) { add(element) }
            override fun removeAt(index: Int) = throw UnsupportedOperationException()
            override fun set(index: Int, element: TrainingExample) = throw UnsupportedOperationException()
        })
        require(header.schema.hash == ranker.schema.hash) { "ranker schema differs from the shards'" }
        log("${lang.id}: $n lists from ${args.getValue("shards")}; ranker ${args.getValue("rank")}")
    } else {
        val corpus = corpus(args, lang)
        val lm = NgramModel.read(File(args.getValue("lm")))
        val test = load(corpus, args, args["max-files"]?.toInt()).test
        val gen = ProxyExampleGenerator(lm.vocab, lm, cacheLambda = args["cache"]?.toDouble() ?: 0.0, baseCount = ranker.schema.baseSize)
        require(gen.schema().hash == ranker.schema.hash) { "ranker schema differs from the generator's" }
        forEachTokenised(corpus, test) { _, t -> gen.generate(t, add) }
        log("${lang.id}: ${test.size} test files, $n lists; ranker ${args.getValue("rank")} with LM ${args.getValue("lm")}")
    }
    System.err.print(m.report("ranker"))
    System.err.print(base.report("baseline: n-gram log-prob only"))
    if (ruleIdx >= 0) System.err.print(rules.report("baseline: plugin rules (rule_rank_log)"))
}

private fun dumpTokens(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    for (t in lang.tokenizer.tokens(File(args.getValue("file")).readText())) println("${t.kind}\t${t.text}")
}
