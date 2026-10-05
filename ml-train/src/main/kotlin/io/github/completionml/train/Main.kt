package io.github.completionml.train

import io.github.completionml.core.eval.RankMetrics
import io.github.completionml.core.lex.Languages
import io.github.completionml.core.ngram.CacheLm
import io.github.completionml.core.ngram.MixedLm
import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.ngram.TokenLm
import io.github.completionml.core.ngram.NgramTrainer
import io.github.completionml.core.ngram.PerplexityAccumulator
import io.github.completionml.core.rank.FeatureSchema
import io.github.completionml.core.rank.LinearRanker
import io.github.completionml.core.rank.LinearRankerTrainer
import io.github.completionml.core.rank.ProxyExampleGenerator
import io.github.completionml.core.rank.TrainingExample
import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.vocab.Vocabulary
import java.io.File
import kotlin.system.exitProcess

private const val USAGE = """
usage:
  l2  --lang csharp|go --data <dir> --out <lm.cml> [--order 4] [--vocab 50000] [--min-count 1,1,2,2] [--split auto|repo|file] [--fp-bits 24] [--eval-exact true] [--min-repos 1,1,1,2] [--cache 0.3] [--smoothing mkn|jm] [--lambda 0.5]
      count n-grams on the train split, estimate modified Kneser-Ney, evaluate on the test split, write the model
  l1  --lang csharp|go --data <dir> --lm <lm.cml> --out <rank.cml> [--epochs 10] [--l2 1e-4] [--lr 0.1] [--max-files N] [--per-file 60] [--cache 0.3] [--features 8]
      NOTE: train the LM on repositories disjoint from the ranker's training repositories (--repos lists), otherwise the
      LM feature is inflated on the ranker's training data and the ranker over-trusts it on unseen projects.
      generate proxy ranking examples, train the listwise logistic regression, report metrics vs. baselines
  eval-lm --lang .. --data <dir> --lm <lm.cml>      re-evaluate an n-gram model on the test split
  common: --repos <file>   restrict the corpus to the repo directory names listed in the file (one per line)
          --dedup 0.8      drop exact duplicates and near-duplicates (MinHash Jaccard over identifiers >= value), train first then test
          --split repo|file  force the split (auto: by repo when >= 20 repos, else by file)
          --test-repos a__b,c__d   hold out exactly these repositories (directory names)
  eval-rank --lang .. --data <dir> --lm <lm.cml> --rank <rank.cml> [--max-files N]
      re-evaluate a ranker (with the LM it was trained with) on the test split of another corpus
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

private fun log(s: String) = System.err.println("[%tT] %s".format(System.currentTimeMillis(), s))

private fun corpus(args: Map<String, String>, lang: io.github.completionml.core.spi.MlLanguage) =
    Corpus(File(args.getValue("data")), lang, args["split"] ?: "auto", includeList = args["repos"]?.let { File(it) },
           testRepos = args["test-repos"]?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }?.toSet() ?: emptySet())

private fun readTokens(c: Corpus, s: Corpus.Source): List<MlToken> = c.language.tokenizer.tokens(s.file.readText())

/** Tokenised train/test files; with `--dedup <jaccard>` exact and near duplicates are dropped (train first, then test). */
class Loaded(val train: List<Pair<Corpus.Source, List<MlToken>>>, val test: List<Pair<Corpus.Source, List<MlToken>>>)

private fun load(corpus: Corpus, args: Map<String, String>, maxFiles: Int? = null): Loaded {
    val files = corpus.files().let { f -> maxFiles?.let { f.take(it) } ?: f }
    val (testSrc, trainSrc) = files.partition { corpus.isTest(it) }
    val dedup = args["dedup"]?.toDouble()?.let { Dedup(it) }
    fun take(list: List<Corpus.Source>) = list.mapNotNull { s -> val t = readTokens(corpus, s); if (dedup == null || dedup.offer(t)) s to t else null }
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
    val train = loaded.train.map { it.first }; val test = loaded.test
    log("order $order")

    // pass 1: vocabulary
    val vb = Vocabulary.Builder()
    val tokenised = ArrayList<List<MlToken>>(train.size)
    for ((_, t) in loaded.train) { tokenised.add(t); vb.addFile(t) }
    val vocab = vb.build(maxVocab, minDocFreq = if (corpus.repos.size >= 20) 2 else 1)
    log("vocabulary: ${vocab.size} entries from ${vb.tokens} tokens in ${vb.files} files")

    // pass 2: counts
    val trainer = NgramTrainer(order, vocab, expectedTokens = vb.tokens.toInt().coerceAtLeast(1 shl 16), trackRepos = minRepos.any { it > 1 })
    val repoIndex = corpus.repos.withIndex().associate { it.value.name to it.index }
    for ((i, t) in tokenised.withIndex()) trainer.addFile(vocab.encode(t), repoIndex.getValue(train[i].repo))
    tokenised.clear()
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

/** Perplexity over all tokens; next-token top-1/top-5 over identifiers that are in the vocabulary (what gray text could predict). */
private fun evaluateLm(model: NgramModel, corpus: Corpus, test: List<Pair<Corpus.Source, List<MlToken>>>, cacheLambda: Double = 0.0) {
    val ppAll = PerplexityAccumulator(); val ppIdent = PerplexityAccumulator()
    var identPositions = 0L; var top1 = 0L; var top5 = 0L; var oov = 0L; var scored = 0L
    val sampleEvery = 20   // top-k over the whole vocabulary is O(|V|): sample positions
    val t0 = System.currentTimeMillis()
    val vocab = model.vocab
    for ((_, tokens) in test) {
        val ids = vocab.encode(tokens)
        val padded = IntArray(ids.size + model.order - 1) { if (it < model.order - 1) Vocabulary.BOS_ID else ids[it - model.order + 1] }
        val cache = if (cacheLambda > 0) CacheLm(3, vocab.size) else null
        val lm: TokenLm = if (cache != null) MixedLm(model, cache, cacheLambda) else model
        for (i in ids.indices) {
            val pos = i + model.order - 1
            val lp = lm.logProb(padded, pos, ids[i])
            ppAll.add(lp)
            if (tokens[i].kind == TokenKind.IDENT) {
                identPositions++
                if (ids[i] == Vocabulary.UNK_ID) { oov++; cache?.add(ids[i]); continue }
                ppIdent.add(lp)
                if (identPositions % sampleEvery == 0L) {
                    scored++
                    val top = if (cache == null) model.topK(padded, pos, 5) { vocab.isIdentifier(it) && it != Vocabulary.UNK_ID }
                              else topKMixed(model, cache, cacheLambda, padded, pos, 5) { vocab.isIdentifier(it) && it != Vocabulary.UNK_ID }
                    if (top.isNotEmpty() && top[0].first == ids[i]) top1++
                    if (top.any { it.first == ids[i] }) top5++
                }
            }
            cache?.add(ids[i])
        }
    }
    log("LM eval on ${test.size} files%s: perplexity all=%.1f (%.2f bits), identifiers in vocab=%.1f; identifier OOV rate=%.1f%%".format(
        if (cacheLambda > 0) " (cache λ=$cacheLambda)" else "", ppAll.perplexity, ppAll.entropyBits, ppIdent.perplexity, 100.0 * oov / identPositions.coerceAtLeast(1)))
    log("next identifier (sampled %d positions, in-vocab): top1=%.3f top5=%.3f; %.1f s".format(scored, top1.toDouble() / scored.coerceAtLeast(1), top5.toDouble() / scored.coerceAtLeast(1), (System.currentTimeMillis() - t0) / 1000.0))
}

/** Top-k over the vocabulary for the cache mixture: global scores via the fixed-context Scorer, cache probabilities on top. */
private fun topKMixed(model: NgramModel, cache: CacheLm, lambda: Double, ctx: IntArray, pos: Int, k: Int, filter: (Int) -> Boolean): List<Pair<Int, Float>> {
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
    val corpus = corpus(args, lang)
    val lm = NgramModel.read(File(args.getValue("lm")))
    val loaded = load(corpus, args, args["max-files"]?.toInt())
    val gen = ProxyExampleGenerator(lm.vocab, lm, maxExamplesPerFile = args["per-file"]?.toInt() ?: 60, cacheLambda = args["cache"]?.toDouble() ?: 0.0,
                                    baseCount = args["features"]?.toInt() ?: FeatureSchema.BASE.size)
    val schema = gen.schema()
    log("${lang.id}: generating examples from ${loaded.train.size} train / ${loaded.test.size} test files, ${schema.size} features")
    val trainEx = ArrayList<TrainingExample>(); val testEx = ArrayList<TrainingExample>()
    for ((_, t) in loaded.train) gen.generate(t) { trainEx.add(it) }
    for ((_, t) in loaded.test) gen.generate(t) { testEx.add(it) }
    log("examples: train ${trainEx.size}, test ${testEx.size}, avg candidates %.1f".format(trainEx.sumOf { it.size }.toDouble() / trainEx.size.coerceAtLeast(1)))

    val trainer = LinearRankerTrainer(schema, l2 = args["l2"]?.toDouble() ?: 1e-4, learningRate = args["lr"]?.toDouble() ?: 0.1, epochs = args["epochs"]?.toInt() ?: 10)
    val (ranker, _) = trainer.train(trainEx) { log(it) }
    val out = File(args.getValue("out"))
    ranker.write(out, lang.id, "proxy examples; lm=${args.getValue("lm")}")
    log("written $out")

    // baselines on the same test lists: single features
    fun single(name: String) = FeatureSchema.BASE.indexOf(name)
    val baselines = linkedMapOf(
        "ranker" to { ex: TrainingExample -> ranker.scores(ex) },
        "baseline: alphabetical (ties -> last)" to { ex: TrainingExample -> FloatArray(ex.size) { 0f } },
        "baseline: n-gram log-prob only" to { ex: TrainingExample -> FloatArray(ex.size) { ex.baseFeature(it, single("lm_logprob")) } },
        "baseline: most frequent in file" to { ex: TrainingExample -> FloatArray(ex.size) { ex.baseFeature(it, single("file_freq_log")) } },
        "baseline: most recent in file" to { ex: TrainingExample -> FloatArray(ex.size) { -ex.baseFeature(it, single("recency_log")) } },
    )
    for ((title, scorer) in baselines) {
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
    val corpus = corpus(args, lang)
    val lm = NgramModel.read(File(args.getValue("lm")))
    val ranker = LinearRanker.read(File(args.getValue("rank")))
    val test = load(corpus, args, args["max-files"]?.toInt()).test
    val gen = ProxyExampleGenerator(lm.vocab, lm, cacheLambda = args["cache"]?.toDouble() ?: 0.0, baseCount = ranker.schema.size / (1 + io.github.completionml.core.spi.ContextKind.values().size))
    require(gen.schema().hash == ranker.schema.hash) { "ranker schema differs from the generator's" }
    val m = RankMetrics(); val base = RankMetrics()
    val lmIdx = FeatureSchema.BASE.indexOf("lm_logprob")
    var n = 0
    for ((_, t) in test) gen.generate(t) { ex ->
        n++
        m.add(ex, ranker.scores(ex))
        base.add(ex, FloatArray(ex.size) { ex.baseFeature(it, lmIdx) })
    }
    log("${lang.id}: ${test.size} test files, $n lists; ranker ${args.getValue("rank")} with LM ${args.getValue("lm")}")
    System.err.print(m.report("ranker"))
    System.err.print(base.report("baseline: n-gram log-prob only"))
}

private fun dumpTokens(args: Map<String, String>) {
    val lang = Languages.byId(args.getValue("lang"))
    for (t in lang.tokenizer.tokens(File(args.getValue("file")).readText())) println("${t.kind}\t${t.text}")
}
