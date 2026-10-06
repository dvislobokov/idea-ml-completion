package io.github.completionml.train

import io.github.completionml.core.lex.Languages
import io.github.completionml.core.ngram.NgramModel
import io.github.completionml.core.ngram.PartitionedNgramTrainer
import io.github.completionml.core.spi.MlLanguage
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.vocab.Vocabulary
import java.io.File

/** One evaluation file in the ids of a given [Vocabulary]; [lineStart] marks tokens that begin a line, [atLineEnd] the lexer's automatic `;` at a line break. */
internal class EvalFile(val ids: IntArray, val ident: BooleanArray, val lineStart: BooleanArray, val atLineEnd: BooleanArray) {
    /** Number of tokens from [i] to the end of its line (an automatic `;` at the line break is not counted). */
    fun restOfLine(i: Int): Int {
        var rest = 0; var j = i
        while (j < ids.size && (j == i || !lineStart[j])) { if (!atLineEnd[j]) rest++; j++ }
        return rest
    }
}

/** Held-out files, read lazily so that several can be processed in parallel. */
internal abstract class TestSet {
    abstract val size: Int
    abstract fun file(i: Int, vocab: Vocabulary): EvalFile
    fun indices(): List<Int> = (0 until size).toList()
}

internal class CorpusTestSet(private val corpus: Corpus, private val sources: List<Corpus.Source>) : TestSet() {
    override val size get() = sources.size
    override fun file(i: Int, vocab: Vocabulary): EvalFile {
        val text = sources[i].file.readText()
        val tokens = corpus.language.tokenizer.tokens(text)
        val ident = BooleanArray(tokens.size) { tokens[it].kind == TokenKind.IDENT }
        val nl = BooleanArray(tokens.size) { k -> k > 0 && (tokens[k - 1].offset until tokens[k].offset).any { text[it] == '\n' } }
        val end = BooleanArray(tokens.size) { tokens[it].offset >= text.length || text[tokens[it].offset] == '\n' }
        return EvalFile(vocab.encode(tokens), ident, nl, end)
    }
}

internal class ShardTestSet(private val shards: TokenShards, private val files: List<TokenFile>) : TestSet() {
    override val size get() = files.size
    private var mapFor: Vocabulary? = null
    private var map = IntArray(0)
    @Synchronized private fun mapping(vocab: Vocabulary): IntArray {
        if (mapFor !== vocab) { map = shardToLm(shards.vocab, vocab); mapFor = vocab }
        return map
    }
    override fun file(i: Int, vocab: Vocabulary): EvalFile {
        val f = files[i]; val m = mapping(vocab)
        return EvalFile(IntArray(f.size) { m[f.id(it)] }, BooleanArray(f.size) { shards.vocab.isIdent(f.id(it)) }, BooleanArray(f.size) { f.lineStart(it) }, BooleanArray(f.size) { f.atLineEnd(it) })
    }
}

/** Shard id -> id in the LM vocabulary (`<ID>` for identifiers it does not know). */
internal fun shardToLm(sv: ShardVocab, vocab: Vocabulary): IntArray = IntArray(sv.size) { if (it == ShardFormat.RARE_ID) Vocabulary.UNK_ID else vocab.id(sv.texts[it]) }

/** Test files from `--token-shards` (fold `--test-fold`, default `test`, thinned to `--max-test-files`) or from the repository tree. */
internal fun testSet(args: Map<String, String>, lang: MlLanguage): TestSet {
    args["token-shards"]?.let { dir ->
        val shards = TokenShards(File(dir)); val fold = args["test-fold"] ?: "test"
        val max = args["max-test-files"]?.toInt()
        val stride = if (max == null) 1 else ((shards.files(fold) + max - 1) / max).toInt().coerceAtLeast(1)
        val files = shards.readAll(fold, stride)
        log("test fold '$fold' of $dir: ${files.size} files, ${files.sumOf { it.size.toLong() }} tokens (every ${stride}th file)")
        return ShardTestSet(shards, files)
    }
    val corpus = corpus(args, lang)
    return CorpusTestSet(corpus, load(corpus, args).test)
}

/**
 * `l2 --token-shards <dir>`: trains the LM on the files of fold `--train-fold` (default lm) with [PartitionedNgramTrainer],
 * then evaluates on the test fold. Vocabulary: identifiers by document frequency over the training fold (>= 2 files), as in the tree path.
 */
internal fun trainLmFromShards(args: Map<String, String>, lang: MlLanguage) {
    val shards = TokenShards(File(args.getValue("token-shards")))
    require(shards.language == lang.id) { "shards are for ${shards.language}" }
    val fold = args["train-fold"] ?: "lm"
    val order = args["order"]?.toInt() ?: 4
    val maxVocab = args["vocab"]?.toInt() ?: 50_000
    fun perOrder(key: String) = (args[key] ?: "1").split(",").map { it.trim().toInt() }.let { l -> IntArray(order) { l.getOrElse(it) { l.last() } } }
    val minCounts = perOrder("min-count"); val minRepos = perOrder("min-repos")
    val threads = args["threads"]?.toInt() ?: (Runtime.getRuntime().availableProcessors() - 2).coerceAtLeast(1)
    val partitions = args["partitions"]?.toInt() ?: 64
    val sv = shards.vocab
    log("order $order, fold $fold: ${shards.files(fold)} files, ${shards.tokens(fold)} tokens, shard vocabulary ${sv.size}")

    // pass 1: document and term frequencies of every shard id over the training fold, in parallel over the parts
    val df = IntArray(sv.size); val tf = LongArray(sv.size)
    var files = 0L
    shards.parts(fold).parallelStream().forEach { part ->
        val ldf = IntArray(sv.size); val ltf = LongArray(sv.size); val stamp = IntArray(sv.size)
        var n = 0
        PartReader(part).use { r ->
            while (true) {
                val f = r.next() ?: break
                n++
                for (i in 0 until f.size) { val id = f.id(i); ltf[id]++; if (stamp[id] != n) { stamp[id] = n; ldf[id]++ } }
            }
        }
        synchronized(df) { for (i in df.indices) { df[i] += ldf[i]; tf[i] += ltf[i] }; files += n }
    }
    val minDocFreq = if (shards.repoFolds.count { it == fold } >= 20) 2 else 1
    val vb = Vocabulary.Builder()
    for (id in 1 until sv.size) if (tf[id] > 0) vb.addCounts(sv.texts[id], sv.isIdent(id), df[id], tf[id])
    vb.addTotals(files.toInt(), tf.sum())
    val vocab = vb.build(maxVocab, minDocFreq)
    log("vocabulary: ${vocab.size} entries from ${tf.sum()} tokens in $files files")

    val toLm = shardToLm(sv, vocab)
    val weights = LongArray(vocab.size)
    for (id in 0 until sv.size) weights[toLm[id]] += tf[id]
    weights[Vocabulary.BOS_ID] += files * order; weights[Vocabulary.EOS_ID] += files
    val trainer = PartitionedNgramTrainer(order, vocab, weights, partitions, threads, minCounts, minRepos)
    log("counting in $partitions partitions on $threads threads")
    val smoothing = (args["smoothing"] ?: "mkn").lowercase()
    require(smoothing == "mkn") { "partitioned training supports only --smoothing mkn" }
    val model = trainer.train({ consumer ->
        var buf = IntArray(1 shl 12)
        shards.forEachFile(fold) { f ->
            if (buf.size < f.size) buf = IntArray(f.size + f.size / 2)
            for (i in 0 until f.size) buf[i] = toLm[f.raw[i] ushr 2]
            consumer(buf, f.size, f.repo)
        }
    }, ::log)
    val bits = args["fp-bits"]?.toInt() ?: 24
    val out = File(args.getValue("out"))
    model.write(out, lang.id, "shards=${shards.dir};fold=$fold", bits)
    log("written $out (${out.length() / 1024} KB, $bits-bit fingerprints, 8-bit values)")
    if (args["no-eval"] != "true") evaluateLm(NgramModel.read(out), testSet(args, lang), args["cache"]?.toDouble() ?: 0.0)
}
