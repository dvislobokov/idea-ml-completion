package io.github.completionml.train

import io.github.completionml.core.lex.Languages
import io.github.completionml.core.spi.MlLanguage
import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import java.io.File
import java.util.Properties
import java.util.concurrent.Callable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/** Repository with its `ok` files (relative paths) from the manifest. */
class ShardRepo(val name: String, val fold: String, val paths: List<String>, val bytes: Long)

/** The `ok` files of a manifest grouped by repository, in manifest order (lm, rank, test; repositories contiguous). */
fun readManifestRepos(manifest: File): List<ShardRepo> {
    val repos = ArrayList<ShardRepo>()
    var cur: String? = null; var fold = ""; var paths = ArrayList<String>(); var bytes = 0L
    fun flush() { val c = cur; if (c != null && paths.isNotEmpty()) repos.add(ShardRepo(c, fold, paths, bytes)); paths = ArrayList(); bytes = 0 }
    manifest.useLines { lines ->
        for (l in lines) {
            if (l.isBlank()) continue
            if (ManifestLine.string(l, "status") != "ok") continue
            val repo = ManifestLine.string(l, "repo") ?: continue
            if (repo != cur) { flush(); cur = repo; fold = ManifestLine.string(l, "fold") ?: "lm" }
            paths.add(ManifestLine.string(l, "path")!!); bytes += ManifestLine.long(l, "bytes") ?: 0
        }
    }
    flush()
    return repos
}

class ShardOptions(
    val language: MlLanguage, val data: File, val manifest: File = File(data, "prepared/manifest.jsonl"), val out: File = File(data, "shards"),
    /** identifiers seen fewer times in the whole corpus share the id `<ID>` */
    val minCount: Long = 2, val threads: Int = Runtime.getRuntime().availableProcessors(),
    /** target source bytes per part file */
    val partBytes: Long = 96L shl 20,
)

private class Cnt(@JvmField var n: Long, @JvmField var kind: Int)

/** True when a line break lies in `text[from, to)`. */
private fun hasNewline(text: String, from: Int, to: Int): Boolean { for (i in from until to) if (text[i] == '\n') return true; return false }

/** Raw `id << 2 | atLineEnd << 1 | lineStart` values for the tokens of [text]; [index] maps token text to the shard id (missing identifiers read as `<ID>`). */
internal fun encodeFile(tokens: List<MlToken>, text: String, index: Map<String, Int>): IntArray {
    val raw = IntArray(tokens.size)
    for (i in tokens.indices) {
        val t = tokens[i]
        val id = index[t.text] ?: ShardFormat.RARE_ID
        val nl = i > 0 && hasNewline(text, tokens[i - 1].offset, t.offset)
        val end = t.offset >= text.length || text[t.offset] == '\n'
        raw[i] = (id shl 2) or (if (end) 2 else 0) or (if (nl) 1 else 0)
    }
    return raw
}

fun buildShards(opt: ShardOptions) {
    val t0 = System.currentTimeMillis()
    val lang = opt.language
    val repos = readManifestRepos(opt.manifest)
    log("shard ${lang.id}: ${repos.size} repositories, ${repos.sumOf { it.paths.size }} files, %.1f GB in the manifest".format(repos.sumOf { it.bytes } / 1e9))
    opt.out.mkdirs()
    // tasks: consecutive repositories of one fold up to ~partBytes of source; the part number is the task's position within its fold
    class Task(val fold: String, val part: Int, val first: Int, val last: Int)   // repos[first until last]
    val tasks = ArrayList<Task>()
    run {
        var start = 0; var acc = 0L; val partNo = HashMap<String, Int>()
        for (i in repos.indices) {
            acc += repos[i].bytes
            val endOfGroup = i == repos.size - 1 || repos[i + 1].fold != repos[i].fold
            if (acc >= opt.partBytes || endOfGroup) {
                val n = partNo.merge(repos[i].fold, 1, Int::plus)!! - 1
                tasks.add(Task(repos[i].fold, n, start, i + 1)); start = i + 1; acc = 0
            }
        }
    }
    val pool = Executors.newFixedThreadPool(opt.threads) { r -> Thread(r, "shard").also { it.isDaemon = true } }
    fun readText(repo: ShardRepo, path: String): String? = try {
        NoAtimeReader.readAllBytes(File(opt.data, "repos/${repo.name}/$path")).let { PrepareFilters.decodeText(it) ?: String(it, Charsets.UTF_8) }
    } catch (e: java.io.IOException) { null }

    // pass A: token counts and kinds
    val global = ConcurrentHashMap<String, Cnt>(1 shl 20)
    val kindConflicts = AtomicLong()
    val done = AtomicInteger(); val tokensA = AtomicLong()
    pool.invokeAll(tasks.map { task -> Callable {
        val local = HashMap<String, Cnt>(1 shl 18)
        for (r in task.first until task.last) for (p in repos[r].paths) {
            val text = readText(repos[r], p) ?: continue
            val toks = lang.tokenizer.tokens(text)
            tokensA.addAndGet(toks.size.toLong())
            for (t in toks) { val c = local.getOrPut(t.text) { Cnt(0, t.kind.ordinal) }; c.n++; if (c.kind != t.kind.ordinal && c.kind != TokenKind.IDENT.ordinal && t.kind == TokenKind.IDENT) c.kind = t.kind.ordinal }
        }
        for ((k, v) in local) global.merge(k, v) { a, b ->
            if (a.kind != b.kind) { kindConflicts.incrementAndGet(); if (b.kind == TokenKind.IDENT.ordinal) a.kind = b.kind }
            a.n += b.n; a
        }
        val d = done.incrementAndGet()
        if (d % 20 == 0 || d == tasks.size) log("pass A: $d/${tasks.size} parts, ${tokensA.get() / 1_000_000} M tokens, ${global.size} distinct texts")
        null
    } }).forEach { it.get() }
    log("pass A done: ${global.size} distinct token texts, ${tokensA.get()} tokens, $kindConflicts kind conflicts merged to IDENT")

    // vocabulary: id 0 = <ID>; the rest by descending count (identifiers below minCount are folded into id 0)
    val identOrd = TokenKind.IDENT.ordinal
    var rareCount = 0L; var rareDistinct = 0L
    val kept = ArrayList<Map.Entry<String, Cnt>>(global.size)
    for (e in global.entries) {
        if (e.value.kind == identOrd && e.value.n < opt.minCount) { rareCount += e.value.n; rareDistinct++ } else kept.add(e)
    }
    kept.sortWith(compareByDescending<Map.Entry<String, Cnt>> { it.value.n }.thenBy { it.key })
    val size = kept.size + 1
    val texts = Array(size) { if (it == 0) "<ID>" else kept[it - 1].key }
    val kinds = ByteArray(size) { if (it == 0) identOrd.toByte() else kept[it - 1].value.kind.toByte() }
    val counts = LongArray(size) { if (it == 0) rareCount else kept[it - 1].value.n }
    val vocab = ShardVocab(texts, kinds, counts)
    val width = ShardFormat.widthFor(size)
    log("vocabulary: $size ids (${kept.count { it.value.kind == identOrd }} identifiers, ${size - 1 - kept.count { it.value.kind == identOrd }} other); $rareDistinct rare identifiers ($rareCount tokens, %.2f%%) -> <ID>; width $width bytes".format(100.0 * rareCount / tokensA.get().coerceAtLeast(1)))
    vocab.write(File(opt.out, "vocab.bin"))
    val index = HashMap<String, Int>(size * 2)
    for (i in 1 until size) index[texts[i]] = i
    global.clear()

    // pass B: encode and write
    File(opt.out, "repos.tsv").bufferedWriter().use { w -> repos.forEachIndexed { i, r -> w.write("$i\t${r.fold}\t${r.name}\n") } }
    val filesByFold = ConcurrentHashMap<String, AtomicLong>(); val tokensByFold = ConcurrentHashMap<String, AtomicLong>()
    val bytesOut = AtomicLong(); done.set(0); val unreadable = AtomicLong(); val tokensB = AtomicLong()
    pool.invokeAll(tasks.map { task -> Callable {
        val file = File(opt.out, "%s-%04d.tok".format(task.fold, task.part))
        PartWriter(file, width).use { w ->
            for (r in task.first until task.last) for (p in repos[r].paths) {
                val text = readText(repos[r], p) ?: run { unreadable.incrementAndGet(); null } ?: continue
                val toks = lang.tokenizer.tokens(text)
                val raw = encodeFile(toks, text, index)
                w.add(r, raw)
                filesByFold.computeIfAbsent(task.fold) { AtomicLong() }.incrementAndGet(); tokensByFold.computeIfAbsent(task.fold) { AtomicLong() }.addAndGet(raw.size.toLong())
                tokensB.addAndGet(raw.size.toLong())
            }
        }
        bytesOut.addAndGet(file.length())
        val d = done.incrementAndGet()
        if (d % 20 == 0 || d == tasks.size) log("pass B: $d/${tasks.size} parts, ${tokensB.get() / 1_000_000} M tokens, %.1f GB written".format(bytesOut.get() / 1e9))
        null
    } }).forEach { it.get() }
    pool.shutdown()

    val p = Properties()
    p["language"] = lang.id; p["width"] = width.toString(); p["min_count"] = opt.minCount.toString(); p["vocab"] = size.toString()
    p["source_manifest"] = opt.manifest.path
    for (f in filesByFold.keys) { p["files.$f"] = filesByFold.getValue(f).get().toString(); p["tokens.$f"] = tokensByFold.getValue(f).get().toString() }
    File(opt.out, "shards.properties").outputStream().use { p.store(it, "token shards, see ShardFormat") }
    val sec = (System.currentTimeMillis() - t0) / 1000.0
    log("sharded %,d tokens (%d unreadable files) into %.2f GB = %.2f bytes/token in %.0f s -> ${opt.out}".format(tokensB.get(), unreadable.get(), bytesOut.get() / 1e9, bytesOut.get().toDouble() / tokensB.get().coerceAtLeast(1), sec))
    for (f in filesByFold.keys.sorted()) log("fold $f: ${filesByFold.getValue(f)} files, ${tokensByFold.getValue(f)} tokens")
}

internal fun shardCommand(args: Map<String, String>) {
    val lang = Languages.byId(args["lang"] ?: "go")
    val data = File(args.getValue("data"))
    buildShards(ShardOptions(
        lang, data,
        manifest = args["manifest"]?.let { File(it) } ?: File(data, "prepared/manifest.jsonl"),
        out = args["out"]?.let { File(it) } ?: File(data, "shards"),
        minCount = args["min-count"]?.toLong() ?: 2,
        threads = args["threads"]?.toInt() ?: Runtime.getRuntime().availableProcessors(),
    ))
}
