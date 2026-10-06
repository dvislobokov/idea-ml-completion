package io.github.completionml.train

import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.Properties

/**
 * Token shards: the corpus lexed once, stored as fixed-width ids so that LM training, ranker example generation and evaluation
 * never touch the source tree again.
 *
 * Directory layout (`<data>/shards`):
 *  - `shards.properties`  language, id width, rare-identifier threshold, per-fold counts
 *  - `vocab.bin`          every token text with its kind and corpus count; id 0 is `<ID>` (identifier below the threshold)
 *  - `repos.tsv`          `index <TAB> fold <TAB> name`; the repository index is what the part records refer to
 *  - `<fold>-NNNN.tok`    parts; whole repositories only, in order (repository counts of the LM trainer need contiguous repos)
 *
 * Part file (little endian): header 32 bytes (magic, version, width, 0, files:i64, tokens:i64), then per file
 * `repo:i32 count:i32` followed by `count` values of `width` bytes. A value is `id << 2 | atLineEnd << 1 | lineStart`: `lineStart` is set when a
 * line break lies between the start of the previous token and the start of this one, `atLineEnd` when the token sits at a line break
 * or at the end of the text (the lexer's automatic `;`); the inline evaluation needs both to find line starts and the rest of a line.
 * Ids are assigned by descending corpus frequency, so 2 bytes hold vocabularies up to 16 384, 3 bytes up to 2 097 152, else 4 bytes.
 */
object ShardFormat {
    const val MAGIC = 0x434d4c53       // "CMLS"
    const val VERSION = 1
    const val HEADER_BYTES = 32
    const val RARE_ID = 0

    fun widthFor(vocabSize: Int): Int = when {
        vocabSize <= 1 shl 14 -> 2
        vocabSize <= 1 shl 21 -> 3
        else -> 4
    }
}

/** Token texts of a shard set; [kinds] holds [TokenKind] ordinals. */
class ShardVocab(val texts: Array<String>, val kinds: ByteArray, val counts: LongArray) {
    val size get() = texts.size
    fun isIdent(id: Int) = kinds[id].toInt() == TokenKind.IDENT.ordinal
    fun kind(id: Int) = KINDS[kinds[id].toInt()]
    fun token(id: Int, offset: Int) = MlToken(kind(id), texts[id], offset)

    fun write(f: File) {
        DataOutputStream(BufferedOutputStream(FileOutputStream(f), 1 shl 20)).use { o ->
            o.writeInt(ShardFormat.MAGIC); o.writeInt(ShardFormat.VERSION); o.writeInt(texts.size)
            for (i in texts.indices) {
                val b = texts[i].toByteArray(Charsets.UTF_8)
                o.writeByte(kinds[i].toInt()); o.writeLong(counts[i]); o.writeInt(b.size); o.write(b)
            }
        }
    }

    companion object {
        val KINDS = TokenKind.entries.toTypedArray()
        fun read(f: File): ShardVocab = DataInputStream(BufferedInputStream(FileInputStream(f), 1 shl 20)).use { i ->
            require(i.readInt() == ShardFormat.MAGIC) { "$f is not a shard vocabulary" }
            require(i.readInt() == ShardFormat.VERSION) { "unsupported shard vocabulary version in $f" }
            val n = i.readInt()
            val kinds = ByteArray(n); val counts = LongArray(n)
            val texts = Array(n) { k -> kinds[k] = i.readByte(); counts[k] = i.readLong(); val b = ByteArray(i.readInt()); i.readFully(b); String(b, Charsets.UTF_8) }
            ShardVocab(texts, kinds, counts)
        }
    }
}

/** One file of a shard part: [raw] holds `id << 2 | atLineEnd << 1 | lineStart` for the first [size] entries (the array may be longer and is reused by streaming readers). */
class TokenFile(val repo: Int, val size: Int, val raw: IntArray) {
    fun id(i: Int) = raw[i] ushr 2
    fun lineStart(i: Int) = raw[i] and 1 != 0
    fun atLineEnd(i: Int) = raw[i] and 2 != 0
}

/** Writes one part file. Not thread-safe; one writer per part. */
class PartWriter(private val file: File, private val width: Int) : AutoCloseable {
    private val out = DataOutputStream(BufferedOutputStream(FileOutputStream(file), 1 shl 20))
    private var files = 0L; private var tokens = 0L
    private var buf = ByteArray(1 shl 16)

    init { out.write(ByteArray(ShardFormat.HEADER_BYTES)) }

    /** [raw] entries are already `id << 2 | atLineEnd << 1 | lineStart`. */
    fun add(repo: Int, raw: IntArray, n: Int = raw.size) {
        if (buf.size < n * width) buf = ByteArray(n * width)
        var p = 0
        for (i in 0 until n) { val v = raw[i]; for (b in 0 until width) buf[p++] = (v ushr (8 * b)).toByte() }
        writeIntLe(repo); writeIntLe(n)
        out.write(buf, 0, n * width)
        files++; tokens += n
    }

    private fun writeIntLe(v: Int) { for (b in 0 until 4) out.write(v ushr (8 * b)) }

    override fun close() {
        out.close()
        java.io.RandomAccessFile(file, "rw").use { f ->
            val h = java.nio.ByteBuffer.allocate(ShardFormat.HEADER_BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            h.putInt(ShardFormat.MAGIC).putInt(ShardFormat.VERSION).putInt(width).putInt(0).putLong(files).putLong(tokens)
            f.seek(0); f.write(h.array())
        }
    }
}

/** Streams the files of one part. */
class PartReader(file: File) : AutoCloseable {
    private val inp = DataInputStream(BufferedInputStream(FileInputStream(file), 1 shl 22))
    val width: Int; val files: Long; val tokens: Long
    private var read = 0L
    private var bytes = ByteArray(1 shl 16)
    private var raw = IntArray(1 shl 12)
    private val b4 = ByteArray(4)

    init {
        val h = ByteArray(ShardFormat.HEADER_BYTES); inp.readFully(h)
        val bb = java.nio.ByteBuffer.wrap(h).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        require(bb.getInt() == ShardFormat.MAGIC) { "$file is not a token shard" }
        require(bb.getInt() == ShardFormat.VERSION) { "unsupported token shard version in $file" }
        width = bb.getInt(); bb.getInt(); files = bb.getLong(); tokens = bb.getLong()
    }

    private fun intLe(): Int { inp.readFully(b4); return (b4[0].toInt() and 0xff) or ((b4[1].toInt() and 0xff) shl 8) or ((b4[2].toInt() and 0xff) shl 16) or ((b4[3].toInt() and 0xff) shl 24) }

    /** Next file, or null at the end. The returned object and its array are reused by the following call. */
    fun next(): TokenFile? {
        if (read >= files) return null
        val repo = try { intLe() } catch (e: EOFException) { throw EOFException("truncated shard part") }
        val n = intLe()
        if (bytes.size < n * width) bytes = ByteArray(n * width)
        if (raw.size < n) raw = IntArray(n + (n shr 1))
        inp.readFully(bytes, 0, n * width)
        var p = 0
        when (width) {
            2 -> for (i in 0 until n) { raw[i] = (bytes[p].toInt() and 0xff) or ((bytes[p + 1].toInt() and 0xff) shl 8); p += 2 }
            3 -> for (i in 0 until n) { raw[i] = (bytes[p].toInt() and 0xff) or ((bytes[p + 1].toInt() and 0xff) shl 8) or ((bytes[p + 2].toInt() and 0xff) shl 16); p += 3 }
            else -> for (i in 0 until n) { raw[i] = (bytes[p].toInt() and 0xff) or ((bytes[p + 1].toInt() and 0xff) shl 8) or ((bytes[p + 2].toInt() and 0xff) shl 16) or ((bytes[p + 3].toInt() and 0xff) shl 24); p += 4 }
        }
        read++
        return TokenFile(repo, n, raw)
    }

    override fun close() = inp.close()
}

/** A shard directory opened for reading. */
class TokenShards(val dir: File) {
    val props = Properties().also { p -> File(dir, "shards.properties").inputStream().use { p.load(it) } }
    val language: String = props.getProperty("language")
    val width: Int = props.getProperty("width").toInt()
    val vocab: ShardVocab = ShardVocab.read(File(dir, "vocab.bin"))
    /** repository names and folds by repository index */
    val repos: List<String>; val repoFolds: List<String>

    init {
        val names = ArrayList<String>(); val folds = ArrayList<String>()
        File(dir, "repos.tsv").forEachLine { l -> val t = l.split('\t'); require(t[0].toInt() == names.size); folds.add(t[1]); names.add(t[2]) }
        repos = names; repoFolds = folds
    }

    fun parts(fold: String): List<File> = (dir.listFiles { f -> f.name.startsWith("$fold-") && f.name.endsWith(".tok") } ?: emptyArray()).sortedBy { it.name }

    fun tokens(fold: String) = props.getProperty("tokens.$fold")?.toLong() ?: 0L
    fun files(fold: String) = props.getProperty("files.$fold")?.toLong() ?: 0L

    /** Sequentially, in corpus order. The [TokenFile] (and its array) is only valid inside [consumer]. */
    fun forEachFile(fold: String, consumer: (TokenFile) -> Unit) {
        for (p in parts(fold)) PartReader(p).use { r -> while (true) consumer(r.next() ?: break) }
    }

    /** Copies of all files of a fold (use for the small test fold; the LM fold has billions of tokens). [stride] > 1 keeps every stride-th file. */
    fun readAll(fold: String, stride: Int = 1): List<TokenFile> {
        val out = ArrayList<TokenFile>(); var k = 0L
        forEachFile(fold) { f -> if (k++ % stride == 0L) out.add(TokenFile(f.repo, f.size, f.raw.copyOf(f.size))) }
        return out
    }

    /** Lexer-token view of a file (offsets are token indices; [MlToken.text] and kind are exact, rare identifiers read as `<ID>`). */
    fun mlTokens(f: TokenFile): List<MlToken> = List(f.size) { i -> vocab.token(f.id(i), i) }
}

/** Minimal reader of the string/number fields of a manifest line (one flat JSON object per line, as written by `prepare`). */
internal object ManifestLine {
    fun string(line: String, key: String): String? {
        val k = "\"$key\":\""
        val s = line.indexOf(k).takeIf { it >= 0 } ?: return null
        val sb = StringBuilder(); var i = s + k.length
        while (i < line.length) {
            val c = line[i]
            when {
                c == '"' -> return sb.toString()
                c == '\\' -> { val e = line[i + 1]; if (e == 'u') { sb.append(line.substring(i + 2, i + 6).toInt(16).toChar()); i += 5 } else { sb.append(when (e) { 'n' -> '\n'; 't' -> '\t'; 'r' -> '\r'; else -> e }); i++ } }
                else -> sb.append(c)
            }
            i++
        }
        return null
    }
    fun long(line: String, key: String): Long? {
        val k = "\"$key\":"; val s = line.indexOf(k).takeIf { it >= 0 } ?: return null
        var e = s + k.length; while (e < line.length && (line[e].isDigit() || line[e] == '-')) e++
        return line.substring(s + k.length, e).toLongOrNull()
    }
}
