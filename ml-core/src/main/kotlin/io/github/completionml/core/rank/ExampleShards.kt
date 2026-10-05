package io.github.completionml.core.rank

import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream

/**
 * On-disk exchange format between the plugin-side example generators (which run inside the IDE platform and see PSI)
 * and `ml-train` (which trains the ranker without it). One shard = one gzip file, self-describing: language, schema,
 * source tag, then a stream of [TrainingExample]s (optionally with candidate names for debugging).
 *
 * ```
 * "CMLX" u8 version | UTF language | UTF source | i32 n, UTF×n schema names | (u8 1, example [, names])* | u8 0
 * ```
 */
object ExampleShards {
    const val MAGIC = "CMLX"
    const val VERSION = 1
    const val EXTENSION = ".cmlx"

    class Header(val language: String, val source: String, val schema: FeatureSchema)

    class Writer(file: File, language: String, source: String, val schema: FeatureSchema, private val withNames: Boolean = false) : Closeable {
        private val out = DataOutputStream(GZIPOutputStream(file.outputStream().buffered(1 shl 16)))
        var count = 0; private set
        init {
            out.writeBytes(MAGIC); out.writeByte(VERSION)
            out.writeUTF(language); out.writeUTF(source)
            out.writeInt(schema.size); for (n in schema.names) out.writeUTF(n)
        }
        fun add(example: TrainingExample) {
            require(example.base.isEmpty() || example.base[0].size == schema.baseSize) { "example base ${example.base[0].size} != schema base ${schema.baseSize}" }
            out.writeByte(1)
            example.write(out)
            val names = example.candidateNames
            if (withNames && names != null) { out.writeBoolean(true); for (n in names) out.writeUTF(n) } else out.writeBoolean(false)
            count++
        }
        override fun close() { out.writeByte(0); out.close() }
    }

    /** Streams the examples of one shard; returns its header. */
    fun forEach(file: File, action: (TrainingExample) -> Unit): Header {
        DataInputStream(GZIPInputStream(file.inputStream().buffered(1 shl 16))).use { inp ->
            val magic = ByteArray(4); inp.readFully(magic)
            require(String(magic, Charsets.ISO_8859_1) == MAGIC) { "$file: not an example shard" }
            val version = inp.readByte().toInt(); require(version == VERSION) { "$file: shard version $version, expected $VERSION" }
            val language = inp.readUTF(); val source = inp.readUTF()
            val n = inp.readInt(); val schema = FeatureSchema(List(n) { inp.readUTF() })
            while (inp.readByte().toInt() == 1) {
                val ex = TrainingExample.read(inp)
                val withNames = inp.readBoolean()
                action(if (withNames) TrainingExample(ex.kind, ex.base, ex.chosen, Array(ex.size) { inp.readUTF() }) else ex)
            }
            return Header(language, source, schema)
        }
    }

    fun header(file: File): Header {
        DataInputStream(GZIPInputStream(file.inputStream().buffered())).use { inp ->
            val magic = ByteArray(4); inp.readFully(magic)
            require(String(magic, Charsets.ISO_8859_1) == MAGIC) { "$file: not an example shard" }
            inp.readByte()
            val language = inp.readUTF(); val source = inp.readUTF()
            val n = inp.readInt(); return Header(language, source, FeatureSchema(List(n) { inp.readUTF() }))
        }
    }

    /** All shards of a directory (or one file); every shard must share the schema. */
    fun readAll(dirOrFile: File, into: MutableList<TrainingExample> = ArrayList(), log: (String) -> Unit = {}): Pair<Header, MutableList<TrainingExample>> {
        val files = if (dirOrFile.isDirectory) dirOrFile.listFiles { f -> f.name.endsWith(EXTENSION) }!!.sortedBy { it.name } else listOf(dirOrFile)
        require(files.isNotEmpty()) { "no $EXTENSION files in $dirOrFile" }
        var first: Header? = null
        for (f in files) {
            val before = into.size
            val h = forEach(f) { into.add(it) }
            if (first == null) first = h
            else {
                require(h.schema.hash == first.schema.hash) { "$f: schema differs from ${files[0]}" }
                require(h.language == first.language) { "$f: language ${h.language} differs from ${first.language}" }
            }
            log("${f.name}: ${into.size - before} lists (${h.source})")
        }
        return first!! to into
    }
}
