package io.github.completionml.core.format

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream

/**
 * `.cml` container: a small header followed by a model-kind-specific body. Gzip-compressed.
 * Bump [VERSION] whenever the body layout of any model kind changes; readers refuse other versions.
 */
object ModelFormat {
    const val MAGIC = "CML1"
    const val VERSION = 2   // v2: n-gram tables stored as CompactFloatMap (24-bit fingerprints, 8-bit quantised values)

    class Header(
        val kind: String,          // "ngram" | "ranker" | "tree-ranker"
        val language: String,      // MlLanguage.id
        val schemaHash: Long,      // FeatureSchema.hash for rankers, vocabulary/order signature for n-grams
        val createdAt: Long,
        val corpusId: String,      // free text: repos.lock SHA-256 or a description
        val version: Int = VERSION,
    )

    fun write(file: File, header: Header, body: (DataOutputStream) -> Unit) {
        file.parentFile?.mkdirs()
        DataOutputStream(BufferedOutputStream(GZIPOutputStream(file.outputStream()), 1 shl 16)).use { out ->
            out.writeBytes(MAGIC)
            out.writeInt(header.version)
            out.writeUTF(header.kind)
            out.writeUTF(header.language)
            out.writeLong(header.schemaHash)
            out.writeLong(header.createdAt)
            out.writeUTF(header.corpusId)
            body(out)
        }
    }

    fun <T> read(file: File, expectedKind: String, body: (Header, DataInputStream) -> T): T =
        read(file.inputStream(), expectedKind, body, file.toString())

    /** Reads a model from a gzip stream (a bundled plugin resource, for instance); [name] is for error messages. The stream is closed. */
    fun <T> read(stream: java.io.InputStream, expectedKind: String, body: (Header, DataInputStream) -> T, name: String = "model"): T =
        read(stream, setOf(expectedKind), body, name)

    /** Same, accepting any of [expectedKinds] (the body dispatches on `header.kind`). */
    fun <T> read(stream: java.io.InputStream, expectedKinds: Set<String>, body: (Header, DataInputStream) -> T, name: String = "model"): T =
        DataInputStream(BufferedInputStream(GZIPInputStream(stream), 1 shl 16)).use { inp ->
            val magic = ByteArray(4).also { inp.readFully(it) }.toString(Charsets.US_ASCII)
            require(magic == MAGIC) { "$name: not a .cml model (magic '$magic')" }
            val version = inp.readInt()
            require(version == VERSION) { "$name: model format $version, this build reads $VERSION; retrain" }
            val header = Header(inp.readUTF(), inp.readUTF(), inp.readLong(), inp.readLong(), inp.readUTF(), version)
            require(header.kind in expectedKinds) { "$name: model kind '${header.kind}', expected ${expectedKinds.joinToString(" | ") { "'$it'" }}" }
            body(header, inp)
        }
}
