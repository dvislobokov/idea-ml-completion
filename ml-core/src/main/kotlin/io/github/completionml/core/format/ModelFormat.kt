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
    const val VERSION = 1

    class Header(
        val kind: String,          // "ngram" | "ranker"
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
        DataInputStream(BufferedInputStream(GZIPInputStream(file.inputStream()), 1 shl 16)).use { inp ->
            val magic = ByteArray(4).also { inp.readFully(it) }.toString(Charsets.US_ASCII)
            require(magic == MAGIC) { "$file: not a .cml model (magic '$magic')" }
            val version = inp.readInt()
            require(version == VERSION) { "$file: model format $version, this build reads $VERSION; retrain" }
            val header = Header(inp.readUTF(), inp.readUTF(), inp.readLong(), inp.readLong(), inp.readUTF(), version)
            require(header.kind == expectedKind) { "$file: model kind '${header.kind}', expected '$expectedKind'" }
            body(header, inp)
        }
}
