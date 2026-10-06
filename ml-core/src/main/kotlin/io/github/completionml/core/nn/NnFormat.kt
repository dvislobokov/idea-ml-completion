package io.github.completionml.core.nn

import java.io.File
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

/**
 * A weight tensor of a neural model.
 *
 * Every 2-D weight is stored as `[in, out]` row-major (i.e. PyTorch `weight.T`), so that `y = x · W`; quantised
 * symmetrically to int8 with one float scale per output column: `W[i, o] = q[i, o] * scales[o]`.
 */
sealed class NnTensor(val name: String) {
    /** int8 matrix `[rows = in, cols = out]`; [data] is a read-only slice of exactly rows*cols bytes (usually off-heap). */
    class Q8(name: String, val rows: Int, val cols: Int, val data: ByteBuffer, val scales: FloatArray) : NnTensor(name) {
        init {
            require(data.remaining() == rows * cols) { "$name: ${data.remaining()} bytes for $rows x $cols" }
            require(scales.size == cols) { "$name: ${scales.size} scales for $cols columns" }
        }

        fun get(i: Int, o: Int): Float = data.get(data.position() + i * cols + o) * scales[o]
    }

    class F32(name: String, val data: FloatArray) : NnTensor(name)
}

/**
 * Neural `.cml` file: uncompressed, little-endian, 64-byte aligned tensors so that it can be memory-mapped directly
 * (the gzip container of [io.github.completionml.core.format.ModelFormat] can't be). Layout: docs/NN-FORMAT.md.
 */
object NnFormat {
    const val MAGIC = "CMLN"
    const val VERSION = 1
    const val KIND = "nn"
    const val DTYPE_F32 = 0
    const val DTYPE_Q8 = 1
    private const val ALIGN = 64L

    class Model(val meta: Map<String, String>, val config: NnConfig, val tensors: Map<String, NnTensor>) {
        fun q8(name: String) = tensors[name] as? NnTensor.Q8 ?: error("tensor '$name' missing or not int8")
        fun f32(name: String) = tensors[name] as? NnTensor.F32 ?: error("tensor '$name' missing or not f32")
    }

    /** Tensor names. Layer tensors are prefixed with `layers.<i>.`. */
    object Names {
        const val TOK_EMB = "tok_emb"          // [dModel, vocab]  (embedding row of token v = column v)
        const val LM_HEAD = "lm_head"          // [dModel, vocab]  only when tiedEmbeddings=false
        const val FINAL_NORM = "final_norm"    // f32 [dModel]
        fun layer(i: Int, t: String) = "layers.$i.$t"
        const val ATTN_NORM = "attn_norm"      // f32 [dModel]
        const val WQ = "wq"                    // [dModel, nHeads*headDim]
        const val WK = "wk"                    // [dModel, nKvHeads*headDim]
        const val WV = "wv"                    // [dModel, nKvHeads*headDim]
        const val WO = "wo"                    // [nHeads*headDim, dModel]
        const val FFN_NORM = "ffn_norm"        // f32 [dModel]
        const val W1 = "w1"                    // gate [dModel, ffnDim]
        const val W3 = "w3"                    // up   [dModel, ffnDim]
        const val W2 = "w2"                    // down [ffnDim, dModel]
    }

    /** Expected tensor shapes for [config]: name -> dims (`[n]` for f32 vectors, `[in, out]` for int8 matrices). */
    fun expectedShapes(c: NnConfig): Map<String, IntArray> {
        val m = LinkedHashMap<String, IntArray>()
        m[Names.TOK_EMB] = intArrayOf(c.dModel, c.vocabSize)
        for (l in 0 until c.nLayers) {
            m[Names.layer(l, Names.ATTN_NORM)] = intArrayOf(c.dModel)
            m[Names.layer(l, Names.WQ)] = intArrayOf(c.dModel, c.dModel)
            m[Names.layer(l, Names.WK)] = intArrayOf(c.dModel, c.kvDim)
            m[Names.layer(l, Names.WV)] = intArrayOf(c.dModel, c.kvDim)
            m[Names.layer(l, Names.WO)] = intArrayOf(c.dModel, c.dModel)
            m[Names.layer(l, Names.FFN_NORM)] = intArrayOf(c.dModel)
            m[Names.layer(l, Names.W1)] = intArrayOf(c.dModel, c.ffnDim)
            m[Names.layer(l, Names.W3)] = intArrayOf(c.dModel, c.ffnDim)
            m[Names.layer(l, Names.W2)] = intArrayOf(c.ffnDim, c.dModel)
        }
        m[Names.FINAL_NORM] = intArrayOf(c.dModel)
        if (!c.tiedEmbeddings) m[Names.LM_HEAD] = intArrayOf(c.dModel, c.vocabSize)
        return m
    }

    // ---------------------------------------------------------------------------------------------------- writing

    /**
     * Writes a model. [meta] holds free-form header fields (language, corpusId, tokenizer, …); config keys and `kind`
     * are added from [config]. Tensors are written in the given order.
     */
    fun write(file: File, config: NnConfig, meta: Map<String, String>, tensors: List<NnTensor>) {
        validate(config, tensors.associateBy { it.name })
        val allMeta = LinkedHashMap<String, String>()
        allMeta["kind"] = KIND
        allMeta.putAll(meta)
        allMeta.putAll(config.toMeta())
        val metaBytes = allMeta.entries.joinToString("") { (k, v) ->
            require('=' !in k && '\n' !in k && '\n' !in v) { "bad meta entry '$k'" }
            "$k=$v\n"
        }.toByteArray(Charsets.UTF_8)

        // Directory size, then the data offsets.
        var dirSize = 4L
        for (t in tensors) dirSize += 2 + t.name.toByteArray(Charsets.UTF_8).size + 2 + 4L * rank(t) + 16
        var pos = align(12L + metaBytes.size + dirSize)
        val dataOff = LongArray(tensors.size); val scaleOff = LongArray(tensors.size)
        tensors.forEachIndexed { k, t ->
            when (t) {
                is NnTensor.F32 -> { dataOff[k] = pos; pos = align(pos + 4L * t.data.size) }
                is NnTensor.Q8 -> {
                    dataOff[k] = pos; pos = align(pos + t.rows.toLong() * t.cols)
                    scaleOff[k] = pos; pos = align(pos + 4L * t.cols)
                }
            }
        }

        file.absoluteFile.parentFile?.mkdirs()
        val tmp = File(file.path + ".tmp")
        FileChannel.open(tmp.toPath(), StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING).use { ch ->
            val head = ByteBuffer.allocate((12 + metaBytes.size + dirSize).toInt()).order(ByteOrder.LITTLE_ENDIAN)
            head.put(MAGIC.toByteArray(Charsets.US_ASCII)).putInt(VERSION).putInt(metaBytes.size).put(metaBytes)
            head.putInt(tensors.size)
            tensors.forEachIndexed { k, t ->
                val nb = t.name.toByteArray(Charsets.UTF_8)
                head.putShort(nb.size.toShort()).put(nb)
                when (t) {
                    is NnTensor.F32 -> head.put(DTYPE_F32.toByte()).put(1).putInt(t.data.size)
                    is NnTensor.Q8 -> head.put(DTYPE_Q8.toByte()).put(2).putInt(t.rows).putInt(t.cols)
                }
                head.putLong(dataOff[k]).putLong(scaleOff[k])
            }
            head.flip(); writeFully(ch, head, 0)
            tensors.forEachIndexed { k, t ->
                when (t) {
                    is NnTensor.F32 -> writeFully(ch, floatsLe(t.data), dataOff[k])
                    is NnTensor.Q8 -> {
                        writeFully(ch, t.data.duplicate(), dataOff[k])
                        writeFully(ch, floatsLe(t.scales), scaleOff[k])
                    }
                }
            }
            if (ch.size() < pos) writeFully(ch, ByteBuffer.allocate((pos - ch.size()).toInt()), ch.size())
        }
        if (!tmp.renameTo(file)) { file.delete(); require(tmp.renameTo(file)) { "cannot rename $tmp to $file" } }
    }

    // ---------------------------------------------------------------------------------------------------- reading

    /** Memory-maps [file] read-only; int8 weights stay off-heap (page cache), only scales and norms are copied. */
    fun read(file: File): Model {
        RandomAccessFile(file, "r").use { raf ->
            val size = raf.length()
            require(size < Int.MAX_VALUE) { "$file: ${size} bytes, > 2 GB is not supported" }
            val buf = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, size)
            return parse(buf, file.toString())
        }
    }

    /** Reads a model from a stream (e.g. a bundled resource) into a direct (off-heap) buffer. The stream is closed. */
    fun read(stream: InputStream, name: String = "model"): Model {
        stream.use { s ->
            var buf = ByteBuffer.allocateDirect(1 shl 20)
            val chunk = ByteArray(1 shl 16)
            while (true) {
                val n = s.read(chunk)
                if (n < 0) break
                if (buf.remaining() < n) {
                    val bigger = ByteBuffer.allocateDirect(maxOf(buf.capacity() * 2L, buf.position() + n.toLong()).coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
                    buf.flip(); bigger.put(buf); buf = bigger
                }
                buf.put(chunk, 0, n)
            }
            buf.flip()
            return parse(buf.slice(), name)
        }
    }

    fun parse(buf0: ByteBuffer, name: String): Model {
        val buf = buf0.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { buf.get(it) }.toString(Charsets.US_ASCII)
        require(magic == MAGIC) { "$name: not a neural .cml model (magic '$magic')" }
        val version = buf.getInt()
        require(version == VERSION) { "$name: neural model format $version, this build reads $VERSION; re-export" }
        val metaBytes = ByteArray(buf.getInt()).also { buf.get(it) }
        val meta = LinkedHashMap<String, String>()
        for (line in metaBytes.toString(Charsets.UTF_8).split('\n')) {
            if (line.isEmpty()) continue
            val eq = line.indexOf('=')
            require(eq > 0) { "$name: bad meta line '$line'" }
            meta[line.substring(0, eq)] = line.substring(eq + 1)
        }
        require(meta["kind"] == KIND) { "$name: model kind '${meta["kind"]}', expected '$KIND'" }
        val config = NnConfig.fromMeta(meta)
        val count = buf.getInt()
        val tensors = LinkedHashMap<String, NnTensor>()
        repeat(count) {
            val nb = ByteArray(buf.getShort().toInt() and 0xFFFF).also { buf.get(it) }
            val tname = nb.toString(Charsets.UTF_8)
            val dtype = buf.get().toInt()
            val rank = buf.get().toInt()
            val dims = IntArray(rank) { buf.getInt() }
            val dataOff = buf.getLong(); val scaleOff = buf.getLong()
            val t = when (dtype) {
                DTYPE_F32 -> {
                    require(rank == 1) { "$name: $tname: f32 tensors must be 1-D" }
                    val fa = FloatArray(dims[0])
                    buf.slice(dataOff.toInt(), 4 * dims[0]).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(fa)
                    NnTensor.F32(tname, fa)
                }
                DTYPE_Q8 -> {
                    require(rank == 2) { "$name: $tname: int8 tensors must be 2-D" }
                    val (rows, cols) = dims[0] to dims[1]
                    val sc = FloatArray(cols)
                    buf.slice(scaleOff.toInt(), 4 * cols).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(sc)
                    NnTensor.Q8(tname, rows, cols, buf.slice(dataOff.toInt(), rows * cols).asReadOnlyBuffer(), sc)
                }
                else -> error("$name: $tname: unknown dtype $dtype")
            }
            tensors[tname] = t
        }
        validate(config, tensors)
        return Model(meta, config, tensors)
    }

    private fun validate(config: NnConfig, tensors: Map<String, NnTensor>) {
        for ((n, dims) in expectedShapes(config)) {
            val t = tensors[n] ?: error("tensor '$n' missing")
            val ok = when (t) {
                is NnTensor.F32 -> dims.size == 1 && t.data.size == dims[0]
                is NnTensor.Q8 -> dims.size == 2 && t.rows == dims[0] && t.cols == dims[1]
            }
            require(ok) { "tensor '$n': expected ${dims.joinToString("x")}" }
        }
    }

    private fun rank(t: NnTensor) = if (t is NnTensor.Q8) 2 else 1
    private fun align(x: Long) = (x + ALIGN - 1) / ALIGN * ALIGN

    private fun floatsLe(a: FloatArray): ByteBuffer {
        val b = ByteBuffer.allocate(4 * a.size).order(ByteOrder.LITTLE_ENDIAN)
        b.asFloatBuffer().put(a)
        return b
    }

    private fun writeFully(ch: FileChannel, b: ByteBuffer, at: Long) {
        var p = at
        while (b.hasRemaining()) p += ch.write(b, p)
    }
}

/** Symmetric per-output-column int8 quantisation of `[rows = in, cols = out]` float matrices. */
object Int8Quant {
    /** Quantises row-major [w] (`rows * cols`) into a heap-backed [NnTensor.Q8]. Zero columns get scale 0. */
    fun quantize(name: String, w: FloatArray, rows: Int, cols: Int): NnTensor.Q8 {
        require(w.size == rows * cols)
        val maxAbs = FloatArray(cols)
        for (i in 0 until rows) for (o in 0 until cols) maxAbs[o] = maxOf(maxAbs[o], kotlin.math.abs(w[i * cols + o]))
        val scales = FloatArray(cols) { maxAbs[it] / 127f }
        val q = ByteArray(rows * cols)
        for (i in 0 until rows) for (o in 0 until cols) {
            val s = scales[o]
            q[i * cols + o] = if (s == 0f) 0 else Math.round(w[i * cols + o] / s).coerceIn(-127, 127).toByte()
        }
        return NnTensor.Q8(name, rows, cols, ByteBuffer.wrap(q), scales)
    }

    fun dequantize(t: NnTensor.Q8): FloatArray {
        val out = FloatArray(t.rows * t.cols)
        val base = t.data.position()
        for (i in 0 until t.rows) for (o in 0 until t.cols) out[i * t.cols + o] = t.data.get(base + i * t.cols + o) * t.scales[o]
        return out
    }
}
