package io.github.completionml.core.util

import java.io.DataInput
import java.io.DataOutput

/** Read-only view shared by the exact ([LongFloatMap]) and the compact ([CompactFloatMap]) storage. Missing keys read as NaN. */
interface FloatLookup {
    fun get(key: Long): Float
    val size: Int
}

/**
 * Compact Long -> Float map for shipped models: open addressing over 64-bit keys, but each slot keeps only a
 * [fingerprintBits]-bit fingerprint of the key (the slot index already fixes another log2(capacity) bits) and an 8-bit
 * quantised value (256 bins chosen at quantiles of the stored values). Memory: 5 bytes per slot; disk: ~5 bytes per entry.
 *
 * A lookup of a key that was never inserted returns a wrong value with probability ≈ probes / 2^fingerprintBits
 * (≈ 1e-7 at 24 bits), which is negligible against the smoothing noise of an n-gram model.
 */
class CompactFloatMap private constructor(
    private val fp: IntArray,          // 0 = empty; otherwise fingerprint + 1
    private val q: ByteArray,
    private val table: FloatArray,     // 256 bin centres
    private val fingerprintBits: Int,
    override val size: Int,
) : FloatLookup {
    private val mask = fp.size - 1
    private val fpMask = (1L shl fingerprintBits) - 1

    override fun get(key: Long): Float {
        val f = fingerprint(key)
        var i = slot(key)
        while (true) {
            val k = fp[i]
            if (k == f) return table[q[i].toInt() and 0xff]
            if (k == 0) return Float.NaN
            i = (i + 1) and mask
        }
    }

    private fun slot(key: Long) = (mix(key) and mask.toLong()).toInt()
    private fun fingerprint(key: Long) = (((key ushr 20) and fpMask) + 1).toInt()   // bits independent of the slot (slot uses mix(key))

    fun write(out: DataOutput) {
        out.writeInt(fingerprintBits)
        out.writeInt(fp.size)
        out.writeInt(size)
        for (t in table) out.writeFloat(t)
        // live entries in slot order: slot delta (varint), fingerprint (3 bytes for ≤ 24 bits, else 4), quantised value
        var prev = -1
        for (i in fp.indices) {
            if (fp[i] == 0) continue
            writeVarint(out, i - prev - 1); prev = i
            if (fingerprintBits <= 24) { out.writeByte(fp[i] ushr 16); out.writeShort(fp[i]) } else out.writeInt(fp[i])
            out.writeByte(q[i].toInt())
        }
    }

    companion object {
        /** Builds from an exact map; [fingerprintBits] 16..32. */
        fun of(src: LongFloatMap, fingerprintBits: Int = 24): CompactFloatMap {
            val n = src.size
            val capacity = capacityFor(n)
            val table = quantiles(src)
            val fp = IntArray(capacity); val q = ByteArray(capacity)
            val mask = capacity - 1
            val fpMask = (1L shl fingerprintBits) - 1
            src.forEach { key, value ->
                var i = (mix(key) and mask.toLong()).toInt()
                while (fp[i] != 0) i = (i + 1) and mask
                fp[i] = (((key ushr 20) and fpMask) + 1).toInt()
                q[i] = nearestBin(table, value).toByte()
            }
            return CompactFloatMap(fp, q, table, fingerprintBits, n)
        }

        fun read(inp: DataInput): CompactFloatMap {
            val bits = inp.readInt(); val capacity = inp.readInt(); val n = inp.readInt()
            val table = FloatArray(256) { inp.readFloat() }
            val fp = IntArray(capacity); val q = ByteArray(capacity)
            var i = -1
            for (e in 0 until n) {
                i += readVarint(inp) + 1
                fp[i] = if (bits <= 24) ((inp.readUnsignedByte() shl 16) or inp.readUnsignedShort()) else inp.readInt()
                q[i] = inp.readByte()
            }
            return CompactFloatMap(fp, q, table, bits, n)
        }

        /** 256 bin centres at value quantiles (sampled); extreme bins hold the min and max exactly. */
        private fun quantiles(src: LongFloatMap): FloatArray {
            val n = src.size
            val step = maxOf(1, n / 200_000)
            val sample = ArrayList<Float>(minOf(n, 200_000) + 1)
            var idx = 0
            src.forEach { _, v -> if (idx++ % step == 0) sample.add(v) }
            sample.sort()
            if (sample.isEmpty()) return FloatArray(256)
            return FloatArray(256) { b -> sample[((b + 0.5) / 256 * sample.size).toInt().coerceIn(0, sample.size - 1)] }
                .also { it[0] = sample.first(); it[255] = sample.last() }
        }

        private fun nearestBin(table: FloatArray, v: Float): Int {
            var lo = 0; var hi = table.size - 1
            while (lo < hi) { val mid = (lo + hi) ushr 1; if (table[mid] < v) lo = mid + 1 else hi = mid }
            if (lo > 0 && v - table[lo - 1] < table[lo] - v) lo--
            return lo
        }

        private fun writeVarint(out: DataOutput, value: Int) {
            var v = value
            while (v >= 0x80) { out.writeByte((v and 0x7f) or 0x80); v = v ushr 7 }
            out.writeByte(v)
        }

        private fun readVarint(inp: DataInput): Int {
            var result = 0; var shift = 0
            while (true) { val b = inp.readUnsignedByte(); result = result or ((b and 0x7f) shl shift); if (b and 0x80 == 0) return result; shift += 7 }
        }
    }
}
