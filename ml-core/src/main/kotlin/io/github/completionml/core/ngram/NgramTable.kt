package io.github.completionml.core.ngram

import io.github.completionml.core.util.NgramHash
import io.github.completionml.core.util.capacityFor
import io.github.completionml.core.util.mix

/**
 * Count table for n-grams of one fixed [order]: open addressing keyed by the n-gram hash, with the token ids stored
 * alongside so that suffixes/contexts can be re-hashed during Kneser–Ney estimation. Memory per entry: 8 + 4 + 4·order bytes.
 */
class NgramTable(val order: Int, expected: Int = 1 shl 16) {
    private var keys = LongArray(capacityFor(expected))
    private var counts = IntArray(keys.size)
    private var ids = IntArray(keys.size * order)
    private var mask = keys.size - 1
    var size = 0; private set

    /** Adds [delta] to the n-gram `tokens[from until from+order]`. */
    fun add(tokens: IntArray, from: Int, delta: Int = 1) {
        val h = NgramHash.of(tokens, from, from + order)
        var i = (mix(h) and mask.toLong()).toInt()
        while (true) {
            val k = keys[i]
            if (k == h) { counts[i] += delta; return }
            if (k == 0L) {
                keys[i] = h; counts[i] = delta
                System.arraycopy(tokens, from, ids, i * order, order)
                size++
                if (size * 10 > keys.size * 7) grow()
                return
            }
            i = (i + 1) and mask
        }
    }

    fun count(hash: Long): Int {
        var i = (mix(hash) and mask.toLong()).toInt()
        while (true) { val k = keys[i]; if (k == hash) return counts[i]; if (k == 0L) return 0; i = (i + 1) and mask }
    }

    /** Iterates live entries: (hash, count, ids array, offset of this entry's first id). */
    inline fun forEach(action: (hash: Long, count: Int, ids: IntArray, offset: Int) -> Unit) {
        val ks = keysRef(); val cs = countsRef(); val xs = idsRef()
        for (i in ks.indices) { val k = ks[i]; if (k != 0L) action(k, cs[i], xs, i * order) }
    }

    /** Removes entries with count below [minCount]; returns how many were dropped. */
    fun prune(minCount: Int): Int {
        if (minCount <= 1) return 0
        val ok = keys; val oc = counts; val oi = ids
        keys = LongArray(ok.size); counts = IntArray(ok.size); ids = IntArray(oi.size); size = 0
        var dropped = 0
        for (i in ok.indices) {
            if (ok[i] == 0L) continue
            if (oc[i] < minCount) { dropped++; continue }
            add(oi, i * order, oc[i])
        }
        return dropped
    }

    fun keysRef() = keys
    fun countsRef() = counts
    fun idsRef() = ids

    private fun grow() {
        val ok = keys; val oc = counts; val oi = ids
        keys = LongArray(ok.size * 2); counts = IntArray(keys.size); ids = IntArray(keys.size * order); mask = keys.size - 1; size = 0
        for (i in ok.indices) if (ok[i] != 0L) add(oi, i * order, oc[i])
    }
}
