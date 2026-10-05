package io.github.completionml.core.ngram

import io.github.completionml.core.util.NgramHash
import io.github.completionml.core.util.capacityFor
import io.github.completionml.core.util.mix

/**
 * Count table for n-grams of one fixed [order]: open addressing keyed by the n-gram hash, with the token ids stored
 * alongside so that suffixes/contexts can be re-hashed during Kneser–Ney estimation. Memory per entry: 8 + 4 + 4·order bytes.
 */
class NgramTable(val order: Int, expected: Int = 1 shl 16, private val trackRepos: Boolean = false) {
    companion object {
        /** Slots per table: `ids` holds `slots × order` ints and must stay below 2^31 (order ≤ 8). ~188 M distinct n-grams at 70 % load. */
        const val MAX_SLOTS = 1 shl 28
        /** Largest `expected` that still fits: pass `minOf(expected, MAX_EXPECTED)` for corpus-sized estimates. */
        const val MAX_EXPECTED = MAX_SLOTS / 10 * 7 - 1
    }
    private var keys = LongArray(capacityFor(minOf(expected, MAX_EXPECTED)))
    private var counts = IntArray(keys.size)
    private var ids = IntArray(keys.size * order)
    // number of distinct repositories an n-gram was seen in (exact when files arrive grouped by repository)
    private var repoCount = if (trackRepos) ShortArray(keys.size) else ShortArray(0)
    private var lastRepo = if (trackRepos) IntArray(keys.size) else IntArray(0)
    private var mask = keys.size - 1
    var size = 0; private set

    /** Adds [delta] to the n-gram `tokens[from until from+order]`; [repo] identifies the source repository (for [repos]). */
    fun add(tokens: IntArray, from: Int, delta: Int = 1, repo: Int = 0, repos: Int = 1) {
        val h = NgramHash.of(tokens, from, from + order)
        var i = (mix(h) and mask.toLong()).toInt()
        while (true) {
            val k = keys[i]
            if (k == h) {
                counts[i] += delta
                if (trackRepos && lastRepo[i] != repo) { lastRepo[i] = repo; if (repoCount[i] < Short.MAX_VALUE) repoCount[i]++ }
                return
            }
            if (k == 0L) {
                keys[i] = h; counts[i] = delta
                System.arraycopy(tokens, from, ids, i * order, order)
                if (trackRepos) { lastRepo[i] = repo; repoCount[i] = repos.coerceAtMost(Short.MAX_VALUE.toInt()).toShort() }
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

    /** Distinct repositories the n-gram with [hash] was seen in (1 when not tracked; 0 if absent). */
    fun reposOf(hash: Long): Int {
        if (!trackRepos) return 1
        var i = (mix(hash) and mask.toLong()).toInt()
        while (true) { val k = keys[i]; if (k == hash) return repoCount[i].toInt(); if (k == 0L) return 0; i = (i + 1) and mask }
    }

    /** Iterates live entries: (hash, count, ids array, offset of this entry's first id). */
    inline fun forEach(action: (hash: Long, count: Int, ids: IntArray, offset: Int) -> Unit) {
        val ks = keysRef(); val cs = countsRef(); val xs = idsRef()
        for (i in ks.indices) { val k = ks[i]; if (k != 0L) action(k, cs[i], xs, i * order) }
    }

    /**
     * Removes entries with count below [minCount] or seen in fewer than [minRepos] repositories; returns how many were dropped.
     * [repos] supplies the repository count per n-gram hash (own tracking by default; a raw-count table for continuation tables).
     */
    fun prune(minCount: Int, minRepos: Int = 1, repos: (Long) -> Int = { h -> reposOf(h) }): Int {
        if (minCount <= 1 && minRepos <= 1) return 0
        val ok = keys; val oc = counts; val oi = ids; val orc = repoCount; val olr = lastRepo
        keys = LongArray(ok.size); counts = IntArray(ok.size); ids = IntArray(oi.size); size = 0
        if (trackRepos) { repoCount = ShortArray(ok.size); lastRepo = IntArray(ok.size) }
        var dropped = 0
        for (i in ok.indices) {
            if (ok[i] == 0L) continue
            if (oc[i] < minCount || (minRepos > 1 && repos(ok[i]) < minRepos)) { dropped++; continue }
            add(oi, i * order, oc[i], if (trackRepos) olr[i] else 0, if (trackRepos) orc[i].toInt() else 1)
        }
        return dropped
    }

    fun keysRef() = keys
    fun countsRef() = counts
    fun idsRef() = ids

    private fun grow() {
        check(keys.size < MAX_SLOTS) { "n-gram table of order $order exceeds $MAX_SLOTS slots: count a smaller corpus (--repos subset) or shard the counting" }
        val ok = keys; val oc = counts; val oi = ids; val orc = repoCount; val olr = lastRepo
        keys = LongArray(ok.size * 2); counts = IntArray(keys.size); ids = IntArray(keys.size * order); mask = keys.size - 1; size = 0
        if (trackRepos) { repoCount = ShortArray(keys.size); lastRepo = IntArray(keys.size) }
        for (i in ok.indices) if (ok[i] != 0L) add(oi, i * order, oc[i], if (trackRepos) olr[i] else 0, if (trackRepos) orc[i].toInt() else 1)
    }
}
