package io.github.completionml.train

import io.github.completionml.core.spi.MlToken
import io.github.completionml.core.spi.TokenKind
import io.github.completionml.core.util.mix

/**
 * Duplicate detection over tokenised files (Allamanis 2019, "The adverse effects of code duplication in machine learning
 * models of code"): exact duplicates by hash of the token sequence, near-duplicates by MinHash/LSH over the set of
 * identifier tokens (Jaccard ≥ [threshold]). Files are offered in order; a file is a duplicate if it matches any file
 * accepted earlier — so offer training files first, then test files, and test duplicates of training files are dropped too.
 */
class Dedup(private val threshold: Double = 0.8, private val permutations: Int = 128, private val bands: Int = 16) {
    private val exact = HashSet<Long>()
    private val buckets = ChainMap()                             // (band, bandHash) -> chain of entries (file index * bands + band)
    private val next = IntList()                                 // entry -> previous entry in the same chain, -1 at the end
    private val signatures = ArrayList<IntArray>()
    private val rows = permutations / bands
    private val seeds = LongArray(permutations) { mix(0x5DEECE66DL + it * 7919L) }
    var exactDuplicates = 0; private set
    var nearDuplicates = 0; private set
    var accepted = 0; private set

    /** Exact hash of the token sequence and (for files with at least 5 distinct identifiers) the MinHash signature. Pure and thread-safe. */
    class Fingerprint(val exact: Long, val signature: IntArray?)

    enum class Verdict { NEW, EXACT, NEAR }

    fun fingerprint(tokens: List<MlToken>): Fingerprint {
        var h = 0x1234567L
        val idents = HashSet<String>()
        for (t in tokens) { h = mix(h xor t.text.hashCode().toLong()); if (t.kind == TokenKind.IDENT) idents.add(t.text) }
        if (idents.size < 5) return Fingerprint(h, null)   // too small to judge near-duplicates
        val sig = IntArray(permutations) { Int.MAX_VALUE }
        for (id in idents) {
            val base = id.hashCode().toLong()
            for (p in 0 until permutations) { val v = (mix(base xor seeds[p]) ushr 33).toInt(); if (v < sig[p]) sig[p] = v }
        }
        return Fingerprint(h, sig)
    }

    /** True if the file is new (accepted into the corpus), false if it duplicates an accepted file. */
    fun offer(tokens: List<MlToken>): Boolean = offer(fingerprint(tokens)) == Verdict.NEW

    /** Not thread-safe: fingerprints are computed in parallel, offered in a fixed order from one thread. */
    fun offer(fp: Fingerprint): Verdict {
        if (!exact.add(fp.exact)) { exactDuplicates++; return Verdict.EXACT }
        val sig = fp.signature
        if (sig == null) { accepted++; return Verdict.NEW }
        // LSH: candidate if any band matches; confirm with signature Jaccard estimate
        val keys = LongArray(bands) { b -> var k = b.toLong(); for (r in 0 until rows) k = mix(k xor sig[b * rows + r].toLong()); k }
        val idx = signatures.size
        var near = false
        scan@ for (b in 0 until bands) {
            var e = buckets.head(keys[b])
            while (e >= 0) {
                val other = signatures[e / bands]
                var same = 0
                for (p in 0 until permutations) if (other[p] == sig[p]) same++
                if (same.toDouble() / permutations >= threshold) { near = true; break@scan }
                e = next.get(e)
            }
        }
        if (near) { nearDuplicates++; return Verdict.NEAR }
        signatures.add(sig)
        for (b in 0 until bands) {
            val e = idx * bands + b
            next.add(buckets.put(keys[b], e))
        }
        accepted++
        return Verdict.NEW
    }

    override fun toString() = "dedup: $accepted kept, $exactDuplicates exact and $nearDuplicates near duplicates dropped (Jaccard ≥ $threshold)"
}

/** Growable int array (no boxing: a corpus of millions of files has hundreds of millions of band entries). */
private class IntList {
    private var a = IntArray(1024); private var n = 0
    fun add(v: Int) { if (n == a.size) a = a.copyOf(n * 2); a[n++] = v }
    fun get(i: Int) = a[i]
}

/** Open-addressing map from a 64-bit band key to the newest entry of its chain; [put] returns the previous head (or -1). */
private class ChainMap {
    private var keys = LongArray(1 shl 16); private var heads = IntArray(1 shl 16) { -1 }; private var used = BooleanArray(1 shl 16); private var size = 0
    private fun slot(k: Long, mask: Int, used: BooleanArray, keys: LongArray): Int {
        var i = (mix(k) ushr 7).toInt() and mask
        while (used[i] && keys[i] != k) i = (i + 1) and mask
        return i
    }
    fun head(k: Long): Int { val i = slot(k, keys.size - 1, used, keys); return if (used[i]) heads[i] else -1 }
    fun put(k: Long, entry: Int): Int {
        if (size * 2 >= keys.size) grow()
        val i = slot(k, keys.size - 1, used, keys)
        val old = if (used[i]) heads[i] else -1
        if (!used[i]) { used[i] = true; keys[i] = k; size++ }
        heads[i] = entry
        return old
    }
    private fun grow() {
        val nk = LongArray(keys.size * 2); val nh = IntArray(keys.size * 2); val nu = BooleanArray(keys.size * 2)
        for (i in keys.indices) if (used[i]) { val j = slot(keys[i], nk.size - 1, nu, nk); nu[j] = true; nk[j] = keys[i]; nh[j] = heads[i] }
        keys = nk; heads = nh; used = nu
    }
}
