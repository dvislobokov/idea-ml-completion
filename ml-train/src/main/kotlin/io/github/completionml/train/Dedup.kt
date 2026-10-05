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
    private val buckets = HashMap<Long, MutableList<Int>>()      // (band, bandHash) -> accepted file indices
    private val signatures = ArrayList<IntArray>()
    private val rows = permutations / bands
    private val seeds = LongArray(permutations) { mix(0x5DEECE66DL + it * 7919L) }
    var exactDuplicates = 0; private set
    var nearDuplicates = 0; private set
    var accepted = 0; private set

    /** True if the file is new (accepted into the corpus), false if it duplicates an accepted file. */
    fun offer(tokens: List<MlToken>): Boolean {
        var h = 0x1234567L
        val idents = HashSet<String>()
        for (t in tokens) { h = mix(h xor t.text.hashCode().toLong()); if (t.kind == TokenKind.IDENT) idents.add(t.text) }
        if (!exact.add(h)) { exactDuplicates++; return false }
        if (idents.size < 5) { accepted++; return true }   // too small to judge; keep
        val sig = IntArray(permutations) { Int.MAX_VALUE }
        for (id in idents) {
            val base = id.hashCode().toLong()
            for (p in 0 until permutations) { val v = (mix(base xor seeds[p]) ushr 33).toInt(); if (v < sig[p]) sig[p] = v }
        }
        // LSH: candidate if any band matches; confirm with signature Jaccard estimate
        val keys = LongArray(bands) { b -> var k = b.toLong(); for (r in 0 until rows) k = mix(k xor sig[b * rows + r].toLong()); k }
        val candidates = HashSet<Int>()
        for (k in keys) buckets[k]?.let { candidates.addAll(it) }
        for (c in candidates) {
            var same = 0
            val other = signatures[c]
            for (p in 0 until permutations) if (other[p] == sig[p]) same++
            if (same.toDouble() / permutations >= threshold) { nearDuplicates++; return false }
        }
        val idx = signatures.size
        signatures.add(sig)
        for (k in keys) buckets.getOrPut(k) { ArrayList(2) }.add(idx)
        accepted++
        return true
    }

    override fun toString() = "dedup: $accepted kept, $exactDuplicates exact and $nearDuplicates near duplicates dropped (Jaccard ≥ $threshold)"
}
