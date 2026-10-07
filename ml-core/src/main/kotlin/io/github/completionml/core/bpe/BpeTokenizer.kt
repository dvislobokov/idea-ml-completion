package io.github.completionml.core.bpe

import java.io.InputStream

/**
 * Byte-level BPE tokenizer for source code, byte-exact twin of the Python reference (`cmlbpe.py`, ~/work/nn/tokenizer).
 *
 * Any byte sequence round-trips exactly ([encodeBytes] / [decodeBytes]); there is no unknown token. Text is encoded as
 * UTF-8 as is (strings, comments and whitespace are data). Special tokens are never produced from text: the literal
 * string `<|endoftext|>` is encoded as ordinary bytes, special ids are only inserted by the caller (FIM prompts etc).
 *
 * Id layout: `0..255` raw bytes, `256 + r` the token created by merge rule `r`, then the specials
 * (`specialBase + i`, in the order of the file; see [endOfText], [fimPrefix], ...). `vocabSize` includes all of them.
 *
 * Pre-tokenizer "go-code-1" (hand-written scanner over bytes, no Unicode tables):
 *  - byte classes: L = `[A-Za-z_]` and every byte >= 0x80, D = `[0-9]`, W = space or tab, N = CR or LF, P = anything else;
 *  - rule 1: newline run `(\r\n | \n | \r)+` followed by all W bytes (indentation belongs to the newline token);
 *  - rule 2: a maximal W run; if it ends with a space and is followed by a byte that is neither W nor N, that last space
 *    is detached and becomes the optional prefix of the next unit, the rest of the run is a token of its own;
 *    otherwise the whole run is one token;
 *  - rule 3: unit = optional single space + (L+ | one D | P+);
 *  - pre-tokens longer than 128 bytes are cut into consecutive 128-byte chunks.
 * Inside a pre-token merges are applied by rank: repeatedly take the pair with the lowest rank and merge all its
 * non-overlapping occurrences left to right (identical to the trainer's order). Results are cached per pre-token.
 *
 * Instances are immutable and thread-safe; the pre-token cache is per thread (~1.5 MB each, bounded).
 */
class BpeTokenizer private constructor(
    private val mergeLeft: IntArray,
    private val mergeRight: IntArray,
    val specials: List<String>,
) {
    val mergeCount: Int = mergeLeft.size
    val specialBase: Int = 256 + mergeCount
    val vocabSize: Int = specialBase + specials.size

    val endOfText: Int get() = special("<|endoftext|>")
    val fimPrefix: Int get() = special("<|fim_prefix|>")
    val fimMiddle: Int get() = special("<|fim_middle|>")
    val fimSuffix: Int get() = special("<|fim_suffix|>")
    val fileSep: Int get() = special("<|file_sep|>")
    val repoName: Int get() = special("<|repo_name|>")
    val pad: Int get() = special("<|pad|>")

    fun specialId(name: String): Int? = specials.indexOf(name).let { if (it < 0) null else specialBase + it }
    private fun special(name: String): Int = specialId(name) ?: error("special token $name is not in this vocabulary")
    fun isSpecial(id: Int): Boolean = id >= specialBase && id < vocabSize

    /** Byte string of every non-special token; index = id. */
    private val tokenBytes: Array<ByteArray> = run {
        val t = arrayOfNulls<ByteArray>(specialBase)
        for (i in 0 until 256) t[i] = byteArrayOf(i.toByte())
        for (r in 0 until mergeCount) t[256 + r] = t[mergeLeft[r]]!! + t[mergeRight[r]]!!
        @Suppress("UNCHECKED_CAST")
        t as Array<ByteArray>
    }

    // pair (left shl 16 | right) -> rank, open addressing; the key -1 marks an empty slot
    private val pairKeys: IntArray
    private val pairRank: IntArray
    private val pairMask: Int

    init {
        var cap = 16
        while (cap < mergeCount * 2 + 2) cap = cap shl 1
        pairKeys = IntArray(cap) { -1 }
        pairRank = IntArray(cap)
        pairMask = cap - 1
        for (r in mergeCount - 1 downTo 0) { // the lowest rank wins if a pair is listed twice
            val key = (mergeLeft[r] shl 16) or mergeRight[r]
            var i = pairSlot(key)
            while (pairKeys[i] != -1 && pairKeys[i] != key) i = (i + 1) and pairMask
            pairKeys[i] = key; pairRank[i] = r
        }
    }

    private fun pairSlot(key: Int): Int = ((key * -0x61c88647) ushr 7) and pairMask

    private fun rankOf(a: Int, b: Int): Int {
        val key = (a shl 16) or b
        var i = pairSlot(key)
        while (true) {
            val k = pairKeys[i]
            if (k == key) return pairRank[i]
            if (k == -1) return Int.MAX_VALUE
            i = (i + 1) and pairMask
        }
    }

    private val caches = ThreadLocal.withInitial { Scratch() }

    // ---------------------------------------------------------------------------------------------------- encoding

    fun encode(text: String): IntArray {
        val b = text.toByteArray(Charsets.UTF_8)
        return encodeBytes(b, 0, b.size)
    }

    fun encodeBytes(bytes: ByteArray, off: Int = 0, len: Int = bytes.size - off): IntArray {
        require(off >= 0 && len >= 0 && off + len <= bytes.size)
        val sc = caches.get()
        sc.out.size = 0
        scan(bytes, off, off + len) { s, e -> emit(bytes, s, e, sc) }
        return sc.out.a.copyOf(sc.out.size)
    }

    /**
     * The pre-tokenizer scanner over `b[from, n)`: calls `emit(start, end)` for every pre-token (before the 128-byte
     * chunking). Inline so that the encoder's hot loop pays nothing for the callback.
     */
    private inline fun scan(b: ByteArray, from: Int, n: Int, emit: (Int, Int) -> Unit) {
        var i = from
        while (i < n) {
            val c = b[i].toInt() and 0xff
            var start: Int
            var k: Int
            if (c == 10 || c == 13) {
                var j = i
                while (j < n) {
                    val x = b[j].toInt()
                    if (x == 13 && j + 1 < n && b[j + 1].toInt() == 10) j += 2
                    else if (x == 10 || x == 13) j++
                    else break
                }
                while (j < n && isW(b[j].toInt() and 0xff)) j++
                emit(i, j); i = j
                continue
            }
            if (c == 32 || c == 9) {
                var j = i
                while (j < n && isW(b[j].toInt() and 0xff)) j++
                val nx = if (j < n) b[j].toInt() and 0xff else -1
                if (nx >= 0 && b[j - 1].toInt() == 32 && !isW(nx) && nx != 10 && nx != 13) {
                    if (j - 1 > i) emit(i, j - 1)
                    start = j - 1; k = j
                } else {
                    emit(i, j); i = j
                    continue
                }
            } else {
                start = i; k = i
            }
            val ck = b[k].toInt() and 0xff
            var e: Int
            if (ck in 48..57) {
                e = k + 1
            } else if (isL(ck)) {
                e = k + 1
                while (e < n && isL(b[e].toInt() and 0xff)) e++
            } else {
                e = k + 1
                while (e < n && isP(b[e].toInt() and 0xff)) e++
            }
            emit(start, e); i = e
        }
    }

    /**
     * Pre-token boundaries of `bytes[from, to)` scanned from [from]: every offset where a pre-token (after the
     * 128-byte chunking) starts or ends, ascending, including [from] and [to]. Token healing uses the largest boundary
     * at or before the cursor — see [lastPreTokenBoundary]. Mirrors `cmlbpe.pretokenize` offsets.
     */
    fun preTokenBoundaries(bytes: ByteArray, from: Int = 0, to: Int = bytes.size): IntArray {
        require(from in 0..to && to <= bytes.size)
        var out = IntArray(16); var n = 0
        fun add(v: Int) { if (n == out.size) out = out.copyOf(n * 2); out[n++] = v }
        add(from)
        scan(bytes, from, to) { s, e ->
            var p = s + MAX_PRETOKEN
            while (p < e) { add(p); p += MAX_PRETOKEN }
            add(e)
        }
        return out.copyOf(n)
    }

    /**
     * Largest pre-token boundary `<= cursor` when `bytes[from, to)` is scanned from [from]. The scan must start at a
     * real boundary (a file start, or the LF byte before the current line: the newline token owns the indentation that
     * follows it) and should extend past the cursor to the end of the line, because what follows the cursor decides
     * where a punctuation run or a word ends (`foo(` + `)` is one pre-token `()`).
     */
    fun lastPreTokenBoundary(bytes: ByteArray, from: Int, cursor: Int, to: Int): Int {
        require(from <= cursor && cursor <= to)
        var last = from
        for (b in preTokenBoundaries(bytes, from, to)) { if (b > cursor) break; last = b }
        return last
    }

    private fun emit(b: ByteArray, s0: Int, e: Int, sc: Scratch) {
        var s = s0
        while (e - s > MAX_PRETOKEN) {
            piece(b, s, s + MAX_PRETOKEN, sc); s += MAX_PRETOKEN
        }
        piece(b, s, e, sc)
    }

    private fun piece(b: ByteArray, s: Int, e: Int, sc: Scratch) {
        val len = e - s
        if (len == 1) { sc.out.add(b[s].toInt() and 0xff); return }
        val out = sc.out
        val cache = sc.cache
        var h = -0x7ee3623b
        for (i in s until e) h = (h xor (b[i].toInt() and 0xff)) * 0x01000193
        h = h xor (h ushr 15)
        var slot = h and PieceCache.MASK
        while (cache.kLen[slot] != 0) {
            if (cache.hash[slot] == h && cache.kLen[slot] == len && cache.same(slot, b, s)) {
                out.addAll(cache.vals, cache.vOff[slot], cache.vLen[slot]); return
            }
            slot = (slot + 1) and PieceCache.MASK
        }
        val ids = sc.work
        for (i in 0 until len) ids[i] = b[s + i].toInt() and 0xff
        val n = bpe(ids, len)
        out.addAll(ids, 0, n)
        if (!cache.hasRoom(len, n)) {
            cache.clear()
            slot = h and PieceCache.MASK // the table is empty now, the first slot is free
        }
        cache.put(slot, h, b, s, len, ids, n)
    }

    /** Merges `ids[0 until n]` in place, returns the new length. */
    private fun bpe(ids: IntArray, n0: Int): Int {
        var n = n0
        while (n > 1) {
            var best = Int.MAX_VALUE
            for (i in 0 until n - 1) {
                val r = rankOf(ids[i], ids[i + 1])
                if (r < best) best = r
            }
            if (best == Int.MAX_VALUE) break
            val a = mergeLeft[best]; val bb = mergeRight[best]; val id = 256 + best
            var w = 0; var i = 0
            while (i < n) {
                if (i + 1 < n && ids[i] == a && ids[i + 1] == bb) { ids[w++] = id; i += 2 } else { ids[w++] = ids[i]; i++ }
            }
            n = w
        }
        return n
    }

    // ---------------------------------------------------------------------------------------------------- decoding

    /** Exact inverse of [encodeBytes]. Special ids are written as their literal text (UTF-8). */
    fun decodeBytes(ids: IntArray, from: Int = 0, to: Int = ids.size): ByteArray {
        var total = 0
        for (p in from until to) total += tokenLength(ids[p])
        val out = ByteArray(total)
        var w = 0
        for (p in from until to) {
            val id = ids[p]
            val t = if (id < specialBase) tokenBytes[id] else specials[id - specialBase].toByteArray(Charsets.UTF_8)
            System.arraycopy(t, 0, out, w, t.size); w += t.size
        }
        return out
    }

    private fun tokenLength(id: Int): Int {
        require(id >= 0 && id < vocabSize) { "token id $id out of range 0..${vocabSize - 1}" }
        return if (id < specialBase) tokenBytes[id].size else specials[id - specialBase].toByteArray(Charsets.UTF_8).size
    }

    /** Decodes as UTF-8; malformed sequences (possible when ids are cut mid-character) become U+FFFD. */
    fun decode(ids: IntArray): String = String(decodeBytes(ids), Charsets.UTF_8)

    /** Raw bytes of one token (a copy). For specials: their literal text. */
    fun tokenBytes(id: Int): ByteArray = decodeBytes(intArrayOf(id))

    // ---------------------------------------------------------------------------------------------------- scratch

    private class IntSink {
        var a = IntArray(1 shl 12)
        var size = 0
        fun add(v: Int) { if (size == a.size) a = a.copyOf(size * 2); a[size++] = v }
        fun addAll(src: IntArray, off: Int, len: Int) {
            if (size + len > a.size) a = a.copyOf(maxOf(a.size * 2, size + len))
            System.arraycopy(src, off, a, size, len); size += len
        }
    }

    private class PieceCache {
        val hash = IntArray(SLOTS)
        val kLen = IntArray(SLOTS) // 0 = empty slot
        val kOff = IntArray(SLOTS)
        val vOff = IntArray(SLOTS)
        val vLen = IntArray(SLOTS)
        var keys = ByteArray(1 shl 18)
        var vals = IntArray(1 shl 16)
        var kUsed = 0
        var vUsed = 0
        var count = 0

        fun same(slot: Int, b: ByteArray, s: Int): Boolean {
            val o = kOff[slot]
            for (i in 0 until kLen[slot]) if (keys[o + i] != b[s + i]) return false
            return true
        }

        fun hasRoom(klen: Int, vlen: Int): Boolean {
            if (count >= SLOTS / 2) return false
            if (kUsed + klen > MAX_KEYS || vUsed + vlen > MAX_VALS) return false
            return true
        }

        fun put(slot: Int, h: Int, b: ByteArray, s: Int, klen: Int, ids: IntArray, n: Int) {
            if (kUsed + klen > keys.size) keys = keys.copyOf(maxOf(keys.size * 2, kUsed + klen))
            if (vUsed + n > vals.size) vals = vals.copyOf(maxOf(vals.size * 2, vUsed + n))
            System.arraycopy(b, s, keys, kUsed, klen)
            System.arraycopy(ids, 0, vals, vUsed, n)
            hash[slot] = h; kLen[slot] = klen; kOff[slot] = kUsed; vOff[slot] = vUsed; vLen[slot] = n
            kUsed += klen; vUsed += n; count++
        }

        fun clear() { java.util.Arrays.fill(kLen, 0); kUsed = 0; vUsed = 0; count = 0 }

        companion object {
            const val SLOTS = 1 shl 17
            const val MASK = SLOTS - 1
            const val MAX_KEYS = 1 shl 21
            const val MAX_VALS = 1 shl 20
        }
    }

    private class Scratch {
        val out = IntSink()
        val cache = PieceCache()
        val work = IntArray(MAX_PRETOKEN + 1)
    }

    companion object {
        const val FORMAT_MAGIC = "CMLBPE"
        const val FORMAT_VERSION = 1
        const val PRETOKENIZER_ID = "go-code-1"
        const val MAX_PRETOKEN = 128

        private fun isW(c: Int) = c == 32 || c == 9
        private fun isL(c: Int) = (c in 65..90) || (c in 97..122) || c == 95 || c >= 128
        private fun isP(c: Int) = !(isL(c) || (c in 48..57) || isW(c) || c == 10 || c == 13)

        /** Builds a tokenizer from merge rules (`left[r], right[r]` -> id `256 + r`). */
        fun of(left: IntArray, right: IntArray, specials: List<String>): BpeTokenizer {
            require(left.size == right.size)
            require(256 + left.size + specials.size <= 65535) { "vocabulary too large for 16-bit ids" }
            for (r in left.indices) require(left[r] in 0 until 256 + r && right[r] in 0 until 256 + r) { "merge $r references a later id" }
            return BpeTokenizer(left, right, specials)
        }

        fun load(stream: InputStream): BpeTokenizer = parse(stream.readBytes().toString(Charsets.UTF_8))

        fun load(path: java.nio.file.Path): BpeTokenizer = parse(java.nio.file.Files.readString(path))

        /** Text format (UTF-8, LF): see `cmlbpe.Vocab.save`. */
        fun parse(text: String): BpeTokenizer {
            val lines = text.split('\n')
            require(lines[0] == "$FORMAT_MAGIC $FORMAT_VERSION") { "not a CMLBPE v$FORMAT_VERSION file: ${lines[0]}" }
            val hdr = HashMap<String, String>()
            var i = 1
            while (!lines[i].startsWith("specials ")) {
                val sp = lines[i].indexOf(' ')
                hdr[lines[i].substring(0, sp)] = lines[i].substring(sp + 1); i++
            }
            val ns = lines[i].substring(9).trim().toInt(); i++
            require(hdr["pretokenizer"] == PRETOKENIZER_ID) { "unsupported pre-tokenizer ${hdr["pretokenizer"]}" }
            require(hdr["max_pretoken"]?.toInt() == MAX_PRETOKEN && hdr["base"]?.toInt() == 256)
            val specials = ArrayList<String>(ns)
            repeat(ns) { specials.add(lines[i++]) }
            val nm = hdr.getValue("merges").toInt()
            val left = IntArray(nm)
            val right = IntArray(nm)
            for (r in 0 until nm) {
                val l = lines[i++]
                val sp = l.indexOf(' ')
                left[r] = l.substring(0, sp).toInt(); right[r] = l.substring(sp + 1).toInt()
            }
            return of(left, right, specials)
        }
    }
}
