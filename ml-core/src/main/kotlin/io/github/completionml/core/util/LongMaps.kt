package io.github.completionml.core.util

/** Open-addressing hash map Long -> Int with linear probing. Key 0 is reserved as "empty"; callers must not use it. */
class LongIntMap(expected: Int = 1024) {
    var keys = LongArray(capacityFor(expected)); private set
    var values = IntArray(keys.size); private set
    var size = 0; private set
    private var mask = keys.size - 1

    fun get(key: Long, default: Int = 0): Int {
        var i = slot(key)
        while (true) {
            val k = keys[i]
            if (k == key) return values[i]
            if (k == 0L) return default
            i = (i + 1) and mask
        }
    }

    fun containsKey(key: Long): Boolean {
        var i = slot(key)
        while (true) { val k = keys[i]; if (k == key) return true; if (k == 0L) return false; i = (i + 1) and mask }
    }

    fun addTo(key: Long, delta: Int) {
        var i = slot(key)
        while (true) {
            val k = keys[i]
            if (k == key) { values[i] += delta; return }
            if (k == 0L) { keys[i] = key; values[i] = delta; size++; if (size * 10 > keys.size * 7) grow(); return }
            i = (i + 1) and mask
        }
    }

    fun put(key: Long, value: Int) {
        var i = slot(key)
        while (true) {
            val k = keys[i]
            if (k == key) { values[i] = value; return }
            if (k == 0L) { keys[i] = key; values[i] = value; size++; if (size * 10 > keys.size * 7) grow(); return }
            i = (i + 1) and mask
        }
    }

    inline fun forEach(action: (Long, Int) -> Unit) {
        val ks = keys; val vs = values
        for (i in ks.indices) { val k = ks[i]; if (k != 0L) action(k, vs[i]) }
    }

    private fun slot(key: Long) = (mix(key) and mask.toLong()).toInt()
    private fun grow() {
        val ok = keys; val ov = values
        keys = LongArray(ok.size * 2); values = IntArray(keys.size); mask = keys.size - 1; size = 0
        for (i in ok.indices) if (ok[i] != 0L) put(ok[i], ov[i])
    }
}

/** Open-addressing hash map Long -> Float. Key 0 reserved. Missing keys read as NaN. */
class LongFloatMap(expected: Int = 1024) : FloatLookup {
    var keys = LongArray(capacityFor(expected)); private set
    var values = FloatArray(keys.size); private set
    override var size = 0; private set
    private var mask = keys.size - 1

    override fun get(key: Long): Float {
        var i = slot(key)
        while (true) { val k = keys[i]; if (k == key) return values[i]; if (k == 0L) return Float.NaN; i = (i + 1) and mask }
    }

    fun put(key: Long, value: Float) {
        var i = slot(key)
        while (true) {
            val k = keys[i]
            if (k == key) { values[i] = value; return }
            if (k == 0L) { keys[i] = key; values[i] = value; size++; if (size * 10 > keys.size * 7) grow(); return }
            i = (i + 1) and mask
        }
    }

    inline fun forEach(action: (Long, Float) -> Unit) {
        val ks = keys; val vs = values
        for (i in ks.indices) { val k = ks[i]; if (k != 0L) action(k, vs[i]) }
    }

    private fun slot(key: Long) = (mix(key) and mask.toLong()).toInt()
    private fun grow() {
        val ok = keys; val ov = values
        keys = LongArray(ok.size * 2); values = FloatArray(keys.size); mask = keys.size - 1; size = 0
        for (i in ok.indices) if (ok[i] != 0L) put(ok[i], ov[i])
    }
}

internal fun capacityFor(expected: Int): Int {
    var c = 16
    while (c.toLong() * 7 < expected.toLong() * 10) c = c shl 1
    return c
}

/** splitmix64 finaliser. */
fun mix(x: Long): Long {
    var z = x
    z = (z xor (z ushr 30)) * -0x40a7b892e31b1a47L
    z = (z xor (z ushr 27)) * -0x6b2fb644ecceee15L
    return z xor (z ushr 31)
}

/** Hash of an n-gram given as token ids; never returns 0. Order matters; `extend(h, w)` appends word `w` to context hash `h`. */
object NgramHash {
    const val EMPTY: Long = -0x61c8864680b583ebL // 0x9E3779B97F4A7C15
    fun extend(context: Long, word: Int): Long {
        val h = mix(context xor ((word.toLong() + 0x632BE59BD9B4E019L) * 0x5851F42D4C957F2DL))
        return if (h == 0L) 1L else h
    }
    fun of(ids: IntArray, from: Int, to: Int): Long {
        var h = EMPTY
        for (i in from until to) h = extend(h, ids[i])
        return h
    }
}
