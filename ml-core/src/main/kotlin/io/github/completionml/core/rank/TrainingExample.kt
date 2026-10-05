package io.github.completionml.core.rank

import io.github.completionml.core.spi.ContextKind
import java.io.DataInput
import java.io.DataOutput

/**
 * One completion list. Stored compactly: the *base* feature vector per candidate plus the context kind; the full
 * schema vector (base ++ base·onehot(kind), see [FeatureSchema.expand]) is materialised on demand into a scratch buffer.
 * Memory per candidate: 4·|BASE| bytes instead of 4·|schema|.
 */
class TrainingExample(
    val kind: ContextKind,
    val base: Array<FloatArray>,
    val chosen: Int,
    val candidateNames: Array<String>? = null,   // for debugging/eval reports; not serialised
) {
    val size get() = base.size

    /** Expands candidate [c] into [out] (length = full schema size). */
    fun expand(c: Int, out: FloatArray) = FeatureSchema.expand(base[c], kind, out)

    /** Base feature [index] of candidate [c] (for single-feature baselines). */
    fun baseFeature(c: Int, index: Int): Float = base[c][index]

    fun write(out: DataOutput) {
        out.writeByte(kind.ordinal)
        out.writeInt(base.size)
        out.writeInt(chosen)
        out.writeInt(base[0].size)
        for (f in base) for (x in f) out.writeFloat(x)
    }

    companion object {
        fun read(inp: DataInput): TrainingExample {
            val kind = ContextKind.values()[inp.readByte().toInt()]
            val n = inp.readInt(); val chosen = inp.readInt(); val dim = inp.readInt()
            val base = Array(n) { FloatArray(dim) }
            for (f in base) for (i in 0 until dim) f[i] = inp.readFloat()
            return TrainingExample(kind, base, chosen)
        }
    }
}
