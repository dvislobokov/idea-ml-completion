package io.github.completionml.core.nn

/**
 * Hyper-parameters of our decoder-only transformer (LLaMA-style: pre-norm RMSNorm, RoPE, SwiGLU MLP, optional GQA,
 * no biases). Stored in the `.cml` neural model meta block as `key=value` lines (see docs/NN-FORMAT.md).
 */
data class NnConfig(
    val vocabSize: Int,
    val dModel: Int,
    val nLayers: Int,
    val nHeads: Int,
    val nKvHeads: Int = nHeads,
    val ffnDim: Int,
    val maxContext: Int = 2048,
    val ropeTheta: Float = 10000f,
    val normEps: Float = 1e-5f,
    val tiedEmbeddings: Boolean = true,
    val activation: String = "swiglu",
    val norm: String = "rmsnorm",
) {
    init {
        require(vocabSize > 0 && dModel > 0 && nLayers > 0 && ffnDim > 0 && maxContext > 0) { "bad config $this" }
        require(dModel % nHeads == 0) { "dModel $dModel not divisible by nHeads $nHeads" }
        require(nHeads % nKvHeads == 0) { "nHeads $nHeads not divisible by nKvHeads $nKvHeads" }
        require(headDim % 2 == 0) { "head dim must be even for RoPE" }
        require(activation == "swiglu") { "unsupported activation '$activation'" }
        require(norm == "rmsnorm") { "unsupported norm '$norm'" }
    }

    val headDim: Int get() = dModel / nHeads
    val kvDim: Int get() = nKvHeads * headDim

    /** Exact parameter count (weights only, no buffers). Tied embeddings are counted once. */
    val paramCount: Long
        get() {
            val d = dModel.toLong()
            val perLayer = d * d + 2 * d * kvDim + d * d + 3 * d * ffnDim + 2 * d
            val emb = vocabSize.toLong() * d
            return emb + nLayers * perLayer + d + (if (tiedEmbeddings) 0 else emb)
        }

    /** Parameters excluding the embedding / output matrices (the part that runs once per prompt token). */
    val nonEmbeddingParams: Long
        get() = paramCount - vocabSize.toLong() * dModel * (if (tiedEmbeddings) 1 else 2)

    fun toMeta(): Map<String, String> = linkedMapOf(
        "vocabSize" to "$vocabSize", "dModel" to "$dModel", "nLayers" to "$nLayers", "nHeads" to "$nHeads",
        "nKvHeads" to "$nKvHeads", "ffnDim" to "$ffnDim", "maxContext" to "$maxContext", "ropeTheta" to "$ropeTheta",
        "normEps" to "$normEps", "tiedEmbeddings" to "$tiedEmbeddings", "activation" to activation, "norm" to norm,
    )

    companion object {
        fun fromMeta(m: Map<String, String>): NnConfig {
            fun int(k: String) = (m[k] ?: error("meta: missing '$k'")).toInt()
            return NnConfig(
                vocabSize = int("vocabSize"), dModel = int("dModel"), nLayers = int("nLayers"), nHeads = int("nHeads"),
                nKvHeads = int("nKvHeads"), ffnDim = int("ffnDim"), maxContext = int("maxContext"),
                ropeTheta = m["ropeTheta"]?.toFloat() ?: 10000f, normEps = m["normEps"]?.toFloat() ?: 1e-5f,
                tiedEmbeddings = m["tiedEmbeddings"]?.toBooleanStrict() ?: true,
                activation = m["activation"] ?: "swiglu", norm = m["norm"] ?: "rmsnorm",
            )
        }
    }
}
