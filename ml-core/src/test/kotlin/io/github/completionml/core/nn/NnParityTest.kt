package io.github.completionml.core.nn

import io.github.completionml.core.bpe.BpeTokenizer
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Parity with the PyTorch model on the real trained weights (docs/NN-PARITY.md). Needs the fixture from
 * `~/work/nn/eval/make_parity.py`, the exported model and the vocabulary (see [NnParity] for the locations);
 * skipped when any of them is absent. The behavioural subset is limited to keep the test under a minute per kernel;
 * the full run is `NnParity.main`. Thresholds are the measured figures with headroom (first-divergence and
 * exact-match rates are inherently noisy: a single near-tie flips a whole line).
 */
class NnParityTest {
    @Test fun kotlinInferenceMatchesPyTorchOnTrainedWeights() {
        assumeTrue(NnParity.available(), "parity fixture / model / vocab not present")
        val fx = NnParity.load(behavLimit = (System.getProperty("completionml.nn.parity.behav") ?: "40").toInt())
        val weights = NnFormat.read(NnParity.defaultModel)
        val tok = BpeTokenizer.load(NnParity.defaultVocab.toPath())
        assertEquals(weights.meta["tokenizerSha256"], NnParity.sha256(NnParity.defaultVocab), "model was exported with a different vocabulary")
        assertEquals(0, NnParity.tokenizerParity(tok, fx, System.out), "prompt ids differ from Python")

        val kernels = listOfNotNull(
            ScalarNnKernels,
            io.github.completionml.core.nn.native.NativeNnKernels.loadOrNull(io.github.completionml.core.nn.native.NativeNnKernels.Mode.F32),
            io.github.completionml.core.nn.native.NativeNnKernels.loadOrNull(io.github.completionml.core.nn.native.NativeNnKernels.Mode.Q8),
        )
        for (k in kernels) {
            val r = NnParity.runKernel(weights, k, 4, tok, fx, null, System.out)
            val q8 = k.name.endsWith("q8")
            // vs the fp32 model with the very same int8 weights: kernels only differ by summation order (f32) or
            // activation quantisation (q8)
            assertTrue(r.vsFq.maxAbs < (if (q8) 1.0f else 0.05f), "${k.name}: max |dlogit| vs fake-quantised model ${r.vsFq.maxAbs}")
            assertTrue(r.vsFq.argmaxAgree >= r.vsFq.n - (if (q8) 2 else 0), "${k.name}: argmax agreement ${r.vsFq.argmaxAgree}/${r.vsFq.n}")
            assertTrue(r.vsFq.lpRmse < (if (q8) 0.05 else 0.005), "${k.name}: log-prob rmse ${r.vsFq.lpRmse}")
            // vs the float model: int8 weight noise only
            assertTrue(r.vsFloat.argmaxAgree >= r.vsFloat.n - 2, "${k.name}: argmax agreement vs float ${r.vsFloat.argmaxAgree}/${r.vsFloat.n}")
            // behaviour: same generated line as the fp32 fake-quantised reference in the vast majority of positions
            assertTrue(r.behav.sameAsFq >= r.behav.n * (if (q8) 0.85 else 0.95), "${k.name}: ${r.behav.sameAsFq}/${r.behav.n} lines identical to the reference")
        }
    }
}
