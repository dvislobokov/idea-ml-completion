package io.github.completionml.core.nn

import io.github.completionml.core.bpe.BpeTokenizer
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The real C# network of `models/` (env `CML_NN_MODEL_CS` / `CML_BPE_VOCAB_CS`) on the positions of the healing-v2 report
 * (2026-10-10): the `Greeter.cs` of the .NET plugin's playground, where the caret-fine-tuned models glued `newHelloReply`,
 * and a `return ⟨⟩` line with the typed space. Skipped without the files. A fine-tuned model trained on prompts of the old
 * rule fails the first test until it is retrained on data built with `heal_trailing_ws` — that is the point of the test.
 */
class NnCSharpModelTest {
    companion object {
        val model: File get() = NnParity.firstExisting("CML_NN_MODEL_CS", "../models/cs-nn-50m-e3-lr2e3.cml")
        val vocab: File get() = NnParity.firstExisting("CML_BPE_VOCAB_CS", "../models/cs-16384.bpe")

        /** `debug-playground/Grpc/Greeter.cs` of idea-dotnet-support as it was in the editor on 2026-10-10 (`Task.FromResult(⟨⟩)` on line 17). */
        val greeter: String by lazy {
            checkNotNull(NnCSharpModelTest::class.java.classLoader.getResourceAsStream("nn/Greeter.cs")).use { String(it.readBytes(), Charsets.UTF_8) }.replace("\r\n", "\n")
        }
    }

    private fun b(s: String) = s.toByteArray(Charsets.UTF_8)

    private fun withNetwork(block: (NnCompletion, NnSession) -> Unit) {
        assumeTrue(model.isFile && vocab.isFile, "C# model / vocabulary not present: $model, $vocab")
        val tok = BpeTokenizer.load(vocab.toPath())
        NnModel(NnFormat.read(model), 4).use { m ->
            val c = NnCompletion(m, tok)
            m.newSession(4096).use { s -> block(c, s) }
        }
    }

    @Test fun newKeepsItsSpaceAtFromResult() = withNetwork { c, s ->
        val caret = greeter.indexOf("Task.FromResult(") + "Task.FromResult(".length
        val r = c.complete(b("Grpc/Greeter.cs"), b(greeter.substring(0, caret)), b(greeter.substring(caret)), s)
        println("NnCSharpModelTest: Task.FromResult(⟨⟩) -> '${r.textString}' typed '${String(r.typed)}' confProd ${"%.3f".format(r.confProd)} (${model.name})")
        assertEquals("(", String(r.typed))                                  // the run `()` is one pre-token; no whitespace before the caret
        assertTrue(r.textString.startsWith("new HelloReply"), "the space after `new` is lost: '${r.textString}'")
    }

    @Test fun theTypedSpaceAfterReturnIsNotDoubled() = withNetwork { c, s ->
        val head = "using System.Collections.Generic;\n\nnamespace Demo;\n\npublic sealed class Order\n{\n    private readonly List<string> items = new();\n\n" +
            "    public int Count()\n    {\n        return "
        val r = c.complete(b("Demo/Order.cs"), b(head), b("\n    }\n}\n"), s)
        println("NnCSharpModelTest: return ⟨⟩ -> '${r.textString}' typed '${String(r.typed)}' confProd ${"%.3f".format(r.confProd)}")
        assertEquals(" ", String(r.typed))                                  // healing v2: the space is typed, the prompt ends at `return`
        assertTrue(String(c.tok.tokenBytes(r.prompt.last())).endsWith("return"), "the prompt should end at `return`")
        assertFalse(r.healMiss)
        assertTrue(r.text.isNotEmpty() && r.text[0] != ' '.code.toByte(), "the inserted text must not start with another space: '${r.textString}'")
        assertTrue(String(c.tok.tokenBytes(r.tokens[0])).startsWith(" "), "the first generated token carries the space")
    }
}
