package io.github.completionml.core.nn

import io.github.completionml.core.bpe.BpeTokenizer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min

/**
 * Prompt assembly of the inline-completion eval, byte-for-byte the same as `eval_inline.build_prompt` (Python,
 * `~/work/nn/eval/eval_inline.py`): header `<|file_sep|> path \n`, prefix tail, and for SPM
 * `<|fim_prefix|> <|fim_suffix|> suffix <|fim_middle|> header prefix`. [cutPrefix] is the eval's 40 KB cut forward to
 * a line start; [NnCompletion] applies it, callers of [plain] / [spm] pass the prefix already cut.
 */
object InlinePrompt {
    /** The eval's prefix window: the last [maxBytes] of `before[0, end)`, moved forward to the next line start. */
    fun cutPrefix(before: ByteArray, end: Int, maxBytes: Int = 40_000): Int {
        var a = maxOf(0, end - maxBytes)
        if (a > 0) {
            var nl = a
            while (nl < end && before[nl] != '\n'.code.toByte()) nl++
            a = if (nl < end) nl + 1 else a
        }
        return a
    }

    fun header(tok: BpeTokenizer, path: ByteArray): IntArray = intArrayOf(tok.fileSep) + tok.encodeBytes(path + "\n".toByteArray())

    private fun tail(a: IntArray, n: Int): IntArray = if (n <= 0 || n >= a.size) a else a.copyOfRange(a.size - n, a.size)

    fun plain(tok: BpeTokenizer, path: ByteArray, prefix: ByteArray, ctx: Int): IntArray {
        val hdr = header(tok, path)
        return hdr + tail(tok.encodeBytes(prefix), ctx - hdr.size)
    }

    fun spm(tok: BpeTokenizer, path: ByteArray, prefix: ByteArray, suffix: ByteArray, ctx: Int, maxPrefix: Int, suffixTokens: Int): IntArray {
        val hdr = header(tok, path)
        val suf = tok.encodeBytes(suffix).let { if (it.size > suffixTokens) it.copyOf(suffixTokens) else it }
        val budget = minOf(maxPrefix, ctx - hdr.size - suf.size - 3)
        require(budget > 0) { "no room for the prefix: ctx $ctx, header ${hdr.size}, suffix ${suf.size}" }
        return intArrayOf(tok.fimPrefix, tok.fimSuffix) + suf + intArrayOf(tok.fimMiddle) + hdr + tail(tok.encodeBytes(prefix), budget)
    }

    /** Ids that end a line in the eval's stop rule: every token starting with LF/CR, and every special token. */
    fun stopIds(tok: BpeTokenizer): IntArray {
        val out = ArrayList<Int>()
        for (id in 0 until tok.specialBase) {
            val b = tok.tokenBytes(id)
            if (b.isNotEmpty() && (b[0] == '\n'.code.toByte() || b[0] == '\r'.code.toByte())) out += id
        }
        for (id in tok.specialBase until tok.vocabSize) out += id
        return out.toIntArray()
    }

    /** Stop set as a lookup table over the vocabulary. */
    fun stopTable(tok: BpeTokenizer): BooleanArray = BooleanArray(tok.vocabSize).also { t -> for (id in stopIds(tok)) t[id] = true }
}

/**
 * Byte-prefix table over the (non-special) vocabulary for constrained decoding: [allowed] lists the ids that may be
 * generated while the typed remainder `rem` still has to be reproduced — every token whose bytes start with `rem`, plus
 * every token that is a proper prefix of `rem` (the rest of `rem` then constrains the next step, see [consume]).
 * Built once per vocabulary (a sort of ~16k byte strings, a few ms); results are cached per remainder. Mirrors
 * `Tokenizer.allowed_ids` / `consume` in the Python harness.
 */
class VocabPrefixIndex(private val tok: BpeTokenizer) {
    private val bytes: Array<ByteArray> = Array(tok.specialBase) { tok.tokenBytes(it) }
    private val order: IntArray = (0 until tok.specialBase).sortedWith { a, b -> compare(bytes[a], bytes[b]) }.toIntArray()
    private val cache = HashMap<String, IntArray>()

    /** Ids allowed under the non-empty remainder [rem], ascending. Never empty: single bytes are always tokens. */
    fun allowed(rem: ByteArray): IntArray {
        require(rem.isNotEmpty())
        return synchronized(cache) {
            cache.getOrPut(String(rem, Charsets.ISO_8859_1)) {
                val out = ArrayList<Int>()
                var i = lowerBound(rem)
                while (i < order.size && startsWith(bytes[order[i]], rem)) { out += order[i]; i++ }
                for (k in 1 until rem.size) {
                    val pre = rem.copyOf(k)
                    var j = lowerBound(pre)
                    while (j < order.size && bytes[order[j]].contentEquals(pre)) { out += order[j]; j++ }
                }
                out.sorted().toIntArray()
            }
        }
    }

    /** Remainder left after generating [id] under [rem]: empty when the token covers it, else `rem` minus the token. */
    fun consume(rem: ByteArray, id: Int): ByteArray {
        val t = bytes[id]
        return if (startsWith(t, rem)) ByteArray(0) else rem.copyOfRange(t.size, rem.size)
    }

    fun tokenBytes(id: Int): ByteArray = bytes[id]

    private fun lowerBound(key: ByteArray): Int {
        var lo = 0; var hi = order.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (compare(bytes[order[mid]], key) < 0) lo = mid + 1 else hi = mid
        }
        return lo
    }

    companion object {
        /** Unsigned lexicographic order, a proper prefix first (Python `bytes` ordering). */
        fun compare(a: ByteArray, b: ByteArray): Int {
            val n = min(a.size, b.size)
            for (i in 0 until n) {
                val d = (a[i].toInt() and 0xff) - (b[i].toInt() and 0xff)
                if (d != 0) return d
            }
            return a.size - b.size
        }

        fun startsWith(s: ByteArray, prefix: ByteArray): Boolean {
            if (s.size < prefix.size) return false
            for (i in prefix.indices) if (s[i] != prefix[i]) return false
            return true
        }
    }
}

/**
 * Whole-line inline completion on top of [NnModel]: the plugin-facing entry point (docs/NN-COMPLETION-API.md).
 *
 * `complete(path, before, after, session)` takes the text before and after the caret and does, exactly like the Python
 * eval harness with `--heal boundary`:
 *  1. **Token healing** — the prompt is cut back to the last pre-token boundary at or before the caret
 *     ([BpeTokenizer.lastPreTokenBoundary] over the current line, newline byte included, up to the end of the line
 *     after the caret), so a caret inside a punctuation run (`foo(⟨⟩)`, `"x"⟨⟩)`, `foo()⟨⟩;`), after a typed space or
 *     inside an identifier never produces a token split the model has not seen in training. The bytes between the
 *     boundary and the caret (the *typed remainder*) constrain decoding: while a remainder is pending, only tokens
 *     that start with it or are a prefix of it ([VocabPrefixIndex.allowed]) may be generated, with probabilities from
 *     the masked softmax; the remainder is stripped from the returned text.
 *  2. **Prompt** — SPM (`<|fim_prefix|><|fim_suffix|> suffix <|fim_middle|> <|file_sep|> path\n prefix`) or plain, with
 *     the eval budgets (prefix ≤ 40 KB cut to a line start, ≤ [Options.maxPrefix] tokens; suffix = the text after the
 *     current line, ≤ 16 KB, ≤ [Options.suffixTokens] tokens; the whole prompt ≤ [Options.ctx]).
 *  3. **Greedy decode** to the end of the line (any token starting with LF/CR or a special token stops), at most
 *     [Options.maxNew] tokens, with the **repetition guard**: a BPE n-gram (n ≤ 4, ≥ 4 bytes) repeated three times in a
 *     row stops decoding ([Stop.REPEAT]).
 *  4. **Show policy** — `confProd` = product of the probabilities of all generated tokens and of the newline that ended
 *     the line, `confMin` = the smallest of them; [Result.show] = `confProd ≥ showThreshold` and the suggestion is
 *     not punctuation only (no letter, digit, underscore, non-ASCII byte or quote: `);`, `}`, `)]`) and decoding was
 *     not stopped by the repetition guard or the token limit.
 *
 * The [NnSession] keeps the KV cache between calls (prefix reuse: a subsequent call recomputes only the tokens after
 * the longest common prefix). One session per editor / thread; the completion object itself is immutable.
 */
class NnCompletion(val model: NnModel, val tok: BpeTokenizer, val options: Options = Options()) {
    enum class Mode { SPM, PLAIN }
    enum class Stop { NEWLINE, SPECIAL, LIMIT, REPEAT }

    data class Options(
        val mode: Mode = Mode.SPM,
        /** Prompt token budget incl. header and suffix; must leave `maxNew` tokens below the model's maxContext. */
        val ctx: Int = 2000,
        val maxPrefix: Int = 1450,
        val suffixTokens: Int = 512,
        val maxNew: Int = 48,
        val prefixBytes: Int = 40_000,
        val suffixBytes: Int = 16_000,
        val heal: Boolean = true,
        /** Where healing cuts the prompt when the caret sits right after a word (see [HealMode]). */
        val healMode: HealMode = HealMode.WORD_EOL,
        val repGuard: Boolean = true,
        /** Gate on `confProd`; 0.7 shows ~26 % of Go / 14 % of C# positions with 93–94 % exact lines, 0.8 ~20 % / 10 % at 95–97 % (CHANGELOG e18). */
        val showThreshold: Double = 0.7,
        val suppressPunctOnly: Boolean = true,
        /** Drop the suggestion's tail that repeats the closers already after the caret (`return len(⟨⟩)`: the editor paired the `)`). */
        val trimClosersAfterCaret: Boolean = true,
    )

    /**
     * Healing at a caret that is itself a pre-token boundary because a word ends there (`return le⟨⟩`). [BOUNDARY]
     * leaves it (the model continues a finished ` le`); [WORD] heals from the start of that word, so the model may
     * choose ` len` instead; [WORD_EOL] does so only when the rest of the line after the caret is empty, whitespace or
     * closers — the typing situation — and leaves `o.Get⟨⟩.Name` alone.
     */
    enum class HealMode { BOUNDARY, WORD, WORD_EOL }

    class Result(
        /** Suggested text to insert at the caret (UTF-8 bytes; the typed remainder already stripped). */
        val text: ByteArray,
        /** Bytes between the healed boundary and the caret that the generation was constrained to reproduce. */
        val typed: ByteArray,
        /** The prompt that was fed to the model. */
        val prompt: IntArray,
        /** Generated token ids (including those that reproduce the typed remainder) and their log-probabilities. */
        val tokens: IntArray,
        val logProbs: FloatArray,
        val stop: Stop,
        /** Log-probability of the token that ended the line (NaN for [Stop.LIMIT] / [Stop.REPEAT]). */
        val stopLogProb: Float,
        val confProd: Double,
        val confMin: Double,
        val punctOnly: Boolean,
        /** True when decoding was cut by the repetition guard or the token limit. */
        val repeated: Boolean,
        val show: Boolean,
        /** True if the model did not reproduce the typed remainder (only possible when the token limit cut it). */
        val healMiss: Boolean,
    ) {
        val textString: String get() = String(text, Charsets.UTF_8)
    }

    val prefixIndex = VocabPrefixIndex(tok)
    private val stop = InlinePrompt.stopTable(tok)

    init {
        require(tok.vocabSize == model.config.vocabSize) { "tokenizer (${tok.vocabSize}) and model (${model.config.vocabSize}) vocabularies differ" }
        require(options.ctx + options.maxNew <= model.config.maxContext) { "ctx ${options.ctx} + maxNew ${options.maxNew} > maxContext ${model.config.maxContext}" }
    }

    /** Where the current line starts in [before] (after the previous LF) and ends in [after] (at the first CR/LF). */
    private fun lineStart(before: ByteArray): Int {
        var i = before.size - 1
        while (i >= 0 && before[i] != '\n'.code.toByte()) i--
        return i + 1
    }

    private fun lineEnd(after: ByteArray): Int {
        var i = 0
        while (i < after.size && after[i] != '\n'.code.toByte() && after[i] != '\r'.code.toByte()) i++
        return i
    }

    /**
     * Healed cut of [before]: the last pre-token boundary at or before the caret, scanning the current line from the
     * newline byte before it (or the file start) to the end of the line in [after].
     */
    fun healedBoundary(before: ByteArray, after: ByteArray): Int {
        val bol = lineStart(before)
        val from = if (bol > 0) bol - 1 else 0
        val restOfLine = lineEnd(after)
        val line = ByteArray(before.size - from + restOfLine)
        System.arraycopy(before, from, line, 0, before.size - from)
        System.arraycopy(after, 0, line, before.size - from, restOfLine)
        val caret = before.size - from
        var boundary = tok.lastPreTokenBoundary(line, 0, caret, line.size)
        if (options.healMode != HealMode.BOUNDARY && boundary == caret && caret > 0 && isWordByte(line[caret - 1]) &&
            (options.healMode == HealMode.WORD || (0 until restOfLine).all { isCloserOrSpace(after[it]) })
        ) boundary = tok.lastPreTokenBoundary(line, 0, caret - 1, line.size)   // the start of the word's pre-token
        return from + boundary
    }

    /** Letters, digits, `_` and non-ASCII bytes: the pre-tokenizer's word classes (`cmlbpe.pretokenize_reference`). */
    private fun isWordByte(b: Byte): Boolean {
        val c = b.toInt() and 0xff
        return c in 65..90 || c in 97..122 || c == 95 || c >= 128 || c in 48..57
    }

    private fun isCloserOrSpace(b: Byte): Boolean = b.toInt().toChar() in CLOSERS

    /** The tail of [text] that repeats the start of [afterLine] and consists of closers only is dropped. */
    fun trimClosers(text: ByteArray, afterLine: ByteArray): ByteArray {
        var k = min(text.size, afterLine.size)
        while (k > 0) {
            var ok = true
            for (i in 0 until k) if (!isCloserOrSpace(afterLine[i]) || text[text.size - k + i] != afterLine[i]) { ok = false; break }
            if (ok) return text.copyOfRange(0, text.size - k)
            k--
        }
        return text
    }

    /** The prompt for a caret with `before` cut at [prefixEnd]. */
    fun buildPrompt(path: ByteArray, before: ByteArray, prefixEnd: Int, after: ByteArray): IntArray {
        val a = InlinePrompt.cutPrefix(before, prefixEnd, options.prefixBytes)
        val prefix = before.copyOfRange(a, prefixEnd)
        return when (options.mode) {
            Mode.PLAIN -> InlinePrompt.plain(tok, path, prefix, options.ctx)
            Mode.SPM -> {
                val s0 = lineEnd(after)
                val suffix = after.copyOfRange(s0, min(after.size, s0 + options.suffixBytes))
                InlinePrompt.spm(tok, path, prefix, suffix, options.ctx, options.maxPrefix, options.suffixTokens)
            }
        }
    }

    /**
     * Completes the line at the caret. [path] is the file path relative to the project root (UTF-8), [before] the
     * text before the caret (a tail of ≥ 40 KB is enough), [after] the text after it (the rest of the line and
     * ≥ 16 KB beyond; may be empty).
     */
    fun complete(path: ByteArray, before: ByteArray, after: ByteArray, session: NnSession): Result {
        val boundary = if (options.heal) healedBoundary(before, after) else before.size
        val typed = before.copyOfRange(boundary, before.size)
        val prompt = buildPrompt(path, before, boundary, after)
        return decode(prompt, typed, session, after.copyOfRange(0, lineEnd(after)))
    }

    /**
     * Greedy decode of [prompt] with the typed-remainder constraint, the stop rule and the repetition guard; [afterLine]
     * (the rest of the current line after the caret) only feeds [Options.trimClosersAfterCaret].
     */
    fun decode(prompt: IntArray, typed: ByteArray, session: NnSession, afterLine: ByteArray = ByteArray(0)): Result {
        val maxNew = options.maxNew
        val gen = IntArray(maxNew); val lp = FloatArray(maxNew)
        var n = 0
        var rem = typed
        var stopKind = Stop.LIMIT
        var stopLp = Float.NaN
        var logits = session.prefill(prompt)
        for (step in 0..maxNew) {
            val t: Int; val l: Float
            if (rem.isNotEmpty()) {
                val allowed = prefixIndex.allowed(rem)
                t = argmaxOver(logits, allowed)
                l = logProbOver(logits, allowed, t)
            } else {
                t = Sampler.argmax(logits)
                l = Sampler.logProb(logits, t)
            }
            if (stop[t]) { stopKind = if (tok.isSpecial(t)) Stop.SPECIAL else Stop.NEWLINE; stopLp = l; break }
            if (step == maxNew) break
            gen[n] = t; lp[n] = l; n++
            if (rem.isNotEmpty()) rem = prefixIndex.consume(rem, t)
            if (options.repGuard && repeated(gen, n)) { stopKind = Stop.REPEAT; break }
            if (session.length >= model.config.maxContext) break
            logits = session.decode(t)
        }
        val tokens = gen.copyOf(n); val logProbs = lp.copyOf(n)
        var sum = 0.0; var mn = 0.0
        for (x in logProbs) { sum += x; if (x < mn) mn = x.toDouble() }
        if (!stopLp.isNaN()) { sum += stopLp; if (stopLp < mn) mn = stopLp.toDouble() }
        val hasAny = n > 0 || !stopLp.isNaN()
        val confProd = if (hasAny) exp(sum) else 0.0
        val confMin = if (hasAny) exp(mn) else 0.0
        val raw = tok.decodeBytes(tokens)
        val healMiss = typed.isNotEmpty() && !VocabPrefixIndex.startsWith(raw, typed)
        var text = if (typed.isNotEmpty() && !healMiss) raw.copyOfRange(typed.size, raw.size) else raw
        if (options.trimClosersAfterCaret && afterLine.isNotEmpty()) text = trimClosers(text, afterLine)
        val punct = punctOnly(text)
        val rep = stopKind == Stop.REPEAT || stopKind == Stop.LIMIT
        val show = confProd >= options.showThreshold && !(options.suppressPunctOnly && punct) && !rep && text.isNotEmpty()
        return Result(text, typed, prompt, tokens, logProbs, stopKind, stopLp, confProd, confMin, punct, rep, show, healMiss)
    }

    /** Repetition guard over the generated ids `gen[0, n)`: the last 3·k tokens are three copies of a k-gram (k ≤ 4)
     * of at least 4 bytes (so digit runs like `000` and bracket pairs are exempt). Mirrors `eval_inline.repetition`. */
    fun repeated(gen: IntArray, n: Int): Boolean {
        for (k in 1..4) {
            if (n < 3 * k) return false
            var same = true
            var bytes = 0
            for (i in 0 until k) {
                val a = gen[n - k + i]
                if (a != gen[n - 2 * k + i] || a != gen[n - 3 * k + i]) { same = false; break }
                bytes += prefixIndex.tokenBytes(a).size
            }
            if (same && bytes >= 4) return true
        }
        return false
    }

    companion object {
        /** No letter, digit, underscore, non-ASCII byte or quote: closers such as `);`, `}`, `)]` (or nothing at all). */
        /** Closers and the whitespace between them, as the editor pairs them: `)` `]` `}` `>` `;` `,` quotes, backtick. */
        const val CLOSERS = " \t\r)]}>;,\"'`"

        fun punctOnly(text: ByteArray): Boolean {
            for (b in text) {
                val c = b.toInt() and 0xff
                if (c >= 0x80 || c in 0x30..0x39 || c in 0x41..0x5a || c in 0x61..0x7a || c == '_'.code || c == '"'.code || c == '\''.code || c == '`'.code) return false
            }
            return true
        }

        fun argmaxOver(logits: FloatArray, ids: IntArray): Int {
            var best = ids[0]
            for (i in 1 until ids.size) if (logits[ids[i]] > logits[best]) best = ids[i]
            return best
        }

        /** log softmax restricted to [ids] (the masked softmax of the Python harness). */
        fun logProbOver(logits: FloatArray, ids: IntArray, token: Int): Float {
            var mx = Float.NEGATIVE_INFINITY
            for (i in ids) if (logits[i] > mx) mx = logits[i]
            var s = 0.0
            for (i in ids) s += exp((logits[i] - mx).toDouble())
            return (logits[token] - mx - ln(s)).toFloat()
        }
    }
}
