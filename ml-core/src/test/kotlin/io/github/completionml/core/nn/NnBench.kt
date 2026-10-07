package io.github.completionml.core.nn

import java.io.File
import java.util.Random

/**
 * Inference benchmark on random int8 weights (test-scope main; run with plain `java`, classpath from
 * `./gradlew :ml-core:printBenchClasspath`).
 *
 *   NnBenchKt --config S|M|L --threads 8 [--prompt 512] [--gen 20] [--kernels best|scalar] [--dir /tmp/nn] [--runs 10]
 *
 * Prints one `RESULT` line (tab-separated) and the per-run latencies of the first runs (JIT warm-up).
 */
object NnBench {
    val configs = mapOf(
        "S" to NnConfig(vocabSize = 32000, dModel = 384, nLayers = 8, nHeads = 6, nKvHeads = 2, ffnDim = 1536, maxContext = 2048),
        "M" to NnConfig(vocabSize = 32000, dModel = 512, nLayers = 8, nHeads = 8, nKvHeads = 4, ffnDim = 2048, maxContext = 2048),
        "L" to NnConfig(vocabSize = 32000, dModel = 768, nLayers = 10, nHeads = 12, nKvHeads = 4, ffnDim = 2560, maxContext = 2048),
    )

    @JvmStatic fun main(args: Array<String>) {
        val opt = args.toList().chunked(2).associate { (k, v) -> k.removePrefix("--") to v }
        val name = opt["config"] ?: "S"
        val cfg0 = configs.getValue(name)
        val threads = (opt["threads"] ?: "1").toInt()
        val promptLen = (opt["prompt"] ?: "512").toInt()
        val gen = (opt["gen"] ?: "20").toInt()
        val runs = (opt["runs"] ?: "10").toInt()
        val reuseNew = (opt["reuse"] ?: "8").toInt()
        val dir = File(opt["dir"] ?: System.getProperty("java.io.tmpdir"))
        val kernels = when (opt["kernels"] ?: "best") {
            "scalar" -> ScalarNnKernels
            "native-f32" -> io.github.completionml.core.nn.native.NativeNnKernels.loadOrNull(io.github.completionml.core.nn.native.NativeNnKernels.Mode.F32)
                ?: error("native: ${io.github.completionml.core.nn.native.NativeLib.status}")
            "native-q8" -> io.github.completionml.core.nn.native.NativeNnKernels.loadOrNull(io.github.completionml.core.nn.native.NativeNnKernels.Mode.Q8)
                ?: error("native: ${io.github.completionml.core.nn.native.NativeLib.status}")
            else -> NnKernels.best()
        }
        System.err.println("native lib: ${io.github.completionml.core.nn.native.NativeLib.status}")
        if (opt["kernel"] != null) { for (nn in opt["kernel"]!!.split(",")) kernelBench(kernels, nn.toInt()); return }
        if (opt["info"] != null) {
            for ((n, c) in configs) println("$n $c params=${c.paramCount} nonEmb=${c.nonEmbeddingParams}")
            return
        }

        val rssBase = rssMb()
        // --file <model.cml>: real weights (config from the file); --prompt-file <parity/prompts.bin>: a real prompt
        val real = opt["file"]?.let(::File)
        val file = real ?: File(dir, "bench-$name.cml")
        if (!file.exists()) NnFormat.write(file, cfg0, mapOf("corpusId" to "random"), NnTestModels.random(cfg0, 1))
        val tLoad0 = System.nanoTime()
        val weights = NnFormat.read(file)
        val loadMmapMs = (System.nanoTime() - tLoad0) / 1e6
        val tTouch0 = System.nanoTime()
        var touched = 0L
        for (t in weights.tensors.values) if (t is NnTensor.Q8) { val b = t.data; var i = 0; while (i < b.remaining()) { touched += b.get(b.position() + i); i += 4096 } }
        val touchMs = (System.nanoTime() - tTouch0) / 1e6
        val tLoad1 = System.nanoTime()
        NnFormat.read(file.inputStream(), "stream").let { touched += it.tensors.size }
        val loadStreamMs = (System.nanoTime() - tLoad1) / 1e6
        val cfg = weights.config
        val rnd = Random(42)
        val prompt = IntArray(promptLen) { rnd.nextInt(cfg.vocabSize) }
        opt["prompt-file"]?.let { pf ->
            val fx = NnParity.load(File(pf).parentFile, 0)
            val src = fx.prompts.filter { it.ids.size >= promptLen }.minByOrNull { it.ids.size } ?: fx.prompts.maxByOrNull { it.ids.size }!!
            for (i in prompt.indices) prompt[i] = src.ids[i % src.ids.size]
        }
        val tPrep0 = System.nanoTime()
        (kernels as? io.github.completionml.core.nn.native.NativeNnKernels)?.prepare(weights.tensors.values.filterIsInstance<NnTensor.Q8>())
        val prepMs = (System.nanoTime() - tPrep0) / 1e6

        NnModel(weights, threads, kernels).use { model ->
            val session = model.newSession(capacity = promptLen + gen + 8)
            val os = java.lang.management.ManagementFactory.getOperatingSystemMXBean() as com.sun.management.OperatingSystemMXBean
            fun line(): DoubleArray {
                val cpu0 = os.processCpuTime
                val t0 = System.nanoTime()
                var logits = session.prefill(prompt.copyOf().also { it[0] = rnd.nextInt(cfg.vocabSize) }) // defeat prefix reuse
                val t1 = System.nanoTime()
                for (i in 1 until gen) logits = session.decode(Sampler.argmax(logits))
                val t2 = System.nanoTime()
                val cpu2 = os.processCpuTime
                // incremental: the same prompt with the last `reuseNew` tokens changed → only those are recomputed
                val p2 = prompt.copyOf().also { for (k in it.size - reuseNew until it.size) it[k] = rnd.nextInt(cfg.vocabSize) }
                session.truncate(0); session.prefill(prompt) // cache = prompt
                val t3 = System.nanoTime()
                logits = session.prefill(p2)
                for (i in 1 until gen) logits = session.decode(Sampler.argmax(logits))
                val t4 = System.nanoTime()
                return doubleArrayOf((t1 - t0) / 1e6, (t2 - t1) / 1e6 / (gen - 1), (t2 - t0) / 1e6, (t4 - t3) / 1e6, (cpu2 - cpu0) / 1e6)
            }
            if (NnSession.Profile.enabled) {   // one profiled prefill + 19 decodes after warm-up
                repeat(3) { line() }
                NnSession.Profile.reset()
                session.truncate(0); var lg = session.prefill(prompt.copyOf().also { it[0] = 7 })
                println("PROFILE prefill: ${NnSession.Profile}")
                NnSession.Profile.reset()
                for (i in 1 until gen) lg = session.decode(Sampler.argmax(lg))
                println("PROFILE decode x${gen - 1}: ${NnSession.Profile}")
            }
            val all = ArrayList<DoubleArray>()
            val start = System.nanoTime()
            // warm-up: at least 3 runs, until 10 runs or 40 s; then `runs` measured runs (or 60 s)
            while (all.size < 3 || (all.size < 10 && System.nanoTime() - start < 40e9)) all += line()
            val warm = all.size
            val mStart = System.nanoTime()
            while (all.size < warm + runs && (all.size < warm + 3 || System.nanoTime() - mStart < 60e9)) all += line()
            val steady = all.subList(warm, all.size)
            fun med(k: Int) = steady.map { it[k] }.sorted()[steady.size / 2]
            val heap = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1e6
            println("RESULT\tconfig=$name\tparams=${cfg.paramCount}\tkernels=${kernels.name}\tthreads=$threads" +
                "\tprefill_ms=%.1f\tdecode_ms_tok=%.2f\tline_ms=%.1f\tline_reuse${reuseNew}_ms=%.1f".format(med(0), med(1), med(2), med(3)) +
                "\tline_cpu_ms=%.0f\tload1=%.1f".format(med(4), os.systemLoadAverage) +
                "\tfirst_line_ms=%.1f\tfirst_prefill_ms=%.1f\trss_mb=%.0f\trss_base_mb=%.0f\theap_mb=%.0f\tfile_mb=%.1f\truns=%d".format(
                    all[0][2], all[0][0], rssMb(), rssBase, heap, file.length() / 1e6, steady.size) +
                "\tload_mmap_ms=%.1f\ttouch_ms=%.1f\tload_stream_ms=%.1f\tprep_ms=%.1f\tprompt=%d\tmodel=%s".format(loadMmapMs, touchMs, loadStreamMs, prepMs, promptLen, file.name))
            println("WARMUP\tconfig=$name\tkernels=${kernels.name}\tthreads=$threads\tline_ms=" + all.take(warm).joinToString(",") { "%.0f".format(it[2]) })
        }
    }

    /** Single-thread GMAC/s of the matmul kernels on a 384x1536 matrix for n tokens. */
    private fun kernelBench(k: NnKernels, n: Int) {
        val rows = (System.getProperty("rows") ?: "384").toInt(); val cols = (System.getProperty("cols") ?: "1536").toInt()
        val chunk = (System.getProperty("chunk") ?: "$cols").toInt()
        val rnd = Random(1)
        val q = ByteArray(rows * cols).also { rnd.nextBytes(it) }
        val file = System.getProperty("file")
        val w = if (file != null) NnFormat.read(File(file)).q8("layers.0.w1")
            else NnTensor.Q8("w", rows, cols, java.nio.ByteBuffer.allocateDirect(q.size).put(q).flip(), FloatArray(cols) { 0.01f })
        val x = Array(n) { FloatArray(rows) { if (System.getProperty("gauss") != null) (rnd.nextGaussian() * 3).toFloat() else rnd.nextFloat() } }
        val y = Array(n) { FloatArray(cols) }
        val tile = KernelScratch(cols); val acc = FloatArray(cols)
        repeat(8) {
            val reps = maxOf(1, 20000 / n)
            val mx = java.lang.management.ManagementFactory.getThreadMXBean()
            val c0 = mx.currentThreadCpuTime
            val t0 = System.nanoTime()
            repeat(reps) {
                if (n == 1 && System.getProperty("rowsplit") != null) k.matvecRowsPartial(w, x[0], 0, rows, acc, tile)
                else { var c = 0; while (c < cols) { k.matmulCols(w, x, n, y, c, minOf(cols, c + chunk), tile); c += chunk } }
            }
            val dt = System.nanoTime() - t0; val dc = mx.currentThreadCpuTime - c0
            println("${k.name} n=$n wall %.2f cpu %.2f GMAC/s".format(reps.toDouble() * n * rows * cols / dt, reps.toDouble() * n * rows * cols / dc))
        }
    }

    /** Resident set size in MB: /proc on Linux, `ps` on macOS; -1 when unavailable (Windows). */
    private fun rssMb(): Double = runCatching {
        val proc = File("/proc/self/status")
        if (proc.exists()) proc.readLines().firstOrNull { it.startsWith("VmRSS:") }
            ?.split(Regex("\\s+"))?.get(1)?.toDouble()?.div(1024)
        else ProcessBuilder("ps", "-o", "rss=", "-p", ProcessHandle.current().pid().toString()).start()
            .inputStream.bufferedReader().readText().trim().toDoubleOrNull()?.div(1024)
    }.getOrNull() ?: -1.0
}
