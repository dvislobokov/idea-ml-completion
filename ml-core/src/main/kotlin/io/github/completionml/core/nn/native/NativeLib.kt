package io.github.completionml.core.nn.native

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/**
 * Loads `libcmlkernels` once per JVM: picks the resource for this OS/arch (`native/libcmlkernels-<os>-<arch>.<ext>`
 * in the ml-core jar), extracts it to `<java.io.tmpdir>/cmlkernels/<sha1>/`, `System.load`s it, checks the ABI version
 * and runs the C self-test (every kernel of the selected ISA level against the scalar reference). Any failure —
 * missing binary, unsupported platform, link error, self-test mismatch — leaves [status] with the reason and the
 * callers fall back to the Kotlin kernels. Never throws.
 *
 * Switches (system properties, checked once):
 *  - `completionml.nn.native=false` (or env `CML_NATIVE=0`): do not load, use Kotlin kernels;
 *  - `completionml.nn.native.path=/abs/lib.so`: load this file instead of the bundled resource (benchmarks);
 *  - `completionml.nn.native.isa=N`: force a lower ISA level (see CmlNative.ISA_*; e.g. 1 = AVX2 on an AVX-512 box).
 */
object NativeLib {
    @Volatile var status: String = "not loaded"; private set
    @Volatile var loaded: Boolean = false; private set
    @Volatile var isaName: String = ""; private set
    private var attempted = false

    val platform: String? by lazy {
        val os = System.getProperty("os.name", "").lowercase()
        val arch = System.getProperty("os.arch", "").lowercase()
        val a = when (arch) { "amd64", "x86_64" -> "x64"; "aarch64", "arm64" -> "arm64"; else -> return@lazy null }
        when {
            os.startsWith("windows") -> if (a == "x64") "windows-x64" else null
            os.startsWith("mac") || os.contains("darwin") -> "macos-$a"
            os.startsWith("linux") -> if (a == "x64") "linux-x64" else null
            else -> null
        }
    }

    fun resourceName(platform: String): String {
        val ext = when { platform.startsWith("windows") -> "dll"; platform.startsWith("macos") -> "dylib"; else -> "so" }
        return "native/libcmlkernels-$platform.$ext"
    }

    /** Loads if not yet attempted; returns whether the library is usable. */
    @Synchronized
    fun ensureLoaded(): Boolean {
        if (attempted) return loaded
        attempted = true
        try {
            if (System.getProperty("completionml.nn.native", "true") == "false" || System.getenv("CML_NATIVE") == "0") {
                status = "disabled by completionml.nn.native=false / CML_NATIVE=0"; return false
            }
            val explicit = System.getProperty("completionml.nn.native.path")
            val file = if (explicit != null) File(explicit) else extract() ?: return false
            System.load(file.absolutePath)
            val abi = CmlNative.abiVersion()
            if (abi != 1) { status = "ABI version $abi, expected 1"; return false }
            val forced = System.getProperty("completionml.nn.native.isa")?.toIntOrNull()
            if (forced != null) CmlNative.setIsa(forced)
            val rc = CmlNative.selftest()
            if (rc != 0) { status = "self-test failed (rc=$rc) at ISA ${CmlNative.isaName(CmlNative.isa())}"; return false }
            isaName = CmlNative.isaName(CmlNative.isa())
            status = "loaded ${file.name}, ISA $isaName (detected ${CmlNative.isaName(CmlNative.isaDetect())})"
            loaded = true
        } catch (e: Throwable) {   // UnsatisfiedLinkError, SecurityException, IOException, ...
            status = "load failed: ${e.javaClass.simpleName}: ${e.message}"
            loaded = false
        }
        return loaded
    }

    private fun extract(): File? {
        val p = platform ?: run { status = "unsupported platform ${System.getProperty("os.name")}/${System.getProperty("os.arch")}"; return null }
        val res = resourceName(p)
        val bytes = NativeLib::class.java.classLoader.getResourceAsStream(res)?.use { it.readBytes() }
            ?: run { status = "resource $res not bundled"; return null }
        val sha = MessageDigest.getInstance("SHA-1").digest(bytes).joinToString("") { "%02x".format(it) }.take(16)
        val dir = File(System.getProperty("java.io.tmpdir"), "cmlkernels/$sha")
        val out = File(dir, res.substringAfterLast('/'))
        if (!(out.isFile && out.length() == bytes.size.toLong())) {
            dir.mkdirs()
            // write-then-rename: another JVM may be extracting the same file (on Windows a loaded DLL is locked)
            val tmp = Files.createTempFile(dir.toPath(), "lib", ".tmp").toFile()
            tmp.writeBytes(bytes)
            try { Files.move(tmp.toPath(), out.toPath(), StandardCopyOption.ATOMIC_MOVE) }
            catch (_: Exception) { tmp.delete(); if (!out.isFile) throw IllegalStateException("cannot create $out") }
        }
        return out
    }
}
