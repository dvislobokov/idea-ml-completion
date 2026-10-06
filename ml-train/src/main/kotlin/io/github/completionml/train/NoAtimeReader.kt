package io.github.completionml.train

import java.io.File
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import java.nio.file.Files

/**
 * Reads files with `O_NOATIME` (libc `open`/`read` through the JDK 21 foreign-function API, needs `--enable-preview`).
 * Why: on the training server the first read of a file updates its atime (tar-extracted files have atime = old mtime, so `relatime`
 * fires once per file), and with ~14 M cached inodes the writeback kernel threads then make cold reads ~1000× slower
 * (60 ms per file, soft lockups). With O_NOATIME the same reads take microseconds. Falls back to [Files.readAllBytes] when the
 * native call is unavailable or refused (e.g. the file is not owned by the user).
 */
object NoAtimeReader {
    private const val O_RDONLY = 0
    private const val O_NOATIME = 262144       // 0x40000 on Linux x86-64 and aarch64 (generic asm)

    private class Natives(val open: MethodHandle, val read: MethodHandle, val close: MethodHandle)

    private val natives: Natives? = try {
        if (!System.getProperty("os.name").lowercase().contains("linux")) null else {
            val linker = Linker.nativeLinker(); val lookup = linker.defaultLookup()
            fun h(name: String, fd: FunctionDescriptor) = linker.downcallHandle(lookup.find(name).get(), fd)
            Natives(
                h("open", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)),
                h("read", FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)),
                h("close", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)))
        }
    } catch (e: Throwable) { null }

    val available get() = natives != null

    fun readAllBytes(f: File): ByteArray {
        val n = natives ?: return Files.readAllBytes(f.toPath())
        Arena.ofConfined().use { arena ->
            val fd = n.open.invoke(arena.allocateUtf8String(f.path), O_RDONLY or O_NOATIME) as Int
            if (fd < 0) return Files.readAllBytes(f.toPath())
            try {
                var buf = arena.allocate(maxOf(f.length(), 1L) + 1)
                var total = 0L
                while (true) {
                    val r = n.read.invoke(fd, buf.asSlice(total), buf.byteSize() - total) as Long
                    if (r < 0) throw java.io.IOException("read failed: ${f.path}")
                    if (r == 0L) break
                    total += r
                    if (total == buf.byteSize()) { val nb = arena.allocate(buf.byteSize() * 2); MemorySegment.copy(buf, 0, nb, 0, total); buf = nb }
                }
                return buf.asSlice(0, total).toArray(ValueLayout.JAVA_BYTE)
            } finally { n.close.invoke(fd) }
        }
    }
}
