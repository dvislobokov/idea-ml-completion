package io.github.completionml.core.nn

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.LockSupport

/**
 * Fixed pool of [nThreads] - 1 daemon workers plus the calling thread, tuned for the many tiny parallel regions of
 * token-by-token decoding (~5 per layer): workers spin briefly between regions (Thread.onSpinWait) and park afterwards,
 * so dispatch costs ~1-2 µs instead of an ExecutorService round trip. Not reentrant: one [run] at a time.
 */
class NnExecutor(val nThreads: Int, private val spinNanos: Long = 100_000) : AutoCloseable {
    fun interface Task { fun run(task: Int, worker: Int) }

    private class Job(val tasks: Int, val body: Task) {
        val next = AtomicInteger()
        val done = AtomicInteger()
        val error = AtomicReference<Throwable?>()
    }

    @Volatile private var current: Job? = null
    private val generation = AtomicLong()
    @Volatile private var closed = false
    private val workers: Array<Thread>

    init {
        require(nThreads >= 1)
        workers = Array(nThreads - 1) { idx ->
            Thread({ workerLoop(idx + 1) }, "nn-worker-${idx + 1}").apply { isDaemon = true; start() }
        }
    }

    /** Runs `body(task, worker)` for task in 0 until [tasks]; worker ids are in 0 until [nThreads] (0 = caller). */
    fun run(tasks: Int, body: Task) {
        if (tasks <= 0) return
        if (nThreads == 1 || tasks == 1) { for (t in 0 until tasks) body.run(t, 0); return }
        val job = Job(tasks, body)
        current = job
        generation.incrementAndGet()
        for (w in workers) LockSupport.unpark(w)
        work(job, 0)
        while (job.done.get() < tasks) Thread.onSpinWait()
        current = null
        job.error.get()?.let { throw RuntimeException("nn worker failed", it) }
    }

    private fun work(job: Job, worker: Int) {
        while (true) {
            val t = job.next.getAndIncrement()
            if (t >= job.tasks) return
            try { job.body.run(t, worker) } catch (e: Throwable) { job.error.compareAndSet(null, e) }
            job.done.incrementAndGet()
        }
    }

    private fun workerLoop(id: Int) {
        var seen = 0L
        while (!closed) {
            var g = generation.get()
            if (g == seen) {
                val start = System.nanoTime()
                while (generation.get().also { g = it } == seen && !closed) {
                    if (System.nanoTime() - start > spinNanos) { LockSupport.park(this); }
                    else Thread.onSpinWait()
                }
                if (closed) return
            }
            seen = g
            val job = current ?: continue
            work(job, id)
        }
    }

    override fun close() {
        closed = true
        for (w in workers) LockSupport.unpark(w)
    }
}
