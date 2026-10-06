#define _POSIX_C_SOURCE 200809L
/*
 * Kernel benchmark without the JVM: GMAC/s of the GEMM / GEMV kernels on the matrix shapes of the S/M/L model
 * configurations, 1 thread and T threads (a tiny spinning pthread pool, tasks = column chunks x token blocks like
 * the Kotlin NnModel does it).
 *
 *   bench [--isa N] [--threads T] [--n 512] [--ms 400] [--shape K,N] [--mode f32|q8|gemv|all] [--selftest]
 */
#include "cml_kernels.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------- spinning pool
 * Fork-join with no overlap between runs: the main thread publishes a job and bumps `gen`; every worker runs tasks
 * until none are left, then bumps `finished`; main returns only when all workers have finished, so the next job's
 * counters cannot race with stragglers. */
typedef void (*task_fn)(int task, void* arg);
static struct {
    int nthreads; pthread_t th[256];
    atomic_int gen; atomic_int next; atomic_int finished; atomic_int tasks; task_fn fn; void* arg; atomic_int stop;
} pool;

static void pool_work(void) {
    int tasks = atomic_load(&pool.tasks);
    for (;;) {
        int t = atomic_fetch_add(&pool.next, 1);
        if (t >= tasks) return;
        pool.fn(t, pool.arg);
    }
}
static void* pool_thread(void* a) {
    (void)a;
    int seen = 0;
    for (;;) {
        int g;
        while ((g = atomic_load(&pool.gen)) == seen) { if (atomic_load(&pool.stop)) return NULL; }
        seen = g;
        pool_work();
        atomic_fetch_add(&pool.finished, 1);
    }
}
static void pool_start(int n) {
    pool.nthreads = n;
    for (int i = 1; i < n; i++) pthread_create(&pool.th[i], NULL, pool_thread, NULL);
}
static void pool_run(int tasks, task_fn fn, void* arg) {
    pool.fn = fn; pool.arg = arg;
    atomic_store(&pool.tasks, tasks);
    atomic_store(&pool.next, 0); atomic_store(&pool.finished, 0);
    atomic_fetch_add(&pool.gen, 1);
    pool_work();
    while (atomic_load(&pool.finished) < pool.nthreads - 1) {}
}

/* ---------------------------------------------------------------- shapes */
typedef struct { const char* cfg; const char* name; int K, N; } shape_t;
static const shape_t shapes[] = {
    {"S", "wq", 384, 384}, {"S", "wk", 384, 128}, {"S", "w1", 384, 1536}, {"S", "w2", 1536, 384}, {"S", "head", 384, 32000},
    {"M", "wq", 512, 512}, {"M", "wk", 512, 256}, {"M", "w1", 512, 2048}, {"M", "w2", 2048, 512}, {"M", "head", 512, 32000},
    {"L", "wq", 768, 768}, {"L", "wk", 768, 256}, {"L", "w1", 768, 2560}, {"L", "w2", 2560, 768}, {"L", "head", 768, 32000},
};

typedef struct {
    int K, N, n, mode, chunk, tblock, ntb;
    const int8_t* q; const float* scale; const float* x; float* y; const void* packed; const void* xq;
} job_t;

static void gemm_task(int task, void* a) {
    job_t* j = (job_t*)a;
    int ct = task / j->ntb, tb = task % j->ntb;
    int c0 = ct * j->chunk, c1 = c0 + j->chunk < j->N ? c0 + j->chunk : j->N;
    int t0 = tb * j->tblock, t1 = t0 + j->tblock < j->n ? t0 + j->tblock : j->n;
    if (j->mode == 0) cml_gemm_f32(j->q, j->scale, j->K, j->N, j->x + (size_t)t0 * j->K, j->K, t1 - t0, j->y + (size_t)t0 * j->N, j->N, c0, c1);
    else cml_gemm_q8(j->packed, j->scale, j->K, j->N, j->xq, t0, t1, j->y + (size_t)t0 * j->N, j->N, c0, c1);
}

static void qact_task(int task, void* a) {
    job_t* j = (job_t*)a;
    int t0 = task * 32, t1 = t0 + 32 < j->n ? t0 + 32 : j->n;
    cml_qact(j->x + (size_t)t0 * j->K, j->K, t1 - t0, j->K, (char*)j->xq + t0 * cml_qact_stride(j->K));
}

static double run_gemm(job_t* j, int threads, double ms) {
    /* task split like NnModel.matmul: 256-column chunks, token blocks so that there are >= 4 tasks per thread */
    j->chunk = 256;
    int colTasks = (j->N + j->chunk - 1) / j->chunk;
    int want = threads * 4;
    int ntb = (want + colTasks - 1) / colTasks; if (ntb < 1) ntb = 1; if (ntb > j->n / 16) ntb = j->n / 16 > 0 ? j->n / 16 : 1;
    j->tblock = ((j->n + ntb - 1) / ntb + 3) / 4 * 4; if (j->tblock < 1) j->tblock = 1;
    j->ntb = (j->n + j->tblock - 1) / j->tblock;
    int tasks = colTasks * j->ntb;
    double best = 0; int reps = 0;
    double t_end = now() + ms * 1e-3;
    do {
        double t0 = now();
        if (j->mode == 1) pool_run((j->n + 31) / 32, qact_task, j);
        pool_run(tasks, gemm_task, j);
        double dt = now() - t0;
        double rate = (double)j->n * j->K * j->N / dt * 1e-9;
        if (rate > best) best = rate;
        reps++;
    } while (now() < t_end || reps < 3);
    return best;
}

static void gemv_task(int task, void* a) {
    job_t* j = (job_t*)a;
    int per = (j->N + pool.nthreads - 1) / pool.nthreads; per = (per + 15) / 16 * 16;
    int c0 = task * per, c1 = c0 + per < j->N ? c0 + per : j->N;
    if (c0 >= c1) return;
    if (j->mode == 0) cml_gemm_f32(j->q, j->scale, j->K, j->N, j->x, j->K, 1, j->y, j->N, c0, c1);
    else cml_gemm_q8(j->packed, j->scale, j->K, j->N, j->xq, 0, 1, j->y, j->N, c0, c1);
}

static double run_gemv(job_t* j, int threads, double ms) {
    double best = 0; int reps = 0;
    double t_end = now() + ms * 1e-3;
    do {
        double t0 = now();
        for (int r = 0; r < 20; r++) pool_run(threads, gemv_task, j);
        double dt = (now() - t0) / 20;
        double rate = (double)j->K * j->N / dt * 1e-9;  /* GB/s of weights == GMAC/s */
        if (rate > best) best = rate;
        reps++;
    } while (now() < t_end || reps < 3);
    return best;
}

int main(int argc, char** argv) {
    int isa = -1, threads = 1, n = 512; double ms = 400; int shapeK = 0, shapeN = 0; const char* mode = "all"; int selftest = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--isa")) isa = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n")) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ms")) ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--shape")) sscanf(argv[++i], "%d,%d", &shapeK, &shapeN);
        else if (!strcmp(argv[i], "--mode")) mode = argv[++i];
        else if (!strcmp(argv[i], "--selftest")) selftest = 1;
    }
    if (isa >= 0) cml_set_isa(isa);
    printf("isa=%s (detected %s) threads=%d n=%d\n", cml_isa_name(cml_isa()), cml_isa_name(cml_isa_detect()), threads, n);
    if (selftest) {
        for (int lvl = 0; lvl <= cml_isa_detect(); lvl++) {
            if (cml_set_isa(lvl) != lvl) continue;
            double t0 = now(); int rc = cml_selftest();
            printf("selftest %-12s rc=%d (%.1f ms)\n", cml_isa_name(lvl), rc, (now() - t0) * 1e3);
        }
        if (isa >= 0) cml_set_isa(isa); else cml_set_isa(cml_isa_detect());
    }
    pool_start(threads);
    int doF32 = !strcmp(mode, "all") || !strcmp(mode, "f32"), doQ8 = !strcmp(mode, "all") || !strcmp(mode, "q8");
    int doGemv = !strcmp(mode, "all") || !strcmp(mode, "gemv");
    printf("%-3s %-5s %5s %6s | %10s %10s | %12s %12s\n", "cfg", "mat", "K", "N", "f32 GMAC/s", "q8 GMAC/s", "gemv f32 GB/s", "gemv q8 GB/s");
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        shape_t sh = shapes[s];
        if (shapeK && (sh.K != shapeK || sh.N != shapeN)) continue;
        if (shapeK && s > 0 && shapes[s - 1].K == shapeK && shapes[s - 1].N == shapeN) continue;
        int K = sh.K, N = sh.N;
        int8_t* q = malloc((size_t)K * N); float* scale = malloc(sizeof(float) * N);
        float* x = malloc(sizeof(float) * (size_t)n * K); float* y = malloc(sizeof(float) * (size_t)n * N);
        void* packed = malloc(cml_pack_size(K, N) + 64); void* xq = malloc(cml_qact_size(n, K) + 64);
        for (size_t i = 0; i < (size_t)K * N; i++) q[i] = (int8_t)(rand() % 255 - 127);
        for (int c = 0; c < N; c++) scale[c] = 0.01f;
        for (size_t i = 0; i < (size_t)n * K; i++) x[i] = (float)rand() / RAND_MAX - 0.5f;
        cml_pack(q, K, N, packed);
        cml_qact(x, K, n, K, xq);
        job_t j = { K, N, n, 0, 0, 0, 0, q, scale, x, y, packed, xq };
        double f32 = 0, q8 = 0, gf = 0, gq = 0;
        if (doF32) { j.mode = 0; f32 = run_gemm(&j, threads, ms); }
        if (doQ8) { j.mode = 1; q8 = run_gemm(&j, threads, ms); }
        if (doGemv) { j.mode = 0; gf = run_gemv(&j, threads, ms / 2); j.mode = 1; gq = run_gemv(&j, threads, ms / 2); }
        printf("%-3s %-5s %5d %6d | %10.1f %10.1f | %12.1f %12.1f\n", sh.cfg, sh.name, K, N, f32, q8, gf, gq);
        fflush(stdout);
        free(q); free(scale); free(x); free(y); free(packed); free(xq);
    }
    atomic_store(&pool.stop, 1);
    return 0;
}
