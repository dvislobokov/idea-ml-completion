/*
 * JNI glue for io.github.completionml.core.nn.native.CmlNative. Thin: every method maps to one cml_* call. Off-heap
 * memory (model weights = mapped/direct ByteBuffers, activation scratch = direct buffers) is passed as raw addresses
 * (jlong) obtained once via address(); the few heap-array entry points use GetPrimitiveArrayCritical (no copies).
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

#include "cml_kernels.h"

#define FN(name) JNIEXPORT JNICALL Java_io_github_completionml_core_nn_native_CmlNative_##name

JNIEXPORT jint FN(abiVersion)(JNIEnv* env, jclass cls) { (void)env; (void)cls; return cml_abi_version(); }
JNIEXPORT jint FN(isaDetect)(JNIEnv* env, jclass cls) { (void)env; (void)cls; return cml_isa_detect(); }
JNIEXPORT jint FN(isa)(JNIEnv* env, jclass cls) { (void)env; (void)cls; return cml_isa(); }
JNIEXPORT jint FN(setIsa)(JNIEnv* env, jclass cls, jint isa) { (void)env; (void)cls; return cml_set_isa(isa); }
JNIEXPORT jint FN(selftest)(JNIEnv* env, jclass cls) { (void)env; (void)cls; return cml_selftest(); }
JNIEXPORT jint FN(qactUnsigned)(JNIEnv* env, jclass cls) { (void)env; (void)cls; return cml_qact_unsigned(); }

JNIEXPORT jstring FN(isaName)(JNIEnv* env, jclass cls, jint isa) {
    (void)cls;
    return (*env)->NewStringUTF(env, cml_isa_name(isa));
}

/* Start address of a direct ByteBuffer (position NOT included; the caller adds it). 0 for heap buffers. */
JNIEXPORT jlong FN(address)(JNIEnv* env, jclass cls, jobject buf) {
    (void)cls;
    void* p = (*env)->GetDirectBufferAddress(env, buf);
    return (jlong)(intptr_t)p;
}

JNIEXPORT jlong FN(malloc)(JNIEnv* env, jclass cls, jlong bytes) {
    (void)env; (void)cls;
    size_t n = (size_t)bytes; if (n < 64) n = 64;
#if defined(_WIN32)
    return (jlong)(intptr_t)_aligned_malloc((n + 63) & ~(size_t)63, 64);
#else
    void* p = NULL;
    if (posix_memalign(&p, 64, (n + 63) & ~(size_t)63) != 0) return 0;
    return (jlong)(intptr_t)p;
#endif
}

JNIEXPORT void FN(free)(JNIEnv* env, jclass cls, jlong p) {
    (void)env; (void)cls;
#if defined(_WIN32)
    _aligned_free((void*)(intptr_t)p);
#else
    free((void*)(intptr_t)p);
#endif
}

JNIEXPORT void FN(gemmF32)(JNIEnv* env, jclass cls, jlong q, jlong scale, jint K, jint N, jlong x, jint ldx, jint n, jlong y,
                           jint ldy, jint c0, jint c1) {
    (void)env; (void)cls;
    cml_gemm_f32((const int8_t*)(intptr_t)q, (const float*)(intptr_t)scale, K, N, (const float*)(intptr_t)x, ldx, n,
                 (float*)(intptr_t)y, ldy, c0, c1);
}

JNIEXPORT void FN(gemmQ8)(JNIEnv* env, jclass cls, jlong packed, jlong scale, jint K, jint N, jlong xq, jint t0, jint t1,
                          jlong y, jint ldy, jint c0, jint c1) {
    (void)env; (void)cls;
    cml_gemm_q8((const void*)(intptr_t)packed, (const float*)(intptr_t)scale, K, N, (const void*)(intptr_t)xq, t0, t1,
                (float*)(intptr_t)y, ldy, c0, c1);
}

/* Heap-array variant for the K-split decode interface: acc[c] += sum_{k in [k0,k1)} x[k] * q[k][c]. */
JNIEXPORT void FN(gemvF32Partial)(JNIEnv* env, jclass cls, jlong q, jint K, jint N, jfloatArray x, jint k0, jint k1,
                                  jfloatArray acc) {
    (void)cls;
    float* xp = (float*)(*env)->GetPrimitiveArrayCritical(env, x, NULL);
    float* ap = (float*)(*env)->GetPrimitiveArrayCritical(env, acc, NULL);
    if (xp && ap) cml_gemv_f32_partial((const int8_t*)(intptr_t)q, K, N, xp, k0, k1, ap);
    if (ap) (*env)->ReleasePrimitiveArrayCritical(env, acc, ap, 0);
    if (xp) (*env)->ReleasePrimitiveArrayCritical(env, x, xp, JNI_ABORT);
}

JNIEXPORT jlong FN(packSize)(JNIEnv* env, jclass cls, jint K, jint N) { (void)env; (void)cls; return (jlong)cml_pack_size(K, N); }
JNIEXPORT void FN(pack)(JNIEnv* env, jclass cls, jlong q, jint K, jint N, jlong packed) {
    (void)env; (void)cls;
    cml_pack((const int8_t*)(intptr_t)q, K, N, (void*)(intptr_t)packed);
}
JNIEXPORT jlong FN(qactSize)(JNIEnv* env, jclass cls, jint n, jint K) { (void)env; (void)cls; return (jlong)cml_qact_size(n, K); }
JNIEXPORT jlong FN(qactStride)(JNIEnv* env, jclass cls, jint K) { (void)env; (void)cls; return (jlong)cml_qact_stride(K); }
JNIEXPORT void FN(qact)(JNIEnv* env, jclass cls, jlong x, jint ldx, jint n, jint K, jlong xq) {
    (void)env; (void)cls;
    cml_qact((const float*)(intptr_t)x, ldx, n, K, (void*)(intptr_t)xq);
}

/* Heap <-> native copies (memcpy under a critical section; faster than FloatBuffer puts for short rows). */
JNIEXPORT void FN(copyIn)(JNIEnv* env, jclass cls, jfloatArray src, jint off, jlong dst, jint n) {
    (void)cls;
    float* p = (float*)(*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (p) { memcpy((void*)(intptr_t)dst, p + off, (size_t)n * sizeof(float)); (*env)->ReleasePrimitiveArrayCritical(env, src, p, JNI_ABORT); }
}
JNIEXPORT void FN(copyOut)(JNIEnv* env, jclass cls, jlong src, jfloatArray dst, jint off, jint n) {
    (void)cls;
    float* p = (float*)(*env)->GetPrimitiveArrayCritical(env, dst, NULL);
    if (p) { memcpy(p + off, (const void*)(intptr_t)src, (size_t)n * sizeof(float)); (*env)->ReleasePrimitiveArrayCritical(env, dst, p, 0); }
}

JNIEXPORT void FN(copyInBytes)(JNIEnv* env, jclass cls, jbyteArray src, jint off, jlong dst, jint n) {
    (void)cls;
    jbyte* p = (jbyte*)(*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (p) { memcpy((void*)(intptr_t)dst, p + off, (size_t)n); (*env)->ReleasePrimitiveArrayCritical(env, src, p, JNI_ABORT); }
}

JNIEXPORT void FN(rmsnorm)(JNIEnv* env, jclass cls, jfloatArray x, jfloatArray w, jint d, jfloat eps, jfloatArray out) {
    (void)cls;
    float* xp = (float*)(*env)->GetPrimitiveArrayCritical(env, x, NULL);
    float* wp = (float*)(*env)->GetPrimitiveArrayCritical(env, w, NULL);
    float* op = (float*)(*env)->GetPrimitiveArrayCritical(env, out, NULL);
    if (xp && wp && op) cml_rmsnorm(xp, wp, d, eps, op);
    if (op) (*env)->ReleasePrimitiveArrayCritical(env, out, op, 0);
    if (wp) (*env)->ReleasePrimitiveArrayCritical(env, w, wp, JNI_ABORT);
    if (xp) (*env)->ReleasePrimitiveArrayCritical(env, x, xp, JNI_ABORT);
}

JNIEXPORT void FN(siluMul)(JNIEnv* env, jclass cls, jfloatArray g, jfloatArray u, jint n) {
    (void)cls;
    float* gp = (float*)(*env)->GetPrimitiveArrayCritical(env, g, NULL);
    float* up = (float*)(*env)->GetPrimitiveArrayCritical(env, u, NULL);
    if (gp && up) cml_silu_mul(gp, up, n);
    if (up) (*env)->ReleasePrimitiveArrayCritical(env, u, up, JNI_ABORT);
    if (gp) (*env)->ReleasePrimitiveArrayCritical(env, g, gp, 0);
}

/* Attention over native KV memory: kT [hd][ldk] and v [len][ldv] at raw addresses, q/out heap arrays. */
JNIEXPORT void FN(attnHead)(JNIEnv* env, jclass cls, jfloatArray q, jint qo, jint hd, jfloat scale, jlong kT, jint ldk, jlong v,
                            jint ldv, jint len, jlong sc, jfloatArray out, jint oo) {
    (void)cls;
    float* qp = (float*)(*env)->GetPrimitiveArrayCritical(env, q, NULL);
    float* op = (float*)(*env)->GetPrimitiveArrayCritical(env, out, NULL);
    if (qp && op) cml_attn_head(qp + qo, hd, scale, (const float*)(intptr_t)kT, ldk, (const float*)(intptr_t)v, ldv, len,
                                (float*)(intptr_t)sc, op + oo);
    if (op) (*env)->ReleasePrimitiveArrayCritical(env, out, op, 0);
    if (qp) (*env)->ReleasePrimitiveArrayCritical(env, q, qp, JNI_ABORT);
}

JNIEXPORT void FN(memcpy)(JNIEnv* env, jclass cls, jlong dst, jlong src, jlong bytes) {
    (void)env; (void)cls;
    memcpy((void*)(intptr_t)dst, (const void*)(intptr_t)src, (size_t)bytes);
}

/* KV cache store for one layer: for every token t < n and kv head h: kT[h][d][p0 + t] = k[t][h*hd + d],
 * v[h][p0 + t][d] = v[t][h*hd + d]. kAddrs/vAddrs hold the per-head base addresses (nKv entries from `base`);
 * key rows are ldk floats apart. */
JNIEXPORT void FN(kvStore)(JNIEnv* env, jclass cls, jlongArray kAddrs, jlongArray vAddrs, jint base, jint nKv, jint ldk, jint hd,
                           jint p0, jobjectArray k, jobjectArray v, jint n) {
    (void)cls;
    jlong* ka = (*env)->GetLongArrayElements(env, kAddrs, NULL);
    jlong* va = (*env)->GetLongArrayElements(env, vAddrs, NULL);
    if (ka && va) {
        for (int t = 0; t < n; t++) {
            jfloatArray kr = (jfloatArray)(*env)->GetObjectArrayElement(env, k, t);
            jfloatArray vr = (jfloatArray)(*env)->GetObjectArrayElement(env, v, t);
            float* kp = (float*)(*env)->GetPrimitiveArrayCritical(env, kr, NULL);
            float* vp = (float*)(*env)->GetPrimitiveArrayCritical(env, vr, NULL);
            if (kp && vp) {
                for (int h = 0; h < nKv; h++) {
                    float* kT = (float*)(intptr_t)ka[base + h];
                    float* vv = (float*)(intptr_t)va[base + h];
                    for (int d = 0; d < hd; d++) kT[(size_t)d * ldk + p0 + t] = kp[h * hd + d];
                    memcpy(vv + (size_t)(p0 + t) * hd, vp + h * hd, sizeof(float) * (size_t)hd);
                }
            }
            if (vp) (*env)->ReleasePrimitiveArrayCritical(env, vr, vp, JNI_ABORT);
            if (kp) (*env)->ReleasePrimitiveArrayCritical(env, kr, kp, JNI_ABORT);
            (*env)->DeleteLocalRef(env, kr); (*env)->DeleteLocalRef(env, vr);
        }
    }
    if (ka) (*env)->ReleaseLongArrayElements(env, kAddrs, ka, JNI_ABORT);
    if (va) (*env)->ReleaseLongArrayElements(env, vAddrs, va, JNI_ABORT);
}

/* Causal attention of one head for tokens [t0, t1): token t attends to positions < p0 + t + 1. q/out are rows of
 * heap arrays (head slice at offset qo); keys transposed at kT (ldk), values at v (hd per position); sc = native
 * scratch of >= p0 + t1 floats. */
JNIEXPORT void FN(attnBlock)(JNIEnv* env, jclass cls, jobjectArray q, jobjectArray out, jint t0, jint t1, jint qo, jint hd,
                             jfloat scale, jlong kT, jint ldk, jlong v, jint p0, jlong sc) {
    (void)cls;
    for (int t = t0; t < t1; t++) {
        jfloatArray qr = (jfloatArray)(*env)->GetObjectArrayElement(env, q, t);
        jfloatArray orow = (jfloatArray)(*env)->GetObjectArrayElement(env, out, t);
        float* qp = (float*)(*env)->GetPrimitiveArrayCritical(env, qr, NULL);
        float* op = (float*)(*env)->GetPrimitiveArrayCritical(env, orow, NULL);
        if (qp && op) cml_attn_head(qp + qo, hd, scale, (const float*)(intptr_t)kT, ldk, (const float*)(intptr_t)v, hd,
                                    p0 + t + 1, (float*)(intptr_t)sc, op + qo);
        if (op) (*env)->ReleasePrimitiveArrayCritical(env, orow, op, 0);
        if (qp) (*env)->ReleasePrimitiveArrayCritical(env, qr, qp, JNI_ABORT);
        (*env)->DeleteLocalRef(env, qr); (*env)->DeleteLocalRef(env, orow);
    }
}
