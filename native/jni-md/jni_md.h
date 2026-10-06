/* Minimal platform header for jni.h, valid for every target we build (all are 64-bit, int = 32 bit, long long = 64
 * bit). Used instead of the per-OS copies of the JDK so that one zig invocation per target cross-compiles the JNI glue. */
#ifndef _JAVASOFT_JNI_MD_H_
#define _JAVASOFT_JNI_MD_H_
#if defined(_WIN32)
#define JNIEXPORT __declspec(dllexport)
#define JNIIMPORT __declspec(dllimport)
#define JNICALL
#else
#define JNIEXPORT __attribute__((visibility("default")))
#define JNIIMPORT
#define JNICALL
#endif
typedef int jint;
typedef long long jlong;
typedef signed char jbyte;
#endif
