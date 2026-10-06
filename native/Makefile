# cmlkernels — native SIMD kernels for ml-core's neural inference (see ../docs/NATIVE-KERNELS.md).
#
#   make            native build with the host compiler (gcc/clang): build/libcmlkernels-linux-x64.so + bench + selftest
#   make cross      the four shipping binaries with zig (zig cc = clang + bundled libc/SDK stubs; no other toolchain needed)
#   make test       runs the self-test at every ISA level the host supports
#   make install    copies the four binaries into ml-core/src/main/resources/native/
#
# Baseline ISA is x86-64 (SSE2) / armv8-a; every SIMD level lives in functions with target attributes and is selected
# at runtime (cpuid), so one binary per OS/arch.

CC      ?= gcc
ZIG     ?= zig
JAVA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(which javac))))
BUILD   ?= build
RES     ?= ../ml-core/src/main/resources/native

CFLAGS  = -O3 -std=c11 -fPIC -fvisibility=hidden -fno-math-errno -fno-trapping-math -Wall -Wextra -Wno-unused-function \
          -DCML_HAVE_AVXVNNI -DNDEBUG
JNIINC  = -I$(JAVA_HOME)/include -Ijni-md
SRCS    = kernels.c jni.c
HDRS    = cml_kernels.h kernels_x86.inc kernels_neon.inc

ZIGCC   = $(ZIG) cc
ZFLAGS  = -O3 -std=c11 -fPIC -fvisibility=hidden -fno-math-errno -fno-trapping-math -Wall -Wextra -Wno-unused-function -s \
          -DCML_HAVE_AVXVNNI -DNDEBUG -Ijni-md $(JNIINC)

.PHONY: all cross test install clean sizes

all: $(BUILD)/libcmlkernels-linux-x64.so $(BUILD)/bench $(BUILD)/selftest

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/kernels.o: kernels.c $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/jni.o: jni.c cml_kernels.h | $(BUILD)
	$(CC) $(CFLAGS) $(JNIINC) -c $< -o $@

$(BUILD)/libcmlkernels-linux-x64.so: $(BUILD)/kernels.o $(BUILD)/jni.o
	$(CC) -shared -Wl,--no-undefined -o $@ $^ -lm -lpthread

$(BUILD)/bench: bench.c $(BUILD)/kernels.o
	$(CC) -O3 -std=c11 -fno-math-errno -Wall -Wextra $< $(BUILD)/kernels.o -o $@ -lm -lpthread

$(BUILD)/selftest: selftest_main.c $(BUILD)/kernels.o
	$(CC) -O2 -std=c11 $< $(BUILD)/kernels.o -o $@ -lm

test: $(BUILD)/selftest
	$(BUILD)/selftest

# ---- cross builds (zig). The macOS targets need no SDK: zig ships the libSystem stubs.
cross: $(BUILD)/libcmlkernels-linux-x64.so.zig $(BUILD)/libcmlkernels-windows-x64.dll $(BUILD)/libcmlkernels-macos-arm64.dylib $(BUILD)/libcmlkernels-macos-x64.dylib sizes

$(BUILD)/libcmlkernels-linux-x64.so.zig: $(SRCS) $(HDRS) | $(BUILD)
	$(ZIGCC) -target x86_64-linux-gnu.2.17 $(ZFLAGS) -shared -o $@ $(SRCS) -lm -lpthread

$(BUILD)/libcmlkernels-windows-x64.dll: $(SRCS) $(HDRS) | $(BUILD)
	$(ZIGCC) -target x86_64-windows-gnu $(ZFLAGS) -shared -o $@ $(SRCS)

$(BUILD)/libcmlkernels-macos-arm64.dylib: $(SRCS) $(HDRS) | $(BUILD)
	$(ZIGCC) -target aarch64-macos $(ZFLAGS) -shared -o $@ $(SRCS)

$(BUILD)/libcmlkernels-macos-x64.dylib: $(SRCS) $(HDRS) | $(BUILD)
	$(ZIGCC) -target x86_64-macos $(ZFLAGS) -shared -o $@ $(SRCS)

sizes:
	@ls -l $(BUILD)/libcmlkernels-* | awk '{printf "%8d  %s\n", $$5, $$9}'

install: cross
	mkdir -p $(RES)
	cp $(BUILD)/libcmlkernels-linux-x64.so.zig $(RES)/libcmlkernels-linux-x64.so
	cp $(BUILD)/libcmlkernels-windows-x64.dll $(BUILD)/libcmlkernels-macos-arm64.dylib $(BUILD)/libcmlkernels-macos-x64.dylib $(RES)/

clean:
	rm -rf $(BUILD)
