EXTRA_DIST = README.md contrib/inject-errors.c
bin_PROGRAMS = xpar
noinst_HEADERS = platform.h crc32c.h jmode.h vmode.h lmode.h common.h yarg.h \
								 sharding.h sse2neon.h blake2b.h blake2b-impl.h \
								 lmode-impl.h
xpar_SOURCES = xpar.c crc32c.c jmode.c vmode.c lmode.c lmode-scalar.c threadpool.c

if XPAR_WIN_HOST
# Windows host: isolate host-win32.c with $(HOST_WIN32_CFLAGS), bypass
# the mingw C runtime so the binary imports only KERNEL32.dll.
xpar_LDFLAGS = -nostartfiles -Wl,-e,xpar_entry -static -static-libgcc -Wl,--subsystem,console $(XPAR_WIN_SUBSYS_LDFLAGS)
xpar_LDADD_EXTRA_WIN = libhost_win32.a -lkernel32
AM_CPPFLAGS = $(XPAR_WIN_CPPFLAGS)
else
if XPAR_DOS_HOST
# DJGPP / MS-DOS host; -static keeps the stubified xpar.exe self-contained.
xpar_SOURCES += host-djgpp.c
xpar_LDFLAGS = -static
xpar_LDADD_EXTRA_WIN =
else
xpar_SOURCES += host-posix.c
xpar_LDADD_EXTRA_WIN =
endif
endif

if HAVE_BLAKE2B
xpar_SOURCES += blake2b.c
endif

if XPAR_X86_64
xpar_SOURCES += xpar-x86_64.asm
SUFFIXES = .asm
.asm.o:
	$(NASM) $(NAFLAGS) -g -o $@ $<
endif

if XPAR_AARCH64
xpar_SOURCES += xpar-aarch64.S
xpar_CCASFLAGS = -march=armv8-a+crypto+crc
# NEON is always available on AArch64; SSSE3/SSE4.1 variants use sse2neon.
xpar_SOURCES += lmode-ssse3.c
if HAVE_BLAKE2B
xpar_SOURCES += blake2b-sse41.c
endif
endif

# SIMD variants live in convenience libs so per-file SSSE3/SSE41/AVX2
# CFLAGS stay out of the rest of the binary (runtime CPUID dispatch).
noinst_LIBRARIES =
xpar_LDADD = $(xpar_LDADD_EXTRA_WIN)
if XPAR_WIN_HOST
noinst_LIBRARIES += libhost_win32.a
libhost_win32_a_SOURCES = host-win32.c
libhost_win32_a_CFLAGS = $(AM_CFLAGS) $(HOST_WIN32_CFLAGS)
endif
if XPAR_X86_64
if HAVE_SSSE3
noinst_LIBRARIES += liblmode_ssse3.a
liblmode_ssse3_a_SOURCES = lmode-ssse3.c
liblmode_ssse3_a_CFLAGS = $(AM_CFLAGS) $(SSSE3_CFLAGS)
xpar_LDADD += liblmode_ssse3.a
endif
if HAVE_AVX2
noinst_LIBRARIES += liblmode_avx2.a
liblmode_avx2_a_SOURCES = lmode-avx2.c
liblmode_avx2_a_CFLAGS = $(AM_CFLAGS) $(AVX2_CFLAGS)
xpar_LDADD += liblmode_avx2.a
endif
endif
if HAVE_BLAKE2B
if XPAR_X86_64
if HAVE_SSE41
noinst_LIBRARIES += libblake2b_sse41.a
libblake2b_sse41_a_SOURCES = blake2b-sse41.c
libblake2b_sse41_a_CFLAGS = $(AM_CFLAGS) $(SSE41_CFLAGS)
xpar_LDADD += libblake2b_sse41.a
endif
if HAVE_AVX2
noinst_LIBRARIES += libblake2b_avx2.a
libblake2b_avx2_a_SOURCES = blake2b-avx2.c
libblake2b_avx2_a_CFLAGS = $(AM_CFLAGS) $(AVX2_CFLAGS)
xpar_LDADD += libblake2b_avx2.a
endif
endif
endif

dist_man_MANS = xpar.1

if XPAR_DOS_HOST
# Swap DJGPP's go32 stub for CWSDSTUB.EXE (CWSDPMI r7) and emit a ./xpar
# wrapper that drives the binary headlessly via DOSBox-X for self-check.
all-local: xpar.wrapper-stamp
xpar.wrapper-stamp: xpar$(EXEEXT) $(srcdir)/contrib/dos/dosbox-x.conf
	$(EXE2COFF) xpar$(EXEEXT)
	cat $(CWSDSTUB) xpar > xpar$(EXEEXT).new
	mv -f xpar$(EXEEXT).new xpar$(EXEEXT)
	rm -f xpar
	@{ \
	  echo '#!/bin/sh'; \
	  echo 'set -e'; \
	  echo 'tmp=$$(mktemp -d)'; \
	  echo 'trap "rm -rf \"$$tmp\"" EXIT'; \
	  echo 'cfg=$$tmp/dosbox-x.conf'; \
	  echo 'printf %s "$$*" > "$$tmp/args.rsp"   # DJGPP @response-file (PSP 126-byte cap)'; \
	  echo 'sed -e "s|@PWD@|$$PWD|g" -e "s|@OUT@|$$tmp|g" '"$(abs_srcdir)/contrib/dos/dosbox-x.conf"' > "$$cfg"'; \
	  echo 'SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy dosbox-x -silent -fastlaunch -nogui -nomenu -conf "$$cfg" -exit >/dev/null 2>&1'; \
	  echo 'if [ -f "$$tmp/stdout.out" ]; then cat "$$tmp/stdout.out"; fi'; \
	  echo 'ec=$$(tr -d " \\t\\r\\n" < "$$tmp/ec.txt" 2>/dev/null || echo FAIL)'; \
	  echo 'case "$$ec" in OK) exit 0;; *) exit 1;; esac'; \
	} > xpar
	@chmod +x xpar
	@touch $@
DOS_WRAPPER_CLEAN = xpar xpar.wrapper-stamp
else
DOS_WRAPPER_CLEAN =
endif
EXTRA_DIST += contrib/dos/dosbox-x.conf

# Developer convenience targets
.PHONY: update-ChangeLog
update-ChangeLog:
	contrib/gitlog-to-changelog > ChangeLog

# Error-injection helper, built for the build host (not the target).
CLEANFILES = inject-errors$(EXEEXT) $(DOS_WRAPPER_CLEAN)
inject-errors$(EXEEXT): contrib/inject-errors.c host-posix.c threadpool.c
	$(CC_FOR_BUILD) $(CFLAGS_FOR_BUILD) -pthread -I. -o $@ $^

CC_FOR_BUILD ?= $(CC)
CFLAGS_FOR_BUILD ?= $(CFLAGS)

# Self-check file patterns: DOS uses 8.3 name derivation (xpar.exe ->
# xpar.xpa joint / xpar.NNN sharded), POSIX/Windows append the full suffix.
if XPAR_DOS_HOST
XPA_BIN_JOINT      = xpar.xpa
XPA_BIN_SHARD_GLOB = xpar.[0-9][0-9][0-9]
XPA_BIN_SHARD_000  = xpar.000
XPA_BIN_SHARD_001  = xpar.001
XPA_BIN_SHARD_002  = xpar.002
XPA_SRC_JOINT      = xpar.xpa
else
XPA_BIN_JOINT      = xpar$(EXEEXT).xpa
XPA_BIN_SHARD_GLOB = xpar$(EXEEXT).xpa.*
XPA_BIN_SHARD_000  = xpar$(EXEEXT).xpa.000
XPA_BIN_SHARD_001  = xpar$(EXEEXT).xpa.001
XPA_BIN_SHARD_002  = xpar$(EXEEXT).xpa.002
XPA_SRC_JOINT      = xpar.src.xpa
endif

.PHONY: self-check
self-check: inject-errors$(EXEEXT)
	./xpar -Jef xpar$(EXEEXT) && ./xpar -Jdf $(XPA_BIN_JOINT) xpar.org \
		&& cmp xpar$(EXEEXT) xpar.org && rm xpar.org $(XPA_BIN_JOINT)
	./xpar -Jef -i 2 xpar$(EXEEXT) && ./xpar -Jdf $(XPA_BIN_JOINT) xpar.org \
		&& cmp xpar$(EXEEXT) xpar.org && rm xpar.org $(XPA_BIN_JOINT)
	./xpar -Wef --dshards=4 --pshards=2 xpar$(EXEEXT) \
	  && ./xpar -Wdf xpar.org $(XPA_BIN_SHARD_GLOB) \
		&& cmp xpar$(EXEEXT) xpar.org
	rm -f xpar.org $(XPA_BIN_SHARD_GLOB)
	./xpar -Wef --dshards=4 --pshards=2 xpar$(EXEEXT) && rm -f $(XPA_BIN_SHARD_000) \
	  && ./xpar -Wdf xpar.org $(XPA_BIN_SHARD_GLOB) \
		&& cmp xpar$(EXEEXT) xpar.org
	rm -f xpar.org $(XPA_BIN_SHARD_GLOB)
	./xpar -Lef --dshards=4 --pshards=2 xpar$(EXEEXT) \
	  && ./xpar -Ldf xpar.org $(XPA_BIN_SHARD_GLOB) \
		&& cmp xpar$(EXEEXT) xpar.org
	rm -f xpar.org $(XPA_BIN_SHARD_GLOB)
	./xpar -Lef --dshards=4 --pshards=2 xpar$(EXEEXT) && rm -f $(XPA_BIN_SHARD_000) \
	  && ./xpar -Ldf xpar.org $(XPA_BIN_SHARD_GLOB) \
		&& cmp xpar$(EXEEXT) xpar.org
	rm -f xpar.org $(XPA_BIN_SHARD_GLOB)
	./xpar -Lef --dshards=12 --pshards=8 xpar$(EXEEXT) && rm -f $(XPA_BIN_SHARD_000) \
	  && ./xpar -Ldf xpar.org $(XPA_BIN_SHARD_GLOB) \
		&& cmp xpar$(EXEEXT) xpar.org
	rm -f xpar.org $(XPA_BIN_SHARD_GLOB)
	cp xpar$(EXEEXT) xpar.src && ./xpar -Jsef xpar.src \
	  && ./xpar -Jsdf xpar.src $(XPA_SRC_JOINT) > xpar.src.rt \
	  && cmp xpar.src xpar.src.rt
	./xpar -Jsdf xpar.src > xpar.src.rt2 && cmp xpar.src xpar.src.rt2
	rm -f xpar.src.rt xpar.src.rt2
	cp xpar.src xpar.src.corrupt && ./inject-errors$(EXEEXT) scatter \
	  xpar.src.corrupt 10 7
	./xpar -Jsdf xpar.src.corrupt $(XPA_SRC_JOINT) > xpar.src.fixed \
	  && cmp xpar.src xpar.src.fixed
	rm -f xpar.src.corrupt xpar.src.fixed
	cp xpar.src xpar.src.burst && ./inject-errors$(EXEEXT) burst \
	  xpar.src.burst 20 500
	! ./xpar -Jsd xpar.src.burst $(XPA_SRC_JOINT) > /dev/null 2>&1
	rm -f xpar.src.burst
	./xpar -Jsef xpar.src && ./xpar -Jstf xpar.src $(XPA_SRC_JOINT)
	cp xpar.src xpar.src.corrupt \
	  && ./inject-errors$(EXEEXT) burst xpar.src.corrupt 20 500 \
	  && ! ./xpar -Jstf xpar.src.corrupt $(XPA_SRC_JOINT) 2>/dev/null
	rm -f xpar.src.corrupt $(XPA_SRC_JOINT)
	./xpar -Jef xpar.src && ./xpar -Jtf $(XPA_SRC_JOINT)
	cp $(XPA_SRC_JOINT) xpar.xpa.corrupt \
	  && ./inject-errors$(EXEEXT) burst xpar.xpa.corrupt 40 200 \
	  && ! ./xpar -Jtf xpar.xpa.corrupt 2>/dev/null
	rm -f $(XPA_SRC_JOINT) xpar.xpa.corrupt
	! ./xpar -Jsef -i 2 xpar.src 2>/dev/null
	! ./xpar -Lse -s --dshards=4 --pshards=2 xpar.src 2>/dev/null
	./xpar -Jef xpar.src \
	  && ! ./xpar -Jsd xpar.src $(XPA_SRC_JOINT) > /dev/null 2>&1
	rm -f $(XPA_SRC_JOINT)
	./xpar -Jsef xpar.src \
	  && ! ./xpar -Jd $(XPA_SRC_JOINT) xpar.src.mm 2>/dev/null
	rm -f xpar.src $(XPA_SRC_JOINT) xpar.src.mm
	./xpar -Wef --dshards=4 --pshards=2 xpar$(EXEEXT) \
	  && ./xpar -Wtf $(XPA_BIN_SHARD_GLOB)
	./inject-errors$(EXEEXT) burst $(XPA_BIN_SHARD_000) 100 1000 \
	  && ./inject-errors$(EXEEXT) burst $(XPA_BIN_SHARD_001) 100 1000 \
	  && ./inject-errors$(EXEEXT) burst $(XPA_BIN_SHARD_002) 100 1000 \
	  && ! ./xpar -Wtf $(XPA_BIN_SHARD_GLOB) 2>/dev/null
	rm -f $(XPA_BIN_SHARD_GLOB)
	./xpar -Lef --dshards=4 --pshards=2 xpar$(EXEEXT) \
	  && ./xpar -Ltf $(XPA_BIN_SHARD_GLOB)
	rm -f $(XPA_BIN_SHARD_000) $(XPA_BIN_SHARD_001) $(XPA_BIN_SHARD_002)
	! ./xpar -Ltf $(XPA_BIN_SHARD_GLOB) 2>/dev/null
	rm -f $(XPA_BIN_SHARD_GLOB)
	./xpar -Jef -H blake2b xpar$(EXEEXT) && ./xpar -Jdf $(XPA_BIN_JOINT) xpar.org \
	  && cmp xpar$(EXEEXT) xpar.org && rm xpar.org $(XPA_BIN_JOINT)
	./xpar -Jef -i 2 -H blake2b xpar$(EXEEXT) && ./xpar -Jdf $(XPA_BIN_JOINT) xpar.org \
	  && cmp xpar$(EXEEXT) xpar.org && rm xpar.org $(XPA_BIN_JOINT)
	./xpar -Jsef -H blake2b xpar$(EXEEXT) \
	  && ./xpar -Jsdf xpar$(EXEEXT) $(XPA_BIN_JOINT) > xpar.org \
	  && cmp xpar$(EXEEXT) xpar.org && rm xpar.org $(XPA_BIN_JOINT)
	head -c 32 /dev/urandom > test.key
	./xpar -Jef --auth=test.key xpar$(EXEEXT)
	./xpar -Jdf --auth=test.key $(XPA_BIN_JOINT) xpar.org
	cmp xpar$(EXEEXT) xpar.org
	head -c 32 /dev/urandom > wrong.key
	! ./xpar -Jd --auth=wrong.key $(XPA_BIN_JOINT) /tmp/bad > /dev/null 2>&1
	! ./xpar -Jd $(XPA_BIN_JOINT) /tmp/bad > /dev/null 2>&1
	rm -f xpar.org $(XPA_BIN_JOINT)
	./xpar -We -H blake2b --dshards=4 --pshards=2 xpar$(EXEEXT)
	./xpar -Wd xpar.org $(XPA_BIN_SHARD_GLOB)
	cmp xpar$(EXEEXT) xpar.org
	./xpar -Wt $(XPA_BIN_SHARD_GLOB)
	rm -f $(XPA_BIN_SHARD_GLOB) xpar.org
	./xpar -Le -H blake2b --dshards=4 --pshards=2 xpar$(EXEEXT)
	./xpar -Ld xpar.org $(XPA_BIN_SHARD_GLOB)
	cmp xpar$(EXEEXT) xpar.org
	./xpar -Lt $(XPA_BIN_SHARD_GLOB)
	rm -f $(XPA_BIN_SHARD_GLOB) xpar.org
	./xpar -We --auth=test.key --dshards=4 --pshards=2 xpar$(EXEEXT)
	./xpar -Wd --auth=test.key xpar.org $(XPA_BIN_SHARD_GLOB)
	cmp xpar$(EXEEXT) xpar.org
	./xpar -Wt --auth=test.key $(XPA_BIN_SHARD_GLOB)
	! ./xpar -Wd --auth=wrong.key /tmp/bad $(XPA_BIN_SHARD_GLOB) > /dev/null 2>&1
	! ./xpar -Wd /tmp/bad $(XPA_BIN_SHARD_GLOB) > /dev/null 2>&1
	rm -f $(XPA_BIN_SHARD_GLOB) xpar.org
	./xpar -Le --auth=test.key --dshards=4 --pshards=2 xpar$(EXEEXT)
	./xpar -Ld --auth=test.key xpar.org $(XPA_BIN_SHARD_GLOB)
	cmp xpar$(EXEEXT) xpar.org
	./xpar -Lt --auth=test.key $(XPA_BIN_SHARD_GLOB)
	! ./xpar -Ld --auth=wrong.key /tmp/bad $(XPA_BIN_SHARD_GLOB) > /dev/null 2>&1
	rm -f $(XPA_BIN_SHARD_GLOB) xpar.org test.key wrong.key
	./xpar -Jef xpar$(EXEEXT)
	./inject-errors$(EXEEXT) truncate $(XPA_BIN_JOINT) 267
	! ./xpar -Jd $(XPA_BIN_JOINT) /tmp/bad > /dev/null 2>&1
	rm -f $(XPA_BIN_JOINT)
	./xpar -Jef xpar$(EXEEXT)
	./inject-errors$(EXEEXT) swap $(XPA_BIN_JOINT) 37 304 267
	! ./xpar -Jd $(XPA_BIN_JOINT) /tmp/bad > /dev/null 2>&1
	rm -f $(XPA_BIN_JOINT)
	./xpar -Wef --dshards=4 --pshards=2 xpar$(EXEEXT)
	./inject-errors$(EXEEXT) swap2 $(XPA_BIN_SHARD_000) $(XPA_BIN_SHARD_001) 7 1
	! ./xpar -Wd /tmp/bad $(XPA_BIN_SHARD_GLOB) > /dev/null 2>&1
	rm -f $(XPA_BIN_SHARD_GLOB)

.PHONY: cloc
cloc:
	cloc $(xpar_SOURCES) Makefile.am configure.ac xpar-aarch64.S \
		xpar-x86_64.asm
