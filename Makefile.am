##  Copyright (C) 2022-2026 Kamila Szewczyk
##
##  This program is free software; you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation; version 3 of the License only.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program. If not, see <http://www.gnu.org/licenses/>.

ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = -std=c99 $(XPAR_WARN_CFLAGS) $(XPAR_HARDEN_CFLAGS) \
            $(XPAR_SAN_CFLAGS) $(XPAR_WIN_CFLAGS) $(XPAR_DOS_CFLAGS)
AM_LDFLAGS = $(XPAR_SAN_LDFLAGS) $(XPAR_WIN_LDFLAGS)

AM_CPPFLAGS = $(XPAR_WIN_CPPFLAGS) $(XPAR_DOS_CPPFLAGS) -I$(top_builddir) \
              -I$(top_srcdir)/src \
              -I$(top_srcdir)/src/kernel \
              -I$(top_srcdir)/src/platform

bin_PROGRAMS = xpar
dist_man1_MANS = xpar.1
dist_doc_DATA = doc/xpar-format.tex

doc/xpar-format.pdf: $(srcdir)/doc/xpar-format.tex
	$(MKDIR_P) doc
	TEXINPUTS=$(abs_top_srcdir)/doc:$$TEXINPUTS \
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
	  -output-directory=$(abs_top_builddir)/doc \
	  $(abs_top_srcdir)/doc/xpar-format.tex

noinst_HEADERS = \
  src/common.h src/xpar2.h src/cli.h src/yarg.h src/v1detect.h \
  src/container.h src/auth.h src/manifest.h src/chunk.h src/slice.h src/resync.h \
  src/plan.h src/json.h src/pathname.h src/volname.h \
  src/ops/ops.h src/ops/chain.h src/ops/vset.h src/ops/volimg.h \
  src/ops/undo.h src/platform/port-win-path.h \
  src/kernel/gf.h src/kernel/gf-impl.h src/kernel/gf-rvv-int.h \
  src/kernel/armour.h src/kernel/armour-impl.h \
  src/kernel/codec.h src/kernel/codec-int.h \
  src/kernel/blake3.h src/kernel/blake3-impl.h \
  src/kernel/crc32c.h src/kernel/crc32c-impl.h \
  src/platform/port.h src/platform/port-fs.h src/platform/port-cpu.h \
  src/platform/port-thread.h

noinst_LIBRARIES = libxpar_core.a

libxpar_core_a_SOURCES = \
  src/cli.c src/progress.c src/v1detect.c src/pathname.c \
  src/volname.c \
  src/container.c src/auth.c src/manifest.c src/chunk.c src/slice.c src/resync.c \
  src/plan.c src/json.c \
  src/ops/create.c src/ops/verify.c src/ops/repair.c src/ops/extract.c \
  src/ops/scrub.c src/ops/listinfo.c src/ops/generation.c \
  src/ops/volimg.c \
  src/kernel/gf.c src/kernel/gf-scalar.c \
  src/kernel/armour.c src/kernel/armour-scalar.c \
  src/kernel/codec-matrix.c src/kernel/codec-fft.c \
  src/kernel/blake3.c src/kernel/blake3-scalar.c \
  src/kernel/crc32c.c src/kernel/crc32c-scalar.c \
  src/platform/port-common.c src/platform/port-crash.c \
  src/platform/port-cpu-x86.c src/platform/port-cpu-arm.c \
  src/platform/port-cpu-other.c

if XPAR_IS_POSIX
libxpar_core_a_SOURCES += src/platform/port-posix.c \
                          src/platform/port-fs-posix.c
endif
if XPAR_IS_WINDOWS
libxpar_core_a_SOURCES += src/platform/port-win32.c \
                          src/platform/port-fs-win32.c \
                          src/platform/port-printf.c
endif
if XPAR_DOS
libxpar_core_a_SOURCES += src/platform/port-djgpp.c \
                          src/platform/port-fs-djgpp.c \
                          src/platform/port-printf.c
endif

if XPAR_THREADS
if XPAR_IS_WINDOWS
libxpar_core_a_SOURCES += src/platform/port-thread-win32.c
else
libxpar_core_a_SOURCES += src/platform/port-thread-posix.c
endif
else
libxpar_core_a_SOURCES += src/platform/port-thread-none.c
endif

xpar_SOURCES = src/xpar.c
xpar_CFLAGS = $(AM_CFLAGS)
xpar_LDFLAGS = $(AM_LDFLAGS)

xpar_LDADD = libxpar_core.a

if XPAR_WIN_LEGACY
xpar_SOURCES += src/platform/port-win95-crt.c
xpar_CFLAGS += -fno-builtin -fno-tree-loop-distribute-patterns
xpar_LDFLAGS += -nostartfiles -nodefaultlibs -Wl,-e,_xpar_entry \
                 -static -static-libgcc -Wl,--subsystem,console
xpar_LDADD += -lgcc -lkernel32
endif

if HAVE_SSSE3
noinst_LIBRARIES          += libxpar_ssse3.a
libxpar_ssse3_a_SOURCES    = src/kernel/gf-ssse3.c
libxpar_ssse3_a_CFLAGS     = $(AM_CFLAGS) $(SSSE3_CFLAGS)
libxpar_ssse3_a_CPPFLAGS   = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_ssse3.a
endif

if HAVE_SSE42
noinst_LIBRARIES          += libxpar_sse42.a
libxpar_sse42_a_SOURCES    = src/kernel/crc32c-sse42.c
libxpar_sse42_a_CFLAGS     = $(AM_CFLAGS) $(SSE42_CFLAGS)
libxpar_sse42_a_CPPFLAGS   = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_sse42.a
endif

if HAVE_AVX2
noinst_LIBRARIES          += libxpar_avx2.a
libxpar_avx2_a_SOURCES     = src/kernel/gf-avx2.c src/kernel/armour-avx2.c \
                             src/kernel/blake3-avx2.c
libxpar_avx2_a_CFLAGS      = $(AM_CFLAGS) $(AVX2_CFLAGS)
libxpar_avx2_a_CPPFLAGS    = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_avx2.a
endif

if HAVE_GFNI
noinst_LIBRARIES          += libxpar_gfni.a
libxpar_gfni_a_SOURCES     = src/kernel/gf-gfni.c src/kernel/armour-gfni.c
libxpar_gfni_a_CFLAGS      = $(AM_CFLAGS) $(GFNI_CFLAGS)
libxpar_gfni_a_CPPFLAGS    = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_gfni.a
endif

if HAVE_GFNI512
noinst_LIBRARIES          += libxpar_gfni512.a
libxpar_gfni512_a_SOURCES  = src/kernel/gf-gfni512.c
libxpar_gfni512_a_CFLAGS   = $(AM_CFLAGS) $(GFNI512_CFLAGS)
libxpar_gfni512_a_CPPFLAGS = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_gfni512.a
endif

if HAVE_VBMI
noinst_LIBRARIES          += libxpar_vbmi.a
libxpar_vbmi_a_SOURCES     = src/kernel/gf-vbmi.c
libxpar_vbmi_a_CFLAGS      = $(AM_CFLAGS) $(VBMI_CFLAGS)
libxpar_vbmi_a_CPPFLAGS    = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_vbmi.a
endif

if HAVE_NEON
noinst_LIBRARIES          += libxpar_neon.a
libxpar_neon_a_SOURCES     = src/kernel/gf-neon.c src/kernel/armour-neon.c \
                             src/kernel/blake3-neon.c
libxpar_neon_a_CFLAGS      = $(AM_CFLAGS) $(NEON_CFLAGS)
libxpar_neon_a_CPPFLAGS    = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_neon.a
endif

if HAVE_NEON_CLMUL
noinst_LIBRARIES          += libxpar_pmull.a
libxpar_pmull_a_SOURCES    = src/kernel/gf-neon-clmul.c
libxpar_pmull_a_CFLAGS     = $(AM_CFLAGS) $(NEON_CLMUL_CFLAGS)
libxpar_pmull_a_CPPFLAGS   = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_pmull.a
endif

if HAVE_SVE
noinst_LIBRARIES          += libxpar_sve.a
libxpar_sve_a_SOURCES      = src/kernel/gf-sve.c
libxpar_sve_a_CFLAGS       = $(AM_CFLAGS) $(SVE_CFLAGS)
libxpar_sve_a_CPPFLAGS     = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_sve.a
endif

if HAVE_VSX
noinst_LIBRARIES          += libxpar_vsx.a
libxpar_vsx_a_SOURCES      = src/kernel/gf-vsx.c
libxpar_vsx_a_CFLAGS       = $(AM_CFLAGS) $(VSX_CFLAGS)
libxpar_vsx_a_CPPFLAGS     = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_vsx.a
endif

if HAVE_RVV_CLMUL
noinst_LIBRARIES          += libxpar_rvvclmul.a
libxpar_rvvclmul_a_SOURCES = src/kernel/gf-rvv-clmul.c
libxpar_rvvclmul_a_CFLAGS  = $(AM_CFLAGS) $(RVV_CLMUL_CFLAGS)
libxpar_rvvclmul_a_CPPFLAGS = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_rvvclmul.a
endif

if HAVE_RVV
noinst_LIBRARIES          += libxpar_rvv.a
libxpar_rvv_a_SOURCES      = src/kernel/gf-rvv-shuffle.c
libxpar_rvv_a_CFLAGS       = $(AM_CFLAGS) $(RVV_CFLAGS)
libxpar_rvv_a_CPPFLAGS     = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_rvv.a
endif

if HAVE_ARM_CRC
noinst_LIBRARIES          += libxpar_armcrc.a
libxpar_armcrc_a_SOURCES   = src/kernel/crc32c-arm.c
libxpar_armcrc_a_CFLAGS    = $(AM_CFLAGS) $(ARM_CRC_CFLAGS)
libxpar_armcrc_a_CPPFLAGS  = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_armcrc.a
endif

xpar_LDADD += libxpar_core.a

## Tests define xpar_main because libxpar_core supplies the platform main.
## Standalone data, damage and timing tools do not link the code under test.

if XPAR_WIN_LEGACY
## Legacy Windows cannot link or run the test programs.
check_PROGRAMS =
TESTS =
else
check_PROGRAMS = tests/t_unit tests/t_codec tests/t_central \
                 tests/mkdata tests/damage tests/forge bench/timeit

tests_t_unit_SOURCES    = tests/t_unit.c
tests_t_unit_CPPFLAGS   = $(AM_CPPFLAGS) -I$(top_srcdir)/tests
tests_t_unit_LDADD      = $(xpar_LDADD)

tests_t_codec_SOURCES   = tests/t_codec.c
tests_t_codec_CPPFLAGS  = $(AM_CPPFLAGS) -I$(top_srcdir)/tests
tests_t_codec_LDADD     = $(xpar_LDADD)

tests_t_central_SOURCES  = tests/t_central.c
tests_t_central_CPPFLAGS = $(AM_CPPFLAGS) -I$(top_srcdir)/tests
tests_t_central_LDADD    = $(xpar_LDADD)

tests_mkdata_SOURCES    = tests/mkdata.c
tests_mkdata_CPPFLAGS   =
tests_mkdata_LDADD      =

tests_damage_SOURCES    = tests/damage.c
tests_damage_CPPFLAGS   =
tests_damage_LDADD      =

## forge links xpar to sign packets.
tests_forge_SOURCES     = tests/forge.c
tests_forge_CPPFLAGS    = $(AM_CPPFLAGS)
tests_forge_LDADD       = $(xpar_LDADD)


TESTS = tests/t_unit tests/t_codec tests/t_central \
        tests/central.sh tests/faults.sh tests/safety.sh tests/sanity.sh \
        tests/build.sh tests/regress.sh tests/bench.sh
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

AM_TESTS_ENVIRONMENT = \
  abs_top_builddir='$(abs_top_builddir)';  export abs_top_builddir; \
  abs_top_srcdir='$(abs_top_srcdir)';      export abs_top_srcdir; \
  srcdir='$(abs_top_srcdir)/tests';        export srcdir; \
  XPAR='$(abs_top_builddir)/xpar$(EXEEXT)';           export XPAR; \
  MKDATA='$(abs_top_builddir)/tests/mkdata$(EXEEXT)'; export MKDATA; \
  DAMAGE='$(abs_top_builddir)/tests/damage$(EXEEXT)'; export DAMAGE; \
  FORGE='$(abs_top_builddir)/tests/forge$(EXEEXT)';   export FORGE; \
  TIMEIT='$(abs_top_builddir)/bench/timeit$(EXEEXT)'; export TIMEIT; \
  XPAR_TEST_LEVEL="$${XPAR_TEST_LEVEL-quick}";        export XPAR_TEST_LEVEL; \
  XPAR_TEST_SEED="$${XPAR_TEST_SEED-20260823}";       export XPAR_TEST_SEED; \
  XPAR_SH='$(SHELL)';                                 export XPAR_SH; \
  CC='$(CC)';                                         export CC;

## Benchmarks are opt-in; tests/bench.sh still requires timeit.

bench_timeit_SOURCES  = bench/timeit.c
bench_timeit_CPPFLAGS =
bench_timeit_LDADD    =

EXTRA_DIST = \
  bootstrap README.md \
  contrib/check-win95-pe.sh contrib/ci-check.sh \
  contrib/gitlog-to-changelog \
  contrib/dos/dosbox-x.conf contrib/dos/ci-check.bat \
  contrib/dos/run-check.sh contrib/dos/run2.c \
  tests/t_harness.h tests/lib.sh \
  tests/central.sh tests/faults.sh tests/safety.sh tests/sanity.sh \
  tests/build.sh tests/regress.sh tests/bench.sh \
  bench/run.sh bench/lib.sh bench/plot.py bench/mktree.sh \
  bench/competitors.sh bench/experiments.sh bench/remote.sh

VENDORED = src/yarg.h

.PHONY: docs doc-check format-check vectors vectors-check fuzz \
        sanitizer-check win95-check check-full check-torture \
        bench-tools bench
docs: doc/xpar-format.pdf

## Wider test matrices for nightly and release builds.
check-full:
	$(MAKE) check XPAR_TEST_LEVEL=full

check-torture:
	$(MAKE) check XPAR_TEST_LEVEL=torture

bench-tools: xpar$(EXEEXT) tests/mkdata$(EXEEXT) tests/damage$(EXEEXT) \
             bench/timeit$(EXEEXT)

bench: bench-tools
	$(SHELL) $(top_srcdir)/bench/run.sh $(BENCHFLAGS)

win95-check: xpar$(EXEEXT)
	OBJDUMP=$(OBJDUMP) NM=$(NM) $(SHELL) \
	  $(top_srcdir)/contrib/check-win95-pe.sh ./xpar$(EXEEXT) ./config.h
