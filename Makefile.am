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

if HAVE_VPCLMUL
noinst_LIBRARIES          += libxpar_vpclmul.a
libxpar_vpclmul_a_SOURCES  = src/kernel/crc32c-vpclmul.c
libxpar_vpclmul_a_CFLAGS    = $(AM_CFLAGS) $(VPCLMUL_CFLAGS)
libxpar_vpclmul_a_CPPFLAGS  = $(AM_CPPFLAGS)
xpar_LDADD                 += libxpar_vpclmul.a
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

if HAVE_PMULL
noinst_LIBRARIES          += libxpar_pmull.a
libxpar_pmull_a_SOURCES    = src/kernel/gf-neon-clmul.c
libxpar_pmull_a_CFLAGS     = $(AM_CFLAGS) $(PMULL_CFLAGS)
libxpar_pmull_a_CPPFLAGS   = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_pmull.a
endif

if HAVE_SVE2
noinst_LIBRARIES          += libxpar_sve2.a
libxpar_sve2_a_SOURCES     = src/kernel/gf-sve2.c
libxpar_sve2_a_CFLAGS      = $(AM_CFLAGS) $(SVE2_CFLAGS)
libxpar_sve2_a_CPPFLAGS    = $(AM_CPPFLAGS)
xpar_LDADD                += libxpar_sve2.a
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

EXTRA_DIST = \
  bootstrap README.md \
  contrib/inject-errors.c contrib/check-win95-pe.sh contrib/ci-check.sh \
  contrib/gitlog-to-changelog \
  contrib/dos/dosbox-x.conf contrib/dos/ci-check.bat \
  contrib/dos/run-check.sh contrib/dos/run2.c

VENDORED = src/yarg.h

.PHONY: docs doc-check format-check vectors vectors-check fuzz \
        sanitizer-check win95-check
docs: doc/xpar-format.pdf

win95-check: xpar$(EXEEXT)
	OBJDUMP=$(OBJDUMP) NM=$(NM) $(SHELL) \
	  $(top_srcdir)/contrib/check-win95-pe.sh ./xpar$(EXEEXT) ./config.h
