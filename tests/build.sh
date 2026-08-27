#!/bin/sh
#  Copyright (C) 2022-2026 Kamila Szewczyk
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; version 3 of the License only.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.

#  A configure probe that is missing its own flags, or a function that was
#  never added to AC_CHECK_FUNCS, fails silently: the feature is simply
#  never compiled in and nothing says so. Every probe below asserts the
#  other direction, that what the compiler accepts, configure found.

. "${srcdir:-.}/lib.sh"

cfg=$abs_top_builddir/config.h
test -f "$cfg" || skip_all "no config.h; nothing to check"

: "${CC:=cc}"

# defined <MACRO>
defined() { grep -q "^#define $1 " "$cfg"; }

# compiles <flags> <code>
compiles() {
  _flags=$1;  shift
  cat > probe.c <<EOF
$*
EOF
  $CC $_flags -c -o probe.o probe.c > "$log" 2>&1
}

# probe <MACRO> <flags> <code...>
#   The compiler accepting the code obliges configure to have defined the
#   macro. The converse is not asserted: configure may refuse a feature for
#   reasons of its own, but it must never miss one it can build.
probe() {
  _macro=$1;  _flags=$2;  shift 2
  if compiles "$_flags" "$*"; then
    if defined "$_macro"; then ok
    else bad "$CC $_flags builds this, but config.h has no $_macro"; fi
  else
    note "$_macro: compiler rejects the probe here, skipped"
  fi
}

step "host functions configure must not miss"

case `uname -s 2> /dev/null` in
  MINGW*|MSYS*|CYGWIN*|*DOS*) note "not a POSIX host; skipping flock" ;;
  *)
    #  flock and fcntl(F_SETLK) differ: an fcntl lock is dropped by any
    #  close of the file, so falling back to it silently is not benign.
    probe HAVE_FLOCK "" '#include <sys/file.h>
int main(void) { return flock(0, LOCK_EX | LOCK_NB); }'
    ;;
esac

step "SIMD probes carry their own flags"

simd=no
for m in HAVE_SSSE3 HAVE_SSE42 HAVE_AVX2 HAVE_GFNI HAVE_GFNI512 \
         HAVE_VBMI HAVE_NEON HAVE_VSX HAVE_RVV; do
  if defined "$m"; then simd=yes;  break; fi
done

if defined XPAR_NO_SIMD; then
  note "SIMD disabled; skipping vector probes"
  if test "$simd" = no; then ok
  else bad "SIMD disabled but a vector tier is configured"; fi
elif test "$simd" = no; then
  case `uname -m 2> /dev/null` in
    i?86|x86_64|amd64)
      bad "all SIMD probes failed unexpectedly on x86" ;;
    *) note "no SIMD tier configured on this architecture; skipping" ;;
  esac
else
  probe HAVE_SSSE3 "-mssse3" '#include <immintrin.h>
int main(void) { __m128i a = _mm_set1_epi8(1);
                 return _mm_cvtsi128_si32(_mm_shuffle_epi8(a, a)); }'

  probe HAVE_AVX2 "-mavx2" '#include <immintrin.h>
int main(void) { __m256i a = _mm256_set1_epi8(1);
                 return _mm256_extract_epi32(_mm256_shuffle_epi8(a, a), 0); }'

  probe HAVE_SSE42 "-msse4.2" '#include <immintrin.h>
int main(void) { return (int) _mm_crc32_u8(0u, 1); }'

  probe HAVE_GFNI "-mgfni -mavx2" '#include <immintrin.h>
int main(void) { __m256i a = _mm256_set1_epi8(1);
                 return _mm256_extract_epi32(
                          _mm256_gf2p8affine_epi64_epi8(a, a, 0), 0); }'

  probe HAVE_GFNI512 "-mgfni -mavx512f -mavx512bw -mavx512vl" \
    '#include <immintrin.h>
int main(void) { __m512i a = _mm512_set1_epi8(1);
                 return (int) _mm512_cvtsi512_si32(
                          _mm512_gf2p8affine_epi64_epi8(a, a, 0)); }'

  probe HAVE_VBMI "-mavx512f -mavx512bw -mavx512vl -mavx512vbmi" \
    '#include <immintrin.h>
int main(void) { __m512i a = _mm512_set1_epi8(1);
                 return (int) _mm512_cvtsi512_si32(
                          _mm512_permutexvar_epi8(a, a)); }'
fi

step "every configured tier was actually archived"

#  A defined macro with no convenience library means the tier is compiled
#  into nothing, which is how GFNI-512 went missing.
for pair in "HAVE_SSSE3 ssse3" "HAVE_SSE42 sse42" "HAVE_AVX2 avx2" \
            "HAVE_GFNI gfni" "HAVE_GFNI512 gfni512" "HAVE_VBMI vbmi" \
            "HAVE_VPCLMUL vpclmul" "HAVE_NEON neon" "HAVE_PMULL pmull" \
            "HAVE_SVE sve" "HAVE_ARM_CRC32 armcrc" "HAVE_VSX vsx" \
            "HAVE_RVV rvv" "HAVE_RVV_CLMUL rvvclmul"; do
  m=`echo "$pair" | cut -d' ' -f1`
  a=`echo "$pair" | cut -d' ' -f2`
  defined "$m" || continue
  if test -f "$abs_top_builddir/libxpar_$a.a"; then ok
  else bad "$m is defined but libxpar_$a.a was not built"; fi
done

summary
