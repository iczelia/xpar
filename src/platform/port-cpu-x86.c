/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  x86 CPUID, XCR0 validation, and tier selection.

    AVX2 requires OSXSAVE with SSE and YMM state enabled.  AVX-512 also
    requires opmask, ZMM_Hi256, and Hi16_ZMM state.  GFNI itself needs only
    XMM state; VEX and EVEX tiers retain their AVX checks.  */

#include "common.h"
#include "port-cpu.h"

#if defined(__x86_64__) || defined(__i386__) ||                               \
    defined(_M_X64) || defined(_M_IX86)

#if defined(__GNUC__) || defined(__clang__)
  #include <cpuid.h>
#elif defined(_MSC_VER)
  #include <intrin.h>
#endif

static void xpar_cpuid(u32 leaf, u32 sub, u32 r[4]) {
#if defined(__GNUC__) || defined(__clang__)
  /*  The macro form, not inline asm: on 32-bit PIC builds ebx is the GOT
      register and cpuid.h is what knows how to save it.  */
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#elif defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, (int) leaf, (int) sub);
  r[0] = (u32) regs[0];  r[1] = (u32) regs[1];
  r[2] = (u32) regs[2];  r[3] = (u32) regs[3];
#else
  (void) leaf;  (void) sub;
  r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

static u64 xpar_xcr0(void) {
#if defined(__GNUC__) || defined(__clang__)
  u32 lo, hi;
  /*  Encoded as bytes rather than as the mnemonic: this file is compiled
      without any -m flag (Makefile.am keeps ISA flags off the program), and
      an assembler that predates -mxsave rejects `xgetbv` outright.  */
  __asm__ volatile (".byte 0x0f, 0x01, 0xd0"
                    : "=a" (lo), "=d" (hi) : "c" (0));
  return ((u64) hi << 32) | (u64) lo;
#elif defined(_MSC_VER)
  return _xgetbv(0);
#else
  return 0;
#endif
}

u32 xpar_cpu_probe(void) {
  u32 f = 0, r[4], maxleaf;
  bool ymm, zmm;

  xpar_cpuid(0, 0, r);
  maxleaf = r[0];
  if (maxleaf < 1) return 0;

  xpar_cpuid(1, 0, r);
  if (r[2] & (1u << 9))  f |= XPAR_CPU_SSSE3;
  if (r[2] & (1u << 20)) f |= XPAR_CPU_SSE42;
  if (r[2] & (1u << 1))  f |= XPAR_CPU_PCLMUL;

  /*  XCR0[1] SSE state, [2] YMM state, [5] opmask, [6] ZMM_Hi256,
      [7] Hi16_ZMM. Reading XCR0 at all requires OSXSAVE, so the guard is
      not optional either: xgetbv itself faults without it.  */
  ymm = zmm = false;
  if ((r[2] & (1u << 27)) && (r[2] & (1u << 28))) {
    u64 x = xpar_xcr0();
    ymm = (x & 0x6ULL) == 0x6ULL;
    zmm = ymm && (x & 0xe0ULL) == 0xe0ULL;
  }

  if (maxleaf >= 7) {
    xpar_cpuid(7, 0, r);
    if (ymm && (r[1] & (1u << 5)))  f |= XPAR_CPU_AVX2;
    if (zmm && (r[1] & (1u << 16))) f |= XPAR_CPU_AVX512F;
    if (zmm && (r[1] & (1u << 30))) f |= XPAR_CPU_AVX512BW;
    if (zmm && (r[1] & (1u << 31))) f |= XPAR_CPU_AVX512VL;
    if (r[2] & (1u << 8))           f |= XPAR_CPU_GFNI;
    if (zmm && (r[2] & (1u << 1)))  f |= XPAR_CPU_VBMI;
    if (ymm && (r[2] & (1u << 10))) f |= XPAR_CPU_VPCLMUL;
  }
  return f;
}

/*  The ladder mirrors the per-ISA libraries Makefile.am builds, and each
    tier requires exactly the features its kernels were compiled with: the
    sse42 row demands PCLMUL only when the crc32c kernel was built with
    -mpclmul, because otherwise there is no pclmulqdq in the binary to
    fault on and requiring it would refuse a tier that works.  */

#define XPAR_T_SSSE3  (XPAR_CPU_SSSE3)
#if defined(HAVE_PCLMUL)
  #define XPAR_T_SSE42  (XPAR_T_SSSE3 | XPAR_CPU_SSE42 | XPAR_CPU_PCLMUL)
#else
  #define XPAR_T_SSE42  (XPAR_T_SSSE3 | XPAR_CPU_SSE42)
#endif
#define XPAR_T_AVX2    (XPAR_T_SSE42 | XPAR_CPU_AVX2)
#define XPAR_T_GFNI    (XPAR_T_AVX2  | XPAR_CPU_GFNI)
#define XPAR_T_AVX512  (XPAR_T_AVX2  | XPAR_CPU_AVX512F |                     \
                        XPAR_CPU_AVX512BW | XPAR_CPU_AVX512VL)
#define XPAR_T_GFNI512 (XPAR_T_AVX512 | XPAR_CPU_GFNI)
#define XPAR_T_VBMI    (XPAR_T_AVX2 | XPAR_CPU_AVX512F |                    \
                        XPAR_CPU_AVX512BW | XPAR_CPU_VBMI)

const xpar_cpu_tier xpar_cpu_tier_table[] = {
  { "scalar",  0              },
#if defined(HAVE_SSSE3)
  { "ssse3",   XPAR_T_SSSE3   },
#endif
#if defined(HAVE_SSE42)
  { "sse42",   XPAR_T_SSE42   },
#endif
#if defined(HAVE_AVX2)
  { "avx2",    XPAR_T_AVX2    },
#endif
#if defined(HAVE_GFNI)
  { "gfni256", XPAR_T_GFNI    },
#endif
#if defined(HAVE_GFNI512)
  { "gfni512", XPAR_T_GFNI512 },
#endif
#if defined(HAVE_VBMI)
  { "vbmi512", XPAR_T_VBMI    },
#endif
  { NULL,      0              }
};

/*  The NULL row is a terminator for readers that want one and is not a
    tier, hence the -1.  */
const int xpar_cpu_tier_table_n =
  (int) (sizeof xpar_cpu_tier_table / sizeof xpar_cpu_tier_table[0]) - 1;

#else

/*  Not an x86 target. ISO C forbids a translation unit with no external
    declaration at all, so one typedef stands in for the whole file.  */
typedef int xpar_cpu_x86_not_here;

#endif
