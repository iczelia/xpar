/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  CPU feature detection and cumulative SIMD tiers.

    Only built kernels appear.  Forced tiers mask probed features down and
    can never enable unsupported instructions.  */

#ifndef XPAR_PORT_CPU_H
#define XPAR_PORT_CPU_H

#include "port.h"

#define XPAR_CPU_SSSE3     (1u << 0)   /*  pshufb  */
#define XPAR_CPU_SSE42     (1u << 1)   /*  the crc32 instruction  */
#define XPAR_CPU_PCLMUL    (1u << 2)
#define XPAR_CPU_AVX2      (1u << 3)
#define XPAR_CPU_AVX512F   (1u << 4)
#define XPAR_CPU_AVX512BW  (1u << 5)
#define XPAR_CPU_AVX512VL  (1u << 6)
#define XPAR_CPU_GFNI      (1u << 7)   /*  gf2p8affineqb  */
#define XPAR_CPU_NEON      (1u << 8)
#define XPAR_CPU_ARMCRC    (1u << 9)   /*  ARMv8 CRC32/CRC32C extension  */
#define XPAR_CPU_VBMI      (1u << 10)  /*  AVX-512 VPERMB  */
#define XPAR_CPU_VPCLMUL   (1u << 11)  /*  vector carry-less multiply  */
#define XPAR_CPU_PMULL     (1u << 12)  /*  ARM polynomial multiply  */
#define XPAR_CPU_SVE2      (1u << 13)
#define XPAR_CPU_RVV       (1u << 14)
#define XPAR_CPU_RVVCLMUL  (1u << 15)
#define XPAR_CPU_VSX       (1u << 16)

#define XPAR_CPU_ALL       0xffffffffu

/*  The probed mask intersected with whatever xpar_cpu_force last allowed.
    Probing happens on the first call. The cache is a plain store with no
    barrier, so the first call must happen before any thread pool exists;
    the dispatcher tables are built at startup, which satisfies that.  */
u32 xpar_cpu_features(void);

/*  Restrict the answer to `allow`. XPAR_CPU_ALL restores the probed set,
    which is what a tier sweep does between iterations. Never widens: a
    forced tier the CPU cannot run stays off.  */
void xpar_cpu_force(u32 allow);

/*  The tier ladder.  */

typedef struct {
  const char * name;   /*  the spelling --simd= accepts  */
  u32          need;   /*  cumulative: every feature at or below the tier  */
} xpar_cpu_tier;

int                   xpar_cpu_tier_count(void);
const xpar_cpu_tier * xpar_cpu_tier_at  (int i);

/*  Index of `name` in the ladder, or -1. Unknown names are the caller's to
    report, with xpar_cpu_tier_at to list what this build does have.  */
int  xpar_cpu_tier_find(const char * name);

/*  True when this CPU can run tier `i`. A compiled tier is not a runnable
    one: a binary built on an AVX-512 host must still start on a Core 2.  */
bool xpar_cpu_tier_usable(int i);

/*  Highest usable tier, which is what the dispatcher selects by default.
    Always >= 0 because tier 0 is `scalar` and needs nothing.  */
int  xpar_cpu_tier_best(void);

/*  Per-architecture back end.
    Provided by port-cpu-x86.c, port-cpu-arm.c or port-cpu-other.c, exactly
    one of which is live per target; the other two compile to nothing. The
    caching, forcing and ladder queries above are shared and live in
    port-cpu-other.c, which is the file that is compiled unconditionally on
    every host.  */

u32 xpar_cpu_probe(void);

extern const xpar_cpu_tier xpar_cpu_tier_table[];
extern const int           xpar_cpu_tier_table_n;

#endif
