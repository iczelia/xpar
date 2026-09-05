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

/*  CPU features and cumulative SIMD tiers.  */

#ifndef XPAR_PORT_CPU_H
#define XPAR_PORT_CPU_H

#include "port.h"

#define XPAR_CPU_SSSE3     (1U << 0)   /*  pshufb  */
#define XPAR_CPU_SSE42     (1U << 1)   /*  the crc32 instruction  */
#define XPAR_CPU_PCLMUL    (1U << 2)
#define XPAR_CPU_AVX2      (1U << 3)
#define XPAR_CPU_AVX512F   (1U << 4)
#define XPAR_CPU_AVX512BW  (1U << 5)
#define XPAR_CPU_AVX512VL  (1U << 6)
#define XPAR_CPU_GFNI      (1U << 7)   /*  gf2p8affineqb  */
#define XPAR_CPU_NEON      (1U << 8)
#define XPAR_CPU_ARMCRC    (1U << 9)   /*  ARMv8 CRC32/CRC32C extension  */
#define XPAR_CPU_VBMI      (1U << 10)  /*  AVX-512 VPERMB  */
#define XPAR_CPU_VPCLMUL   (1U << 11)  /*  vector carry-less multiply  */
#define XPAR_CPU_SVE      (1U << 13)
#define XPAR_CPU_RVV       (1U << 14)
#define XPAR_CPU_RVVCLMUL  (1U << 15)
#define XPAR_CPU_VSX       (1U << 16)

#define XPAR_CPU_ALL       0xffffffffU

/*  Probed features restricted by xpar_cpu_force. Call before threading.  */
u32 xpar_cpu_features(void);

/*  Restrict features; XPAR_CPU_ALL restores the probe. Never widen.  */
void xpar_cpu_force(u32 allow);

/*  The tier ladder.  */

typedef struct {
  const char * name;   /*  the spelling --simd= accepts  */
  u32          need;   /*  cumulative: every feature at or below the tier  */
} xpar_cpu_tier;

int                   xpar_cpu_tier_count(void);
const xpar_cpu_tier * xpar_cpu_tier_at  (int i);

/*  Tier index, or -1.  */
int  xpar_cpu_tier_find(const char * name);

/*  Whether this CPU can run tier `i`.  */
bool xpar_cpu_tier_usable(int i);

/*  Highest usable tier; always at least scalar.  */
int  xpar_cpu_tier_best(void);

/*  Per-architecture feature probe and tier table.  */

u32 xpar_cpu_probe(void);

extern const xpar_cpu_tier xpar_cpu_tier_table[];
extern const int           xpar_cpu_tier_table_n;

#endif
