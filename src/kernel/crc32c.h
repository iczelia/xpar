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

/*  CRC-32C (Castagnoli), range combination, rolling windows, and hardware
    dispatch.  Polynomial 0x1EDC6F41 is reflected as 0x82F63B78; init and
    xorout are 0xFFFFFFFF.  */

#ifndef XPAR_CRC32C_H
#define XPAR_CRC32C_H

#include "common.h"

/*  Running CRC over successive buffers: pass 0 for the first call and the
    previous return value afterwards, so that
    crc32c(crc32c(0, a, na), b, nb) is the CRC of a followed by b.  */
u32 xpar_crc32c(u32 crc, const void * buf, sz n);

/*  CRC of A followed by B from the CRC of each and the length of B. The
    length of A is not needed and does not enter the result.  */
u32 xpar_crc32c_combine(u32 a, u32 b, u64 len_b);

/*  Builds the tables and the cached shift operators.  */
void xpar_crc32c_init(void);

/*  Name of the dispatched kernel, for `info` and the selftest.  */
const char * xpar_crc32c_variant(void);

/*  The rolling window.  */

typedef struct {
  u32 enter[256];
  u32 leave[256];
  u32 fold;
  sz  window;
} xpar_crc32c_roll;

void xpar_crc32c_roll_init(xpar_crc32c_roll *, sz window);

/*  CRC of [i+1, i+1+W) given the CRC of [i, i+W), the byte at i that is
    leaving, and the byte at i+W that is entering.  */
static inline u32 xpar_crc32c_roll_step(const xpar_crc32c_roll * r, u32 crc,
                                        u8 leaving, u8 entering) {
  u32 v = crc ^ r->fold ^ r->leave[leaving];
  return ((v >> 8) ^ r->enter[(v ^ entering) & 0xFF]) ^ r->fold;
}

/*  Internal: shared between crc32c.c and the ISA variants.  */

/*  Bytes per chain in the three-way split.  */
#define XPAR_CRC32C_LONG   8192
#define XPAR_CRC32C_SHORT   256

/*  Slice-by-eight tables, filled by xpar_crc32c_init.  */
extern u32 xpar_crc32c_tab[8][256];

/*  Advance a bare register through `n` zero bytes, which is
    multiplication by x^(8n) modulo the polynomial.  */
u32 xpar_crc32c_shift(u32 crc, u64 n);

/*  Fold three bare registers covering three consecutive equal-length
    runs into one.  */
u32 xpar_crc32c_join3_long (u32 a, u32 b, u32 c);
u32 xpar_crc32c_join3_short(u32 a, u32 b, u32 c);

u32 xpar_crc32c_scalar(u32 crc, const u8 * p, sz n);
#ifdef HAVE_SSE42
u32 xpar_crc32c_sse42 (u32 crc, const u8 * p, sz n);
#endif
#ifdef HAVE_ARM_CRC32
u32 xpar_crc32c_arm   (u32 crc, const u8 * p, sz n);
#endif
#ifdef HAVE_VPCLMUL
u32 xpar_crc32c_vpclmul(u32 crc, const u8 * p, sz n);
#endif

#endif
