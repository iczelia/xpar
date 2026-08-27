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

/*  CRC-32C tables, GF(2) combination, rolling windows, and dispatch.  */

#include "crc32c.h"

#include "platform/port-cpu.h"

/*  Reflected CRC-32C polynomial: 0x1EDC6F41 read from the other end, the
    x^32 term implicit. Not 0xEDB88320, which is the zip polynomial and
    would produce an archive nothing else can read.  */
#define XPAR_CRC32C_POLY 0x82F63B78u

u32 xpar_crc32c_tab[8][256];

static u32 xpar_crc_apply(const u32 * op, u32 v) {
  u32 r = 0;
  int k = 0;
  while (v) {
    if (v & 1) r ^= op[k];
    v >>= 1;  k++;
  }
  return r;
}

static void xpar_crc_compose(u32 * out, const u32 * a, const u32 * b) {
  Fi(32, out[i] = xpar_crc_apply(a, b[i]))
}

static void xpar_crc_op_bit(u32 * op) {
  op[0] = XPAR_CRC32C_POLY;
  Fi0(32, 1, op[i] = (u32) 1 << (i - 1))
}

/*  The operator for `n` zero bytes, by square and multiply over the
    bits of n.  */
static void xpar_crc_op_zeros(u32 * op, u64 n) {
  u32 cur[32], acc[32], tmp[32];
  xpar_crc_op_bit(cur);
  Fi(3, xpar_crc_compose(tmp, cur, cur);  Fj(32, cur[j] = tmp[j]))
  Fi(32, acc[i] = (u32) 1 << i)
  while (n) {
    if (n & 1) {
      xpar_crc_compose(tmp, cur, acc);
      Fi(32, acc[i] = tmp[i])
    }
    n >>= 1;
    if (n) {
      xpar_crc_compose(tmp, cur, cur);
      Fi(32, cur[i] = tmp[i])
    }
  }
  Fi(32, op[i] = acc[i])
}

/*  Dispatch and one-time state.  */

static u32 xpar_crc_op_long[32], xpar_crc_op_long2[32];
static u32 xpar_crc_op_short[32], xpar_crc_op_short2[32];

typedef u32 (* xpar_crc_fn)(u32 crc, const u8 * p, sz n);

static xpar_crc_fn   xpar_crc_impl = xpar_crc32c_scalar;
static const char *  xpar_crc_name = "scalar";
static bool          xpar_crc_ready;
static u32           xpar_crc_seen;  /*  Features the choice was made on.  */

static void xpar_crc_dispatch(u32 f) {
  xpar_crc_impl = xpar_crc32c_scalar;  xpar_crc_name = "scalar";
#ifdef HAVE_SSE42
  if (f & XPAR_CPU_SSE42) {
    xpar_crc_impl = xpar_crc32c_sse42;  xpar_crc_name = "sse4.2";
  }
#endif
#ifdef HAVE_VPCLMUL
  if ((f & (XPAR_CPU_SSE42 | XPAR_CPU_AVX2 | XPAR_CPU_VPCLMUL)) ==
           (XPAR_CPU_SSE42 | XPAR_CPU_AVX2 | XPAR_CPU_VPCLMUL)) {
    xpar_crc_impl = xpar_crc32c_vpclmul;  xpar_crc_name = "vpclmul";
  }
#endif
#ifdef HAVE_ARM_CRC32
  if (f & XPAR_CPU_ARMCRC) {
    xpar_crc_impl = xpar_crc32c_arm;  xpar_crc_name = "armv8";
  }
#endif
  xpar_crc_seen = f;
}

void xpar_crc32c_init(void) {
  u32 f = xpar_cpu_features();
  if (xpar_crc_ready && f == xpar_crc_seen) return;
  if (xpar_crc_ready) {
    xpar_crc_dispatch(f);
    return;
  }
  Fi(256, u32 c = (u32) i;
          Fj(8, c = (c & 1) ? (c >> 1) ^ XPAR_CRC32C_POLY : c >> 1)
          xpar_crc32c_tab[0][i] = c)
  Fi(256, Fj0(8, 1, u32 v = xpar_crc32c_tab[j - 1][i];
                    xpar_crc32c_tab[j][i] =
                      (v >> 8) ^ xpar_crc32c_tab[0][v & 0xFF]))
  xpar_crc_op_zeros(xpar_crc_op_long,   XPAR_CRC32C_LONG);
  xpar_crc_op_zeros(xpar_crc_op_long2,  2 * XPAR_CRC32C_LONG);
  xpar_crc_op_zeros(xpar_crc_op_short,  XPAR_CRC32C_SHORT);
  xpar_crc_op_zeros(xpar_crc_op_short2, 2 * XPAR_CRC32C_SHORT);
  xpar_crc_dispatch(f);
  xpar_crc_ready = true;
}

const char * xpar_crc32c_variant(void) {
  xpar_crc32c_init();
  return xpar_crc_name;
}

u32 xpar_crc32c(u32 crc, const void * buf, sz n) {
  xpar_crc32c_init();
  return ~xpar_crc_impl(~crc, (const u8 *) buf, n);
}

u32 xpar_crc32c_shift(u32 crc, u64 n) {
  u32 op[32];
  xpar_crc_op_zeros(op, n);
  return xpar_crc_apply(op, crc);
}

/*  Zero-extending A by the length of B lines the two up, and the stored
    form's constant offset cancels, so the length of A never appears.  */
u32 xpar_crc32c_combine(u32 a, u32 b, u64 len_b) {
  xpar_crc32c_init();
  return xpar_crc32c_shift(a, len_b) ^ b;
}

void xpar_crc32c_shift_op(u32 op[XPAR_CRC32C_OP_WORDS], u64 n) {
  xpar_crc32c_init();
  xpar_crc_op_zeros(op, n);
}

u32 xpar_crc32c_combine_op(const u32 op[XPAR_CRC32C_OP_WORDS],
                           u32 a, u32 b) {
  return xpar_crc_apply(op, a) ^ b;
}

u32 xpar_crc32c_join3_long(u32 a, u32 b, u32 c) {
  return xpar_crc_apply(xpar_crc_op_long2, a) ^
         xpar_crc_apply(xpar_crc_op_long, b) ^ c;
}

u32 xpar_crc32c_join3_short(u32 a, u32 b, u32 c) {
  return xpar_crc_apply(xpar_crc_op_short2, a) ^
         xpar_crc_apply(xpar_crc_op_short, b) ^ c;
}

/*  The rolling window.  */

void xpar_crc32c_roll_init(xpar_crc32c_roll * r, sz window) {
  u32 op[32];
  xpar_crc32c_init();
  xpar_assert(window >= 1);
  xpar_crc_op_zeros(op, (u64) window - 1);
  Fi(256, r->enter[i] = xpar_crc32c_tab[0][i];
          r->leave[i] = xpar_crc_apply(op, xpar_crc32c_tab[0][i]))
  r->fold = 0xFFFFFFFFu ^ xpar_crc32c_shift(0xFFFFFFFFu, (u64) window);
}
