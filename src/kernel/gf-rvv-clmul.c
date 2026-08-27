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

/*  xpar: RISC-V Zvbc carry-less multiply GF(2^16) kernels.  */

#include "gf-rvv-int.h"

#include <riscv_vector.h>

/*  GF16 symbols are little-endian; vector loads use host byte order.  */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&               \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
static vuint16mf4_t rc_byteswap16(vuint16mf4_t v, sz vl) {
  return __riscv_vor_vv_u16mf4(
    __riscv_vsll_vx_u16mf4(v, 8, vl),
    __riscv_vsrl_vx_u16mf4(v, 8, vl), vl);
}
#endif

static vuint16mf4_t rc_load16(const u16 * p, sz vl) {
  vuint16mf4_t v = __riscv_vle16_v_u16mf4(p, vl);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&               \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = rc_byteswap16(v, vl);
#endif
  return v;
}

static void rc_store16(u16 * p, vuint16mf4_t v, sz vl) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&               \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = rc_byteswap16(v, vl);
#endif
  __riscv_vse16_v_u16mf4(p, v, vl);
}

static vuint16mf4_t rc_mul16(vuint16mf4_t in, u16 c, sz vl) {
  vuint64m1_t v = __riscv_vzext_vf4_u64m1(in, vl);
  vuint64m1_t product = __riscv_vclmul_vx_u64m1(v, c, vl);
  vuint64m1_t high = __riscv_vsrl_vx_u64m1(product, 16, vl);
  vuint64m1_t low = __riscv_vand_vx_u64m1(product, 0xFFFF, vl);
  vuint64m1_t fold = __riscv_vxor_vv_u64m1(
    low, __riscv_vclmul_vx_u64m1(high, 0x2D, vl), vl);
  high = __riscv_vsrl_vx_u64m1(fold, 16, vl);
  fold = __riscv_vxor_vv_u64m1(
    __riscv_vand_vx_u64m1(fold, 0xFFFF, vl),
    __riscv_vclmul_vx_u64m1(high, 0x2D, vl), vl);
  return __riscv_vnsrl_wx_u16mf4(
    __riscv_vnsrl_wx_u32mf2(fold, 0, vl), 0, vl);
}

/*  Cache the coefficient because destination writes may alias it.  */
static void rc_mac16(u8 * db, const u8 * sb, sz n,
                     const xpar_gf16_coef * m) {
  u16 * d = (u16 *) (void *) db;
  const u16 * s = (const u16 *) (const void *) sb;
  const u16 c = m->c;
  sz symbols = n / 2;
  while (symbols) {
    sz vl = __riscv_vsetvl_e16mf4(symbols);
    vuint16mf4_t p = rc_mul16(rc_load16(s, vl), c, vl);
    p = __riscv_vxor_vv_u16mf4(p, rc_load16(d, vl), vl);
    rc_store16(d, p, vl);
    d += vl;  s += vl;  symbols -= vl;
  }
}

static void rc_mac16x2(u8 * const d[2], const u8 * s, sz n,
                       const xpar_gf16_coef m[2]) {
  rc_mac16(d[0], s, n, &m[0]);
  rc_mac16(d[1], s, n, &m[1]);
}

static void rc_mul16_region(u8 * db, const u8 * sb, sz n,
                            const xpar_gf16_coef * m) {
  u16 * d = (u16 *) (void *) db;
  const u16 * s = (const u16 *) (const void *) sb;
  const u16 c = m->c;
  sz symbols = n / 2;
  while (symbols) {
    sz vl = __riscv_vsetvl_e16mf4(symbols);
    vuint16mf4_t p = rc_mul16(rc_load16(s, vl), c, vl);
    rc_store16(d, p, vl);
    d += vl;  s += vl;  symbols -= vl;
  }
}

#define RC_FFT16(name, inverse)                                              \
static void name(u8 * xb, u8 * yb, sz n, const xpar_gf16_coef * m) {        \
  u16 * x = (u16 *) (void *) xb, * y = (u16 *) (void *) yb;                 \
  const u16 c = m->c;                                                       \
  sz symbols = n / 2;                                                       \
  while (symbols) {                                                         \
    sz vl = __riscv_vsetvl_e16mf4(symbols);                                 \
    vuint16mf4_t a = rc_load16(x, vl);                                      \
    vuint16mf4_t b = rc_load16(y, vl);                                      \
    if (inverse) {                                                          \
      b = __riscv_vxor_vv_u16mf4(b, a, vl);                                \
      a = __riscv_vxor_vv_u16mf4(a, rc_mul16(b, c, vl), vl);                \
    } else {                                                                \
      a = __riscv_vxor_vv_u16mf4(a, rc_mul16(b, c, vl), vl);                 \
      b = __riscv_vxor_vv_u16mf4(b, a, vl);                                \
    }                                                                       \
    rc_store16(x, a, vl);                                                   \
    rc_store16(y, b, vl);                                                   \
    x += vl;  y += vl;  symbols -= vl;                                     \
  }                                                                         \
}

RC_FFT16(rc_fft16, 0)
RC_FFT16(rc_ifft16, 1)

const xpar_gf_kernels xpar_gf_kernels_rvv_clmul = {
  "rvv-clmul", xpar_rvv_mac8, xpar_rvv_mac8x2,
  rc_mac16, rc_mac16x2,
  xpar_rvv_mul8, rc_mul16_region,
  xpar_rvv_xor2, xpar_rvv_xor3, xpar_rvv_fft8, rc_fft16,
  xpar_rvv_ifft8, rc_ifft16
};
