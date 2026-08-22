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

/*  xpar: RISC-V V table-gather Galois-field kernels.  */

#include "gf-rvv-int.h"

#include <riscv_vector.h>

static vuint8m1_t rv_tab(const u8 * p) {
  sz vl = __riscv_vsetvl_e8m1(16);
  return __riscv_vle8_v_u8m1(p, vl);
}

static vuint8m1_t rv_mul8_v(vuint8m1_t v, const xpar_gf8_coef * m, sz vl) {
  vuint8m1_t lo = rv_tab(m->tab), hi = rv_tab(m->tab + 16);
  vuint8m1_t mask = __riscv_vand_vx_u8m1(v, 15, vl);
  vuint8m1_t upper = __riscv_vsrl_vx_u8m1(v, 4, vl);
  return __riscv_vxor_vv_u8m1(
    __riscv_vrgather_vv_u8m1(lo, mask, vl),
    __riscv_vrgather_vv_u8m1(hi, upper, vl), vl);
}

void xpar_rvv_mac8(u8 * d, const u8 * s, sz n,
                   const xpar_gf8_coef * m) {
  while (n) {
    sz vl = __riscv_vsetvl_e8m1(MIN(n, (sz) 16));
    vuint8m1_t p = rv_mul8_v(__riscv_vle8_v_u8m1(s, vl), m, vl);
    p = __riscv_vxor_vv_u8m1(p, __riscv_vle8_v_u8m1(d, vl), vl);
    __riscv_vse8_v_u8m1(d, p, vl);
    d += vl;  s += vl;  n -= vl;
  }
}

void xpar_rvv_mac8x2(u8 * const d[2], const u8 * s, sz n,
                     const xpar_gf8_coef m[2]) {
  u8 * dp[2];
  for (u32 j = 0; j < 2; j++) dp[j] = d[j];
  while (n) {
    sz vl = __riscv_vsetvl_e8m1(MIN(n, (sz) 16));
    vuint8m1_t v = __riscv_vle8_v_u8m1(s, vl);
    for (u32 j = 0; j < 2; j++) {
      vuint8m1_t prod = rv_mul8_v(v, &m[j], vl);
      prod = __riscv_vxor_vv_u8m1(
        prod, __riscv_vle8_v_u8m1(dp[j], vl), vl);
      __riscv_vse8_v_u8m1(dp[j], prod, vl);
      dp[j] += vl;
    }
    s += vl;  n -= vl;
  }
}

void xpar_rvv_mul8(u8 * d, const u8 * s, sz n,
                   const xpar_gf8_coef * m) {
  while (n) {
    sz vl = __riscv_vsetvl_e8m1(MIN(n, (sz) 16));
    vuint8m1_t p = rv_mul8_v(__riscv_vle8_v_u8m1(s, vl), m, vl);
    __riscv_vse8_v_u8m1(d, p, vl);
    d += vl;  s += vl;  n -= vl;
  }
}

static void rv_mul16_v(vuint8m1_t lo, vuint8m1_t hi,
                       const xpar_gf16_coef * m, sz vl,
                       vuint8m1_t * ol, vuint8m1_t * oh) {
  vuint8m1_t n0 = __riscv_vand_vx_u8m1(lo, 15, vl);
  vuint8m1_t n1 = __riscv_vsrl_vx_u8m1(lo, 4, vl);
  vuint8m1_t n2 = __riscv_vand_vx_u8m1(hi, 15, vl);
  vuint8m1_t n3 = __riscv_vsrl_vx_u8m1(hi, 4, vl);
  vuint8m1_t t0 = rv_tab(m->tab[0]), t1 = rv_tab(m->tab[1]);
  vuint8m1_t t2 = rv_tab(m->tab[2]), t3 = rv_tab(m->tab[3]);
  vuint8m1_t t4 = rv_tab(m->tab[4]), t5 = rv_tab(m->tab[5]);
  vuint8m1_t t6 = rv_tab(m->tab[6]), t7 = rv_tab(m->tab[7]);
#define RV_GATHER(t, n) __riscv_vrgather_vv_u8m1((t), (n), vl)
  *ol = __riscv_vxor_vv_u8m1(
    __riscv_vxor_vv_u8m1(RV_GATHER(t0, n0), RV_GATHER(t2, n1), vl),
    __riscv_vxor_vv_u8m1(RV_GATHER(t4, n2), RV_GATHER(t6, n3), vl), vl);
  *oh = __riscv_vxor_vv_u8m1(
    __riscv_vxor_vv_u8m1(RV_GATHER(t1, n0), RV_GATHER(t3, n1), vl),
    __riscv_vxor_vv_u8m1(RV_GATHER(t5, n2), RV_GATHER(t7, n3), vl), vl);
#undef RV_GATHER
}

static void rv_mac16(u8 * d, const u8 * s, sz n,
                     const xpar_gf16_coef * m) {
  sz symbols = n / 2;
  while (symbols) {
    sz vl = __riscv_vsetvl_e8m1(MIN(symbols, (sz) 16));
    vuint8m1_t lo = __riscv_vlse8_v_u8m1(s, 2, vl);
    vuint8m1_t hi = __riscv_vlse8_v_u8m1(s + 1, 2, vl), ol, oh;
    rv_mul16_v(lo, hi, m, vl, &ol, &oh);
    ol = __riscv_vxor_vv_u8m1(ol, __riscv_vlse8_v_u8m1(d, 2, vl), vl);
    oh = __riscv_vxor_vv_u8m1(oh,
                               __riscv_vlse8_v_u8m1(d + 1, 2, vl), vl);
    __riscv_vsse8_v_u8m1(d, 2, ol, vl);
    __riscv_vsse8_v_u8m1(d + 1, 2, oh, vl);
    d += vl * 2;  s += vl * 2;  symbols -= vl;
  }
}

static void rv_mac16x2(u8 * const d[2], const u8 * s, sz n,
                       const xpar_gf16_coef m[2]) {
  rv_mac16(d[0], s, n, &m[0]);
  rv_mac16(d[1], s, n, &m[1]);
}

static void rv_mul16_region(u8 * d, const u8 * s, sz n,
                            const xpar_gf16_coef * m) {
  sz symbols = n / 2;
  while (symbols) {
    sz vl = __riscv_vsetvl_e8m1(MIN(symbols, (sz) 16));
    vuint8m1_t lo = __riscv_vlse8_v_u8m1(s, 2, vl);
    vuint8m1_t hi = __riscv_vlse8_v_u8m1(s + 1, 2, vl), ol, oh;
    rv_mul16_v(lo, hi, m, vl, &ol, &oh);
    __riscv_vsse8_v_u8m1(d, 2, ol, vl);
    __riscv_vsse8_v_u8m1(d + 1, 2, oh, vl);
    d += vl * 2;  s += vl * 2;  symbols -= vl;
  }
}

void xpar_rvv_xor2(u8 * d, const u8 * s, sz n) {
  while (n) {
    sz vl = __riscv_vsetvl_e8m1(n);
    vuint8m1_t v = __riscv_vxor_vv_u8m1(
      __riscv_vle8_v_u8m1(d, vl), __riscv_vle8_v_u8m1(s, vl), vl);
    __riscv_vse8_v_u8m1(d, v, vl);
    d += vl;  s += vl;  n -= vl;
  }
}

void xpar_rvv_xor3(u8 * d, const u8 * a, const u8 * b, sz n) {
  while (n) {
    sz vl = __riscv_vsetvl_e8m1(n);
    vuint8m1_t v = __riscv_vxor_vv_u8m1(
      __riscv_vle8_v_u8m1(a, vl), __riscv_vle8_v_u8m1(b, vl), vl);
    __riscv_vse8_v_u8m1(d, v, vl);
    d += vl;  a += vl;  b += vl;  n -= vl;
  }
}

#define RV_FFT8(name, inverse)                                               \
void name(u8 * x, u8 * y, sz n, const xpar_gf8_coef * m) {                  \
  while (n) {                                                               \
    sz vl = __riscv_vsetvl_e8m1(MIN(n, (sz) 16));                           \
    vuint8m1_t a = __riscv_vle8_v_u8m1(x, vl);                              \
    vuint8m1_t b = __riscv_vle8_v_u8m1(y, vl);                              \
    if (inverse) {                                                          \
      b = __riscv_vxor_vv_u8m1(b, a, vl);                                  \
      a = __riscv_vxor_vv_u8m1(a, rv_mul8_v(b, m, vl), vl);                 \
    } else {                                                                \
      a = __riscv_vxor_vv_u8m1(a, rv_mul8_v(b, m, vl), vl);                 \
      b = __riscv_vxor_vv_u8m1(b, a, vl);                                  \
    }                                                                       \
    __riscv_vse8_v_u8m1(x, a, vl);  __riscv_vse8_v_u8m1(y, b, vl);         \
    x += vl;  y += vl;  n -= vl;                                           \
  }                                                                         \
}

RV_FFT8(xpar_rvv_fft8, 0)
RV_FFT8(xpar_rvv_ifft8, 1)

#define RV_FFT16(name, inverse)                                              \
static void name(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m) {          \
  sz symbols = n / 2;                                                       \
  while (symbols) {                                                         \
    sz vl = __riscv_vsetvl_e8m1(MIN(symbols, (sz) 16));                     \
    vuint8m1_t xl = __riscv_vlse8_v_u8m1(x, 2, vl);                         \
    vuint8m1_t xh = __riscv_vlse8_v_u8m1(x + 1, 2, vl);                     \
    vuint8m1_t yl = __riscv_vlse8_v_u8m1(y, 2, vl);                         \
    vuint8m1_t yh = __riscv_vlse8_v_u8m1(y + 1, 2, vl), ml, mh;             \
    if (inverse) {                                                          \
      yl = __riscv_vxor_vv_u8m1(yl, xl, vl);                               \
      yh = __riscv_vxor_vv_u8m1(yh, xh, vl);                               \
      rv_mul16_v(yl, yh, m, vl, &ml, &mh);                                 \
      xl = __riscv_vxor_vv_u8m1(xl, ml, vl);                               \
      xh = __riscv_vxor_vv_u8m1(xh, mh, vl);                               \
    } else {                                                                \
      rv_mul16_v(yl, yh, m, vl, &ml, &mh);                                 \
      xl = __riscv_vxor_vv_u8m1(xl, ml, vl);                               \
      xh = __riscv_vxor_vv_u8m1(xh, mh, vl);                               \
      yl = __riscv_vxor_vv_u8m1(yl, xl, vl);                               \
      yh = __riscv_vxor_vv_u8m1(yh, xh, vl);                               \
    }                                                                       \
    __riscv_vsse8_v_u8m1(x, 2, xl, vl);                                    \
    __riscv_vsse8_v_u8m1(x + 1, 2, xh, vl);                                \
    __riscv_vsse8_v_u8m1(y, 2, yl, vl);                                    \
    __riscv_vsse8_v_u8m1(y + 1, 2, yh, vl);                                \
    x += vl * 2;  y += vl * 2;  symbols -= vl;                             \
  }                                                                         \
}

RV_FFT16(rv_fft16, 0)
RV_FFT16(rv_ifft16, 1)

const xpar_gf_kernels xpar_gf_kernels_rvv_shuffle = {
  "rvv-shuffle", xpar_rvv_mac8, xpar_rvv_mac8x2,
  rv_mac16, rv_mac16x2,
  xpar_rvv_mul8, rv_mul16_region,
  xpar_rvv_xor2, xpar_rvv_xor3, xpar_rvv_fft8, rv_fft16,
  xpar_rvv_ifft8, rv_ifft16
};
