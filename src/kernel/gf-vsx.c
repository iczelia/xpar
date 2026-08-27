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

/*  xpar: PowerPC VSX Galois-field region kernels.  */

#include "gf.h"

#include <altivec.h>

typedef __vector unsigned char vx_u8;

static vx_u8 vx_splat(u8 x) { return vec_splats((unsigned char) x); }

/*  vec_perm implements the split-table lookup used by other shuffle tiers.  */
typedef struct { vx_u8 lo, hi; } vx_c8;
typedef struct { vx_u8 t[8]; } vx_c16;

static vx_c8 vx_load8(const xpar_gf8_coef * m) {
  vx_c8 c;
  c.lo = vec_vsx_ld(0, m->tab);
  c.hi = vec_vsx_ld(0, m->tab + 16);
  return c;
}

static vx_c16 vx_load16(const xpar_gf16_coef * m) {
  vx_c16 c;
  u32 i;
  for (i = 0; i < 8; i++) c.t[i] = vec_vsx_ld(0, m->tab[i]);
  return c;
}

static vx_u8 vx_mul8(vx_u8 v, vx_c8 c) {
  vx_u8 nl = vec_and(v, vx_splat(15));
  vx_u8 nh = vec_sr(v, vx_splat(4));
  return vec_perm(c.lo, c.lo, nl) ^ vec_perm(c.hi, c.hi, nh);
}

/*  Split little-endian GF16 symbols into byte planes, then rejoin them.  */
static void vx_mul16(vx_u8 v0, vx_u8 v1, vx_c16 c,
                     vx_u8 * o0, vx_u8 * o1) {
  static const u8 even_bytes[16] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
  };
  static const u8 odd_bytes[16] = {
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
  };
  vx_u8 pe = vec_vsx_ld(0, even_bytes), po = vec_vsx_ld(0, odd_bytes);
  vx_u8 lo = vec_perm(v0, v1, pe), hi = vec_perm(v0, v1, po);
  vx_u8 m15 = vx_splat(15), sh4 = vx_splat(4);
  vx_u8 n0 = vec_and(lo, m15), n1 = vec_sr(lo, sh4);
  vx_u8 n2 = vec_and(hi, m15), n3 = vec_sr(hi, sh4);
  vx_u8 rl = (vec_perm(c.t[0], c.t[0], n0) ^ vec_perm(c.t[2], c.t[2], n1)) ^
             (vec_perm(c.t[4], c.t[4], n2) ^ vec_perm(c.t[6], c.t[6], n3));
  vx_u8 rh = (vec_perm(c.t[1], c.t[1], n0) ^ vec_perm(c.t[3], c.t[3], n1)) ^
             (vec_perm(c.t[5], c.t[5], n2) ^ vec_perm(c.t[7], c.t[7], n3));
  *o0 = vec_mergeh(rl, rh);
  *o1 = vec_mergel(rl, rh);
}

/*  Cache the coefficient because destination writes may alias it.  */
static void vx_mac8(u8 * d, const u8 * s, sz n,
                    const xpar_gf8_coef * m) {
  const vx_c8 c = vx_load8(m);
  sz i = 0;
  for (; i + 16 <= n; i += 16) {
    vx_u8 dv = vec_vsx_ld(0, d + i), sv = vec_vsx_ld(0, s + i);
    vec_vsx_st(dv ^ vx_mul8(sv, c), 0, d + i);
  }
  xpar_gf8_mac_ref(d + i, s + i, n - i, m->c);
}

static void vx_mac8x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf8_coef m[2]) {
  const vx_c8 c0 = vx_load8(&m[0]), c1 = vx_load8(&m[1]);
  sz i = 0;
  for (; i + 16 <= n; i += 16) {
    vx_u8 v = vec_vsx_ld(0, s + i);
    vec_vsx_st(vec_vsx_ld(0, d[0] + i) ^ vx_mul8(v, c0), 0, d[0] + i);
    vec_vsx_st(vec_vsx_ld(0, d[1] + i) ^ vx_mul8(v, c1), 0, d[1] + i);
  }
  for (u32 j = 0; j < 2; j++)
    xpar_gf8_mac_ref(d[j] + i, s + i, n - i, m[j].c);
}

static void vx_mul8_region(u8 * d, const u8 * s, sz n,
                           const xpar_gf8_coef * m) {
  const vx_c8 c = vx_load8(m);
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vec_vsx_st(vx_mul8(vec_vsx_ld(0, s + i), c), 0, d + i);
  xpar_gf8_mul_ref(d + i, s + i, n - i, m->c);
}

static void vx_mac16(u8 * d, const u8 * s, sz n,
                     const xpar_gf16_coef * m) {
  const vx_c16 c = vx_load16(m);
  sz i = 0;
  for (; i + 32 <= n; i += 32) {
    vx_u8 a, b;
    vx_mul16(vec_vsx_ld(0, s + i), vec_vsx_ld(0, s + i + 16),
             c, &a, &b);
    vec_vsx_st(vec_vsx_ld(0, d + i) ^ a, 0, d + i);
    vec_vsx_st(vec_vsx_ld(0, d + i + 16) ^ b, 0, d + i + 16);
  }
  xpar_gf16_mac_ref(d + i, s + i, n - i, m->c);
}

static void vx_mac16x2(u8 * const d[2], const u8 * s, sz n,
                       const xpar_gf16_coef m[2]) {
  vx_mac16(d[0], s, n, &m[0]);
  vx_mac16(d[1], s, n, &m[1]);
}

static void vx_mul16_region(u8 * d, const u8 * s, sz n,
                            const xpar_gf16_coef * m) {
  const vx_c16 c = vx_load16(m);
  sz i = 0;
  for (; i + 32 <= n; i += 32) {
    vx_u8 a, b;
    vx_mul16(vec_vsx_ld(0, s + i), vec_vsx_ld(0, s + i + 16),
             c, &a, &b);
    vec_vsx_st(a, 0, d + i);  vec_vsx_st(b, 0, d + i + 16);
  }
  xpar_gf16_mul_ref(d + i, s + i, n - i, m->c);
}

static void vx_xor2(u8 * d, const u8 * s, sz n) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vec_vsx_st(vec_vsx_ld(0, d + i) ^ vec_vsx_ld(0, s + i), 0, d + i);
  xpar_xor2_ref(d + i, s + i, n - i);
}

static void vx_xor3(u8 * d, const u8 * a, const u8 * b, sz n) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vec_vsx_st(vec_vsx_ld(0, a + i) ^ vec_vsx_ld(0, b + i), 0, d + i);
  xpar_xor3_ref(d + i, a + i, b + i, n - i);
}

#define VX_FFT8(name, ref, inverse)                                          \
static void name(u8 * x, u8 * y, sz n, const xpar_gf8_coef * m) {           \
  const vx_c8 c = vx_load8(m);                                              \
  sz i = 0;                                                                 \
  for (; i + 16 <= n; i += 16) {                                            \
    vx_u8 a = vec_vsx_ld(0, x + i), b = vec_vsx_ld(0, y + i);               \
    if (inverse) { b ^= a;  a ^= vx_mul8(b, c); }                           \
    else { a ^= vx_mul8(b, c);  b ^= a; }                                   \
    vec_vsx_st(a, 0, x + i);  vec_vsx_st(b, 0, y + i);                      \
  }                                                                         \
  ref(x + i, y + i, n - i, m->c);                                          \
}

#define VX_FFT16(name, ref, inverse)                                         \
static void name(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m) {          \
  const vx_c16 c = vx_load16(m);                                            \
  sz i = 0;                                                                 \
  for (; i + 32 <= n; i += 32) {                                            \
    vx_u8 x0 = vec_vsx_ld(0, x + i), x1 = vec_vsx_ld(0, x + i + 16);        \
    vx_u8 y0 = vec_vsx_ld(0, y + i), y1 = vec_vsx_ld(0, y + i + 16);        \
    vx_u8 a, b;                                                             \
    if (inverse) {                                                          \
      y0 ^= x0;  y1 ^= x1;  vx_mul16(y0, y1, c, &a, &b);                    \
      x0 ^= a;  x1 ^= b;                                                    \
    } else {                                                                \
      vx_mul16(y0, y1, c, &a, &b);  x0 ^= a;  x1 ^= b;                      \
      y0 ^= x0;  y1 ^= x1;                                                  \
    }                                                                       \
    vec_vsx_st(x0, 0, x + i);  vec_vsx_st(x1, 0, x + i + 16);              \
    vec_vsx_st(y0, 0, y + i);  vec_vsx_st(y1, 0, y + i + 16);              \
  }                                                                         \
  ref(x + i, y + i, n - i, m->c);                                          \
}

VX_FFT8(vx_fft8, xpar_gf8_fft2_ref, 0)
VX_FFT8(vx_ifft8, xpar_gf8_ifft2_ref, 1)
VX_FFT16(vx_fft16, xpar_gf16_fft2_ref, 0)
VX_FFT16(vx_ifft16, xpar_gf16_ifft2_ref, 1)

const xpar_gf_kernels xpar_gf_kernels_vsx = {
  "vsx", vx_mac8, vx_mac8x2, vx_mac16, vx_mac16x2,
  vx_mul8_region, vx_mul16_region,
  vx_xor2, vx_xor3, vx_fft8, vx_fft16, vx_ifft8, vx_ifft16
};
