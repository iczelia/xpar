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

/*  NEON PMULL kernels, including GF(2^16) carry-less multiply.  */

#include "gf.h"
#include <arm_neon.h>

/*  Cache coefficients because destination writes may alias them.  */
typedef struct { uint8x16_t lo, hi; } nc_c8;
typedef struct { poly8x8_t  lo, hi; } nc_c16;

static nc_c8 nc_load8(const xpar_gf8_coef * m) {
  nc_c8 c;
  c.lo = vld1q_u8(m->tab);
  c.hi = vld1q_u8(m->tab + 16);
  return c;
}

static nc_c16 nc_load16(const xpar_gf16_coef * m) {
  nc_c16 c;
  c.lo = vdup_n_p8((poly8_t) m->c);
  c.hi = vdup_n_p8((poly8_t) (m->c >> 8));
  return c;
}

static uint8x16_t nc_mul8(uint8x16_t v, nc_c8 c) {
  uint8x16_t mask = vdupq_n_u8(15);
  return veorq_u8(vqtbl1q_u8(c.lo, vandq_u8(v, mask)),
                  vqtbl1q_u8(c.hi, vshrq_n_u8(v, 4)));
}

static uint16x8_t nc_clmul(uint8x8_t a, poly8x8_t b) {
  return vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(a), b));
}

/*  A 16x16 carry-less product is assembled from four 8x8 PMULLs. Reduction
    uses x^16 = x^5+x^3+x^2+1 twice; after the first fold only four high
    bits remain, so the second byte PMULL completes it.  */
static uint8x16_t nc_mul16(uint8x16_t v, nc_c16 c) {
  const poly8x8_t k = vdup_n_p8((poly8_t) 0x2D);
  uint8x8x2_t in = vuzp_u8(vget_low_u8(v), vget_high_u8(v));
  uint16x8_t p0 = nc_clmul(in.val[0], c.lo);
  uint16x8_t p1 = veorq_u16(nc_clmul(in.val[0], c.hi),
                            nc_clmul(in.val[1], c.lo));
  uint16x8_t p2 = nc_clmul(in.val[1], c.hi);
  uint16x8_t low = veorq_u16(p0, vshlq_n_u16(p1, 8));
  uint16x8_t high = veorq_u16(p2, vshrq_n_u16(p1, 8));
  uint8x16_t hb = vreinterpretq_u8_u16(high);
  uint8x8x2_t hz = vuzp_u8(vget_low_u8(hb), vget_high_u8(hb));
  uint16x8_t f1 = nc_clmul(hz.val[1], k);
  uint16x8_t fold = veorq_u16(nc_clmul(hz.val[0], k),
                              vshlq_n_u16(f1, 8));
  uint8x8_t top = vmovn_u16(vshrq_n_u16(f1, 8));
  uint16x8_t out = veorq_u16(low, veorq_u16(fold, nc_clmul(top, k)));
  uint8x8_t ol = vmovn_u16(out);
  uint8x8_t oh = vmovn_u16(vshrq_n_u16(out, 8));
  uint8x8x2_t zip = vzip_u8(ol, oh);
  return vcombine_u8(zip.val[0], zip.val[1]);
}

static void nc_mac8(u8 * d, const u8 * s, sz n,
                    const xpar_gf8_coef * m) {
  const nc_c8 c = nc_load8(m);
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, veorq_u8(vld1q_u8(d + i),
                             nc_mul8(vld1q_u8(s + i), c)));
  xpar_gf8_mac_ref(d + i, s + i, n - i, m->c);
}

static void nc_mac8x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf8_coef m[2]) {
  const nc_c8 c0 = nc_load8(&m[0]), c1 = nc_load8(&m[1]);
  sz i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t v = vld1q_u8(s + i);
    vst1q_u8(d[0] + i, veorq_u8(vld1q_u8(d[0] + i), nc_mul8(v, c0)));
    vst1q_u8(d[1] + i, veorq_u8(vld1q_u8(d[1] + i), nc_mul8(v, c1)));
  }
  u32 j;
  Fj(2, xpar_gf8_mac_ref(d[j] + i, s + i, n - i, m[j].c));
}

static void nc_mul8_region(u8 * d, const u8 * s, sz n,
                           const xpar_gf8_coef * m) {
  const nc_c8 c = nc_load8(m);
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, nc_mul8(vld1q_u8(s + i), c));
  xpar_gf8_mul_ref(d + i, s + i, n - i, m->c);
}

static void nc_mac16(u8 * d, const u8 * s, sz n,
                     const xpar_gf16_coef * m) {
  const nc_c16 c = nc_load16(m);
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, veorq_u8(vld1q_u8(d + i),
                             nc_mul16(vld1q_u8(s + i), c)));
  xpar_gf16_mac_ref(d + i, s + i, n - i, m->c);
}

static void nc_mac16x2(u8 * const d[2], const u8 * s, sz n,
                       const xpar_gf16_coef m[2]) {
  nc_mac16(d[0], s, n, &m[0]);
  nc_mac16(d[1], s, n, &m[1]);
}

static void nc_mul16_region(u8 * d, const u8 * s, sz n,
                            const xpar_gf16_coef * m) {
  const nc_c16 c = nc_load16(m);
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, nc_mul16(vld1q_u8(s + i), c));
  xpar_gf16_mul_ref(d + i, s + i, n - i, m->c);
}

static void nc_xor2(u8 * d, const u8 * s, sz n) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, veorq_u8(vld1q_u8(d + i), vld1q_u8(s + i)));
  xpar_xor2_ref(d + i, s + i, n - i);
}

static void nc_xor3(u8 * d, const u8 * a, const u8 * b, sz n) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i)));
  xpar_xor3_ref(d + i, a + i, b + i, n - i);
}

#define NC_FFT(name, coef_t, ctype, load, mul, ref, inverse)                 \
static void name(u8 * x, u8 * y, sz n, const coef_t * m) {                  \
  const ctype gc = load(m);                                                 \
  sz i = 0;                                                                 \
  for (; i + 16 <= n; i += 16) {                                            \
    uint8x16_t vx = vld1q_u8(x + i), vy = vld1q_u8(y + i);                  \
    if (inverse) {                                                          \
      vy = veorq_u8(vy, vx);  vx = veorq_u8(vx, mul(vy, gc));               \
    } else {                                                                \
      vx = veorq_u8(vx, mul(vy, gc));  vy = veorq_u8(vy, vx);               \
    }                                                                       \
    vst1q_u8(x + i, vx);  vst1q_u8(y + i, vy);                              \
  }                                                                         \
  ref(x + i, y + i, n - i, m->c);                                          \
}

NC_FFT(nc_fft8,   xpar_gf8_coef,  nc_c8,  nc_load8,  nc_mul8,
       xpar_gf8_fft2_ref,   0)
NC_FFT(nc_ifft8,  xpar_gf8_coef,  nc_c8,  nc_load8,  nc_mul8,
       xpar_gf8_ifft2_ref,  1)
NC_FFT(nc_fft16,  xpar_gf16_coef, nc_c16, nc_load16, nc_mul16,
       xpar_gf16_fft2_ref,  0)
NC_FFT(nc_ifft16, xpar_gf16_coef, nc_c16, nc_load16, nc_mul16,
       xpar_gf16_ifft2_ref, 1)

const xpar_gf_kernels xpar_gf_kernels_neon_clmul = {
  "clmul-neon",
  nc_mac8, nc_mac8x2, nc_mac16, nc_mac16x2,
  nc_mul8_region, nc_mul16_region,
  nc_xor2, nc_xor3, nc_fft8, nc_fft16, nc_ifft8, nc_ifft16
};
