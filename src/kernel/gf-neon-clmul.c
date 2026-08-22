/*  xpar: NEON PMULL kernels, including GF(2^16) carry-less multiply.

    Copyright (C) 2022-2026 Kamila Szewczyk.  GPLv3-only (see COPYING).  */

#include "gf.h"

#include <arm_neon.h>

static uint8x16_t nc_mul8(uint8x16_t v, const xpar_gf8_coef * m) {
  uint8x16_t mask = vdupq_n_u8(15);
  uint8x16_t lo = vld1q_u8(m->tab);
  uint8x16_t hi = vld1q_u8(m->tab + 16);
  return veorq_u8(vqtbl1q_u8(lo, vandq_u8(v, mask)),
                  vqtbl1q_u8(hi, vshrq_n_u8(v, 4)));
}

static poly8x8_t nc_poly8(uint8x8_t v) {
  return vreinterpret_p8_u8(v);
}

static uint16x8_t nc_clmul_byte(uint8x8_t a, u8 b) {
  return vreinterpretq_u16_p16(vmull_p8(nc_poly8(a),
                                         vdup_n_p8((poly8_t) b)));
}

/*  A 16x16 carry-less product is assembled from four 8x8 PMULLs. Reduction
    uses x^16 = x^5+x^3+x^2+1 twice; after the first fold only four high
    bits remain, so the second byte PMULL completes it.  */
static uint8x16_t nc_mul16(uint8x16_t v, const xpar_gf16_coef * m) {
  uint8x8x2_t in = vuzp_u8(vget_low_u8(v), vget_high_u8(v));
  uint16x8_t p0 = nc_clmul_byte(in.val[0], (u8) m->c);
  uint16x8_t p1 = veorq_u16(
    nc_clmul_byte(in.val[0], (u8) (m->c >> 8)),
    nc_clmul_byte(in.val[1], (u8) m->c));
  uint16x8_t p2 = nc_clmul_byte(in.val[1], (u8) (m->c >> 8));
  uint16x8_t low = veorq_u16(p0, vshlq_n_u16(p1, 8));
  uint16x8_t high = veorq_u16(p2, vshrq_n_u16(p1, 8));
  uint8x16_t hb = vreinterpretq_u8_u16(high);
  uint8x8x2_t hz = vuzp_u8(vget_low_u8(hb), vget_high_u8(hb));
  uint16x8_t f1 = nc_clmul_byte(hz.val[1], 0x2D);
  uint16x8_t fold = veorq_u16(nc_clmul_byte(hz.val[0], 0x2D),
                              vshlq_n_u16(f1, 8));
  uint8x8_t top = vmovn_u16(vshrq_n_u16(f1, 8));
  uint16x8_t out = veorq_u16(low,
    veorq_u16(fold, nc_clmul_byte(top, 0x2D)));
  uint8x8_t ol = vmovn_u16(out);
  uint8x8_t oh = vmovn_u16(vshrq_n_u16(out, 8));
  uint8x8x2_t zip = vzip_u8(ol, oh);
  return vcombine_u8(zip.val[0], zip.val[1]);
}

static void nc_mac8(u8 * d, const u8 * s, sz n,
                    const xpar_gf8_coef * m) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, veorq_u8(vld1q_u8(d + i),
                             nc_mul8(vld1q_u8(s + i), m)));
  xpar_gf8_mac_ref(d + i, s + i, n - i, m->c);
}

static void nc_mac8x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf8_coef m[2]) {
  sz i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t v = vld1q_u8(s + i);
    for (u32 j = 0; j < 2; j++)
      vst1q_u8(d[j] + i, veorq_u8(vld1q_u8(d[j] + i),
                                   nc_mul8(v, &m[j])));
  }
  for (u32 j = 0; j < 2; j++)
    xpar_gf8_mac_ref(d[j] + i, s + i, n - i, m[j].c);
}

static void nc_mul8_region(u8 * d, const u8 * s, sz n,
                           const xpar_gf8_coef * m) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, nc_mul8(vld1q_u8(s + i), m));
  xpar_gf8_mul_ref(d + i, s + i, n - i, m->c);
}

static void nc_mac16(u8 * d, const u8 * s, sz n,
                     const xpar_gf16_coef * m) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, veorq_u8(vld1q_u8(d + i),
                             nc_mul16(vld1q_u8(s + i), m)));
  xpar_gf16_mac_ref(d + i, s + i, n - i, m->c);
}

static void nc_mac16x2(u8 * const d[2], const u8 * s, sz n,
                       const xpar_gf16_coef m[2]) {
  nc_mac16(d[0], s, n, &m[0]);
  nc_mac16(d[1], s, n, &m[1]);
}

static void nc_mul16_region(u8 * d, const u8 * s, sz n,
                            const xpar_gf16_coef * m) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_u8(d + i, nc_mul16(vld1q_u8(s + i), m));
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

#define NC_FFT(name, coef_t, mul, ref, inverse)                              \
static void name(u8 * x, u8 * y, sz n, const coef_t * m) {                  \
  sz i = 0;                                                                 \
  for (; i + 16 <= n; i += 16) {                                            \
    uint8x16_t vx = vld1q_u8(x + i), vy = vld1q_u8(y + i);                  \
    if (inverse) {                                                          \
      vy = veorq_u8(vy, vx);  vx = veorq_u8(vx, mul(vy, m));                \
    } else {                                                                \
      vx = veorq_u8(vx, mul(vy, m));  vy = veorq_u8(vy, vx);                \
    }                                                                       \
    vst1q_u8(x + i, vx);  vst1q_u8(y + i, vy);                              \
  }                                                                         \
  ref(x + i, y + i, n - i, m->c);                                          \
}

NC_FFT(nc_fft8,  xpar_gf8_coef,  nc_mul8,  xpar_gf8_fft2_ref,  0)
NC_FFT(nc_ifft8, xpar_gf8_coef,  nc_mul8,  xpar_gf8_ifft2_ref, 1)
NC_FFT(nc_fft16, xpar_gf16_coef, nc_mul16, xpar_gf16_fft2_ref, 0)
NC_FFT(nc_ifft16, xpar_gf16_coef, nc_mul16, xpar_gf16_ifft2_ref, 1)

const xpar_gf_kernels xpar_gf_kernels_neon_clmul = {
  "clmul-neon",
  nc_mac8, nc_mac8x2, nc_mac16, nc_mac16x2,
  nc_mul8_region, nc_mul16_region,
  nc_xor2, nc_xor3, nc_fft8, nc_fft16, nc_ifft8, nc_ifft16
};
