/*  xpar: PowerPC VSX Galois-field region kernels.

    Copyright (C) 2022-2026 Kamila Szewczyk.  GPLv3-only (see COPYING).  */

#include "gf.h"

#include <altivec.h>

typedef __vector unsigned char vx_u8;

static vx_u8 vx_splat(u8 x) { return vec_splats((unsigned char) x); }

static vx_u8 vx_mul8(vx_u8 v, u8 c) {
  vx_u8 a = v, out = vx_splat(0), one = vx_splat(1);
  vx_u8 sh1 = vx_splat(1), sh7 = vx_splat(7), poly = vx_splat(0x1D);
  for (u32 bit = 0; bit < 8; bit++) {
    vx_u8 carry;
    if ((c >> bit) & 1) out ^= a;
    carry = vec_sr(a, sh7) & one;
    a = vec_sl(a, sh1) ^ (vec_sub(vx_splat(0), carry) & poly);
  }
  return out;
}

static void vx_mul16(vx_u8 v0, vx_u8 v1, u16 c,
                     vx_u8 * o0, vx_u8 * o1) {
  static const u8 even_bytes[16] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
  };
  static const u8 odd_bytes[16] = {
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
  };
  vx_u8 pe = vec_vsx_ld(0, even_bytes), po = vec_vsx_ld(0, odd_bytes);
  vx_u8 lo = vec_perm(v0, v1, pe), hi = vec_perm(v0, v1, po);
  vx_u8 al = lo, ah = hi, rl = vx_splat(0), rh = vx_splat(0);
  vx_u8 one = vx_splat(1), sh1 = vx_splat(1), sh7 = vx_splat(7);
  vx_u8 poly = vx_splat(0x2D);
  for (u32 bit = 0; bit < 16; bit++) {
    vx_u8 carry, bridge;
    if ((c >> bit) & 1) { rl ^= al;  rh ^= ah; }
    carry = vec_sr(ah, sh7) & one;
    bridge = vec_sr(al, sh7) & one;
    ah = vec_sl(ah, sh1) | bridge;
    al = vec_sl(al, sh1) ^ (vec_sub(vx_splat(0), carry) & poly);
  }
  *o0 = vec_mergeh(rl, rh);
  *o1 = vec_mergel(rl, rh);
}

static void vx_mac8(u8 * d, const u8 * s, sz n,
                    const xpar_gf8_coef * m) {
  sz i = 0;
  for (; i + 16 <= n; i += 16) {
    vx_u8 dv = vec_vsx_ld(0, d + i), sv = vec_vsx_ld(0, s + i);
    vec_vsx_st(dv ^ vx_mul8(sv, m->c), 0, d + i);
  }
  xpar_gf8_mac_ref(d + i, s + i, n - i, m->c);
}

static void vx_mac8x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf8_coef m[2]) {
  sz i = 0;
  for (; i + 16 <= n; i += 16) {
    vx_u8 v = vec_vsx_ld(0, s + i);
    for (u32 j = 0; j < 2; j++) {
      vx_u8 out = vec_vsx_ld(0, d[j] + i) ^ vx_mul8(v, m[j].c);
      vec_vsx_st(out, 0, d[j] + i);
    }
  }
  for (u32 j = 0; j < 2; j++)
    xpar_gf8_mac_ref(d[j] + i, s + i, n - i, m[j].c);
}

static void vx_mul8_region(u8 * d, const u8 * s, sz n,
                           const xpar_gf8_coef * m) {
  sz i = 0;
  for (; i + 16 <= n; i += 16)
    vec_vsx_st(vx_mul8(vec_vsx_ld(0, s + i), m->c), 0, d + i);
  xpar_gf8_mul_ref(d + i, s + i, n - i, m->c);
}

static void vx_mac16(u8 * d, const u8 * s, sz n,
                     const xpar_gf16_coef * m) {
  sz i = 0;
  for (; i + 32 <= n; i += 32) {
    vx_u8 a, b;
    vx_mul16(vec_vsx_ld(0, s + i), vec_vsx_ld(0, s + i + 16),
             m->c, &a, &b);
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
  sz i = 0;
  for (; i + 32 <= n; i += 32) {
    vx_u8 a, b;
    vx_mul16(vec_vsx_ld(0, s + i), vec_vsx_ld(0, s + i + 16),
             m->c, &a, &b);
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
  sz i = 0;                                                                 \
  for (; i + 16 <= n; i += 16) {                                            \
    vx_u8 a = vec_vsx_ld(0, x + i), b = vec_vsx_ld(0, y + i);               \
    if (inverse) { b ^= a;  a ^= vx_mul8(b, m->c); }                         \
    else { a ^= vx_mul8(b, m->c);  b ^= a; }                                \
    vec_vsx_st(a, 0, x + i);  vec_vsx_st(b, 0, y + i);                      \
  }                                                                         \
  ref(x + i, y + i, n - i, m->c);                                          \
}

#define VX_FFT16(name, ref, inverse)                                         \
static void name(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m) {          \
  sz i = 0;                                                                 \
  for (; i + 32 <= n; i += 32) {                                            \
    vx_u8 x0 = vec_vsx_ld(0, x + i), x1 = vec_vsx_ld(0, x + i + 16);        \
    vx_u8 y0 = vec_vsx_ld(0, y + i), y1 = vec_vsx_ld(0, y + i + 16);        \
    vx_u8 a, b;                                                             \
    if (inverse) {                                                          \
      y0 ^= x0;  y1 ^= x1;  vx_mul16(y0, y1, m->c, &a, &b);                 \
      x0 ^= a;  x1 ^= b;                                                    \
    } else {                                                                \
      vx_mul16(y0, y1, m->c, &a, &b);  x0 ^= a;  x1 ^= b;                   \
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
