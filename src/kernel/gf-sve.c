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

/*  xpar: base-SVE Galois-field region kernels.  */

#include "gf.h"

#include <arm_sve.h>

/*  SVE halfword loads are host-endian; GF16 stream symbols are
    little-endian.  */
static svuint16_t sv_load16(svbool_t pg, const u16 * p) {
  svuint16_t v = svld1_u16(pg, p);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&               \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = svrevb_u16_x(pg, v);
#endif
  return v;
}

static void sv_store16(svbool_t pg, u16 * p, svuint16_t v) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&               \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = svrevb_u16_x(pg, v);
#endif
  svst1_u16(pg, p, v);
}

static svuint8_t sv_mul8(svbool_t pg, svuint8_t v, u8 c) {
  svuint8_t a = v, out = svdup_n_u8(0);
  for (u32 bit = 0; bit < 8; bit++) {
    svuint8_t carry;
    if ((c >> bit) & 1) out = sveor_u8_x(pg, out, a);
    carry = svlsr_n_u8_x(pg, a, 7);
    a = svlsl_n_u8_x(pg, a, 1);
    a = sveor_u8_x(pg, a, svmul_n_u8_x(pg, carry, 0x1D));
  }
  return out;
}

static svuint16_t sv_mul16(svbool_t pg, svuint16_t v, u16 c) {
  svuint16_t a = v, out = svdup_n_u16(0);
  for (u32 bit = 0; bit < 16; bit++) {
    svuint16_t carry;
    if ((c >> bit) & 1) out = sveor_u16_x(pg, out, a);
    carry = svlsr_n_u16_x(pg, a, 15);
    a = svlsl_n_u16_x(pg, a, 1);
    a = sveor_u16_x(pg, a, svmul_n_u16_x(pg, carry, 0x2D));
  }
  return out;
}

/*  Cache the coefficient because destination writes may alias it.  */
static void sv_mac8(u8 * d, const u8 * s, sz n,
                    const xpar_gf8_coef * m) {
  const u8 c = m->c;
  sz i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b8((u64) i, (u64) n);
    svuint8_t v = sv_mul8(pg, svld1_u8(pg, s + i), c);
    svst1_u8(pg, d + i, sveor_u8_x(pg, svld1_u8(pg, d + i), v));
    i += svcntb();
  }
}

static void sv_mac8x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf8_coef m[2]) {
  const u8 c0 = m[0].c, c1 = m[1].c;
  sz i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b8((u64) i, (u64) n);
    svuint8_t v = svld1_u8(pg, s + i);
    svst1_u8(pg, d[0] + i,
             sveor_u8_x(pg, svld1_u8(pg, d[0] + i), sv_mul8(pg, v, c0)));
    svst1_u8(pg, d[1] + i,
             sveor_u8_x(pg, svld1_u8(pg, d[1] + i), sv_mul8(pg, v, c1)));
    i += svcntb();
  }
}

static void sv_mul8_region(u8 * d, const u8 * s, sz n,
                           const xpar_gf8_coef * m) {
  const u8 c = m->c;
  sz i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b8((u64) i, (u64) n);
    svst1_u8(pg, d + i, sv_mul8(pg, svld1_u8(pg, s + i), c));
    i += svcntb();
  }
}

static void sv_mac16(u8 * d, const u8 * s, sz n,
                     const xpar_gf16_coef * m) {
  const u16 c = m->c;
  sz i = 0, symbols = n / 2;
  for (; i < symbols; i += svcnth()) {
    svbool_t pg = svwhilelt_b16((u64) i, (u64) symbols);
    const u16 * sp = (const u16 *) (const void *) s;
    u16 * dp = (u16 *) (void *) d;
    svuint16_t v = sv_mul16(pg, sv_load16(pg, sp + i), c);
    sv_store16(pg, dp + i, sveor_u16_x(pg, sv_load16(pg, dp + i), v));
  }
}

static void sv_mac16x2(u8 * const d[2], const u8 * s, sz n,
                       const xpar_gf16_coef m[2]) {
  sv_mac16(d[0], s, n, &m[0]);
  sv_mac16(d[1], s, n, &m[1]);
}

static void sv_mul16_region(u8 * d, const u8 * s, sz n,
                            const xpar_gf16_coef * m) {
  const u16 c = m->c;
  sz i = 0, symbols = n / 2;
  for (; i < symbols; i += svcnth()) {
    svbool_t pg = svwhilelt_b16((u64) i, (u64) symbols);
    const u16 * sp = (const u16 *) (const void *) s;
    u16 * dp = (u16 *) (void *) d;
    sv_store16(pg, dp + i, sv_mul16(pg, sv_load16(pg, sp + i), c));
  }
}

static void sv_xor2(u8 * d, const u8 * s, sz n) {
  sz i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b8((u64) i, (u64) n);
    svst1_u8(pg, d + i, sveor_u8_x(pg, svld1_u8(pg, d + i),
                                   svld1_u8(pg, s + i)));
    i += svcntb();
  }
}

static void sv_xor3(u8 * d, const u8 * a, const u8 * b, sz n) {
  sz i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b8((u64) i, (u64) n);
    svst1_u8(pg, d + i, sveor_u8_x(pg, svld1_u8(pg, a + i),
                                   svld1_u8(pg, b + i)));
    i += svcntb();
  }
}

#define S2_FFT8(name, inverse)                                               \
static void name(u8 * x, u8 * y, sz n, const xpar_gf8_coef * m) {           \
  const u8 c = m->c;                                                        \
  sz i = 0;                                                                 \
  while (i < n) {                                                           \
    svbool_t pg = svwhilelt_b8((u64) i, (u64) n);                            \
    svuint8_t vx = svld1_u8(pg, x + i), vy = svld1_u8(pg, y + i);           \
    if (inverse) {                                                          \
      vy = sveor_u8_x(pg, vy, vx);                                          \
      vx = sveor_u8_x(pg, vx, sv_mul8(pg, vy, c));                         \
    } else {                                                                \
      vx = sveor_u8_x(pg, vx, sv_mul8(pg, vy, c));                          \
      vy = sveor_u8_x(pg, vy, vx);                                          \
    }                                                                       \
    svst1_u8(pg, x + i, vx);  svst1_u8(pg, y + i, vy);                      \
    i += svcntb();                                                          \
  }                                                                         \
}

#define S2_FFT16(name, inverse)                                              \
static void name(u8 * xb, u8 * yb, sz n, const xpar_gf16_coef * m) {        \
  u16 * x = (u16 *) (void *) xb, * y = (u16 *) (void *) yb;                 \
  const u16 c = m->c;                                                       \
  sz i = 0, symbols = n / 2;                                                \
  for (; i < symbols; i += svcnth()) {                                      \
    svbool_t pg = svwhilelt_b16((u64) i, (u64) symbols);                     \
    svuint16_t vx = sv_load16(pg, x + i), vy = sv_load16(pg, y + i);        \
    if (inverse) {                                                          \
      vy = sveor_u16_x(pg, vy, vx);                                         \
      vx = sveor_u16_x(pg, vx, sv_mul16(pg, vy, c));                        \
    } else {                                                                \
      vx = sveor_u16_x(pg, vx, sv_mul16(pg, vy, c));                         \
      vy = sveor_u16_x(pg, vy, vx);                                         \
    }                                                                       \
    sv_store16(pg, x + i, vx);  sv_store16(pg, y + i, vy);                  \
  }                                                                         \
}

S2_FFT8(sv_fft8, 0)
S2_FFT8(sv_ifft8, 1)
S2_FFT16(sv_fft16, 0)
S2_FFT16(sv_ifft16, 1)

const xpar_gf_kernels xpar_gf_kernels_sve = {
  "sve", sv_mac8, sv_mac8x2, sv_mac16, sv_mac16x2,
  sv_mul8_region, sv_mul16_region,
  sv_xor2, sv_xor3, sv_fft8, sv_fft16, sv_ifft8, sv_ifft16
};
