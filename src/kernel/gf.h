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

/*  GF(2^8) and GF(2^16) arithmetic interface.

    Scalar operations use log tables.  Region operations dispatch to a
    supported vector tier and operate on interleaved symbols without a
    dependency between lanes.  */

#ifndef XPAR_GF_H
#define XPAR_GF_H

#include "common.h"

/*  The two fields.
    alpha = 2, the class of x, generates the multiplicative group of both
    fields. xpar_gf_init asserts that alpha returns to 1 only after the
    full group order: a modulus that is irreducible but not primitive
    repeats early and leaves most of the field without a logarithm.  */

#define XPAR_GF8_POLY   0x11DU     /*  x^8+x^4+x^3+x^2+1  */
#define XPAR_GF16_POLY  0x1002DU   /*  x^16+x^5+x^3+x^2+1  */

/*  Tables.  */

extern u8 xpar_gf8_exp[512];      /*  exp[i] = alpha^(i mod 255); i <= 508  */
extern u8 xpar_gf8_log[256];
extern u8 xpar_gf8_inv_tab[256];  /*  inv_tab[0] = 0, by convention.  */

/*  131,070 + 65,536 entries of u16 = 384 KiB, allocated by xpar_gf_init.  */
extern const u16 * xpar_gf16_exp;
extern const u16 * xpar_gf16_log;

/*  Lifecycle.  */
void xpar_gf_init(void);

/*  Scalars.  */

static inline u8 xpar_gf8_mul(u8 a, u8 b) {
  return (a && b) ? xpar_gf8_exp[xpar_gf8_log[a] + xpar_gf8_log[b]] : 0;
}
static inline u8 xpar_gf8_inv(u8 a) { return xpar_gf8_inv_tab[a]; }
static inline u8 xpar_gf8_div(u8 a, u8 b) {
  return xpar_gf8_mul(a, xpar_gf8_inv_tab[b]);
}
static inline u8 xpar_gf8_alpha_pow(u32 e) { return xpar_gf8_exp[e % 255u]; }

static inline u16 xpar_gf16_mul(u16 a, u16 b) {
  return (a && b) ? xpar_gf16_exp[(u32) xpar_gf16_log[a] +
                                  (u32) xpar_gf16_log[b]] : 0;
}
static inline u16 xpar_gf16_inv(u16 a) {
  return a ? xpar_gf16_exp[65535u - xpar_gf16_log[a]] : 0;
}
static inline u16 xpar_gf16_div(u16 a, u16 b) {
  return xpar_gf16_mul(a, xpar_gf16_inv(b));
}
static inline u16 xpar_gf16_alpha_pow(u32 e) {
  return xpar_gf16_exp[e % 65535u];
}

/*  Cantor basis.  */
extern const u8  xpar_gf8_cantor[8];
extern const u16 xpar_gf16_cantor[16];

/*  Prepared coefficients contain both shuffle tables and affine matrices.
    Callers prepare once and reuse across regions.  Every tier's fields are
    filled, so a coefficient remains valid after dispatch changes.  */

typedef struct {
  u64 affine;    /*  The 8x8 GF(2) matrix of `x -> x * c`, GFNI order.  */
  u8  tab[32];   /*  [0..15] indexed by the low nibble, [16..31] high.  */
  u8  c;
} xpar_gf8_coef;

typedef struct {
  /*  The 16x16 matrix of `x -> x * c` as four 8x8 blocks: [0] low byte
      of the product from the low byte of the input, [1] low from high,
      [2] high from low, [3] high from high.  */
  u64 affine[4];
  /*  tab[2k] and tab[2k+1] are the low and high bytes of
      `(i << 4k) * c`, i in [0,16), so a product is four table lookups
      per output byte.  */
  u8  tab[8][16];
  /*  AVX-512 VBMI indexes a whole 64-byte register. Three input groups
      of 6, 6 and 4 bits therefore replace the four nibble groups; each
      pair is the low and high output byte of the GF(2^16) product.  */
  u8  tab6[6][64];
  u16 c;
} xpar_gf16_coef;

void xpar_gf8_prepare (xpar_gf8_coef  * m, u8  c);
void xpar_gf16_prepare(xpar_gf16_coef * m, u16 c);

/*  Region kernels.  */

typedef struct {
  const char * name;
  /*  dst ^= src * c  */
  void (* mac8  )(u8 * d, const u8 * s, sz n, const xpar_gf8_coef  * m);
  void (* mac8x2)(u8 * const d[2], const u8 * s, sz n,
                  const xpar_gf8_coef m[2]);
  void (* mac16 )(u8 * d, const u8 * s, sz n, const xpar_gf16_coef * m);
  void (* mac16x2)(u8 * const d[2], const u8 * s, sz n,
                   const xpar_gf16_coef m[2]);
  /*  dst = src * c  */
  void (* mul8  )(u8 * d, const u8 * s, sz n, const xpar_gf8_coef  * m);
  void (* mul16 )(u8 * d, const u8 * s, sz n, const xpar_gf16_coef * m);
  /*  dst ^= src, and dst = a ^ b  */
  void (* xor2  )(u8 * d, const u8 * s, sz n);
  void (* xor3  )(u8 * d, const u8 * a, const u8 * b, sz n);
  /*  x ^= y * c; y ^= x   (the decimation-in-time butterfly)  */
  void (* fft8  )(u8 * x, u8 * y, sz n, const xpar_gf8_coef  * m);
  void (* fft16 )(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m);
  /*  y ^= x; x ^= y * c   (its inverse)  */
  void (* ifft8 )(u8 * x, u8 * y, sz n, const xpar_gf8_coef  * m);
  void (* ifft16)(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m);
} xpar_gf_kernels;

/*  The resolved tier. Hot loops load it once outside every loop they
    have; the convenience wrappers below go through it per call.  */
const xpar_gf_kernels * xpar_gf_active(void);

/*  Convenience wrappers.  */

void xpar_gf8_mac        (u8 * dst, const u8 * src, sz n, u8  c);
void xpar_gf16_mac       (u8 * dst, const u8 * src, sz n, u16 c);
void xpar_gf8_mul_region (u8 * dst, const u8 * src, sz n, u8  c);
void xpar_gf16_mul_region(u8 * dst, const u8 * src, sz n, u16 c);
void xpar_xor_region     (u8 * dst, const u8 * src, sz n);
void xpar_xor_region3    (u8 * dst, const u8 * a, const u8 * b, sz n);

/*  Reference kernels.  */

void xpar_gf8_mac_ref  (u8 * d, const u8 * s, sz n, u8  c);
void xpar_gf16_mac_ref (u8 * d, const u8 * s, sz n, u16 c);
void xpar_gf8_mul_ref  (u8 * d, const u8 * s, sz n, u8  c);
void xpar_gf16_mul_ref (u8 * d, const u8 * s, sz n, u16 c);
void xpar_xor2_ref     (u8 * d, const u8 * s, sz n);
void xpar_xor3_ref     (u8 * d, const u8 * a, const u8 * b, sz n);
void xpar_gf8_fft2_ref (u8 * x, u8 * y, sz n, u8  c);
void xpar_gf16_fft2_ref(u8 * x, u8 * y, sz n, u16 c);
void xpar_gf8_ifft2_ref(u8 * x, u8 * y, sz n, u8  c);
void xpar_gf16_ifft2_ref(u8 * x, u8 * y, sz n, u16 c);

/*  Tiers.  */

int          xpar_gf_tier_count(void);
const char * xpar_gf_tier_name(int tier);
bool         xpar_gf_tier_usable(int tier);  /*  Compiled and supported.  */
int          xpar_gf_tier(void);             /*  The active index.  */

/*  Force a tier. False, and no change, when the index is out of range or
    the host lacks the instructions; running a kernel the host cannot
    execute is a fault, not a slow path.  */
bool xpar_gf_use_tier(int tier);
bool xpar_gf_use_tier_name(const char * name);  /*  The --simd override.  */

/*  Re-resolve the preference list against the current feature mask.
    xpar_gf_init calls it; call it again after xpar_cpu_force, or the
    selection made at startup outlives the mask it was made under.  */
void xpar_gf_use_default_tier(void);

/*  Per-ISA tables are built in separately flagged translation units.  */

extern const xpar_gf_kernels xpar_gf_kernels_scalar;
#ifdef HAVE_SSSE3
extern const xpar_gf_kernels xpar_gf_kernels_ssse3;
#endif
#ifdef HAVE_AVX2
extern const xpar_gf_kernels xpar_gf_kernels_avx2;
#endif
#ifdef HAVE_GFNI
extern const xpar_gf_kernels xpar_gf_kernels_gfni256;
#endif
#ifdef HAVE_GFNI512
extern const xpar_gf_kernels xpar_gf_kernels_gfni512;
#endif
#ifdef HAVE_VBMI
extern const xpar_gf_kernels xpar_gf_kernels_vbmi512;
#endif
#ifdef HAVE_NEON
extern const xpar_gf_kernels xpar_gf_kernels_neon;
#endif
#ifdef HAVE_PMULL
extern const xpar_gf_kernels xpar_gf_kernels_neon_clmul;
#endif
#ifdef HAVE_SVE2
extern const xpar_gf_kernels xpar_gf_kernels_sve2;
#endif
#ifdef HAVE_VSX
extern const xpar_gf_kernels xpar_gf_kernels_vsx;
#endif
#ifdef HAVE_RVV
extern const xpar_gf_kernels xpar_gf_kernels_rvv_shuffle;
#endif
#ifdef HAVE_RVV_CLMUL
extern const xpar_gf_kernels xpar_gf_kernels_rvv_clmul;
#endif

#endif
