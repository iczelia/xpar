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

/*  Reed-Solomon inner-code interface.

    Parameters describe shortened RS(n, k = n - 2t) codes over GF(2^8)
    or GF(2^16).  Frames interleave D codewords so frame symbol s belongs
    to codeword s mod D at position floor(s / D).  Plaintext therefore
    occupies the frame prefix verbatim.  Callers check the frame tag
    before decoding and must validate any correction with the tag.  */

#ifndef XPAR_ARMOUR_H
#define XPAR_ARMOUR_H

#include "common.h"
#include "gf.h"

/*  On-disk parameters in host form.  `poly` excludes the x^w term.  */

typedef struct {
  u32 symbol_bits;   /*  8 or 16.  */
  u32 poly;          /*  Low bits of the field modulus.  */
  u32 n;             /*  Codeword length in symbols, 3 <= n <= 2^w - 1.  */
  u32 k;             /*  Data symbols; n - k = 2t and must be even.  */
  u32 fcr;           /*  Generator roots are alpha^(fcr + i*prim).  */
  u32 prim;          /*  Must be a unit modulo 2^w - 1.  */
  u64 depth;         /*  D, 1 <= D <= 2^24.  */
} xpar_armour_params;

/*  Normative field and code defaults.  */
void xpar_armour_defaults(xpar_armour_params * p, u32 symbol_bits);

/*  Return NULL when the parameters are supported, else an error.  */
const char * xpar_armour_check(const xpar_armour_params * p);

/*  A codec owns scratch and is not thread-safe.  Build one per worker.
    Construction binds the current kernel tier and requires valid
    parameters and an initialised field.  */

typedef struct xpar_armour xpar_armour;

xpar_armour * xpar_armour_new (const xpar_armour_params * p);
void          xpar_armour_free(xpar_armour * a);

const xpar_armour_params * xpar_armour_params_of(const xpar_armour * a);

/*  Plaintext occupies the frame prefix; the final frame is zero-padded.  */

u64 xpar_armour_frame_plain(const xpar_armour * a);   /*  D*k*W  */
u64 xpar_armour_frame_disk (const xpar_armour * a);   /*  D*n*W  */
u64 xpar_armour_size(const xpar_armour * a, u64 plain_length);

/*  Alignment-independent correctable burst: (t*D - 1)*W bytes.  */
u64 xpar_armour_burst(const xpar_armour * a);

/*  Return g[i], the coefficient of x^i for 0 <= i <= 2t.  */
void xpar_armour_generator(const xpar_armour * a, u32 * g);

/*  Encode one prepared frame, or copy and encode a whole region.  */

void xpar_armour_encode_frame(const xpar_armour * a, u8 * frame);
void xpar_armour_encode(const xpar_armour * a, u8 * out,
                        const u8 * plain, u64 plain_length);

/*  Extract plaintext from an undamaged region without field arithmetic.  */
void xpar_armour_extract(const xpar_armour * a, u8 * plain, u64 plain_length,
                         const u8 * region);

typedef enum {
  XPAR_ARMOUR_CLEAN     = 0,   /*  Every syndrome zero; nothing touched.  */
  XPAR_ARMOUR_CORRECTED = 1,   /*  Errors found and, provisionally, fixed.  */
  XPAR_ARMOUR_FAILED    = 2    /*  At least one codeword past capacity.  */
} xpar_armour_status;

/*  Per-codeword statistics.  `hist`, when non-NULL, has t+1 entries.  */

typedef struct {
  u64   frames;      /*  Frames examined.  */
  u64   codewords;   /*  Codewords examined, = frames * D.  */
  u64   clean;       /*  Codewords whose syndromes were all zero.  */
  u64   corrected;   /*  Codewords decoded with 1..t errors.  */
  u64   failed;      /*  Codewords the decoder would not accept.  */
  u64   symbols;     /*  Symbols corrected, summed over codewords.  */
  u32   worst;       /*  Largest single-codeword count corrected.  */
  u64 * hist;        /*  hist[e] counts codewords that took e errors,
                         e = 0 included; t+1 entries covers every case.  */
  u32   hist_len;    /*  Entries in hist; counts past it are dropped.  */
} xpar_armour_stat;

xpar_armour_status xpar_armour_decode_frame(const xpar_armour * a, u8 * frame,
                                            xpar_armour_stat * st);

/*  Decode with n*depth erasure flags in disk order.  Up to 2t known
    erasures per codeword are recoverable.  The result is provisional.  */
xpar_armour_status xpar_armour_decode_frame_erasures(
                                      const xpar_armour * a, u8 * frame,
                                      const u8 * erased,
                                      xpar_armour_stat * st);

/*  Return true when recovered plaintext passes its integrity check.  */
typedef bool (* xpar_armour_check_fn)(void * ctx, const u8 * plain, u64 len);

xpar_armour_status xpar_armour_decode(const xpar_armour * a,
                                      u8 * region, u64 region_length,
                                      u8 * plain, u64 plain_length,
                                      xpar_armour_check_fn check, void * ctx,
                                      xpar_armour_stat * st);

/*  Checked region decoding with frames*n*depth erasure flags.  */
xpar_armour_status xpar_armour_decode_erasures(
                                      const xpar_armour * a,
                                      u8 * region, u64 region_length,
                                      u8 * plain, u64 plain_length,
                                      const u8 * erased,
                                      xpar_armour_check_fn check, void * ctx,
                                      xpar_armour_stat * st);

/*  Region kernels over D*W bytes.  The encoder uses a rotating parity
    register; the decoder fuses Horner multiply and XOR.  Source buffers
    must not overlap the register.  */

typedef struct {
  const char * name;
  void (* taps8   )(u8 * restrict par, sz stride, u32 t2, u32 head,
                    const xpar_gf8_coef  * gen, const u8 * restrict fb, sz n);
  void (* taps16  )(u8 * restrict par, sz stride, u32 t2, u32 head,
                    const xpar_gf16_coef * gen, const u8 * restrict fb, sz n);
  void (* horner8 )(u8 * restrict syn, sz stride, u32 t2,
                    const xpar_gf8_coef  * rt, const u8 * restrict sym, sz n);
  void (* horner16)(u8 * restrict syn, sz stride, u32 t2,
                    const xpar_gf16_coef * rt, const u8 * restrict sym, sz n);
} xpar_armour_kernels;

/*  The scalar tier, and the tail of every vector kernel, so a tail that
    disagrees with a body is not expressible. T-TIERS compares against
    these.  */

void xpar_armour_taps8_ref   (u8 * restrict par, sz stride, u32 t2, u32 head,
                              const xpar_gf8_coef  * gen,
                              const u8 * restrict fb, sz n);
void xpar_armour_taps16_ref  (u8 * restrict par, sz stride, u32 t2, u32 head,
                              const xpar_gf16_coef * gen,
                              const u8 * restrict fb, sz n);
void xpar_armour_horner8_ref (u8 * restrict syn, sz stride, u32 t2,
                              const xpar_gf8_coef  * rt,
                              const u8 * restrict sym, sz n);
void xpar_armour_horner16_ref(u8 * restrict syn, sz stride, u32 t2,
                              const xpar_gf16_coef * rt,
                              const u8 * restrict sym, sz n);

/*  Tiers.  */

int          xpar_armour_tier_count(void);
const char * xpar_armour_tier_name (int tier);
bool         xpar_armour_tier_usable(int tier);
int          xpar_armour_tier(void);

bool xpar_armour_use_tier(int tier);
bool xpar_armour_use_tier_name(const char * name);
void xpar_armour_use_default_tier(void);

/*  Per-ISA tables from separately flagged translation units.  */

extern const xpar_armour_kernels xpar_armour_kernels_scalar;
#ifdef HAVE_AVX2
extern const xpar_armour_kernels xpar_armour_kernels_avx2;
#endif
#ifdef HAVE_GFNI
extern const xpar_armour_kernels xpar_armour_kernels_gfni256;
#endif
#ifdef HAVE_NEON
extern const xpar_armour_kernels xpar_armour_kernels_neon;
#endif

#endif
