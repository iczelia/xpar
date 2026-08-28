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

/*  Inner-code region kernels, instantiated once per ISA.

    Encoder sweeps retain one source across 2t destinations; syndrome
    sweeps fuse multiply and XOR.  Each selects a loop order by register
    size.  Scalar reference functions handle vector tails.  */

#include "armour.h"

/*  Variant selection.  */

#if defined(XPAR_ARM_VARIANT_SCALAR)
  #define ARM_SUF   scalar
  #define ARM_NAME  "scalar"
#elif defined(XPAR_ARM_VARIANT_AVX2)
  #define ARM_SUF   avx2
  #define ARM_NAME  "avx2"
  #define ARM_ISA_AVX2
  #define ARM_SPLIT
#elif defined(XPAR_ARM_VARIANT_GFNI256)
  #define ARM_SUF   gfni256
  #define ARM_NAME  "gfni256"
  #define ARM_ISA_AVX2
  #define ARM_AFFINE
#elif defined(XPAR_ARM_VARIANT_NEON)
  #define ARM_SUF   neon
  #define ARM_NAME  "neon"
  #define ARM_ISA_NEON
  #define ARM_SPLIT
#else
  #error "armour-impl.h included without an XPAR_ARM_VARIANT_* selection"
#endif

#define ARM_CAT_(a, b) a##_##b
#define ARM_CAT(a, b)  ARM_CAT_(a, b)

#ifdef XPAR_ARM_HAVE_SIMD

/*  Vector primitives.
    ARM_TAB replicates a 16-byte table into every 128-bit lane, which is
    what the wide shuffles want: pshufb and vqtbl1q index inside their
    own lane at every width.  */

#if defined(ARM_ISA_AVX2)
  #include <immintrin.h>
  #define ARM_V           __m256i
  #define ARM_VB          32
  #define ARM_LD(p)       _mm256_loadu_si256((const __m256i *)               \
                                             (const void *) (p))
  #define ARM_ST(p, v)    _mm256_storeu_si256((__m256i *) (void *) (p), (v))
  #define ARM_XOR(a, b)   _mm256_xor_si256((a), (b))
  #define ARM_TAB(p)      _mm256_broadcastsi128_si256(                       \
                            _mm_loadu_si128((const __m128i *)                \
                                            (const void *) (p)))
  #define ARM_SHUF(t, i)  _mm256_shuffle_epi8((t), (i))
  #define ARM_SET64(x)    _mm256_set1_epi64x((long long) (x))
  #define ARM_AN4(v)      _mm256_and_si256((v), _mm256_set1_epi8(0x0F))
  #define ARM_SR4(v)      ARM_AN4(_mm256_srli_epi64((v), 4))
  #define ARM_AFF(v, mx)  _mm256_gf2p8affine_epi64_epi8((v), (mx), 0)
#elif defined(ARM_ISA_NEON)
  #include <arm_neon.h>
  #define ARM_V           uint8x16_t
  #define ARM_VB          16
  #define ARM_LD(p)       vld1q_u8((const u8 *) (const void *) (p))
  #define ARM_ST(p, v)    vst1q_u8((u8 *) (void *) (p), (v))
  #define ARM_XOR(a, b)   veorq_u8((a), (b))
  #define ARM_TAB(p)      ARM_LD(p)
  #define ARM_SHUF(t, i)  vqtbl1q_u8((t), (i))
  #define ARM_AN4(v)      vandq_u8((v), vdupq_n_u8(0x0F))
  #define ARM_SR4(v)      vshrq_n_u8((v), 4)
#endif

/*  GF(2^8): the multiply.
    ARM_PRE8 splits a source vector into the two nibble indices the
    shuffle method wants, once per vector rather than once per tap, and
    the affine method has nothing to split. The void cast in the affine
    form is there so that the unused half is still a use: the two macros
    have to present the same interface to the loops below.  */

#ifdef ARM_SPLIT
  #define ARM_PRE8(v, lo, hi)  do {                                          \
      (lo) = ARM_AN4(v);  (hi) = ARM_SR4(v);                                 \
    } while (0)
  #define ARM_MUL8P(lo, hi, m)                                               \
    ARM_XOR(ARM_SHUF(ARM_TAB((m)->tab), (lo)),                               \
            ARM_SHUF(ARM_TAB((m)->tab + 16), (hi)))
#else
  #define ARM_PRE8(v, lo, hi)  do { (lo) = (v);  (hi) = (v); } while (0)
  #define ARM_MUL8P(lo, hi, m)                                               \
    ARM_AFF(((void) (hi), (lo)), ARM_SET64((m)->affine))
#endif

/*  The same multiply where the source is not loop-invariant and there is
    nothing to hoist. The affine form takes the whole byte, not a nibble,
    so this cannot be written in terms of ARM_MUL8P.  */
#ifdef ARM_SPLIT
  #define ARM_MUL8V(v, m)                                                    \
    ARM_XOR(ARM_SHUF(ARM_TAB((m)->tab), ARM_AN4(v)),                         \
            ARM_SHUF(ARM_TAB((m)->tab + 16), ARM_SR4(v)))
#else
  #define ARM_MUL8V(v, m)  ARM_AFF((v), ARM_SET64((m)->affine))
#endif

/*  GF(2^16): planes.
    A region is interleaved little-endian u16 and the multiply wants the
    low and the high bytes apart, because a nibble table or a matrix
    applies to a byte and the two halves of a symbol need different ones.
    aarch64 separates them outright; x86 gathers within each lane with a
    byte shuffle and recombines the halves of two vectors with a 64-bit
    unpack, which permutes the symbols inside each plane. That does not
    matter, because the multiply is elementwise and ARM_INT undoes
    exactly the permutation ARM_DEINT introduced.  */

#ifdef ARM_ISA_NEON
  #define ARM_PLANE_VARS
  #define ARM_DEINT(v0, v1, olo, ohi) do {                                   \
      (olo) = vuzp1q_u8((v0), (v1));  (ohi) = vuzp2q_u8((v0), (v1));         \
    } while (0)
  #define ARM_INT(ilo, ihi, o0, o1) do {                                     \
      (o0) = vzip1q_u8((ilo), (ihi));  (o1) = vzip2q_u8((ilo), (ihi));       \
    } while (0)
#else
static const u8 arm_deint_idx[16] = { 0, 2, 4, 6, 8, 10, 12, 14,
                                      1, 3, 5, 7, 9, 11, 13, 15 };
static const u8 arm_reint_idx[16] = { 0, 8, 1, 9, 2, 10, 3, 11,
                                      4, 12, 5, 13, 6, 14, 7, 15 };
  #define ARM_PLANE_VARS                                                     \
    const ARM_V amdi = ARM_TAB(arm_deint_idx), amii = ARM_TAB(arm_reint_idx);
  #define ARM_DEINT(v0, v1, olo, ohi) do {                                   \
      ARM_V amu = ARM_SHUF((v0), amdi), amv = ARM_SHUF((v1), amdi);          \
      (olo) = _mm256_unpacklo_epi64(amu, amv);                               \
      (ohi) = _mm256_unpackhi_epi64(amu, amv);                               \
    } while (0)
  #define ARM_INT(ilo, ihi, o0, o1) do {                                     \
      ARM_V amu = _mm256_unpacklo_epi64((ilo), (ihi));                       \
      ARM_V amv = _mm256_unpackhi_epi64((ilo), (ihi));                       \
      (o0) = ARM_SHUF(amu, amii);  (o1) = ARM_SHUF(amv, amii);               \
    } while (0)
#endif

#ifdef ARM_SPLIT
  /*  Eight nibble tables against four affine matrices, which is the
      other half of why the affine tier leads the ladder.  */
  #define ARM_MUL16P(lo, hi, olo, ohi, m) do {                               \
      ARM_V an0 = ARM_AN4(lo), an1 = ARM_SR4(lo),                            \
            an2 = ARM_AN4(hi), an3 = ARM_SR4(hi);                            \
      (olo) = ARM_XOR(ARM_XOR(ARM_SHUF(ARM_TAB((m)->tab[0]), an0),           \
                              ARM_SHUF(ARM_TAB((m)->tab[2]), an1)),          \
                      ARM_XOR(ARM_SHUF(ARM_TAB((m)->tab[4]), an2),           \
                              ARM_SHUF(ARM_TAB((m)->tab[6]), an3)));         \
      (ohi) = ARM_XOR(ARM_XOR(ARM_SHUF(ARM_TAB((m)->tab[1]), an0),           \
                              ARM_SHUF(ARM_TAB((m)->tab[3]), an1)),          \
                      ARM_XOR(ARM_SHUF(ARM_TAB((m)->tab[5]), an2),           \
                              ARM_SHUF(ARM_TAB((m)->tab[7]), an3)));         \
    } while (0)
#else
  #define ARM_MUL16P(lo, hi, olo, ohi, m) do {                               \
      (olo) = ARM_XOR(ARM_AFF((lo), ARM_SET64((m)->affine[0])),              \
                      ARM_AFF((hi), ARM_SET64((m)->affine[1])));             \
      (ohi) = ARM_XOR(ARM_AFF((lo), ARM_SET64((m)->affine[2])),              \
                      ARM_AFF((hi), ARM_SET64((m)->affine[3])));             \
    } while (0)
#endif

/*  Whole-vector steps only; the remainder uses the reference kernel.
    SIMD starts when depth reaches ARM_VB.  */
#define ARM_BODY8(n)   ((n) & ~(sz) (ARM_VB - 1))
#define ARM_BODY16(n)  ((n) & ~(sz) (2 * ARM_VB - 1))

/*  Hoisted coefficients.
    The tap-major sweeps below hold one coefficient for a whole pass over
    the lane, which is where the shuffle tiers get their eight tables and
    the affine tiers their matrix into registers instead of reloading
    them per vector. The chunk-major sweeps cannot: their inner loop is
    the tap.  */

#ifdef ARM_SPLIT
  #define ARM_H8_VARS(m)                                                     \
    const ARM_V ah0 = ARM_TAB((m)->tab), ah1 = ARM_TAB((m)->tab + 16);
  #define ARM_H8P(lo, hi)                                                    \
    ARM_XOR(ARM_SHUF(ah0, (lo)), ARM_SHUF(ah1, (hi)))
  #define ARM_H16_VARS(m)                                                    \
    const ARM_V ah0 = ARM_TAB((m)->tab[0]), ah1 = ARM_TAB((m)->tab[1]),      \
                ah2 = ARM_TAB((m)->tab[2]), ah3 = ARM_TAB((m)->tab[3]),      \
                ah4 = ARM_TAB((m)->tab[4]), ah5 = ARM_TAB((m)->tab[5]),      \
                ah6 = ARM_TAB((m)->tab[6]), ah7 = ARM_TAB((m)->tab[7]);
  #define ARM_H16P(lo, hi, olo, ohi) do {                                    \
      ARM_V an0 = ARM_AN4(lo), an1 = ARM_SR4(lo),                            \
            an2 = ARM_AN4(hi), an3 = ARM_SR4(hi);                            \
      (olo) = ARM_XOR(ARM_XOR(ARM_SHUF(ah0, an0), ARM_SHUF(ah2, an1)),       \
                      ARM_XOR(ARM_SHUF(ah4, an2), ARM_SHUF(ah6, an3)));      \
      (ohi) = ARM_XOR(ARM_XOR(ARM_SHUF(ah1, an0), ARM_SHUF(ah3, an1)),       \
                      ARM_XOR(ARM_SHUF(ah5, an2), ARM_SHUF(ah7, an3)));      \
    } while (0)
#else
  #define ARM_H8_VARS(m)   const ARM_V ah0 = ARM_SET64((m)->affine);
  #define ARM_H8P(lo, hi)  ARM_AFF(((void) (hi), (lo)), ah0)
  #define ARM_H16_VARS(m)                                                    \
    const ARM_V ah0 = ARM_SET64((m)->affine[0]),                             \
                ah1 = ARM_SET64((m)->affine[1]),                             \
                ah2 = ARM_SET64((m)->affine[2]),                             \
                ah3 = ARM_SET64((m)->affine[3]);
  #define ARM_H16P(lo, hi, olo, ohi) do {                                    \
      (olo) = ARM_XOR(ARM_AFF((lo), ah0), ARM_AFF((hi), ah1));               \
      (ohi) = ARM_XOR(ARM_AFF((lo), ah2), ARM_AFF((hi), ah3));               \
    } while (0)
#endif

/*  The two whole-byte forms, for the sweeps whose multiplicand is the
    accumulator and therefore has nothing pre-split.  */
#ifdef ARM_SPLIT
  #define ARM_H8V(v)  ARM_H8P(ARM_AN4(v), ARM_SR4(v))
#else
  #define ARM_H8V(v)  ARM_AFF((v), ah0)
#endif

/*  Chunk-major keeps the source live while the parity register fits in L1.
    Tap-major streams each slot once and avoids cache-set collisions for
    larger interleave depths.  */

#define ARM_TILE  4096

/*  par[(head + u) mod t2] ^= gen[u] * fb.  Split the rotation into two
    contiguous runs to avoid modulo address arithmetic.  */

#define ARM_RUN8(p0, g0, cnt) do {                                           \
    u8 * ap = (p0);  const xpar_gf8_coef * ag = (g0);  u32 ac = (cnt);       \
    while (ac--) {                                                           \
      ARM_ST(ap, ARM_XOR(ARM_LD(ap), ARM_MUL8P(aflo, afhi, ag)));            \
      ap += stride;  ag++;                                                   \
    }                                                                        \
  } while (0)

#define ARM_SWEEP8(p0, g0, cnt) do {                                         \
    u8 * ap = (p0);  const xpar_gf8_coef * ag = (g0);  u32 ac = (cnt);       \
    while (ac--) {                                                           \
      ARM_H8_VARS(ag)                                                        \
      sz aq;                                                                 \
      for (aq = 0; aq < body; aq += ARM_VB) {                                \
        ARM_V afv = ARM_LD(fb + aq), aflo, afhi;                             \
        ARM_PRE8(afv, aflo, afhi);                                           \
        ARM_ST(ap + aq, ARM_XOR(ARM_LD(ap + aq), ARM_H8P(aflo, afhi)));      \
      }                                                                      \
      ap += stride;  ag++;                                                   \
    }                                                                        \
  } while (0)

static void k_taps8(u8 * restrict par, sz stride, u32 t2, u32 head,
                    const xpar_gf8_coef * gen,
                    const u8 * restrict fb, sz n) {
  sz ai = 0, body = ARM_BODY8(n);
  u32 first = t2 - head;
  if ((u64) t2 * (u64) n <= ARM_TILE) {
    for (; ai < body; ai += ARM_VB) {
      ARM_V afv = ARM_LD(fb + ai), aflo, afhi;
      ARM_PRE8(afv, aflo, afhi);
      ARM_RUN8(par + (sz) head * stride + ai, gen, first);
      ARM_RUN8(par + ai, gen + first, head);
    }
  } else {
    ARM_SWEEP8(par + (sz) head * stride, gen, first);
    ARM_SWEEP8(par, gen + first, head);
    ai = body;
  }
  if (ai < n)
    xpar_armour_taps8_ref(par + ai, stride, t2, head, gen, fb + ai, n - ai);
}

#define ARM_RUN16(p0, g0, cnt) do {                                          \
    u8 * ap = (p0);  const xpar_gf16_coef * ag = (g0);  u32 ac = (cnt);      \
    while (ac--) {                                                           \
      ARM_V aol, aoh, ar0, ar1;                                              \
      ARM_MUL16P(aflo, afhi, aol, aoh, ag);                                  \
      ARM_INT(aol, aoh, ar0, ar1);                                           \
      ARM_ST(ap, ARM_XOR(ARM_LD(ap), ar0));                                  \
      ARM_ST(ap + ARM_VB, ARM_XOR(ARM_LD(ap + ARM_VB), ar1));                \
      ap += stride;  ag++;                                                   \
    }                                                                        \
  } while (0)

#define ARM_SWEEP16(p0, g0, cnt) do {                                        \
    u8 * ap = (p0);  const xpar_gf16_coef * ag = (g0);  u32 ac = (cnt);      \
    while (ac--) {                                                           \
      ARM_H16_VARS(ag)                                                       \
      sz aq;                                                                 \
      for (aq = 0; aq < body; aq += 2 * ARM_VB) {                            \
        ARM_V aflo, afhi, aol, aoh, ar0, ar1;                                \
        ARM_DEINT(ARM_LD(fb + aq), ARM_LD(fb + aq + ARM_VB), aflo, afhi);    \
        ARM_H16P(aflo, afhi, aol, aoh);                                      \
        ARM_INT(aol, aoh, ar0, ar1);                                         \
        ARM_ST(ap + aq, ARM_XOR(ARM_LD(ap + aq), ar0));                      \
        ARM_ST(ap + aq + ARM_VB,                                             \
               ARM_XOR(ARM_LD(ap + aq + ARM_VB), ar1));                      \
      }                                                                      \
      ap += stride;  ag++;                                                   \
    }                                                                        \
  } while (0)

static void k_taps16(u8 * restrict par, sz stride, u32 t2, u32 head,
                     const xpar_gf16_coef * gen,
                     const u8 * restrict fb, sz n) {
  sz ai = 0, body = ARM_BODY16(n);
  u32 first = t2 - head;
  ARM_PLANE_VARS
  if ((u64) t2 * (u64) n <= ARM_TILE) {
    for (; ai < body; ai += 2 * ARM_VB) {
      ARM_V aflo, afhi;
      ARM_DEINT(ARM_LD(fb + ai), ARM_LD(fb + ai + ARM_VB), aflo, afhi);
      ARM_RUN16(par + (sz) head * stride + ai, gen, first);
      ARM_RUN16(par + ai, gen + first, head);
    }
  } else {
    ARM_SWEEP16(par + (sz) head * stride, gen, first);
    ARM_SWEEP16(par, gen + first, head);
    ai = body;
  }
  if (ai < n)
    xpar_armour_taps16_ref(par + ai, stride, t2, head, gen, fb + ai, n - ai);
}

/*  The Horner sweep.
    syn[j] = syn[j] * rt[j] ^ sym, fused so that syn is read once and
    written once where a region multiply followed by a region XOR would
    read and write it twice. The accumulator is the multiply's source, so
    nothing about it can be hoisted; what is hoisted is the codeword
    symbol, which all 2t recurrences add. Slots are in order here, so
    neither order has a wrap to split.  */

static void k_horner8(u8 * restrict syn, sz stride, u32 t2,
                      const xpar_gf8_coef * rt,
                      const u8 * restrict sym, sz n) {
  sz ai = 0, body = ARM_BODY8(n);
  u32 aj;
  if ((u64) t2 * (u64) n <= ARM_TILE) {
    for (; ai < body; ai += ARM_VB) {
      ARM_V asv = ARM_LD(sym + ai);
      u8 * ap = syn + ai;
      for (aj = 0; aj < t2; aj++) {
        ARM_ST(ap, ARM_XOR(ARM_MUL8V(ARM_LD(ap), rt + aj), asv));
        ap += stride;
      }
    }
  } else {
    u8 * ap = syn;
    for (aj = 0; aj < t2; aj++) {
      ARM_H8_VARS(rt + aj)
      sz aq;
      for (aq = 0; aq < body; aq += ARM_VB)
        ARM_ST(ap + aq, ARM_XOR(ARM_H8V(ARM_LD(ap + aq)), ARM_LD(sym + aq)));
      ap += stride;
    }
    ai = body;
  }
  if (ai < n)
    xpar_armour_horner8_ref(syn + ai, stride, t2, rt, sym + ai, n - ai);
}

static void k_horner16(u8 * restrict syn, sz stride, u32 t2,
                       const xpar_gf16_coef * rt,
                       const u8 * restrict sym, sz n) {
  sz ai = 0, body = ARM_BODY16(n);
  u32 aj;
  ARM_PLANE_VARS
  if ((u64) t2 * (u64) n <= ARM_TILE) {
    for (; ai < body; ai += 2 * ARM_VB) {
      ARM_V av0 = ARM_LD(sym + ai), av1 = ARM_LD(sym + ai + ARM_VB);
      u8 * ap = syn + ai;
      for (aj = 0; aj < t2; aj++) {
        ARM_V alo, ahi, aol, aoh, ar0, ar1;
        ARM_DEINT(ARM_LD(ap), ARM_LD(ap + ARM_VB), alo, ahi);
        ARM_MUL16P(alo, ahi, aol, aoh, rt + aj);
        ARM_INT(aol, aoh, ar0, ar1);
        ARM_ST(ap, ARM_XOR(ar0, av0));
        ARM_ST(ap + ARM_VB, ARM_XOR(ar1, av1));
        ap += stride;
      }
    }
  } else {
    u8 * ap = syn;
    for (aj = 0; aj < t2; aj++) {
      ARM_H16_VARS(rt + aj)
      sz aq;
      for (aq = 0; aq < body; aq += 2 * ARM_VB) {
        ARM_V alo, ahi, aol, aoh, ar0, ar1;
        ARM_DEINT(ARM_LD(ap + aq), ARM_LD(ap + aq + ARM_VB), alo, ahi);
        ARM_H16P(alo, ahi, aol, aoh);
        ARM_INT(aol, aoh, ar0, ar1);
        ARM_ST(ap + aq, ARM_XOR(ar0, ARM_LD(sym + aq)));
        ARM_ST(ap + aq + ARM_VB,
               ARM_XOR(ar1, ARM_LD(sym + aq + ARM_VB)));
      }
      ap += stride;
    }
    ai = body;
  }
  if (ai < n)
    xpar_armour_horner16_ref(syn + ai, stride, t2, rt, sym + ai, n - ai);
}

#else

/*  Scalar tier.  */

#define ARM_VB 0   /*  Scalar tier.  */

static void k_taps8(u8 * restrict par, sz stride, u32 t2, u32 head,
                    const xpar_gf8_coef * gen,
                    const u8 * restrict fb, sz n) {
  xpar_armour_taps8_ref(par, stride, t2, head, gen, fb, n);
}
static void k_taps16(u8 * restrict par, sz stride, u32 t2, u32 head,
                     const xpar_gf16_coef * gen,
                     const u8 * restrict fb, sz n) {
  xpar_armour_taps16_ref(par, stride, t2, head, gen, fb, n);
}
static void k_horner8(u8 * restrict syn, sz stride, u32 t2,
                      const xpar_gf8_coef * rt,
                      const u8 * restrict sym, sz n) {
  xpar_armour_horner8_ref(syn, stride, t2, rt, sym, n);
}
static void k_horner16(u8 * restrict syn, sz stride, u32 t2,
                       const xpar_gf16_coef * rt,
                       const u8 * restrict sym, sz n) {
  xpar_armour_horner16_ref(syn, stride, t2, rt, sym, n);
}

#endif

const xpar_armour_kernels ARM_CAT(xpar_armour_kernels, ARM_SUF) = {
  ARM_NAME, ARM_VB, k_taps8, k_taps16, k_horner8, k_horner16
};
