/*  Galois-field region kernels, instantiated once per ISA.

    Split-nibble tiers use two byte shuffles; GFNI tiers apply the prepared
    GF(2) matrix.  GF(2^16) deinterleaves low and high byte planes except
    on the masked AVX-512 path.  Scalar reference functions handle tails.

    Copyright (C) 2022-2026 Kamila Szewczyk.  GPLv3-only (see COPYING).  */

#include "gf.h"

/*  Variant selection.  */

#if defined(XPAR_GF_VARIANT_SCALAR)
  #define GF_SUF    scalar
  #define GF_NAME   "scalar"
#elif defined(XPAR_GF_VARIANT_SSSE3)
  #define GF_SUF    ssse3
  #define GF_NAME   "ssse3"
  #define GF_ISA_SSE
  #define GF_SPLIT
#elif defined(XPAR_GF_VARIANT_AVX2)
  #define GF_SUF    avx2
  #define GF_NAME   "avx2"
  #define GF_ISA_AVX2
  #define GF_SPLIT
#elif defined(XPAR_GF_VARIANT_GFNI256)
  #define GF_SUF    gfni256
  #define GF_NAME   "gfni256"
  #define GF_ISA_AVX2
  #define GF_AFFINE
#elif defined(XPAR_GF_VARIANT_GFNI512)
  #define GF_SUF    gfni512
  #define GF_NAME   "gfni512"
  #define GF_ISA_AVX512
  #define GF_AFFINE
#elif defined(XPAR_GF_VARIANT_NEON)
  #define GF_SUF    neon
  #define GF_NAME   "neon"
  #define GF_ISA_NEON
  #define GF_SPLIT
#elif defined(XPAR_GF_VARIANT_VBMI512)
  #define GF_SUF    vbmi512
  #define GF_NAME   "vbmi512"
  #define GF_ISA_AVX512
  #define GF_SPLIT
  #define GF_VBMI
#else
  #error "gf-impl.h included without an XPAR_GF_VARIANT_* selection"
#endif

#define GF_CAT_(a, b) a##_##b
#define GF_CAT(a, b)  GF_CAT_(a, b)

#ifdef XPAR_GF_HAVE_SIMD

/*  Vector primitives.
    GF_TAB loads a 16-byte table and replicates it into every 128-bit
    lane, which is what the wide shuffles want: pshufb and vqtbl1q index
    inside their own lane on every width, so one pattern serves 128, 256
    and 512 bits. GF_UNPKL and GF_UNPKH are likewise per-lane, which is
    why GF_INT below is an exact inverse of GF_DEINT even though the
    symbol order inside the two planes is permuted.  */

#if defined(GF_ISA_SSE)
  #include <immintrin.h>
  #define GF_V           __m128i
  #define GF_VB          16
  #define GF_LD(p)       _mm_loadu_si128((const __m128i *) (const void *) (p))
  #define GF_ST(p, v)    _mm_storeu_si128((__m128i *) (void *) (p), (v))
  #define GF_XOR(a, b)   _mm_xor_si128((a), (b))
  #define GF_TAB(p)      GF_LD(p)
  #define GF_SHUF(t, i)  _mm_shuffle_epi8((t), (i))
  #define GF_SET64(x)    _mm_set1_epi64x((long long) (x))
  #define GF_UNPKL(a, b) _mm_unpacklo_epi64((a), (b))
  #define GF_UNPKH(a, b) _mm_unpackhi_epi64((a), (b))
  #define GF_AN4(v)      _mm_and_si128((v), _mm_set1_epi8(0x0F))
  #define GF_SR4(v)      GF_AN4(_mm_srli_epi64((v), 4))
#elif defined(GF_ISA_AVX2)
  #include <immintrin.h>
  #define GF_V           __m256i
  #define GF_VB          32
  #define GF_LD(p)       _mm256_loadu_si256((const __m256i *) (const void *) \
                                            (p))
  #define GF_ST(p, v)    _mm256_storeu_si256((__m256i *) (void *) (p), (v))
  #define GF_XOR(a, b)   _mm256_xor_si256((a), (b))
  #define GF_TAB(p)      _mm256_broadcastsi128_si256(                        \
                           _mm_loadu_si128((const __m128i *) (const void *)  \
                                           (p)))
  #define GF_SHUF(t, i)  _mm256_shuffle_epi8((t), (i))
  #define GF_SET64(x)    _mm256_set1_epi64x((long long) (x))
  #define GF_UNPKL(a, b) _mm256_unpacklo_epi64((a), (b))
  #define GF_UNPKH(a, b) _mm256_unpackhi_epi64((a), (b))
  #define GF_AN4(v)      _mm256_and_si256((v), _mm256_set1_epi8(0x0F))
  #define GF_SR4(v)      GF_AN4(_mm256_srli_epi64((v), 4))
  #define GF_AFF(v, mx)  _mm256_gf2p8affine_epi64_epi8((v), (mx), 0)
#elif defined(GF_ISA_AVX512)
  #include <immintrin.h>
  #define GF_V           __m512i
  #define GF_VB          64
  #define GF_LD(p)       _mm512_loadu_si512((const void *) (p))
  #define GF_ST(p, v)    _mm512_storeu_si512((void *) (p), (v))
  #define GF_XOR(a, b)   _mm512_xor_si512((a), (b))
  #define GF_TAB(p)      _mm512_broadcast_i32x4(                             \
                           _mm_loadu_si128((const __m128i *) (const void *)  \
                                           (p)))
  #define GF_SHUF(t, i)  _mm512_shuffle_epi8((t), (i))
  #define GF_SET64(x)    _mm512_set1_epi64((long long) (x))
  #define GF_UNPKL(a, b) _mm512_unpacklo_epi64((a), (b))
  #define GF_UNPKH(a, b) _mm512_unpackhi_epi64((a), (b))
  #define GF_AN4(v)      _mm512_and_si512((v), _mm512_set1_epi8(0x0F))
  #define GF_SR4(v)      GF_AN4(_mm512_srli_epi64((v), 4))
  #define GF_AFF(v, mx)  _mm512_gf2p8affine_epi64_epi8((v), (mx), 0)
#elif defined(GF_ISA_NEON)
  #include <arm_neon.h>
  #define GF_V           uint8x16_t
  #define GF_VB          16
  #define GF_LD(p)       vld1q_u8((const u8 *) (const void *) (p))
  #define GF_ST(p, v)    vst1q_u8((u8 *) (void *) (p), (v))
  #define GF_XOR(a, b)   veorq_u8((a), (b))
  #define GF_TAB(p)      GF_LD(p)
  #define GF_SHUF(t, i)  vqtbl1q_u8((t), (i))
  #define GF_AN4(v)      vandq_u8((v), vdupq_n_u8(0x0F))
  #define GF_SR4(v)      vshrq_n_u8((v), 4)
#endif

/*  GF(2^8): the multiply.
    GF_MUL8_VARS hoists the coefficient into registers once per call, so
    the loops below carry no per-iteration setup. Each of these macros
    carries its own semicolon, because on some variants one of them
    expands to nothing and a stray semicolon between declarations would
    be a statement.  */

#ifdef GF_SPLIT
  #define GF_MUL8_VARS                                                       \
    const GF_V gt0 = GF_TAB(m->tab), gt1 = GF_TAB(m->tab + 16);
  #define GF_MUL8(v)                                                         \
    GF_XOR(GF_SHUF(gt0, GF_AN4(v)), GF_SHUF(gt1, GF_SR4(v)))
#else
  #define GF_MUL8_VARS                                                       \
    const GF_V gma = GF_SET64(m->affine);
  #define GF_MUL8(v)  GF_AFF((v), gma)
#endif

/*  GF(2^16): the multiply.
    GF_MUL16_PAIR turns two source vectors into two product vectors and
    is the only GF(2^16) primitive the kernels below use, so a tier is
    free to reach the result any way it likes. Two ways are used.

    Planes. A GF(2^16) region is interleaved little-endian u16 and both
    the nibble table and the block matrix apply to a byte, so the low
    bytes and the high bytes have to be separated first. aarch64 has the
    separation outright; x86 gathers within each lane with a byte shuffle
    and recombines the halves of two vectors with a 64-bit unpack. The
    unpack leaves the symbols permuted inside each plane, which does not
    matter: the multiply is elementwise and GF_INT undoes exactly the
    permutation GF_DEINT introduced.

    A mask register instead. A masked affine writes each product directly
    into its destination byte lane, avoiding plane construction.  Only the
    AVX-512 tier has an efficient mask form; other tiers build planes.  */

#if defined(GF_VBMI)

static const u8 gf_dup_even_idx[16] = { 0, 0, 2, 2, 4, 4, 6, 6,
                                        8, 8, 10, 10, 12, 12, 14, 14 };
  #define GF_EVEN ((__mmask64) 0x5555555555555555ULL)
  #define GF_MUL16_VARS                                                      \
    const GF_V gt0 = GF_LD(m->tab6[0]), gt1 = GF_LD(m->tab6[1]),             \
               gt2 = GF_LD(m->tab6[2]), gt3 = GF_LD(m->tab6[3]),             \
               gt4 = GF_LD(m->tab6[4]), gt5 = GF_LD(m->tab6[5]),             \
               gdup = GF_TAB(gf_dup_even_idx);
  #define GF_MUL16_ONE(s, o) do {                                            \
      GF_V gv = (s), gi0, gi1, gi2, gl, gh;                                  \
      gi0 = GF_SHUF(gv, gdup);                                               \
      gi0 = _mm512_and_si512(gi0, _mm512_set1_epi8(0x3F));                   \
      gi1 = _mm512_srli_epi16(gv, 6);                                        \
      gi1 = GF_SHUF(gi1, gdup);                                              \
      gi1 = _mm512_and_si512(gi1, _mm512_set1_epi8(0x3F));                   \
      gi2 = _mm512_srli_epi16(gv, 12);                                       \
      gi2 = GF_SHUF(gi2, gdup);                                              \
      gl = GF_XOR(_mm512_permutexvar_epi8(gi0, gt0),                          \
                  _mm512_permutexvar_epi8(gi1, gt2));                         \
      gh = GF_XOR(_mm512_permutexvar_epi8(gi0, gt1),                          \
                  _mm512_permutexvar_epi8(gi1, gt3));                         \
      gl = GF_XOR(gl, _mm512_permutexvar_epi8(gi2, gt4));                     \
      gh = GF_XOR(gh, _mm512_permutexvar_epi8(gi2, gt5));                     \
      (o) = _mm512_mask_blend_epi8(GF_EVEN, gh, gl);                          \
    } while (0)
  #define GF_MUL16_PAIR(s0, s1, o0, o1) do {                                 \
      GF_MUL16_ONE((s0), (o0));  GF_MUL16_ONE((s1), (o1));                   \
    } while (0)

#elif defined(GF_ISA_AVX512) && defined(GF_AFFINE)

static const u8 gf_swap_idx[16] = { 1, 0, 3, 2, 5, 4, 7, 6,
                                    9, 8, 11, 10, 13, 12, 15, 14 };
  #define GF_EVEN ((__mmask64) 0x5555555555555555ULL)
  #define GF_MUL16_VARS                                                      \
    const GF_V gma = GF_SET64(m->affine[0]), gmb = GF_SET64(m->affine[1]),   \
               gmc = GF_SET64(m->affine[2]), gmd = GF_SET64(m->affine[3]),   \
               gsw = GF_TAB(gf_swap_idx);
  /*  Even lanes hold a symbol's low byte and odd lanes its high byte, so
      the even half of the result is A*low ^ B*high and the odd half is
      C*low ^ D*high. gv carries low in the even lanes; gw, the
      byte-swapped copy, carries it in the odd ones.  */
  #define GF_MUL16_ONE(s, o) do {                                            \
      GF_V gv = (s), gw = GF_SHUF(gv, gsw);                                  \
      GF_V gx = GF_AFF(gv, gmd), gy = GF_AFF(gw, gmc);                       \
      gx = _mm512_mask_gf2p8affine_epi64_epi8(gx, GF_EVEN, gv, gma, 0);      \
      gy = _mm512_mask_gf2p8affine_epi64_epi8(gy, GF_EVEN, gw, gmb, 0);      \
      (o) = GF_XOR(gx, gy);                                                  \
    } while (0)
  #define GF_MUL16_PAIR(s0, s1, o0, o1) do {                                 \
      GF_MUL16_ONE((s0), (o0));  GF_MUL16_ONE((s1), (o1));                   \
    } while (0)

#else

#ifdef GF_ISA_NEON
  #define GF_PLANE_VARS
  #define GF_DEINT(v0, v1, olo, ohi) do {                                    \
      (olo) = vuzp1q_u8((v0), (v1));  (ohi) = vuzp2q_u8((v0), (v1));         \
    } while (0)
  #define GF_INT(ilo, ihi, o0, o1) do {                                      \
      (o0) = vzip1q_u8((ilo), (ihi));  (o1) = vzip2q_u8((ilo), (ihi));       \
    } while (0)
#else
static const u8 gf_deint_idx[16] = { 0, 2, 4, 6, 8, 10, 12, 14,
                                     1, 3, 5, 7, 9, 11, 13, 15 };
static const u8 gf_reint_idx[16] = { 0, 8, 1, 9, 2, 10, 3, 11,
                                     4, 12, 5, 13, 6, 14, 7, 15 };
  #define GF_PLANE_VARS                                                      \
    const GF_V gfdi = GF_TAB(gf_deint_idx), gfii = GF_TAB(gf_reint_idx);
  #define GF_DEINT(v0, v1, olo, ohi) do {                                    \
      GF_V gfu = GF_SHUF((v0), gfdi), gfv = GF_SHUF((v1), gfdi);             \
      (olo) = GF_UNPKL(gfu, gfv);  (ohi) = GF_UNPKH(gfu, gfv);               \
    } while (0)
  #define GF_INT(ilo, ihi, o0, o1) do {                                      \
      GF_V gfu = GF_UNPKL((ilo), (ihi)), gfv = GF_UNPKH((ilo), (ihi));       \
      (o0) = GF_SHUF(gfu, gfii);  (o1) = GF_SHUF(gfv, gfii);                 \
    } while (0)
#endif

#ifdef GF_SPLIT
  /*  Nine vectors live at once, which is most of the register file on
      SSSE3 and AVX2. That is inherent to eight nibble tables and is the
      other half of why the affine tiers are preferred.  */
  #define GF_MUL16_VARS                                                      \
    const GF_V gt0 = GF_TAB(m->tab[0]), gt1 = GF_TAB(m->tab[1]),             \
               gt2 = GF_TAB(m->tab[2]), gt3 = GF_TAB(m->tab[3]),             \
               gt4 = GF_TAB(m->tab[4]), gt5 = GF_TAB(m->tab[5]),             \
               gt6 = GF_TAB(m->tab[6]), gt7 = GF_TAB(m->tab[7]);             \
    GF_PLANE_VARS
  #define GF_MUL16(lo, hi, olo, ohi) do {                                    \
      GF_V gn0 = GF_AN4(lo), gn1 = GF_SR4(lo),                               \
           gn2 = GF_AN4(hi), gn3 = GF_SR4(hi);                               \
      (olo) = GF_XOR(GF_XOR(GF_SHUF(gt0, gn0), GF_SHUF(gt2, gn1)),           \
                     GF_XOR(GF_SHUF(gt4, gn2), GF_SHUF(gt6, gn3)));          \
      (ohi) = GF_XOR(GF_XOR(GF_SHUF(gt1, gn0), GF_SHUF(gt3, gn1)),           \
                     GF_XOR(GF_SHUF(gt5, gn2), GF_SHUF(gt7, gn3)));          \
    } while (0)
#else
  #define GF_MUL16_VARS                                                      \
    const GF_V gma = GF_SET64(m->affine[0]), gmb = GF_SET64(m->affine[1]),   \
               gmc = GF_SET64(m->affine[2]), gmd = GF_SET64(m->affine[3]);   \
    GF_PLANE_VARS
  #define GF_MUL16(lo, hi, olo, ohi) do {                                    \
      (olo) = GF_XOR(GF_AFF((lo), gma), GF_AFF((hi), gmb));                  \
      (ohi) = GF_XOR(GF_AFF((lo), gmc), GF_AFF((hi), gmd));                  \
    } while (0)
#endif

  #define GF_MUL16_PAIR(s0, s1, o0, o1) do {                                 \
      GF_V glo, ghi, gol, goh;                                               \
      GF_DEINT((s0), (s1), glo, ghi);                                        \
      GF_MUL16(glo, ghi, gol, goh);                                          \
      GF_INT(gol, goh, (o0), (o1));                                          \
    } while (0)

#endif

/*  Whole vector steps only; the remainder goes to the reference kernel.  */
#define GF_BODY8(n)   ((n) & ~(sz) (GF_VB - 1))
#define GF_BODY16(n)  ((n) & ~(sz) (2 * GF_VB - 1))

/*  GF(2^8).  */

static void k_mac8(u8 * d, const u8 * s, sz n, const xpar_gf8_coef * m) {
  sz i = 0, body = GF_BODY8(n);
  GF_MUL8_VARS
  for (; i < body; i += GF_VB)
    GF_ST(d + i, GF_XOR(GF_LD(d + i), GF_MUL8(GF_LD(s + i))));
  if (i < n) xpar_gf8_mac_ref(d + i, s + i, n - i, m->c);
}

static void k_mac8x2(u8 * const d[2], const u8 * s, sz n,
                     const xpar_gf8_coef m[2]) {
  sz i = 0, body = GF_BODY8(n);
#ifdef GF_SPLIT
  const GF_V a0 = GF_TAB(m[0].tab), a1 = GF_TAB(m[0].tab + 16);
  const GF_V b0 = GF_TAB(m[1].tab), b1 = GF_TAB(m[1].tab + 16);
#else
  const GF_V a0 = GF_SET64(m[0].affine), b0 = GF_SET64(m[1].affine);
#endif
  for (; i < body; i += GF_VB) {
    GF_V v = GF_LD(s + i), p;
#ifdef GF_SPLIT
    GF_V lo = GF_AN4(v), hi = GF_SR4(v);
    p = GF_XOR(GF_SHUF(a0, lo), GF_SHUF(a1, hi));
    GF_ST(d[0] + i, GF_XOR(GF_LD(d[0] + i), p));
    p = GF_XOR(GF_SHUF(b0, lo), GF_SHUF(b1, hi));
#else
    p = GF_AFF(v, a0);
    GF_ST(d[0] + i, GF_XOR(GF_LD(d[0] + i), p));
    p = GF_AFF(v, b0);
#endif
    GF_ST(d[1] + i, GF_XOR(GF_LD(d[1] + i), p));
  }
  if (i < n)
    for (u32 j = 0; j < 2; j++)
      xpar_gf8_mac_ref(d[j] + i, s + i, n - i, m[j].c);
}

static void k_mul8(u8 * d, const u8 * s, sz n, const xpar_gf8_coef * m) {
  sz i = 0, body = GF_BODY8(n);
  GF_MUL8_VARS
  for (; i < body; i += GF_VB) GF_ST(d + i, GF_MUL8(GF_LD(s + i)));
  if (i < n) xpar_gf8_mul_ref(d + i, s + i, n - i, m->c);
}

static void k_fft8(u8 * x, u8 * y, sz n, const xpar_gf8_coef * m) {
  sz i = 0, body = GF_BODY8(n);
  GF_MUL8_VARS
  for (; i < body; i += GF_VB) {
    GF_V vx = GF_LD(x + i), vy = GF_LD(y + i);
    vx = GF_XOR(vx, GF_MUL8(vy));  vy = GF_XOR(vy, vx);
    GF_ST(x + i, vx);  GF_ST(y + i, vy);
  }
  if (i < n) xpar_gf8_fft2_ref(x + i, y + i, n - i, m->c);
}

static void k_ifft8(u8 * x, u8 * y, sz n, const xpar_gf8_coef * m) {
  sz i = 0, body = GF_BODY8(n);
  GF_MUL8_VARS
  for (; i < body; i += GF_VB) {
    GF_V vx = GF_LD(x + i), vy = GF_XOR(GF_LD(y + i), vx);
    vx = GF_XOR(vx, GF_MUL8(vy));
    GF_ST(x + i, vx);  GF_ST(y + i, vy);
  }
  if (i < n) xpar_gf8_ifft2_ref(x + i, y + i, n - i, m->c);
}

/*  GF(2^16).  */

static void k_mac16(u8 * d, const u8 * s, sz n, const xpar_gf16_coef * m) {
  sz i = 0, body = GF_BODY16(n);
  GF_MUL16_VARS
  for (; i < body; i += 2 * GF_VB) {
    GF_V r0, r1;
    GF_MUL16_PAIR(GF_LD(s + i), GF_LD(s + i + GF_VB), r0, r1);
    GF_ST(d + i,         GF_XOR(GF_LD(d + i),         r0));
    GF_ST(d + i + GF_VB, GF_XOR(GF_LD(d + i + GF_VB), r1));
  }
  if (i < n) xpar_gf16_mac_ref(d + i, s + i, n - i, m->c);
}

#if defined(GF_AFFINE) && defined(GF_ISA_AVX2)
static void k_mac16x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf16_coef m[2]) {
  sz i = 0, body = GF_BODY16(n);
  const GF_V a0 = GF_SET64(m[0].affine[0]);
  const GF_V a1 = GF_SET64(m[0].affine[1]);
  const GF_V a2 = GF_SET64(m[0].affine[2]);
  const GF_V a3 = GF_SET64(m[0].affine[3]);
  const GF_V b0 = GF_SET64(m[1].affine[0]);
  const GF_V b1 = GF_SET64(m[1].affine[1]);
  const GF_V b2 = GF_SET64(m[1].affine[2]);
  const GF_V b3 = GF_SET64(m[1].affine[3]);
  GF_PLANE_VARS
  for (; i < body; i += 2 * GF_VB) {
    GF_V lo, hi, al, ah, bl, bh, p0, p1;
    GF_DEINT(GF_LD(s + i), GF_LD(s + i + GF_VB), lo, hi);
    al = GF_XOR(GF_AFF(lo, a0), GF_AFF(hi, a1));
    ah = GF_XOR(GF_AFF(lo, a2), GF_AFF(hi, a3));
    GF_INT(al, ah, p0, p1);
    GF_ST(d[0] + i, GF_XOR(GF_LD(d[0] + i), p0));
    GF_ST(d[0] + i + GF_VB,
          GF_XOR(GF_LD(d[0] + i + GF_VB), p1));
    bl = GF_XOR(GF_AFF(lo, b0), GF_AFF(hi, b1));
    bh = GF_XOR(GF_AFF(lo, b2), GF_AFF(hi, b3));
    GF_INT(bl, bh, p0, p1);
    GF_ST(d[1] + i, GF_XOR(GF_LD(d[1] + i), p0));
    GF_ST(d[1] + i + GF_VB,
          GF_XOR(GF_LD(d[1] + i + GF_VB), p1));
  }
  if (i < n)
    for (u32 j = 0; j < 2; j++)
      xpar_gf16_mac_ref(d[j] + i, s + i, n - i, m[j].c);
}
#else
static void k_mac16x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf16_coef m[2]) {
  k_mac16(d[0], s, n, &m[0]);
  k_mac16(d[1], s, n, &m[1]);
}
#endif

static void k_mul16(u8 * d, const u8 * s, sz n, const xpar_gf16_coef * m) {
  sz i = 0, body = GF_BODY16(n);
  GF_MUL16_VARS
  for (; i < body; i += 2 * GF_VB) {
    GF_V r0, r1;
    GF_MUL16_PAIR(GF_LD(s + i), GF_LD(s + i + GF_VB), r0, r1);
    GF_ST(d + i, r0);  GF_ST(d + i + GF_VB, r1);
  }
  if (i < n) xpar_gf16_mul_ref(d + i, s + i, n - i, m->c);
}

static void k_fft16(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m) {
  sz i = 0, body = GF_BODY16(n);
  GF_MUL16_VARS
  for (; i < body; i += 2 * GF_VB) {
    GF_V r0, r1, y0 = GF_LD(y + i), y1 = GF_LD(y + i + GF_VB);
    GF_MUL16_PAIR(y0, y1, r0, r1);
    r0 = GF_XOR(GF_LD(x + i), r0);  r1 = GF_XOR(GF_LD(x + i + GF_VB), r1);
    GF_ST(x + i, r0);              GF_ST(x + i + GF_VB, r1);
    GF_ST(y + i, GF_XOR(y0, r0));  GF_ST(y + i + GF_VB, GF_XOR(y1, r1));
  }
  if (i < n) xpar_gf16_fft2_ref(x + i, y + i, n - i, m->c);
}

static void k_ifft16(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m) {
  sz i = 0, body = GF_BODY16(n);
  GF_MUL16_VARS
  for (; i < body; i += 2 * GF_VB) {
    GF_V r0, r1, x0 = GF_LD(x + i), x1 = GF_LD(x + i + GF_VB);
    GF_V y0 = GF_XOR(GF_LD(y + i), x0), y1 = GF_XOR(GF_LD(y + i + GF_VB), x1);
    GF_ST(y + i, y0);  GF_ST(y + i + GF_VB, y1);
    GF_MUL16_PAIR(y0, y1, r0, r1);
    GF_ST(x + i, GF_XOR(x0, r0));  GF_ST(x + i + GF_VB, GF_XOR(x1, r1));
  }
  if (i < n) xpar_gf16_ifft2_ref(x + i, y + i, n - i, m->c);
}

/*  XOR.  */

static void k_xor2(u8 * d, const u8 * s, sz n) {
  sz i = 0, body = GF_BODY8(n);
  for (; i < body; i += GF_VB)
    GF_ST(d + i, GF_XOR(GF_LD(d + i), GF_LD(s + i)));
  if (i < n) xpar_xor2_ref(d + i, s + i, n - i);
}

static void k_xor3(u8 * d, const u8 * a, const u8 * b, sz n) {
  sz i = 0, body = GF_BODY8(n);
  for (; i < body; i += GF_VB)
    GF_ST(d + i, GF_XOR(GF_LD(a + i), GF_LD(b + i)));
  if (i < n) xpar_xor3_ref(d + i, a + i, b + i, n - i);
}

#else

/*  Scalar tier.
    The reference kernels are the tier. Keeping them in gf.c rather than
    here means the fallback and the vector tails are one body of code
    that cannot drift apart.  */

static void k_mac8(u8 * d, const u8 * s, sz n, const xpar_gf8_coef * m) {
  xpar_gf8_mac_ref(d, s, n, m->c);
}
static void k_mac8x2(u8 * const d[2], const u8 * s, sz n,
                     const xpar_gf8_coef m[2]) {
  for (u32 j = 0; j < 2; j++) xpar_gf8_mac_ref(d[j], s, n, m[j].c);
}
static void k_mul8(u8 * d, const u8 * s, sz n, const xpar_gf8_coef * m) {
  xpar_gf8_mul_ref(d, s, n, m->c);
}
static void k_fft8(u8 * x, u8 * y, sz n, const xpar_gf8_coef * m) {
  xpar_gf8_fft2_ref(x, y, n, m->c);
}
static void k_ifft8(u8 * x, u8 * y, sz n, const xpar_gf8_coef * m) {
  xpar_gf8_ifft2_ref(x, y, n, m->c);
}
static void k_mac16(u8 * d, const u8 * s, sz n, const xpar_gf16_coef * m) {
  xpar_gf16_mac_ref(d, s, n, m->c);
}
static void k_mac16x2(u8 * const d[2], const u8 * s, sz n,
                      const xpar_gf16_coef m[2]) {
  for (u32 j = 0; j < 2; j++) xpar_gf16_mac_ref(d[j], s, n, m[j].c);
}
static void k_mul16(u8 * d, const u8 * s, sz n, const xpar_gf16_coef * m) {
  xpar_gf16_mul_ref(d, s, n, m->c);
}
static void k_fft16(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m) {
  xpar_gf16_fft2_ref(x, y, n, m->c);
}
static void k_ifft16(u8 * x, u8 * y, sz n, const xpar_gf16_coef * m) {
  xpar_gf16_ifft2_ref(x, y, n, m->c);
}
static void k_xor2(u8 * d, const u8 * s, sz n) { xpar_xor2_ref(d, s, n); }
static void k_xor3(u8 * d, const u8 * a, const u8 * b, sz n) {
  xpar_xor3_ref(d, a, b, n);
}

#endif

const xpar_gf_kernels GF_CAT(xpar_gf_kernels, GF_SUF) = {
  GF_NAME,
  k_mac8, k_mac8x2, k_mac16, k_mac16x2, k_mul8, k_mul16, k_xor2, k_xor3,
  k_fft8,  k_fft16,  k_ifft8, k_ifft16
};
