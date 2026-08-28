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

/*  BLAKE3 compression instantiated once per ISA.  */

#include "blake3.h"

#if defined(XPAR_BLAKE3_VARIANT_AVX2)
  #define XPAR_B3_SFX  _avx2
  #define XPAR_B3_DEG  8
  #include <immintrin.h>
#elif defined(XPAR_BLAKE3_VARIANT_NEON)
  #define XPAR_B3_SFX  _neon
  #define XPAR_B3_DEG  4
  #include <arm_neon.h>
#else
  #define XPAR_B3_SFX  _scalar
  #define XPAR_B3_DEG  1
#endif

#define XPAR_B3_CAT2(a, b) a##b
#define XPAR_B3_CAT(a, b)  XPAR_B3_CAT2(a, b)
#define XPAR_B3_FN(name)   XPAR_B3_CAT(name, XPAR_B3_SFX)

/*  The scalar compression function is external in the scalar TU, where it
    is the one the driver calls, and internal everywhere else, where it
    only serves the lanes left over past a multiple of the degree.  */
#if defined(XPAR_BLAKE3_VARIANT_SCALAR)
  #define XPAR_B3_LINK
#else
  #define XPAR_B3_LINK static
#endif

/*  Message permutation. Row 0 is the identity and row r+1 is row r
    permuted by (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8),
    which is BLAKE3's sigma; seven rows because the permutation runs
    seven rounds.  */
static const u8 xpar_b3_schedule[7][16] = {
  {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
  {  2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8 },
  {  3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1 },
  { 10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6 },
  { 12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4 },
  {  9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7 },
  { 11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13 }
};

/*  Scalar core.  */

static inline u32 xpar_b3_ror(u32 x, int n) {
  return (x >> n) | (x << (32 - n));
}

/*  The quarter-round. Rotations are 16, 12, 8, 7 to the right, and the
    two message words enter at the two additions.  */
#define XPAR_B3_G(a, b, c, d, x, y)                                           \
  do {                                                                        \
    a = a + b + (x);  d = xpar_b3_ror(d ^ a, 16);                             \
    c = c + d;        b = xpar_b3_ror(b ^ c, 12);                             \
    a = a + b + (y);  d = xpar_b3_ror(d ^ a, 8);                              \
    c = c + d;        b = xpar_b3_ror(b ^ c, 7);                              \
  } while (0)

static inline void xpar_b3_round(u32 * v, const u32 * m, int r) {
  const u8 * s = xpar_b3_schedule[r];
  XPAR_B3_G(v[0], v[4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
  XPAR_B3_G(v[1], v[5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
  XPAR_B3_G(v[2], v[6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
  XPAR_B3_G(v[3], v[7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
  XPAR_B3_G(v[0], v[5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
  XPAR_B3_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
  XPAR_B3_G(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]]);
  XPAR_B3_G(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]]);
}

/*  The 16-word state after the seven rounds, before either of the two
    feed-forward foldings that produce a chaining value or output bytes.
    Words 12 and 13 carry the counter split low then high, 14 the block
    length, 15 the flags.  */
static void xpar_b3_permute(u32 * v, const u32 * cv, const u8 * block,
                            u8 block_len, u64 counter, u8 flags) {
  u32 m[16];
  Fi(16, m[i] = xpar_rd32(block + 4 * i))
  Fi(8, v[i] = cv[i])
  v[ 8] = xpar_blake3_iv[0];  v[ 9] = xpar_blake3_iv[1];
  v[10] = xpar_blake3_iv[2];  v[11] = xpar_blake3_iv[3];
  v[12] = (u32) counter;      v[13] = (u32) (counter >> 32);
  v[14] = block_len;          v[15] = flags;
  xpar_b3_round(v, m, 0);  xpar_b3_round(v, m, 1);
  xpar_b3_round(v, m, 2);  xpar_b3_round(v, m, 3);
  xpar_b3_round(v, m, 4);  xpar_b3_round(v, m, 5);
  xpar_b3_round(v, m, 6);
}

XPAR_B3_LINK void XPAR_B3_FN(xpar_blake3_compress)(u32 * cv, const u8 * block,
                                                   u8 block_len, u64 counter,
                                                   u8 flags) {
  u32 v[16];
  xpar_b3_permute(v, cv, block, block_len, counter, flags);
  Fi(8, cv[i] = v[i] ^ v[i + 8])
}

#if defined(XPAR_BLAKE3_VARIANT_SCALAR)
/*  The full 64-byte output block. The upper half folds in the input
    chaining value, which is what makes the output extendable rather than
    just 32 bytes wide.  */
void xpar_blake3_xof_scalar(const u32 * cv, const u8 * block, u8 block_len,
                            u64 counter, u8 flags, u8 * out) {
  u32 v[16];
  xpar_b3_permute(v, cv, block, block_len, counter, flags);
  Fi(8, xpar_wr32(out + 4 * i, v[i] ^ v[i + 8]);
        xpar_wr32(out + 32 + 4 * i, v[i + 8] ^ cv[i]))
}
#endif

/*  One input, all of its blocks, chaining value out. This is the tail
    path of every hash_many and the whole of the scalar one.  */
static void xpar_b3_one(const u8 * in, sz blocks, const u32 * key,
                        u64 counter, u8 flags, u8 first, u8 last, u8 * out) {
  u32 cv[8];
  Fi(8, cv[i] = key[i])
  for (sz b = 0; b < blocks; b++) {
    u8 bf = (u8) (flags | (b == 0 ? first : 0) |
                  (b + 1 == blocks ? last : 0));
    XPAR_B3_FN(xpar_blake3_compress)(cv, in + b * XPAR_BLAKE3_BLOCK_LEN,
                                     XPAR_BLAKE3_BLOCK_LEN, counter, bf);
  }
  Fi(8, xpar_wr32(out + 4 * i, cv[i]))
}

/*  Vector core.  */

#if defined(XPAR_BLAKE3_HAVE_SIMD)

/*  Prefetch each strided lane; out-of-range hints are safe no-ops.  */
#define XPAR_B3_AHEAD (4 * XPAR_BLAKE3_BLOCK_LEN)
#if defined(__GNUC__) || defined(__clang__)
  #define XPAR_B3_PREFETCH(p) __builtin_prefetch((const void *) (p))
#else
  #define XPAR_B3_PREFETCH(p) ((void) 0)
#endif

#if defined(XPAR_BLAKE3_VARIANT_AVX2)

typedef __m256i xpar_b3_vec;

#define XPAR_B3_SET1(x)   _mm256_set1_epi32((int) (u32) (x))
#define XPAR_B3_ADD(a, b) _mm256_add_epi32(a, b)
#define XPAR_B3_XOR(a, b) _mm256_xor_si256(a, b)

/*  A 32-bit rotate by a multiple of 8 is a byte shuffle, which is one
    port-5 operation instead of a shift pair and an or.  */
static inline xpar_b3_vec xpar_b3_rot16(xpar_b3_vec x) {
  return _mm256_shuffle_epi8(x, _mm256_setr_epi8(
    2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
    2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13));
}

static inline xpar_b3_vec xpar_b3_rot8(xpar_b3_vec x) {
  return _mm256_shuffle_epi8(x, _mm256_setr_epi8(
    1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
    1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12));
}

static inline xpar_b3_vec xpar_b3_rot12(xpar_b3_vec x) {
  return _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20));
}

static inline xpar_b3_vec xpar_b3_rot7(xpar_b3_vec x) {
  return _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25));
}

static inline xpar_b3_vec xpar_b3_ldw(const u32 * p) {
  return _mm256_loadu_si256((const __m256i *) p);
}

/*  Eight vectors of eight words in, the same 64 words out with the roles
    of lane and index exchanged. Unpack pairs, unpack quads, then swap the
    128-bit halves, which is the only step that crosses the lane
    boundary AVX2 imposes.  */
static inline void xpar_b3_transpose(xpar_b3_vec * v) {
  xpar_b3_vec t[8], u[8];
  Fi(4, t[2 * i]     = _mm256_unpacklo_epi32(v[2 * i], v[2 * i + 1]);
        t[2 * i + 1] = _mm256_unpackhi_epi32(v[2 * i], v[2 * i + 1]))
  Fi(2, u[4 * i]     = _mm256_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 1] = _mm256_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 2] = _mm256_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        u[4 * i + 3] = _mm256_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]))
  Fi(4, v[i]     = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
        v[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31))
}

static inline void xpar_b3_load_msg(const u8 * const * in, sz off,
                                    xpar_b3_vec * m) {
  xpar_b3_vec t[8];
  Fi(8, t[i] = _mm256_loadu_si256((const __m256i *) (in[i] + off)))
  xpar_b3_transpose(t);
  Fi(8, m[i] = t[i])
  Fi(8, t[i] = _mm256_loadu_si256((const __m256i *) (in[i] + off + 32)))
  Fi(8, XPAR_B3_PREFETCH(in[i] + off + XPAR_B3_AHEAD))
  xpar_b3_transpose(t);
  Fi(8, m[i + 8] = t[i])
}

static inline void xpar_b3_store_cv(xpar_b3_vec * h, u8 * out) {
  xpar_b3_transpose(h);
  Fi(8, _mm256_storeu_si256((__m256i *) (out + 32 * i), h[i]))
}

#elif defined(XPAR_BLAKE3_VARIANT_NEON)

typedef uint32x4_t xpar_b3_vec;

#define XPAR_B3_SET1(x)   vdupq_n_u32((u32) (x))
#define XPAR_B3_ADD(a, b) vaddq_u32(a, b)
#define XPAR_B3_XOR(a, b) veorq_u32(a, b)

/*  Shift-right-and-insert fuses the two halves of a rotate into one
    instruction after the left shift, so a rotate is two operations
    rather than three.  */
static inline xpar_b3_vec xpar_b3_rot16(xpar_b3_vec x) {
  return vsriq_n_u32(vshlq_n_u32(x, 16), x, 16);
}

static inline xpar_b3_vec xpar_b3_rot12(xpar_b3_vec x) {
  return vsriq_n_u32(vshlq_n_u32(x, 20), x, 12);
}

static inline xpar_b3_vec xpar_b3_rot8(xpar_b3_vec x) {
  return vsriq_n_u32(vshlq_n_u32(x, 24), x, 8);
}

static inline xpar_b3_vec xpar_b3_rot7(xpar_b3_vec x) {
  return vsriq_n_u32(vshlq_n_u32(x, 25), x, 7);
}

static inline xpar_b3_vec xpar_b3_ldw(const u32 * p) {
  return vld1q_u32(p);
}

static inline void xpar_b3_transpose(xpar_b3_vec * v) {
  uint32x4x2_t a = vtrnq_u32(v[0], v[1]), b = vtrnq_u32(v[2], v[3]);
  v[0] = vcombine_u32(vget_low_u32 (a.val[0]), vget_low_u32 (b.val[0]));
  v[1] = vcombine_u32(vget_low_u32 (a.val[1]), vget_low_u32 (b.val[1]));
  v[2] = vcombine_u32(vget_high_u32(a.val[0]), vget_high_u32(b.val[0]));
  v[3] = vcombine_u32(vget_high_u32(a.val[1]), vget_high_u32(b.val[1]));
}

static inline void xpar_b3_load_msg(const u8 * const * in, sz off,
                                    xpar_b3_vec * m) {
  Fj(4, xpar_b3_vec t[4];
        Fi(4, t[i] = vreinterpretq_u32_u8(vld1q_u8(in[i] + off + 16 * j)))
        xpar_b3_transpose(t);
        Fi(4, m[4 * j + i] = t[i]))
  Fi(4, XPAR_B3_PREFETCH(in[i] + off + XPAR_B3_AHEAD))
}

static inline void xpar_b3_store_cv(xpar_b3_vec * h, u8 * out) {
  xpar_b3_transpose(h);
  xpar_b3_transpose(h + 4);
  Fi(4, vst1q_u8(out + 32 * i, vreinterpretq_u8_u32(h[i]));
        vst1q_u8(out + 32 * i + 16, vreinterpretq_u8_u32(h[i + 4])))
}

#endif

#define XPAR_B3_GV(a, b, c, d, x, y)                                          \
  do {                                                                        \
    a = XPAR_B3_ADD(XPAR_B3_ADD(a, b), x);                                    \
    d = xpar_b3_rot16(XPAR_B3_XOR(d, a));                                     \
    c = XPAR_B3_ADD(c, d);  b = xpar_b3_rot12(XPAR_B3_XOR(b, c));             \
    a = XPAR_B3_ADD(XPAR_B3_ADD(a, b), y);                                    \
    d = xpar_b3_rot8(XPAR_B3_XOR(d, a));                                      \
    c = XPAR_B3_ADD(c, d);  b = xpar_b3_rot7(XPAR_B3_XOR(b, c));              \
  } while (0)

/*  `r` is always a literal at the call site so that the schedule folds to
    sixteen constant indices; a runtime r would turn every message word
    into a dependent load.  */
static inline void xpar_b3_vround(xpar_b3_vec * v, const xpar_b3_vec * m,
                                  int r) {
  const u8 * s = xpar_b3_schedule[r];
  XPAR_B3_GV(v[0], v[4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
  XPAR_B3_GV(v[1], v[5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
  XPAR_B3_GV(v[2], v[6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
  XPAR_B3_GV(v[3], v[7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
  XPAR_B3_GV(v[0], v[5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
  XPAR_B3_GV(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
  XPAR_B3_GV(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]]);
  XPAR_B3_GV(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]]);
}

/*  Exactly XPAR_B3_DEG inputs, one per lane, same key and flags.  */
static void xpar_b3_batch(const u8 * const * in, sz blocks, const u32 * key,
                          u64 counter, bool inc, u8 flags, u8 first, u8 last,
                          u8 * out) {
  xpar_b3_vec h[8], v[16], m[16], ctr_lo, ctr_hi;
  u32 lo[XPAR_B3_DEG], hi[XPAR_B3_DEG];
  Fi(XPAR_B3_DEG, u64 c = counter + (inc ? (u64) i : 0);
                  lo[i] = (u32) c;  hi[i] = (u32) (c >> 32))
  ctr_lo = xpar_b3_ldw(lo);  ctr_hi = xpar_b3_ldw(hi);
  Fi(8, h[i] = XPAR_B3_SET1(key[i]))
  for (sz b = 0; b < blocks; b++) {
    u8 bf = (u8) (flags | (b == 0 ? first : 0) |
                  (b + 1 == blocks ? last : 0));
    xpar_b3_load_msg(in, b * XPAR_BLAKE3_BLOCK_LEN, m);
    Fi(8, v[i] = h[i])
    v[ 8] = XPAR_B3_SET1(xpar_blake3_iv[0]);
    v[ 9] = XPAR_B3_SET1(xpar_blake3_iv[1]);
    v[10] = XPAR_B3_SET1(xpar_blake3_iv[2]);
    v[11] = XPAR_B3_SET1(xpar_blake3_iv[3]);
    v[12] = ctr_lo;  v[13] = ctr_hi;
    v[14] = XPAR_B3_SET1(XPAR_BLAKE3_BLOCK_LEN);
    v[15] = XPAR_B3_SET1(bf);
    xpar_b3_vround(v, m, 0);  xpar_b3_vround(v, m, 1);
    xpar_b3_vround(v, m, 2);  xpar_b3_vround(v, m, 3);
    xpar_b3_vround(v, m, 4);  xpar_b3_vround(v, m, 5);
    xpar_b3_vround(v, m, 6);
    Fi(8, h[i] = XPAR_B3_XOR(v[i], v[i + 8]))
  }
  xpar_b3_store_cv(h, out);
}

#endif

void XPAR_B3_FN(xpar_blake3_hash_many)(const u8 * const * inputs, sz count,
                                       sz blocks, const u32 * key,
                                       u64 counter, bool inc, u8 flags,
                                       u8 first, u8 last, u8 * out) {
#if defined(XPAR_BLAKE3_HAVE_SIMD)
  while (count >= XPAR_B3_DEG) {
    xpar_b3_batch(inputs, blocks, key, counter, inc, flags, first, last, out);
    inputs += XPAR_B3_DEG;  count -= XPAR_B3_DEG;
    out += XPAR_B3_DEG * XPAR_BLAKE3_OUT_LEN;
    if (inc) counter += XPAR_B3_DEG;
  }
  /*  Pad partial SIMD groups with lane 0 and discard the extra outputs.  */
  if (XPAR_B3_DEG > 1 && count > 1) {
    const u8 * pad[XPAR_B3_DEG];
    u8 tmp[XPAR_B3_DEG * XPAR_BLAKE3_OUT_LEN];
    sz i;
    for (i = 0; i < count; i++) pad[i] = inputs[i];
    for (; i < XPAR_B3_DEG; i++) pad[i] = inputs[0];
    xpar_b3_batch(pad, blocks, key, counter, inc, flags, first, last, tmp);
    xpar_memcpy(out, tmp, count * XPAR_BLAKE3_OUT_LEN);
    return;
  }
#endif
  while (count) {
    xpar_b3_one(inputs[0], blocks, key, counter, flags, first, last, out);
    inputs++;  count--;  out += XPAR_BLAKE3_OUT_LEN;
    if (inc) counter++;
  }
}
