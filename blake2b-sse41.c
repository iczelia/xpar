/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    SSE4.1 BLAKE2b compression; aarch64 builds use sse2neon.  */

#include "blake2b-impl.h"

#if defined(HAVE_SSE41) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))

#if defined(XPAR_AARCH64) && defined(HAVE_NEON)
  #include "sse2neon.h"
#else
  #include <smmintrin.h>
#endif

void blake2b_compress_sse41(blake2b_state * restrict s,
                            const u8 * restrict block) {
  const __m128i rot24 = _mm_setr_epi8( 3, 4, 5, 6, 7, 0, 1, 2,
                                      11,12,13,14,15, 8, 9,10);
  const __m128i rot16 = _mm_setr_epi8( 2, 3, 4, 5, 6, 7, 0, 1,
                                      10,11,12,13,14,15, 8, 9);

  u64 m[16];
  for (int i = 0; i < 16; i++) m[i] = blake2b_load64(block + i * 8);

  __m128i row1l = _mm_loadu_si128((const __m128i *)&s->h[0]);
  __m128i row1h = _mm_loadu_si128((const __m128i *)&s->h[2]);
  __m128i row2l = _mm_loadu_si128((const __m128i *)&s->h[4]);
  __m128i row2h = _mm_loadu_si128((const __m128i *)&s->h[6]);
  __m128i row3l = _mm_loadu_si128((const __m128i *)&blake2b_IV[0]);
  __m128i row3h = _mm_loadu_si128((const __m128i *)&blake2b_IV[2]);
  __m128i row4l = _mm_xor_si128(
    _mm_loadu_si128((const __m128i *)&blake2b_IV[4]),
    _mm_loadu_si128((const __m128i *)&s->t[0]));
  __m128i row4h = _mm_xor_si128(
    _mm_loadu_si128((const __m128i *)&blake2b_IV[6]),
    _mm_loadu_si128((const __m128i *)&s->f[0]));

  #define G1(b0, b1) do {                                         \
    row1l = _mm_add_epi64(_mm_add_epi64(row1l, b0), row2l);       \
    row1h = _mm_add_epi64(_mm_add_epi64(row1h, b1), row2h);       \
    row4l = _mm_shuffle_epi32(_mm_xor_si128(row4l, row1l),        \
                              _MM_SHUFFLE(2,3,0,1));              \
    row4h = _mm_shuffle_epi32(_mm_xor_si128(row4h, row1h),        \
                              _MM_SHUFFLE(2,3,0,1));              \
    row3l = _mm_add_epi64(row3l, row4l);                          \
    row3h = _mm_add_epi64(row3h, row4h);                          \
    row2l = _mm_shuffle_epi8(_mm_xor_si128(row2l, row3l), rot24); \
    row2h = _mm_shuffle_epi8(_mm_xor_si128(row2h, row3h), rot24); \
  } while (0)

  #define G2(b0, b1) do {                                         \
    row1l = _mm_add_epi64(_mm_add_epi64(row1l, b0), row2l);       \
    row1h = _mm_add_epi64(_mm_add_epi64(row1h, b1), row2h);       \
    row4l = _mm_shuffle_epi8(_mm_xor_si128(row4l, row1l), rot16); \
    row4h = _mm_shuffle_epi8(_mm_xor_si128(row4h, row1h), rot16); \
    row3l = _mm_add_epi64(row3l, row4l);                          \
    row3h = _mm_add_epi64(row3h, row4h);                          \
    { __m128i t = _mm_xor_si128(row2l, row3l);                    \
      row2l = _mm_xor_si128(_mm_srli_epi64(t, 63),                \
                            _mm_add_epi64(t, t)); }               \
    { __m128i t = _mm_xor_si128(row2h, row3h);                    \
      row2h = _mm_xor_si128(_mm_srli_epi64(t, 63),                \
                            _mm_add_epi64(t, t)); }               \
  } while (0)

  #define DIAG() do {                                             \
    __m128i t = _mm_alignr_epi8(row2h, row2l, 8);                 \
    row2h = _mm_alignr_epi8(row2l, row2h, 8);                     \
    row2l = t;                                                    \
    t = row3l; row3l = row3h; row3h = t;                          \
    t = _mm_alignr_epi8(row4h, row4l, 8);                         \
    row4l = _mm_alignr_epi8(row4l, row4h, 8);                     \
    row4h = t;                                                    \
  } while (0)

  #define UNDIAG() do {                                           \
    __m128i t = _mm_alignr_epi8(row2l, row2h, 8);                 \
    row2h = _mm_alignr_epi8(row2h, row2l, 8);                     \
    row2l = t;                                                    \
    t = row3l; row3l = row3h; row3h = t;                          \
    t = _mm_alignr_epi8(row4l, row4h, 8);                         \
    row4l = _mm_alignr_epi8(row4h, row4l, 8);                     \
    row4h = t;                                                    \
  } while (0)

  for (int r = 0; r < 12; r++) {
    const u8 * p = blake2b_sigma[r];
    __m128i b0, b1;

    b0 = _mm_set_epi64x(m[p[ 2]], m[p[ 0]]);
    b1 = _mm_set_epi64x(m[p[ 6]], m[p[ 4]]);
    G1(b0, b1);
    b0 = _mm_set_epi64x(m[p[ 3]], m[p[ 1]]);
    b1 = _mm_set_epi64x(m[p[ 7]], m[p[ 5]]);
    G2(b0, b1);
    DIAG();
    b0 = _mm_set_epi64x(m[p[10]], m[p[ 8]]);
    b1 = _mm_set_epi64x(m[p[14]], m[p[12]]);
    G1(b0, b1);
    b0 = _mm_set_epi64x(m[p[11]], m[p[ 9]]);
    b1 = _mm_set_epi64x(m[p[15]], m[p[13]]);
    G2(b0, b1);
    UNDIAG();
  }

  __m128i h0 = _mm_loadu_si128((const __m128i *)&s->h[0]);
  __m128i h2 = _mm_loadu_si128((const __m128i *)&s->h[2]);
  __m128i h4 = _mm_loadu_si128((const __m128i *)&s->h[4]);
  __m128i h6 = _mm_loadu_si128((const __m128i *)&s->h[6]);
  _mm_storeu_si128((__m128i *)&s->h[0],
    _mm_xor_si128(h0, _mm_xor_si128(row1l, row3l)));
  _mm_storeu_si128((__m128i *)&s->h[2],
    _mm_xor_si128(h2, _mm_xor_si128(row1h, row3h)));
  _mm_storeu_si128((__m128i *)&s->h[4],
    _mm_xor_si128(h4, _mm_xor_si128(row2l, row4l)));
  _mm_storeu_si128((__m128i *)&s->h[6],
    _mm_xor_si128(h6, _mm_xor_si128(row2h, row4h)));
}

#endif
