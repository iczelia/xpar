/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    AVX2 BLAKE2b compression; each state row fits one __m256i.  */

#include "blake2b-impl.h"

#ifdef HAVE_AVX2

#include <immintrin.h>

void blake2b_compress_avx2(blake2b_state *restrict s,
                           const u8 *restrict block) {
  const __m256i rot24 =
      _mm256_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3,
                       4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10);
  const __m256i rot16 =
      _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2,
                       3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9);

  u64 m[16];
  for (int i = 0; i < 16; i++)
    m[i] = blake2b_load64(block + i * 8);

  __m256i row1 = _mm256_loadu_si256((const __m256i *)&s->h[0]);
  __m256i row2 = _mm256_loadu_si256((const __m256i *)&s->h[4]);
  __m256i row3 = _mm256_loadu_si256((const __m256i *)&blake2b_IV[0]);
  __m256i row4 =
      _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)&blake2b_IV[4]),
                       _mm256_set_epi64x((int64_t)s->f[1], (int64_t)s->f[0],
                                         (int64_t)s->t[1], (int64_t)s->t[0]));

#define G1(buf)                                                                \
  do {                                                                         \
    row1 = _mm256_add_epi64(_mm256_add_epi64(row1, buf), row2);                \
    row4 = _mm256_shuffle_epi32(_mm256_xor_si256(row4, row1),                  \
                                _MM_SHUFFLE(2, 3, 0, 1));                      \
    row3 = _mm256_add_epi64(row3, row4);                                       \
    row2 = _mm256_shuffle_epi8(_mm256_xor_si256(row2, row3), rot24);           \
  } while (0)

#define G2(buf)                                                                \
  do {                                                                         \
    row1 = _mm256_add_epi64(_mm256_add_epi64(row1, buf), row2);                \
    row4 = _mm256_shuffle_epi8(_mm256_xor_si256(row4, row1), rot16);           \
    row3 = _mm256_add_epi64(row3, row4);                                       \
    {                                                                          \
      __m256i t = _mm256_xor_si256(row2, row3);                                \
      row2 =                                                                   \
          _mm256_xor_si256(_mm256_srli_epi64(t, 63), _mm256_add_epi64(t, t));  \
    }                                                                          \
  } while (0)

#define DIAG()                                                                 \
  do {                                                                         \
    row2 = _mm256_permute4x64_epi64(row2, _MM_SHUFFLE(0, 3, 2, 1));            \
    row3 = _mm256_permute4x64_epi64(row3, _MM_SHUFFLE(1, 0, 3, 2));            \
    row4 = _mm256_permute4x64_epi64(row4, _MM_SHUFFLE(2, 1, 0, 3));            \
  } while (0)

#define UNDIAG()                                                               \
  do {                                                                         \
    row2 = _mm256_permute4x64_epi64(row2, _MM_SHUFFLE(2, 1, 0, 3));            \
    row3 = _mm256_permute4x64_epi64(row3, _MM_SHUFFLE(1, 0, 3, 2));            \
    row4 = _mm256_permute4x64_epi64(row4, _MM_SHUFFLE(0, 3, 2, 1));            \
  } while (0)

  for (int r = 0; r < 12; r++) {
    const u8 *p = blake2b_sigma[r];
    __m256i buf;
    buf = _mm256_set_epi64x(m[p[6]], m[p[4]], m[p[2]], m[p[0]]);
    G1(buf);
    buf = _mm256_set_epi64x(m[p[7]], m[p[5]], m[p[3]], m[p[1]]);
    G2(buf);
    DIAG();
    buf = _mm256_set_epi64x(m[p[14]], m[p[12]], m[p[10]], m[p[8]]);
    G1(buf);
    buf = _mm256_set_epi64x(m[p[15]], m[p[13]], m[p[11]], m[p[9]]);
    G2(buf);
    UNDIAG();
  }

  _mm256_storeu_si256(
      (__m256i *)&s->h[0],
      _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)&s->h[0]),
                       _mm256_xor_si256(row1, row3)));
  _mm256_storeu_si256(
      (__m256i *)&s->h[4],
      _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)&s->h[4]),
                       _mm256_xor_si256(row2, row4)));
}

#endif
