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

/*  xpar: SSE4.2 CRC chains with VPCLMULQDQ recombination.  */

#include "crc32c.h"

#include <immintrin.h>

static u32 vc_join(u32 a, u32 b, u32 c, u32 ka, u32 kb) {
  __m128i av = _mm_cvtsi32_si128((int) a);
  __m128i bv = _mm_cvtsi32_si128((int) b);
  __m128i ak = _mm_cvtsi32_si128((int) ka);
  __m128i bk = _mm_cvtsi32_si128((int) kb);
  __m256i values = _mm256_inserti128_si256(_mm256_castsi128_si256(av), bv, 1);
  __m256i keys = _mm256_inserti128_si256(_mm256_castsi128_si256(ak), bk, 1);
  __m256i product = _mm256_clmulepi64_epi128(values, keys, 0x00);
  u64 pa = (u64) _mm_cvtsi128_si64(_mm256_castsi256_si128(product));
  u64 pb = (u64) _mm_cvtsi128_si64(_mm256_extracti128_si256(product, 1));
  return (u32) _mm_crc32_u64(0, pa << 1) ^
         (u32) _mm_crc32_u64(0, pb << 1) ^ c;
}

static u32 vc_run(u32 crc, const u8 * p, sz run, u32 ka, u32 kb) {
  u32 c0 = crc, c1 = 0, c2 = 0;
  for (sz i = 0; i < run; i += 8) {
    c0 = (u32) _mm_crc32_u64(c0, xpar_rd64(p + i));
    c1 = (u32) _mm_crc32_u64(c1, xpar_rd64(p + run + i));
    c2 = (u32) _mm_crc32_u64(c2, xpar_rd64(p + 2 * run + i));
  }
  return vc_join(c0, c1, c2, ka, kb);
}

u32 xpar_crc32c_vpclmul(u32 crc, const u8 * p, sz n) {
  while (n >= 3 * XPAR_CRC32C_LONG) {
    crc = vc_run(crc, p, XPAR_CRC32C_LONG, XPAR_CRC_K_LONG2, XPAR_CRC_K_LONG);
    p += 3 * XPAR_CRC32C_LONG;  n -= 3 * XPAR_CRC32C_LONG;
  }
  while (n >= 3 * XPAR_CRC32C_SHORT) {
    crc = vc_run(crc, p, XPAR_CRC32C_SHORT,
                 XPAR_CRC_K_SHORT2, XPAR_CRC_K_SHORT);
    p += 3 * XPAR_CRC32C_SHORT;  n -= 3 * XPAR_CRC32C_SHORT;
  }
  while (n >= 8) {
    crc = (u32) _mm_crc32_u64(crc, xpar_rd64(p));
    p += 8;  n -= 8;
  }
  while (n) { crc = _mm_crc32_u8(crc, *p++);  n--; }
  return crc;
}
