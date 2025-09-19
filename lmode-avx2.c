/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    AVX2 variant of lmode; x86_64 only.  */

#include "platform.h"

#if defined(HAVE_AVX2)

#define LEO_FUNC(n) n ## _avx2
#define LEO_TRY_AVX2
#define LEO_ALIGN_BYTES 32
#define LEO_M256 __m256i

#include <x86intrin.h>

#include "lmode-impl.h"

/*  Fill 256-bit pshufb tables; called once at startup.  */
void FillPshufbTables_avx2(void) {
  Multiply256LUT_t * tbl = (Multiply256LUT_t *) Multiply256LUT_storage;
  for (unsigned log_m = 0; log_m < kOrder; ++log_m) {
    for (unsigned i = 0, shift = 0; i < 2; ++i, shift += 4) {
      uint8_t lut[16];
      for (ffe_t x = 0; x < 16; ++x)
        lut[x] = MultiplyLog(x << shift, (ffe_t) (log_m));
      const __m128i * v_ptr = (const __m128i *) (&lut[0]);
      const __m128i value = _mm_loadu_si128(v_ptr);
      _mm256_storeu_si256((LEO_M256 *) &tbl[log_m].Value[i],
                          _mm256_broadcastsi128_si256(value));
    }
  }
}

#endif /*  HAVE_AVX2  */
