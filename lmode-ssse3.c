/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    SSSE3 variant of lmode; aarch64 builds use sse2neon.  */

#include "platform.h"

#if defined(HAVE_SSSE3) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))

#define LEO_FUNC(n) n ## _ssse3
#define LEO_TRY_SSSE3
#define LEO_M128 __m128i

#if defined(XPAR_AARCH64) && defined(HAVE_NEON)
  #include "sse2neon.h"
#else
  #include <emmintrin.h>
  #include <tmmintrin.h>
#endif

#include "lmode-impl.h"

/*  Fill 128-bit pshufb tables; called once at startup.  */
void FillPshufbTables_ssse3(void) {
  Multiply128LUT_t * tbl = (Multiply128LUT_t *) Multiply128LUT_storage;
  for (unsigned log_m = 0; log_m < kOrder; ++log_m) {
    for (unsigned i = 0, shift = 0; i < 2; ++i, shift += 4) {
      uint8_t lut[16];
      for (ffe_t x = 0; x < 16; ++x)
        lut[x] = MultiplyLog(x << shift, (ffe_t) (log_m));
      const LEO_M128 * v_ptr = (const LEO_M128 *) (&lut[0]);
      const LEO_M128 value = _mm_loadu_si128(v_ptr);
      _mm_storeu_si128((LEO_M128 *) &tbl[log_m].Value[i], value);
    }
  }
}

#endif /*  HAVE_SSSE3 || (XPAR_AARCH64 && HAVE_NEON)  */
