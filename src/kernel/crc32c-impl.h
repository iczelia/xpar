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

/*  CRC-32C inner loops instantiated once per ISA.  */

#include "crc32c.h"

#if defined(XPAR_CRC32C_VARIANT_SSE42)
  #include <immintrin.h>
  #include "platform/port-cpu.h"
  #define XPAR_CRC_SFX _sse42
  #if defined(__x86_64__) || defined(_M_X64)
    #define XPAR_CRC_W 8
    #define XPAR_CRC_WORD(c, v) ((u32) _mm_crc32_u64((u64) (c), (u64) (v)))
  #else
    #define XPAR_CRC_W 4
    #define XPAR_CRC_WORD(c, v) _mm_crc32_u32((c), (u32) (v))
  #endif
  #define XPAR_CRC_BYTE(c, b) _mm_crc32_u8((c), (b))
#elif defined(XPAR_CRC32C_VARIANT_ARM)
  #include <arm_acle.h>
  #define XPAR_CRC_SFX _arm
  #if defined(__aarch64__)
    #define XPAR_CRC_W 8
    #define XPAR_CRC_WORD(c, v) __crc32cd((c), (u64) (v))
  #else
    #define XPAR_CRC_W 4
    #define XPAR_CRC_WORD(c, v) __crc32cw((c), (u32) (v))
  #endif
  #define XPAR_CRC_BYTE(c, b) __crc32cb((c), (b))
#else
  #define XPAR_CRC_SFX _scalar
#endif

#define XPAR_CRC_CAT2(a, b) a##b
#define XPAR_CRC_CAT(a, b)  XPAR_CRC_CAT2(a, b)
#define XPAR_CRC_FN(name)   XPAR_CRC_CAT(name, XPAR_CRC_SFX)

#if defined(XPAR_CRC_W)

#if XPAR_CRC_W == 8
static inline u64 xpar_crc_load(const u8 * p) { return xpar_rd64(p); }
#else
static inline u64 xpar_crc_load(const u8 * p) { return (u64) xpar_rd32(p); }
#endif

#if defined(XPAR_CRC32C_VARIANT_SSE42) && defined(HAVE_PCLMUL) &&             \
    (defined(__x86_64__) || defined(_M_X64))

/*  Multiply a bare register by x^(8n) mod G. The carry-less product of
    two reflected 32-bit values sits one bit low, because reflecting a
    63-bit product inside 64 bits loses the top position, so it is shifted
    up by one; the CRC32 instruction over the resulting word supplies the
    missing x^32 and performs the reduction.  */
static inline u32 xpar_crc_clmul(u32 v, u32 k) {
  __m128i r = _mm_clmulepi64_si128(_mm_cvtsi32_si128((int) v),
                                   _mm_cvtsi32_si128((int) k), 0x00);
  return (u32) _mm_crc32_u64(0, (u64) _mm_cvtsi128_si64(r) << 1);
}

/*  A CPU can have SSE4.2 and not PCLMULQDQ (Nehalem), and both live in
    this translation unit, so the recombine is chosen at run time rather
    than by the build flags. Once per twenty-four kilobytes.  */
static int xpar_crc_clmul_ok(void) {
  static int ok = -1;
  if (ok < 0) ok = (xpar_cpu_features() & XPAR_CPU_PCLMUL) != 0;
  return ok;
}

static u32 xpar_crc_join_long(u32 a, u32 b, u32 c) {
  if (!xpar_crc_clmul_ok()) return xpar_crc32c_join3_long(a, b, c);
  return xpar_crc_clmul(a, XPAR_CRC_K_LONG2) ^
         xpar_crc_clmul(b, XPAR_CRC_K_LONG) ^ c;
}

static u32 xpar_crc_join_short(u32 a, u32 b, u32 c) {
  if (!xpar_crc_clmul_ok()) return xpar_crc32c_join3_short(a, b, c);
  return xpar_crc_clmul(a, XPAR_CRC_K_SHORT2) ^
         xpar_crc_clmul(b, XPAR_CRC_K_SHORT) ^ c;
}

#else

static u32 xpar_crc_join_long(u32 a, u32 b, u32 c) {
  return xpar_crc32c_join3_long(a, b, c);
}

static u32 xpar_crc_join_short(u32 a, u32 b, u32 c) {
  return xpar_crc32c_join3_short(a, b, c);
}

#endif

/*  One three-way pass over 3 * `run` bytes.  */
#define XPAR_CRC_THREE(run, join)                                             \
  do {                                                                        \
    u32 c0 = crc, c1 = 0, c2 = 0;                                             \
    const u8 * q = p;                                                         \
    sz i;                                                                     \
    for (i = 0; i < (run); i += XPAR_CRC_W) {                                 \
      c0 = XPAR_CRC_WORD(c0, xpar_crc_load(q));                               \
      c1 = XPAR_CRC_WORD(c1, xpar_crc_load(q + (run)));                       \
      c2 = XPAR_CRC_WORD(c2, xpar_crc_load(q + 2 * (run)));                   \
      q += XPAR_CRC_W;                                                        \
    }                                                                         \
    crc = join(c0, c1, c2);                                                   \
    p += 3 * (run);  n -= 3 * (run);                                          \
  } while (0)

u32 XPAR_CRC_FN(xpar_crc32c)(u32 crc, const u8 * p, sz n) {
  while (n >= 3 * XPAR_CRC32C_LONG)
    XPAR_CRC_THREE(XPAR_CRC32C_LONG, xpar_crc_join_long);
  while (n >= 3 * XPAR_CRC32C_SHORT)
    XPAR_CRC_THREE(XPAR_CRC32C_SHORT, xpar_crc_join_short);
  while (n >= XPAR_CRC_W) {
    crc = XPAR_CRC_WORD(crc, xpar_crc_load(p));
    p += XPAR_CRC_W;  n -= XPAR_CRC_W;
  }
  while (n) { crc = XPAR_CRC_BYTE(crc, *p++);  n--; }
  return crc;
}

#else

u32 xpar_crc32c_scalar(u32 crc, const u8 * p, sz n) {
  while (n >= 8) {
    u32 a = crc ^ xpar_rd32(p), b = xpar_rd32(p + 4);
    crc = xpar_crc32c_tab[7][ a        & 0xFF] ^
          xpar_crc32c_tab[6][(a >>  8) & 0xFF] ^
          xpar_crc32c_tab[5][(a >> 16) & 0xFF] ^
          xpar_crc32c_tab[4][ a >> 24        ] ^
          xpar_crc32c_tab[3][ b        & 0xFF] ^
          xpar_crc32c_tab[2][(b >>  8) & 0xFF] ^
          xpar_crc32c_tab[1][(b >> 16) & 0xFF] ^
          xpar_crc32c_tab[0][ b >> 24        ];
    p += 8;  n -= 8;
  }
  while (n) {
    crc = (crc >> 8) ^ xpar_crc32c_tab[0][(crc ^ *p++) & 0xFF];
    n--;
  }
  return crc;
}

#endif
