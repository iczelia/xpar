/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    Shared body for lmode-{scalar,ssse3,avx2}.c. Each TU defines
    LEO_FUNC(n) (symbol suffix) and optional LEO_TRY_SSSE3/AVX2.  */

#ifndef _LMODE_IMPL_H_
#define _LMODE_IMPL_H_

#include "common.h"
#include "config.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifndef LEO_ALIGN_BYTES
  #define LEO_ALIGN_BYTES 16
#endif

#define LEO_FORCE_INLINE inline __attribute__((always_inline))
#define LEO_ALIGNED __attribute__((aligned(LEO_ALIGN_BYTES)))

#define LEO_ERROR_BITFIELD_OPT
#define LEO_INTERLEAVE_BUTTERFLY4_OPT
#define LEO_M1_OPT
#define LEO_USE_VECTOR4_OPT

/*  Finite field element type.  */
typedef uint8_t ffe_t;

#define kBits 8
#define kOrder 256
#define kModulus 255
#define kPolynomial 0x11D

/*  Shared state, defined in lmode.c.  */
extern ffe_t LogLUT[kOrder];
extern ffe_t ExpLUT[kOrder];
extern ffe_t FFTSkew[kModulus];
extern ffe_t LogWalsh[kOrder];
extern const ffe_t * Multiply8LUT;
extern void * Multiply128LUT_storage;
extern void * Multiply256LUT_storage;

/*  FWHT stays scalar in lmode.c.  */
void FWHT(ffe_t * data, const unsigned m, const unsigned m_truncated);

/*  Portable helpers.  */
static LEO_FORCE_INLINE uint32_t LastNonzeroBit32(uint32_t x) {
  return 31 - (uint32_t) __builtin_clz(x);
}

static LEO_FORCE_INLINE uint32_t NextPow2(uint32_t n) {
  return 2UL << LastNonzeroBit32(n - 1);
}

/*  Scalar field ops; static inline so each TU has its own copy.  */
static inline ffe_t AddMod(const ffe_t a, const ffe_t b) {
  const unsigned sum = (unsigned) (a) + b;
  return (ffe_t) (sum + (sum >> kBits));
}
static inline ffe_t SubMod(const ffe_t a, const ffe_t b) {
  const unsigned dif = (unsigned) (a) - b;
  return (ffe_t) (dif + (dif >> kBits));
}

#define FWHT_2(a, b)                \
  {                                 \
    const ffe_t sum = AddMod(a, b); \
    const ffe_t dif = SubMod(a, b); \
    a = sum;                        \
    b = dif;                        \
  }

static LEO_FORCE_INLINE void FWHT_4(ffe_t * data, unsigned s) {
  const unsigned s2 = s << 1;
  ffe_t t0 = data[0];
  ffe_t t1 = data[s];
  ffe_t t2 = data[s2];
  ffe_t t3 = data[s2 + s];
  FWHT_2(t0, t1);
  FWHT_2(t2, t3);
  FWHT_2(t0, t2);
  FWHT_2(t1, t3);
  data[0] = t0;
  data[s] = t1;
  data[s2] = t2;
  data[s2 + s] = t3;
}

static inline ffe_t MultiplyLog(ffe_t a, ffe_t log_b) {
  if (a == 0) return 0;
  return ExpLUT[AddMod(LogLUT[a], log_b)];
}

/*  SIMD LUT types; instantiated only under matching LEO_TRY_* macro.  */
#if defined(LEO_TRY_SSSE3)
typedef struct {
  LEO_M128 Value[2];
} Multiply128LUT_t;
#define Multiply128LUT ((const Multiply128LUT_t *) Multiply128LUT_storage)

/*   128-bit x_reg ^= y_reg * log_m  */
#define LEO_MULADD_128(x_reg, y_reg, table_lo, table_hi) \
  {                                                      \
    LEO_M128 lo = _mm_and_si128(y_reg, clr_mask);        \
    lo = _mm_shuffle_epi8(table_lo, lo);                 \
    LEO_M128 hi = _mm_srli_epi64(y_reg, 4);              \
    hi = _mm_and_si128(hi, clr_mask);                    \
    hi = _mm_shuffle_epi8(table_hi, hi);                 \
    x_reg = _mm_xor_si128(x_reg, _mm_xor_si128(lo, hi)); \
  }
#endif

#if defined(LEO_TRY_AVX2)
typedef struct {
  LEO_M256 Value[2];
} Multiply256LUT_t;
#define Multiply256LUT ((const Multiply256LUT_t *) Multiply256LUT_storage)

/*   256-bit x_reg ^= y_reg * log_m  */
#define LEO_MULADD_256(x_reg, y_reg, table_lo, table_hi)       \
  {                                                            \
    LEO_M256 lo = _mm256_and_si256(y_reg, clr_mask);           \
    lo = _mm256_shuffle_epi8(table_lo, lo);                    \
    LEO_M256 hi = _mm256_srli_epi64(y_reg, 4);                 \
    hi = _mm256_and_si256(hi, clr_mask);                       \
    hi = _mm256_shuffle_epi8(table_hi, hi);                    \
    x_reg = _mm256_xor_si256(x_reg, _mm256_xor_si256(lo, hi)); \
  }
#endif

/*  XOR primitives; callers use LEO_FUNC(name) for per-variant symbols.  */
void LEO_FUNC(xor_mem)(void * restrict x,
                       const void * restrict y, uint64_t bytes);
#ifdef LEO_M1_OPT
void LEO_FUNC(xor_mem_2to1)(void * restrict x,
                            const void * restrict y,
                            const void * restrict z,
                            uint64_t bytes);
#endif
#ifdef LEO_USE_VECTOR4_OPT
void LEO_FUNC(xor_mem4)(void * restrict x_0, const void * restrict y_0,
                        void * restrict x_1, const void * restrict y_1,
                        void * restrict x_2, const void * restrict y_2,
                        void * restrict x_3, const void * restrict y_3,
                        uint64_t bytes);
#endif
void LEO_FUNC(VectorXOR)(const uint64_t bytes, unsigned count,
                         void ** x, void ** y);

/*  XORSummer: tiny inline helper state.  */
typedef struct {
  void * DestBuffer;
  const void * Waiting;
} XORSummer;

static LEO_FORCE_INLINE void XORSummer_Initialize(XORSummer * summer,
                                                  void * dest) {
  summer->DestBuffer = dest;
  summer->Waiting = NULL;
}

static LEO_FORCE_INLINE void XORSummer_Add(XORSummer * summer,
                                           const void * src,
                                           const uint64_t bytes) {
#ifdef LEO_M1_OPT
  if (summer->Waiting) {
    LEO_FUNC(xor_mem_2to1)(summer->DestBuffer, src, summer->Waiting, bytes);
    summer->Waiting = NULL;
  } else
    summer->Waiting = src;
#else
  LEO_FUNC(xor_mem)(summer->DestBuffer, src, bytes);
#endif
}

static LEO_FORCE_INLINE void XORSummer_Finalize(XORSummer * summer,
                                                const uint64_t bytes) {
#ifdef LEO_M1_OPT
  if (summer->Waiting)
    LEO_FUNC(xor_mem)(summer->DestBuffer, summer->Waiting, bytes);
#endif
}

/*  Reference scalar multipliers; fallback when no intrinsic path.  */

static LEO_FORCE_INLINE void RefMulAdd(void * restrict x,
                                       const void * restrict y,
                                       ffe_t log_m,
                                       uint64_t bytes) {
  const ffe_t * restrict lut = Multiply8LUT + (unsigned) log_m * 256;
  const ffe_t * restrict y1 = (const ffe_t *) (y);
  uint64_t * restrict x8 = (uint64_t *) (x);
  do {
    for (unsigned j = 0; j < 8; ++j) {
      uint64_t x_0 = x8[j];
      x_0 ^= (uint64_t) lut[y1[0]];
      x_0 ^= (uint64_t) lut[y1[1]] << 8;
      x_0 ^= (uint64_t) lut[y1[2]] << 16;
      x_0 ^= (uint64_t) lut[y1[3]] << 24;
      x_0 ^= (uint64_t) lut[y1[4]] << 32;
      x_0 ^= (uint64_t) lut[y1[5]] << 40;
      x_0 ^= (uint64_t) lut[y1[6]] << 48;
      x_0 ^= (uint64_t) lut[y1[7]] << 56;
      x8[j] = x_0;
      y1 += 8;
    }
    x8 += 8;
    bytes -= 64;
  } while (bytes > 0);
}

static LEO_FORCE_INLINE void RefMul(void * restrict x,
                                    const void * restrict y,
                                    ffe_t log_m,
                                    uint64_t bytes) {
  const ffe_t * restrict lut = Multiply8LUT + (unsigned) log_m * 256;
  const ffe_t * restrict y1 = (const ffe_t *) (y);
  uint64_t * restrict x8 = (uint64_t *) (x);
  do {
    for (unsigned j = 0; j < 8; ++j) {
      uint64_t x_0 = (uint64_t) lut[y1[0]];
      x_0 ^= (uint64_t) lut[y1[1]] << 8;
      x_0 ^= (uint64_t) lut[y1[2]] << 16;
      x_0 ^= (uint64_t) lut[y1[3]] << 24;
      x_0 ^= (uint64_t) lut[y1[4]] << 32;
      x_0 ^= (uint64_t) lut[y1[5]] << 40;
      x_0 ^= (uint64_t) lut[y1[6]] << 48;
      x_0 ^= (uint64_t) lut[y1[7]] << 56;
      x8[j] = x_0;
      y1 += 8;
    }
    x8 += 8;
    bytes -= 64;
  } while (bytes > 0);
}

/*  ----------------------------------------------------------------------
    Public encode/decode entry points (per-variant).  */

/*  Wrapper prototypes for the external dispatcher in lmode.c.  */
void LEO_FUNC(EncodeM1)(uint64_t buffer_bytes,
                        unsigned original_count,
                        const void * const * const original_data,
                        void * recovery_data);
void LEO_FUNC(DecodeM1)(uint64_t buffer_bytes,
                        unsigned original_count,
                        const void * const * original_data,
                        const void * recovery_data,
                        void * work_data);
void LEO_FUNC(ReedSolomonEncode)(uint64_t buffer_bytes,
                                 unsigned original_count,
                                 unsigned recovery_count,
                                 unsigned m,
                                 const void * const * const data,
                                 void ** work);
void LEO_FUNC(ReedSolomonDecode)(uint64_t buffer_bytes,
                                 unsigned original_count,
                                 unsigned recovery_count,
                                 unsigned m,
                                 unsigned n,
                                 const void * const * const original,
                                 const void * const * const recovery,
                                 void ** work);

/*  ======================================================================
    Bodies follow.
    ======================================================================  */

/*   recovery_data = parity of original_data (xor sum)  */
void LEO_FUNC(EncodeM1)(uint64_t buffer_bytes,
                        unsigned original_count,
                        const void * const * const original_data,
                        void * recovery_data) {
  memcpy(recovery_data, original_data[0], buffer_bytes);

  XORSummer summer;
  XORSummer_Initialize(&summer, recovery_data);

  for (unsigned i = 1; i < original_count; ++i)
    XORSummer_Add(&summer, original_data[i], buffer_bytes);

  XORSummer_Finalize(&summer, buffer_bytes);
}

void LEO_FUNC(DecodeM1)(uint64_t buffer_bytes,
                        unsigned original_count,
                        const void * const * original_data,
                        const void * recovery_data,
                        void * work_data) {
  memcpy(work_data, recovery_data, buffer_bytes);

  XORSummer summer;
  XORSummer_Initialize(&summer, work_data);

  for (unsigned i = 0; i < original_count; ++i)
    if (original_data[i])
      XORSummer_Add(&summer, original_data[i], buffer_bytes);

  XORSummer_Finalize(&summer, buffer_bytes);
}

/*  ----------------------------------------------------------------------
    mul_mem  */
static void LEO_FUNC(mul_mem)(void * restrict x,
                              const void * restrict y,
                              ffe_t log_m,
                              uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    LEO_M256 * restrict x32 = (LEO_M256 *) (x);
    const LEO_M256 * restrict y32 = (const LEO_M256 *) (y);
    do {
#define LEO_MUL_256(x_ptr, y_ptr)                         \
  {                                                       \
    LEO_M256 data = _mm256_loadu_si256(y_ptr);            \
    LEO_M256 lo = _mm256_and_si256(data, clr_mask);       \
    lo = _mm256_shuffle_epi8(table_lo_y, lo);             \
    LEO_M256 hi = _mm256_srli_epi64(data, 4);             \
    hi = _mm256_and_si256(hi, clr_mask);                  \
    hi = _mm256_shuffle_epi8(table_hi_y, hi);             \
    _mm256_storeu_si256(x_ptr, _mm256_xor_si256(lo, hi)); \
  }
      LEO_MUL_256(x32 + 1, y32 + 1);
      LEO_MUL_256(x32, y32);
      y32 += 2, x32 += 2;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#elif defined(LEO_TRY_SSSE3)
  {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);
    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);
    LEO_M128 * restrict x16 = (LEO_M128 *) (x);
    const LEO_M128 * restrict y16 = (const LEO_M128 *) (y);
    do {
#define LEO_MUL_128(x_ptr, y_ptr)                   \
  {                                                 \
    LEO_M128 data = _mm_loadu_si128(y_ptr);         \
    LEO_M128 lo = _mm_and_si128(data, clr_mask);    \
    lo = _mm_shuffle_epi8(table_lo_y, lo);          \
    LEO_M128 hi = _mm_srli_epi64(data, 4);          \
    hi = _mm_and_si128(hi, clr_mask);               \
    hi = _mm_shuffle_epi8(table_hi_y, hi);          \
    _mm_storeu_si128(x_ptr, _mm_xor_si128(lo, hi)); \
  }
      LEO_MUL_128(x16 + 3, y16 + 3);
      LEO_MUL_128(x16 + 2, y16 + 2);
      LEO_MUL_128(x16 + 1, y16 + 1);
      LEO_MUL_128(x16, y16);
      x16 += 4, y16 += 4;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#else
  RefMul(x, y, log_m, bytes);
#endif
}

/*  ----------------------------------------------------------------------
    IFFT_DIT2  */
static void LEO_FUNC(IFFT_DIT2)(void * restrict x,
                                void * restrict y,
                                ffe_t log_m,
                                uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    LEO_M256 * restrict x32 = (LEO_M256 *) (x);
    LEO_M256 * restrict y32 = (LEO_M256 *) (y);
    do {
#define LEO_IFFTB_256(x_ptr, y_ptr)                         \
  {                                                         \
    LEO_M256 x_data = _mm256_loadu_si256(x_ptr);            \
    LEO_M256 y_data = _mm256_loadu_si256(y_ptr);            \
    y_data = _mm256_xor_si256(y_data, x_data);              \
    _mm256_storeu_si256(y_ptr, y_data);                     \
    LEO_MULADD_256(x_data, y_data, table_lo_y, table_hi_y); \
    _mm256_storeu_si256(x_ptr, x_data);                     \
  }
      LEO_IFFTB_256(x32 + 1, y32 + 1);
      LEO_IFFTB_256(x32, y32);
      y32 += 2, x32 += 2;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#elif defined(LEO_TRY_SSSE3)
  {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);
    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);
    LEO_M128 * restrict x16 = (LEO_M128 *) (x);
    LEO_M128 * restrict y16 = (LEO_M128 *) (y);
    do {
#define LEO_IFFTB_128(x_ptr, y_ptr)                         \
  {                                                         \
    LEO_M128 x_data = _mm_loadu_si128(x_ptr);               \
    LEO_M128 y_data = _mm_loadu_si128(y_ptr);               \
    y_data = _mm_xor_si128(y_data, x_data);                 \
    _mm_storeu_si128(y_ptr, y_data);                        \
    LEO_MULADD_128(x_data, y_data, table_lo_y, table_hi_y); \
    _mm_storeu_si128(x_ptr, x_data);                        \
  }
      LEO_IFFTB_128(x16 + 3, y16 + 3);
      LEO_IFFTB_128(x16 + 2, y16 + 2);
      LEO_IFFTB_128(x16 + 1, y16 + 1);
      LEO_IFFTB_128(x16, y16);
      x16 += 4, y16 += 4;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#else
  LEO_FUNC(xor_mem)(y, x, bytes);
  RefMulAdd(x, y, log_m, bytes);
#endif
}

/*  ----------------------------------------------------------------------
    IFFT_DIT4  */
static void LEO_FUNC(IFFT_DIT4)(uint64_t bytes,
                                void ** work,
                                unsigned dist,
                                const ffe_t log_m01,
                                const ffe_t log_m23,
                                const ffe_t log_m02) {
#if defined(LEO_INTERLEAVE_BUTTERFLY4_OPT) && defined(LEO_TRY_AVX2)
  {
    const LEO_M256 t01_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m01].Value[0]);
    const LEO_M256 t01_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m01].Value[1]);
    const LEO_M256 t23_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m23].Value[0]);
    const LEO_M256 t23_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m23].Value[1]);
    const LEO_M256 t02_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m02].Value[0]);
    const LEO_M256 t02_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m02].Value[1]);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    LEO_M256 * restrict work0 = (LEO_M256 *) (work[0]);
    LEO_M256 * restrict work1 = (LEO_M256 *) (work[dist]);
    LEO_M256 * restrict work2 = (LEO_M256 *) (work[dist * 2]);
    LEO_M256 * restrict work3 = (LEO_M256 *) (work[dist * 3]);
    do {
      LEO_M256 work0_reg = _mm256_loadu_si256(work0);
      LEO_M256 work1_reg = _mm256_loadu_si256(work1);
      work1_reg = _mm256_xor_si256(work0_reg, work1_reg);
      if (log_m01 != kModulus)
        LEO_MULADD_256(work0_reg, work1_reg, t01_lo, t01_hi);
      LEO_M256 work2_reg = _mm256_loadu_si256(work2);
      LEO_M256 work3_reg = _mm256_loadu_si256(work3);
      work3_reg = _mm256_xor_si256(work2_reg, work3_reg);
      if (log_m23 != kModulus)
        LEO_MULADD_256(work2_reg, work3_reg, t23_lo, t23_hi);
      work2_reg = _mm256_xor_si256(work0_reg, work2_reg);
      work3_reg = _mm256_xor_si256(work1_reg, work3_reg);
      if (log_m02 != kModulus) {
        LEO_MULADD_256(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_256(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      _mm256_storeu_si256(work0, work0_reg);
      _mm256_storeu_si256(work1, work1_reg);
      _mm256_storeu_si256(work2, work2_reg);
      _mm256_storeu_si256(work3, work3_reg);
      work0++, work1++, work2++, work3++;
      bytes -= 32;
    } while (bytes > 0);
    return;
  }
#elif defined(LEO_INTERLEAVE_BUTTERFLY4_OPT) && defined(LEO_TRY_SSSE3)
  {
    const LEO_M128 t01_lo = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[0]);
    const LEO_M128 t01_hi = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[1]);
    const LEO_M128 t23_lo = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[0]);
    const LEO_M128 t23_hi = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[1]);
    const LEO_M128 t02_lo = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[0]);
    const LEO_M128 t02_hi = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[1]);
    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);
    LEO_M128 * restrict work0 = (LEO_M128 *) (work[0]);
    LEO_M128 * restrict work1 = (LEO_M128 *) (work[dist]);
    LEO_M128 * restrict work2 = (LEO_M128 *) (work[dist * 2]);
    LEO_M128 * restrict work3 = (LEO_M128 *) (work[dist * 3]);
    do {
      LEO_M128 work0_reg = _mm_loadu_si128(work0);
      LEO_M128 work1_reg = _mm_loadu_si128(work1);
      work1_reg = _mm_xor_si128(work0_reg, work1_reg);
      if (log_m01 != kModulus)
        LEO_MULADD_128(work0_reg, work1_reg, t01_lo, t01_hi);
      LEO_M128 work2_reg = _mm_loadu_si128(work2);
      LEO_M128 work3_reg = _mm_loadu_si128(work3);
      work3_reg = _mm_xor_si128(work2_reg, work3_reg);
      if (log_m23 != kModulus)
        LEO_MULADD_128(work2_reg, work3_reg, t23_lo, t23_hi);
      work2_reg = _mm_xor_si128(work0_reg, work2_reg);
      work3_reg = _mm_xor_si128(work1_reg, work3_reg);
      if (log_m02 != kModulus) {
        LEO_MULADD_128(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_128(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      _mm_storeu_si128(work0, work0_reg);
      _mm_storeu_si128(work1, work1_reg);
      _mm_storeu_si128(work2, work2_reg);
      _mm_storeu_si128(work3, work3_reg);
      work0++, work1++, work2++, work3++;
      bytes -= 16;
    } while (bytes > 0);
    return;
  }
#else
  if (log_m01 == kModulus)
    LEO_FUNC(xor_mem)(work[dist], work[0], bytes);
  else
    LEO_FUNC(IFFT_DIT2)(work[0], work[dist], log_m01, bytes);

  if (log_m23 == kModulus)
    LEO_FUNC(xor_mem)(work[dist * 3], work[dist * 2], bytes);
  else
    LEO_FUNC(IFFT_DIT2)(work[dist * 2], work[dist * 3], log_m23, bytes);

  if (log_m02 == kModulus) {
    LEO_FUNC(xor_mem)(work[dist * 2], work[0], bytes);
    LEO_FUNC(xor_mem)(work[dist * 3], work[dist], bytes);
  } else {
    LEO_FUNC(IFFT_DIT2)(work[0], work[dist * 2], log_m02, bytes);
    LEO_FUNC(IFFT_DIT2)(work[dist], work[dist * 3], log_m02, bytes);
  }
#endif
}

/*  ----------------------------------------------------------------------
    IFFT_DIT2_xor  */
static void LEO_FUNC(IFFT_DIT2_xor)(void * restrict x_in,
                                    void * restrict y_in,
                                    void * restrict x_out,
                                    void * restrict y_out,
                                    const ffe_t log_m,
                                    uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    const LEO_M256 * restrict x32_in = (const LEO_M256 *) (x_in);
    const LEO_M256 * restrict y32_in = (const LEO_M256 *) (y_in);
    LEO_M256 * restrict x32_out = (LEO_M256 *) (x_out);
    LEO_M256 * restrict y32_out = (LEO_M256 *) (y_out);
    do {
#define LEO_IFFTB_256_XOR(x_ptr_in, y_ptr_in, x_ptr_out, y_ptr_out) \
  {                                                                 \
    LEO_M256 x_data_out = _mm256_loadu_si256(x_ptr_out);            \
    LEO_M256 y_data_out = _mm256_loadu_si256(y_ptr_out);            \
    LEO_M256 x_data_in = _mm256_loadu_si256(x_ptr_in);              \
    LEO_M256 y_data_in = _mm256_loadu_si256(y_ptr_in);              \
    y_data_in = _mm256_xor_si256(y_data_in, x_data_in);             \
    y_data_out = _mm256_xor_si256(y_data_out, y_data_in);           \
    _mm256_storeu_si256(y_ptr_out, y_data_out);                     \
    LEO_MULADD_256(x_data_in, y_data_in, table_lo_y, table_hi_y);   \
    x_data_out = _mm256_xor_si256(x_data_out, x_data_in);           \
    _mm256_storeu_si256(x_ptr_out, x_data_out);                     \
  }
      LEO_IFFTB_256_XOR(x32_in + 1, y32_in + 1, x32_out + 1, y32_out + 1);
      LEO_IFFTB_256_XOR(x32_in, y32_in, x32_out, y32_out);
      y32_in += 2, x32_in += 2, y32_out += 2, x32_out += 2;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#elif defined(LEO_TRY_SSSE3)
  {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);
    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);
    const LEO_M128 * restrict x16_in = (const LEO_M128 *) (x_in);
    const LEO_M128 * restrict y16_in = (const LEO_M128 *) (y_in);
    LEO_M128 * restrict x16_out = (LEO_M128 *) (x_out);
    LEO_M128 * restrict y16_out = (LEO_M128 *) (y_out);
    do {
#define LEO_IFFTB_128_XOR(x_ptr_in, y_ptr_in, x_ptr_out, y_ptr_out) \
  {                                                                 \
    LEO_M128 x_data_out = _mm_loadu_si128(x_ptr_out);               \
    LEO_M128 y_data_out = _mm_loadu_si128(y_ptr_out);               \
    LEO_M128 x_data_in = _mm_loadu_si128(x_ptr_in);                 \
    LEO_M128 y_data_in = _mm_loadu_si128(y_ptr_in);                 \
    y_data_in = _mm_xor_si128(y_data_in, x_data_in);                \
    y_data_out = _mm_xor_si128(y_data_out, y_data_in);              \
    _mm_storeu_si128(y_ptr_out, y_data_out);                        \
    LEO_MULADD_128(x_data_in, y_data_in, table_lo_y, table_hi_y);   \
    x_data_out = _mm_xor_si128(x_data_out, x_data_in);              \
    _mm_storeu_si128(x_ptr_out, x_data_out);                        \
  }
      LEO_IFFTB_128_XOR(x16_in + 3, y16_in + 3, x16_out + 3, y16_out + 3);
      LEO_IFFTB_128_XOR(x16_in + 2, y16_in + 2, x16_out + 2, y16_out + 2);
      LEO_IFFTB_128_XOR(x16_in + 1, y16_in + 1, x16_out + 1, y16_out + 1);
      LEO_IFFTB_128_XOR(x16_in, y16_in, x16_out, y16_out);
      y16_in += 4, x16_in += 4, y16_out += 4, x16_out += 4;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#else
  LEO_FUNC(xor_mem)(y_in, x_in, bytes);
  RefMulAdd(x_in, y_in, log_m, bytes);
  LEO_FUNC(xor_mem)(y_out, y_in, bytes);
  LEO_FUNC(xor_mem)(x_out, x_in, bytes);
#endif
}

/*  ----------------------------------------------------------------------
    IFFT_DIT4_xor  */
static void LEO_FUNC(IFFT_DIT4_xor)(uint64_t bytes,
                                    void ** work_in,
                                    void ** xor_out,
                                    unsigned dist,
                                    const ffe_t log_m01,
                                    const ffe_t log_m23,
                                    const ffe_t log_m02) {
#if defined(LEO_INTERLEAVE_BUTTERFLY4_OPT) && defined(LEO_TRY_AVX2)
  {
    const LEO_M256 t01_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m01].Value[0]);
    const LEO_M256 t01_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m01].Value[1]);
    const LEO_M256 t23_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m23].Value[0]);
    const LEO_M256 t23_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m23].Value[1]);
    const LEO_M256 t02_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m02].Value[0]);
    const LEO_M256 t02_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m02].Value[1]);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    const LEO_M256 * restrict work0 = (const LEO_M256 *) (work_in[0]);
    const LEO_M256 * restrict work1 = (const LEO_M256 *) (work_in[dist]);
    const LEO_M256 * restrict work2 = (const LEO_M256 *) (work_in[dist * 2]);
    const LEO_M256 * restrict work3 = (const LEO_M256 *) (work_in[dist * 3]);
    LEO_M256 * restrict xor0 = (LEO_M256 *) (xor_out[0]);
    LEO_M256 * restrict xor1 = (LEO_M256 *) (xor_out[dist]);
    LEO_M256 * restrict xor2 = (LEO_M256 *) (xor_out[dist * 2]);
    LEO_M256 * restrict xor3 = (LEO_M256 *) (xor_out[dist * 3]);
    do {
      LEO_M256 work0_reg = _mm256_loadu_si256(work0);
      LEO_M256 work1_reg = _mm256_loadu_si256(work1);
      work0++, work1++;
      work1_reg = _mm256_xor_si256(work0_reg, work1_reg);
      if (log_m01 != kModulus)
        LEO_MULADD_256(work0_reg, work1_reg, t01_lo, t01_hi);
      LEO_M256 work2_reg = _mm256_loadu_si256(work2);
      LEO_M256 work3_reg = _mm256_loadu_si256(work3);
      work2++, work3++;
      work3_reg = _mm256_xor_si256(work2_reg, work3_reg);
      if (log_m23 != kModulus)
        LEO_MULADD_256(work2_reg, work3_reg, t23_lo, t23_hi);
      work2_reg = _mm256_xor_si256(work0_reg, work2_reg);
      work3_reg = _mm256_xor_si256(work1_reg, work3_reg);
      if (log_m02 != kModulus) {
        LEO_MULADD_256(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_256(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      work0_reg = _mm256_xor_si256(work0_reg, _mm256_loadu_si256(xor0));
      work1_reg = _mm256_xor_si256(work1_reg, _mm256_loadu_si256(xor1));
      work2_reg = _mm256_xor_si256(work2_reg, _mm256_loadu_si256(xor2));
      work3_reg = _mm256_xor_si256(work3_reg, _mm256_loadu_si256(xor3));
      _mm256_storeu_si256(xor0, work0_reg);
      _mm256_storeu_si256(xor1, work1_reg);
      _mm256_storeu_si256(xor2, work2_reg);
      _mm256_storeu_si256(xor3, work3_reg);
      xor0++, xor1++, xor2++, xor3++;
      bytes -= 32;
    } while (bytes > 0);
    return;
  }
#elif defined(LEO_INTERLEAVE_BUTTERFLY4_OPT) && defined(LEO_TRY_SSSE3)
  {
    const LEO_M128 t01_lo = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[0]);
    const LEO_M128 t01_hi = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[1]);
    const LEO_M128 t23_lo = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[0]);
    const LEO_M128 t23_hi = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[1]);
    const LEO_M128 t02_lo = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[0]);
    const LEO_M128 t02_hi = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[1]);
    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);
    const LEO_M128 * restrict work0 = (const LEO_M128 *) (work_in[0]);
    const LEO_M128 * restrict work1 = (const LEO_M128 *) (work_in[dist]);
    const LEO_M128 * restrict work2 = (const LEO_M128 *) (work_in[dist * 2]);
    const LEO_M128 * restrict work3 = (const LEO_M128 *) (work_in[dist * 3]);
    LEO_M128 * restrict xor0 = (LEO_M128 *) (xor_out[0]);
    LEO_M128 * restrict xor1 = (LEO_M128 *) (xor_out[dist]);
    LEO_M128 * restrict xor2 = (LEO_M128 *) (xor_out[dist * 2]);
    LEO_M128 * restrict xor3 = (LEO_M128 *) (xor_out[dist * 3]);
    do {
      LEO_M128 work0_reg = _mm_loadu_si128(work0);
      LEO_M128 work1_reg = _mm_loadu_si128(work1);
      work0++, work1++;
      work1_reg = _mm_xor_si128(work0_reg, work1_reg);
      if (log_m01 != kModulus)
        LEO_MULADD_128(work0_reg, work1_reg, t01_lo, t01_hi);
      LEO_M128 work2_reg = _mm_loadu_si128(work2);
      LEO_M128 work3_reg = _mm_loadu_si128(work3);
      work2++, work3++;
      work3_reg = _mm_xor_si128(work2_reg, work3_reg);
      if (log_m23 != kModulus)
        LEO_MULADD_128(work2_reg, work3_reg, t23_lo, t23_hi);
      work2_reg = _mm_xor_si128(work0_reg, work2_reg);
      work3_reg = _mm_xor_si128(work1_reg, work3_reg);
      if (log_m02 != kModulus) {
        LEO_MULADD_128(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_128(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      work0_reg = _mm_xor_si128(work0_reg, _mm_loadu_si128(xor0));
      work1_reg = _mm_xor_si128(work1_reg, _mm_loadu_si128(xor1));
      work2_reg = _mm_xor_si128(work2_reg, _mm_loadu_si128(xor2));
      work3_reg = _mm_xor_si128(work3_reg, _mm_loadu_si128(xor3));
      _mm_storeu_si128(xor0, work0_reg);
      _mm_storeu_si128(xor1, work1_reg);
      _mm_storeu_si128(xor2, work2_reg);
      _mm_storeu_si128(xor3, work3_reg);
      xor0++, xor1++, xor2++, xor3++;
      bytes -= 16;
    } while (bytes > 0);
    return;
  }
#else
  if (log_m01 == kModulus)
    LEO_FUNC(xor_mem)(work_in[dist], work_in[0], bytes);
  else
    LEO_FUNC(IFFT_DIT2)(work_in[0], work_in[dist], log_m01, bytes);

  if (log_m23 == kModulus)
    LEO_FUNC(xor_mem)(work_in[dist * 3], work_in[dist * 2], bytes);
  else
    LEO_FUNC(IFFT_DIT2)(work_in[dist * 2], work_in[dist * 3], log_m23, bytes);

  if (log_m02 == kModulus) {
    LEO_FUNC(xor_mem)(work_in[dist * 2], work_in[0], bytes);
    LEO_FUNC(xor_mem)(work_in[dist * 3], work_in[dist], bytes);
  } else {
    LEO_FUNC(IFFT_DIT2)(work_in[0], work_in[dist * 2], log_m02, bytes);
    LEO_FUNC(IFFT_DIT2)(work_in[dist], work_in[dist * 3], log_m02, bytes);
  }

  LEO_FUNC(xor_mem)(xor_out[0], work_in[0], bytes);
  LEO_FUNC(xor_mem)(xor_out[dist], work_in[dist], bytes);
  LEO_FUNC(xor_mem)(xor_out[dist * 2], work_in[dist * 2], bytes);
  LEO_FUNC(xor_mem)(xor_out[dist * 3], work_in[dist * 3], bytes);
#endif
}

/*  ----------------------------------------------------------------------
    IFFT_DIT_Encoder / IFFT_DIT_Decoder  */
static void LEO_FUNC(IFFT_DIT_Encoder)(const uint64_t bytes,
                                       const void * const * data,
                                       const unsigned m_truncated,
                                       void ** work,
                                       void ** xor_result,
                                       const unsigned m,
                                       const ffe_t * skewLUT) {
  for (unsigned i = 0; i < m_truncated; ++i)
    memcpy(work[i], data[i], bytes);
  for (unsigned i = m_truncated; i < m; ++i)
    memset(work[i], 0, bytes);

  unsigned dist = 1, dist4 = 4;
  for (; dist4 <= m; dist = dist4, dist4 <<= 2) {
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      const unsigned i_end = r + dist;
      const ffe_t log_m01 = skewLUT[i_end];
      const ffe_t log_m02 = skewLUT[i_end + dist];
      const ffe_t log_m23 = skewLUT[i_end + dist * 2];

      if (dist4 == m && xor_result) {
        for (unsigned i = r; i < i_end; ++i) {
          LEO_FUNC(IFFT_DIT4_xor)(bytes, work + i, xor_result + i, dist,
                                  log_m01, log_m23, log_m02);
        }
      } else {
        for (unsigned i = r; i < i_end; ++i) {
          LEO_FUNC(IFFT_DIT4)(
            bytes, work + i, dist, log_m01, log_m23, log_m02);
        }
      }
    }
  }

  if (dist < m) {
    const ffe_t log_m = skewLUT[dist];

    if (xor_result) {
      if (log_m == kModulus) {
        for (unsigned i = 0; i < dist; ++i)
          LEO_FUNC(xor_mem_2to1)(
            xor_result[i], work[i], work[i + dist], bytes);
      } else {
        for (unsigned i = 0; i < dist; ++i) {
          LEO_FUNC(IFFT_DIT2_xor)(work[i], work[i + dist], xor_result[i],
                                  xor_result[i + dist], log_m, bytes);
        }
      }
    } else {
      if (log_m == kModulus)
        LEO_FUNC(VectorXOR)(bytes, dist, work + dist, work);
      else {
        for (unsigned i = 0; i < dist; ++i) {
          LEO_FUNC(IFFT_DIT2)(work[i], work[i + dist], log_m, bytes);
        }
      }
    }
  }
}

static void LEO_FUNC(IFFT_DIT_Decoder)(const uint64_t bytes,
                                       const unsigned m_truncated,
                                       void ** work,
                                       const unsigned m) {
  unsigned dist = 1, dist4 = 4;
  for (; dist4 <= m; dist = dist4, dist4 <<= 2) {
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      const unsigned i_end = r + dist;
      const ffe_t log_m01 = FFTSkew[i_end - 1];
      const ffe_t log_m02 = FFTSkew[i_end + dist - 1];
      const ffe_t log_m23 = FFTSkew[i_end + dist * 2 - 1];

      for (unsigned i = r; i < i_end; ++i) {
        LEO_FUNC(IFFT_DIT4)(bytes, work + i, dist, log_m01, log_m23, log_m02);
      }
    }
  }

  if (dist < m) {
    const ffe_t log_m = FFTSkew[dist - 1];

    if (log_m == kModulus)
      LEO_FUNC(VectorXOR)(bytes, dist, work + dist, work);
    else {
      for (unsigned i = 0; i < dist; ++i) {
        LEO_FUNC(IFFT_DIT2)(work[i], work[i + dist], log_m, bytes);
      }
    }
  }
}

/*  ----------------------------------------------------------------------
    FFT_DIT2  */
static void LEO_FUNC(FFT_DIT2)(void * restrict x,
                               void * restrict y,
                               ffe_t log_m,
                               uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    LEO_M256 * restrict x32 = (LEO_M256 *) (x);
    LEO_M256 * restrict y32 = (LEO_M256 *) (y);
    do {
#define LEO_FFTB_256(x_ptr, y_ptr)                          \
  {                                                         \
    LEO_M256 y_data = _mm256_loadu_si256(y_ptr);            \
    LEO_M256 x_data = _mm256_loadu_si256(x_ptr);            \
    LEO_MULADD_256(x_data, y_data, table_lo_y, table_hi_y); \
    y_data = _mm256_xor_si256(y_data, x_data);              \
    _mm256_storeu_si256(x_ptr, x_data);                     \
    _mm256_storeu_si256(y_ptr, y_data);                     \
  }
      LEO_FFTB_256(x32 + 1, y32 + 1);
      LEO_FFTB_256(x32, y32);
      y32 += 2, x32 += 2;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#elif defined(LEO_TRY_SSSE3)
  {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);
    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);
    LEO_M128 * restrict x16 = (LEO_M128 *) (x);
    LEO_M128 * restrict y16 = (LEO_M128 *) (y);
    do {
#define LEO_FFTB_128(x_ptr, y_ptr)                          \
  {                                                         \
    LEO_M128 y_data = _mm_loadu_si128(y_ptr);               \
    LEO_M128 x_data = _mm_loadu_si128(x_ptr);               \
    LEO_MULADD_128(x_data, y_data, table_lo_y, table_hi_y); \
    y_data = _mm_xor_si128(y_data, x_data);                 \
    _mm_storeu_si128(x_ptr, x_data);                        \
    _mm_storeu_si128(y_ptr, y_data);                        \
  }
      LEO_FFTB_128(x16 + 3, y16 + 3);
      LEO_FFTB_128(x16 + 2, y16 + 2);
      LEO_FFTB_128(x16 + 1, y16 + 1);
      LEO_FFTB_128(x16, y16);
      x16 += 4, y16 += 4;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#else
  RefMulAdd(x, y, log_m, bytes);
  LEO_FUNC(xor_mem)(y, x, bytes);
#endif
}

/*  ----------------------------------------------------------------------
    FFT_DIT4  */
static void LEO_FUNC(FFT_DIT4)(uint64_t bytes,
                               void ** work,
                               unsigned dist,
                               const ffe_t log_m01,
                               const ffe_t log_m23,
                               const ffe_t log_m02) {
#if defined(LEO_INTERLEAVE_BUTTERFLY4_OPT) && defined(LEO_TRY_AVX2)
  {
    const LEO_M256 t01_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m01].Value[0]);
    const LEO_M256 t01_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m01].Value[1]);
    const LEO_M256 t23_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m23].Value[0]);
    const LEO_M256 t23_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m23].Value[1]);
    const LEO_M256 t02_lo =
        _mm256_loadu_si256(&Multiply256LUT[log_m02].Value[0]);
    const LEO_M256 t02_hi =
        _mm256_loadu_si256(&Multiply256LUT[log_m02].Value[1]);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    LEO_M256 * restrict work0 = (LEO_M256 *) (work[0]);
    LEO_M256 * restrict work1 = (LEO_M256 *) (work[dist]);
    LEO_M256 * restrict work2 = (LEO_M256 *) (work[dist * 2]);
    LEO_M256 * restrict work3 = (LEO_M256 *) (work[dist * 3]);
    do {
      LEO_M256 work0_reg = _mm256_loadu_si256(work0);
      LEO_M256 work2_reg = _mm256_loadu_si256(work2);
      LEO_M256 work1_reg = _mm256_loadu_si256(work1);
      LEO_M256 work3_reg = _mm256_loadu_si256(work3);
      if (log_m02 != kModulus) {
        LEO_MULADD_256(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_256(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      work2_reg = _mm256_xor_si256(work0_reg, work2_reg);
      work3_reg = _mm256_xor_si256(work1_reg, work3_reg);
      if (log_m01 != kModulus)
        LEO_MULADD_256(work0_reg, work1_reg, t01_lo, t01_hi);
      work1_reg = _mm256_xor_si256(work0_reg, work1_reg);
      _mm256_storeu_si256(work0, work0_reg);
      _mm256_storeu_si256(work1, work1_reg);
      work0++, work1++;
      if (log_m23 != kModulus)
        LEO_MULADD_256(work2_reg, work3_reg, t23_lo, t23_hi);
      work3_reg = _mm256_xor_si256(work2_reg, work3_reg);
      _mm256_storeu_si256(work2, work2_reg);
      _mm256_storeu_si256(work3, work3_reg);
      work2++, work3++;
      bytes -= 32;
    } while (bytes > 0);
    return;
  }
#elif defined(LEO_INTERLEAVE_BUTTERFLY4_OPT) && defined(LEO_TRY_SSSE3)
  {
    const LEO_M128 t01_lo = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[0]);
    const LEO_M128 t01_hi = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[1]);
    const LEO_M128 t23_lo = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[0]);
    const LEO_M128 t23_hi = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[1]);
    const LEO_M128 t02_lo = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[0]);
    const LEO_M128 t02_hi = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[1]);
    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);
    LEO_M128 * restrict work0 = (LEO_M128 *) (work[0]);
    LEO_M128 * restrict work1 = (LEO_M128 *) (work[dist]);
    LEO_M128 * restrict work2 = (LEO_M128 *) (work[dist * 2]);
    LEO_M128 * restrict work3 = (LEO_M128 *) (work[dist * 3]);
    do {
      LEO_M128 work0_reg = _mm_loadu_si128(work0);
      LEO_M128 work2_reg = _mm_loadu_si128(work2);
      LEO_M128 work1_reg = _mm_loadu_si128(work1);
      LEO_M128 work3_reg = _mm_loadu_si128(work3);
      if (log_m02 != kModulus) {
        LEO_MULADD_128(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_128(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      work2_reg = _mm_xor_si128(work0_reg, work2_reg);
      work3_reg = _mm_xor_si128(work1_reg, work3_reg);
      if (log_m01 != kModulus)
        LEO_MULADD_128(work0_reg, work1_reg, t01_lo, t01_hi);
      work1_reg = _mm_xor_si128(work0_reg, work1_reg);
      _mm_storeu_si128(work0, work0_reg);
      _mm_storeu_si128(work1, work1_reg);
      work0++, work1++;
      if (log_m23 != kModulus)
        LEO_MULADD_128(work2_reg, work3_reg, t23_lo, t23_hi);
      work3_reg = _mm_xor_si128(work2_reg, work3_reg);
      _mm_storeu_si128(work2, work2_reg);
      _mm_storeu_si128(work3, work3_reg);
      work2++, work3++;
      bytes -= 16;
    } while (bytes > 0);
    return;
  }
#else
  if (log_m02 == kModulus) {
    LEO_FUNC(xor_mem)(work[dist * 2], work[0], bytes);
    LEO_FUNC(xor_mem)(work[dist * 3], work[dist], bytes);
  } else {
    LEO_FUNC(FFT_DIT2)(work[0], work[dist * 2], log_m02, bytes);
    LEO_FUNC(FFT_DIT2)(work[dist], work[dist * 3], log_m02, bytes);
  }
  if (log_m01 == kModulus)
    LEO_FUNC(xor_mem)(work[dist], work[0], bytes);
  else
    LEO_FUNC(FFT_DIT2)(work[0], work[dist], log_m01, bytes);
  if (log_m23 == kModulus)
    LEO_FUNC(xor_mem)(work[dist * 3], work[dist * 2], bytes);
  else
    LEO_FUNC(FFT_DIT2)(work[dist * 2], work[dist * 3], log_m23, bytes);
#endif
}

/*  ----------------------------------------------------------------------
    FFT_DIT  */
static void LEO_FUNC(FFT_DIT)(const uint64_t bytes,
                              void ** work,
                              const unsigned m_truncated,
                              const unsigned m) {
  unsigned dist4 = m, dist = m >> 2;
  for (; dist != 0; dist4 = dist, dist >>= 2) {
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      const unsigned i_end = r + dist;
      const ffe_t log_m01 = FFTSkew[i_end - 1];
      const ffe_t log_m02 = FFTSkew[i_end + dist - 1];
      const ffe_t log_m23 = FFTSkew[i_end + dist * 2 - 1];

      for (unsigned i = r; i < i_end; ++i) {
        LEO_FUNC(FFT_DIT4)(bytes, work + i, dist, log_m01, log_m23, log_m02);
      }
    }
  }

  if (dist4 == 2) {
    for (unsigned r = 0; r < m_truncated; r += 2) {
      const ffe_t log_m = FFTSkew[r];
      if (log_m == kModulus)
        LEO_FUNC(xor_mem)(work[r + 1], work[r], bytes);
      else {
        LEO_FUNC(FFT_DIT2)(work[r], work[r + 1], log_m, bytes);
      }
    }
  }
}

/*  ----------------------------------------------------------------------
    Reed-Solomon Encode (variant)  */
void LEO_FUNC(ReedSolomonEncode)(uint64_t buffer_bytes,
                                 unsigned original_count,
                                 unsigned recovery_count,
                                 unsigned m,
                                 const void * const * data,
                                 void ** work) {
  const ffe_t * skewLUT = FFTSkew + m - 1;

  LEO_FUNC(IFFT_DIT_Encoder)(buffer_bytes, data,
                             original_count < m ? original_count : m,
                             work, NULL, m, skewLUT);

  const unsigned last_count = original_count % m;
  if (m >= original_count)
    goto skip_body;

  for (unsigned i = m; i + m <= original_count; i += m) {
    data += m;
    skewLUT += m;
    LEO_FUNC(IFFT_DIT_Encoder)(buffer_bytes, data, m, work + m, work,
                               m, skewLUT);
  }

  if (last_count != 0) {
    data += m;
    skewLUT += m;
    LEO_FUNC(IFFT_DIT_Encoder)(buffer_bytes, data, last_count, work + m, work,
                               m, skewLUT);
  }

skip_body:
  LEO_FUNC(FFT_DIT)(buffer_bytes, work, recovery_count, m);
}

/*  ----------------------------------------------------------------------
    ErrorBitfield  */
#ifdef LEO_ERROR_BITFIELD_OPT

#define kWords (kOrder / 64)

typedef struct {
  uint64_t Words[7][kWords];
} ErrorBitfield;

static LEO_FORCE_INLINE void
ErrorBitfield_Set(ErrorBitfield * bf, unsigned i) {
  bf->Words[0][i / 64] |= (uint64_t) 1 << (i % 64);
}

static LEO_FORCE_INLINE bool ErrorBitfield_IsNeeded(const ErrorBitfield * bf,
                                                    unsigned mip_level,
                                                    unsigned bit) {
  if (mip_level >= 8) return true;
  return 0 !=
         (bf->Words[mip_level - 1][bit / 64] & ((uint64_t) 1 << (bit % 64)));
}

static const uint64_t kHiMasks[5] = {
    0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
    0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL,
};

static void ErrorBitfield_Prepare(ErrorBitfield * bf) {
  for (unsigned i = 0; i < kWords; ++i) {
    uint64_t w_i = bf->Words[0][i];
    const uint64_t hi2lo0 = w_i | ((w_i & kHiMasks[0]) >> 1);
    const uint64_t lo2hi0 = ((w_i & (kHiMasks[0] >> 1)) << 1);
    bf->Words[0][i] = w_i = hi2lo0 | lo2hi0;

    for (unsigned j = 1, bits = 2; j < 5; ++j, bits <<= 1) {
      const uint64_t hi2lo_j = w_i | ((w_i & kHiMasks[j]) >> bits);
      const uint64_t lo2hi_j = ((w_i & (kHiMasks[j] >> bits)) << bits);
      bf->Words[j][i] = w_i = hi2lo_j | lo2hi_j;
    }
  }

  for (unsigned i = 0; i < kWords; ++i) {
    uint64_t w = bf->Words[4][i];
    w |= w >> 32;
    w |= w << 32;
    bf->Words[5][i] = w;
  }

  for (unsigned i = 0; i < kWords; i += 2)
    bf->Words[6][i] = bf->Words[6][i + 1] =
        bf->Words[5][i] | bf->Words[5][i + 1];
}

static void LEO_FUNC(FFT_DIT_ErrorBits)(const uint64_t bytes,
                                        void ** work,
                                        const unsigned n_truncated,
                                        const unsigned n,
                                        const ErrorBitfield * error_bits) {
  unsigned mip_level = LastNonzeroBit32(n);

  unsigned dist4 = n, dist = n >> 2;
  for (; dist != 0; dist4 = dist, dist >>= 2, mip_level -= 2) {
    for (unsigned r = 0; r < n_truncated; r += dist4) {
      if (!ErrorBitfield_IsNeeded(error_bits, mip_level, r))
        continue;

      const ffe_t log_m01 = FFTSkew[r + dist - 1];
      const ffe_t log_m23 = FFTSkew[r + dist * 3 - 1];
      const ffe_t log_m02 = FFTSkew[r + dist * 2 - 1];

      for (unsigned i = r; i < r + dist; ++i) {
        LEO_FUNC(FFT_DIT4)(bytes, work + i, dist, log_m01, log_m23, log_m02);
      }
    }
  }

  if (dist4 == 2) {
    for (unsigned r = 0; r < n_truncated; r += 2) {
      if (!ErrorBitfield_IsNeeded(error_bits, mip_level, r))
        continue;

      const ffe_t log_m = FFTSkew[r];
      if (log_m == kModulus)
        LEO_FUNC(xor_mem)(work[r + 1], work[r], bytes);
      else {
        LEO_FUNC(FFT_DIT2)(work[r], work[r + 1], log_m, bytes);
      }
    }
  }
}

#endif /*  LEO_ERROR_BITFIELD_OPT  */

/*  ----------------------------------------------------------------------
    Reed-Solomon Decode (variant)  */
void LEO_FUNC(ReedSolomonDecode)(
    uint64_t buffer_bytes,
    unsigned original_count,
    unsigned recovery_count,
    unsigned m,
    unsigned n,
    const void * const * const original,
    const void * const * const recovery,
    void ** work) {
#ifdef LEO_ERROR_BITFIELD_OPT
  ErrorBitfield error_bits;
  memset(&error_bits, 0, sizeof(error_bits));
#endif

  ffe_t error_locations[kOrder] = {};
  Fi(recovery_count, if (!recovery[i]) error_locations[i] = 1)
  Fi0(m, recovery_count, error_locations[i] = 1)
  for (unsigned i = 0; i < original_count; ++i) {
    if (!original[i]) {
      error_locations[i + m] = 1;
#ifdef LEO_ERROR_BITFIELD_OPT
      ErrorBitfield_Set(&error_bits, i + m);
#endif
    }
  }

#ifdef LEO_ERROR_BITFIELD_OPT
  ErrorBitfield_Prepare(&error_bits);
#endif

  FWHT(error_locations, kOrder, m + original_count);

  Fi(kOrder, error_locations[i] =
      ((unsigned) error_locations[i] * (unsigned) LogWalsh[i]) % kModulus)
  FWHT(error_locations, kOrder, kOrder);

  Fi(recovery_count,
    if (recovery[i])
      LEO_FUNC(mul_mem)(
        work[i], recovery[i], error_locations[i], buffer_bytes);
    else
      memset(work[i], 0, buffer_bytes))
  Fi0(m, recovery_count, memset(work[i], 0, buffer_bytes))

  Fi(original_count,
    if (original[i])
      LEO_FUNC(mul_mem)(work[m + i], original[i],
                        error_locations[m + i], buffer_bytes);
    else
      memset(work[m + i], 0, buffer_bytes))

  Fi0(n, m + original_count, memset(work[i], 0, buffer_bytes))

  LEO_FUNC(IFFT_DIT_Decoder)(buffer_bytes, m + original_count, work, n);

  Fi0(n, 1,
    const unsigned width = ((i ^ (i - 1)) + 1) >> 1;
    LEO_FUNC(VectorXOR)(buffer_bytes, width, work + i - width, work + i))

  const unsigned output_count = m + original_count;

#ifdef LEO_ERROR_BITFIELD_OPT
  LEO_FUNC(FFT_DIT_ErrorBits)(
    buffer_bytes, work, output_count, n, &error_bits);
#else
  LEO_FUNC(FFT_DIT)(buffer_bytes, work, output_count, n);
#endif

  Fi(original_count, if (!original[i])
    LEO_FUNC(mul_mem)(work[i], work[i + m],
                      kModulus - error_locations[i + m],
                      buffer_bytes))
}

/*  ======================================================================
    XOR memory primitives
    ======================================================================  */

void LEO_FUNC(xor_mem)(void * restrict vx,
                       const void * restrict vy, uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  {
    LEO_M256 * restrict x32 = (LEO_M256 *) (vx);
    const LEO_M256 * restrict y32 = (const LEO_M256 *) (vy);
    while (bytes >= 128) {
      const LEO_M256 x0 =
          _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
      const LEO_M256 x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1),
                                           _mm256_loadu_si256(y32 + 1));
      const LEO_M256 x2 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 2),
                                           _mm256_loadu_si256(y32 + 2));
      const LEO_M256 x3 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 3),
                                           _mm256_loadu_si256(y32 + 3));
      _mm256_storeu_si256(x32, x0);
      _mm256_storeu_si256(x32 + 1, x1);
      _mm256_storeu_si256(x32 + 2, x2);
      _mm256_storeu_si256(x32 + 3, x3);
      x32 += 4, y32 += 4;
      bytes -= 128;
    };
    if (bytes > 0) {
      const LEO_M256 x0 =
          _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
      const LEO_M256 x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1),
                                           _mm256_loadu_si256(y32 + 1));
      _mm256_storeu_si256(x32, x0);
      _mm256_storeu_si256(x32 + 1, x1);
    }
    return;
  }
#elif defined(LEO_TRY_SSSE3)
  {
    LEO_M128 * restrict x16 = (LEO_M128 *) (vx);
    const LEO_M128 * restrict y16 = (const LEO_M128 *) (vy);
    do {
      const LEO_M128 x0 =
          _mm_xor_si128(_mm_loadu_si128(x16), _mm_loadu_si128(y16));
      const LEO_M128 x1 =
          _mm_xor_si128(_mm_loadu_si128(x16 + 1), _mm_loadu_si128(y16 + 1));
      const LEO_M128 x2 =
          _mm_xor_si128(_mm_loadu_si128(x16 + 2), _mm_loadu_si128(y16 + 2));
      const LEO_M128 x3 =
          _mm_xor_si128(_mm_loadu_si128(x16 + 3), _mm_loadu_si128(y16 + 3));
      _mm_storeu_si128(x16, x0);
      _mm_storeu_si128(x16 + 1, x1);
      _mm_storeu_si128(x16 + 2, x2);
      _mm_storeu_si128(x16 + 3, x3);
      x16 += 4, y16 += 4;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#else
  {
    uint8_t * restrict x8 = (uint8_t *) (vx);
    const uint8_t * restrict y8 = (const uint8_t *) (vy);
    do *x8++ ^= *y8++; while (--bytes > 0);
    return;
  }
#endif
}

#ifdef LEO_M1_OPT
void LEO_FUNC(xor_mem_2to1)(void * restrict x,
                            const void * restrict y,
                            const void * restrict z,
                            uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  {
    LEO_M256 * restrict x32 = (LEO_M256 *) (x);
    const LEO_M256 * restrict y32 = (const LEO_M256 *) (y);
    const LEO_M256 * restrict z32 = (const LEO_M256 *) (z);
    while (bytes >= 128) {
      LEO_M256 x0 =
          _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
      x0 = _mm256_xor_si256(x0, _mm256_loadu_si256(z32));
      LEO_M256 x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1),
                                     _mm256_loadu_si256(y32 + 1));
      x1 = _mm256_xor_si256(x1, _mm256_loadu_si256(z32 + 1));
      LEO_M256 x2 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 2),
                                     _mm256_loadu_si256(y32 + 2));
      x2 = _mm256_xor_si256(x2, _mm256_loadu_si256(z32 + 2));
      LEO_M256 x3 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 3),
                                     _mm256_loadu_si256(y32 + 3));
      x3 = _mm256_xor_si256(x3, _mm256_loadu_si256(z32 + 3));
      _mm256_storeu_si256(x32, x0);
      _mm256_storeu_si256(x32 + 1, x1);
      _mm256_storeu_si256(x32 + 2, x2);
      _mm256_storeu_si256(x32 + 3, x3);
      x32 += 4, y32 += 4, z32 += 4;
      bytes -= 128;
    };

    if (bytes > 0) {
      LEO_M256 x0 =
          _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
      x0 = _mm256_xor_si256(x0, _mm256_loadu_si256(z32));
      LEO_M256 x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1),
                                     _mm256_loadu_si256(y32 + 1));
      x1 = _mm256_xor_si256(x1, _mm256_loadu_si256(z32 + 1));
      _mm256_storeu_si256(x32, x0);
      _mm256_storeu_si256(x32 + 1, x1);
    }
    return;
  }
#elif defined(LEO_TRY_SSSE3)
  {
    LEO_M128 * restrict x16 = (LEO_M128 *) (x);
    const LEO_M128 * restrict y16 = (const LEO_M128 *) (y);
    const LEO_M128 * restrict z16 = (const LEO_M128 *) (z);
    do {
      LEO_M128 x0 = _mm_xor_si128(_mm_loadu_si128(x16), _mm_loadu_si128(y16));
      x0 = _mm_xor_si128(x0, _mm_loadu_si128(z16));
      LEO_M128 x1 =
          _mm_xor_si128(_mm_loadu_si128(x16 + 1), _mm_loadu_si128(y16 + 1));
      x1 = _mm_xor_si128(x1, _mm_loadu_si128(z16 + 1));
      LEO_M128 x2 =
          _mm_xor_si128(_mm_loadu_si128(x16 + 2), _mm_loadu_si128(y16 + 2));
      x2 = _mm_xor_si128(x2, _mm_loadu_si128(z16 + 2));
      LEO_M128 x3 =
          _mm_xor_si128(_mm_loadu_si128(x16 + 3), _mm_loadu_si128(y16 + 3));
      x3 = _mm_xor_si128(x3, _mm_loadu_si128(z16 + 3));
      _mm_storeu_si128(x16, x0);
      _mm_storeu_si128(x16 + 1, x1);
      _mm_storeu_si128(x16 + 2, x2);
      _mm_storeu_si128(x16 + 3, x3);
      x16 += 4, y16 += 4, z16 += 4;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#else
  {
    uint8_t * restrict x8 = (uint8_t *) (x);
    const uint8_t * restrict y8 = (const uint8_t *) (y);
    const uint8_t * restrict z8 = (const uint8_t *) (z);
    do *x8++ ^= *y8++ ^ *z8++; while (--bytes > 0);
    return;
  }
#endif
}
#endif /*  LEO_M1_OPT  */

#ifdef LEO_USE_VECTOR4_OPT
void LEO_FUNC(xor_mem4)(void * restrict vx_0, const void * restrict vy_0,
                        void * restrict vx_1, const void * restrict vy_1,
                        void * restrict vx_2, const void * restrict vy_2,
                        void * restrict vx_3, const void * restrict vy_3,
                        uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  {
    LEO_M256 * restrict x32_0 = (LEO_M256 *) (vx_0);
    const LEO_M256 * restrict y32_0 = (const LEO_M256 *) (vy_0);
    LEO_M256 * restrict x32_1 = (LEO_M256 *) (vx_1);
    const LEO_M256 * restrict y32_1 = (const LEO_M256 *) (vy_1);
    LEO_M256 * restrict x32_2 = (LEO_M256 *) (vx_2);
    const LEO_M256 * restrict y32_2 = (const LEO_M256 *) (vy_2);
    LEO_M256 * restrict x32_3 = (LEO_M256 *) (vx_3);
    const LEO_M256 * restrict y32_3 = (const LEO_M256 *) (vy_3);
    while (bytes >= 128) {
      const LEO_M256 x0_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0),
                                             _mm256_loadu_si256(y32_0));
      const LEO_M256 x1_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 1),
                                             _mm256_loadu_si256(y32_0 + 1));
      const LEO_M256 x2_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 2),
                                             _mm256_loadu_si256(y32_0 + 2));
      const LEO_M256 x3_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 3),
                                             _mm256_loadu_si256(y32_0 + 3));
      _mm256_storeu_si256(x32_0, x0_0);
      _mm256_storeu_si256(x32_0 + 1, x1_0);
      _mm256_storeu_si256(x32_0 + 2, x2_0);
      _mm256_storeu_si256(x32_0 + 3, x3_0);
      x32_0 += 4, y32_0 += 4;
      const LEO_M256 x0_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1),
                                             _mm256_loadu_si256(y32_1));
      const LEO_M256 x1_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 1),
                                             _mm256_loadu_si256(y32_1 + 1));
      const LEO_M256 x2_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 2),
                                             _mm256_loadu_si256(y32_1 + 2));
      const LEO_M256 x3_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 3),
                                             _mm256_loadu_si256(y32_1 + 3));
      _mm256_storeu_si256(x32_1, x0_1);
      _mm256_storeu_si256(x32_1 + 1, x1_1);
      _mm256_storeu_si256(x32_1 + 2, x2_1);
      _mm256_storeu_si256(x32_1 + 3, x3_1);
      x32_1 += 4, y32_1 += 4;
      const LEO_M256 x0_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2),
                                             _mm256_loadu_si256(y32_2));
      const LEO_M256 x1_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 1),
                                             _mm256_loadu_si256(y32_2 + 1));
      const LEO_M256 x2_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 2),
                                             _mm256_loadu_si256(y32_2 + 2));
      const LEO_M256 x3_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 3),
                                             _mm256_loadu_si256(y32_2 + 3));
      _mm256_storeu_si256(x32_2, x0_2);
      _mm256_storeu_si256(x32_2 + 1, x1_2);
      _mm256_storeu_si256(x32_2 + 2, x2_2);
      _mm256_storeu_si256(x32_2 + 3, x3_2);
      x32_2 += 4, y32_2 += 4;
      const LEO_M256 x0_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3),
                                             _mm256_loadu_si256(y32_3));
      const LEO_M256 x1_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 1),
                                             _mm256_loadu_si256(y32_3 + 1));
      const LEO_M256 x2_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 2),
                                             _mm256_loadu_si256(y32_3 + 2));
      const LEO_M256 x3_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 3),
                                             _mm256_loadu_si256(y32_3 + 3));
      _mm256_storeu_si256(x32_3, x0_3);
      _mm256_storeu_si256(x32_3 + 1, x1_3);
      _mm256_storeu_si256(x32_3 + 2, x2_3);
      _mm256_storeu_si256(x32_3 + 3, x3_3);
      x32_3 += 4, y32_3 += 4;
      bytes -= 128;
    }
    if (bytes > 0) {
      const LEO_M256 x0_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0),
                                             _mm256_loadu_si256(y32_0));
      const LEO_M256 x1_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 1),
                                             _mm256_loadu_si256(y32_0 + 1));
      const LEO_M256 x0_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1),
                                             _mm256_loadu_si256(y32_1));
      const LEO_M256 x1_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 1),
                                             _mm256_loadu_si256(y32_1 + 1));
      _mm256_storeu_si256(x32_0, x0_0);
      _mm256_storeu_si256(x32_0 + 1, x1_0);
      _mm256_storeu_si256(x32_1, x0_1);
      _mm256_storeu_si256(x32_1 + 1, x1_1);
      const LEO_M256 x0_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2),
                                             _mm256_loadu_si256(y32_2));
      const LEO_M256 x1_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 1),
                                             _mm256_loadu_si256(y32_2 + 1));
      const LEO_M256 x0_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3),
                                             _mm256_loadu_si256(y32_3));
      const LEO_M256 x1_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 1),
                                             _mm256_loadu_si256(y32_3 + 1));
      _mm256_storeu_si256(x32_2, x0_2);
      _mm256_storeu_si256(x32_2 + 1, x1_2);
      _mm256_storeu_si256(x32_3, x0_3);
      _mm256_storeu_si256(x32_3 + 1, x1_3);
    }
    return;
  }
#elif defined(LEO_TRY_SSSE3)
  {
    LEO_M128 * restrict x16_0 = (LEO_M128 *) (vx_0);
    const LEO_M128 * restrict y16_0 = (const LEO_M128 *) (vy_0);
    LEO_M128 * restrict x16_1 = (LEO_M128 *) (vx_1);
    const LEO_M128 * restrict y16_1 = (const LEO_M128 *) (vy_1);
    LEO_M128 * restrict x16_2 = (LEO_M128 *) (vx_2);
    const LEO_M128 * restrict y16_2 = (const LEO_M128 *) (vy_2);
    LEO_M128 * restrict x16_3 = (LEO_M128 *) (vx_3);
    const LEO_M128 * restrict y16_3 = (const LEO_M128 *) (vy_3);
    do {
      const LEO_M128 x0_0 =
          _mm_xor_si128(_mm_loadu_si128(x16_0), _mm_loadu_si128(y16_0));
      const LEO_M128 x1_0 =
          _mm_xor_si128(_mm_loadu_si128(x16_0 + 1), _mm_loadu_si128(y16_0 + 1));
      const LEO_M128 x2_0 =
          _mm_xor_si128(_mm_loadu_si128(x16_0 + 2), _mm_loadu_si128(y16_0 + 2));
      const LEO_M128 x3_0 =
          _mm_xor_si128(_mm_loadu_si128(x16_0 + 3), _mm_loadu_si128(y16_0 + 3));
      _mm_storeu_si128(x16_0, x0_0);
      _mm_storeu_si128(x16_0 + 1, x1_0);
      _mm_storeu_si128(x16_0 + 2, x2_0);
      _mm_storeu_si128(x16_0 + 3, x3_0);
      x16_0 += 4, y16_0 += 4;
      const LEO_M128 x0_1 =
          _mm_xor_si128(_mm_loadu_si128(x16_1), _mm_loadu_si128(y16_1));
      const LEO_M128 x1_1 =
          _mm_xor_si128(_mm_loadu_si128(x16_1 + 1), _mm_loadu_si128(y16_1 + 1));
      const LEO_M128 x2_1 =
          _mm_xor_si128(_mm_loadu_si128(x16_1 + 2), _mm_loadu_si128(y16_1 + 2));
      const LEO_M128 x3_1 =
          _mm_xor_si128(_mm_loadu_si128(x16_1 + 3), _mm_loadu_si128(y16_1 + 3));
      _mm_storeu_si128(x16_1, x0_1);
      _mm_storeu_si128(x16_1 + 1, x1_1);
      _mm_storeu_si128(x16_1 + 2, x2_1);
      _mm_storeu_si128(x16_1 + 3, x3_1);
      x16_1 += 4, y16_1 += 4;
      const LEO_M128 x0_2 =
          _mm_xor_si128(_mm_loadu_si128(x16_2), _mm_loadu_si128(y16_2));
      const LEO_M128 x1_2 =
          _mm_xor_si128(_mm_loadu_si128(x16_2 + 1), _mm_loadu_si128(y16_2 + 1));
      const LEO_M128 x2_2 =
          _mm_xor_si128(_mm_loadu_si128(x16_2 + 2), _mm_loadu_si128(y16_2 + 2));
      const LEO_M128 x3_2 =
          _mm_xor_si128(_mm_loadu_si128(x16_2 + 3), _mm_loadu_si128(y16_2 + 3));
      _mm_storeu_si128(x16_2, x0_2);
      _mm_storeu_si128(x16_2 + 1, x1_2);
      _mm_storeu_si128(x16_2 + 2, x2_2);
      _mm_storeu_si128(x16_2 + 3, x3_2);
      x16_2 += 4, y16_2 += 4;
      const LEO_M128 x0_3 =
          _mm_xor_si128(_mm_loadu_si128(x16_3), _mm_loadu_si128(y16_3));
      const LEO_M128 x1_3 =
          _mm_xor_si128(_mm_loadu_si128(x16_3 + 1), _mm_loadu_si128(y16_3 + 1));
      const LEO_M128 x2_3 =
          _mm_xor_si128(_mm_loadu_si128(x16_3 + 2), _mm_loadu_si128(y16_3 + 2));
      const LEO_M128 x3_3 =
          _mm_xor_si128(_mm_loadu_si128(x16_3 + 3), _mm_loadu_si128(y16_3 + 3));
      _mm_storeu_si128(x16_3, x0_3);
      _mm_storeu_si128(x16_3 + 1, x1_3);
      _mm_storeu_si128(x16_3 + 2, x2_3);
      _mm_storeu_si128(x16_3 + 3, x3_3);
      x16_3 += 4, y16_3 += 4;
      bytes -= 64;
    } while (bytes > 0);
    return;
  }
#else
  {
    uint8_t * restrict x8_0 = (uint8_t *) (vx_0);
    const uint8_t * restrict y8_0 = (const uint8_t *) (vy_0);
    uint8_t * restrict x8_1 = (uint8_t *) (vx_1);
    const uint8_t * restrict y8_1 = (const uint8_t *) (vy_1);
    uint8_t * restrict x8_2 = (uint8_t *) (vx_2);
    const uint8_t * restrict y8_2 = (const uint8_t *) (vy_2);
    uint8_t * restrict x8_3 = (uint8_t *) (vx_3);
    const uint8_t * restrict y8_3 = (const uint8_t *) (vy_3);
    do {
      *x8_0++ ^= *y8_0++;
      *x8_1++ ^= *y8_1++;
      *x8_2++ ^= *y8_2++;
      *x8_3++ ^= *y8_3++;
    } while (--bytes > 0);
    return;
  }
#endif
}
#endif /*  LEO_USE_VECTOR4_OPT  */

void LEO_FUNC(VectorXOR)(const uint64_t bytes, unsigned count,
                         void ** x, void ** y) {
#ifdef LEO_USE_VECTOR4_OPT
  if (count >= 4) {
    int i_end = count - 4;
    for (int i = 0; i <= i_end; i += 4) {
      LEO_FUNC(xor_mem4)(x[i + 0], y[i + 0], x[i + 1], y[i + 1],
                         x[i + 2], y[i + 2], x[i + 3], y[i + 3], bytes);
    }
    count %= 4;
    i_end -= count;
    x += i_end;
    y += i_end;
  }
#endif

  Fi(count, LEO_FUNC(xor_mem)(x[i], y[i], bytes))
}

#endif /*  _LMODE_IMPL_H_  */
