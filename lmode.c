/*
   Copyright (C) 2022-2025 Kamila Szewczyk

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

/*
    Copyright (c) 2017 Christopher A. Taylor.  All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of Leopard-RS nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/

#include "lmode.h"
#include "platform.h"

#include <assert.h>
#include <sys/stat.h>

#if defined(XPAR_OPENMP)
  #include <omp.h>
#endif

#include <stdint.h>
#include <stdbool.h>

#if defined(XPAR_X86_64)
  #ifdef HAVE_FUNC_ATTRIBUTE_SYSV_ABI
    #define EXTERNAL_ABI __attribute__((sysv_abi))
  #else
    #define EXTERNAL_ABI
  #endif

  extern EXTERNAL_ABI int xpar_leo_x86_64_cpuflags(void);

  #if HAVE_AVX2
    #define LEO_TRY_AVX2 /* 256-bit */
    #define LEO_ALIGN_BYTES 32
    #define LEO_M256 __m256i
  #endif

  #if HAVE_SSSE3
    #define LEO_TRY_SSSE3   /* 128-bit */
    #include <emmintrin.h>  // SSE2
    #include <tmmintrin.h>  // SSSE3: _mm_shuffle_epi8
    #define LEO_M128 __m128i
  #endif

  #include <x86intrin.h>
#elif defined(XPAR_AARCH64)
  // TODO.
#endif

#ifndef LEO_ALIGN_BYTES
  #define LEO_ALIGN_BYTES 16
#endif

typedef enum LeopardResultT {
  Leopard_Success = 0,         // Operation succeeded
  Leopard_NeedMoreData = -1,   // Not enough recovery data received
  Leopard_TooMuchData = -2,    // Buffer counts are too high
  Leopard_InvalidSize = -3,    // Buffer size must be a multiple of 64 bytes
  Leopard_InvalidCounts = -4,  // Invalid counts provided
  Leopard_InvalidInput = -5,   // A function parameter was invalid
  Leopard_Platform = -6,       // Platform is unsupported
} LeopardResult;

// ----------------------------------------------------------------------------
// Constants

// Avoid calculating final FFT values in decoder using bitfield
#define LEO_ERROR_BITFIELD_OPT

// Interleave butterfly operations between layer pairs in FFT
#define LEO_INTERLEAVE_BUTTERFLY4_OPT

// Optimize M=1 case
#define LEO_M1_OPT

// Unroll inner loops 4 times
#define LEO_USE_VECTOR4_OPT

// Define if unaligned accesses are undesired.
// #define LEO_ALIGNED_ACCESSES

// ----------------------------------------------------------------------------
// Platform/Architecture

#define LEO_FORCE_INLINE inline __attribute__((always_inline))
#define LEO_ALIGNED __attribute__((aligned(LEO_ALIGN_BYTES)))

// ----------------------------------------------------------------------------
// Runtime CPU Architecture Check

static bool CpuHasAVX2 = false;
static bool CpuHasSSSE3 = false;

// ----------------------------------------------------------------------------
// Portable Intrinsics

// Returns highest bit index 0..31 where the first non-zero bit is found
// Precondition: x != 0
LEO_FORCE_INLINE uint32_t LastNonzeroBit32(uint32_t x) {
  return 31 - (uint32_t)__builtin_clz(x);
}

// Returns next power of two at or above given value
LEO_FORCE_INLINE uint32_t NextPow2(uint32_t n) {
  return 2UL << LastNonzeroBit32(n - 1);
}

// ----------------------------------------------------------------------------
// XOR Memory
//
// This works for both 8-bit and 16-bit finite fields

// x[] ^= y[]
void xor_mem(void* restrict x, const void* restrict y, uint64_t bytes);

#ifdef LEO_M1_OPT

// x[] ^= y[] ^ z[]
void xor_mem_2to1(void* restrict x,
                  const void* restrict y,
                  const void* restrict z,
                  uint64_t bytes);

#endif  // LEO_M1_OPT

#ifdef LEO_USE_VECTOR4_OPT

// For i = {0, 1, 2, 3}: x_i[] ^= x_i[]
void xor_mem4(void* restrict x_0,
              const void* restrict y_0,
              void* restrict x_1,
              const void* restrict y_1,
              void* restrict x_2,
              const void* restrict y_2,
              void* restrict x_3,
              const void* restrict y_3,
              uint64_t bytes);

#endif  // LEO_USE_VECTOR4_OPT

// x[] ^= y[]
void VectorXOR(const uint64_t bytes, unsigned count, void** x, void** y);

// ----------------------------------------------------------------------------
// XORSummer

typedef struct {
  void* DestBuffer;
  const void* Waiting;
} XORSummer;

LEO_FORCE_INLINE void XORSummer_Initialize(XORSummer* summer, void* dest) {
  summer->DestBuffer = dest;
  summer->Waiting = NULL;
}

LEO_FORCE_INLINE void XORSummer_Add(XORSummer* summer,
                                    const void* src,
                                    const uint64_t bytes) {
#ifdef LEO_M1_OPT
  if (summer->Waiting) {
    xor_mem_2to1(summer->DestBuffer, src, summer->Waiting, bytes);
    summer->Waiting = NULL;
  } else
    summer->Waiting = src;
#else   // LEO_M1_OPT
  xor_mem(summer->DestBuffer, src, bytes);
#endif  // LEO_M1_OPT
}

LEO_FORCE_INLINE void XORSummer_Finalize(XORSummer* summer,
                                         const uint64_t bytes) {
#ifdef LEO_M1_OPT
  if (summer->Waiting)
    xor_mem(summer->DestBuffer, summer->Waiting, bytes);
#endif  // LEO_M1_OPT
}

// ----------------------------------------------------------------------------
// SIMD-Safe Aligned Memory Allocations

static const unsigned kAlignmentBytes = LEO_ALIGN_BYTES;

static LEO_FORCE_INLINE uint8_t* SIMDSafeAllocate(size_t size) {
  uint8_t* data = (uint8_t*)calloc(1, kAlignmentBytes + size);
  if (!data)
    return NULL;
  unsigned offset = (unsigned)((uintptr_t)data % kAlignmentBytes);
  data += kAlignmentBytes - offset;
  data[-1] = (uint8_t)offset;
  return data;
}

static LEO_FORCE_INLINE void SIMDSafeFree(void* ptr) {
  if (!ptr)
    return;
  uint8_t* data = (uint8_t*)ptr;
  unsigned offset = data[-1];
  if (offset >= kAlignmentBytes) {
    return;
  }
  data -= kAlignmentBytes - offset;
  free(data);
}

/*
  8-bit Finite Field Math

  This finite field contains 256 elements and so each element is one byte.
  This library is designed for data that is a multiple of 64 bytes in size.

  Algorithms are described in LeopardCommon.h
*/

// ----------------------------------------------------------------------------
// Datatypes and Constants

// Finite field element type
typedef uint8_t ffe_t;

// Number of bits per element
#define kBits 8

// Finite field order: Number of elements in the field
#define kOrder 256

// Modulus for field operations
#define kModulus 255

// LFSR Polynomial that generates the field elements
#define kPolynomial 0x11D

// ----------------------------------------------------------------------------
// API

// Returns false if the self-test fails
bool Initialize();

void ReedSolomonEncode(uint64_t buffer_bytes,
                       unsigned original_count,
                       unsigned recovery_count,
                       unsigned m,  // = NextPow2(recovery_count)
                       const void* const* const data,
                       void** work);  // m * 2 elements

void ReedSolomonDecode(
    uint64_t buffer_bytes,
    unsigned original_count,
    unsigned recovery_count,
    unsigned m,                         // = NextPow2(recovery_count)
    unsigned n,                         // = NextPow2(m + original_count)
    const void* const* const original,  // original_count elements
    const void* const* const recovery,  // recovery_count elements
    void** work);                       // n elements

#include <string.h>

// ----------------------------------------------------------------------------
// Encoder API

// recovery_data = parity of original_data (xor sum)
static void EncodeM1(uint64_t buffer_bytes,
                     unsigned original_count,
                     const void* const* const original_data,
                     void* recovery_data) {
  memcpy(recovery_data, original_data[0], buffer_bytes);

  XORSummer summer;
  XORSummer_Initialize(&summer, recovery_data);

  for (unsigned i = 1; i < original_count; ++i)
    XORSummer_Add(&summer, original_data[i], buffer_bytes);

  XORSummer_Finalize(&summer, buffer_bytes);
}

// ----------------------------------------------------------------------------
// Decoder API
static void DecodeM1(uint64_t buffer_bytes,
                     unsigned original_count,
                     const void* const* original_data,
                     const void* recovery_data,
                     void* work_data) {
  memcpy(work_data, recovery_data, buffer_bytes);

  XORSummer summer;
  XORSummer_Initialize(&summer, work_data);

  for (unsigned i = 0; i < original_count; ++i)
    if (original_data[i])
      XORSummer_Add(&summer, original_data[i], buffer_bytes);

  XORSummer_Finalize(&summer, buffer_bytes);
}

// ----------------------------------------------------------------------------
// Datatypes and Constants

// Basis used for generating logarithm tables
static const ffe_t kCantorBasis[kBits] = {1, 214, 152, 146, 86, 200, 88, 230};

// Using the Cantor basis {2} here enables us to avoid a lot of extra
// calculations when applying the formal derivative in decoding.

// ----------------------------------------------------------------------------
// Field Operations

// z = x + y (mod kModulus)
static inline ffe_t AddMod(const ffe_t a, const ffe_t b) {
  const unsigned sum = (unsigned)(a) + b;

  // Partial reduction step, allowing for kModulus to be returned
  return (ffe_t)(sum + (sum >> kBits));
}

// z = x - y (mod kModulus)
static inline ffe_t SubMod(const ffe_t a, const ffe_t b) {
  const unsigned dif = (unsigned)(a)-b;

  // Partial reduction step, allowing for kModulus to be returned
  return (ffe_t)(dif + (dif >> kBits));
}

// ----------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (FWHT) (mod kModulus)

#define FWHT_2(a, b)                \
  {                                 \
    const ffe_t sum = AddMod(a, b); \
    const ffe_t dif = SubMod(a, b); \
    a = sum;                        \
    b = dif;                        \
  }

static LEO_FORCE_INLINE void FWHT_4(ffe_t* data, unsigned s) {
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

// Decimation in time (DIT) Fast Walsh-Hadamard Transform
// Unrolls pairs of layers to perform cross-layer operations in registers
// m_truncated: Number of elements that are non-zero at the front of data
static void FWHT(ffe_t* data, const unsigned m, const unsigned m_truncated) {
  // Decimation in time: Unroll 2 layers at a time
  unsigned dist = 1, dist4 = 4;
  for (; dist4 <= m; dist = dist4, dist4 <<= 2) {
    // For each set of dist*4 elements:
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      // For each set of dist elements:
      for (unsigned i = r; i < r + dist; ++i)
        FWHT_4(data + i, dist);
    }
  }

  // If there is one layer left:
  if (dist < m)
    for (unsigned i = 0; i < dist; ++i)
      FWHT_2(data[i], data[i + dist]);
}

// ----------------------------------------------------------------------------
// Logarithm Tables

static ffe_t LogLUT[kOrder];
static ffe_t ExpLUT[kOrder];

// Returns a * Log(b)
static ffe_t MultiplyLog(ffe_t a, ffe_t log_b) {
  /*
      Note that this operation is not a normal multiplication in a finite
      field because the right operand is already a logarithm.  This is done
      because it moves K table lookups from the Decode() method into the
      initialization step that is less performance critical.  The LogWalsh[]
      table below contains precalculated logarithms so it is easier to do
      all the other multiplies in that form as well.
  */
  if (a == 0)
    return 0;
  return ExpLUT[AddMod(LogLUT[a], log_b)];
}

// Initialize LogLUT[], ExpLUT[]
static void InitializeLogarithmTables() {
  // LFSR table generation:

  unsigned state = 1;
  for (unsigned i = 0; i < kModulus; ++i) {
    ExpLUT[state] = (ffe_t)(i);
    state <<= 1;
    if (state >= kOrder)
      state ^= kPolynomial;
  }
  ExpLUT[0] = kModulus;

  // Conversion to Cantor basis {2}:

  LogLUT[0] = 0;
  for (unsigned i = 0; i < kBits; ++i) {
    const ffe_t basis = kCantorBasis[i];
    const unsigned width = (unsigned)(1UL << i);

    for (unsigned j = 0; j < width; ++j)
      LogLUT[j + width] = LogLUT[j] ^ basis;
  }

  for (unsigned i = 0; i < kOrder; ++i)
    LogLUT[i] = ExpLUT[LogLUT[i]];

  // Generate Exp table from Log table:

  for (unsigned i = 0; i < kOrder; ++i)
    ExpLUT[LogLUT[i]] = i;

  // Note: Handles modulus wrap around with LUT
  ExpLUT[kModulus] = ExpLUT[0];
}

// ----------------------------------------------------------------------------
// Multiplies

/*
  The multiplication algorithm used follows the approach outlined in {4}.
  Specifically section 6 outlines the algorithm used here for 8-bit fields.
*/

#if defined(LEO_TRY_SSSE3)
typedef struct {
  LEO_M128 Value[2];
} Multiply128LUT_t;

static const Multiply128LUT_t* Multiply128LUT = NULL;

// 128-bit x_reg ^= y_reg * log_m
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

static const Multiply256LUT_t* Multiply256LUT = NULL;

// 256-bit x_reg ^= y_reg * log_m
#define LEO_MULADD_256(x_reg, y_reg, table_lo, table_hi)       \
  {                                                            \
    LEO_M256 lo = _mm256_and_si256(y_reg, clr_mask);           \
    lo = _mm256_shuffle_epi8(table_lo, lo);                    \
    LEO_M256 hi = _mm256_srli_epi64(y_reg, 4);                 \
    hi = _mm256_and_si256(hi, clr_mask);                       \
    hi = _mm256_shuffle_epi8(table_hi, hi);                    \
    x_reg = _mm256_xor_si256(x_reg, _mm256_xor_si256(lo, hi)); \
  }

#endif  // LEO_TRY_AVX2

// Stores the product of x * y at offset x + y * 256
// Repeated accesses from the same y value are faster
static const ffe_t* Multiply8LUT = NULL;

// Reference version of muladd: x[] ^= y[] * log_m
static LEO_FORCE_INLINE void RefMulAdd(void* restrict x,
                                       const void* restrict y,
                                       ffe_t log_m,
                                       uint64_t bytes) {
  const ffe_t* restrict lut = Multiply8LUT + (unsigned)log_m * 256;
  const ffe_t* restrict y1 = (const ffe_t*)(y);

#ifdef LEO_ALIGNED_ACCESSES
  ffe_t* restrict x1 = (ffe_t*)(x);

  do {
    for (unsigned j = 0; j < 64; ++j)
      x1[j] ^= lut[y1[j]];

    x1 += 64, y1 += 64;
    bytes -= 64;
  } while (bytes > 0);
#else
  uint64_t* restrict x8 = (uint64_t*)(x);

  do {
    for (unsigned j = 0; j < 8; ++j) {
      uint64_t x_0 = x8[j];
      x_0 ^= (uint64_t)lut[y1[0]];
      x_0 ^= (uint64_t)lut[y1[1]] << 8;
      x_0 ^= (uint64_t)lut[y1[2]] << 16;
      x_0 ^= (uint64_t)lut[y1[3]] << 24;
      x_0 ^= (uint64_t)lut[y1[4]] << 32;
      x_0 ^= (uint64_t)lut[y1[5]] << 40;
      x_0 ^= (uint64_t)lut[y1[6]] << 48;
      x_0 ^= (uint64_t)lut[y1[7]] << 56;
      x8[j] = x_0;
      y1 += 8;
    }

    x8 += 8;
    bytes -= 64;
  } while (bytes > 0);
#endif
}

// Reference version of mul: x[] = y[] * log_m
static LEO_FORCE_INLINE void RefMul(void* restrict x,
                                    const void* restrict y,
                                    ffe_t log_m,
                                    uint64_t bytes) {
  const ffe_t* restrict lut = Multiply8LUT + (unsigned)log_m * 256;
  const ffe_t* restrict y1 = (const ffe_t*)(y);

#ifdef LEO_ALIGNED_ACCESSES
  ffe_t* restrict x1 = (ffe_t*)(x);

  do {
    for (unsigned j = 0; j < 64; ++j)
      x1[j] ^= lut[y1[j]];

    x1 += 64, y1 += 64;
    bytes -= 64;
  } while (bytes > 0);
#else
  uint64_t* restrict x8 = (uint64_t*)(x);

  do {
    for (unsigned j = 0; j < 8; ++j) {
      uint64_t x_0 = (uint64_t)lut[y1[0]];
      x_0 ^= (uint64_t)lut[y1[1]] << 8;
      x_0 ^= (uint64_t)lut[y1[2]] << 16;
      x_0 ^= (uint64_t)lut[y1[3]] << 24;
      x_0 ^= (uint64_t)lut[y1[4]] << 32;
      x_0 ^= (uint64_t)lut[y1[5]] << 40;
      x_0 ^= (uint64_t)lut[y1[6]] << 48;
      x_0 ^= (uint64_t)lut[y1[7]] << 56;
      x8[j] = x_0;
      y1 += 8;
    }

    x8 += 8;
    bytes -= 64;
  } while (bytes > 0);
#endif
}

static void InitializeMultiplyTables() {
  Multiply8LUT = calloc(256, 256);

  // For each left-multiplicand:
  for (unsigned x = 0; x < 256; ++x) {
    ffe_t* lut = (ffe_t*)Multiply8LUT + x;

    if (x == 0) {
      for (unsigned log_y = 0; log_y < 256; ++log_y, lut += 256)
        *lut = 0;
    } else {
      const ffe_t log_x = LogLUT[x];

      for (unsigned log_y = 0; log_y < 256; ++log_y, lut += 256) {
        const ffe_t prod = ExpLUT[AddMod(log_x, log_y)];
        *lut = prod;
      }
    }
  }

#ifdef LEO_TRY_AVX2
  if (CpuHasAVX2)
    Multiply256LUT = (const Multiply256LUT_t*)(SIMDSafeAllocate(
        sizeof(Multiply256LUT_t) * kOrder));
#endif
#ifdef LEO_TRY_SSSE3
  if (CpuHasSSSE3 && !Multiply256LUT)
    Multiply128LUT = (const Multiply128LUT_t*)(SIMDSafeAllocate(
        sizeof(Multiply128LUT_t) * kOrder));
#endif

#if defined(LEO_TRY_AVX2) || defined(LEO_TRY_SSSE3)
  // For each value we could multiply by:
  for (unsigned log_m = 0; log_m < kOrder; ++log_m) {
    // For each 4 bits of the finite field width in bits:
    for (unsigned i = 0, shift = 0; i < 2; ++i, shift += 4) {
      // Construct 16 entry LUT for PSHUFB
      uint8_t lut[16];
      for (ffe_t x = 0; x < 16; ++x)
        lut[x] = MultiplyLog(x << shift, (ffe_t)(log_m));

      const LEO_M128* v_ptr = (const LEO_M128*)(&lut[0]);
      const LEO_M128 value = _mm_loadu_si128(v_ptr);

      // Store in 128-bit wide table
#if defined(LEO_TRY_AVX2)
      if (!CpuHasAVX2)
#endif  // LEO_TRY_AVX2
        _mm_storeu_si128((LEO_M128*)&Multiply128LUT[log_m].Value[i], value);

        // Store in 256-bit wide table
#if defined(LEO_TRY_AVX2)
      if (CpuHasAVX2) {
        _mm256_storeu_si256((LEO_M256*)&Multiply256LUT[log_m].Value[i],
                            _mm256_broadcastsi128_si256(value));
      }
#endif  // LEO_TRY_AVX2
    }
  }
#endif
}

static void mul_mem(void* restrict x,
                    const void* restrict y,
                    ffe_t log_m,
                    uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);

    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);

    LEO_M256* restrict x32 = (LEO_M256*)(x);
    const LEO_M256* restrict y32 = (const LEO_M256*)(y);

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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);

    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);

    LEO_M128* restrict x16 = (LEO_M128*)(x);
    const LEO_M128* restrict y16 = (const LEO_M128*)(y);

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
#endif

  // Reference version:
  RefMul(x, y, log_m, bytes);
}

// ----------------------------------------------------------------------------
// FFT

// Twisted factors used in FFT
static ffe_t FFTSkew[kModulus];

// Factors used in the evaluation of the error locator polynomial
static ffe_t LogWalsh[kOrder];

static void FFTInitialize() {
  ffe_t temp[kBits - 1];

  // Generate FFT skew vector {1}:

  for (unsigned i = 1; i < kBits; ++i)
    temp[i - 1] = (ffe_t)(1UL << i);

  for (unsigned m = 0; m < (kBits - 1); ++m) {
    const unsigned step = 1UL << (m + 1);

    FFTSkew[(1UL << m) - 1] = 0;

    for (unsigned i = m; i < (kBits - 1); ++i) {
      const unsigned s = (1UL << (i + 1));

      for (unsigned j = (1UL << m) - 1; j < s; j += step)
        FFTSkew[j + s] = FFTSkew[j] ^ temp[i];
    }

    temp[m] = kModulus - LogLUT[MultiplyLog(temp[m], LogLUT[temp[m] ^ 1])];

    for (unsigned i = m + 1; i < (kBits - 1); ++i) {
      const ffe_t sum = AddMod(LogLUT[temp[i] ^ 1], temp[m]);
      temp[i] = MultiplyLog(temp[i], sum);
    }
  }

  for (unsigned i = 0; i < kModulus; ++i)
    FFTSkew[i] = LogLUT[FFTSkew[i]];

  // Precalculate FWHT(Log[i]):

  for (unsigned i = 0; i < kOrder; ++i)
    LogWalsh[i] = LogLUT[i];
  LogWalsh[0] = 0;

  FWHT(LogWalsh, kOrder, kOrder);
}

// 2-way butterfly
static void IFFT_DIT2(void* restrict x,
                      void* restrict y,
                      ffe_t log_m,
                      uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);

    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);

    LEO_M256* restrict x32 = (LEO_M256*)(x);
    LEO_M256* restrict y32 = (LEO_M256*)(y);

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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);

    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);

    LEO_M128* restrict x16 = (LEO_M128*)(x);
    LEO_M128* restrict y16 = (LEO_M128*)(y);

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
#endif

  // Reference version:
  xor_mem(y, x, bytes);
  RefMulAdd(x, y, log_m, bytes);
}

// 4-way butterfly
static void IFFT_DIT4(uint64_t bytes,
                      void** work,
                      unsigned dist,
                      const ffe_t log_m01,
                      const ffe_t log_m23,
                      const ffe_t log_m02) {
#ifdef LEO_INTERLEAVE_BUTTERFLY4_OPT

#if defined(LEO_TRY_AVX2)

  if (CpuHasAVX2) {
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

    LEO_M256* restrict work0 = (LEO_M256*)(work[0]);
    LEO_M256* restrict work1 = (LEO_M256*)(work[dist]);
    LEO_M256* restrict work2 = (LEO_M256*)(work[dist * 2]);
    LEO_M256* restrict work3 = (LEO_M256*)(work[dist * 3]);

    do {
      // First layer:
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

      // Second layer:
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

#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    const LEO_M128 t01_lo = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[0]);
    const LEO_M128 t01_hi = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[1]);
    const LEO_M128 t23_lo = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[0]);
    const LEO_M128 t23_hi = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[1]);
    const LEO_M128 t02_lo = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[0]);
    const LEO_M128 t02_hi = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[1]);

    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);

    LEO_M128* restrict work0 = (LEO_M128*)(work[0]);
    LEO_M128* restrict work1 = (LEO_M128*)(work[dist]);
    LEO_M128* restrict work2 = (LEO_M128*)(work[dist * 2]);
    LEO_M128* restrict work3 = (LEO_M128*)(work[dist * 3]);

    do {
      // First layer:
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

      // Second layer:
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
#endif

#endif  // LEO_INTERLEAVE_BUTTERFLY4_OPT

  // First layer:
  if (log_m01 == kModulus)
    xor_mem(work[dist], work[0], bytes);
  else
    IFFT_DIT2(work[0], work[dist], log_m01, bytes);

  if (log_m23 == kModulus)
    xor_mem(work[dist * 3], work[dist * 2], bytes);
  else
    IFFT_DIT2(work[dist * 2], work[dist * 3], log_m23, bytes);

  // Second layer:
  if (log_m02 == kModulus) {
    xor_mem(work[dist * 2], work[0], bytes);
    xor_mem(work[dist * 3], work[dist], bytes);
  } else {
    IFFT_DIT2(work[0], work[dist * 2], log_m02, bytes);
    IFFT_DIT2(work[dist], work[dist * 3], log_m02, bytes);
  }
}

// {x_out, y_out} ^= IFFT_DIT2( {x_in, y_in} )
static void IFFT_DIT2_xor(void* restrict x_in,
                          void* restrict y_in,
                          void* restrict x_out,
                          void* restrict y_out,
                          const ffe_t log_m,
                          uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);

    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);

    const LEO_M256* restrict x32_in = (const LEO_M256*)(x_in);
    const LEO_M256* restrict y32_in = (const LEO_M256*)(y_in);
    LEO_M256* restrict x32_out = (LEO_M256*)(x_out);
    LEO_M256* restrict y32_out = (LEO_M256*)(y_out);

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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);

    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);

    const LEO_M128* restrict x16_in = (const LEO_M128*)(x_in);
    const LEO_M128* restrict y16_in = (const LEO_M128*)(y_in);
    LEO_M128* restrict x16_out = (LEO_M128*)(x_out);
    LEO_M128* restrict y16_out = (LEO_M128*)(y_out);

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
#endif

  // Reference version:
  xor_mem(y_in, x_in, bytes);
  RefMulAdd(x_in, y_in, log_m, bytes);
  xor_mem(y_out, y_in, bytes);
  xor_mem(x_out, x_in, bytes);
}

// xor_result ^= IFFT_DIT4(work)
static void IFFT_DIT4_xor(uint64_t bytes,
                          void** work_in,
                          void** xor_out,
                          unsigned dist,
                          const ffe_t log_m01,
                          const ffe_t log_m23,
                          const ffe_t log_m02) {
#ifdef LEO_INTERLEAVE_BUTTERFLY4_OPT

#if defined(LEO_TRY_AVX2)

  if (CpuHasAVX2) {
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

    const LEO_M256* restrict work0 = (const LEO_M256*)(work_in[0]);
    const LEO_M256* restrict work1 = (const LEO_M256*)(work_in[dist]);
    const LEO_M256* restrict work2 = (const LEO_M256*)(work_in[dist * 2]);
    const LEO_M256* restrict work3 = (const LEO_M256*)(work_in[dist * 3]);
    LEO_M256* restrict xor0 = (LEO_M256*)(xor_out[0]);
    LEO_M256* restrict xor1 = (LEO_M256*)(xor_out[dist]);
    LEO_M256* restrict xor2 = (LEO_M256*)(xor_out[dist * 2]);
    LEO_M256* restrict xor3 = (LEO_M256*)(xor_out[dist * 3]);

    do {
      // First layer:
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

      // Second layer:
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

#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    const LEO_M128 t01_lo = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[0]);
    const LEO_M128 t01_hi = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[1]);
    const LEO_M128 t23_lo = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[0]);
    const LEO_M128 t23_hi = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[1]);
    const LEO_M128 t02_lo = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[0]);
    const LEO_M128 t02_hi = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[1]);

    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);

    const LEO_M128* restrict work0 = (const LEO_M128*)(work_in[0]);
    const LEO_M128* restrict work1 = (const LEO_M128*)(work_in[dist]);
    const LEO_M128* restrict work2 = (const LEO_M128*)(work_in[dist * 2]);
    const LEO_M128* restrict work3 = (const LEO_M128*)(work_in[dist * 3]);
    LEO_M128* restrict xor0 = (LEO_M128*)(xor_out[0]);
    LEO_M128* restrict xor1 = (LEO_M128*)(xor_out[dist]);
    LEO_M128* restrict xor2 = (LEO_M128*)(xor_out[dist * 2]);
    LEO_M128* restrict xor3 = (LEO_M128*)(xor_out[dist * 3]);

    do {
      // First layer:
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

      // Second layer:
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
#endif

#endif  // LEO_INTERLEAVE_BUTTERFLY4_OPT

  // First layer:
  if (log_m01 == kModulus)
    xor_mem(work_in[dist], work_in[0], bytes);
  else
    IFFT_DIT2(work_in[0], work_in[dist], log_m01, bytes);

  if (log_m23 == kModulus)
    xor_mem(work_in[dist * 3], work_in[dist * 2], bytes);
  else
    IFFT_DIT2(work_in[dist * 2], work_in[dist * 3], log_m23, bytes);

  // Second layer:
  if (log_m02 == kModulus) {
    xor_mem(work_in[dist * 2], work_in[0], bytes);
    xor_mem(work_in[dist * 3], work_in[dist], bytes);
  } else {
    IFFT_DIT2(work_in[0], work_in[dist * 2], log_m02, bytes);
    IFFT_DIT2(work_in[dist], work_in[dist * 3], log_m02, bytes);
  }

  xor_mem(xor_out[0], work_in[0], bytes);
  xor_mem(xor_out[dist], work_in[dist], bytes);
  xor_mem(xor_out[dist * 2], work_in[dist * 2], bytes);
  xor_mem(xor_out[dist * 3], work_in[dist * 3], bytes);
}

// Unrolled IFFT for encoder
static void IFFT_DIT_Encoder(const uint64_t bytes,
                             const void* const* data,
                             const unsigned m_truncated,
                             void** work,
                             void** xor_result,
                             const unsigned m,
                             const ffe_t* skewLUT) {
  // I tried rolling the memcpy/memset into the first layer of the FFT and
  // found that it only yields a 4% performance improvement, which is not
  // worth the extra complexity.
  for (unsigned i = 0; i < m_truncated; ++i)
    memcpy(work[i], data[i], bytes);
  for (unsigned i = m_truncated; i < m; ++i)
    memset(work[i], 0, bytes);

  // I tried splitting up the first few layers into L3-cache sized blocks but
  // found that it only provides about 5% performance boost, which is not
  // worth the extra complexity.

  // Decimation in time: Unroll 2 layers at a time
  unsigned dist = 1, dist4 = 4;
  for (; dist4 <= m; dist = dist4, dist4 <<= 2) {
    // For each set of dist*4 elements:
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      const unsigned i_end = r + dist;
      const ffe_t log_m01 = skewLUT[i_end];
      const ffe_t log_m02 = skewLUT[i_end + dist];
      const ffe_t log_m23 = skewLUT[i_end + dist * 2];

      if (dist4 == m && xor_result) {
        // For each set of dist elements:
        for (unsigned i = r; i < i_end; ++i) {
          IFFT_DIT4_xor(bytes, work + i, xor_result + i, dist, log_m01,
                        log_m23, log_m02);
        }
      } else {
        // For each set of dist elements:
        for (unsigned i = r; i < i_end; ++i) {
          IFFT_DIT4(bytes, work + i, dist, log_m01, log_m23, log_m02);
        }
      }
    }

    // I tried alternating sweeps left->right and right->left to reduce cache
    // misses. It provides about 1% performance boost when done for both FFT
    // and IFFT, so it does not seem to be worth the extra complexity.
  }

  // If there is one layer left:
  if (dist < m) {
    // Assuming that dist = m / 2
    const ffe_t log_m = skewLUT[dist];

    if (xor_result) {
      if (log_m == kModulus) {
        for (unsigned i = 0; i < dist; ++i)
          xor_mem_2to1(xor_result[i], work[i], work[i + dist], bytes);
      } else {
        for (unsigned i = 0; i < dist; ++i) {
          IFFT_DIT2_xor(work[i], work[i + dist], xor_result[i],
                        xor_result[i + dist], log_m, bytes);
        }
      }
    } else {
      if (log_m == kModulus)
        VectorXOR(bytes, dist, work + dist, work);
      else {
        for (unsigned i = 0; i < dist; ++i) {
          IFFT_DIT2(work[i], work[i + dist], log_m, bytes);
        }
      }
    }
  }
}

// Basic no-frills version for decoder
static void IFFT_DIT_Decoder(const uint64_t bytes,
                             const unsigned m_truncated,
                             void** work,
                             const unsigned m) {
  // Decimation in time: Unroll 2 layers at a time
  unsigned dist = 1, dist4 = 4;
  for (; dist4 <= m; dist = dist4, dist4 <<= 2) {
    // For each set of dist*4 elements:
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      const unsigned i_end = r + dist;
      const ffe_t log_m01 = FFTSkew[i_end - 1];
      const ffe_t log_m02 = FFTSkew[i_end + dist - 1];
      const ffe_t log_m23 = FFTSkew[i_end + dist * 2 - 1];

      // For each set of dist elements:
      for (unsigned i = r; i < i_end; ++i) {
        IFFT_DIT4(bytes, work + i, dist, log_m01, log_m23, log_m02);
      }
    }
  }

  // If there is one layer left:
  if (dist < m) {
    // Assuming that dist = m / 2
    const ffe_t log_m = FFTSkew[dist - 1];

    if (log_m == kModulus)
      VectorXOR(bytes, dist, work + dist, work);
    else {
      for (unsigned i = 0; i < dist; ++i) {
        IFFT_DIT2(work[i], work[i + dist], log_m, bytes);
      }
    }
  }
}

// 2-way butterfly
static void FFT_DIT2(void* restrict x,
                     void* restrict y,
                     ffe_t log_m,
                     uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
    const LEO_M256 table_lo_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
    const LEO_M256 table_hi_y =
        _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);

    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);

    LEO_M256* restrict x32 = (LEO_M256*)(x);
    LEO_M256* restrict y32 = (LEO_M256*)(y);

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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    const LEO_M128 table_lo_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[0]);
    const LEO_M128 table_hi_y =
        _mm_loadu_si128(&Multiply128LUT[log_m].Value[1]);

    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);

    LEO_M128* restrict x16 = (LEO_M128*)(x);
    LEO_M128* restrict y16 = (LEO_M128*)(y);

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
#endif

  // Reference version:
  RefMulAdd(x, y, log_m, bytes);
  xor_mem(y, x, bytes);
}

// 4-way butterfly
static void FFT_DIT4(uint64_t bytes,
                     void** work,
                     unsigned dist,
                     const ffe_t log_m01,
                     const ffe_t log_m23,
                     const ffe_t log_m02) {
#ifdef LEO_INTERLEAVE_BUTTERFLY4_OPT

#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
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

    LEO_M256* restrict work0 = (LEO_M256*)(work[0]);
    LEO_M256* restrict work1 = (LEO_M256*)(work[dist]);
    LEO_M256* restrict work2 = (LEO_M256*)(work[dist * 2]);
    LEO_M256* restrict work3 = (LEO_M256*)(work[dist * 3]);

    do {
      LEO_M256 work0_reg = _mm256_loadu_si256(work0);
      LEO_M256 work2_reg = _mm256_loadu_si256(work2);
      LEO_M256 work1_reg = _mm256_loadu_si256(work1);
      LEO_M256 work3_reg = _mm256_loadu_si256(work3);

      // First layer:
      if (log_m02 != kModulus) {
        LEO_MULADD_256(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_256(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      work2_reg = _mm256_xor_si256(work0_reg, work2_reg);
      work3_reg = _mm256_xor_si256(work1_reg, work3_reg);

      // Second layer:
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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    const LEO_M128 t01_lo = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[0]);
    const LEO_M128 t01_hi = _mm_loadu_si128(&Multiply128LUT[log_m01].Value[1]);
    const LEO_M128 t23_lo = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[0]);
    const LEO_M128 t23_hi = _mm_loadu_si128(&Multiply128LUT[log_m23].Value[1]);
    const LEO_M128 t02_lo = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[0]);
    const LEO_M128 t02_hi = _mm_loadu_si128(&Multiply128LUT[log_m02].Value[1]);

    const LEO_M128 clr_mask = _mm_set1_epi8(0x0f);

    LEO_M128* restrict work0 = (LEO_M128*)(work[0]);
    LEO_M128* restrict work1 = (LEO_M128*)(work[dist]);
    LEO_M128* restrict work2 = (LEO_M128*)(work[dist * 2]);
    LEO_M128* restrict work3 = (LEO_M128*)(work[dist * 3]);

    do {
      LEO_M128 work0_reg = _mm_loadu_si128(work0);
      LEO_M128 work2_reg = _mm_loadu_si128(work2);
      LEO_M128 work1_reg = _mm_loadu_si128(work1);
      LEO_M128 work3_reg = _mm_loadu_si128(work3);

      // First layer:
      if (log_m02 != kModulus) {
        LEO_MULADD_128(work0_reg, work2_reg, t02_lo, t02_hi);
        LEO_MULADD_128(work1_reg, work3_reg, t02_lo, t02_hi);
      }
      work2_reg = _mm_xor_si128(work0_reg, work2_reg);
      work3_reg = _mm_xor_si128(work1_reg, work3_reg);

      // Second layer:
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
#endif

#endif  // LEO_INTERLEAVE_BUTTERFLY4_OPT

  // First layer:
  if (log_m02 == kModulus) {
    xor_mem(work[dist * 2], work[0], bytes);
    xor_mem(work[dist * 3], work[dist], bytes);
  } else {
    FFT_DIT2(work[0], work[dist * 2], log_m02, bytes);
    FFT_DIT2(work[dist], work[dist * 3], log_m02, bytes);
  }

  // Second layer:
  if (log_m01 == kModulus)
    xor_mem(work[dist], work[0], bytes);
  else
    FFT_DIT2(work[0], work[dist], log_m01, bytes);

  if (log_m23 == kModulus)
    xor_mem(work[dist * 3], work[dist * 2], bytes);
  else
    FFT_DIT2(work[dist * 2], work[dist * 3], log_m23, bytes);
}

// In-place FFT for encoder and decoder
static void FFT_DIT(const uint64_t bytes,
                    void** work,
                    const unsigned m_truncated,
                    const unsigned m) {
  // Decimation in time: Unroll 2 layers at a time
  unsigned dist4 = m, dist = m >> 2;
  for (; dist != 0; dist4 = dist, dist >>= 2) {
    // For each set of dist*4 elements:
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      const unsigned i_end = r + dist;
      const ffe_t log_m01 = FFTSkew[i_end - 1];
      const ffe_t log_m02 = FFTSkew[i_end + dist - 1];
      const ffe_t log_m23 = FFTSkew[i_end + dist * 2 - 1];

      // For each set of dist elements:
      for (unsigned i = r; i < i_end; ++i) {
        FFT_DIT4(bytes, work + i, dist, log_m01, log_m23, log_m02);
      }
    }
  }

  // If there is one layer left:
  if (dist4 == 2) {
    for (unsigned r = 0; r < m_truncated; r += 2) {
      const ffe_t log_m = FFTSkew[r];

      if (log_m == kModulus)
        xor_mem(work[r + 1], work[r], bytes);
      else {
        FFT_DIT2(work[r], work[r + 1], log_m, bytes);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Reed-Solomon Encode

void ReedSolomonEncode(uint64_t buffer_bytes,
                       unsigned original_count,
                       unsigned recovery_count,
                       unsigned m,
                       const void* const* data,
                       void** work) {
  // work <- IFFT(data, m, m)

  const ffe_t* skewLUT = FFTSkew + m - 1;

  IFFT_DIT_Encoder(buffer_bytes, data, original_count < m ? original_count : m,
                   work,
                   NULL,  // No xor output
                   m, skewLUT);

  const unsigned last_count = original_count % m;
  if (m >= original_count)
    goto skip_body;

  // For sets of m data pieces:
  for (unsigned i = m; i + m <= original_count; i += m) {
    data += m;
    skewLUT += m;

    // work <- work xor IFFT(data + i, m, m + i)

    IFFT_DIT_Encoder(buffer_bytes,
                     data,  // data source
                     m,
                     work + m,  // temporary workspace
                     work,      // xor destination
                     m, skewLUT);
  }

  // Handle final partial set of m pieces:
  if (last_count != 0) {
    data += m;
    skewLUT += m;

    // work <- work xor IFFT(data + i, m, m + i)

    IFFT_DIT_Encoder(buffer_bytes,
                     data,  // data source
                     last_count,
                     work + m,  // temporary workspace
                     work,      // xor destination
                     m, skewLUT);
  }

skip_body:

  // work <- FFT(work, m, 0)
  FFT_DIT(buffer_bytes, work, recovery_count, m);
}

// ----------------------------------------------------------------------------
// ErrorBitfield

#ifdef LEO_ERROR_BITFIELD_OPT

#define kWords (kOrder / 64)

typedef struct {
  uint64_t Words[7][kWords];
} ErrorBitfield;

LEO_FORCE_INLINE void ErrorBitfield_Set(ErrorBitfield* bf, unsigned i) {
  bf->Words[0][i / 64] |= (uint64_t)1 << (i % 64);
}

LEO_FORCE_INLINE bool ErrorBitfield_IsNeeded(const ErrorBitfield* bf,
                                             unsigned mip_level,
                                             unsigned bit) {
  if (mip_level >= 8)
    return true;
  return 0 !=
         (bf->Words[mip_level - 1][bit / 64] & ((uint64_t)1 << (bit % 64)));
}

static const uint64_t kHiMasks[5] = {
    0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
    0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL,
};

void ErrorBitfield_Prepare(ErrorBitfield* bf) {
  // First mip level is for final layer of FFT: pairs of data
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

static void FFT_DIT_ErrorBits(const uint64_t bytes,
                              void** work,
                              const unsigned n_truncated,
                              const unsigned n,
                              const ErrorBitfield* error_bits) {
  unsigned mip_level = LastNonzeroBit32(n);

  // Decimation in time: Unroll 2 layers at a time
  unsigned dist4 = n, dist = n >> 2;
  for (; dist != 0; dist4 = dist, dist >>= 2, mip_level -= 2) {
    // For each set of dist*4 elements:
    for (unsigned r = 0; r < n_truncated; r += dist4) {
      if (!ErrorBitfield_IsNeeded(error_bits, mip_level, r))
        continue;

      const ffe_t log_m01 = FFTSkew[r + dist - 1];
      const ffe_t log_m23 = FFTSkew[r + dist * 3 - 1];
      const ffe_t log_m02 = FFTSkew[r + dist * 2 - 1];

      // For each set of dist elements:
      for (unsigned i = r; i < r + dist; ++i) {
        FFT_DIT4(bytes, work + i, dist, log_m01, log_m23, log_m02);
      }
    }
  }

  // If there is one layer left:
  if (dist4 == 2) {
    for (unsigned r = 0; r < n_truncated; r += 2) {
      if (!ErrorBitfield_IsNeeded(error_bits, mip_level, r))
        continue;

      const ffe_t log_m = FFTSkew[r];

      if (log_m == kModulus)
        xor_mem(work[r + 1], work[r], bytes);
      else {
        FFT_DIT2(work[r], work[r + 1], log_m, bytes);
      }
    }
  }
}

#endif  // LEO_ERROR_BITFIELD_OPT

// ----------------------------------------------------------------------------
// Reed-Solomon Decode

void ReedSolomonDecode(
    uint64_t buffer_bytes,
    unsigned original_count,
    unsigned recovery_count,
    unsigned m,  // NextPow2(recovery_count)
    unsigned n,  // NextPow2(m + original_count) = work_count
    const void* const* const original,  // original_count entries
    const void* const* const recovery,  // recovery_count entries
    void** work)                        // n entries
{
  // Fill in error locations

#ifdef LEO_ERROR_BITFIELD_OPT
  ErrorBitfield error_bits;
  memset(&error_bits, 0, sizeof(error_bits));
#endif  // LEO_ERROR_BITFIELD_OPT

  ffe_t error_locations[kOrder] = {};
  Fi(recovery_count, if (!recovery[i]) error_locations[i] = 1)
  Fi0(m, recovery_count, error_locations[i] = 1)
  for (unsigned i = 0; i < original_count; ++i) {
    if (!original[i]) {
      error_locations[i + m] = 1;
#ifdef LEO_ERROR_BITFIELD_OPT
      ErrorBitfield_Set(&error_bits, i + m);
#endif  // LEO_ERROR_BITFIELD_OPT
    }
  }

#ifdef LEO_ERROR_BITFIELD_OPT
  ErrorBitfield_Prepare(&error_bits);
#endif  // LEO_ERROR_BITFIELD_OPT

  // Evaluate error locator polynomial

  FWHT(error_locations, kOrder, m + original_count);

  Fi(kOrder, error_locations[i] =
      ((unsigned)error_locations[i] * (unsigned)LogWalsh[i]) % kModulus)
  FWHT(error_locations, kOrder, kOrder);

  // work <- recovery data
  Fi(recovery_count,
    if (recovery[i])
      mul_mem(work[i], recovery[i], error_locations[i], buffer_bytes);
    else
      memset(work[i], 0, buffer_bytes))
  Fi0(m, recovery_count, memset(work[i], 0, buffer_bytes))

  // work <- original data
  Fi(original_count,
    if (original[i])
      mul_mem(work[m + i], original[i], error_locations[m + i], buffer_bytes);
    else
      memset(work[m + i], 0, buffer_bytes))

  Fi0(n, m + original_count, memset(work[i], 0, buffer_bytes))

  // work <- IFFT(work, n, 0)

  IFFT_DIT_Decoder(buffer_bytes, m + original_count, work, n);

  // work <- FormalDerivative(work, n)

  Fi0(n, 1,
    const unsigned width = ((i ^ (i - 1)) + 1) >> 1;
    VectorXOR(buffer_bytes, width, work + i - width, work + i))
  
  // work <- FFT(work, n, 0) truncated to m + original_count

  const unsigned output_count = m + original_count;

#ifdef LEO_ERROR_BITFIELD_OPT
  FFT_DIT_ErrorBits(buffer_bytes, work, output_count, n, &error_bits);
#else
  FFT_DIT(buffer_bytes, work, output_count, n);
#endif

  // Reveal erasures
  Fi(original_count, if (!original[i])
    mul_mem(work[i], work[i + m], kModulus - error_locations[i + m],
        buffer_bytes))
}

// ----------------------------------------------------------------------------
// XOR Memory

void xor_mem(void* restrict vx, const void* restrict vy, uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
    LEO_M256* restrict x32 = (LEO_M256*)(vx);
    const LEO_M256* restrict y32 = (const LEO_M256*)(vy);
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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    LEO_M128* restrict x16 = (LEO_M128*)(vx);
    const LEO_M128* restrict y16 = (const LEO_M128*)(vy);
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
#endif

  // Simple reference version:
  uint8_t* restrict x8 = (uint8_t*)(vx);
  const uint8_t* restrict y8 = (const uint8_t*)(vy);
  do *x8++ ^= *y8++; while (--bytes > 0);
  return;
}

#ifdef LEO_M1_OPT

void xor_mem_2to1(void* restrict x,
                  const void* restrict y,
                  const void* restrict z,
                  uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
    LEO_M256* restrict x32 = (LEO_M256*)(x);
    const LEO_M256* restrict y32 = (const LEO_M256*)(y);
    const LEO_M256* restrict z32 = (const LEO_M256*)(z);
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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    LEO_M128* restrict x16 = (LEO_M128*)(x);
    const LEO_M128* restrict y16 = (const LEO_M128*)(y);
    const LEO_M128* restrict z16 = (const LEO_M128*)(z);
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
#endif

  // Simple reference version:
  uint8_t* restrict x8 = (uint8_t*)(x);
  const uint8_t* restrict y8 = (const uint8_t*)(y);
  const uint8_t* restrict z8 = (const uint8_t*)(z);
  do *x8++ ^= *y8++ ^ *z8++; while (--bytes > 0);
  return;
}

#endif  // LEO_M1_OPT

#ifdef LEO_USE_VECTOR4_OPT

void xor_mem4(void* restrict vx_0,
              const void* restrict vy_0,
              void* restrict vx_1,
              const void* restrict vy_1,
              void* restrict vx_2,
              const void* restrict vy_2,
              void* restrict vx_3,
              const void* restrict vy_3,
              uint64_t bytes) {
#if defined(LEO_TRY_AVX2)
  if (CpuHasAVX2) {
    LEO_M256* restrict x32_0 = (LEO_M256*)(vx_0);
    const LEO_M256* restrict y32_0 = (const LEO_M256*)(vy_0);
    LEO_M256* restrict x32_1 = (LEO_M256*)(vx_1);
    const LEO_M256* restrict y32_1 = (const LEO_M256*)(vy_1);
    LEO_M256* restrict x32_2 = (LEO_M256*)(vx_2);
    const LEO_M256* restrict y32_2 = (const LEO_M256*)(vy_2);
    LEO_M256* restrict x32_3 = (LEO_M256*)(vx_3);
    const LEO_M256* restrict y32_3 = (const LEO_M256*)(vy_3);
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
#endif  // LEO_TRY_AVX2

#if defined(LEO_TRY_SSSE3)
  if (CpuHasSSSE3) {
    LEO_M128* restrict x16_0 = (LEO_M128*)(vx_0);
    const LEO_M128* restrict y16_0 = (const LEO_M128*)(vy_0);
    LEO_M128* restrict x16_1 = (LEO_M128*)(vx_1);
    const LEO_M128* restrict y16_1 = (const LEO_M128*)(vy_1);
    LEO_M128* restrict x16_2 = (LEO_M128*)(vx_2);
    const LEO_M128* restrict y16_2 = (const LEO_M128*)(vy_2);
    LEO_M128* restrict x16_3 = (LEO_M128*)(vx_3);
    const LEO_M128* restrict y16_3 = (const LEO_M128*)(vy_3);
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
  }
#endif

  // Simple reference version:
  uint8_t* restrict x8_0 = (uint8_t*)(vx_0);
  const uint8_t* restrict y8_0 = (const uint8_t*)(vy_0);
  uint8_t* restrict x8_1 = (uint8_t*)(vx_1);
  const uint8_t* restrict y8_1 = (const uint8_t*)(vy_1);
  uint8_t* restrict x8_2 = (uint8_t*)(vx_2);
  const uint8_t* restrict y8_2 = (const uint8_t*)(vy_2);
  uint8_t* restrict x8_3 = (uint8_t*)(vx_3);
  const uint8_t* restrict y8_3 = (const uint8_t*)(vy_3);
  do {
    *x8_0++ ^= *y8_0++;
    *x8_1++ ^= *y8_1++;
    *x8_2++ ^= *y8_2++;
    *x8_3++ ^= *y8_3++;
  } while (--bytes > 0);
  return;
}

#endif  // LEO_USE_VECTOR4_OPT

void VectorXOR(const uint64_t bytes, unsigned count, void** x, void** y) {
#ifdef LEO_USE_VECTOR4_OPT
  if (count >= 4) {
    int i_end = count - 4;
    for (int i = 0; i <= i_end; i += 4) {
      xor_mem4(x[i + 0], y[i + 0], x[i + 1], y[i + 1], x[i + 2], y[i + 2],
               x[i + 3], y[i + 3], bytes);
    }
    count %= 4;
    i_end -= count;
    x += i_end;
    y += i_end;
  }
#endif  // LEO_USE_VECTOR4_OPT

  Fi(count, xor_mem(x[i], y[i], bytes))
} 

void lmode_gentab() {
#if defined(LEO_TRY_SSSE3) || defined(LEO_TRY_AVX2)
  int flags = xpar_leo_x86_64_cpuflags();
  // rax := (CpuHasSSSE3 << 1) | CpuHasAVX2.
  CpuHasSSSE3 = (flags & 2) != 0;
  CpuHasAVX2 = (flags & 1) != 0;
#endif
  InitializeLogarithmTables();
  InitializeMultiplyTables();
  FFTInitialize();
}

// ============================================================================
//  Implementation of RS sharding erasure codes in O(n log n).
// ============================================================================
typedef struct {
  int data, parity, total, ebuf, dbuf;
} rs;
static rs * rs_init(int data_shards, int parity_shards) {
  rs * r = xmalloc(sizeof(rs));
  r->data = data_shards; r->parity = parity_shards;
  r->total = data_shards + parity_shards;
  r->ebuf = data_shards == 1 ? parity_shards :
              parity_shards == 1 ? 1 : NextPow2(parity_shards) * 2;
  r->dbuf = data_shards == 1 || parity_shards == 1 ? data_shards :
              NextPow2(data_shards + NextPow2(parity_shards));
  return r;
}
static void rs_encode(rs * r, uint8_t ** in, sz len, bool verbose) {
  void ** ework = xmalloc(r->ebuf * sizeof(void *));
  Fi(r->ebuf, ework[i] = xmalloc(len))
  if (verbose)
    fprintf(stderr, "The workspace is r->ebuf * len = %d * %zu = %zu bytes\n",
            r->ebuf, len, r->ebuf * len);
  if (r->data == 1) {
    Fi(r->parity, memcpy(ework[i], in[i], len))
  } else if (r->parity == 1) {
    EncodeM1(len, r->data, (const void * const * const) in, ework[0]);
  } else {
    const unsigned m = NextPow2(r->parity);
    ReedSolomonEncode(len, r->data, r->parity, m, (const void **) in, ework);
  }
  Fi(r->parity, in[r->data + i] = ework[i])
  Fi0(r->ebuf, r->parity, free(ework[i]))  free(ework);
}
static bool rs_correct(rs * r, uint8_t ** in, uint8_t * shards_present,
                       sz len, bool verbose) {
  int present = 0, di = 0, pi = 0;
  Fi(r->total, present += !!shards_present[i])
  if (present < r->data) return false;
  if (present == r->total) return true;
  if (verbose)
    fprintf(stderr, "The workspace is r->dbuf * len = %d * %zu = %zu bytes\n",
                    r->dbuf, len, r->dbuf * len);
  void ** dwork = xmalloc(r->dbuf * sizeof(void*));
  Fi(r->dbuf, dwork[i] = xmalloc(len))
  void ** dshards = xmalloc(r->data * sizeof(void*));
  void ** pshards = xmalloc(r->parity * sizeof(void*));
  Fi(r->total, 
    void * ptr = shards_present[i] ? in[i] : NULL;
    if (i < r->data) dshards[di++] = ptr; else pshards[pi++] = ptr;
  )
  // Check if not enough recovery data arrived
  sz orig_lost = 0, recovery_got = 0, recovery_got_i = 0;
  Fi(r->data, if (!dshards[i]) ++orig_lost)
  Fi(r->parity, if (pshards[i]) { ++recovery_got; recovery_got_i = i; })
  if (recovery_got < orig_lost)
    FATAL("Not enough recovery data received: lost %lu, got %lu\n",
          orig_lost, recovery_got);
  if (r->data == 1) {
    memcpy(dwork[0], pshards[recovery_got_i], len);
  } else if (orig_lost == 0) {
    Fi(r->data, memcpy(dwork[i], dshards[i], len))
  } else if (r->parity == 1) {
    DecodeM1(
        len, r->data,
        (const void * const * const) dshards,
        pshards[0], dwork[orig_lost]);
  } else {
    const sz m = NextPow2(r->parity), n = NextPow2(m + r->data);
    ReedSolomonDecode(
        len, r->data, r->parity, m, n,
        (const void * const * const) dshards,
        (const void * const * const) pshards,
        dwork);
  }
  Fi(r->total, if (!shards_present[i]) memcpy(in[i], dwork[i], len))
  Fi(r->dbuf, free(dwork[i]))  free(dwork); free(dshards);  free(pshards);
  return true;
}
static void rs_destroy(rs * r) { free(r); }

// ============================================================================
//  Sharded mode encoders/decoders.
// ============================================================================
static void do_sharded_encode(sharded_encoding_options_t o,
                              u8 * buf, sz size) {
  FILE * out[MAX_TOTAL_SHARDS] = { NULL };
  if (o.pshards >= o.dshards) FATAL("Too many parity shards.");
  Fi(o.dshards + o.pshards,
    char * name; asprintf(&name, "%s.xpa.%03d", o.output_prefix, i);
    struct stat st;  memset(&st, 0, sizeof(struct stat));
    int exists = stat(name, &st);
    if ((st.st_size || S_ISDIR(st.st_mode)) && exists != -1 && !o.force)
      FATAL("Output file `%s' exists and is not empty.", name);
    if (!(out[i] = fopen(name, "wb"))) FATAL_PERROR("fopen");
    free(name);
  )
  u8 * shards[MAX_TOTAL_SHARDS] = { NULL };
  sz shard_size = (size + o.dshards - 1) / o.dshards;
  if (shard_size & 63) shard_size = (shard_size + 63) & ~63;
  if (shard_size <= 8192)
    FATAL("Input file too small to be sharded with the given parameters.");
  Fi(o.dshards - 1, shards[i] = buf + i * shard_size);
  // last shard: use a temporary buffer to avoid overflowing
  shards[o.dshards - 1] = xmalloc(shard_size);
  if (size > (o.dshards - 1) * shard_size)
    memcpy(shards[o.dshards - 1], buf + (o.dshards - 1) * shard_size,
      size - (o.dshards - 1) * shard_size);
  rs * r = rs_init(o.dshards, o.pshards);
  rs_encode(r, shards, shard_size, o.verbose);
  rs_destroy(r);
  u8 size_bytes[8] = { 0 };
  Fj(8, size_bytes[j] = size >> (56 - 8 * j));
  Fi(o.dshards + o.pshards,
    u32 checksum = crc32c(shards[i], shard_size);  u8 checksum_bytes[4];
    xfwrite("XPAL", 4, out[i]);
    Fj(4, checksum_bytes[j] = checksum >> (24 - 8 * j));
    xfwrite(checksum_bytes, 4, out[i]);
    xfwrite(&o.dshards, 1, out[i]);
    xfwrite(&o.pshards, 1, out[i]);
    xfwrite(&i, 1, out[i]);
    xfwrite(size_bytes, 8, out[i]);
    xfwrite(shards[i], shard_size, out[i]);
  )
  Fi(o.dshards + o.pshards, xfclose(out[i]));
  Fi0(o.dshards + o.pshards, o.dshards, free(shards[i]));
  free(shards[o.dshards - 1]);
}

void log_sharded_encode(sharded_encoding_options_t o) {
  if(!o.no_map) {
    #if defined(XPAR_ALLOW_MAPPING)
    mmap_t map = xpar_map(o.input_name);
    if (map.map) {
      do_sharded_encode(o, map.map, map.size);
      xpar_unmap(&map);
      return;
    }
    #endif
  }
  FILE * in = fopen(o.input_name, "rb");
  if (!in) FATAL_PERROR("fopen");
  if (!is_seekable(in)) FATAL("Input not seekable.");
  fseek(in, 0, SEEK_END);
  sz size = ftell(in);
  fseek(in, 0, SEEK_SET);
  u8 * buffer = xmalloc(size);
  if (xfread(buffer, size, in) != size) FATAL("Short read.");
  fclose(in);
  do_sharded_encode(o, buffer, size);
  free(buffer);
}
void log_sharded_decode(sharded_decoding_options_t opt) {
  sharded_hv_result_t res[MAX_TOTAL_SHARDS];
  if (opt.n_input_shards > MAX_TOTAL_SHARDS)
    FATAL(
      "Too many input shards. While many of them may be wrong and\n"
      "subsequently discarded, this functionality is not implemented\n"
      "yet. Please throw away some of the input shards and try again.\n"
    );
  Fi(opt.n_input_shards,
    res[i] = validate_shard_header(opt.input_files[i], opt, "XPAL");
    if (!res[i].valid) {
      if (!opt.quiet)
        fprintf(stderr,
          "Invalid shard header in `%s', skipping.\n", opt.input_files[i]);
      if (!opt.force) exit(1);
    }
  )
  // Consensus voting.
  u8 consensus_dshards, consensus_pshards;
  sz consensus_size, consensus_shard_size;
  {
    u8 b[MAX_TOTAL_SHARDS];
    Fi(opt.n_input_shards, b[i] = res[i].dshards);
    consensus_dshards = *(u8 *) most_frequent(b, opt.n_input_shards, 1);
    Fi(opt.n_input_shards, b[i] = res[i].pshards);
    consensus_pshards = *(u8 *) most_frequent(b, opt.n_input_shards, 1);
  }
  {
    sz b[MAX_TOTAL_SHARDS];
    Fi(opt.n_input_shards, b[i] = res[i].total_size);
    consensus_size =
      *(sz *) most_frequent((u8 *) b, opt.n_input_shards, sizeof(sz));
    Fi(opt.n_input_shards, b[i] = res[i].shard_size);
    consensus_shard_size =
      *(sz *) most_frequent((u8 *) b, opt.n_input_shards, sizeof(sz));
  }
  // Kick out shards that don't match the consensus.
  Fi(opt.n_input_shards,
    if (res[i].dshards != consensus_dshards
     || res[i].pshards != consensus_pshards
     || res[i].total_size != consensus_size
     || res[i].shard_size != consensus_shard_size) {
      res[i].valid = false;
      if (!opt.quiet)
        fprintf(stderr,
          "Shard `%s' does not match the consensus, skipping.\n",
          opt.input_files[i]);
      if (!opt.force) exit(1);
    }
  )
  if (consensus_shard_size & 63) {
    FATAL(
      "Consensus shard size is not a multiple of 64 bytes. This\n"
      "should never happen with shards produced by this tool.\n"
    );
  }
  // Check if we have a duplicate of any shard.
  Fi(opt.n_input_shards, Fj0(opt.n_input_shards, i + 1, 
    if (res[i].shard_number == res[j].shard_number
      && res[i].valid && res[j].valid) {
      FATAL(
        "Duplicate shard number %u. A future version of this tool\n"
        "will have the capability to try combinations of shards\n"
        "to pick the valid shard %u out of your input. For now,\n"
        "this is a fatal error. Please remove one of the offending\n"
        "shards:\n%s\n%s\n",
        res[i].shard_number, res[i].shard_number,
        opt.input_files[i], opt.input_files[j]
      );
    }
  ))
  // Free the invalid buffers, compact the valid ones.
  Fi(opt.n_input_shards,
    if (!res[i].valid) free(res[i].buf), res[i].shard_number = 0xFF);
  int n_valid_shards = 0;
  Fi(opt.n_input_shards, if (res[i].valid) {
    n_valid_shards++;
    for (int j = i; j && res[j].shard_number < res[j - 1].shard_number; j--) {
      sharded_hv_result_t tmp = res[j];
      res[j] = res[j - 1];
      res[j - 1] = tmp;
    }
  })
  // Log some information.
  if (opt.verbose) {
    fprintf(stderr,
      "Sharding consensus: %u data shards, %u parity shards\n",
      consensus_dshards, consensus_pshards);
    fprintf(stderr,
      "Sharding consensus: %zu bytes total, %zu bytes per shard\n",
      consensus_size, consensus_shard_size);
    fprintf(stderr, "Valid shards:\n");
    Fi(n_valid_shards, 
      fprintf(stderr,
        "  %s (#%d)\n",
        opt.input_files[i], res[i].shard_number);
    )
  }
  if (n_valid_shards < consensus_dshards) {
    FATAL(
      "Not enough valid shards to recover the data. The consensus\n"
      "requires %u shards, but only %d are available.\n",
      consensus_dshards, n_valid_shards);
  }
  FILE * out = fopen(opt.output_file, "wb");
  if (!out) FATAL_PERROR("fopen");
  if (n_valid_shards == consensus_dshards + consensus_pshards) {
    Fi(consensus_dshards,
      sz w = MIN(consensus_size, consensus_shard_size);
      xfwrite(res[i].buf + SHARD_HEADER_SIZE, w, out);
      consensus_size -= w)
    Fi(n_valid_shards, unmap_shard(&res[i]));
    return;
  }
  rs * r = rs_init(consensus_dshards, consensus_pshards);
  u8 * buffers[MAX_TOTAL_SHARDS] = { NULL }, pres[MAX_TOTAL_SHARDS] = { 0 };
  Fi(n_valid_shards,
    buffers[res[i].shard_number] = res[i].buf + SHARD_HEADER_SIZE,
    pres[res[i].shard_number] = 1)
  Fi(consensus_dshards + consensus_pshards, if (!buffers[i]) {
    buffers[i] = xmalloc(consensus_shard_size);
    memset(buffers[i], 0, consensus_shard_size);
  })
  if(!rs_correct(r, buffers, pres, consensus_shard_size, opt.verbose))
    FATAL("Failed to correct the data.");
  Fi(consensus_dshards,
    sz w = MIN(consensus_size, consensus_shard_size);
    xfwrite(buffers[i], w, out);
    consensus_size -= w)
  xfclose(out);
  rs_destroy(r);
  Fi(n_valid_shards, unmap_shard(&res[i]));
  Fi(consensus_dshards + consensus_pshards, if (!pres[i]) free(buffers[i]));
}
