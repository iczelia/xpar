/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Copyright (c) 2017 Christopher A. Taylor.  All rights reserved.

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
    POSSIBILITY OF SUCH DAMAGE.  */

/*  Baseline Leopard-RS dispatcher. Intrinsics live in
    lmode-{scalar,ssse3,avx2}.c; keep this file intrinsic-free.  */

#include "lmode.h"
#include "platform.h"
#include "io_uring_host.h"

#if defined(XPAR_X86_64)
  #ifdef HAVE_FUNC_ATTRIBUTE_SYSV_ABI
    #define EXTERNAL_ABI __attribute__((sysv_abi))
  #else
    #define EXTERNAL_ABI
  #endif

  extern EXTERNAL_ABI int xpar_leo_x86_64_cpuflags(void);
#endif

/*  -----------------------------------------------------------------------
  Constants / field shape.  */

typedef uint8_t ffe_t;

#define kBits 8
#define kOrder 256
#define kModulus 255
#define kPolynomial 0x11D

/*  32-byte align for AVX2; at most 16B waste on 128-bit-only paths.  */
#define kAlignmentBytes 32

/*  -----------------------------------------------------------------------
  Runtime CPU dispatch state.  */

bool CpuHasAVX2 = false;
bool CpuHasSSSE3 = false;

/*  Field tables shared via lmode-impl.h; filled by lmode_gentab().  */

ffe_t LogLUT[kOrder];
ffe_t ExpLUT[kOrder];
ffe_t FFTSkew[kModulus];
ffe_t LogWalsh[kOrder];
const ffe_t * Multiply8LUT = NULL;

/*  Typed accessors live in lmode-impl.h; we only hold raw storage.  */
void * Multiply128LUT_storage = NULL;
void * Multiply256LUT_storage = NULL;

/*  Basis used for generating logarithm tables.  */
static const ffe_t kCantorBasis[kBits] = {1, 214, 152, 146, 86, 200, 88, 230};

/*  Scalar helpers duplicated from lmode-impl.h to avoid including it.  */

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

static inline void FWHT_4(ffe_t * data, unsigned s) {
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

/*  Decimation-in-time FWHT (mod kModulus); called from variants.  */
void FWHT(ffe_t * data, const unsigned m, const unsigned m_truncated) {
  unsigned dist = 1, dist4 = 4;
  for (; dist4 <= m; dist = dist4, dist4 <<= 2) {
    for (unsigned r = 0; r < m_truncated; r += dist4) {
      for (unsigned i = r; i < r + dist; ++i)
        FWHT_4(data + i, dist);
    }
  }
  if (dist < m)
    for (unsigned i = 0; i < dist; ++i)
      FWHT_2(data[i], data[i + dist]);
}

static inline ffe_t MultiplyLog(ffe_t a, ffe_t log_b) {
  if (a == 0) return 0;
  return ExpLUT[AddMod(LogLUT[a], log_b)];
}

/*  Used for NextPow2(); also referenced by rs_init().  */
static inline uint32_t LastNonzeroBit32(uint32_t x) {
  return 31 - (uint32_t) __builtin_clz(x);
}

static inline uint32_t NextPow2(uint32_t n) {
  return 2UL << LastNonzeroBit32(n - 1);
}

/*  -----------------------------------------------------------------------
  SIMD-safe aligned allocation. Pure pointer arithmetic, no intrinsics.  */

static inline uint8_t * SIMDSafeAllocate(size_t size) {
  uint8_t * data = (uint8_t *) xpar_malloc(kAlignmentBytes + size);
  if (!data) return NULL;
  unsigned offset = (unsigned) ((uintptr_t) data % kAlignmentBytes);
  data += kAlignmentBytes - offset;
  data[-1] = (uint8_t) offset;
  return data;
}

static inline void SIMDSafeFree(void * ptr) {
  if (!ptr) return;
  uint8_t * data = (uint8_t *) ptr;
  unsigned offset = data[-1];
  if (offset >= kAlignmentBytes) return;
  data -= kAlignmentBytes - offset;
  xpar_free(data);
}

/*  Per-variant entry points; definitions in lmode-{scalar,ssse3,avx2}.c.  */

extern void ReedSolomonEncode_scalar(uint64_t buffer_bytes,
                                     unsigned original_count,
                                     unsigned recovery_count,
                                     unsigned m,
                                     const void * const * data,
                                     void ** work);
extern void ReedSolomonDecode_scalar(uint64_t buffer_bytes,
                                     unsigned original_count,
                                     unsigned recovery_count,
                                     unsigned m,
                                     unsigned n,
                                     const void * const * const original,
                                     const void * const * const recovery,
                                     void ** work);
extern void EncodeM1_scalar(uint64_t buffer_bytes,
                            unsigned original_count,
                            const void * const * const original_data,
                            void * recovery_data);
extern void DecodeM1_scalar(uint64_t buffer_bytes,
                            unsigned original_count,
                            const void * const * original_data,
                            const void * recovery_data,
                            void * work_data);

#if defined(HAVE_SSSE3) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))
extern void ReedSolomonEncode_ssse3(uint64_t buffer_bytes,
                                    unsigned original_count,
                                    unsigned recovery_count,
                                    unsigned m,
                                    const void * const * data,
                                    void ** work);
extern void ReedSolomonDecode_ssse3(uint64_t buffer_bytes,
                                    unsigned original_count,
                                    unsigned recovery_count,
                                    unsigned m,
                                    unsigned n,
                                    const void * const * const original,
                                    const void * const * const recovery,
                                    void ** work);
extern void EncodeM1_ssse3(uint64_t buffer_bytes,
                           unsigned original_count,
                           const void * const * const original_data,
                           void * recovery_data);
extern void DecodeM1_ssse3(uint64_t buffer_bytes,
                           unsigned original_count,
                           const void * const * original_data,
                           const void * recovery_data,
                           void * work_data);
extern void FillPshufbTables_ssse3(void);
#endif

#if defined(HAVE_AVX2)
extern void ReedSolomonEncode_avx2(uint64_t buffer_bytes,
                                   unsigned original_count,
                                   unsigned recovery_count,
                                   unsigned m,
                                   const void * const * data,
                                   void ** work);
extern void ReedSolomonDecode_avx2(uint64_t buffer_bytes,
                                   unsigned original_count,
                                   unsigned recovery_count,
                                   unsigned m,
                                   unsigned n,
                                   const void * const * const original,
                                   const void * const * const recovery,
                                   void ** work);
extern void EncodeM1_avx2(uint64_t buffer_bytes,
                          unsigned original_count,
                          const void * const * const original_data,
                          void * recovery_data);
extern void DecodeM1_avx2(uint64_t buffer_bytes,
                          unsigned original_count,
                          const void * const * original_data,
                          const void * recovery_data,
                          void * work_data);
extern void FillPshufbTables_avx2(void);
#endif

/*  -----------------------------------------------------------------------
  Table initialization.  */

static void InitializeLogarithmTables(void) {
  unsigned state = 1;
  for (unsigned i = 0; i < kModulus; ++i) {
    ExpLUT[state] = (ffe_t) (i);
    state <<= 1;
    if (state >= kOrder)
      state ^= kPolynomial;
  }
  ExpLUT[0] = kModulus;

  LogLUT[0] = 0;
  for (unsigned i = 0; i < kBits; ++i) {
    const ffe_t basis = kCantorBasis[i];
    const unsigned width = (unsigned) (1UL << i);
    for (unsigned j = 0; j < width; ++j)
      LogLUT[j + width] = LogLUT[j] ^ basis;
  }

  for (unsigned i = 0; i < kOrder; ++i)
    LogLUT[i] = ExpLUT[LogLUT[i]];

  for (unsigned i = 0; i < kOrder; ++i)
    ExpLUT[LogLUT[i]] = i;

  ExpLUT[kModulus] = ExpLUT[0];
}

/*  Bytes per pshufb LUT entry; must match Multiply{128,256}LUT_t.  */
#define MULT128_LUT_BYTES  (2 * 16)
#define MULT256_LUT_BYTES  (2 * 32)

static void InitializeMultiplyTables(void) {
  /*  Plain 8-bit LUT: 256 x 256 bytes. Same for every variant.  */
  Multiply8LUT = xpar_malloc(256 * 256);

  for (unsigned x = 0; x < 256; ++x) {
    ffe_t * lut = (ffe_t *) Multiply8LUT + x;
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

  /*  Allocate pshufb LUTs; per-variant TU performs the broadcast-fill.  */
#if defined(HAVE_AVX2)
  if (CpuHasAVX2) {
    Multiply256LUT_storage = SIMDSafeAllocate(MULT256_LUT_BYTES * kOrder);
    FillPshufbTables_avx2();
  }
#endif
#if defined(HAVE_SSSE3) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))
  if (CpuHasSSSE3 && !Multiply256LUT_storage) {
    Multiply128LUT_storage = SIMDSafeAllocate(MULT128_LUT_BYTES * kOrder);
    FillPshufbTables_ssse3();
  }
#endif
}

static void FFTInitialize(void) {
  ffe_t temp[kBits - 1];

  for (unsigned i = 1; i < kBits; ++i)
    temp[i - 1] = (ffe_t) (1UL << i);

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

  for (unsigned i = 0; i < kOrder; ++i)
    LogWalsh[i] = LogLUT[i];
  LogWalsh[0] = 0;

  FWHT(LogWalsh, kOrder, kOrder);
}

void lmode_gentab(void) {
#if defined(XPAR_X86_64)
  int flags = xpar_leo_x86_64_cpuflags();
  /*   rax := (CpuHasSSSE3 << 1) | CpuHasAVX2.  */
  CpuHasSSSE3 = (flags & 2) != 0;
  CpuHasAVX2 = (flags & 1) != 0;
#elif defined(XPAR_AARCH64) && defined(HAVE_NEON)
  /*  sse2neon maps SSSE3 onto NEON; safe on any aarch64.  */
  CpuHasSSSE3 = true;
  CpuHasAVX2 = false;
#endif
  InitializeLogarithmTables();
  InitializeMultiplyTables();
  FFTInitialize();
}

/*  Public dispatchers; route by CpuHas* set in lmode_gentab().  */

void ReedSolomonEncode(uint64_t buffer_bytes,
                       unsigned original_count,
                       unsigned recovery_count,
                       unsigned m,
                       const void * const * data,
                       void ** work) {
#if defined(HAVE_AVX2)
  if (CpuHasAVX2) {
    ReedSolomonEncode_avx2(buffer_bytes, original_count, recovery_count, m,
                           data, work);
    return;
  }
#endif
#if defined(HAVE_SSSE3) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))
  if (CpuHasSSSE3) {
    ReedSolomonEncode_ssse3(buffer_bytes, original_count, recovery_count, m,
                            data, work);
    return;
  }
#endif
  ReedSolomonEncode_scalar(buffer_bytes, original_count, recovery_count, m,
                           data, work);
}

void ReedSolomonDecode(uint64_t buffer_bytes,
                       unsigned original_count,
                       unsigned recovery_count,
                       unsigned m,
                       unsigned n,
                       const void * const * const original,
                       const void * const * const recovery,
                       void ** work) {
#if defined(HAVE_AVX2)
  if (CpuHasAVX2) {
    ReedSolomonDecode_avx2(buffer_bytes, original_count, recovery_count, m, n,
                           original, recovery, work);
    return;
  }
#endif
#if defined(HAVE_SSSE3) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))
  if (CpuHasSSSE3) {
    ReedSolomonDecode_ssse3(buffer_bytes, original_count, recovery_count, m, n,
                            original, recovery, work);
    return;
  }
#endif
  ReedSolomonDecode_scalar(buffer_bytes, original_count, recovery_count, m, n,
                           original, recovery, work);
}

static void EncodeM1(uint64_t buffer_bytes,
                     unsigned original_count,
                     const void * const * const original_data,
                     void * recovery_data) {
#if defined(HAVE_AVX2)
  if (CpuHasAVX2) {
    EncodeM1_avx2(buffer_bytes, original_count, original_data, recovery_data);
    return;
  }
#endif
#if defined(HAVE_SSSE3) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))
  if (CpuHasSSSE3) {
    EncodeM1_ssse3(buffer_bytes, original_count, original_data, recovery_data);
    return;
  }
#endif
  EncodeM1_scalar(buffer_bytes, original_count, original_data, recovery_data);
}

static void DecodeM1(uint64_t buffer_bytes,
                     unsigned original_count,
                     const void * const * original_data,
                     const void * recovery_data,
                     void * work_data) {
#if defined(HAVE_AVX2)
  if (CpuHasAVX2) {
    DecodeM1_avx2(buffer_bytes, original_count, original_data, recovery_data,
                  work_data);
    return;
  }
#endif
#if defined(HAVE_SSSE3) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))
  if (CpuHasSSSE3) {
    DecodeM1_ssse3(buffer_bytes, original_count, original_data, recovery_data,
                   work_data);
    return;
  }
#endif
  DecodeM1_scalar(buffer_bytes, original_count, original_data, recovery_data,
                  work_data);
}

/*  -----------------------------------------------------------------------
  Implementation of RS sharding erasure codes in O(n log n).  */
typedef struct {
  int data, parity, total, ebuf, dbuf;
} rs;
static rs * rs_init(int data_shards, int parity_shards) {
  rs * r = xpar_malloc(sizeof(rs));
  r->data = data_shards; r->parity = parity_shards;
  r->total = data_shards + parity_shards;
  r->ebuf = data_shards == 1 ? parity_shards :
              parity_shards == 1 ? 1 : NextPow2(parity_shards) * 2;
  r->dbuf = data_shards == 1 || parity_shards == 1 ? data_shards :
              NextPow2(data_shards + NextPow2(parity_shards));
  return r;
}
static void rs_encode(rs * r, uint8_t ** in, sz len, bool verbose,
                      xpar_progress_t * prog) {
  void ** ework = xpar_malloc(r->ebuf * sizeof(void *));
  Fi(r->ebuf, ework[i] = SIMDSafeAllocate(len))
  if (verbose)
    xpar_fprintf(xpar_stderr,
      "The workspace is r->ebuf * len = %d * %zu = %zu bytes\n",
      r->ebuf, len, r->ebuf * len);
  /*  Chunk the byte dimension so progress ticks periodically. The FFT is
      over the shard axis; byte offsets within a shard are independent, so
      processing (off, off+cl) slices in sequence yields the same output.
      Chunk at a multiple of 64 to preserve SIMD alignment of shard starts.  */
  sz chunk = MiB(64);
  if (chunk > len) chunk = len;
  if (!chunk) chunk = 1;
  void ** in_sl = xpar_malloc(r->total * sizeof(void *));
  void ** ew_sl = xpar_malloc(r->ebuf * sizeof(void *));
  for (sz off = 0; off < len; off += chunk) {
    sz cl = MIN(chunk, len - off);
    Fi(r->total,    in_sl[i] = in[i] ? ((u8 *) in[i]) + off : NULL);
    Fi(r->ebuf,     ew_sl[i] = ((u8 *) ework[i]) + off);
    if (r->data == 1) {
      Fi(r->parity, xpar_memcpy(ew_sl[i], in_sl[0], cl))
    } else if (r->parity == 1) {
      EncodeM1(cl, r->data,
               (const void * const * const) in_sl, ew_sl[0]);
    } else {
      const unsigned m = NextPow2(r->parity);
      ReedSolomonEncode(cl, r->data, r->parity, m,
                        (const void **) in_sl, ew_sl);
    }
    xpar_progress_tick(prog, cl * (sz) r->data);
  }
  xpar_free(in_sl); xpar_free(ew_sl);
  Fi(r->parity, in[r->data + i] = ework[i])
  Fi0(r->ebuf, r->parity, SIMDSafeFree(ework[i]))  xpar_free(ework);
}
static bool rs_correct(rs * r, uint8_t ** in, uint8_t * shards_present,
                       sz len, bool verbose, xpar_progress_t * prog) {
  int present = 0, di = 0, pi = 0;
  Fi(r->total, present += !!shards_present[i])
  if (present < r->data) return false;
  if (present == r->total) return true;
  if (verbose)
    xpar_fprintf(xpar_stderr,
      "The workspace is r->dbuf * len = %d * %zu = %zu bytes\n",
      r->dbuf, len, r->dbuf * len);
  void ** dwork = xpar_malloc(r->dbuf * sizeof(void*));
  Fi(r->dbuf, dwork[i] = SIMDSafeAllocate(len))
  void ** dshards = xpar_malloc(r->data * sizeof(void*));
  void ** pshards = xpar_malloc(r->parity * sizeof(void*));
  Fi(r->total,
    void * ptr = shards_present[i] ? in[i] : NULL;
    if (i < r->data) dshards[di++] = ptr; else pshards[pi++] = ptr;
  )
  /*   Check if not enough recovery data arrived  */
  sz orig_lost = 0, recovery_got = 0, recovery_got_i = 0;
  Fi(r->data, if (!dshards[i]) ++orig_lost)
  Fi(r->parity, if (pshards[i]) { ++recovery_got; recovery_got_i = i; })
  if (recovery_got < orig_lost)
    FATAL("Not enough recovery data received: lost %lu, got %lu\n",
          orig_lost, recovery_got);
  sz chunk = MiB(64);
  if (chunk > len) chunk = len;
  if (!chunk) chunk = 1;
  void ** dsh_sl = xpar_malloc(r->data * sizeof(void *));
  void ** psh_sl = xpar_malloc(r->parity * sizeof(void *));
  void ** dw_sl  = xpar_malloc(r->dbuf * sizeof(void *));
  for (sz off = 0; off < len; off += chunk) {
    sz cl = MIN(chunk, len - off);
    Fi(r->data,   dsh_sl[i] = dshards[i] ? ((u8 *) dshards[i]) + off : NULL);
    Fi(r->parity, psh_sl[i] = pshards[i] ? ((u8 *) pshards[i]) + off : NULL);
    Fi(r->dbuf,   dw_sl[i]  = ((u8 *) dwork[i]) + off);
    if (r->data == 1) {
      xpar_memcpy(dw_sl[0], psh_sl[recovery_got_i], cl);
    } else if (orig_lost == 0) {
      Fi(r->data, xpar_memcpy(dw_sl[i], dsh_sl[i], cl))
    } else if (r->parity == 1) {
      DecodeM1(cl, r->data,
               (const void * const * const) dsh_sl,
               psh_sl[0], dw_sl[orig_lost]);
    } else {
      const sz m = NextPow2(r->parity), n = NextPow2(m + r->data);
      ReedSolomonDecode(cl, r->data, r->parity, m, n,
                        (const void * const * const) dsh_sl,
                        (const void * const * const) psh_sl,
                        dw_sl);
    }
    xpar_progress_tick(prog, cl * (sz) r->data);
  }
  xpar_free(dsh_sl); xpar_free(psh_sl); xpar_free(dw_sl);
  Fi(r->total, if (!shards_present[i]) xpar_memcpy(in[i], dwork[i], len))
  Fi(r->dbuf, SIMDSafeFree(dwork[i]))
  xpar_free(dwork);  xpar_free(dshards);  xpar_free(pshards);
  return true;
}
static void rs_destroy(rs * r) { xpar_free(r); }

/*  -----------------------------------------------------------------------
  Sharded mode encoders/decoders.  */
#ifdef XPAR_HAS_LIBURING
/*  Emit every shard (data + parity) via io_uring. The data shards can
    be enqueued BEFORE rs_encode runs since their bytes are just slices
    of the input; parity shards must wait. Returns false if io_uring is
    not usable at runtime, and the caller takes the sync path.  */
static bool sharded_encode_uring(sharded_encoding_options_t o,
                                 xpar_file ** out, u8 ** shards,
                                 sz shard_size, sz size) {
  xpar_iogroup * iog = xpar_iogroup_new(
      4u * (unsigned)(o.dshards + o.pshards));
  if (!iog) return false;
  int fid[MAX_TOTAL_SHARDS];
  for (int i = 0; i < o.dshards + o.pshards; i++) {
    fid[i] = xpar_iogroup_register_file(iog, out[i]);
    if (fid[i] < 0) { xpar_iogroup_free(iog); return false; }
  }
  u8 (*hdrs)[SHARD_HEADER_BLAKE2B_SIZE]
      = xpar_malloc((sz)(o.dshards + o.pshards) * sizeof *hdrs);
  sz hsz[MAX_TOTAL_SHARDS];
  /*  Phase A: enqueue k data shards before the FFT runs.  */
  Fi0(o.dshards, 0,
    hsz[i] = pack_shard_header(hdrs[i], "XPAL",
                               o.dshards, o.pshards, (u8) i,
                               size, shards[i], shard_size,
                               o.integrity, o.auth_key, o.auth_keylen);
    xpar_iogroup_enqueue_write(iog, fid[i], hdrs[i],   0,
                               hsz[i],                (u64) i);
    xpar_iogroup_enqueue_write(iog, fid[i], shards[i], hsz[i],
                               shard_size,            (u64) i);
  )
  xpar_iogroup_submit(iog);
  rs * r = rs_init(o.dshards, o.pshards);
  xpar_progress_t prog;
  xpar_progress_init(&prog, o.progress,
    (u64) shard_size * (u64) o.dshards, "Encoding");
  rs_encode(r, shards, shard_size, o.verbose, &prog);
  xpar_progress_end(&prog);
  rs_destroy(r);
  /*  Phase C: parity shards exist now -- enqueue, fsync, drain.  */
  Fi0(o.dshards + o.pshards, o.dshards,
    hsz[i] = pack_shard_header(hdrs[i], "XPAL",
                               o.dshards, o.pshards, (u8) i,
                               size, shards[i], shard_size,
                               o.integrity, o.auth_key, o.auth_keylen);
    xpar_iogroup_enqueue_write(iog, fid[i], hdrs[i],   0,
                               hsz[i],                (u64) i);
    xpar_iogroup_enqueue_write(iog, fid[i], shards[i], hsz[i],
                               shard_size,            (u64) i);
  )
  Fi(o.dshards + o.pshards, xpar_iogroup_fsync(iog, fid[i]));
  xpar_iogroup_free(iog);   /*  drains + FATALs on CQE error  */
  xpar_free(hdrs);
  return true;
}
#endif

static void do_sharded_encode(sharded_encoding_options_t o,
                              u8 * buf, sz size) {
  xpar_file * out[MAX_TOTAL_SHARDS] = { NULL };
  if (o.pshards >= o.dshards) FATAL("Too many parity shards.");
  Fi(o.dshards + o.pshards,
    char * name;
#if defined(XPAR_DOS)
    /*  8.3 DOS: shards get a numeric extension, no .xpa marker.  */
    char suf[8]; xpar_snprintf(suf, sizeof(suf), ".%03d", i);
    name = xpar_derive_name(o.output_prefix, suf);
#else
    xpar_asprintf(&name, "%s.xpa.%03d", o.output_prefix, i);
#endif
    xpar_stat_t st;
    int exists = xpar_stat_path(name, &st);
    if (exists == 0 && (st.size || st.is_dir) && !o.force)
      FATAL("Output file `%s' exists and is not empty.", name);
    if (!(out[i] = xpar_open(name,
        XPAR_O_WRITE | XPAR_O_CREATE | XPAR_O_TRUNCATE)))
      FATAL_PERROR("fopen");
    xpar_free(name);
  )
  u8 * shards[MAX_TOTAL_SHARDS] = { NULL };
  bool owned[MAX_TOTAL_SHARDS] = { false };
  sz shard_size = (size + o.dshards - 1) / o.dshards;
  if (shard_size & 63) shard_size = (shard_size + 63) & ~63;
  if (shard_size <= 8192)
    FATAL("Input file too small to be sharded with the given parameters.");
  /*   Any data shard whose slice would cross the input end (possible here
       because shard_size is rounded up to 64) gets its own zeroed buffer
       with the available bytes copied in: avoids OOB reads on `buf` and
       keeps heap garbage out of the shard body / MAC domain.  */
  Fi(o.dshards,
    sz start = (sz) i * shard_size;
    if (start + shard_size <= size)
      shards[i] = buf + start;
    else {
      shards[i] = SIMDSafeAllocate(shard_size);
      xpar_memset(shards[i], 0, shard_size);
      if (start < size)
        xpar_memcpy(shards[i], buf + start, size - start);
      owned[i] = true;
    })
#ifdef XPAR_HAS_LIBURING
  if (!sharded_encode_uring(o, out, shards, shard_size, size))
#endif
  {
    rs * r = rs_init(o.dshards, o.pshards);
    xpar_progress_t prog;
    xpar_progress_init(&prog, o.progress,
      (u64) shard_size * (u64) o.dshards, "Encoding");
    rs_encode(r, shards, shard_size, o.verbose, &prog);
    xpar_progress_end(&prog);
    rs_destroy(r);
    Fi(o.dshards + o.pshards,
      u8 hdr[SHARD_HEADER_BLAKE2B_SIZE];
      sz hs = pack_shard_header(hdr, "XPAL", o.dshards, o.pshards, (u8) i,
                                size, shards[i], shard_size,
                                o.integrity, o.auth_key, o.auth_keylen);
      xpar_xwrite(out[i], hdr, hs);
      xpar_xwrite(out[i], shards[i], shard_size);
    )
  }

  Fi(o.dshards + o.pshards, xpar_xclose(out[i]));
  Fi0(o.dshards + o.pshards, o.dshards, SIMDSafeFree(shards[i]));
  Fi(o.dshards, if (owned[i]) SIMDSafeFree(shards[i]));
}

void log_sharded_encode(sharded_encoding_options_t o) {
  if (!o.no_map) {
    #if defined(XPAR_ALLOW_MAPPING)
    xpar_mmap map = xpar_map(o.input_name);
    if (map.map) {
      do_sharded_encode(o, map.map, map.size);
      xpar_unmap(&map);
      return;
    }
    #endif
  }
  xpar_file * in = xpar_open(o.input_name, XPAR_O_READ);
  if (!in) FATAL_PERROR("fopen");
  if (!xpar_is_seekable(in)) FATAL("Input not seekable.");
  xpar_seek(in, 0, XPAR_SEEK_END);
  i64 size_raw = xpar_tell(in);
  if (size_raw < 0) FATAL_PERROR("ftell");
  sz size = (sz) size_raw;
  xpar_seek(in, 0, XPAR_SEEK_SET);
  u8 * buffer = SIMDSafeAllocate(size);
  if (xpar_xread(in, buffer, size) != size) FATAL("Short read.");
  xpar_close(in);
  do_sharded_encode(o, buffer, size);
  SIMDSafeFree(buffer);
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
        xpar_fprintf(xpar_stderr,
          "Invalid shard header in `%s', skipping.\n", opt.input_files[i]);
      if (!opt.force) xpar_exit(1);
    }
  )
  /*   Consensus voting.  */
  shard_consensus_t sc = consensus_of_valid(res, opt.n_input_shards);
  u8 consensus_dshards = sc.dshards, consensus_pshards = sc.pshards;
  sz consensus_size = sc.total_size, consensus_shard_size = sc.shard_size;
  if ((u64) consensus_size
      > (u64) consensus_dshards * (u64) consensus_shard_size)
    FATAL("Header total_size (%zu) exceeds %u data shards of %zu bytes.",
          consensus_size, consensus_dshards, consensus_shard_size);
  /*   Kick out shards that don't match the consensus.  */
  Fi(opt.n_input_shards,
    if (res[i].dshards != consensus_dshards
     || res[i].pshards != consensus_pshards
     || res[i].total_size != consensus_size
     || res[i].shard_size != consensus_shard_size
     || res[i].shard_number >= consensus_dshards + consensus_pshards) {
      res[i].valid = false;
      if (!opt.quiet)
        xpar_fprintf(xpar_stderr,
          "Shard `%s' does not match the consensus, skipping.\n",
          opt.input_files[i]);
      if (!opt.force) xpar_exit(1);
    }
  )
  if (consensus_shard_size & 63) {
    FATAL(
      "Consensus shard size is not a multiple of 64 bytes. This\n"
      "should never happen with shards produced by this tool.\n"
    );
  }
  /*   Check if we have a duplicate of any shard.  */
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
  /*   Free the invalid buffers, compact the valid ones.  */
  Fi(opt.n_input_shards,
    if (!res[i].valid) xpar_free(res[i].buf), res[i].shard_number = 0xFF);
  int n_valid_shards = 0;
  Fi(opt.n_input_shards, if (res[i].valid) {
    n_valid_shards++;
    for (int j = i; j && res[j].shard_number < res[j - 1].shard_number; j--) {
      sharded_hv_result_t tmp = res[j];
      res[j] = res[j - 1];
      res[j - 1] = tmp;
    }
  })
  /*   Log some information.  */
  if (opt.verbose) {
    xpar_fprintf(xpar_stderr,
      "Sharding consensus: %u data shards, %u parity shards\n",
      consensus_dshards, consensus_pshards);
    xpar_fprintf(xpar_stderr,
      "Sharding consensus: %zu bytes total, %zu bytes per shard\n",
      consensus_size, consensus_shard_size);
    xpar_fprintf(xpar_stderr, "Valid shards:\n");
    Fi(n_valid_shards,
      xpar_fprintf(xpar_stderr,
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
  xpar_file * out = xpar_open(opt.output_file,
    XPAR_O_WRITE | XPAR_O_CREATE | XPAR_O_TRUNCATE);
  if (!out) FATAL_PERROR("fopen");
  if (n_valid_shards == consensus_dshards + consensus_pshards) {
    Fi(consensus_dshards,
      sz w = MIN(consensus_size, consensus_shard_size);
      xpar_xwrite(out, res[i].buf + res[i].hdr_size, w);
      consensus_size -= w)
    Fi(n_valid_shards, unmap_shard(&res[i]));
    xpar_xclose(out);
    return;
  }
  rs * r = rs_init(consensus_dshards, consensus_pshards);
  u8 * buffers[MAX_TOTAL_SHARDS] = { NULL }, pres[MAX_TOTAL_SHARDS] = { 0 };
  Fi(n_valid_shards,
    buffers[res[i].shard_number] = res[i].buf + res[i].hdr_size,
    pres[res[i].shard_number] = 1)
  Fi(consensus_dshards + consensus_pshards, if (!buffers[i]) {
    buffers[i] = SIMDSafeAllocate(consensus_shard_size);
    xpar_memset(buffers[i], 0, consensus_shard_size);
  })
  xpar_progress_t prog;
  xpar_progress_init(&prog, opt.progress,
    (u64) consensus_shard_size * (u64) consensus_dshards, "Decoding");
  if (!rs_correct(r, buffers, pres, consensus_shard_size, opt.verbose, &prog))
    FATAL("Failed to correct the data.");
  xpar_progress_end(&prog);
  Fi(consensus_dshards,
    sz w = MIN(consensus_size, consensus_shard_size);
    xpar_xwrite(out, buffers[i], w);
    consensus_size -= w)
  xpar_xclose(out);
  rs_destroy(r);
  Fi(n_valid_shards, unmap_shard(&res[i]));
  Fi(consensus_dshards + consensus_pshards,
    if (!pres[i]) SIMDSafeFree(buffers[i]));
}
/*  Dry-run of log_sharded_decode; returns invalid-or-missing count.
    Exits 1 when unrecoverable.  */
void log_sharded_test(sharded_decoding_options_t opt) {
  sharded_hv_result_t * res = xpar_malloc(MAX_TOTAL_SHARDS * sizeof(*res));
  if (opt.n_input_shards > MAX_TOTAL_SHARDS) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr,
        "Too many input shards (max %d).\n", MAX_TOTAL_SHARDS);
    xpar_free(res); xpar_exit(1);
  }
  Fi(opt.n_input_shards,
    res[i] = validate_shard_header(opt.input_files[i], opt, "XPAL");
    if (!res[i].valid && !opt.quiet)
      xpar_fprintf(xpar_stderr, "Invalid shard `%s'.\n", opt.input_files[i]);
  )
  int n_valid = 0;
  Fi(opt.n_input_shards, if (res[i].valid) n_valid++);
  shard_consensus_t sc = consensus_of_valid(res, opt.n_input_shards);
  u8 consensus_dshards = sc.dshards, consensus_pshards = sc.pshards;
  sz consensus_shard_size = sc.shard_size;
  if (n_valid < consensus_dshards) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr,
        "Only %d valid shards, need %u to recover.\n",
        n_valid, consensus_dshards);
    Fi(opt.n_input_shards, if (res[i].buf) unmap_shard(&res[i]));
    xpar_free(res);
    xpar_exit(1);
  }
  int n_total = consensus_dshards + consensus_pshards;
  int bad = n_total - n_valid;
  if (n_valid < n_total) {
    if (consensus_shard_size & 63)
      FATAL("Consensus shard size is not a multiple of 64 bytes.");
    rs * r = rs_init(consensus_dshards, consensus_pshards);
    u8 * buffers[MAX_TOTAL_SHARDS] = { NULL };
    u8 pres[MAX_TOTAL_SHARDS] = { 0 };
    Fi(opt.n_input_shards, if (res[i].valid &&
        res[i].shard_number < consensus_dshards + consensus_pshards) {
      buffers[res[i].shard_number] = res[i].buf + res[i].hdr_size;
      pres[res[i].shard_number] = 1;
    })
    Fi(n_total, if (!buffers[i]) {
      buffers[i] = SIMDSafeAllocate(consensus_shard_size);
      xpar_memset(buffers[i], 0, consensus_shard_size);
    })
    xpar_progress_t prog;
    xpar_progress_init(&prog, opt.progress,
      (u64) consensus_shard_size * (u64) consensus_dshards, "Testing");
    if (!rs_correct(r, buffers, pres, consensus_shard_size, opt.verbose,
                    &prog)) {
      if (!opt.quiet)
        xpar_fprintf(xpar_stderr, "RS reconstruction failed.\n");
      bad++;
    }
    xpar_progress_end(&prog);
    rs_destroy(r);
    Fi(n_total, if (!pres[i]) SIMDSafeFree(buffers[i]));
  }
  if (opt.verbose)
    xpar_fprintf(xpar_stderr,
      "Checked %zu shards, %d valid (need %u), %d bad.\n",
      opt.n_input_shards, n_valid, consensus_dshards, bad);
  Fi(opt.n_input_shards, if (res[i].buf) unmap_shard(&res[i]));
  xpar_free(res);
  if (bad) xpar_exit(1);
}
