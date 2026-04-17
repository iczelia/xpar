/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    Shared between blake2b.c and its SIMD variants.  */

#ifndef _BLAKE2B_IMPL_H_
#define _BLAKE2B_IMPL_H_

#include "blake2b.h"
#include "config.h"

extern const u64 blake2b_IV[8];
extern const u8 blake2b_sigma[12][16];

static inline u64 blake2b_load64(const void *restrict src) {
  const u8 *restrict p = (const u8 *)src;
  return (u64)p[0] | ((u64)p[1] << 8) | ((u64)p[2] << 16) | ((u64)p[3] << 24) |
         ((u64)p[4] << 32) | ((u64)p[5] << 40) | ((u64)p[6] << 48) |
         ((u64)p[7] << 56);
}

#if defined(HAVE_SSE41) || (defined(XPAR_AARCH64) && defined(HAVE_NEON))
void blake2b_compress_sse41(blake2b_state *restrict s,
                            const u8 *restrict block);
#endif
#if defined(HAVE_AVX2)
void blake2b_compress_avx2(blake2b_state *restrict s, const u8 *restrict block);
#endif

#endif
