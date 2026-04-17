/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    BLAKE2b: constants, ref compress, init/update/final, SIMD dispatch.  */

#include "blake2b-impl.h"

const u64 blake2b_IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
                           0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                           0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                           0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

const u8 blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

static inline void store64(void *restrict dst, u64 w) {
  u8 *restrict p = (u8 *)dst;
  p[0] = (u8)(w);
  p[1] = (u8)(w >> 8);
  p[2] = (u8)(w >> 16);
  p[3] = (u8)(w >> 24);
  p[4] = (u8)(w >> 32);
  p[5] = (u8)(w >> 40);
  p[6] = (u8)(w >> 48);
  p[7] = (u8)(w >> 56);
}
static inline u64 rotr64(u64 x, unsigned n) {
  return (x >> n) | (x << (64 - n));
}

#define G(r, i, a, b, c, d)                                                    \
  do {                                                                         \
    a = a + b + m[blake2b_sigma[r][2 * i + 0]];                                \
    d = rotr64(d ^ a, 32);                                                     \
    c = c + d;                                                                 \
    b = rotr64(b ^ c, 24);                                                     \
    a = a + b + m[blake2b_sigma[r][2 * i + 1]];                                \
    d = rotr64(d ^ a, 16);                                                     \
    c = c + d;                                                                 \
    b = rotr64(b ^ c, 63);                                                     \
  } while (0)

#define ROUND(r)                                                               \
  do {                                                                         \
    G(r, 0, v[0], v[4], v[8], v[12]);                                          \
    G(r, 1, v[1], v[5], v[9], v[13]);                                          \
    G(r, 2, v[2], v[6], v[10], v[14]);                                         \
    G(r, 3, v[3], v[7], v[11], v[15]);                                         \
    G(r, 4, v[0], v[5], v[10], v[15]);                                         \
    G(r, 5, v[1], v[6], v[11], v[12]);                                         \
    G(r, 6, v[2], v[7], v[8], v[13]);                                          \
    G(r, 7, v[3], v[4], v[9], v[14]);                                          \
  } while (0)

static void blake2b_compress_ref(blake2b_state *restrict s,
                                 const u8 *restrict block) {
  u64 m[16], v[16];
  for (int i = 0; i < 16; i++)
    m[i] = blake2b_load64(block + i * 8);
  for (int i = 0; i < 8; i++)
    v[i] = s->h[i];
  v[8] = blake2b_IV[0];
  v[9] = blake2b_IV[1];
  v[10] = blake2b_IV[2];
  v[11] = blake2b_IV[3];
  v[12] = s->t[0] ^ blake2b_IV[4];
  v[13] = s->t[1] ^ blake2b_IV[5];
  v[14] = s->f[0] ^ blake2b_IV[6];
  v[15] = s->f[1] ^ blake2b_IV[7];
  ROUND(0);
  ROUND(1);
  ROUND(2);
  ROUND(3);
  ROUND(4);
  ROUND(5);
  ROUND(6);
  ROUND(7);
  ROUND(8);
  ROUND(9);
  ROUND(10);
  ROUND(11);
  for (int i = 0; i < 8; i++)
    s->h[i] ^= v[i] ^ v[i + 8];
}

typedef void (*blake2b_compress_fn)(blake2b_state *restrict s,
                                    const u8 *restrict block);

#if defined(XPAR_X86_64)
#if defined(HAVE_FUNC_ATTRIBUTE_SYSV_ABI)
#define EXTERNAL_ABI __attribute__((sysv_abi))
#else
#define EXTERNAL_ABI
#endif
extern EXTERNAL_ABI int xpar_x86_64_cpuflags(void);
extern EXTERNAL_ABI int xpar_leo_x86_64_cpuflags(void);
#endif

static blake2b_compress_fn select_compress(void) {
#if defined(XPAR_X86_64)
#ifdef HAVE_AVX2
  if (xpar_leo_x86_64_cpuflags() & 1)
    return blake2b_compress_avx2;
#endif
#ifdef HAVE_SSE41
  if (xpar_x86_64_cpuflags() & 1)
    return blake2b_compress_sse41;
#endif
  return blake2b_compress_ref;
#elif defined(XPAR_AARCH64) && defined(HAVE_NEON)
  return blake2b_compress_sse41;
#else
  return blake2b_compress_ref;
#endif
}

static void blake2b_compress(blake2b_state *restrict s,
                             const u8 *restrict block) {
  static blake2b_compress_fn fn = NULL;
  if (!fn)
    fn = select_compress();
  fn(s, block);
}

int blake2b_init_key(blake2b_state *restrict s, sz outlen,
                     const void *restrict key, sz keylen) {
  if (!outlen || outlen > BLAKE2B_OUTBYTES)
    return -1;
  if (keylen > BLAKE2B_KEYBYTES)
    return -1;
  if (keylen && !key)
    return -1;
  xpar_memset(s, 0, sizeof(*s));
  for (int i = 0; i < 8; i++)
    s->h[i] = blake2b_IV[i];
  u64 param0 =
      (u64)outlen | ((u64)keylen << 8) | ((u64)1 << 16) | ((u64)1 << 24);
  s->h[0] ^= param0;
  s->outlen = outlen;
  if (keylen) {
    u8 block[BLAKE2B_BLOCKBYTES];
    xpar_memset(block, 0, BLAKE2B_BLOCKBYTES);
    xpar_memcpy(block, key, keylen);
    blake2b_update(s, block, BLAKE2B_BLOCKBYTES);
  }
  return 0;
}
int blake2b_init(blake2b_state *restrict s, sz outlen) {
  return blake2b_init_key(s, outlen, NULL, 0);
}

int blake2b_update(blake2b_state *restrict s, const void *restrict in,
                   sz inlen) {
  const u8 *restrict p = (const u8 *)in;
  if (!inlen)
    return 0;
  sz left = BLAKE2B_BLOCKBYTES - s->buflen;
  if (inlen > left) {
    xpar_memcpy(s->buf + s->buflen, p, left);
    s->t[0] += BLAKE2B_BLOCKBYTES;
    if (s->t[0] < BLAKE2B_BLOCKBYTES)
      s->t[1]++;
    blake2b_compress(s, s->buf);
    s->buflen = 0;
    p += left;
    inlen -= left;
    while (inlen > BLAKE2B_BLOCKBYTES) {
      s->t[0] += BLAKE2B_BLOCKBYTES;
      if (s->t[0] < BLAKE2B_BLOCKBYTES)
        s->t[1]++;
      blake2b_compress(s, p);
      p += BLAKE2B_BLOCKBYTES;
      inlen -= BLAKE2B_BLOCKBYTES;
    }
  }
  xpar_memcpy(s->buf + s->buflen, p, inlen);
  s->buflen += inlen;
  return 0;
}

int blake2b_final(blake2b_state *restrict s, void *restrict out, sz outlen) {
  if (outlen < s->outlen)
    return -1;
  if (s->f[0])
    return -1;
  s->t[0] += s->buflen;
  if (s->t[0] < s->buflen)
    s->t[1]++;
  s->f[0] = (u64)-1;
  xpar_memset(s->buf + s->buflen, 0, BLAKE2B_BLOCKBYTES - s->buflen);
  blake2b_compress(s, s->buf);
  u8 hash[BLAKE2B_OUTBYTES];
  for (int i = 0; i < 8; i++)
    store64(hash + i * 8, s->h[i]);
  xpar_memcpy(out, hash, s->outlen);
  xpar_memset(hash, 0, sizeof(hash));
  return 0;
}

int blake2b(void *restrict out, sz outlen, const void *restrict in, sz inlen,
            const void *restrict key, sz keylen) {
  blake2b_state s;
  if (blake2b_init_key(&s, outlen, key, keylen) < 0)
    return -1;
  blake2b_update(&s, in, inlen);
  return blake2b_final(&s, out, outlen);
}
