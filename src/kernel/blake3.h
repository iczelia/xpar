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

/*  Streaming BLAKE3 interface.  */

#ifndef XPAR_BLAKE3_H
#define XPAR_BLAKE3_H

#include "common.h"

#define XPAR_BLAKE3_OUT_LEN    32
#define XPAR_BLAKE3_KEY_LEN    32
#define XPAR_BLAKE3_BLOCK_LEN  64
#define XPAR_BLAKE3_CHUNK_LEN  1024

/*  An input is at most 2^64 bytes, so at most 2^54 chunks, so the stack
    of pending subtree chaining values never exceeds 54 entries plus the
    one being pushed.  */
#define XPAR_BLAKE3_MAX_DEPTH  54

/*  Lanes of the widest hash_many built here (AVX2). A wider kernel must
    raise this, or the subtree compressor overruns its chaining-value
    array; blake3.c asserts the dispatched degree against it.  */
#define XPAR_BLAKE3_MAX_DEGREE 8

/*  C99 forbids repeating a typedef, so every declarer shares a guard.  */
#ifndef XPAR_POOL_TYPEDEF
#define XPAR_POOL_TYPEDEF
typedef struct xpar_pool xpar_pool;
#endif

typedef struct {
  u32 cv[8];
  u64 counter;              /*  Index of the chunk being filled.  */
  u8  buf[XPAR_BLAKE3_BLOCK_LEN];
  u8  buf_len;              /*  Bytes buffered, 0..64.  */
  u8  blocks;               /*  Blocks of this chunk absorbed, 0..15.  */
  u8  flags;                /*  Mode flags: keyed / derive-key material.  */
} xpar_blake3_chunk;

typedef struct {
  u32 key[8];
  xpar_blake3_chunk chunk;
  u64 chunk_base;             /*  first counter in a standalone subtree  */
  u8  stack_len;
  u8  stack[(XPAR_BLAKE3_MAX_DEPTH + 1) * XPAR_BLAKE3_OUT_LEN];
} xpar_blake3_t;

void xpar_blake3_init      (xpar_blake3_t *);
void xpar_blake3_init_keyed(xpar_blake3_t *, const u8 * key);
void xpar_blake3_init_derive_key(xpar_blake3_t *, const char * context);

void xpar_blake3_update(xpar_blake3_t *, const void * buf, sz n);
void xpar_blake3_update_parallel(xpar_blake3_t *, const void * buf, sz n,
                                 xpar_pool *);
void xpar_blake3_final(const xpar_blake3_t *, u8 * out, sz n);

/*  Extendable output past 32 bytes.  */
void xpar_blake3_final_seek(const xpar_blake3_t *, u64 seek, u8 * out, sz n);

/*  One-shots. `n` may be anything.  */
void xpar_blake3_hash      (const void * buf, sz len, u8 * out, sz n);
void xpar_blake3_hash_keyed(const u8 * key, const void * buf, sz len,
                            u8 * out, sz n);

/*  Chaining-value tag of one complete BLAKE3 subtree.  */
void xpar_blake3_subtree_tag(const void * buf, sz len, u64 chunk_counter,
                             u8 * out, sz n);
void xpar_blake3_subtree_tag_keyed(const u8 * key, const void * buf, sz len,
                                   u64 chunk_counter, u8 * out, sz n);

/*  Streaming form of the subtree tag for repair gates whose slice is larger
    than the memory budget.  */
void xpar_blake3_subtree_stream_init(xpar_blake3_t *, const u8 * key,
                                     u64 chunk_counter);
void xpar_blake3_subtree_stream_final(const xpar_blake3_t *, u8 * out, sz n);

/*  Subkey derivation.  */
void xpar_blake3_derive_key(const char * context, const void * material,
                            sz len, u8 * out);

/*  Compare a tag against a stored one.  */
static inline bool xpar_blake3_tag_equal(const void * a, const void * b,
                                         sz n) {
  return xpar_ct_equal(a, b, n);
}

/*  Name of the dispatched kernel, for `info` and the benchmark.  */
const char * xpar_blake3_variant(void);

/*  Internal: the per-ISA kernels, dispatched by blake3.c.  */
extern const u32 xpar_blake3_iv[8];

/*  One 64-byte block, folded into `cv` in place.  */
void xpar_blake3_compress_scalar(u32 * cv, const u8 * block, u8 block_len,
                                 u64 counter, u8 flags);

/*  One 64-byte block, producing the full 64-byte output word set rather
    than a chaining value. Only the root node needs it, and only it can
    extend the output past 32 bytes.  */
void xpar_blake3_xof_scalar(const u32 * cv, const u8 * block, u8 block_len,
                            u64 counter, u8 flags, u8 * out);

/*  hash_many compresses `count` inputs of `blocks` full 64-byte blocks
    each, in parallel lanes, all from the same key and flags, writing one
    32-byte chaining value per input.  */
#define XPAR_BLAKE3_HASH_MANY_DECL(sfx)                                       \
  void xpar_blake3_hash_many##sfx(const u8 * const * inputs, sz count,        \
                                  sz blocks, const u32 * key, u64 counter,   \
                                  bool inc, u8 flags, u8 first, u8 last,     \
                                  u8 * out)

XPAR_BLAKE3_HASH_MANY_DECL(_scalar);
#ifdef HAVE_AVX2
XPAR_BLAKE3_HASH_MANY_DECL(_avx2);
#endif
#ifdef HAVE_NEON
XPAR_BLAKE3_HASH_MANY_DECL(_neon);
#endif

#endif
