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

/*  Streaming BLAKE3 and kernel dispatch.  */

#include "blake3.h"

#include "platform/port-cpu.h"
#include "platform/port-thread.h"

/*  BLAKE3's SHA-2-derived initialisation vector.  */
const u32 xpar_blake3_iv[8] = {
  0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
  0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

/*  Compression-position and keying-mode domain flags.  */
#define XPAR_B3_CHUNK_START  (1u << 0)
#define XPAR_B3_CHUNK_END    (1u << 1)
#define XPAR_B3_PARENT       (1u << 2)
#define XPAR_B3_ROOT         (1u << 3)
#define XPAR_B3_KEYED_HASH   (1u << 4)
#define XPAR_B3_DERIVE_CTX   (1u << 5)
#define XPAR_B3_DERIVE_MAT   (1u << 6)

/*  Kernel dispatch.  */

typedef void (* xpar_b3_many_fn)(const u8 * const * inputs, sz count,
                                 sz blocks, const u32 * key, u64 counter,
                                 bool inc, u8 flags, u8 first, u8 last,
                                 u8 * out);

static xpar_b3_many_fn xpar_b3_many = xpar_blake3_hash_many_scalar;
static const char *    xpar_b3_name = "scalar";
static sz              xpar_b3_deg  = 1;
static bool            xpar_b3_ready;
static u32             xpar_b3_seen;   /*  Features the table was built on.  */

/*  The transposed lane loads in the SIMD kernels read 32-bit words with
    vector loads, so they are little-endian only. A big-endian host takes
    the scalar kernel, which goes through xpar_rd32 and is correct
    everywhere.  */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) &&               \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  #define XPAR_B3_HOST_BE 1
#endif

static void xpar_b3_dispatch(u32 f) {
  xpar_b3_many = xpar_blake3_hash_many_scalar;
  xpar_b3_name = "scalar";  xpar_b3_deg = 1;
#if !defined(XPAR_B3_HOST_BE)
  #ifdef HAVE_AVX2
  if (f & XPAR_CPU_AVX2) {
    xpar_b3_many = xpar_blake3_hash_many_avx2;
    xpar_b3_name = "avx2";  xpar_b3_deg = 8;
  }
  #endif
  #ifdef HAVE_NEON
  if (f & XPAR_CPU_NEON) {
    xpar_b3_many = xpar_blake3_hash_many_neon;
    xpar_b3_name = "neon";  xpar_b3_deg = 4;
  }
  #endif
#endif
  xpar_assert(xpar_b3_deg <= XPAR_BLAKE3_MAX_DEGREE);
  xpar_b3_seen = f;  xpar_b3_ready = true;
}

/*  Re-selects when xpar_cpu_force has narrowed the feature set since the
    last call, which is what --simd and the benchmark's tier sweep do. Once
    per hasher, never inside a loop.  */
static void xpar_b3_need_dispatch(void) {
  u32 f = xpar_cpu_features();
  if (!xpar_b3_ready || f != xpar_b3_seen) xpar_b3_dispatch(f);
}

const char * xpar_blake3_variant(void) {
  xpar_b3_need_dispatch();
  return xpar_b3_name;
}

sz xpar_blake3_degree(void) {
  xpar_b3_need_dispatch();
  return xpar_b3_deg;
}

/*  Chunk state.  */

static void xpar_b3_chunk_reset(xpar_blake3_chunk * c, const u32 * key,
                                u64 counter) {
  Fi(8, c->cv[i] = key[i])
  c->counter = counter;  c->buf_len = 0;  c->blocks = 0;
  xpar_memset(c->buf, 0, sizeof(c->buf));
}

static sz xpar_b3_chunk_len(const xpar_blake3_chunk * c) {
  return (sz) XPAR_BLAKE3_BLOCK_LEN * c->blocks + c->buf_len;
}

/*  CHUNK_START rides on the first block of a chunk and nothing else, so
    it is a function of how many blocks this chunk has already taken.  */
static u8 xpar_b3_start_flag(const xpar_blake3_chunk * c) {
  return c->blocks == 0 ? (u8) XPAR_B3_CHUNK_START : 0;
}

static sz xpar_b3_fill(xpar_blake3_chunk * c, const u8 * p, sz n) {
  sz take = XPAR_BLAKE3_BLOCK_LEN - c->buf_len;
  if (take > n) take = n;
  xpar_memcpy(c->buf + c->buf_len, p, take);
  c->buf_len = (u8) (c->buf_len + take);
  return take;
}

/*  The last block of a chunk is never compressed here: it may turn out to
    be the root, and the root needs a flag the finaliser owns. So a full
    buffer is only flushed once more input proves it is not the last.  */
static void xpar_b3_chunk_update(xpar_blake3_chunk * c, const u8 * p, sz n) {
  if (c->buf_len > 0) {
    sz take = xpar_b3_fill(c, p, n);
    p += take;  n -= take;
    if (n > 0) {
      xpar_blake3_compress_scalar(c->cv, c->buf, XPAR_BLAKE3_BLOCK_LEN,
                                  c->counter,
                                  (u8) (c->flags | xpar_b3_start_flag(c)));
      c->blocks++;  c->buf_len = 0;
      xpar_memset(c->buf, 0, sizeof(c->buf));
    }
  }
  while (n > XPAR_BLAKE3_BLOCK_LEN) {
    xpar_blake3_compress_scalar(c->cv, p, XPAR_BLAKE3_BLOCK_LEN, c->counter,
                                (u8) (c->flags | xpar_b3_start_flag(c)));
    c->blocks++;
    p += XPAR_BLAKE3_BLOCK_LEN;  n -= XPAR_BLAKE3_BLOCK_LEN;
  }
  xpar_b3_fill(c, p, n);
}

/*  Node output.  */

typedef struct {
  u32 cv[8];
  u8  block[XPAR_BLAKE3_BLOCK_LEN];
  u64 counter;
  u8  block_len;
  u8  flags;
} xpar_b3_node;

static void xpar_b3_chunk_node(const xpar_blake3_chunk * c,
                               xpar_b3_node * o) {
  Fi(8, o->cv[i] = c->cv[i])
  xpar_memcpy(o->block, c->buf, XPAR_BLAKE3_BLOCK_LEN);
  o->counter   = c->counter;
  o->block_len = c->buf_len;
  o->flags     = (u8) (c->flags | xpar_b3_start_flag(c) | XPAR_B3_CHUNK_END);
}

/*  A parent's block is its two children's chaining values, and its
    counter is always zero: position in the tree is carried by the PARENT
    flag and by the children, not by a counter.  */
static void xpar_b3_parent_node(const u8 * block, const u32 * key, u8 flags,
                                xpar_b3_node * o) {
  Fi(8, o->cv[i] = key[i])
  xpar_memcpy(o->block, block, XPAR_BLAKE3_BLOCK_LEN);
  o->counter   = 0;
  o->block_len = XPAR_BLAKE3_BLOCK_LEN;
  o->flags     = (u8) (flags | XPAR_B3_PARENT);
}

static void xpar_b3_node_cv(const xpar_b3_node * o, u8 * out) {
  u32 cv[8];
  Fi(8, cv[i] = o->cv[i])
  xpar_blake3_compress_scalar(cv, o->block, o->block_len, o->counter,
                              o->flags);
  Fi(8, xpar_wr32(out + 4 * i, cv[i]))
}

/*  Output bytes from the root node. The block counter is reused as the
    output block index, which is what makes the output an XOF: block i of
    the stream is one compression with counter i.  */
static void xpar_b3_root(const xpar_b3_node * o, u64 seek, u8 * out, sz n) {
  u8 wide[XPAR_BLAKE3_BLOCK_LEN];
  u64 blk = seek / XPAR_BLAKE3_BLOCK_LEN;
  sz  off = (sz) (seek % XPAR_BLAKE3_BLOCK_LEN);
  while (n > 0) {
    sz take = sizeof(wide) - off;
    if (take > n) take = n;
    xpar_blake3_xof_scalar(o->cv, o->block, o->block_len, blk,
                           (u8) (o->flags | XPAR_B3_ROOT), wide);
    xpar_memcpy(out, wide + off, take);
    out += take;  n -= take;  off = 0;  blk++;
  }
}

/*  Wide subtree compression.  */

static sz xpar_b3_chunks_wide(const u8 * in, sz len, const u32 * key,
                              u64 counter, u8 flags, u8 * out) {
  const u8 * chunks[XPAR_BLAKE3_MAX_DEGREE];
  sz pos = 0, count = 0;
  xpar_assert(len <= XPAR_BLAKE3_MAX_DEGREE * XPAR_BLAKE3_CHUNK_LEN);
  while (len - pos >= XPAR_BLAKE3_CHUNK_LEN) {
    chunks[count++] = in + pos;
    pos += XPAR_BLAKE3_CHUNK_LEN;
  }
  xpar_b3_many(chunks, count,
               XPAR_BLAKE3_CHUNK_LEN / XPAR_BLAKE3_BLOCK_LEN, key, counter,
               true, flags, XPAR_B3_CHUNK_START, XPAR_B3_CHUNK_END, out);
  if (len > pos) {
    /*  A short trailing chunk cannot be batched: its last block is not a
        full one.  */
    xpar_blake3_chunk c;
    xpar_b3_node o;
    c.flags = flags;
    xpar_b3_chunk_reset(&c, key, counter + count);
    xpar_b3_chunk_update(&c, in + pos, len - pos);
    xpar_b3_chunk_node(&c, &o);
    xpar_b3_node_cv(&o, out + count * XPAR_BLAKE3_OUT_LEN);
    count++;
  }
  return count;
}

static sz xpar_b3_parents_wide(const u8 * cvs, sz count, const u32 * key,
                               u8 flags, u8 * out) {
  const u8 * parents[XPAR_BLAKE3_MAX_DEGREE];
  sz np = 0;
  while (count - 2 * np >= 2) {
    parents[np] = cvs + 2 * np * XPAR_BLAKE3_OUT_LEN;
    np++;
  }
  xpar_b3_many(parents, np, 1, key, 0, false, (u8) (flags | XPAR_B3_PARENT),
               0, 0, out);
  /*  An odd child has no sibling at this level and rises unchanged; the
      binary-counter shape of the tree puts it at the next level up.  */
  if (count > 2 * np) {
    xpar_memcpy(out + np * XPAR_BLAKE3_OUT_LEN,
                cvs + 2 * np * XPAR_BLAKE3_OUT_LEN, XPAR_BLAKE3_OUT_LEN);
    np++;
  }
  return np;
}

/*  Largest power-of-two multiple of the chunk length strictly inside
    `len`, which is BLAKE3's left subtree.  */
static sz xpar_b3_left_len(sz len) {
  u64 chunks = (u64) ((len - 1) / XPAR_BLAKE3_CHUNK_LEN);
  return (sz) (((u64) 1 << xpar_log2_floor(chunks)) *
               XPAR_BLAKE3_CHUNK_LEN);
}

static sz xpar_b3_compress_wide(const u8 * in, sz len, const u32 * key,
                                u64 counter, u8 flags, sz deg, u8 * out) {
  sz left, right, ln, rn;
  if (len <= deg * XPAR_BLAKE3_CHUNK_LEN)
    return xpar_b3_chunks_wide(in, len, key, counter, flags, out);
  left = xpar_b3_left_len(len);  right = len - left;
  /*  A parent needs two children, so above the chunk level the halves are
      placed at least two apart even on a one-lane kernel. At the chunk
      level the two single chaining values are already adjacent, and the
      `ln == 1` return below hands them over as a pair.  */
  if (deg == 1 && left > XPAR_BLAKE3_CHUNK_LEN) deg = 2;
  {
    u8 cvs[2 * XPAR_BLAKE3_MAX_DEGREE * XPAR_BLAKE3_OUT_LEN];
    u8 * rcvs = cvs + deg * XPAR_BLAKE3_OUT_LEN;
    ln = xpar_b3_compress_wide(in, left, key, counter, flags, deg, cvs);
    rn = xpar_b3_compress_wide(in + left, right, key,
                               counter + left / XPAR_BLAKE3_CHUNK_LEN, flags,
                               deg, rcvs);
    /*  Degree one: both halves are single chunks and are already the two
        children the caller needs.  */
    if (ln == 1) {
      xpar_memcpy(out, cvs, 2 * XPAR_BLAKE3_OUT_LEN);
      return 2;
    }
    return xpar_b3_parents_wide(cvs, ln + rn, key, flags, out);
  }
}

/*  Reduce a whole subtree to the two chaining values its parent needs.  */
static void xpar_b3_subtree_node(const u8 * in, sz len, const u32 * key,
                                 u64 counter, u8 flags, u8 * out) {
  u8 cvs[2 * XPAR_BLAKE3_MAX_DEGREE * XPAR_BLAKE3_OUT_LEN];
  u8 up[XPAR_BLAKE3_MAX_DEGREE * XPAR_BLAKE3_OUT_LEN];
  sz n = xpar_b3_compress_wide(in, len, key, counter, flags, xpar_b3_deg,
                               cvs);
  while (n > 2) {
    n = xpar_b3_parents_wide(cvs, n, key, flags, up);
    xpar_memcpy(cvs, up, n * XPAR_BLAKE3_OUT_LEN);
  }
  xpar_memcpy(out, cvs, 2 * XPAR_BLAKE3_OUT_LEN);
}

static void subtree_tag(const u8 * key_bytes, const void * buf, sz len,
                        u64 chunk_counter, u8 * out, sz n) {
  const u8 * p = (const u8 *) buf;
  u32 key[8];
  u8 flags = key_bytes ? XPAR_B3_KEYED_HASH : 0;
  u8 cv[XPAR_BLAKE3_OUT_LEN];
  u32 i;
  xpar_b3_need_dispatch();
  xpar_assert(len >= XPAR_BLAKE3_CHUNK_LEN);
  xpar_assert(len % XPAR_BLAKE3_CHUNK_LEN == 0);
  xpar_assert((len & (len - 1)) == 0);
  xpar_assert(n <= XPAR_BLAKE3_OUT_LEN);
  for (i = 0; i < 8; i++)
    key[i] = key_bytes ? xpar_rd32(key_bytes + 4 * i) : xpar_blake3_iv[i];
  if (len == XPAR_BLAKE3_CHUNK_LEN) {
    xpar_blake3_chunk c;
    xpar_b3_node node;
    c.flags = flags;
    xpar_b3_chunk_reset(&c, key, chunk_counter);
    xpar_b3_chunk_update(&c, p, len);
    xpar_b3_chunk_node(&c, &node);
    xpar_b3_node_cv(&node, cv);
  } else {
    u8 pair[2 * XPAR_BLAKE3_OUT_LEN];
    xpar_b3_node node;
    xpar_b3_subtree_node(p, len, key, chunk_counter, flags, pair);
    xpar_b3_parent_node(pair, key, flags, &node);
    xpar_b3_node_cv(&node, cv);
  }
  xpar_memcpy(out, cv, n);
  xpar_secure_zero(key, sizeof key);
}

void xpar_blake3_subtree_tag(const void * buf, sz len, u64 chunk_counter,
                             u8 * out, sz n) {
  subtree_tag(NULL, buf, len, chunk_counter, out, n);
}

void xpar_blake3_subtree_tag_keyed(const u8 * key, const void * buf, sz len,
                                   u64 chunk_counter, u8 * out, sz n) {
  subtree_tag(key, buf, len, chunk_counter, out, n);
}

/*  The hasher.  */

static int xpar_b3_popcount(u64 v) {
  int n = 0;
  while (v) { n += (int) (v & 1);  v >>= 1; }
  return n;
}

/*  The stack holds one chaining value per set bit of the number of chunks
    already absorbed, so after absorbing up to `total`, merging down to
    that population count is exactly the set of subtrees that have just
    been completed.  */
static void xpar_b3_merge(xpar_blake3_t * h, u64 total) {
  int want;
  xpar_assert(total >= h->chunk_base);
  want = xpar_b3_popcount(total - h->chunk_base);
  while ((int) h->stack_len > want) {
    u8 * pair = h->stack + (h->stack_len - 2) * XPAR_BLAKE3_OUT_LEN;
    xpar_b3_node o;
    xpar_b3_parent_node(pair, h->key, h->chunk.flags, &o);
    xpar_b3_node_cv(&o, pair);
    h->stack_len--;
  }
}

static void xpar_b3_push(xpar_blake3_t * h, const u8 * cv, u64 counter) {
  xpar_b3_merge(h, counter);
  xpar_assert(h->stack_len <= XPAR_BLAKE3_MAX_DEPTH);
  xpar_memcpy(h->stack + h->stack_len * XPAR_BLAKE3_OUT_LEN, cv,
              XPAR_BLAKE3_OUT_LEN);
  h->stack_len++;
}

#define XPAR_B3_MT_SUBTREE ((sz) 256 * 1024)

typedef struct {
  const u8 * in;
  sz         span;
  u64        counter;
  const u32 * key;
  u8         flags;
  u8 *       cv;
} xpar_b3_parallel;

static void xpar_b3_parallel_one(sz index, void * opaque) {
  xpar_b3_parallel * p = (xpar_b3_parallel *) opaque;
  const u8 * in = p->in + index * p->span;
  u64 counter = p->counter +
                (u64) index * p->span / XPAR_BLAKE3_CHUNK_LEN;
  u8 * cv = p->cv + index * XPAR_BLAKE3_OUT_LEN;
  if (p->span == XPAR_BLAKE3_CHUNK_LEN) {
    xpar_blake3_chunk c;
    xpar_b3_node node;
    c.flags = p->flags;
    xpar_b3_chunk_reset(&c, p->key, counter);
    xpar_b3_chunk_update(&c, in, p->span);
    xpar_b3_chunk_node(&c, &node);
    xpar_b3_node_cv(&node, cv);
  } else {
    u8 pair[2 * XPAR_BLAKE3_OUT_LEN];
    xpar_b3_node node;
    xpar_b3_subtree_node(in, p->span, p->key, counter, p->flags, pair);
    xpar_b3_parent_node(pair, p->key, p->flags, &node);
    xpar_b3_node_cv(&node, cv);
  }
}

/*  Absorb one aligned, complete subtree.  */
static void xpar_b3_absorb_parallel(xpar_blake3_t * h, const u8 * p, sz sub,
                                    xpar_pool * pool) {
  sz jobs, span, i;
  u8 * cv;
  xpar_b3_parallel task;
  int threads = xpar_pool_threads(pool);
  if (threads <= 1 || sub < 2 * XPAR_B3_MT_SUBTREE) {
    u8 pair[2 * XPAR_BLAKE3_OUT_LEN];
    u64 chunks = sub / XPAR_BLAKE3_CHUNK_LEN;
    xpar_b3_subtree_node(p, sub, h->key, h->chunk.counter,
                         h->chunk.flags, pair);
    xpar_b3_push(h, pair, h->chunk.counter);
    xpar_b3_push(h, pair + XPAR_BLAKE3_OUT_LEN,
                 h->chunk.counter + chunks / 2);
    return;
  }
  jobs = 1;
  while (jobs < (sz) threads && jobs * 2 * XPAR_B3_MT_SUBTREE <= sub)
    jobs *= 2;
  span = sub / jobs;
  cv = (u8 *) xpar_alloc_raw(jobs * XPAR_BLAKE3_OUT_LEN);
  task.in = p;  task.span = span;  task.counter = h->chunk.counter;
  task.key = h->key;  task.flags = h->chunk.flags;  task.cv = cv;
  xpar_pool_run(pool, jobs, xpar_b3_parallel_one, &task);
  for (i = 0; i < jobs; i++)
    xpar_b3_push(h, cv + i * XPAR_BLAKE3_OUT_LEN,
                 h->chunk.counter +
                   (u64) i * span / XPAR_BLAKE3_CHUNK_LEN);
  xpar_free(cv);
}

static void xpar_b3_hasher_init(xpar_blake3_t * h, const u32 * key,
                                u8 flags) {
  xpar_b3_need_dispatch();
  Fi(8, h->key[i] = key[i])
  h->chunk.flags = flags;
  h->chunk_base = 0;
  xpar_b3_chunk_reset(&h->chunk, key, 0);
  h->stack_len = 0;
}

void xpar_blake3_init(xpar_blake3_t * h) {
  xpar_b3_hasher_init(h, xpar_blake3_iv, 0);
}

void xpar_blake3_init_keyed(xpar_blake3_t * h, const u8 * key) {
  u32 k[8];
  Fi(8, k[i] = xpar_rd32(key + 4 * i))
  xpar_b3_hasher_init(h, k, XPAR_B3_KEYED_HASH);
}

void xpar_blake3_subtree_stream_init(xpar_blake3_t * h, const u8 * key,
                                     u64 chunk_counter) {
  u32 k[8];
  u32 i;
  for (i = 0; i < 8; i++)
    k[i] = key ? xpar_rd32(key + 4 * i) : xpar_blake3_iv[i];
  xpar_b3_hasher_init(h, k, key ? XPAR_B3_KEYED_HASH : 0);
  h->chunk_base = chunk_counter;
  xpar_b3_chunk_reset(&h->chunk, h->key, chunk_counter);
  xpar_secure_zero(k, sizeof k);
}

void xpar_blake3_init_derive_key(xpar_blake3_t * h, const char * context) {
  xpar_blake3_t ctx;
  u8 sub[XPAR_BLAKE3_KEY_LEN];
  u32 k[8];
  xpar_b3_hasher_init(&ctx, xpar_blake3_iv, XPAR_B3_DERIVE_CTX);
  xpar_blake3_update(&ctx, context, xpar_strlen(context));
  xpar_blake3_final(&ctx, sub, sizeof(sub));
  Fi(8, k[i] = xpar_rd32(sub + 4 * i))
  xpar_b3_hasher_init(h, k, XPAR_B3_DERIVE_MAT);
}

void xpar_blake3_update(xpar_blake3_t * h, const void * buf, sz n) {
  const u8 * p = (const u8 *) buf;
  if (n == 0) return;
  /*  Finish the chunk in progress first, but only compress it once more
      input has proved it is not the last one.  */
  if (xpar_b3_chunk_len(&h->chunk) > 0) {
    sz take = XPAR_BLAKE3_CHUNK_LEN - xpar_b3_chunk_len(&h->chunk);
    if (take > n) take = n;
    xpar_b3_chunk_update(&h->chunk, p, take);
    p += take;  n -= take;
    if (n == 0) return;
    {
      xpar_b3_node o;
      u8 cv[XPAR_BLAKE3_OUT_LEN];
      xpar_b3_chunk_node(&h->chunk, &o);
      xpar_b3_node_cv(&o, cv);
      xpar_b3_push(h, cv, h->chunk.counter);
      xpar_b3_chunk_reset(&h->chunk, h->key, h->chunk.counter + 1);
    }
  }
  while (n > XPAR_BLAKE3_CHUNK_LEN) {
    u64 sub = (u64) 1 << xpar_log2_floor((u64) n);
    u64 done = (h->chunk.counter - h->chunk_base) *
               XPAR_BLAKE3_CHUNK_LEN;
    u64 nchunks;
    while ((sub - 1) & done) sub /= 2;
    nchunks = sub / XPAR_BLAKE3_CHUNK_LEN;
    if (sub <= XPAR_BLAKE3_CHUNK_LEN) {
      xpar_blake3_chunk c;
      xpar_b3_node o;
      u8 cv[XPAR_BLAKE3_OUT_LEN];
      c.flags = h->chunk.flags;
      xpar_b3_chunk_reset(&c, h->key, h->chunk.counter);
      xpar_b3_chunk_update(&c, p, (sz) sub);
      xpar_b3_chunk_node(&c, &o);
      xpar_b3_node_cv(&o, cv);
      xpar_b3_push(h, cv, h->chunk.counter);
    } else {
      u8 pair[2 * XPAR_BLAKE3_OUT_LEN];
      xpar_b3_subtree_node(p, (sz) sub, h->key, h->chunk.counter,
                           h->chunk.flags, pair);
      xpar_b3_push(h, pair, h->chunk.counter);
      xpar_b3_push(h, pair + XPAR_BLAKE3_OUT_LEN,
                   h->chunk.counter + nchunks / 2);
    }
    h->chunk.counter += nchunks;
    p += (sz) sub;  n -= (sz) sub;
  }
  if (n > 0) {
    xpar_b3_chunk_update(&h->chunk, p, n);
    xpar_b3_merge(h, h->chunk.counter);
  }
}

void xpar_blake3_update_parallel(xpar_blake3_t * h, const void * buf, sz n,
                                 xpar_pool * pool) {
  const u8 * p = (const u8 *) buf;
  if (!pool || xpar_pool_threads(pool) <= 1 || n < 2 * XPAR_B3_MT_SUBTREE) {
    xpar_blake3_update(h, buf, n);
    return;
  }
  /*  Finish an incomplete chunk exactly as update() does.  */
  if (xpar_b3_chunk_len(&h->chunk) > 0) {
    sz take = XPAR_BLAKE3_CHUNK_LEN - xpar_b3_chunk_len(&h->chunk);
    if (take > n) take = n;
    xpar_b3_chunk_update(&h->chunk, p, take);
    p += take;  n -= take;
    if (n == 0) return;
    {
      xpar_b3_node o;
      u8 cv[XPAR_BLAKE3_OUT_LEN];
      xpar_b3_chunk_node(&h->chunk, &o);
      xpar_b3_node_cv(&o, cv);
      xpar_b3_push(h, cv, h->chunk.counter);
      xpar_b3_chunk_reset(&h->chunk, h->key, h->chunk.counter + 1);
    }
  }
  while (n > XPAR_BLAKE3_CHUNK_LEN) {
    u64 sub = (u64) 1 << xpar_log2_floor((u64) n);
    u64 done = (h->chunk.counter - h->chunk_base) *
               XPAR_BLAKE3_CHUNK_LEN;
    u64 chunks;
    while ((sub - 1) & done) sub /= 2;
    chunks = sub / XPAR_BLAKE3_CHUNK_LEN;
    if (sub <= XPAR_BLAKE3_CHUNK_LEN) {
      xpar_blake3_chunk c;
      xpar_b3_node o;
      u8 cv[XPAR_BLAKE3_OUT_LEN];
      c.flags = h->chunk.flags;
      xpar_b3_chunk_reset(&c, h->key, h->chunk.counter);
      xpar_b3_chunk_update(&c, p, (sz) sub);
      xpar_b3_chunk_node(&c, &o);
      xpar_b3_node_cv(&o, cv);
      xpar_b3_push(h, cv, h->chunk.counter);
      h->chunk.counter++;
    } else {
      xpar_b3_absorb_parallel(h, p, (sz) sub, pool);
      h->chunk.counter += chunks;
    }
    p += (sz) sub;  n -= (sz) sub;
  }
  if (n) xpar_blake3_update(h, p, n);
}

void xpar_blake3_final_seek(const xpar_blake3_t * h, u64 seek, u8 * out,
                            sz n) {
  xpar_b3_node o;
  sz left;
  if (n == 0) return;
  if (h->stack_len == 0) {
    xpar_b3_chunk_node(&h->chunk, &o);
    xpar_b3_root(&o, seek, out, n);
    return;
  }
  if (xpar_b3_chunk_len(&h->chunk) > 0) {
    left = h->stack_len;
    xpar_b3_chunk_node(&h->chunk, &o);
  } else {
    /*  Only reachable if a caller finalises a hasher whose last update
        ended exactly on a subtree boundary.  */
    xpar_assert(h->stack_len >= 2);
    left = h->stack_len - 2;
    xpar_b3_parent_node(h->stack + left * XPAR_BLAKE3_OUT_LEN, h->key,
                        h->chunk.flags, &o);
  }
  while (left > 0) {
    u8 block[XPAR_BLAKE3_BLOCK_LEN];
    left--;
    xpar_memcpy(block, h->stack + left * XPAR_BLAKE3_OUT_LEN,
                XPAR_BLAKE3_OUT_LEN);
    xpar_b3_node_cv(&o, block + XPAR_BLAKE3_OUT_LEN);
    xpar_b3_parent_node(block, h->key, h->chunk.flags, &o);
  }
  xpar_b3_root(&o, seek, out, n);
}

void xpar_blake3_final(const xpar_blake3_t * h, u8 * out, sz n) {
  xpar_blake3_final_seek(h, 0, out, n);
}

void xpar_blake3_subtree_stream_final(const xpar_blake3_t * h, u8 * out,
                                      sz n) {
  xpar_b3_node o;
  u8 cv[XPAR_BLAKE3_OUT_LEN];
  sz left;
  u64 chunks = h->chunk.counter - h->chunk_base +
               (xpar_b3_chunk_len(&h->chunk) != 0);
  xpar_assert(n <= sizeof cv);
  xpar_assert(chunks && !(chunks & (chunks - 1)));
  if (h->stack_len == 0) {
    xpar_b3_chunk_node(&h->chunk, &o);
  } else if (xpar_b3_chunk_len(&h->chunk) > 0) {
    left = h->stack_len;
    xpar_b3_chunk_node(&h->chunk, &o);
    while (left > 0) {
      u8 block[XPAR_BLAKE3_BLOCK_LEN];
      left--;
      xpar_memcpy(block, h->stack + left * XPAR_BLAKE3_OUT_LEN,
                  XPAR_BLAKE3_OUT_LEN);
      xpar_b3_node_cv(&o, block + XPAR_BLAKE3_OUT_LEN);
      xpar_b3_parent_node(block, h->key, h->chunk.flags, &o);
    }
  } else {
    xpar_assert(h->stack_len >= 2);
    left = h->stack_len - 2;
    xpar_b3_parent_node(h->stack + left * XPAR_BLAKE3_OUT_LEN, h->key,
                        h->chunk.flags, &o);
    while (left > 0) {
      u8 block[XPAR_BLAKE3_BLOCK_LEN];
      left--;
      xpar_memcpy(block, h->stack + left * XPAR_BLAKE3_OUT_LEN,
                  XPAR_BLAKE3_OUT_LEN);
      xpar_b3_node_cv(&o, block + XPAR_BLAKE3_OUT_LEN);
      xpar_b3_parent_node(block, h->key, h->chunk.flags, &o);
    }
  }
  xpar_b3_node_cv(&o, cv);
  xpar_memcpy(out, cv, n);
}

u64 xpar_blake3_final_u64(const xpar_blake3_t * h) {
  u8 out[8];
  xpar_blake3_final_seek(h, 0, out, sizeof(out));
  return xpar_rd64(out);
}

void xpar_blake3_hash(const void * buf, sz len, u8 * out, sz n) {
  xpar_blake3_t h;
  xpar_blake3_init(&h);
  xpar_blake3_update(&h, buf, len);
  xpar_blake3_final(&h, out, n);
}

void xpar_blake3_hash_keyed(const u8 * key, const void * buf, sz len,
                            u8 * out, sz n) {
  xpar_blake3_t h;
  xpar_blake3_init_keyed(&h, key);
  xpar_blake3_update(&h, buf, len);
  xpar_blake3_final(&h, out, n);
}

void xpar_blake3_derive_key(const char * context, const void * material,
                            sz len, u8 * out) {
  xpar_blake3_t h;
  xpar_blake3_init_derive_key(&h, context);
  xpar_blake3_update(&h, material, len);
  xpar_blake3_final(&h, out, XPAR_BLAKE3_KEY_LEN);
}
