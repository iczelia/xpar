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

#include "jmode.h"
#include "crc32c.h"
#include "platform.h"
#include "io_uring_host.h"
#ifdef HAVE_BLAKE2B
  #include "blake2b.h"
#endif

/*  Reed-Solomon (K=223 data, T=32 parity, N=255 codeword).  */
#define K 223
#define N 255
#define T 32

/*  BCH view; originally Phil Karn KA9Q (1999), restructured for speed.  */
static u8 LOG[256], EXP[256], PROD[256][256], DP[256][256];
u8 PROD_GEN[256][32];
void jmode_gf256_gentab(u8 poly) {
  for (int l = 0, b = 1; l < 255; l++) {
    LOG[b] = l;  EXP[l] = b;
    if ((b <<= 1) >= 256)
      b = (b - 256) ^ poly;
  }
  LOG[0] = 255; EXP[255] = 0;
  Fi(256, Fj(256,
    PROD[i][j] = (i && j) ? EXP[(LOG[i] + LOG[j]) % 255] : 0,
    DP[i][j] = (i != 255 && j) ? EXP[(i + LOG[j]) % 255] : 0))
  static const u8 gen[T] = {
    1, 91, 127, 86, 16, 30, 13, 235, 97, 165, 8, 42, 54, 86, 171, 32,
    113, 32, 171, 86, 54, 42, 8, 165, 97, 235, 13, 30, 16, 86, 127, 91
  };
  Fi(256, Fj(T, PROD_GEN[i][j] = PROD[i][gen[j]]))
}
static u8 gf256_div(u8 a, u8 b) {
  if (!a || !b) return 0;
  int d = LOG[a] - LOG[b];
  return EXP[d < 0 ? d + 255 : d];
}
#if defined(XPAR_X86_64)
#ifdef HAVE_FUNC_ATTRIBUTE_SYSV_ABI
  #define EXTERNAL_ABI __attribute__((sysv_abi))
#else
  #define EXTERNAL_ABI
#endif

extern EXTERNAL_ABI int xpar_x86_64_cpuflags(void);
extern EXTERNAL_ABI void rse32_inplace_x86_64_avx512(u8 out[N]);
extern EXTERNAL_ABI void rse32_inplace_x86_64_generic(u8 out[N]);
extern EXTERNAL_ABI void rse32_scatter_x86_64(const u8 * data, u8 * out, sz n);

static int rse32_cpuflags(void) {
  static int flags = -1;
  if (flags == -1) flags = xpar_x86_64_cpuflags();
  return flags;
}
void rse32_inplace(u8 out[N]) {
  if (rse32_cpuflags() & 0xC) rse32_inplace_x86_64_avx512(out);
  else rse32_inplace_x86_64_generic(out);
}
void rse32_scatter(const u8 * data, u8 * out, sz n) {
  rse32_scatter_x86_64(data, out, n);
}
#else
static void rse32_inplace_c(u8 out[N]) {
  xpar_memset(out + K, 0, N - K);
  for (int i = K - 1; i >= 0; i--) {
    u8 x = out[i] ^ out[K + T - 1];
    for (int j = T - 1; j > 0; j--)
      out[K + j] = out[K + j - 1] ^ PROD_GEN[x][j];
    out[K] = PROD_GEN[x][0];
  }
}
void rse32_inplace(u8 out[N]) { rse32_inplace_c(out); }
void rse32_scatter(const u8 * data, u8 * out, sz n) {
  for (sz i = 0; i < n; i++) xpar_memcpy(out + i * N, data + i * K, K);
}
#endif

void rse32(const u8 data[K], u8 out[N]) {
  rse32_scatter(data, out, 1);
  rse32_inplace(out);
}

/*  Scatter src into K-strided slots of out, zero-padding the tail.
    Parity region untouched. Requires src_len <= n_blocks * K.  */
void rse32_scatter_pad(const u8 * src, sz src_len, u8 * out, sz n_blocks) {
  sz full_blocks = src_len / K;
  if (full_blocks > n_blocks) full_blocks = n_blocks;
  if (full_blocks) rse32_scatter(src, out, full_blocks);
  if (full_blocks >= n_blocks) return;
  sz remainder = src_len - full_blocks * K;
  u8 * blk = out + full_blocks * N;
  if (remainder) {
    xpar_memcpy(blk, src + full_blocks * K, remainder);
    xpar_memset(blk + remainder, 0, K - remainder);
  } else {
    xpar_memset(blk, 0, K);
  }
  for (sz i = full_blocks + 1; i < n_blocks; i++)
    xpar_memset(out + i * N, 0, K);
}
int rsd32(u8 data[N]) {
  int deg_lambda, el, deg_omega = 0;
  int i, j, r, k, syn_error, count;
  u8 q, tmp, num1, den, discr_r;
  u8 lambda[T + 1] = { 0 }, omega[T + 1] = { 0 }, eras_pos[T] = { 0 };
  u8 t[T + 1], s[T], root[T], reg[T + 1] = { 0 };
  u8 b_backing[3 * T + 1] = { 0 }, * b = b_backing + 2 * T;
  xpar_memset(s, data[0], T);
  /*   Fast syndrome computation: idea discovered by Marshall Lochbaum.  */
  for (int jb = 0; jb < 51; jb++) {
    u8 a5 = 255, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    for (j = jb; j < 255; j += 51) {
      if (j == 0 || !data[j]) continue;
      tmp = data[j];
      tmp = PROD[EXP[(212 * j) % 255]][tmp];
      u8 a1 = EXP[(11 * j) % 255]; /*  a_j1  */
      u8 a2 = PROD[a1][a1]; /*  a_j2  */
      if (a5 == 255) a5 = PROD[a2][PROD[a1][a2]];
      t1 ^=                tmp; /*  t1 = sum of t_j1, j = jb (mod 51)  */
      t2 ^=       PROD[a1][tmp]; /*  etc.  */
      t3 ^= tmp = PROD[a2][tmp];
      t4 ^=       PROD[a1][tmp];
      t5 ^=       PROD[a2][tmp];
    }
    if (a5 == 255) continue; /*  No j values do anything (unlikely)  */
    for (i = 0; ; i += 5) {
      s[i ] ^= t1; t1 = PROD[a5][t1];
      s[i+1] ^= t2; t2 = PROD[a5][t2];
      if (i+2 == 32) break;
      s[i+2] ^= t3; t3 = PROD[a5][t3];
      s[i+3] ^= t4; t4 = PROD[a5][t4];
      s[i+4] ^= t5; t5 = PROD[a5][t5];
    }
  }
  syn_error = 0; Fi(T, syn_error |= s[i])
  if (!syn_error) return 0;
  lambda[0] = 1;  r = el = 0;  xpar_memcpy(b, lambda, T + 1);
  while (++r <= T) {
    for (discr_r = 0, i = 0; i < r; i++)
      discr_r ^= PROD[lambda[i]][s[r - i - 1]];
    if (!discr_r) --b; else {
      t[0] = lambda[0];
      Fi(T, t[i + 1] = lambda[i + 1] ^ PROD[discr_r][b[i]])
      if (2 * el <= r - 1) {
        el = r - el;
        Fi(T + 1, b[i] = gf256_div(lambda[i], discr_r))
      } else --b;
      xpar_memcpy(lambda, t, T + 1);
    }
  }
  deg_lambda = 0;
  Fi(T + 1, if (lambda[i]) deg_lambda = i, reg[i] = lambda[i])
  for (count = 0, i = 1, k = 139; i <= 255; i++, k = (k + 139) % 255) {
    for (q = 1, j = deg_lambda; j > 0; j--)
      q ^= reg[j] = DP[j][reg[j]];
    if (q) continue;
    root[count] = i, eras_pos[count] = k;
    if (++count == deg_lambda) break; /*  Early exit.  */
  }
  if (deg_lambda != count) return -1;
  Fi(T,
    for (tmp = 0, j = MIN(deg_lambda, i); j >= 0; j--)
      tmp ^= PROD[s[i - j]][lambda[j]];
    if (tmp) deg_omega = i, omega[i] = tmp)
  for (j = count - 1; j >= 0; j--) {
    for (num1 = 0, i = deg_omega; i >= 0; i--)
      num1 ^= DP[(i * root[j]) % 255][omega[i]];
    for (den = 0, i = MIN(deg_lambda, T - 1) & ~1; i >= 0; i -= 2)
      den ^= DP[(i * root[j]) % 255][lambda[i + 1]];
    if (den == 0) return -1;
    data[eras_pos[j]] ^= gf256_div(DP[(root[j] * 111) % 255][num1], den);
  }
  return count;
}

/*  Strategies: mapped+serial encode (3); unmapped in/out (4).  */
static int compute_interlacing_bs(int ifactor) {
  switch (ifactor) {
    default: case 1: return 1; break;
    case 2: return N; break;
    case 3: return N * N; break;
  }
}
static void trans2D(u8 * restrict in, u8 * restrict out) {
  Fi(N, Fj(N, out[j * N + i] = in[i * N + j]))
}
struct _pf_ctx_trans3D { u8 * restrict in; u8 * restrict out; };
static void _pf_fn_trans3D(sz i, void * p) {
  struct _pf_ctx_trans3D * c = p;
  Fj(N, Fk(N, c->out[k * N * N + j * N + i] = c->in[i * N * N + j * N + k]))
}
static void trans3D(u8 * restrict in, u8 * restrict out) {
  struct _pf_ctx_trans3D ctx = { in, out };
  xpar_parallel_for(N, _pf_fn_trans3D, &ctx);
}

/*  Shared per-lace RS encode kernel: rse32_inplace on o1 + i*N.  */
struct _pf_ctx_rse32 { u8 * o1; };
static void _pf_fn_rse32(sz i, void * p) {
  struct _pf_ctx_rse32 * c = p;
  rse32_inplace(c->o1 + i * N);
}

/*  Per-lace RS decode: rsd32 then copy K corrected bytes.
    Accumulates ecc; unrecoverable blocks bump bad (soft) or exit (strict).  */
struct _pf_ctx_decode {
  u8 * in2;
  u8 * out_buffer;
  xpar_atomic_int * ecc;
  xpar_atomic_int * bad;   /*  NULL means strict: exit on failure  */
  unsigned laces;
  int ibs;
  bool quiet;
  bool force;
  bool terse; /*  true: "Block U (lace V)"; false: with byte range.  */
};
static void _pf_fn_decode(sz i, void * p) {
  struct _pf_ctx_decode * c = p;
  int n = rsd32(c->in2 + i * N);
  if (n < 0) {
    const unsigned lace_ibs = c->laces * (unsigned) c->ibs + (unsigned) i;
    if (!c->quiet) {
      if (c->terse)
        xpar_fprintf(xpar_stderr,
          "Block %u (lace %u) irrecoverable.\n", lace_ibs, c->laces);
      else
        xpar_fprintf(xpar_stderr,
          "Block %u (lace %u, bytes %u-%u) irrecoverable.\n",
          lace_ibs, c->laces, lace_ibs * N, lace_ibs * N + N - 1);
    }
    if (c->bad) xpar_atomic_add_int(c->bad, 1);
    else if (!c->force) xpar_exit(1);
  } else {
    xpar_atomic_add_int(c->ecc, n);
  }
  xpar_memcpy(c->out_buffer + i * K, c->in2 + i * N, K);
}
static void do_interlacing(u8 * restrict in, u8 * restrict out, int ifactor) {
  switch (ifactor) {
    case 1: xpar_memcpy(out, in, N); break;
    case 2: trans2D(in, out); break;
    case 3: trans3D(in, out); break;
  }
}
/*  File header: 15 payload + 32 RS-parity = 47 bytes on disk.
    Payload h[0..15) is recovered via rsd32 on decode; also bound into the
    per-block MAC domain (ctx) so any tamper with header fields fails all
    block/trailer MAC verifications.
    Layout: "XP"|major|minor|ifactor('1'..'4','4'=systematic)
            |integrity|auth|total_bytes(u64 BE, ~0=unknown).  */
#define FHDR_PAYLOAD 15
#define FHDR_DISK_SIZE (FHDR_PAYLOAD + (N - K))
typedef struct {
  int ifactor;
  int integrity;
  int auth_flag;
  u64 total_bytes;
  u8 ctx[FHDR_PAYLOAD];
} file_hdr;
static void pack_u64_be(u8 * b, u64 v) {
  Fi(8, b[i] = (u8)(v >> (56 - 8 * i)));
}
static u64 unpack_u64_be(const u8 * b) {
  u64 v = 0;
  Fi(8, v |= ((u64) b[i]) << (56 - 8 * i));
  return v;
}
static void write_header(xpar_file * des, file_hdr * fh) {
  u8 h[K] = { 0 }, out[N];
  h[0] = 'X'; h[1] = 'P'; h[2] = XPAR_MAJOR; h[3] = XPAR_MINOR;
  h[4] = fh->ifactor + '0';
  h[5] = (u8) fh->integrity;
  h[6] = (u8) fh->auth_flag;
  pack_u64_be(h + 7, fh->total_bytes);
  xpar_memcpy(fh->ctx, h, FHDR_PAYLOAD);
  rse32(h, out);
  xpar_xwrite(des, h, FHDR_PAYLOAD);
  xpar_xwrite(des, out + K, N - K);
}
static file_hdr parse_header(u8 out[N], int force, int ifactor_override) {
  file_hdr fh = { .ifactor = ifactor_override, .integrity = INTEGRITY_CRC32C,
                  .auth_flag = 0, .total_bytes = (u64) -1 };
  if (out[0] != 'X' || out[1] != 'P')
    FATAL_UNLESS("Invalid header.", !force);
  out[0] = 'X'; out[1] = 'P'; xpar_memset(out + FHDR_PAYLOAD, 0,
                                          K - FHDR_PAYLOAD);
  if (rsd32(out) < 0)
    FATAL_UNLESS("Invalid header.", !force);
  xpar_memcpy(fh.ctx, out, FHDR_PAYLOAD);
  if (out[2] != XPAR_MAJOR) {
    FATAL_UNLESS("File was produced by an incompatible xpar major version.",
                 !force);
    if (force) return fh;
  }
  int ifactor = out[4] - '0';
  if (ifactor < 1 || ifactor > 4) {
    FATAL_UNLESS("Invalid header.", !force);
    if (force) ifactor = ifactor_override;
  }
  fh.ifactor = ifactor;
  fh.integrity = out[5];
  fh.auth_flag = out[6];
  fh.total_bytes = unpack_u64_be(out + 7);
  if (fh.integrity != INTEGRITY_CRC32C && fh.integrity != INTEGRITY_BLAKE2B) {
    FATAL_UNLESS("Unknown integrity algorithm in header.", !force);
    if (force) fh.integrity = INTEGRITY_CRC32C;
  }
  if (fh.auth_flag != 0 && fh.auth_flag != 1) {
    FATAL_UNLESS("Invalid auth flag in header.", !force);
    if (force) fh.auth_flag = 0;
  }
#ifndef HAVE_BLAKE2B
  if (fh.integrity == INTEGRITY_BLAKE2B) {
    FATAL("This file uses BLAKE2b integrity, but xpar was built with "
          "BLAKE2b support disabled (configure --enable-blake2b).");
  }
#endif
  return fh;
}
static file_hdr read_header(xpar_file * des, int force, int ifactor_override) {
  u8 out[N];
  xpar_xread(des, out, FHDR_PAYLOAD);
  xpar_xread(des, out + K, N - K);
  return parse_header(out, force, ifactor_override);
}
#ifdef XPAR_ALLOW_MAPPING
static file_hdr read_header_from_map(xpar_mmap map, int force,
                                     int ifactor_override) {
  if (map.size < FHDR_DISK_SIZE)
    FATAL("Truncated file.");
  u8 out[N];
  xpar_memcpy(out, map.map, FHDR_PAYLOAD);
  xpar_memcpy(out + K, map.map + FHDR_PAYLOAD, N - K);
  return parse_header(out, force, ifactor_override);
}
#endif

/*  Block header:  CRC32C  12B: 'X'+bytes(3 BE)+seq(4 BE)+crc(4 BE);
                    BLAKE2B 24B: 'X'+bytes(3 BE)+seq(4 BE)+tag(16).
    The 8-byte prefix is bound into the MAC domain along with the file
    header payload, so any tamper with bytes/seq/header fails the MAC.
    bytes=0 is reserved for the end-of-stream trailer frame.  */
#define BHDR_PREFIX 8
#define BHDR_HASH_MAX 16
typedef struct { u32 bytes, seq; u8 hash[BHDR_HASH_MAX]; } block_hdr;
static inline sz bhdr_size(int algo) {
  return BHDR_PREFIX + (algo == INTEGRITY_CRC32C ? 4 : 16);
}
static void pack_bhdr_prefix(u8 prefix[BHDR_PREFIX], u32 bytes, u32 seq) {
  if (bytes > 0xFFFFFF)
    FATAL("Could not write the header: block too big.");
  prefix[0] = 'X';
  prefix[1] = (u8)(bytes >> 16);
  prefix[2] = (u8)(bytes >>  8);
  prefix[3] = (u8) bytes;
  prefix[4] = (u8)(seq   >> 24);
  prefix[5] = (u8)(seq   >> 16);
  prefix[6] = (u8)(seq   >>  8);
  prefix[7] = (u8) seq;
}
static void write_block_header(xpar_file * des,
                               const u8 prefix[BHDR_PREFIX],
                               const u8 hash[BHDR_HASH_MAX], int algo) {
  xpar_xwrite(des, prefix, BHDR_PREFIX);
  xpar_xwrite(des, hash, algo == INTEGRITY_CRC32C ? 4 : 16);
}
static block_hdr parse_block_header(const u8 * b, int algo, bool force) {
  block_hdr h;
  if (b[0] != 'X') {
    FATAL_UNLESS("Invalid block header.", !force);
    h.bytes = 0xFFFFFF; h.seq = 0;
    xpar_memset(h.hash, 0, BHDR_HASH_MAX);
    return h;
  }
  h.bytes = ((u32) b[1] << 16) | ((u32) b[2] << 8) | b[3];
  h.seq   = ((u32) b[4] << 24) | ((u32) b[5] << 16)
          | ((u32) b[6] <<  8) | b[7];
  if (algo == INTEGRITY_CRC32C) {
    xpar_memcpy(h.hash, b + BHDR_PREFIX, 4);
    xpar_memset(h.hash + 4, 0, BHDR_HASH_MAX - 4);
  } else {
    xpar_memcpy(h.hash, b + BHDR_PREFIX, 16);
  }
  return h;
}
/*  Per-lace integrity tag into dst[16]: CRC32C writes 4 (BE),
    BLAKE2b writes 16 (raw). MAC domain is always:
    fhdr_ctx(15) || bhdr_prefix(8) || body(len).  */
static void integrity_tag(u8 * dst, int algo,
                          const u8 * key, sz keylen,
                          const u8 fhdr_ctx[FHDR_PAYLOAD],
                          const u8 bhdr_prefix[BHDR_PREFIX],
                          const void * buf, sz len) {
  if (algo == INTEGRITY_CRC32C) {
    u32 c = 0xFFFFFFFFu;
    c = crc32c_partial(c, fhdr_ctx, FHDR_PAYLOAD);
    c = crc32c_partial(c, bhdr_prefix, BHDR_PREFIX);
    if (len) c = crc32c_partial(c, buf, len);
    c ^= 0xFFFFFFFFu;
    dst[0] = (u8)(c >> 24); dst[1] = (u8)(c >> 16);
    dst[2] = (u8)(c >>  8); dst[3] = (u8) c;
  } else {
#ifdef HAVE_BLAKE2B
    blake2b_state s;
    if (keylen) blake2b_init_key(&s, 16, key, keylen);
    else        blake2b_init(&s, 16);
    blake2b_update(&s, fhdr_ctx, FHDR_PAYLOAD);
    blake2b_update(&s, bhdr_prefix, BHDR_PREFIX);
    if (len) blake2b_update(&s, buf, len);
    blake2b_final(&s, dst, 16);
#else
    (void) dst; (void) key; (void) keylen;
    (void) fhdr_ctx; (void) bhdr_prefix; (void) buf; (void) len;
    FATAL("This build has BLAKE2b integrity disabled (configure with "
          "--enable-blake2b).");
#endif
  }
}
static bool integrity_match(const u8 a[BHDR_HASH_MAX],
                            const u8 b[BHDR_HASH_MAX], int algo) {
  sz n = algo == INTEGRITY_CRC32C ? 4 : 16;
  u8 d = 0;
  Fi(n, d |= a[i] ^ b[i])
  return d == 0;
}
/*  End-of-stream trailer: a bhdr-sized record with bytes=0 and seq set to
    the block/lace count. MAC is computed over fhdr_ctx || trailer_prefix
    || (empty body). Lets decoders detect tail truncation even when
    total_bytes is unknown (stdin input).  */
static void write_trailer(xpar_file * des, const file_hdr * fh,
                          int algo, const u8 * key, sz keylen,
                          u32 seq_count) {
  u8 prefix[BHDR_PREFIX];
  u8 hash[BHDR_HASH_MAX] = { 0 };
  pack_bhdr_prefix(prefix, 0, seq_count);
  integrity_tag(hash, algo, key, keylen, fh->ctx, prefix, NULL, 0);
  write_block_header(des, prefix, hash, algo);
}
static bool validate_trailer(const u8 * buf, int algo,
                             const file_hdr * fh,
                             const u8 * key, sz keylen,
                             u32 expected_seq) {
  if (buf[0] != 'X') return false;
  block_hdr th = parse_block_header(buf, algo, true);
  if (th.bytes != 0 || th.seq != expected_seq) return false;
  u8 exp[BHDR_HASH_MAX];
  integrity_tag(exp, algo, key, keylen, fh->ctx, buf, NULL, 0);
  return integrity_match(exp, th.hash, algo);
}
#ifdef XPAR_HAS_LIBURING
/*  Uring fast path for encode4: packs (interlaced body || bhdr) laces
    into a ~2 MiB arena. For ifactor=1 (rec=267 B) it coalesces thousands
    of laces per write; for ifactor=3 (rec=16 MiB+) it collapses to one
    lace per write, which still wins by replacing two syscalls with one
    plus avoiding the stdio buffer. Shares the scratch buffers with the
    caller so the arithmetic state is identical to the sync path.  */
static bool encode4_uring(xpar_file * in, xpar_file * out, int ifactor,
                          int algo, const u8 * key, sz keylen,
                          const file_hdr * fh,
                          u8 * in_buffer, u8 * o1, u8 * o2) {
  xpar_iogroup * iog = xpar_iogroup_new(8);
  if (!iog) return false;
  int fid = xpar_iogroup_register_file(iog, out);
  if (fid < 0) { xpar_iogroup_free(iog); return false; }

  int ibs = compute_interlacing_bs(ifactor);
  sz bsz = bhdr_size(algo);
  sz rec = (sz) ibs * N + bsz;
  sz target   = (sz) 2 * 1024 * 1024;
  sz laces_ar = target / rec; if (!laces_ar) laces_ar = 1;
  sz arena_sz = laces_ar * rec;
  u8 * arena = xpar_malloc(arena_sz);
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  u32 seq = 0;
  sz  filled = 0;
  u64 off    = FHDR_DISK_SIZE;
  for (size_t n; n = xpar_xread(in, in_buffer, ibs * K); seq++) {
    if (n < ibs * K) xpar_memset(in_buffer + n, 0, ibs * K - n);
    rse32_scatter(in_buffer, o1, ibs);
    if (ifactor == 3) {
      struct _pf_ctx_rse32 c = { o1 };
      xpar_parallel_for(ibs, _pf_fn_rse32, &c);
    } else {
      Fi(ibs, rse32_inplace(o1 + i * N));
    }
    do_interlacing(o1, o2, ifactor);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh->ctx, prefix, in_buffer, n);
    xpar_memcpy(arena + filled,               o2,    (sz) ibs * N);
    xpar_memcpy(arena + filled + (sz)ibs * N, prefix, BHDR_PREFIX);
    xpar_memcpy(arena + filled + (sz)ibs * N + BHDR_PREFIX,
                hash, bsz - BHDR_PREFIX);
    filled += rec;
    if (filled == arena_sz) {
      xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
      xpar_iogroup_drain(iog);
      off += filled; filled = 0;
    }
  }
  if (filled) {
    xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
    xpar_iogroup_drain(iog);
    off += filled;
  }
  pack_bhdr_prefix(prefix, 0, seq);
  u8 thash[BHDR_HASH_MAX] = { 0 };
  integrity_tag(thash, algo, key, keylen, fh->ctx, prefix, NULL, 0);
  xpar_memcpy(arena,               prefix, BHDR_PREFIX);
  xpar_memcpy(arena + BHDR_PREFIX, thash,  bsz - BHDR_PREFIX);
  xpar_iogroup_enqueue_write(iog, fid, arena, off, bsz, (u64) seq);
  xpar_iogroup_fsync(iog, fid);
  xpar_iogroup_free(iog);
  xpar_free(arena);
  return true;
}
#endif
static void encode4(xpar_file * in, xpar_file * out, int ifactor,
                    int algo, const u8 * key, sz keylen, u64 total_bytes) {
  xpar_notty(out);
  u8 * in_buffer, * o1, * o2;
  int ibs = compute_interlacing_bs(ifactor);
  in_buffer = xpar_malloc(ibs * K);
  o1 = xpar_malloc(ibs * N);  o2 = xpar_malloc(ibs * N);
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  file_hdr fh = { .ifactor = ifactor, .integrity = algo,
                  .auth_flag = keylen ? 1 : 0, .total_bytes = total_bytes };
  write_header(out, &fh);
#ifdef XPAR_HAS_LIBURING
  if (encode4_uring(in, out, ifactor, algo, key, keylen, &fh,
                    in_buffer, o1, o2)) {
    xpar_free(in_buffer); xpar_free(o1); xpar_free(o2);
    xpar_xclose(out); return;
  }
#endif
  u32 seq = 0;
  for (size_t n; n = xpar_xread(in, in_buffer, ibs * K); seq++) {
    if (n < ibs * K) xpar_memset(in_buffer + n, 0, ibs * K - n);
    rse32_scatter(in_buffer, o1, ibs);
    if (ifactor == 3) {
      struct _pf_ctx_rse32 c = { o1 };
      xpar_parallel_for(ibs, _pf_fn_rse32, &c);
    } else {
      Fi(ibs, rse32_inplace(o1 + i * N));
    }
    do_interlacing(o1, o2, ifactor);
    xpar_xwrite(out, o2, ibs * N);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh.ctx, prefix, in_buffer, n);
    write_block_header(out, prefix, hash, algo);
  }
  write_trailer(out, &fh, algo, key, keylen, seq);
  xpar_free(in_buffer), xpar_free(o1), xpar_free(o2); xpar_xclose(out);
}
#ifdef XPAR_ALLOW_MAPPING
#ifdef XPAR_HAS_LIBURING
static bool encode3_uring(xpar_mmap in, xpar_file * out, int ifactor,
                          int algo, const u8 * key, sz keylen,
                          const file_hdr * fh, u8 * o1, u8 * o2) {
  xpar_iogroup * iog = xpar_iogroup_new(8);
  if (!iog) return false;
  int fid = xpar_iogroup_register_file(iog, out);
  if (fid < 0) { xpar_iogroup_free(iog); return false; }

  int ibs = compute_interlacing_bs(ifactor);
  sz bsz = bhdr_size(algo);
  sz rec = (sz) ibs * N + bsz;
  sz target   = (sz) 2 * 1024 * 1024;
  sz laces_ar = target / rec; if (!laces_ar) laces_ar = 1;
  sz arena_sz = laces_ar * rec;
  u8 * arena = xpar_malloc(arena_sz);
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  u32 seq = 0;
  sz  filled = 0;
  u64 off    = FHDR_DISK_SIZE;
  for (sz n; n = MIN(in.size, ibs * K), n; in.size -= n, in.map += n, seq++) {
    rse32_scatter_pad(in.map, n, o1, ibs);
    if (ifactor == 3) {
      struct _pf_ctx_rse32 c = { o1 };
      xpar_parallel_for(ibs, _pf_fn_rse32, &c);
    } else {
      Fi(ibs, rse32_inplace(o1 + i * N));
    }
    do_interlacing(o1, o2, ifactor);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh->ctx, prefix, in.map, n);
    xpar_memcpy(arena + filled,               o2,    (sz) ibs * N);
    xpar_memcpy(arena + filled + (sz)ibs * N, prefix, BHDR_PREFIX);
    xpar_memcpy(arena + filled + (sz)ibs * N + BHDR_PREFIX,
                hash, bsz - BHDR_PREFIX);
    filled += rec;
    if (filled == arena_sz) {
      xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
      xpar_iogroup_drain(iog);
      off += filled; filled = 0;
    }
  }
  if (filled) {
    xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
    xpar_iogroup_drain(iog);
    off += filled;
  }
  pack_bhdr_prefix(prefix, 0, seq);
  u8 thash[BHDR_HASH_MAX] = { 0 };
  integrity_tag(thash, algo, key, keylen, fh->ctx, prefix, NULL, 0);
  xpar_memcpy(arena,               prefix, BHDR_PREFIX);
  xpar_memcpy(arena + BHDR_PREFIX, thash,  bsz - BHDR_PREFIX);
  xpar_iogroup_enqueue_write(iog, fid, arena, off, bsz, (u64) seq);
  xpar_iogroup_fsync(iog, fid);
  xpar_iogroup_free(iog);
  xpar_free(arena);
  return true;
}
#endif
static void encode3(xpar_mmap in, xpar_file * out, int ifactor,
                    int algo, const u8 * key, sz keylen) {
  xpar_notty(out);
  u8 * o1, * o2;
  int ibs = compute_interlacing_bs(ifactor);
  o1 = xpar_malloc(ibs * N), o2 = xpar_malloc(ibs * N);
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  file_hdr fh = { .ifactor = ifactor, .integrity = algo,
                  .auth_flag = keylen ? 1 : 0, .total_bytes = in.size };
  write_header(out, &fh);
#ifdef XPAR_HAS_LIBURING
  if (encode3_uring(in, out, ifactor, algo, key, keylen, &fh, o1, o2)) {
    xpar_free(o1); xpar_free(o2); xpar_xclose(out); return;
  }
#endif
  u32 seq = 0;
  for (sz n; n = MIN(in.size, ibs * K), n; in.size -= n, in.map += n, seq++) {
    rse32_scatter_pad(in.map, n, o1, ibs);
    if (ifactor == 3) {
      struct _pf_ctx_rse32 c = { o1 };
      xpar_parallel_for(ibs, _pf_fn_rse32, &c);
    } else {
      Fi(ibs, rse32_inplace(o1 + i * N));
    }
    do_interlacing(o1, o2, ifactor);
    xpar_xwrite(out, o2, ibs * N);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh.ctx, prefix, in.map, n);
    write_block_header(out, prefix, hash, algo);
  }
  write_trailer(out, &fh, algo, key, keylen, seq);
  xpar_free(o1), xpar_free(o2); xpar_xclose(out);
}
#endif
static void decode4(xpar_file * in, xpar_file * out, int force,
    int ifactor_override, bool quiet, bool verbose,
    const u8 * key, sz keylen) {
  xpar_notty(in);
  u8 * in1, * in2, * out_buffer;  u32 laces = 0;
  xpar_atomic_int ecc = 0;
  u64 bytes_out = 0;
  block_hdr bhdr; u8 tmp[24];
  file_hdr fh = read_header(in, force, ifactor_override);
  if (fh.ifactor == 4)
    FATAL("File is a systematic .xpa; pass -s to decode it.");
  if (fh.auth_flag && !keylen) FATAL("File requires --auth=<keyfile>.");
  if (!fh.auth_flag && keylen)
    FATAL("File is not authenticated; drop --auth.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  sz ibs = compute_interlacing_bs(fh.ifactor);
  in1 = xpar_malloc(ibs * N), in2 = xpar_malloc(ibs * N);
  out_buffer = xpar_malloc(ibs * K);
  bool saw_trailer = false;
  for (;;) {
    sz n = xpar_xread(in, in1, ibs * N);
    if (n == 0) break;
    if (n < ibs * N) {
      if (n == bsize && validate_trailer(in1, algo, &fh, key, keylen, laces)) {
        saw_trailer = true; break;
      }
      if (!quiet)
        xpar_fprintf(xpar_stderr, "Short read, lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) xpar_exit(1);
      xpar_memset(in1 + n, 0, ibs * N - n);
    }
    if (xpar_xread(in, tmp, bsize) != bsize) {
      if (!quiet)
        xpar_fprintf(xpar_stderr,
          "Short read (block header), lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) xpar_exit(1);
    }
    bhdr = parse_block_header(tmp, algo, force);
    if (bhdr.seq != laces) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "Lace %u out of order (header says %u).\n",
          laces, bhdr.seq);
      if (!force) xpar_exit(1);
    }
    do_interlacing(in1, in2, fh.ifactor);
    struct _pf_ctx_decode c = {
      in2, out_buffer, &ecc, NULL, laces, (int) ibs, quiet, force, false
    };
    if (fh.ifactor == 3) xpar_parallel_for(ibs, _pf_fn_decode, &c);
    else                 Fi(ibs, _pf_fn_decode(i, &c));
    sz size = MIN(ibs * K, bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, out_buffer, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "%s mismatch, lace %u (bytes %zu-%zu).\n",
          fh.auth_flag ? "MAC" : "Integrity",
          laces, laces * ibs * N, laces * ibs * N + size - 1);
      if (!force) xpar_exit(1);
    }
    xpar_xwrite(out, out_buffer, size);
    bytes_out += size;
    laces++;
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    if (!force) xpar_exit(1);
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: decoded %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    if (!force) xpar_exit(1);
  }
  xpar_free(in1), xpar_free(in2), xpar_free(out_buffer); xpar_xclose(out);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr, "Decoded %u laces, %u errors corrected.\n",
                 laces, (unsigned) xpar_atomic_load_int(&ecc));
}
#ifdef XPAR_ALLOW_MAPPING
static void decode3(xpar_mmap in, xpar_file * out, int force,
    int ifactor_override, bool quiet, bool verbose,
    const u8 * key, sz keylen) {
  u8 * in1, * in2, * out_buffer;  u32 laces = 0;
  xpar_atomic_int ecc = 0;
  u64 bytes_out = 0;
  block_hdr bhdr; u8 tmp[24];
  file_hdr fh = read_header_from_map(in, force, ifactor_override);
  if (fh.ifactor == 4)
    FATAL("File is a systematic .xpa; pass -s to decode it.");
  if (fh.auth_flag && !keylen) FATAL("File requires --auth=<keyfile>.");
  if (!fh.auth_flag && keylen)
    FATAL("File is not authenticated; drop --auth.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  in.size -= FHDR_DISK_SIZE; in.map += FHDR_DISK_SIZE;
  sz ibs = compute_interlacing_bs(fh.ifactor);
  in1 = xpar_malloc(ibs * N), in2 = xpar_malloc(ibs * N);
  out_buffer = xpar_malloc(ibs * K);
  bool saw_trailer = false;
  for (;;) {
    if (in.size == 0) break;
    sz n = MIN(in.size, ibs * N);
    if (n < ibs * N) {
      if (n == bsize) {
        u8 trbuf[24]; xpar_memcpy(trbuf, in.map, bsize);
        if (validate_trailer(trbuf, algo, &fh, key, keylen, laces)) {
          in.map += bsize; in.size -= bsize;
          saw_trailer = true; break;
        }
      }
      if (!quiet)
        xpar_fprintf(xpar_stderr, "Short read, lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) xpar_exit(1);
      xpar_memcpy(in1, in.map, n); in.map += n; in.size = 0;
      xpar_memset(in1 + n, 0, ibs * N - n);
    } else {
      xpar_memcpy(in1, in.map, n); in.map += n; in.size -= n;
    }
    if (in.size < bsize) {
      if (!quiet)
        xpar_fprintf(xpar_stderr,
          "Short read (block header), lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) xpar_exit(1);
      xpar_memcpy(tmp, in.map, in.size); in.map += in.size; in.size = 0;
    } else {
      xpar_memcpy(tmp, in.map, bsize); in.size -= bsize; in.map += bsize;
    }
    bhdr = parse_block_header(tmp, algo, force);
    if (bhdr.seq != laces) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "Lace %u out of order (header says %u).\n",
          laces, bhdr.seq);
      if (!force) xpar_exit(1);
    }
    do_interlacing(in1, in2, fh.ifactor);
    struct _pf_ctx_decode c = {
      in2, out_buffer, &ecc, NULL, laces, (int) ibs, quiet, force, false
    };
    if (fh.ifactor == 3) xpar_parallel_for(ibs, _pf_fn_decode, &c);
    else                 Fi(ibs, _pf_fn_decode(i, &c));
    sz size = MIN(ibs * K, bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, out_buffer, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "%s mismatch, lace %u (bytes %zu-%zu).\n",
          fh.auth_flag ? "MAC" : "Integrity",
          laces, laces * ibs * N, laces * ibs * N + size - 1);
      if (!force) xpar_exit(1);
    }
    xpar_xwrite(out, out_buffer, size);
    bytes_out += size;
    laces++;
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    if (!force) xpar_exit(1);
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: decoded %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    if (!force) xpar_exit(1);
  }
  xpar_free(in1), xpar_free(in2), xpar_free(out_buffer); xpar_xclose(out);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr, "Decoded %u laces, %u errors corrected.\n",
                 laces, (unsigned) xpar_atomic_load_int(&ecc));
}
#endif

#ifdef XPAR_HAS_LIBURING
/*  Uring fast path for encode_systematic4: same on-disk format, but the
    per-block (parity || bhdr) records are packed into a scratch arena
    and emitted as one pwrite per batch instead of three fwrites per
    block. Returns false if io_uring isn't available at runtime, and the
    caller falls through to the synchronous path.  */
static bool encode_systematic4_uring(xpar_file * in, xpar_file * out,
                                     int algo, const u8 * key, sz keylen,
                                     const file_hdr * fh) {
  xpar_iogroup * iog = xpar_iogroup_new(8);
  if (!iog) return false;
  int fid = xpar_iogroup_register_file(iog, out);
  if (fid < 0) { xpar_iogroup_free(iog); return false; }

  sz bsz = bhdr_size(algo);
  sz rec = (N - K) + bsz;
  unsigned B = xpar_iogroup_batch_records();
  u8 * arena = xpar_malloc((sz) B * rec);
  u8 buf[K], block[N];
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  u32 seq = 0;
  sz  filled = 0;
  u64 off    = FHDR_DISK_SIZE;
  bool short_block = false;
  for (sz n; (n = xpar_xread(in, buf, K)); seq++) {
    if (n < K) { xpar_memset(buf + n, 0, K - n); short_block = true; }
    rse32(buf, block);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh->ctx, prefix, buf, n);
    xpar_memcpy(arena + filled,              block + K, N - K);
    xpar_memcpy(arena + filled + (N - K),    prefix,    BHDR_PREFIX);
    xpar_memcpy(arena + filled + (N - K) + BHDR_PREFIX,
                hash, bsz - BHDR_PREFIX);
    filled += rec;
    if (filled == (sz) B * rec) {
      xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
      xpar_iogroup_drain(iog);   /*  arena safe to reuse  */
      off += filled; filled = 0;
    }
    if (short_block) { seq++; break; }
  }
  if (filled) {
    xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
    xpar_iogroup_drain(iog);
    off += filled;
  }
  pack_bhdr_prefix(prefix, 0, seq);
  u8 thash[BHDR_HASH_MAX] = { 0 };
  integrity_tag(thash, algo, key, keylen, fh->ctx, prefix, NULL, 0);
  xpar_memcpy(arena,               prefix, BHDR_PREFIX);
  xpar_memcpy(arena + BHDR_PREFIX, thash,  bsz - BHDR_PREFIX);
  xpar_iogroup_enqueue_write(iog, fid, arena, off, bsz, (u64) seq);
  xpar_iogroup_fsync(iog, fid);
  xpar_iogroup_free(iog);
  xpar_free(arena);
  return true;
}
#endif

static void encode_systematic4(xpar_file * in, xpar_file * out,
                               int algo, const u8 * key, sz keylen,
                               u64 total_bytes) {
  xpar_notty(out);
  u8 buf[K], block[N];
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  file_hdr fh = { .ifactor = 4, .integrity = algo,
                  .auth_flag = keylen ? 1 : 0, .total_bytes = total_bytes };
  write_header(out, &fh);
#ifdef XPAR_HAS_LIBURING
  if (encode_systematic4_uring(in, out, algo, key, keylen, &fh)) {
    xpar_xclose(out); return;
  }
#endif
  u32 seq = 0;
  bool short_block = false;
  for (sz n; (n = xpar_xread(in, buf, K)); seq++) {
    if (n < K) { xpar_memset(buf + n, 0, K - n); short_block = true; }
    rse32(buf, block);
    xpar_xwrite(out, block + K, N - K);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh.ctx, prefix, buf, n);
    write_block_header(out, prefix, hash, algo);
    if (short_block) { seq++; break; }
  }
  write_trailer(out, &fh, algo, key, keylen, seq);
  xpar_xclose(out);
}
#ifdef XPAR_ALLOW_MAPPING
#ifdef XPAR_HAS_LIBURING
static bool encode_systematic3_uring(xpar_mmap in, xpar_file * out,
                                     int algo, const u8 * key, sz keylen,
                                     const file_hdr * fh) {
  xpar_iogroup * iog = xpar_iogroup_new(8);
  if (!iog) return false;
  int fid = xpar_iogroup_register_file(iog, out);
  if (fid < 0) { xpar_iogroup_free(iog); return false; }

  sz bsz = bhdr_size(algo);
  sz rec = (N - K) + bsz;
  unsigned B = xpar_iogroup_batch_records();
  u8 * arena = xpar_malloc((sz) B * rec);
  u8 buf[K], block[N];
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  u32 seq = 0;
  sz  filled = 0;
  u64 off    = FHDR_DISK_SIZE;
  while (in.size) {
    sz n = MIN(in.size, (sz)K);
    xpar_memcpy(buf, in.map, n);
    if (n < K) xpar_memset(buf + n, 0, K - n);
    rse32(buf, block);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh->ctx, prefix, in.map, n);
    xpar_memcpy(arena + filled,              block + K, N - K);
    xpar_memcpy(arena + filled + (N - K),    prefix,    BHDR_PREFIX);
    xpar_memcpy(arena + filled + (N - K) + BHDR_PREFIX,
                hash, bsz - BHDR_PREFIX);
    filled += rec;
    if (filled == (sz) B * rec) {
      xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
      xpar_iogroup_drain(iog);
      off += filled; filled = 0;
    }
    in.size -= n; in.map += n;
    seq++;
  }
  if (filled) {
    xpar_iogroup_enqueue_write(iog, fid, arena, off, filled, (u64) seq);
    xpar_iogroup_drain(iog);
    off += filled;
  }
  pack_bhdr_prefix(prefix, 0, seq);
  u8 thash[BHDR_HASH_MAX] = { 0 };
  integrity_tag(thash, algo, key, keylen, fh->ctx, prefix, NULL, 0);
  xpar_memcpy(arena,               prefix, BHDR_PREFIX);
  xpar_memcpy(arena + BHDR_PREFIX, thash,  bsz - BHDR_PREFIX);
  xpar_iogroup_enqueue_write(iog, fid, arena, off, bsz, (u64) seq);
  xpar_iogroup_fsync(iog, fid);
  xpar_iogroup_free(iog);
  xpar_free(arena);
  return true;
}
#endif
static void encode_systematic3(xpar_mmap in, xpar_file * out,
                               int algo, const u8 * key, sz keylen) {
  xpar_notty(out);
  u8 buf[K], block[N];
  u8 prefix[BHDR_PREFIX], hash[BHDR_HASH_MAX];
  file_hdr fh = { .ifactor = 4, .integrity = algo,
                  .auth_flag = keylen ? 1 : 0, .total_bytes = in.size };
  write_header(out, &fh);
#ifdef XPAR_HAS_LIBURING
  if (encode_systematic3_uring(in, out, algo, key, keylen, &fh)) {
    xpar_xclose(out); return;
  }
#endif
  u32 seq = 0;
  while (in.size) {
    sz n = MIN(in.size, (sz)K);
    xpar_memcpy(buf, in.map, n);
    if (n < K) xpar_memset(buf + n, 0, K - n);
    rse32(buf, block);
    xpar_xwrite(out, block + K, N - K);
    pack_bhdr_prefix(prefix, n, seq);
    integrity_tag(hash, algo, key, keylen, fh.ctx, prefix, in.map, n);
    write_block_header(out, prefix, hash, algo);
    in.size -= n; in.map += n;
    seq++;
  }
  write_trailer(out, &fh, algo, key, keylen, seq);
  xpar_xclose(out);
}
#endif
static void decode_systematic4(xpar_file * data_in,
    xpar_file * parity_in, xpar_file * out,
    bool force, bool quiet, bool verbose,
    const u8 * key, sz keylen) {
  xpar_notty(parity_in);
  u8 block[N], tmp[24];  u32 blk = 0, ecc = 0;  u64 bytes_out = 0;
  file_hdr fh = read_header(parity_in, force, 4);
  if (fh.ifactor != 4)
    FATAL("File is not a systematic .xpa; omit -s to decode it.");
  if (fh.auth_flag && !keylen) FATAL("File requires --auth=<keyfile>.");
  if (!fh.auth_flag && keylen)
    FATAL("File is not authenticated; drop --auth.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  bool saw_trailer = false, done = false;
  for (;;) {
    sz pn = xpar_xread(parity_in, block + K, N - K);
    if (!pn) break;
    if (pn < (sz)(N - K)) {
      if (pn == bsize && validate_trailer(block + K, algo,
                                          &fh, key, keylen, blk)) {
        saw_trailer = true; break;
      }
      FATAL_UNLESS("Short parity block.", !force);
      xpar_memset(block + K + pn, 0, (N - K) - pn);
    }
    if (xpar_xread(parity_in, tmp, bsize) != bsize)
      FATAL_UNLESS("Short block header.", !force);
    block_hdr bhdr = parse_block_header(tmp, algo, force);
    if (bhdr.seq != blk) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "Block %u out of order (header says %u).\n",
          blk, bhdr.seq);
      if (!force) xpar_exit(1);
    }
    sz dn = xpar_xread(data_in, block, K);
    if (dn < (sz)K) xpar_memset(block + dn, 0, K - dn);
    int n = rsd32(block);
    if (n < 0) {
      if (!quiet) xpar_fprintf(xpar_stderr, "Block %u irrecoverable.\n", blk);
      if (!force) xpar_exit(1);
    } else ecc += n;
    sz size = MIN((sz)K, (sz)bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, block, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "%s mismatch, block %u.\n",
          fh.auth_flag ? "MAC" : "Integrity", blk);
      /*  Systematic + unauthenticated: the CRC is the only witness that
          the headerless data file is paired with this parity file, so a
          mismatch means wrong pairing; abort even under --force.  */
      if (!force || !fh.auth_flag) xpar_exit(1);
    }
    xpar_xwrite(out, block, size);
    bytes_out += size;
    blk++;
    if (size < (sz)K) { done = true; break; }
  }
  if (!saw_trailer) {
    /*  For systematic with a short final block, the encoder still emits
        a trailer afterwards; read it here.  */
    if (done) {
      sz pn = xpar_xread(parity_in, tmp, bsize);
      if (pn == bsize && validate_trailer(tmp, algo, &fh, key, keylen, blk))
        saw_trailer = true;
    }
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    if (!force) xpar_exit(1);
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: decoded %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    if (!force) xpar_exit(1);
  }
  xpar_xclose(out);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr,
      "Decoded %u blocks, %u errors corrected.\n", blk, ecc);
}
#ifdef XPAR_ALLOW_MAPPING
static void decode_systematic3(xpar_mmap data_in,
    xpar_file * parity_in, xpar_file * out,
    bool force, bool quiet, bool verbose,
    const u8 * key, sz keylen) {
  xpar_notty(parity_in);
  u8 block[N], tmp[24];  u32 blk = 0, ecc = 0;  u64 bytes_out = 0;
  file_hdr fh = read_header(parity_in, force, 4);
  if (fh.ifactor != 4)
    FATAL("File is not a systematic .xpa; omit -s to decode it.");
  if (fh.auth_flag && !keylen) FATAL("File requires --auth=<keyfile>.");
  if (!fh.auth_flag && keylen)
    FATAL("File is not authenticated; drop --auth.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  bool saw_trailer = false, done = false;
  for (;;) {
    sz pn = xpar_xread(parity_in, block + K, N - K);
    if (!pn) break;
    if (pn < (sz)(N - K)) {
      if (pn == bsize && validate_trailer(block + K, algo,
                                          &fh, key, keylen, blk)) {
        saw_trailer = true; break;
      }
      FATAL_UNLESS("Short parity block.", !force);
      xpar_memset(block + K + pn, 0, (N - K) - pn);
    }
    if (xpar_xread(parity_in, tmp, bsize) != bsize)
      FATAL_UNLESS("Short block header.", !force);
    block_hdr bhdr = parse_block_header(tmp, algo, force);
    if (bhdr.seq != blk) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "Block %u out of order (header says %u).\n",
          blk, bhdr.seq);
      if (!force) xpar_exit(1);
    }
    sz dn = MIN(data_in.size, (sz)K);
    xpar_memcpy(block, data_in.map, dn);
    if (dn < (sz)K) xpar_memset(block + dn, 0, K - dn);
    data_in.size -= dn; data_in.map += dn;
    int n = rsd32(block);
    if (n < 0) {
      if (!quiet) xpar_fprintf(xpar_stderr, "Block %u irrecoverable.\n", blk);
      if (!force) xpar_exit(1);
    } else ecc += n;
    sz size = MIN((sz)K, (sz)bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, block, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet)
        xpar_fprintf(xpar_stderr, "%s mismatch, block %u.\n",
          fh.auth_flag ? "MAC" : "Integrity", blk);
      /*  Systematic + unauthenticated: the CRC is the only witness that
          the headerless data file is paired with this parity file, so a
          mismatch means wrong pairing; abort even under --force.  */
      if (!force || !fh.auth_flag) xpar_exit(1);
    }
    xpar_xwrite(out, block, size);
    bytes_out += size;
    blk++;
    if (size < (sz)K) { done = true; break; }
  }
  if (!saw_trailer && done) {
    sz pn = xpar_xread(parity_in, tmp, bsize);
    if (pn == bsize && validate_trailer(tmp, algo, &fh, key, keylen, blk))
      saw_trailer = true;
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    if (!force) xpar_exit(1);
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: decoded %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    if (!force) xpar_exit(1);
  }
  xpar_xclose(out);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr,
      "Decoded %u blocks, %u errors corrected.\n", blk, ecc);
}
#endif

/*  Integrity check: decode path sans output write, continues on errors.  */
static int test4(xpar_file * in, int ifactor_override,
    bool quiet, bool verbose, const u8 * key, sz keylen) {
  xpar_notty(in);
  u8 * in1, * in2, * out_buffer;
  u32 laces = 0;
  xpar_atomic_int ecc = 0, bad = 0;
  u64 bytes_out = 0;
  block_hdr bhdr; u8 tmp[24];
  file_hdr fh = read_header(in, true, ifactor_override);
  if (fh.ifactor == 4)
    FATAL("File is a systematic .xpa; pass -s to test it.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  sz ibs = compute_interlacing_bs(fh.ifactor);
  in1 = xpar_malloc(ibs * N), in2 = xpar_malloc(ibs * N);
  out_buffer = xpar_malloc(ibs * K);
  bool saw_trailer = false;
  for (;;) {
    sz n = xpar_xread(in, in1, ibs * N);
    if (n == 0) break;
    if (n < ibs * N) {
      if (n == bsize && validate_trailer(in1, algo, &fh, key, keylen, laces)) {
        saw_trailer = true; break;
      }
      if (!quiet) xpar_fprintf(xpar_stderr, "Short read, lace %u.\n", laces);
      xpar_atomic_add_int(&bad, 1); xpar_memset(in1 + n, 0, ibs * N - n);
    }
    if (xpar_xread(in, tmp, bsize) != bsize) {
      if (!quiet) xpar_fprintf(xpar_stderr,
        "Short block header, lace %u.\n", laces);
      xpar_atomic_add_int(&bad, 1);
    }
    bhdr = parse_block_header(tmp, algo, true);
    if (bhdr.seq != laces) {
      if (!quiet) xpar_fprintf(xpar_stderr,
        "Lace %u out of order (header says %u).\n",
        laces, bhdr.seq);
      xpar_atomic_add_int(&bad, 1);
    }
    do_interlacing(in1, in2, fh.ifactor);
    struct _pf_ctx_decode c = {
      in2, out_buffer, &ecc, &bad, laces, (int) ibs, quiet, false, true
    };
    if (fh.ifactor == 3) xpar_parallel_for(ibs, _pf_fn_decode, &c);
    else                 Fi(ibs, _pf_fn_decode(i, &c));
    sz size = MIN(ibs * K, bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, out_buffer, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet) xpar_fprintf(xpar_stderr, "%s mismatch, lace %u.\n",
        fh.auth_flag ? "MAC" : "Integrity", laces);
      xpar_atomic_add_int(&bad, 1);
    }
    bytes_out += size;
    laces++;
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    xpar_atomic_add_int(&bad, 1);
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: tested %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    xpar_atomic_add_int(&bad, 1);
  }
  xpar_free(in1), xpar_free(in2), xpar_free(out_buffer);
  int ecc_val = xpar_atomic_load_int(&ecc);
  int bad_val = xpar_atomic_load_int(&bad);
  if (ecc_val && !quiet)
    xpar_fprintf(xpar_stderr,
      "%u byte(s) needed RS correction: file is not pristine.\n",
      (unsigned) ecc_val);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr, "Tested %u laces, %u errors corrected.\n",
                 laces, (unsigned) ecc_val);
  return bad_val + (ecc_val ? 1 : 0);
}
#ifdef XPAR_ALLOW_MAPPING
static int test3(xpar_mmap in, int ifactor_override, bool quiet, bool verbose,
                 const u8 * key, sz keylen) {
  u8 * in1, * in2, * out_buffer;
  u32 laces = 0;
  xpar_atomic_int ecc = 0, bad = 0;
  u64 bytes_out = 0;
  block_hdr bhdr; u8 tmp[24];
  file_hdr fh = read_header_from_map(in, true, ifactor_override);
  if (fh.ifactor == 4)
    FATAL("File is a systematic .xpa; pass -s to test it.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  in.size -= FHDR_DISK_SIZE; in.map += FHDR_DISK_SIZE;
  sz ibs = compute_interlacing_bs(fh.ifactor);
  in1 = xpar_malloc(ibs * N), in2 = xpar_malloc(ibs * N);
  out_buffer = xpar_malloc(ibs * K);
  bool saw_trailer = false;
  for (;;) {
    if (in.size == 0) break;
    sz n = MIN(in.size, ibs * N);
    if (n < ibs * N) {
      if (n == bsize) {
        u8 trbuf[24]; xpar_memcpy(trbuf, in.map, bsize);
        if (validate_trailer(trbuf, algo, &fh, key, keylen, laces)) {
          in.map += bsize; in.size -= bsize;
          saw_trailer = true; break;
        }
      }
      if (!quiet) xpar_fprintf(xpar_stderr, "Short read, lace %u.\n", laces);
      xpar_atomic_add_int(&bad, 1);
      xpar_memcpy(in1, in.map, n); in.map += n; in.size = 0;
      xpar_memset(in1 + n, 0, ibs * N - n);
    } else {
      xpar_memcpy(in1, in.map, n); in.map += n; in.size -= n;
    }
    if (in.size < bsize) {
      if (!quiet) xpar_fprintf(xpar_stderr,
        "Short block header, lace %u.\n", laces);
      xpar_atomic_add_int(&bad, 1);
      xpar_memcpy(tmp, in.map, in.size); in.map += in.size; in.size = 0;
    } else {
      xpar_memcpy(tmp, in.map, bsize); in.size -= bsize; in.map += bsize;
    }
    bhdr = parse_block_header(tmp, algo, true);
    if (bhdr.seq != laces) {
      if (!quiet) xpar_fprintf(xpar_stderr,
        "Lace %u out of order (header says %u).\n",
        laces, bhdr.seq);
      xpar_atomic_add_int(&bad, 1);
    }
    do_interlacing(in1, in2, fh.ifactor);
    struct _pf_ctx_decode c = {
      in2, out_buffer, &ecc, &bad, laces, (int) ibs, quiet, false, true
    };
    if (fh.ifactor == 3) xpar_parallel_for(ibs, _pf_fn_decode, &c);
    else                 Fi(ibs, _pf_fn_decode(i, &c));
    sz size = MIN(ibs * K, bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, out_buffer, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet) xpar_fprintf(xpar_stderr, "%s mismatch, lace %u.\n",
        fh.auth_flag ? "MAC" : "Integrity", laces);
      xpar_atomic_add_int(&bad, 1);
    }
    bytes_out += size;
    laces++;
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    xpar_atomic_add_int(&bad, 1);
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: tested %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    xpar_atomic_add_int(&bad, 1);
  }
  xpar_free(in1), xpar_free(in2), xpar_free(out_buffer);
  int ecc_val = xpar_atomic_load_int(&ecc);
  int bad_val = xpar_atomic_load_int(&bad);
  if (ecc_val && !quiet)
    xpar_fprintf(xpar_stderr,
      "%u byte(s) needed RS correction: file is not pristine.\n",
      (unsigned) ecc_val);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr, "Tested %u laces, %u errors corrected.\n",
                 laces, (unsigned) ecc_val);
  return bad_val + (ecc_val ? 1 : 0);
}
#endif
static int test_systematic4(xpar_file * data_in, xpar_file * parity_in,
                             bool quiet, bool verbose,
                             const u8 * key, sz keylen) {
  xpar_notty(parity_in);
  u8 block[N], tmp[24];  u32 blk = 0, ecc = 0, bad = 0;  u64 bytes_out = 0;
  file_hdr fh = read_header(parity_in, true, 4);
  if (fh.ifactor != 4)
    FATAL("File is not a systematic .xpa; omit -s to test it.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  bool saw_trailer = false, done = false;
  for (;;) {
    sz pn = xpar_xread(parity_in, block + K, N - K);
    if (!pn) break;
    if (pn < (sz)(N - K)) {
      if (pn == bsize && validate_trailer(block + K, algo,
                                          &fh, key, keylen, blk)) {
        saw_trailer = true;
      } else bad++;
      break;
    }
    if (xpar_xread(parity_in, tmp, bsize) != bsize) { bad++; break; }
    block_hdr bhdr = parse_block_header(tmp, algo, true);
    if (bhdr.seq != blk) {
      if (!quiet) xpar_fprintf(xpar_stderr,
        "Block %u out of order (header says %u).\n",
        blk, bhdr.seq);
      bad++;
    }
    sz dn = xpar_xread(data_in, block, K);
    if (dn < (sz)K) xpar_memset(block + dn, 0, K - dn);
    int n = rsd32(block);
    if (n < 0) {
      if (!quiet) xpar_fprintf(xpar_stderr, "Block %u irrecoverable.\n", blk);
      bad++;
    } else ecc += n;
    sz size = MIN((sz)K, (sz)bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, block, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet) xpar_fprintf(xpar_stderr, "%s mismatch, block %u.\n",
        fh.auth_flag ? "MAC" : "Integrity", blk);
      bad++;
    }
    bytes_out += size;
    blk++;
    if (size < (sz)K) { done = true; break; }
  }
  if (!saw_trailer && done) {
    sz pn = xpar_xread(parity_in, tmp, bsize);
    if (pn == bsize && validate_trailer(tmp, algo, &fh, key, keylen, blk))
      saw_trailer = true;
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    bad++;
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: tested %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    bad++;
  }
  if (ecc && !quiet)
    xpar_fprintf(xpar_stderr,
      "%u byte(s) needed RS correction: file is not pristine.\n", ecc);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr,
      "Tested %u blocks, %u errors corrected.\n", blk, ecc);
  return bad + (ecc ? 1 : 0);
}
#ifdef XPAR_ALLOW_MAPPING
static int test_systematic3(xpar_mmap data_in, xpar_file * parity_in,
                             bool quiet, bool verbose,
                             const u8 * key, sz keylen) {
  xpar_notty(parity_in);
  u8 block[N], tmp[24];  u32 blk = 0, ecc = 0, bad = 0;  u64 bytes_out = 0;
  file_hdr fh = read_header(parity_in, true, 4);
  if (fh.ifactor != 4)
    FATAL("File is not a systematic .xpa; omit -s to test it.");
  int algo = fh.integrity;
  sz bsize = bhdr_size(algo);
  bool saw_trailer = false, done = false;
  for (;;) {
    sz pn = xpar_xread(parity_in, block + K, N - K);
    if (!pn) break;
    if (pn < (sz)(N - K)) {
      if (pn == bsize && validate_trailer(block + K, algo,
                                          &fh, key, keylen, blk)) {
        saw_trailer = true;
      } else bad++;
      break;
    }
    if (xpar_xread(parity_in, tmp, bsize) != bsize) { bad++; break; }
    block_hdr bhdr = parse_block_header(tmp, algo, true);
    if (bhdr.seq != blk) {
      if (!quiet) xpar_fprintf(xpar_stderr,
        "Block %u out of order (header says %u).\n",
        blk, bhdr.seq);
      bad++;
    }
    sz dn = MIN(data_in.size, (sz)K);
    xpar_memcpy(block, data_in.map, dn);
    if (dn < (sz)K) xpar_memset(block + dn, 0, K - dn);
    data_in.size -= dn; data_in.map += dn;
    int n = rsd32(block);
    if (n < 0) {
      if (!quiet) xpar_fprintf(xpar_stderr, "Block %u irrecoverable.\n", blk);
      bad++;
    } else ecc += n;
    sz size = MIN((sz)K, (sz)bhdr.bytes);
    u8 tag[BHDR_HASH_MAX];
    integrity_tag(tag, algo, key, keylen, fh.ctx, tmp, block, size);
    if (!integrity_match(tag, bhdr.hash, algo)) {
      if (!quiet) xpar_fprintf(xpar_stderr, "%s mismatch, block %u.\n",
        fh.auth_flag ? "MAC" : "Integrity", blk);
      bad++;
    }
    bytes_out += size;
    blk++;
    if (size < (sz)K) { done = true; break; }
  }
  if (!saw_trailer && done) {
    sz pn = xpar_xread(parity_in, tmp, bsize);
    if (pn == bsize && validate_trailer(tmp, algo, &fh, key, keylen, blk))
      saw_trailer = true;
  }
  if (!saw_trailer) {
    if (!quiet) xpar_fprintf(xpar_stderr,
      "Missing end-of-stream trailer; file is truncated.\n");
    bad++;
  }
  if (fh.total_bytes != (u64) -1 && bytes_out != fh.total_bytes) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
        "File truncated: tested %llu bytes, header declared %llu.\n",
        (unsigned long long) bytes_out,
        (unsigned long long) fh.total_bytes);
    bad++;
  }
  if (ecc && !quiet)
    xpar_fprintf(xpar_stderr,
      "%u byte(s) needed RS correction: file is not pristine.\n", ecc);
  if (!quiet && verbose)
    xpar_fprintf(xpar_stderr,
      "Tested %u blocks, %u errors corrected.\n", blk, ecc);
  return bad + (ecc ? 1 : 0);
}
#endif

static xpar_stat_t validate_file(const char * filename) {
  xpar_stat_t st;
  if (xpar_stat_path(filename, &st) != 0) FATAL_PERROR("stat");
  if (st.is_dir) FATAL("Input is a directory.");
  if (!st.is_regular) FATAL("Input is not a regular file.");
  return st;
}
static xpar_file * open_output(joint_options_t o) {
  xpar_file * out = xpar_stdout;
  if (o.output_name) {
    xpar_stat_t st;
    int exists = xpar_stat_path(o.output_name, &st);
    if (exists == 0 && (st.size || st.is_dir) && !o.force)
      FATAL("Output file `%s' exists and is not empty.", o.output_name);
    /*  Refuse to truncate the input via O_TRUNCATE before we've read it.  */
    if (exists == 0 && o.input_name
        && xpar_same_file(o.input_name, o.output_name) == 1)
      FATAL("Input and output refer to the same file.");
    if (!(out = xpar_open(o.output_name,
        XPAR_O_WRITE | XPAR_O_CREATE | XPAR_O_TRUNCATE)))
      FATAL_PERROR("fopen");
  }
  return out;
}
void do_joint_encode(joint_options_t o) {
  xpar_file * out = open_output(o), * in = xpar_stdin;
  u64 total = (u64) -1;
  if (o.input_name) {
    xpar_stat_t st = validate_file(o.input_name);
    total = (u64) st.size;
    if (!o.no_map) {
      #if defined(XPAR_ALLOW_MAPPING)
      xpar_mmap map = xpar_map(o.input_name);
      if (map.map) {
        if (o.interlacing == 4)
          encode_systematic3(map, out, o.integrity, o.auth_key, o.auth_keylen);
        else
          encode3(map, out, o.interlacing,
                  o.integrity, o.auth_key, o.auth_keylen);
        xpar_unmap(&map);
        return;
      }
      #endif
    }
    if (!(in = xpar_open(o.input_name, XPAR_O_READ))) FATAL_PERROR("fopen");
  }
  if (o.interlacing == 4)
    encode_systematic4(in, out, o.integrity, o.auth_key, o.auth_keylen, total);
  else
    encode4(in, out, o.interlacing,
            o.integrity, o.auth_key, o.auth_keylen, total);
}
void do_joint_decode(joint_options_t o) {
  if (o.interlacing == 4) {
    if (!o.output_name)
      FATAL("Systematic mode requires a parity .xpa file.");
    xpar_file * parity = xpar_open(o.output_name, XPAR_O_READ);
    if (!parity) FATAL_PERROR("fopen");
    if (o.input_name) {
      validate_file(o.input_name);
      if (!o.no_map) {
        #if defined(XPAR_ALLOW_MAPPING)
        xpar_mmap map = xpar_map(o.input_name);
        if (map.map) {
          decode_systematic3(map, parity, xpar_stdout,
                             o.force, o.quiet, o.verbose,
                             o.auth_key, o.auth_keylen);
          xpar_unmap(&map);
          xpar_close(parity);
          return;
        }
        #endif
      }
      xpar_file * data = xpar_open(o.input_name, XPAR_O_READ);
      if (!data) FATAL_PERROR("fopen");
      decode_systematic4(data, parity, xpar_stdout,
        o.force, o.quiet, o.verbose,
        o.auth_key, o.auth_keylen);
      xpar_close(data);
    } else {
      decode_systematic4(xpar_stdin, parity, xpar_stdout,
        o.force, o.quiet, o.verbose,
        o.auth_key, o.auth_keylen);
    }
    xpar_close(parity);
    return;
  }
  xpar_file * out = open_output(o), * in = xpar_stdin;
  if (o.input_name) {
    xpar_stat_t st = validate_file(o.input_name);
    if (!o.no_map) {
      #if defined(XPAR_ALLOW_MAPPING)
      xpar_mmap map = xpar_map(o.input_name);
      if (map.map) {
        decode3(map, out, o.force, o.interlacing, o.quiet, o.verbose,
                o.auth_key, o.auth_keylen);
        xpar_unmap(&map);
        return;
      }
      #endif
    }
    if (!(in = xpar_open(o.input_name, XPAR_O_READ))) FATAL_PERROR("fopen");
  }
  decode4(in, out, o.force, o.interlacing, o.quiet, o.verbose,
          o.auth_key, o.auth_keylen);
}
void do_joint_test(joint_options_t o) {
  int bad;
  if (o.interlacing == 4) {
    if (!o.output_name)
      FATAL("Systematic mode requires a parity .xpa file.");
    xpar_file * parity = xpar_open(o.output_name, XPAR_O_READ);
    if (!parity) FATAL_PERROR("fopen");
    if (o.input_name) {
      validate_file(o.input_name);
      if (!o.no_map) {
        #if defined(XPAR_ALLOW_MAPPING)
        xpar_mmap map = xpar_map(o.input_name);
        if (map.map) {
          bad = test_systematic3(map, parity, o.quiet, o.verbose,
                                 o.auth_key, o.auth_keylen);
          xpar_unmap(&map);
          xpar_close(parity);
          goto done;
        }
        #endif
      }
      xpar_file * data = xpar_open(o.input_name, XPAR_O_READ);
      if (!data) FATAL_PERROR("fopen");
      bad = test_systematic4(data, parity, o.quiet, o.verbose,
                             o.auth_key, o.auth_keylen);
      xpar_close(data);
    } else {
      bad = test_systematic4(xpar_stdin, parity, o.quiet, o.verbose,
                             o.auth_key, o.auth_keylen);
    }
    xpar_close(parity);
    goto done;
  }
  if (o.input_name) {
    validate_file(o.input_name);
    if (!o.no_map) {
      #if defined(XPAR_ALLOW_MAPPING)
      xpar_mmap map = xpar_map(o.input_name);
      if (map.map) {
        bad = test3(map, o.interlacing, o.quiet, o.verbose,
                    o.auth_key, o.auth_keylen);
        xpar_unmap(&map);
        goto done;
      }
      #endif
    }
    xpar_file * in = xpar_open(o.input_name, XPAR_O_READ);
    if (!in) FATAL_PERROR("fopen");
    bad = test4(in, o.interlacing, o.quiet, o.verbose,
                o.auth_key, o.auth_keylen);
    xpar_close(in);
  } else {
    bad = test4(xpar_stdin, o.interlacing, o.quiet, o.verbose,
                o.auth_key, o.auth_keylen);
  }
done:
  if (bad) xpar_exit(1);
}