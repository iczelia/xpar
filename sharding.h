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

#ifndef _SHARDING_H_
#define _SHARDING_H_

#include "common.h"
#include "platform.h"
#include "crc32c.h"

#ifdef HAVE_BLAKE2B
  #include "blake2b.h"
#endif


/*  Vandermonde / Log-time sharded mode: shared encode/decode utilities.  */
#define MAX_DATA_SHARDS 128
#define MAX_PARITY_SHARDS 64
#define MAX_TOTAL_SHARDS (MAX_DATA_SHARDS + MAX_PARITY_SHARDS)

typedef struct {
  const char * input_name, * output_prefix;
  bool force, quiet, verbose, no_map;
  u8 dshards, pshards;
  /*  INTEGRITY_CRC32C or _BLAKE2B. With key: authenticated MAC.  */
  int integrity;
  const u8 * auth_key;
  sz auth_keylen;
} sharded_encoding_options_t;

typedef struct {
  const char * output_file, ** input_files;
  bool force, quiet, verbose, no_map;
  sz n_input_shards;
  const u8 * auth_key;
  sz auth_keylen;
} sharded_decoding_options_t;

/*  v1.x shard header. Version byte picks plain(0x01,20B,CRC32C),
    hash(0x81,32B,BLAKE2b-128), or keyed(0xC1,32B,BLAKE2b-128 MAC).
    Layout: magic[4] | ver | dshards | pshards | shard# | total_size(u64 BE)
            | tag(4 or 16) | body | EOS. All shards in a set share the mode.
    EOS trailer: magic "XPAE" | ver | shard# | 00 | 00 | tag (4 or 16).
    EOS MAC domain: header[0..16) || eos[0..8).  */
#define SHARD_HEADER_SIZE         20
#define SHARD_HEADER_BLAKE2B_SIZE 32
#define SHARD_EOS_SIZE            12
#define SHARD_EOS_BLAKE2B_SIZE    24
#define SHARD_VERSION             1
#define SHARD_VERSION_BLAKE2B     0x80
#define SHARD_VERSION_KEYED       0x40
#define SHARD_VERSION_MASK        0x3F
static inline sz shard_eos_size(bool has_b2b) {
  return has_b2b ? SHARD_EOS_BLAKE2B_SIZE : SHARD_EOS_SIZE;
}
typedef struct {
  bool valid, mapped, auth; u32 crc; u8 * buf;
  u8 shard_number, dshards, pshards;
  sz shard_size, total_size, hdr_size;
#if defined(XPAR_ALLOW_MAPPING)
  xpar_mmap map;
#endif
} sharded_hv_result_t;
static void unmap_shard(sharded_hv_result_t * res) {
  if (res->mapped && res->buf) {
    #if defined(XPAR_ALLOW_MAPPING)
    xpar_unmap(&res->map);
    res->buf = NULL;  res->mapped = false;
    #endif
  } else {
    xpar_free(res->buf);  res->buf = NULL;
  }
}
static u8 * most_frequent(u8 * tab, sz nmemb, sz size) {
  xpar_assert(nmemb >= 1);  xpar_assert(size <= 16);
  u8 tmp[16]; sz i, j;
  Fi0(nmemb, 1,
    for (j = i;
         j && xpar_memcmp(&tab[j * size], &tab[(j - 1) * size], size) < 0;
         j--) {
      xpar_memcpy(tmp, tab + j * size, size);
      xpar_memcpy(tab + j * size, tab + (j - 1) * size, size);
      xpar_memcpy(tab + (j - 1) * size, tmp, size);
    }
  )
  u8 * best = tab;  sz best_count = 1, current_count = 1;
  Fi0(nmemb, 1,
    if (!xpar_memcmp(tab + i * size, tab + (i - 1) * size, size))
      current_count++;
    else {
      if (current_count > best_count)
        best = tab + (i - 1) * size, best_count = current_count;
      current_count = 1;
    }
  )
  return best;
}

/*  Validate EOS trailer at buf[0..eos_size). Checks magic, version/shard#
    match header, and MAC over header[0..16) || eos[0..8).  */
static bool validate_eos_marker(const u8 * buf, const u8 header[16],
    bool has_b2b, bool has_key,
    const u8 * key, sz keylen) {
  if (buf[0] != 'X' || buf[1] != 'P' || buf[2] != 'A' || buf[3] != 'E')
    return false;
  if (buf[4] != header[4] || buf[5] != header[7]) return false;
  if (has_b2b) {
#ifdef HAVE_BLAKE2B
    u8 tag[16];
    blake2b_state s;
    if (has_key) blake2b_init_key(&s, 16, key, keylen);
    else         blake2b_init(&s, 16);
    blake2b_update(&s, header, 16);
    blake2b_update(&s, buf, 8);
    blake2b_final(&s, tag, 16);
    u8 d = 0;
    Fi(16, d |= tag[i] ^ buf[8 + i])
    return d == 0;
#else
    (void) key; (void) keylen;  return false;
#endif
  }
  u32 c = crc32c_partial(0xFFFFFFFFL, header, 16);
  c = crc32c_partial(c, buf, 8) ^ 0xFFFFFFFFL;
  u32 stored = 0;
  Fi(4, stored |= ((u32) buf[8 + i]) << (24 - 8 * i))
  return c == stored;
}

/*  Unpack+verify shard header; fills res. True iff magic, version,
    tag, keyed-flag, and EOS marker all match.  */
static bool unpack_shard_header(const u8 * buf, sz file_size,
    sharded_decoding_options_t opt, const char * file_name,
    const char * magic, sharded_hv_result_t * res) {
  if (file_size < SHARD_HEADER_SIZE) return false;
  if (xpar_memcmp(buf, magic, 4)) return false;
  u8 vb = buf[4];
  bool has_b2b = (vb & SHARD_VERSION_BLAKE2B) != 0;
  bool has_key = (vb & SHARD_VERSION_KEYED) != 0;
  u8 base_ver = vb & SHARD_VERSION_MASK;
  if (base_ver != SHARD_VERSION) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr, "Shard `%s': unsupported version %u.\n",
        file_name, base_ver);
    return false;
  }
  if (has_key && !has_b2b) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr,
        "Shard `%s': invalid keyed-without-BLAKE2b mode.\n", file_name);
    return false;
  }
  sz hdr_size = has_b2b ? SHARD_HEADER_BLAKE2B_SIZE : SHARD_HEADER_SIZE;
  sz eos_sz = shard_eos_size(has_b2b);
  if (file_size < hdr_size + eos_sz) return false;
  if (has_key && !opt.auth_keylen)
    FATAL("Shard `%s' is authenticated; pass --auth=<keyfile>.", file_name);
  if (!has_key && opt.auth_keylen)
    FATAL("Shard `%s' is not authenticated; drop --auth.", file_name);
  res->dshards = buf[5];
  res->pshards = buf[6];
  res->shard_number = buf[7];
  if (res->dshards < 1 || res->dshards > MAX_DATA_SHARDS
   || res->pshards < 1 || res->pshards > MAX_PARITY_SHARDS
   || res->dshards + res->pshards > MAX_TOTAL_SHARDS
   || res->shard_number >= res->dshards + res->pshards) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr,
        "Shard `%s': invalid layout "
        "(dshards=%u, pshards=%u, shard#=%u).\n",
        file_name, res->dshards, res->pshards, res->shard_number);
    return false;
  }
  /*  Read into u64 (sz may be 32-bit; shifts >=32 would be UB),
      check for overflow, then narrow.  */
  u64 ts = 0;
  Fi(8, ts |= ((u64) buf[8 + i]) << (56 - 8 * i))
  if (SIZEOF_SIZE_T == 4) {
    if (ts >> 32) {
      if (!opt.quiet)
        xpar_fprintf(xpar_stderr,
          "Shard `%s' is too large for 32-bit architectures.\n", file_name);
      return false;
    }
  }
  res->total_size = (sz) ts;
  res->auth = has_key;
  res->hdr_size = hdr_size;
  res->shard_size = file_size - hdr_size - eos_sz;
  const u8 * body = buf + hdr_size;
  const u8 * eos = buf + file_size - eos_sz;
  if (has_b2b) {
#ifdef HAVE_BLAKE2B
    u8 tag[16];
    blake2b_state s;
    if (has_key) blake2b_init_key(&s, 16, opt.auth_key, opt.auth_keylen);
    else         blake2b_init(&s, 16);
    blake2b_update(&s, buf, 16);
    blake2b_update(&s, body, res->shard_size);
    blake2b_final(&s, tag, 16);
    u8 d = 0;
    Fi(16, d |= tag[i] ^ buf[16 + i])
    if (d != 0) return false;
#else
    FATAL("Shard `%s' uses BLAKE2b but that support was disabled at "
          "configure time.", file_name);
#endif
  } else {
    Fi(4, res->crc |= ((u32) buf[16 + i]) << (24 - 8 * i))
    u32 c = crc32c_partial(0xFFFFFFFFL, (u8 *) buf, 16);
    c = crc32c_partial(c, (u8 *) body, res->shard_size) ^ 0xFFFFFFFFL;
    if (c != res->crc) return false;
  }
  if (!validate_eos_marker(eos, buf, has_b2b, has_key,
                           opt.auth_key, opt.auth_keylen)) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr,
        "Shard `%s': missing or invalid EOS trailer.\n", file_name);
    return false;
  }
  return true;
}

static sharded_hv_result_t validate_shard_header(const char * file_name,
    sharded_decoding_options_t opt, const char * hdr) {
  sharded_hv_result_t res;  xpar_memset(&res, 0, sizeof(res));
  if (!opt.no_map) {
    #if defined(XPAR_ALLOW_MAPPING)
    xpar_mmap map = xpar_map(file_name);
    if (map.map) {
      if (!unpack_shard_header(map.map, map.size, opt, file_name, hdr, &res)) {
        xpar_unmap(&map);  return res;
      }
      res.valid = true;  res.mapped = true;
      res.buf = map.map;   res.map = map;  return res;
    }
    #endif
  }
  xpar_file * in = xpar_open(file_name, XPAR_O_READ);
  if (!in) return res;
  xpar_seek(in, 0, XPAR_SEEK_END);
  i64 size_raw = xpar_tell(in);
  if (size_raw < 0) { xpar_close(in);  return res; }
  sz size = (sz) size_raw;
  xpar_seek(in, 0, XPAR_SEEK_SET);
  if (size < SHARD_HEADER_SIZE) { xpar_close(in);  return res; }
  u8 * buffer = (u8 *) xpar_malloc(size);
  if (xpar_xread(in, buffer, size) != size) {
    xpar_free(buffer);  xpar_close(in);  return res;
  }
  xpar_close(in);
  if (!unpack_shard_header(buffer, size, opt, file_name, hdr, &res)) {
    xpar_free(buffer);  return res;
  }
  res.buf = buffer; res.valid = true;  return res;
}

/*  Majority-vote the four consensus fields (dshards, pshards, total_size,
    shard_size) over the currently-valid shards. Zeroed result if none.  */
typedef struct {
  u8 dshards, pshards;
  sz total_size, shard_size;
} shard_consensus_t;
static shard_consensus_t consensus_of_valid(
    sharded_hv_result_t * res, sz n) {
  shard_consensus_t c = { 0, 0, 0, 0 };
  u8  b8[MAX_TOTAL_SHARDS];  sz bs[MAX_TOTAL_SHARDS];  int k;
  k = 0;  Fi(n, if (res[i].valid) b8[k++] = res[i].dshards);
  if (!k) return c;
  c.dshards = *(u8 *) most_frequent(b8, k, 1);
  k = 0;  Fi(n, if (res[i].valid) b8[k++] = res[i].pshards);
  c.pshards = *(u8 *) most_frequent(b8, k, 1);
  k = 0;  Fi(n, if (res[i].valid) bs[k++] = res[i].total_size);
  c.total_size = *(sz *) most_frequent((u8 *) bs, k, sizeof(sz));
  k = 0;  Fi(n, if (res[i].valid) bs[k++] = res[i].shard_size);
  c.shard_size = *(sz *) most_frequent((u8 *) bs, k, sizeof(sz));
  return c;
}

/*  Build shard header into dst, tag over header[0..16)+body.
    Returns bytes written: CRC32C=20, BLAKE2b(keyed or not)=32.  */
static sz pack_shard_header(u8 dst[SHARD_HEADER_BLAKE2B_SIZE],
                            const char * magic,
                            u8 dshards, u8 pshards, u8 shard_number,
                            sz total_size, const u8 * body, sz body_len,
                            int algo, const u8 * key, sz keylen) {
  xpar_memcpy(dst, magic, 4);
  u8 vb = SHARD_VERSION;
  if (algo == INTEGRITY_BLAKE2B) {
    vb |= SHARD_VERSION_BLAKE2B;
    if (keylen) vb |= SHARD_VERSION_KEYED;
  } else if (keylen) {
    FATAL("Keyed MAC requested with a non-BLAKE2b integrity algorithm.");
  }
  dst[4] = vb;
  dst[5] = dshards;
  dst[6] = pshards;
  dst[7] = shard_number;
  /*  Widen to u64 before shifting; sz may be 32-bit.  */
  { u64 ts = (u64) total_size;
    Fi(8, dst[8 + i] = (u8)(ts >> (56 - 8 * i))); }
  if (algo == INTEGRITY_BLAKE2B) {
#ifdef HAVE_BLAKE2B
    u8 tag[16];
    blake2b_state s;
    if (keylen) blake2b_init_key(&s, 16, key, keylen);
    else        blake2b_init(&s, 16);
    blake2b_update(&s, dst, 16);
    blake2b_update(&s, body, body_len);
    blake2b_final(&s, tag, 16);
    xpar_memcpy(dst + 16, tag, 16);
    return SHARD_HEADER_BLAKE2B_SIZE;
#else
    (void) body; (void) body_len; (void) key;
    FATAL("BLAKE2b requested but support was disabled at configure time.");
#endif
  }
  u32 c = crc32c_partial(0xFFFFFFFFL, dst, 16);
  c = crc32c_partial(c, (u8 *) body, body_len) ^ 0xFFFFFFFFL;
  Fi(4, dst[16 + i] = (u8)(c >> (24 - 8 * i)));
  return SHARD_HEADER_SIZE;
}

/*  Build EOS trailer into dst, tag over header[0..16)+eos[0..8).
    Returns bytes written: CRC32C=12, BLAKE2b=24.  */
static sz pack_eos_marker(u8 dst[SHARD_EOS_BLAKE2B_SIZE],
                          const u8 header[16], int algo,
                          const u8 * key, sz keylen) {
  dst[0] = 'X'; dst[1] = 'P'; dst[2] = 'A'; dst[3] = 'E';
  dst[4] = header[4];  /*  version byte  */
  dst[5] = header[7];  /*  shard_number  */
  dst[6] = dst[7] = 0;
  if (algo == INTEGRITY_BLAKE2B) {
#ifdef HAVE_BLAKE2B
    u8 tag[16];
    blake2b_state s;
    if (keylen) blake2b_init_key(&s, 16, key, keylen);
    else        blake2b_init(&s, 16);
    blake2b_update(&s, header, 16);
    blake2b_update(&s, dst, 8);
    blake2b_final(&s, tag, 16);
    xpar_memcpy(dst + 8, tag, 16);
    return SHARD_EOS_BLAKE2B_SIZE;
#else
    (void) key;
    FATAL("BLAKE2b requested but disabled at configure time.");
#endif
  }
  u32 c = crc32c_partial(0xFFFFFFFFL, header, 16);
  c = crc32c_partial(c, dst, 8) ^ 0xFFFFFFFFL;
  Fi(4, dst[8 + i] = (u8)(c >> (24 - 8 * i)));
  return SHARD_EOS_SIZE;
}

#endif
