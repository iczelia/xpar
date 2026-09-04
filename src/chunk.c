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

/*  content-defined chunking and transient chunk fingerprints.  */

#include "chunk.h"
#include "pathname.h"
#include "blake3.h"
#include "container.h"
#include "crc32c.h"
#include "port.h"
#include "port-fs.h"

#define CHUNK_IO ((sz) 1 << 16)

static u64 gear_mix(u64 x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

static u64 low_mask(u32 bits) {
  if (bits >= 63) return UINT64_MAX;
  return ((u64) 1 << bits) - 1;
}

bool xpar_chunk_file(const char * path, u64 average,
                     xpar_chunk_emit emit, void * user,
                     u8 content_hash[32], u8 prefix_hash[16]) {
  xpar_file * f;
  xpar_blake3_t ch, whole, prefix;
  u64 gear[256], h = 0, file_at = 0, chunk_at = 0;
  u64 min, max, strong, weak;
  u64 chunk_len = 0;
  u32 bits = 0, b;
  u8 hash[16];
  u8 * buf;
  sz got;
  u64 prefix_left = 16384;

  if (!average) average = (u64) 1 << 20;
  if (average > ((u64) 1 << 30)) average = (u64) 1 << 30;
  min = MAX(average / 4, 1);
  max = MIN(average * 4, (u64) UINT32_MAX);
  while (bits < 62 && ((u64) 1 << bits) < average) bits++;
  strong = low_mask(bits < 62 ? bits + 1 : bits);
  weak = low_mask(bits > 1 ? bits - 1 : 1);
  for (b = 0; b < 256; b++) gear[b] = gear_mix(b);

  f = xpar_open(path, XPAR_O_RDONLY);
  if (!f) return false;
  buf = (u8 *) xpar_alloc_raw(CHUNK_IO);
  xpar_blake3_init(&ch);
  xpar_blake3_init(&whole);
  xpar_blake3_init(&prefix);
  while ((got = xpar_xread(f, buf, CHUNK_IO)) != 0) {
    sz i, run = 0;
    xpar_blake3_update(&whole, buf, got);
    if (prefix_left) {
      sz take = (sz) MIN(prefix_left, got);
      xpar_blake3_update(&prefix, buf, take);
      prefix_left -= take;
    }
    Fi(got,
      bool cut = false;
      h = (h << 1) + gear[buf[i]];
      chunk_len++;
      if (chunk_len >= max) cut = true;
      else if (chunk_len >= min) {
        u64 mask = chunk_len < average ? strong : weak;
        cut = (h & mask) == 0;
      }
      if (!cut) continue;
      xpar_blake3_update(&ch, buf + run, i + 1 - run);
      xpar_blake3_final(&ch, hash, sizeof hash);
      if (!emit(user, chunk_at, (u32) chunk_len, hash)) {
        xpar_free(buf);  xpar_xclose(f);
        return false;
      }
      chunk_at += chunk_len;
      chunk_len = 0;  h = 0;  run = i + 1;
      xpar_blake3_init(&ch));
    if (run < got) xpar_blake3_update(&ch, buf + run, got - run);
    file_at += got;
  }
  if (chunk_len) {
    xpar_blake3_final(&ch, hash, sizeof hash);
    if (!emit(user, chunk_at, (u32) chunk_len, hash)) {
      xpar_free(buf);  xpar_xclose(f);
      return false;
    }
  }
  xpar_assert(chunk_at + chunk_len == file_at);
  if (content_hash) xpar_blake3_final(&whole, content_hash, 32);
  if (prefix_hash) xpar_blake3_final(&prefix, prefix_hash, 16);
  xpar_free(buf);  xpar_xclose(f);
  return true;
}

static u32 chunk_home(const u8 hash[16], u32 length, u32 capacity) {
  u64 h = xpar_rd64(hash) ^ xpar_rd64(hash + 8) ^ length;
  h ^= h >> 33;  h *= UINT64_C(0xff51afd7ed558ccd);
  h ^= h >> 33;
  return (u32) h & (capacity - 1);
}

static xpar_chunk_slot * chunk_probe(xpar_chunk_index * x,
                                     const u8 hash[16], u32 length) {
  u32 at = chunk_home(hash, length, x->capacity);
  while (x->slot[at].length &&
         (x->slot[at].length != length ||
          xpar_memcmp(x->slot[at].hash, hash, 16)))
    at = (at + 1) & (x->capacity - 1);
  return &x->slot[at];
}

bool xpar_chunk_index_init(xpar_chunk_index * x, u64 max_bytes) {
  xpar_memset(x, 0, sizeof *x);
  x->max_bytes = max_bytes;
  x->capacity = 16;
  if (max_bytes < (u64) x->capacity * sizeof(xpar_chunk_slot)) return false;
  x->slot = (xpar_chunk_slot *) xpar_calloc(x->capacity, sizeof *x->slot);
  return true;
}

void xpar_chunk_index_free(xpar_chunk_index * x) {
  xpar_free(x->slot);
  xpar_memset(x, 0, sizeof *x);
}

xpar_chunk_slot * xpar_chunk_index_find(xpar_chunk_index * x,
                                        const u8 hash[16], u32 length) {
  xpar_chunk_slot * s = chunk_probe(x, hash, length);
  return s->length ? s : NULL;
}

static bool chunk_grow(xpar_chunk_index * x) {
  xpar_chunk_slot * old = x->slot;
  u32 oldn = x->capacity, i;
  if ((u64) oldn * 2 * sizeof *old > x->max_bytes) return false;
  x->capacity *= 2;
  x->slot = (xpar_chunk_slot *) xpar_calloc(x->capacity, sizeof *x->slot);
  Fi(oldn, if (old[i].length) *chunk_probe(x, old[i].hash, old[i].length) = old[i]);
  xpar_free(old);
  return true;
}

bool xpar_chunk_index_put(xpar_chunk_index * x, const u8 hash[16], u32 length,
                          u64 stream_offset) {
  xpar_chunk_slot * s;
  if (!length) return true;
  s = chunk_probe(x, hash, length);
  if (!s->length && (u64) (x->count + 1) * 4 >
                    (u64) x->capacity * 3) {
    if (!chunk_grow(x)) return false;
    s = chunk_probe(x, hash, length);
  }
  if (!s->length) x->count++;
  xpar_memcpy(s->hash, hash, 16);
  s->stream_offset = stream_offset;
  s->length = length;
  s->refs = 1;
  s->trust = 1;
  return true;
}

#define CACHE_HDR 64u
#define CACHE_REC 40u

bool xpar_chunk_cache_load(const char * path,
                           const u8 set_id[XPAR_SET_ID_LEN], u64 average,
                           xpar_chunk_index * x) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  u8 h[CACHE_HDR], r[CACHE_REC];
  i64 size;
  u64 count, i, need;
  u32 payload_crc = 0;
  bool ok = false;
  if (!f) return false;
  size = xpar_size(f);
  if (size < CACHE_HDR || (u64) size > (u64) (sz) -1) goto done;
  if (xpar_xread(f, h, sizeof h) != sizeof h) goto done;
  count = xpar_rd64(h + 40);
  if (count > ((u64) -1 - CACHE_HDR) / CACHE_REC) goto done;
  need = CACHE_HDR + count * CACHE_REC;
  xpar_crc32c_init();
  if ((u64) size != need || xpar_memcmp(h, "XPARIDX\0", 8) ||
      xpar_rd32(h + 8) != 1 || xpar_rd32(h + 12) ||
      xpar_rd64(h + 16) != average ||
      xpar_memcmp(h + 24, set_id, XPAR_SET_ID_LEN) ||
      xpar_rd32(h + 52) != xpar_crc32c(0, h, 52) || xpar_rd64(h + 56))
    goto done;
  ok = true;
  Fi(count,
    u32 len;
    if (xpar_xread(f, r, sizeof r) != sizeof r) { ok = false;  break; }
    payload_crc = xpar_crc32c(payload_crc, r, sizeof r);
    len = xpar_rd32(r + 24);
    if (!len || xpar_rd32(r + 28) ||
        !xpar_rd64(r + 32) ||
        !xpar_chunk_index_put(x, r, len, xpar_rd64(r + 16))) {
      ok = false;
      break;
    }
    chunk_probe(x, r, len)->refs = xpar_rd64(r + 32);
    chunk_probe(x, r, len)->trust = 0);
  if (ok && payload_crc != xpar_rd32(h + 48)) ok = false;
done:
  xpar_xclose(f);
  return ok;
}

bool xpar_chunk_cache_write(const char * path,
                            const u8 set_id[XPAR_SET_ID_LEN], u64 average,
                            const xpar_chunk_index * x) {
  xpar_file * f = NULL;
  char * tmp = NULL;
  u8 h[CACHE_HDR], r[CACHE_REC];
  u32 i, suffix, payload_crc = 0;
  xpar_memset(h, 0, sizeof h);
  xpar_memcpy(h, "XPARIDX\0", 8);
  xpar_wr32(h + 8, 1);
  xpar_wr64(h + 16, average);
  xpar_memcpy(h + 24, set_id, XPAR_SET_ID_LEN);
  xpar_wr64(h + 40, x->count);
  xpar_crc32c_init();
  /*  Compute the payload checksum without materialising a second copy of
      the index. This cache is explicitly bounded by --dedup-memory.  */
  Fi(x->capacity,
    const xpar_chunk_slot * s = &x->slot[i];
    if (!s->length) continue;
    xpar_memset(r, 0, sizeof r);
    xpar_memcpy(r, s->hash, 16);
    xpar_wr64(r + 16, s->stream_offset);
    xpar_wr32(r + 24, s->length);
    xpar_wr64(r + 32, s->refs);
    payload_crc = xpar_crc32c(payload_crc, r, sizeof r));
  xpar_wr32(h + 48, payload_crc);
  xpar_wr32(h + 52, xpar_crc32c(0, h, 52));
  for (suffix = 0; suffix < 1000; suffix++) {
#if defined(XPAR_DOS) || defined(__MSDOS__)
    tmp = xpar_dos_numbered(path, "CHK", "TMP", suffix);
#else
    xpar_asprintf(&tmp, "%s.tmp-%03" PRIu32, path, suffix);
#endif
    f = xpar_open(tmp, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_EXCL);
    if (f) break;
    xpar_free(tmp);  tmp = NULL;
  }
  if (!f) return false;
  xpar_xwrite(f, h, sizeof h);
  Fi(x->capacity,
    const xpar_chunk_slot * s = &x->slot[i];
    if (!s->length) continue;
    xpar_memset(r, 0, sizeof r);
    xpar_memcpy(r, s->hash, 16);
    xpar_wr64(r + 16, s->stream_offset);
    xpar_wr32(r + 24, s->length);
    xpar_wr64(r + 32, s->refs);
    xpar_xwrite(f, r, sizeof r));
  if (xpar_fsync(f) != 0) { xpar_xclose(f);  xpar_remove(tmp);  xpar_free(tmp);  return false; }
  xpar_xclose(f);
  if (xpar_rename(tmp, path) != 0) { xpar_remove(tmp);  xpar_free(tmp);  return false; }
  if (xpar_fsync_dir(path) != 0) { xpar_free(tmp);  return false; }
  xpar_free(tmp);
  return true;
}

bool xpar_chunk_cache_rebind(const char * path,
                             const u8 set_id[XPAR_SET_ID_LEN]) {
  xpar_file * f = xpar_open(path, XPAR_O_RDWR);
  u8 h[CACHE_HDR];
  bool ok = false;
  if (!f) return false;
  xpar_crc32c_init();
  if (xpar_pread(f, h, sizeof h, 0) != sizeof h ||
      xpar_memcmp(h, "XPARIDX\0", 8) || xpar_rd32(h + 8) != 1 ||
      xpar_rd32(h + 52) != xpar_crc32c(0, h, 52)) goto done;
  xpar_memcpy(h + 24, set_id, XPAR_SET_ID_LEN);
  xpar_wr32(h + 52, xpar_crc32c(0, h, 52));
  if (xpar_pwrite(f, h, sizeof h, 0) != sizeof h) goto done;
  if (xpar_fsync(f) != 0) goto done;
  ok = true;
done:
  xpar_xclose(f);
  return ok;
}
