/*  xpar: sliding-window misplaced-data search.

    Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

#include "resync.h"

#include "crc32c.h"

#define RS_IO       ((sz) 1 << 20)
#define RS_MAX_BINS ((u32) 1 << 20)
#define RS_NONE     UINT32_MAX

typedef struct {
  i64 delta;
  u64 count;
  bool used;
} rs_bin;

typedef struct {
  const xpar_resync_probe * probe;
  u32 probe_count;
  u32 * bucket, * next;
  u8 * unique;
  u32 mask;
} rs_index;

typedef bool (*rs_hit_fn)(void *, u32, u64);

static u32 rs_pow2(u32 n) {
  u32 v = 1;
  while (v < n && v < (UINT32_MAX >> 1)) v <<= 1;
  return v;
}

static u32 rs_scale_sat(u32 n, u32 scale) {
  return n > UINT32_MAX / scale ? UINT32_MAX : n * scale;
}

static u32 rs_hash32(u32 x) {
  x ^= x >> 16;  x *= 0x7feb352du;
  x ^= x >> 15;  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

static u32 rs_hash64(i64 v) {
  u64 x = (u64) v;
  x ^= x >> 30;  x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27;  x *= UINT64_C(0x94d049bb133111eb);
  return (u32) (x ^ (x >> 31));
}

static void rs_index_init(rs_index * x, const xpar_resync_probe * p, u32 n) {
  u32 slots = rs_pow2(MAX(rs_scale_sat(n, 2), 2u)), i;
  xpar_memset(x, 0, sizeof *x);
  x->probe = p;  x->probe_count = n;  x->mask = slots - 1;
  x->bucket = (u32 *) xpar_alloc_raw((sz) slots * sizeof(u32));
  x->next = (u32 *) xpar_alloc_raw((sz) MAX(n, 1u) * sizeof(u32));
  x->unique = (u8 *) xpar_calloc(MAX(n, 1u), 1);
  for (i = 0; i < slots; i++) x->bucket[i] = RS_NONE;
  for (i = 0; i < n; i++) {
    u32 b = rs_hash32(p[i].crc) & x->mask;
    x->next[i] = x->bucket[b];
    x->bucket[b] = i;
  }
  for (i = 0; i < n; i++) {
    u32 b = rs_hash32(p[i].crc) & x->mask, q, count = 0;
    for (q = x->bucket[b]; q != RS_NONE; q = x->next[q])
      if (p[q].crc == p[i].crc) count++;
    x->unique[i] = count == 1;
  }
}

static void rs_index_free(rs_index * x) {
  xpar_free(x->bucket);  xpar_free(x->next);  xpar_free(x->unique);
}

static bool rs_delta(u64 physical, u64 expected, i64 * out) {
  u64 d;
  if (physical >= expected) {
    d = physical - expected;
    if (d > (u64) INT64_MAX) return false;
    *out = (i64) d;
  } else {
    d = expected - physical;
    if (d > (u64) INT64_MAX) return false;
    *out = -(i64) d;
  }
  return true;
}

static bool rs_within(i64 delta, u64 limit) {
  u64 n;
  if (!limit) return true;
  n = delta < 0 ? (u64) (-(delta + 1)) + 1 : (u64) delta;
  return n <= limit;
}

static bool rs_scan(xpar_file * f, u64 size, u64 window, const rs_index * ix,
                    u32 step, u64 max_delta, rs_hit_fn hit, void * user) {
  xpar_crc32c_roll roll;
  u8 * ring, * input;
  u64 pos, at;
  u32 crc;
  if (!ix->probe_count || !window || window > size || window > (u64) (sz) -1)
    return true;
  ring = (u8 *) xpar_alloc_raw((sz) window);
  input = (u8 *) xpar_alloc_raw(RS_IO);
  if (xpar_pread(f, ring, (sz) window, 0) != (sz) window) {
    xpar_free(ring);  xpar_free(input);  return false;
  }
  xpar_crc32c_roll_init(&roll, (sz) window);
  crc = xpar_crc32c(0, ring, (sz) window);
  pos = 0;
  if (pos % step == 0) {
    u32 b = rs_hash32(crc) & ix->mask, q;
    for (q = ix->bucket[b]; q != RS_NONE; q = ix->next[q]) {
      i64 delta;
      if (!ix->unique[q] || ix->probe[q].crc != crc ||
          !rs_delta(pos, ix->probe[q].expected, &delta) ||
          !rs_within(delta, max_delta)) continue;
      if (!hit(user, q, pos)) goto done;
    }
  }
  while (pos + window < size) {
    at = pos + window;
    { sz n = (sz) MIN((u64) RS_IO, size - at), i;
      if (xpar_pread(f, input, n, at) != n) {
        xpar_free(ring);  xpar_free(input);  return false;
      }
      for (i = 0; i < n; i++) {
        u64 slot = pos % window;
        crc = xpar_crc32c_roll_step(&roll, crc, ring[slot], input[i]);
        ring[slot] = input[i];
        pos++;
        if (pos % step == 0) {
          u32 b = rs_hash32(crc) & ix->mask, q;
          for (q = ix->bucket[b]; q != RS_NONE; q = ix->next[q]) {
            i64 delta;
            if (!ix->unique[q] || ix->probe[q].crc != crc ||
                !rs_delta(pos, ix->probe[q].expected, &delta) ||
                !rs_within(delta, max_delta)) continue;
            if (!hit(user, q, pos)) goto done;
          }
        }
      }
    }
  }
done:
  xpar_free(ring);  xpar_free(input);
  return true;
}

typedef struct {
  rs_bin * bin;
  u32 mask, used, limit;
  u64 candidates;
  bool overflow;
  const xpar_resync_probe * probe;
} rs_hist;

static bool rs_hist_hit(void * user, u32 probe, u64 physical) {
  rs_hist * h = (rs_hist *) user;
  i64 delta;
  u32 at;
  if (!rs_delta(physical, h->probe[probe].expected, &delta)) return true;
  h->candidates++;
  at = rs_hash64(delta) & h->mask;
  while (h->bin[at].used && h->bin[at].delta != delta)
    at = (at + 1) & h->mask;
  if (!h->bin[at].used) {
    if (h->used >= h->limit) { h->overflow = true;  return false; }
    h->bin[at].used = true;  h->bin[at].delta = delta;  h->used++;
  }
  h->bin[at].count++;
  return true;
}

static void rs_insert_best(xpar_resync_result * out, i64 delta, u64 count) {
  u32 at = out->count, i;
  if (at < XPAR_RESYNC_DELTAS) out->count++;
  else if (count <= out->delta[at - 1].votes) return;
  else at--;
  while (at && count > out->delta[at - 1].votes) {
    if (at < XPAR_RESYNC_DELTAS) out->delta[at] = out->delta[at - 1];
    at--;
  }
  out->delta[at].delta = delta;
  out->delta[at].votes = count;
  for (i = out->count; i < XPAR_RESYNC_DELTAS; i++) {
    out->delta[i].delta = 0;  out->delta[i].votes = 0;
  }
}

bool xpar_resync_search(xpar_file * f, u64 file_size, u64 window,
                        const xpar_resync_probe * probe, u32 probe_count,
                        u32 step, u64 max_delta, xpar_resync_result * out) {
  rs_index ix;
  rs_hist h;
  u32 slots, i;
  xpar_memset(out, 0, sizeof *out);
  if (!probe_count || !window || window > file_size) return true;
  if (!step) step = 1;
  slots = rs_pow2(MIN(MAX(rs_scale_sat(probe_count, 8), 1024u),
                      RS_MAX_BINS * 2));
  xpar_memset(&h, 0, sizeof h);
  h.bin = (rs_bin *) xpar_calloc(slots, sizeof(rs_bin));
  h.mask = slots - 1;
  h.limit = MIN(RS_MAX_BINS, slots - slots / 4);
  h.probe = probe;
  rs_index_init(&ix, probe, probe_count);
  if (!rs_scan(f, file_size, window, &ix, step, max_delta,
               rs_hist_hit, &h)) {
    rs_index_free(&ix);  xpar_free(h.bin);  return false;
  }
  for (i = 0; i < slots; i++)
    if (h.bin[i].used)
      rs_insert_best(out, h.bin[i].delta, h.bin[i].count);
  out->candidates = h.candidates;
  out->overflow = h.overflow;
  if (out->count) {
    u64 first = out->delta[0].votes;
    u64 second = out->count > 1 ? out->delta[1].votes : 0;
    out->dominant = first && (!second || second <= first / 2);
  }
  rs_index_free(&ix);  xpar_free(h.bin);
  return true;
}

typedef struct {
  xpar_resync_confirm_fn confirm;
  void * user;
  u64 * located;
  u64 confirms;
} rs_exhaust;

static bool rs_exhaust_hit(void * user, u32 probe, u64 physical) {
  rs_exhaust * x = (rs_exhaust *) user;
  if (x->located[probe] != UINT64_MAX) return true;
  x->confirms++;
  if (x->confirm(x->user, probe, physical)) x->located[probe] = physical;
  return true;
}

u64 xpar_resync_exhaustive(xpar_file * f, u64 file_size, u64 window,
                           const xpar_resync_probe * probe, u32 probe_count,
                           u32 step, u64 max_delta,
                           xpar_resync_confirm_fn confirm, void * user,
                           u64 * located) {
  rs_index ix;
  rs_exhaust x;
  u32 i;
  for (i = 0; i < probe_count; i++) located[i] = UINT64_MAX;
  x.confirm = confirm;  x.user = user;  x.located = located;  x.confirms = 0;
  if (!probe_count || !window || window > file_size) return 0;
  if (!step) step = 1;
  rs_index_init(&ix, probe, probe_count);
  if (!rs_scan(f, file_size, window, &ix, step, max_delta,
               rs_exhaust_hit, &x)) x.confirms = 0;
  rs_index_free(&ix);
  return x.confirms;
}
