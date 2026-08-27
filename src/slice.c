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

/*  Stream, slice, and cell geometry.

    Z and S tile the stream; Y sets the recorded erasure granularity.  C
    is only an in-memory codec tile.  Columns may have distinct erasure
    patterns and share decode plans when those patterns match.  */

#include "slice.h"

#include "common.h"
#include "kernel/blake3.h"

/*  Choosing Z and S.  */

static u64 round_up_64(u64 v) {
  if (v > XPAR_SLICE_MAX) return XPAR_SLICE_MAX;
  return (v + 63) & ~(u64) 63;
}

const char * xpar_geom_reason(xpar_geom_status s) {
  switch (s) {
    case XPAR_GEOM_OK:        return "ok";
    case XPAR_GEOM_EXCLUSIVE: return "-s and -b are mutually exclusive";
    case XPAR_GEOM_QUANTUM:   return "slice size must be a multiple of 64";
    case XPAR_GEOM_HUGE:      return "slice size past 1 GiB; split the set";
    case XPAR_GEOM_FIELD:     return "S + R does not fit the field";
    case XPAR_GEOM_UNREACHABLE:
      return "more slices requested than the stream has bytes";
    case XPAR_GEOM_CELLS:
      return "slice exceeds 65536 cells; raise --cell or use more slices";
  }
  return "geometry refused";
}

/*  The field bound is S + R <= 2^field and S <= 2^field - 1.
    R is a hint because three of -r's four forms are a function of S, so
    a planner iterates: choose with R unknown, derive R, choose again.
    One is assumed when nothing is known, which keeps the bound honest
    rather than optimistic.  */
static u64 max_slices(u8 field_log2, u64 recovery) {
  u64 field = (u64) 1 << field_log2;
  u64 r = recovery ? recovery : 1;
  if (r >= field) return 0;
  return MIN(field - r, field - 1);
}

xpar_geom_status xpar_geom_choose(const xpar_geom_req * req,
                                  xpar_geom * out) {
  u64 z, s, cap = max_slices(req->field_log2, req->recovery);
  bool explicit_z = req->slice_size != 0;

  xpar_memset(out, 0, sizeof(*out));
  out->stream_length = req->stream_length;
  out->stream_base   = req->stream_base;
  if (req->slice_size && req->slice_count) return XPAR_GEOM_EXCLUSIVE;
  if (!cap) return XPAR_GEOM_FIELD;

  if (explicit_z) {
    z = req->slice_size;
    if (z < XPAR_SLICE_MIN || z % 64) return XPAR_GEOM_QUANTUM;
  } else if (req->slice_count) {
    if (req->slice_count > req->stream_length && req->stream_length)
      return XPAR_GEOM_UNREACHABLE;
    if (!req->stream_length) z = XPAR_SLICE_FLOOR;
    else z = round_up_64(xpar_ceil_div(req->stream_length,
                                       req->slice_count));
    if (z < XPAR_SLICE_MIN) z = XPAR_SLICE_MIN;
  } else if (!req->stream_length) {
    z = XPAR_SLICE_FLOOR;
  } else {
    z = round_up_64(req->stream_length / XPAR_SLICE_TARGET);
    if (z < XPAR_SLICE_FLOOR) z = XPAR_SLICE_FLOOR;
    if (z > XPAR_SLICE_CEIL)  z = XPAR_SLICE_CEIL;
  }

  s = req->stream_length ? xpar_ceil_div(req->stream_length, z) : 0;
  if (s > cap) {
    if (explicit_z || req->slice_count) return XPAR_GEOM_FIELD;
    z = round_up_64(xpar_ceil_div(req->stream_length, cap));
    s = xpar_ceil_div(req->stream_length, z);
    if (s > cap) return XPAR_GEOM_FIELD;
  }
  if (z > XPAR_SLICE_REFUSE || z > XPAR_SLICE_MAX) return XPAR_GEOM_HUGE;

  out->slice_size  = z;
  out->slice_count = s;
  out->cell_bytes  = xpar_cell_choose(z, req->cell_bytes,
                                      req->armour_frame);
  out->cells_per_slice = out->cell_bytes
                           ? (u32) xpar_ceil_div(z, out->cell_bytes) : 1;
  if (out->cell_bytes && xpar_ceil_div(z, out->cell_bytes) > XPAR_CELLS_MAX)
    return XPAR_GEOM_CELLS;
  return XPAR_GEOM_OK;
}

u32 xpar_cell_choose(u64 slice_size, u32 want, u32 armour_frame) {
  u64 y;
  if (slice_size < XPAR_CELL_MIN) return 0;
  y = want ? want : XPAR_CELL_DEFAULT;
  if (armour_frame) {
    u64 f = xpar_ceil_div(y, armour_frame) * armour_frame;
    if (f <= slice_size) y = f;
  }
  y = (y + 63) & ~(u64) 63;
  if (y > slice_size) y = slice_size & ~(u64) 63;
  if (y < XPAR_CELL_MIN) y = XPAR_CELL_MIN;
  if (y > slice_size) return 0;
  return (u32) y;
}

bool xpar_geom_from_setd(const xpar_setd * sd, xpar_geom * out) {
  xpar_memset(out, 0, sizeof(*out));
  if (sd->slice_size % 64 || sd->slice_size < XPAR_SLICE_MIN ||
      sd->slice_size > XPAR_SLICE_MAX) return false;
  /*  Reject slice sizes beyond the host address space.  */
  if (sd->slice_size > (u64) (sz) -1) return false;
  if (sd->data_slice_count == 0) {
    if (sd->stream_length != 0) return false;
  } else {
    u64 hi = sd->data_slice_count * sd->slice_size;
    u64 lo = (sd->data_slice_count - 1) * sd->slice_size;
    if (sd->stream_length > hi || sd->stream_length <= lo) return false;
  }
  if (sd->cell_bytes) {
    if (sd->cell_bytes < XPAR_CELL_MIN || sd->cell_bytes % 64) return false;
    if (sd->cell_bytes > sd->slice_size) return false;
    if (xpar_ceil_div(sd->slice_size, sd->cell_bytes) > XPAR_CELLS_MAX)
      return false;
  }
  out->slice_size      = sd->slice_size;
  out->slice_count     = sd->data_slice_count;
  out->stream_length   = sd->stream_length;
  out->stream_base     = sd->stream_base;
  out->cell_bytes      = sd->cell_bytes;
  out->cells_per_slice = sd->cell_bytes
                           ? (u32) xpar_ceil_div(sd->slice_size,
                                                 sd->cell_bytes) : 1;
  return true;
}

/*  Slice and cell mapping.  */

u64 xpar_slice_begin(const xpar_geom * g, u64 slice) {
  return g->stream_base + slice * g->slice_size;
}

u64 xpar_slice_of(const xpar_geom * g, u64 stream_off) {
  xpar_assert(stream_off >= g->stream_base);
  return (stream_off - g->stream_base) / g->slice_size;
}

u64 xpar_slice_bytes(const xpar_geom * g, u64 slice) {
  u64 begin = slice * g->slice_size;
  if (begin >= g->stream_length) return 0;
  return MIN(g->slice_size, g->stream_length - begin);
}

void xpar_slice_tag(const xpar_setd * sd, u64 slice, const u8 * bytes,
                    u8 * out, sz n) {
  if (sd->required_features & XPAR_FEAT_B3_SUBTREE) {
    u64 begin = sd->stream_base + slice * sd->slice_size;
    xpar_blake3_subtree_tag(bytes, (sz) sd->slice_size,
                            begin / XPAR_BLAKE3_CHUNK_LEN, out, n);
  } else {
    xpar_blake3_hash(bytes, (sz) sd->slice_size, out, n);
  }
}

void xpar_slice_tag_keyed(const xpar_setd * sd, u64 slice, const u8 * bytes,
                          const u8 * key, u8 * out, sz n) {
  if (sd->required_features & XPAR_FEAT_B3_SUBTREE) {
    u64 begin = sd->stream_base + slice * sd->slice_size;
    xpar_blake3_subtree_tag_keyed(key, bytes, (sz) sd->slice_size,
                                  begin / XPAR_BLAKE3_CHUNK_LEN, out, n);
  } else {
    xpar_blake3_hash_keyed(key, bytes, (sz) sd->slice_size, out, n);
  }
}

u32 xpar_cell_of(const xpar_geom * g, u64 stream_off) {
  u64 in = (stream_off - g->stream_base) % g->slice_size;
  if (!g->cell_bytes) return 0;
  return (u32) (in / g->cell_bytes);
}

u64 xpar_cell_begin(const xpar_geom * g, u64 slice, u32 col) {
  return xpar_slice_begin(g, slice) +
         (u64) col * (g->cell_bytes ? g->cell_bytes : g->slice_size);
}

u64 xpar_cell_size(const xpar_geom * g, u32 col) {
  u64 y = g->cell_bytes ? g->cell_bytes : g->slice_size;
  u64 off = (u64) col * y;
  if (off >= g->slice_size) return 0;
  return MIN(y, g->slice_size - off);
}

u64 xpar_cell_bytes(const xpar_geom * g, u64 slice, u32 col) {
  u64 begin = slice * g->slice_size +
              (u64) col * (g->cell_bytes ? g->cell_bytes : g->slice_size);
  u64 size  = xpar_cell_size(g, col);
  if (!size || begin >= g->stream_length) return 0;
  return MIN(size, g->stream_length - begin);
}

/*  Erasures.  */

void xpar_erasures_init(xpar_erasures * e, u64 slices, u32 cells) {
  u64 n = slices * (cells ? cells : 1);
  if (n > 0x7FFFFFFFu) FATAL("Erasure table too large for this host.");
  e->slice_count     = slices;
  e->cells_per_slice = cells ? cells : 1;
  e->bad_count       = 0;
  e->bad             = (u8 *) xpar_calloc(n ? (sz) n : 1, 1);
}

void xpar_erasures_free(xpar_erasures * e) {
  xpar_free(e->bad);
  e->bad = NULL;  e->slice_count = 0;  e->bad_count = 0;
}

void xpar_erasures_clear(xpar_erasures * e) {
  xpar_memset(e->bad, 0, (sz) (e->slice_count * e->cells_per_slice));
  e->bad_count = 0;
}

void xpar_erasures_mark_slice(xpar_erasures * e, u64 slice) {
  u32 c;
  for (c = 0; c < e->cells_per_slice; c++) xpar_cell_mark(e, slice, c);
}

void xpar_erasures_mark_range(xpar_erasures * e, const xpar_geom * g,
                              u64 off, u64 len) {
  u64 end, p;
  if (!len) return;
  if (off < g->stream_base) off = g->stream_base;
  end = off + len;
  if (end > g->stream_base + g->slice_count * g->slice_size)
    end = g->stream_base + g->slice_count * g->slice_size;
  for (p = off; p < end;) {
    u64 slice = xpar_slice_of(g, p);
    u32 col   = xpar_cell_of(g, p);
    xpar_cell_mark(e, slice, col);
    p = xpar_cell_begin(g, slice, col) + xpar_cell_size(g, col);
  }
}

u64 xpar_erasures_max_depth(const xpar_erasures * e) {
  u64 worst = 0, i;
  u32 c;
  for (c = 0; c < e->cells_per_slice; c++) {
    u64 depth = 0;
    for (i = 0; i < e->slice_count; i++)
      if (e->bad[i * e->cells_per_slice + c]) depth++;
    if (depth > worst) worst = depth;
  }
  return worst;
}

static u64 pattern_hash(const xpar_erasures * e, u32 col) {
  u64 h = 0xCBF29CE484222325ull, i;
  for (i = 0; i < e->slice_count; i++) {
    h ^= e->bad[i * e->cells_per_slice + col];
    h *= 0x100000001B3ull;
  }
  return h;
}

static bool pattern_eq(const xpar_erasures * e, u32 a, const u8 * present) {
  u64 i;
  for (i = 0; i < e->slice_count; i++) {
    u8 bad = e->bad[i * e->cells_per_slice + a];
    if ((present[i] == 0) != (bad != 0)) return false;
  }
  return true;
}

void xpar_col_groups_build(const xpar_erasures * e, xpar_col_groups * g) {
  u32 k = e->cells_per_slice, col, mask, * bucket, b;
  u64 size = 16;
  u64 * hash;
  g->slice_count     = e->slice_count;
  g->cells_per_slice = k;
  g->group_count     = 0;
  g->group = (xpar_col_group *) xpar_alloc_raw((sz) k *
                                               sizeof(xpar_col_group));
  while (size < (u64) k * 2 + 2) size *= 2;
  mask   = (u32) (size - 1);
  bucket = (u32 *) xpar_alloc_raw(((sz) mask + 1) * sizeof(u32));
  hash   = (u64 *) xpar_alloc_raw((sz) k * sizeof(u64));
  for (b = 0; b <= mask; b++) bucket[b] = 0xFFFFFFFFu;

  for (col = 0; col < k; col++) {
    u64 h = pattern_hash(e, col);
    u32 slot = (u32) ((h ^ (h >> 32)) & mask), gi = 0xFFFFFFFFu;
    while (bucket[slot] != 0xFFFFFFFFu) {
      u32 cand = bucket[slot];
      if (hash[cand] == h && pattern_eq(e, col, g->group[cand].present)) {
        gi = cand;
        break;
      }
      slot = (slot + 1) & mask;
    }
    if (gi == 0xFFFFFFFFu) {
      u64 i;
      xpar_col_group * ng = &g->group[g->group_count];
      ng->present = (u8 *) xpar_alloc_raw(e->slice_count
                                            ? (sz) e->slice_count : 1);
      ng->column  = (u32 *) xpar_alloc_raw((sz) k * sizeof(u32));
      ng->column_count = 0;
      ng->erased  = 0;
      for (i = 0; i < e->slice_count; i++) {
        u8 bad = e->bad[i * k + col];
        ng->present[i] = !bad;
        if (bad) ng->erased++;
      }
      gi = g->group_count++;
      hash[gi] = h;
      bucket[slot] = gi;
    }
    g->group[gi].column[g->group[gi].column_count++] = col;
  }
  xpar_free(bucket);  xpar_free(hash);
}

void xpar_col_groups_free(xpar_col_groups * g) {
  For(u32, i, g->group_count,
      xpar_free(g->group[i].present);  xpar_free(g->group[i].column))
  xpar_free(g->group);
  g->group = NULL;  g->group_count = 0;
}

/*  Stream to entry.  */

bool xpar_stream_locate(const xpar_occindex * ix, u64 off,
                        xpar_span * out) {
  xpar_occurrence o;
  u64 run;
  if (!xpar_occindex_canonical(ix, off, &o, &run)) return false;
  out->stream_offset = off;
  out->file_offset   = o.file_offset + (off - o.stream_offset);
  out->length        = run;
  out->entry         = o.entry;
  return true;
}

u32 xpar_slice_spans(const xpar_geom * g, const xpar_occindex * ix,
                     u64 slice, xpar_span * out, u32 max) {
  u64 begin = xpar_slice_begin(g, slice);
  u64 have  = xpar_slice_bytes(g, slice);
  u64 p     = begin;
  u32 n     = 0;
  while (p < begin + have) {
    xpar_span s;
    if (!xpar_stream_locate(ix, p, &s)) break;
    if (s.length > begin + have - p) s.length = begin + have - p;
    if (n < max) out[n] = s;
    n++;
    p += s.length;
  }
  return n;
}
