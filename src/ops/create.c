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

/*  Protected-set writer, from manifest scan through volume publication.  */

#include "ops.h"
#include "chain.h"
#include "vset.h"
#include "volimg.h"

#include "common.h"
#include "auth.h"
#include "chunk.h"
#include "container.h"
#include "json.h"
#include "manifest.h"
#include "pathname.h"
#include "plan.h"
#include "slice.h"
#include "volname.h"
#include "kernel/armour.h"
#include "kernel/codec.h"
#include "kernel/crc32c.h"
#include "platform/port-fs.h"
#include "platform/port-thread.h"

typedef struct {
  const xpar_options * o;
  xpar_manifest   m;
  xpar_occindex   ix;
  xpar_plan       plan;
  xpar_geom       geom;
  xpar_setd       sd;
  xpar_wropt      wr;
  xpar_key        key;
  u8              master[XPAR_BLAKE3_KEY_LEN];
  xpar_auth       auth;
  bool            keyed;
  u8              set_id[XPAR_SET_ID_LEN];
  u64             recovery;
  u32           * slice_crc;      /*  S entries.  */
  u8            * slice_tag;      /*  S * tag_len bytes.  */
  u32           * cell_crc;       /*  S * K entries, or NULL.  */
  u8              tag_len;
  char          * base;           /*  Output base name, path included.  */
  xpar_json       js;
  xpar_progress_t prog;
  xpar_armour   * arm;            /*  Critical-group armour, or NULL.  */
  xpar_armour_params region_ap;   /*  The armoured layout's region code.  */
  xpar_chunk_index chunk_cache;
  char * chunk_cache_path, * chunk_cache_stage;
  u8 * stream_cache;
  u64 stream_cache_length;
} ctx;

static const xpar_key * create_key(const ctx * c) {
  return c->keyed ? &c->key : NULL;
}

static char * base_name(const xpar_options * o) {
  char * b;
  sz n;
  if (o->output && o->output[0]) return xpar_strdup(o->output);
  n = xpar_strlen(o->paths[0]);
  while (n > 1 && xpar_path_sep(o->paths[0][n - 1]))
    n--;
  b = xpar_strndup(o->paths[0], n);
  return b;
}

/*  Power-of-two volume ladder, with any remainder in the last volume.  */

typedef struct { u64 first, count; } vol_span;

static u32 ladder(const xpar_options * o, u64 r, vol_span * out, u32 max) {
  u64 at = 0, step = 1;
  u32 n = 0;
  if (!r) return 0;
  if (o->volumes == XPAR_VOLS_EQUAL || o->volumes == XPAR_VOLS_FIXED) {
    u64 want = o->volume_count, per;
    if (o->volumes == XPAR_VOLS_EQUAL || !want) {
      u64 t = 0, st = 1;
      want = 0;
      while (t < r) { u64 k = MIN(st, r - t);
                      if (t + 2 * st > r) k = r - t;
                      t += k;  want++;  st *= 2; }
    }
    if (want > r) want = r;
    if (!want) want = 1;
    per = xpar_ceil_div(r, want);
    while (at < r && n < max) {
      out[n].first = at;
      out[n].count = MIN(per, r - at);
      at += out[n].count;  n++;
    }
    return n;
  }
  while (at < r && n < max) {
    u64 take = MIN(step, r - at);
    /*  Avoid a final volume smaller than its predecessor.  */
    if (at + 2 * step > r) take = r - at;
    out[n].first = at;  out[n].count = take;
    at += take;  n++;  step *= 2;
  }
  return n;
}

/*  Solve codeword length and parity jointly; short objects must not inherit
    the overhead of a full-width codeword.  */
static void armour_params(const xpar_options * o, u64 object_bytes,
                          bool metadata, xpar_armour_params * p) {
  u32 w, sym, t2, n;
  f64 pct = o->armour_pct;

  /*  Small metadata favours correctable fraction; regions favour low
      overhead.  */
  sym = 8;
  if (o->armour_field == 8 || o->armour_field == 16)
    sym = (u32) o->armour_field;
  else if (!metadata)
    sym = pct > 1.0 ? 8 : 16;
  else if (object_bytes >= ((u64) 4 << 10) && pct > 0.0 && pct <= 1.0)
    sym = 16;
  xpar_armour_defaults(p, sym);
  w = sym / 8;

  /* Keep the field's full-width code when parity is disabled. */
  if (!o->armour_t && pct <= 0.0) {
    p->depth = 1;
    return;
  }

  /*  Iterate n' = min(n, ceil(bytes/W) + 2t) with 2t = round(p*n').
      The fixed bound prevents malformed percentages from looping.  */
  n = p->n;
  t2 = o->armour_t ? 2 * o->armour_t : 2;
  {
    int it;
    for (it = 0; it < 8; it++) {
      u64 need = xpar_ceil_div(object_bytes, w) + t2;
      u32 n2 = need < (u64) n ? (u32) need : n;
      u32 t3 = o->armour_t ? 2 * o->armour_t
                           : (u32) (pct * (f64) n2 / 100.0 + 0.5);
      if (t3 < 2) t3 = 2;
      t3 &= ~1u;
      if (t3 >= n2) t3 = (n2 - 1) & ~1u;
      if (n2 == n && t3 == t2) break;
      n = n2;  t2 = t3;
    }
  }
  if (t2 < 2) t2 = 2;
  if (n <= t2) n = t2 + 1;
  p->n = n;  p->k = n - t2;
  p->depth = 1;
}

/*  A burst spans at most (t*D - 1)*W bytes. Clamp D until two frame
    buffers fit within one quarter of the memory budget.  */
static void armour_depth(const xpar_options * o, u64 budget,
                         xpar_armour_params * p) {
  u64 w = p->symbol_bits / 8, t = (p->n - p->k) / 2, d = 1;
  if (o->depth) d = o->depth;
  else if (o->burst) {
    /*  Saturate the increment to prevent wraparound.  */
    u64 sym = o->burst / w;
    d = sym == (u64) -1 ? (u64) -1 : xpar_ceil_div(sym + 1, t ? t : 1);
  }
  if (!d) d = 1;
  /*  Clamp before computing the footprint to avoid overflow.  */
  if (d > XPAR_ARMG_DEPTH_MAX) d = XPAR_ARMG_DEPTH_MAX;
  while (d > 1 && 2 * d * p->n * w > budget / 4) d /= 2;
  p->depth = d;
}

/*  Resolve logical stream bytes through canonical occurrences. Bytes past
    the stored length are the zero padding protected by the outer code.  */

typedef struct {
  ctx * c;
  xpar_file * f;
  u32 entry;
  bool open;
} reader;

static void rd_init(reader * r, ctx * c) {
  r->c = c;  r->f = NULL;  r->entry = 0;  r->open = false;
}

static void rd_free(reader * r) {
  if (r->open) xpar_close(r->f);
  r->open = false;  r->f = NULL;
}

static void rd_bytes(reader * r, u64 off, u8 * buf, u64 len) {
  ctx * c = r->c;
  if (c->stream_cache) {
    u64 rel, take;
    if (off < c->m.stream_base) {
      xpar_memset(buf, 0, (sz) len);
      return;
    }
    rel = off - c->m.stream_base;
    take = rel < c->stream_cache_length
             ? MIN(len, c->stream_cache_length - rel) : 0;
    if (take) xpar_memcpy(buf, c->stream_cache + rel, (sz) take);
    if (take < len) xpar_memset(buf + take, 0, (sz) (len - take));
    return;
  }
  while (len) {
    xpar_span sp;
    u64 take;
    if (!xpar_stream_locate(&c->ix, off, &sp)) {
      /*  Zero only to the next extent; gaps may be interior.  */
      u64 gap = xpar_occindex_next(&c->ix, off, off + len) - off;
      if (!gap) gap = len;
      xpar_memset(buf, 0, (sz) gap);
      buf += gap;  off += gap;  len -= gap;
      continue;
    }
    take = MIN(sp.length, len);
    if (!r->open || r->entry != sp.entry) {
      if (r->open) xpar_close(r->f);
      r->f = xpar_open(c->m.source[sp.entry], XPAR_O_RDONLY);
      if (!r->f) FATAL_PERROR(c->m.source[sp.entry]);
      r->entry = sp.entry;  r->open = true;
    }
    if (xpar_seek(r->f, (i64) sp.file_offset, XPAR_SEEK_SET) != 0)
      FATAL_IO("Cannot seek in '%s'.", c->m.source[sp.entry]);
    if (xpar_xread(r->f, buf, (sz) take) != (sz) take)
      FATAL_IO("'%s' is shorter than it was when it was scanned.",
               c->m.source[sp.entry]);
    buf += take;  off += take;  len -= take;
  }
}

static void tag_alloc(ctx * c) {
  u64 s = c->geom.slice_count;
  u32 k = c->geom.cells_per_slice;
  c->slice_crc = (u32 *) xpar_alloc_raw((sz) s * sizeof(u32));
  if (c->tag_len)
    c->slice_tag = (u8 *) xpar_alloc_raw((sz) s * c->tag_len);
  if (c->geom.cell_bytes)
    c->cell_crc = (u32 *) xpar_alloc_raw((sz) s * k * sizeof(u32));
}

static void tag_slice(ctx * c, u64 slice, const u8 * buf) {
  u64 z = c->geom.slice_size;
  u32 k = c->geom.cells_per_slice;
  c->slice_crc[slice] = xpar_crc32c(0, buf, (sz) z);
  if (c->tag_len) {
    u8 t[32];
    if (c->keyed)
      xpar_slice_tag_keyed(&c->sd, slice, buf, c->key.k_slice, t, sizeof t);
    else
      xpar_slice_tag(&c->sd, slice, buf, t, sizeof t);
    xpar_memcpy(c->slice_tag + slice * c->tag_len, t, c->tag_len);
  }
  if (c->cell_crc) {
    u32 col;
    for (col = 0; col < k; col++) {
      u64 at = (u64) col * c->geom.cell_bytes;
      u64 n = xpar_cell_size(&c->geom, col);
      c->cell_crc[slice * k + col] = xpar_crc32c(0, buf + at, (sz) n);
    }
  }
}

/*  Slice tags include the protected zero padding.  */
static void tag_pass(ctx * c) {
  reader r;
  u64 z = c->geom.slice_size, i, s = c->geom.slice_count;
  u8 * buf;
  if (!s) return;
  buf = (u8 *) xpar_alloc_raw((sz) z);
  tag_alloc(c);
  rd_init(&r, c);
  for (i = 0; i < s; i++) {
    rd_bytes(&r, xpar_slice_begin(&c->geom, i), buf, z);
    tag_slice(c, i, buf);
    xpar_progress_tick(&c->prog, xpar_slice_bytes(&c->geom, i));
  }
  rd_free(&r);
  xpar_free(buf);
}

typedef struct {
  u8 * mem;             /*  R * Z, when it fits the budget.  */
  xpar_file * spill;    /*  Otherwise a scratch file of the same shape.  */
  char * path;
  u64 z, count;
} recstore;

typedef struct {
  recstore rs;
  u64      slices;
  u8       field_log2;
  bool     encoded;
  char *   source_path;
  char *   final_path;
} pipe_ready;

static void rs_open(recstore * rs, u64 count, u64 z, u64 budget,
                    const char * base) {
  /*  Avoid spilling sub-megabyte stores.  */
  if (budget < ((u64) 1 << 20)) budget = (u64) 1 << 20;
  rs->z = z;  rs->count = count;  rs->mem = NULL;
  rs->spill = NULL;  rs->path = NULL;
  if (!count) return;
  if (count * z <= budget && count * z <= (u64) (sz) -1) {
    rs->mem = (u8 *) xpar_alloc_raw((sz) (count * z));
    return;
  }
  xpar_asprintf(&rs->path, "%s.xpar-tmp", base);
  rs->spill = xpar_open(rs->path, XPAR_O_RDWR | XPAR_O_CREAT | XPAR_O_TRUNC);
  if (!rs->spill) FATAL_PERROR(rs->path);
}

static void rs_put(recstore * rs, u64 idx, u64 off, const u8 * p, u64 n) {
  if (rs->mem) { xpar_memcpy(rs->mem + idx * rs->z + off, p, (sz) n);  return; }
  if (xpar_pwrite(rs->spill, p, (sz) n, idx * rs->z + off) != (sz) n)
    FATAL_PERROR(rs->path);
}

static const u8 * rs_data(recstore * rs, u64 idx, u8 * scratch) {
  if (rs->mem) return rs->mem + idx * rs->z;
  if (xpar_pread(rs->spill, scratch, (sz) rs->z,
                 idx * rs->z) != (sz) rs->z)
    FATAL_PERROR(rs->path);
  return scratch;
}

static void rs_close(recstore * rs) {
  xpar_free(rs->mem);
  if (rs->spill) { xpar_close(rs->spill);  xpar_remove(rs->path); }
  xpar_free(rs->path);
  rs->mem = NULL;  rs->spill = NULL;  rs->path = NULL;
}

/*  FFT uses the planner's complete footprint. Matrix streaming holds one
    input column and R accumulator columns.  */
static u64 encode_chunk(const ctx * c, u64 budget) {
  u64 s = c->geom.slice_count, r = c->recovery, z = c->geom.slice_size;
  u64 want, per;
  if (XPAR_CODEC_IS_FFT(c->plan.codec)) {
    want = c->plan.column_chunk;
    if (want > z) want = z;
    if (!want) want = 64;
    while (want > 64 &&
           xpar_codec_encode_footprint_axis(
             c->plan.codec, c->plan.field_log2, s, r,
             c->sd.recovery_axis_log2, (sz) want) > budget)
      want = (want / 2) & ~(u64) 63;
    FATAL_UNLESS("The reserved FFT axis does not fit the memory limit.",
                 xpar_codec_encode_footprint_axis(
                   c->plan.codec, c->plan.field_log2, s, r,
                   c->sd.recovery_axis_log2, (sz) want) <= budget);
    return want;
  }
  (void) s;
  per  = r + 1;
  want = (budget / per) & ~(u64) 63;
  if (want > z) want = z;
  if (!want)
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "The matrix encoder needs %" PRIu64 " resident column buffers and "
               "-m admits none at the 64-byte minimum; raise -m to at "
               "least %" PRIu64 " bytes.", per,
               (per * 64));
  return want;
}

typedef struct {
  xpar_codec * cd;
  const u8 * const * data;
  u8 ** rec;
  u64 data_first, data_count, recovery_count, jobs;
  sz bytes;
  bool clear;
} matrix_batch;

/*  Fixed row partitions make worker writes disjoint and deterministic.  */
static void matrix_accumulate_job(sz index, void * arg) {
  matrix_batch * b = (matrix_batch *) arg;
  u64 per = xpar_ceil_div(b->recovery_count, b->jobs);
  u64 first = (u64) index * per;
  u64 count = first < b->recovery_count
                ? MIN(per, b->recovery_count - first) : 0;
  xpar_codec_status st;
  if (!count) return;
  st = xpar_codec_matrix_accumulate_many(
         b->cd, b->data_first, b->data, b->data_count, first,
         b->rec + first, count, b->bytes, b->clear);
  FATAL_UNLESS("internal: matrix streaming encoder rejected a planned "
               "recovery-row range.", st == XPAR_CODEC_OK);
}

static void encode_matrix(ctx * c, recstore * rs, u64 budget) {
  xpar_codec * cd;
  xpar_pool * workers;
  reader r;
  matrix_batch batch;
  u64 s = c->geom.slice_count, rr = c->recovery, z = c->geom.slice_size;
  u64 chunk = encode_chunk(c, budget), at, i, first, count, input_count;
  u64 used, spare;
  u8 * pool = NULL, ** data, ** rec;
  bool direct = rs->mem != NULL;
  bool cached;
  if (!s || !rr) return;
  cached = c->stream_cache && chunk == z;
  if (chunk == z) tag_alloc(c);
  else            tag_pass(c);
  if (rr > ((u64) -1 - chunk) / (direct ? z : chunk))
    FATAL_CODE(XPAR_EXIT_NOPLAN, "Matrix column buffers exceed this host's "
               "address space; lower -m or -b.");
  used = direct ? rr * z + chunk : (rr + 1) * chunk;
  spare = budget > used ? budget - used : 0;
  input_count = cached ? s : 1 + spare / chunk;
  if (input_count > s) input_count = s;
  if (input_count + (direct ? 0 : rr) > (u64) (sz) -1 / chunk)
    FATAL_CODE(XPAR_EXIT_NOPLAN, "Matrix column buffers exceed this host's "
               "address space; lower -m or -b.");
  if (!cached || !direct)
    pool = (u8 *) xpar_alloc_aligned(
             (sz) ((cached ? 0 : input_count) * chunk +
                    (direct ? 0 : rr) * chunk), 64);
  data = (u8 **) xpar_alloc_raw(
           (sz) (input_count + rr) * sizeof(u8 *));
  rec = data + input_count;
  for (i = 0; i < input_count; i++)
    data[i] = cached ? c->stream_cache + i * z : pool + i * chunk;
  cd = xpar_codec_new_axis(c->plan.codec, c->plan.field_log2, s, rr,
                           c->sd.recovery_axis_log2);
  workers = xpar_pool_create(c->o->jobs > 0 ? c->o->jobs
                                             : xpar_core_count());
  xpar_memset(&batch, 0, sizeof batch);
  batch.cd = cd;  batch.data = (const u8 * const *) data;  batch.rec = rec;
  batch.recovery_count = rr;
  batch.jobs = MIN(rr, (u64) xpar_pool_threads(workers));
  if (!batch.jobs) batch.jobs = 1;
  rd_init(&r, c);
  for (at = 0; at < z; at += chunk) {
    u64 n = MIN(chunk, z - at);
    for (i = 0; i < rr; i++)
      rec[i] = direct ? rs->mem + i * z + at
                      : pool + (cached ? 0 : input_count * chunk) +
                               i * chunk;
    batch.bytes = (sz) n;
    for (first = 0; first < s; first += input_count) {
      count = MIN(input_count, s - first);
      for (i = 0; i < count; i++) {
        if (!cached)
          rd_bytes(&r, xpar_slice_begin(&c->geom, first + i) + at,
                   data[i], n);
        if (chunk == z) {
          tag_slice(c, first + i, data[i]);
          xpar_progress_tick(&c->prog,
                             xpar_slice_bytes(&c->geom, first + i));
        }
      }
      batch.data_first = first;
      batch.data_count = count;
      batch.clear = first == 0;
      xpar_pool_run(workers, (sz) batch.jobs,
                    matrix_accumulate_job, &batch);
      xpar_progress_tick(&c->prog, count * n);
    }
    if (!direct)
      for (i = 0; i < rr; i++) rs_put(rs, i, at, rec[i], n);
  }
  rd_free(&r);
  xpar_pool_destroy(workers);
  xpar_codec_free(cd);
  xpar_free(data);
  if (pool) xpar_free_aligned(pool);
}

static void encode(ctx * c, recstore * rs, u64 budget) {
  xpar_codec * cd;
  reader r;
  u64 s = c->geom.slice_count, rr = c->recovery, z = c->geom.slice_size;
  u64 chunk = encode_chunk(c, budget), at, i;
  u8 * pool, ** data, ** rec;
  if (!s) return;
  if (!rr) { tag_pass(c);  return; }
  if (c->plan.codec == XPAR_CODEC_MATRIX) {
    encode_matrix(c, rs, budget);
    return;
  }
  tag_pass(c);
  if ((s + rr) * chunk > (u64) (sz) -1)
    FATAL_CODE(XPAR_EXIT_NOPLAN, "Column buffers exceed this host's "
               "address space; lower -m or -b.");
  pool = (u8 *) xpar_alloc_aligned((sz) ((s + rr) * chunk), 64);
  data = (u8 **) xpar_alloc_raw((sz) (s + rr) * sizeof(u8 *));
  rec  = data + s;
  for (i = 0; i < s + rr; i++) data[i] = pool + i * chunk;
  cd = xpar_codec_new_axis(c->plan.codec, c->plan.field_log2, s, rr,
                           c->sd.recovery_axis_log2);
  rd_init(&r, c);
  for (at = 0; at < z; at += chunk) {
    u64 n = MIN(chunk, z - at);
    for (i = 0; i < s; i++)
      rd_bytes(&r, xpar_slice_begin(&c->geom, i) + at, data[i], n);
    if (xpar_codec_encode(cd, (const u8 * const *) data, rec, (sz) n) !=
        XPAR_CODEC_OK)
      FATAL_CODE(XPAR_EXIT_INTERNAL,
                 "internal: the codec refused a geometry the planner "
                 "accepted.");
    for (i = 0; i < rr; i++) rs_put(rs, i, at, rec[i], n);
    xpar_progress_tick(&c->prog, s * n);
  }
  rd_free(&r);
  xpar_codec_free(cd);
  xpar_free(data);
  xpar_free_aligned(pool);
}

/*  Derive set_id from container.c's canonical SETD and FILE bodies, then
    serialise the packets again with that identity.  */
static void compute_set_id(ctx * c) {
  xpar_buf b;
  xpar_set_id_ctx h;
  u8 zero[XPAR_SET_ID_LEN];
  sz p = 0;
  bool first = true;
  xpar_memset(zero, 0, sizeof zero);
  xpar_buf_init(&b);
  xpar_setd_write(&b, &c->sd, zero, NULL);
  For(u32, i, c->m.count,
      xpar_entry_write(&b, &c->m.entry[i], zero, NULL, &c->wr))
  while (p + XPAR_PKT_HDR <= b.len) {
    u64 len = xpar_rd64(b.data + p + 8);
    const u8 * body = b.data + p + XPAR_PKT_HDR;
    sz n = (sz) (len - XPAR_PKT_HDR);
    if (len < XPAR_PKT_HDR || p + len > b.len) break;
    if (first) { xpar_set_id_begin(&h, c->keyed ? c->key.k_set : NULL,
                                   body, n);  first = false; }
    else       xpar_set_id_update(&h, body, n);
    p += (sz) len;
  }
  xpar_set_id_final(&h, c->set_id);
  xpar_buf_free(&b);
}

/*  --auth-only replaces the public whole-file hash with K_file MAC. Prefix
    hashes remain public for moved-file recognition.  */
static void auth_only_hashes(ctx * c) {
  xpar_nameidx ix;
  u32 i;
  xpar_nameidx_build(&c->m, &ix);
  for (i = 0; i < c->m.count; i++) {
    xpar_entry * e = &c->m.entry[i];
    xpar_blake3_t h;
    if (e->entry_type == XPAR_ENTRY_HARDLINK) continue;
    xpar_blake3_init_keyed(&h, c->key.k_file);
    if (e->entry_type == XPAR_ENTRY_SYMLINK) {
      xpar_blake3_update(&h, e->extra, e->extra_len);
    } else if (e->entry_type == XPAR_ENTRY_REGULAR) {
      xpar_file * f = xpar_open(c->m.source[i], XPAR_O_RDONLY);
      u8 buf[16384];
      if (!f) FATAL_PERROR(c->m.source[i]);
      for (;;) {
        sz n = xpar_read(f, buf, sizeof buf);
        if (n) xpar_blake3_update(&h, buf, n);
        if (n < sizeof buf) {
          if (xpar_error(f)) FATAL_IO("Reading '%s' failed.", c->m.source[i]);
          if (xpar_eof(f) || !n) break;
        }
      }
      xpar_xclose(f);
      xpar_secure_zero(buf, sizeof buf);
    }
    xpar_blake3_final(&h, e->content_hash, 32);
    xpar_secure_zero(&h, sizeof h);
  }
  for (i = 0; i < c->m.count; i++) {
    xpar_entry * e = &c->m.entry[i];
    i64 t;
    if (e->entry_type != XPAR_ENTRY_HARDLINK) continue;
    t = xpar_link_target(&c->m, &ix, i);
    FATAL_UNLESS("Hard-link entry '%.*s' has no canonical target.",
                 t >= 0, (int) e->name_len, e->name);
    xpar_memcpy(e->content_hash, c->m.entry[t].content_hash, 32);
  }
  xpar_nameidx_free(&ix);
}

/*  Write recovery payloads without assembling R * Z bytes twice.  */

static void put_file(const char * path, const xpar_buf * b) {
  xpar_file * f = xpar_open(path, XPAR_O_WRONLY | XPAR_O_CREAT |
                                  XPAR_O_TRUNC);
  if (!f) FATAL_PERROR(path);
  xpar_xwrite(f, b->data, b->len);
  xpar_xclose(f);
}

static void emit_crit(ctx * c, xpar_buf * out, const xpar_crit * cr) {
  xpar_buf plain;
  if (c->o->armour == XPAR_ARMOUR_NONE || !c->arm) {
    xpar_crit_write(out, cr, c->set_id, create_key(c), &c->wr);
    return;
  }
  xpar_buf_init(&plain);
  xpar_crit_write(&plain, cr, c->set_id, create_key(c), &c->wr);
  {
    xpar_armg g;
    const xpar_armour_params * ap = xpar_armour_params_of(c->arm);
    u64 n = xpar_armour_size(c->arm, plain.len);
    u8 * enc = (u8 *) xpar_alloc_raw((sz) n);
    xpar_armour_encode(c->arm, enc, plain.data, plain.len);
    g.symbol_bits     = (u8) ap->symbol_bits;
    g.poly            = ap->poly;
    g.n               = ap->n;
    g.k               = ap->k;
    g.fcr             = ap->fcr;
    g.prim            = ap->prim;
    g.depth           = ap->depth;
    g.plain_length    = plain.len;
    g.armoured_length = n;
    g.data            = NULL;
    xpar_armg_write(out, &g, enc, c->set_id, create_key(c));
    xpar_free(enc);
  }
  xpar_buf_free(&plain);
}

static void emit_head(ctx * c, xpar_buf * out, const xpar_crit * cr,
                      u32 index, u32 kind, bool tables, bool crit) {
  xpar_volh v;
  v.volume_index  = index;
  v.volume_kind   = kind;
  v.version_major = XPAR_FORMAT_MAJOR;
  v.version_minor = XPAR_FORMAT_MINOR;
  xpar_volh_write(out, &v, c->set_id, create_key(c));
  if (crit) emit_crit(c, out, cr);
  if (tables && c->geom.slice_count) {
    if (c->tag_len)
      xpar_sltg_write_all(out, c->slice_tag, c->geom.slice_count,
                          c->tag_len, c->set_id, create_key(c));
    if (c->cell_crc)
      xpar_slcl_write_all(out, c->cell_crc, c->geom.slice_count,
                          c->geom.cell_bytes, c->geom.cells_per_slice,
                          c->set_id, create_key(c));
  }
}

/*  Stream armour frames so the L-byte STRM body is never resident. The
    final frame is zero-padded, making encoded length deterministic.  */

/*  Reject unreachable sidecar entries; piped input is published locally.  */
static void check_reachable(ctx * c) {
  char * dir;
  const xpar_entry * lost;
  if (c->o->layout != XPAR_LAYOUT_SIDECAR) return;
  dir = xpar_path_dir(c->base);
  lost = xpar_manifest_unreachable(&c->m, dir, c->o->stdin_name);
  xpar_free(dir);
  if (lost)
    FATAL("Sidecar entry '%.*s' is unreachable; place the set beside its "
          "data or use --base.",
          (int) lost->name_len, lost->name);
}

/*  Rebuild and hash every entry through its extents, then reparse all
    written packet volumes. The first check catches invalid deduplication
    aliases that an archive could otherwise verify against itself.  */
static void verify_after(ctx * c, char * const * names, u32 count) {
  (void) count;
  FATAL_UNLESS("internal: no index/archive path was retained for read-back.",
               names && names[0]);
  xpar_verify_written_set_sources(c->o, names[0], &c->m);
}

static u64 create_stream_tag(ctx * c, u64 at, u64 length) {
  reader r;
  xpar_blake3_t h;
  u8 * buf = (u8 *) xpar_alloc_raw(1u << 16);
  u64 left = length;
  xpar_vol_tag_begin(&h);
  rd_init(&r, c);
  while (left) {
    u64 take = MIN(left, (u64) 1 << 16);
    rd_bytes(&r, at, buf, take);
    xpar_blake3_update(&h, buf, (sz) take);
    at += take;  left -= take;
  }
  rd_free(&r);
  xpar_free(buf);
  return xpar_vol_tag_final(&h);
}

static void emit_json_set(ctx * c) {
  if (!c->o->json) return;
  xpar_json_begin(&c->js, "set");
  xpar_json_u64(&c->js, "schema", XPAR_JSON_SCHEMA);
  xpar_json_hex(&c->js, "set_id", c->set_id, XPAR_SET_ID_LEN);
  xpar_json_u64(&c->js, "slice_size", c->geom.slice_size);
  xpar_json_u64(&c->js, "slices", c->geom.slice_count);
  xpar_json_u64(&c->js, "recovery", c->recovery);
  xpar_json_u64(&c->js, "field", c->plan.field_log2);
  xpar_json_str(&c->js, "codec",
                c->plan.codec == XPAR_CODEC_FFT_LOW ? "fft-low"
              : c->plan.codec == XPAR_CODEC_FFT ? "fft" : "matrix");
  xpar_json_str(&c->js, "layout",
                c->o->layout == XPAR_LAYOUT_SIDECAR ? "sidecar"
                  : (c->o->layout == XPAR_LAYOUT_SPLIT ? "split"
                                                       : "armoured"));
  xpar_json_u64(&c->js, "files", c->m.count);
  xpar_json_end(&c->js);
}

static void emit_json_files(ctx * c) {
  u32 i;
  if (!c->o->json) return;
  for (i = 0; i < c->m.count; i++) {
    const xpar_entry * e = &c->m.entry[i];
    xpar_json_begin(&c->js, "file");
    xpar_json_u64 (&c->js, "index", i);
    xpar_json_strn(&c->js, "name", e->name, e->name_len);
    xpar_json_u64 (&c->js, "length", e->length);
    xpar_json_u64 (&c->js, "extents", e->extent_count);
    xpar_json_hex (&c->js, "content_hash", e->content_hash, 32);
    xpar_json_end (&c->js);
  }
}

static void build_walk(const xpar_options * o, xpar_walk_opts * w, u64 z) {
  u64 budget = o->memory;
  xpar_walk_opts_default(w);
  if (!budget) {
    u64 phys = xpar_physical_memory();
    u64 cap = sizeof(void *) >= 8 ? ((u64) 1 << 30)
                                 : ((u64) 512 << 20);
    budget = phys ? phys / 4 : cap;
    if (budget > cap) budget = cap;
    if (budget < ((u64) 1 << 20)) budget = (u64) 1 << 20;
  }
  w->dedup     = (u8) o->dedup;
  w->align     = (u8) o->align;
  w->slice_size      = z;
  w->stream_base     = 0;
  w->dedup_max_refs  = o->dedup_max_refs;
  w->dedup_chunk     = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
  w->dedup_memory    = o->dedup_memory ? o->dedup_memory : budget / 4;
  w->preserve        = o->preserve;
  w->preserve_explicit = o->preserve_explicit;
  w->caps_mask     = (o->preserve & XPAR_PRES_LINKS) ? 0xFFFFFFFFu
                                                     : ~(u32) XPAR_FS_LINKID;
  w->path_flags    = 0;
  w->base_dir      = o->base_dir;
  w->exclude       = o->exclude;
  w->exclude_count = o->exclude_count;
  w->include       = o->include;
  w->include_count = o->include_count;
  w->recurse       = o->recurse;
  w->follow_symlinks = o->follow_symlinks;
  w->reproducible  = o->reproducible;
}

static void stage_chunk_cache(ctx * c, const xpar_walk_opts * w) {
  u32 i;
  if (!c->chunk_cache.slot) return;
  xpar_asprintf(&c->chunk_cache_path, "%s.xparidx", c->base);
  for (i = 0; i < 1000; i++) {
    xpar_stat_t st;
    xpar_asprintf(&c->chunk_cache_stage, "%s.xpar-cache-%03" PRIu32,
                  c->chunk_cache_path, i);
    if (xpar_lstat(c->chunk_cache_stage, &st) != 0) break;
    xpar_free(c->chunk_cache_stage);  c->chunk_cache_stage = NULL;
  }
  if (!c->chunk_cache_stage ||
      !xpar_chunk_cache_write(c->chunk_cache_stage, c->set_id,
                              w->dedup_chunk, &c->chunk_cache)) {
    if (c->o->verbose)
      xpar_fprintf(xpar_stderr, "xpar: could not stage chunk cache '%s'.\n",
                   c->chunk_cache_path);
    xpar_free(c->chunk_cache_stage);  c->chunk_cache_stage = NULL;
  }
  xpar_chunk_index_free(&c->chunk_cache);
}

static void publish_chunk_cache(ctx * c) {
  if (!c->chunk_cache_stage) return;
  if (xpar_rename(c->chunk_cache_stage, c->chunk_cache_path) != 0 ||
      xpar_fsync_dir(c->chunk_cache_path) != 0) {
    if (c->o->verbose)
      xpar_fprintf(xpar_stderr, "xpar: could not publish chunk cache '%s'.\n",
                   c->chunk_cache_path);
    xpar_remove(c->chunk_cache_stage);
  }
}

/*  A non-zero recovery percentage always yields at least one slice.  */
static u64 resolve_recovery(const xpar_rspec * rs, u64 s, u64 z, u64 floor) {
  f64 v;
  u64 r = 0;
  if (!s) return 0;
  switch (rs->kind) {
    case XPAR_R_NONE:    r = xpar_ceil_div(s, 20);  break;
    case XPAR_R_COUNT:   r = rs->count;  break;
    case XPAR_R_PERCENT:
      v = (f64) s * rs->factor / 100.0 + 0.5;
      r = v >= (f64) UINT64_MAX ? UINT64_MAX : (u64) v;
      /* Positive fractions round up; explicit zero stays zero. */
      if (!r && rs->factor > 0.0) r = 1;
      break;
    case XPAR_R_BYTES:   r = xpar_ceil_div(rs->count, z);  break;
    case XPAR_R_TIMES:
      v = (f64) s * rs->factor + 0.5;
      r = v >= (f64) UINT64_MAX ? UINT64_MAX : (u64) v;
      if (!r && rs->factor > 0.0) r = 1;
      break;
  }
  if (r < floor) r = floor;
  return r;
}

/*  Direct pipe input accumulates fixed Cauchy rows while publishing the
    final data object. The pipe is neither replayed nor copied to scratch.  */

/*  A volume is never assembled under its final name: a crash then leaves
    a stale temporary rather than a truncated set.  */
static xpar_file * create_stage_open(const char * dir, char ** path) {
  char * stem = xpar_path_join(dir, ".xpar-stdin-");
  xpar_file * f = xpar_stage_open(stem, XPAR_O_WRONLY, 0, path);
  xpar_free(stem);
  if (!f)
    FATAL_IO("Cannot create a secure pipe staging file in '%s': %s.", dir,
             xpar_strerror(xpar_errno()));
  return f;
}

static char * create_output_stage(const char * base) {
  char * parent = xpar_path_dir(base);
  char * stem = xpar_path_join(parent, ".xpar-create-");
  char * path = xpar_stage_dir(stem);
  xpar_free(parent);  xpar_free(stem);
  if (!path)
    FATAL_IO("Cannot create an output staging directory beside '%s'.", base);
  return path;
}

static void publish_outputs(const xpar_options * o, char * const * stage,
                            char * const * final, u32 count, u32 label_first,
                            u32 labels, const char * extra_from,
                            const char * extra_to, const char * stage_dir) {
  u32 extra = extra_from && extra_to;
  u32 total = count + labels + extra, at = 0, i, published = 0;
  char ** from = (char **) xpar_calloc(total, sizeof(char *));
  char ** to = (char **) xpar_calloc(total, sizeof(char *));
  char ** backup = (char **) xpar_calloc(total, sizeof(char *));
  bool * had = (bool *) xpar_calloc(total, sizeof(bool));
  bool collision = false, irregular = false;
  xpar_stat_t st;
  if (extra) {
    from[at] = xpar_strdup(extra_from);
    to[at++] = xpar_strdup(extra_to);
  }
  for (i = 1; i < count; i++, at++) {
    from[at] = xpar_strdup(stage[i]);
    to[at] = xpar_strdup(final[i]);
  }
  for (i = 0; i < labels; i++, at++) {
    from[at] = xpar_vname_label(stage[label_first + i]);
    to[at]   = xpar_vname_label(final[label_first + i]);
  }
  from[at] = xpar_strdup(stage[0]);
  to[at] = xpar_strdup(final[0]);
  for (i = 0; i < total; i++) {
    xpar_asprintf(&backup[i], "%s/.backup-%" PRIu32, stage_dir, i);
    if (xpar_lstat(to[i], &st) != 0) continue;
    if (!st.is_regular) { irregular = true;  goto rollback_old; }
    if (!o->force) { collision = true;  goto rollback_old; }
  }
  /* Recheck shared outputs and preserve rollback on refusal. */
  for (i = 0; i < total; i++) {
    if (xpar_lstat(to[i], &st) != 0) continue;
    if (!st.is_regular) { irregular = true;  goto rollback_old; }
    if (!o->force) { collision = true;  goto rollback_old; }
    if (xpar_rename(to[i], backup[i]) != 0) goto rollback_old;
    had[i] = true;
  }
  for (i = 0; i < total; i++) {
    if (xpar_rename(from[i], to[i]) != 0) goto rollback_new;
    published++;
  }
  if (xpar_fsync_dir(to[total - 1]) != 0) goto rollback_new;
  for (i = 0; i < total; i++)
    if (had[i] && xpar_remove(backup[i]) != 0)
      xpar_fprintf(xpar_stderr,
                   "xpar: warning: old output remains at '%s'.\n", backup[i]);
  if (xpar_rmdir(stage_dir) != 0)
    xpar_fprintf(xpar_stderr,
                 "xpar: warning: output staging directory '%s' remains.\n",
                 stage_dir);
  for (i = 0; i < total; i++) {
    xpar_free(from[i]);  xpar_free(to[i]);  xpar_free(backup[i]);
  }
  xpar_free(from);  xpar_free(to);  xpar_free(backup);  xpar_free(had);
  return;

rollback_new:
  while (published) { published--;  (void) xpar_remove(to[published]); }
rollback_old:
  for (u32 j = total; j > 0; j--)
    if (had[j - 1]) (void) xpar_rename(backup[j - 1], to[j - 1]);
  if (irregular)
    FATAL("Refusing non-regular output '%s'; set remains in '%s'.",
          to[i], stage_dir);
  if (collision)
    FATAL("Output '%s' appeared; set remains in '%s'.", to[i], stage_dir);
  FATAL_IO("Cannot publish '%s': %s; set remains in '%s'.",
           to[i < total ? i : total - 1], xpar_strerror(xpar_errno()),
           stage_dir);
}

static bool create_copy_file(const char * from, const char * to) {
  xpar_file * in = xpar_open(from, XPAR_O_RDONLY);
  xpar_file * out;
  u8 * buf;
  bool ok = true;
  if (!in) return false;
  out = xpar_open(to, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_EXCL);
  if (!out) { xpar_close(in);  return false; }
  buf = (u8 *) xpar_alloc_raw((sz) 1 << 20);
  for (;;) {
    sz n = xpar_read(in, buf, (sz) 1 << 20);
    if (n && xpar_write(out, buf, n) != n) { ok = false;  break; }
    if (n < ((sz) 1 << 20)) {
      if (xpar_error(in)) ok = false;
      break;
    }
  }
  if (ok && xpar_fsync(out) != 0) ok = false;
  xpar_free(buf);
  xpar_close(in);  xpar_close(out);
  if (!ok) xpar_remove(to);
  return ok;
}

static void create_stage_input(pipe_ready * ready, const char * to,
                               xpar_manifest * m) {
  const char * from = ready->source_path;
  u32 i;
  if (xpar_rename(from, to) != 0 && !create_copy_file(from, to))
    FATAL_IO("Cannot stage pipe input '%s': %s.", to,
             xpar_strerror(xpar_errno()));
  for (i = 0; i < m->count; i++)
    if (m->source[i] && !xpar_strcmp(m->source[i], from)) {
      xpar_free(m->source[i]);
      m->source[i] = xpar_strdup(to);
    }
}

char * xpar_spool_stdin(const xpar_options * o) {
  const char * anchor = o->output && o->output[0] ? o->output : o->set;
  char * outdir;
  FATAL_UNLESS("A pipe spool needs an output or set path.",
               anchor && anchor[0]);
  outdir = xpar_path_dir(anchor);
  char * dir = o->spool_dir ? xpar_strdup(o->spool_dir) : outdir;
  char * stage = NULL;
  xpar_file * f;
  u8 * buf;
  if (dir != outdir) xpar_free(outdir);
  if (xpar_mkdir_p(dir, 0777) != 0) FATAL_PERROR(dir);
  f = create_stage_open(dir, &stage);
  buf = (u8 *) xpar_alloc_raw((sz) 1 << 20);
  for (;;) {
    sz got = xpar_read(xpar_stdin, buf, (sz) 1 << 20);
    if (got) xpar_xwrite(f, buf, got);
    if (got < ((sz) 1 << 20)) {
      if (xpar_error(xpar_stdin)) FATAL_IO("Reading standard input failed.");
      if (xpar_eof(xpar_stdin) || !got) break;
    }
  }
  xpar_free(buf);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("Cannot flush pipe staging file '%s'.", stage);
  xpar_xclose(f);
  xpar_free(dir);
  return stage;
}

char * xpar_publish_spooled_stdin(const xpar_options * o,
                                  const char * stage) {
  const char * anchor = o->output && o->output[0] ? o->output : o->set;
  char * outdir, * final, * parent, * local = NULL, * backup = NULL;
  u32 i;
  bool had = false;
  xpar_stat_t st;
  FATAL_UNLESS("Publishing a pipe input needs an output or set path.",
               anchor && anchor[0]);
  outdir = xpar_path_dir(anchor);
  final = xpar_path_join(outdir, o->stdin_name);
  parent = xpar_path_dir(final);
  if (xpar_mkdir_p(parent, 0777) != 0) FATAL_PERROR(parent);
  if (xpar_lstat(final, &st) == 0) {
    if (!st.is_regular)
      FATAL("Refusing to replace non-regular pipe destination '%s'.", final);
    if (!o->force) FATAL("'%s' exists; -f overwrites it.", final);
    had = true;
  }
  for (i = 0; i < 1000; i++) {
    xpar_asprintf(&local, "%s.xpar-input-%03" PRIu32, final, i);
    if (xpar_lstat(local, &st) != 0 &&
        (xpar_rename(stage, local) == 0 || create_copy_file(stage, local)))
      break;
    xpar_free(local);  local = NULL;
  }
  if (!local) FATAL_IO("Cannot stage pipe input beside '%s'.", final);
  if (had) {
    for (i = 0; i < 1000; i++) {
      xpar_asprintf(&backup, "%s.xpar-old-%03" PRIu32, final, i);
      if (xpar_lstat(backup, &st) != 0) break;
      xpar_free(backup);  backup = NULL;
    }
    if (!backup || xpar_rename(final, backup) != 0) {
      xpar_remove(local);
      FATAL_IO("Cannot stage the old pipe destination '%s'.", final);
    }
  }
  if (xpar_rename(local, final) != 0 || xpar_fsync_dir(final) != 0) {
    int saved = xpar_errno();
    if (xpar_lstat(final, &st) == 0) (void) xpar_rename(final, local);
    if (backup) (void) xpar_rename(backup, final);
    FATAL_IO("Cannot publish pipe input as '%s': %s.", final,
             xpar_strerror(saved));
  }
  if (backup && xpar_remove(backup) != 0)
    xpar_fprintf(xpar_stderr, "xpar: old pipe input remains at '%s'.\n",
                 backup);
  if (xpar_lstat(stage, &st) == 0) (void) xpar_remove(stage);
  xpar_free(local);  xpar_free(backup);
  xpar_free(parent); xpar_free(outdir);
  return final;
}

static int create_regular(const xpar_options *, pipe_ready *);

static int create_from_pipe_spooled(const xpar_options * o) {
  xpar_options nested = *o;
  pipe_ready ready;
  char * outdir = xpar_path_dir(o->output);
  char * stage = xpar_spool_stdin(o), * final = NULL;
  char * one[1];
  int rc;
  xpar_stat_t st;
  xpar_memset(&ready, 0, sizeof ready);
  if (o->layout == XPAR_LAYOUT_SIDECAR) {
    final = xpar_path_join(outdir, o->stdin_name);
    { char * parent = xpar_path_dir(final);
      if (xpar_mkdir_p(parent, 0777) != 0) FATAL_PERROR(parent);
      xpar_free(parent); }
    if (!o->force && xpar_lstat(final, &st) == 0)
      FATAL("'%s' exists; -f overwrites it.", final);
  }
  one[0] = stage;
  nested.paths = one;  nested.path_count = 1;  nested.from_stdin = false;
  nested.base_dir = NULL;
  ready.source_path = stage;
  ready.final_path = final;
  rc = create_regular(&nested, &ready);
  if (xpar_lstat(stage, &st) == 0 && xpar_remove(stage) != 0 && o->verbose)
    xpar_fprintf(xpar_stderr, "xpar: warning: could not remove spool '%s'.\n",
                 stage);
  xpar_free(final);  xpar_free(stage);
  xpar_free(outdir);
  return rc;
}

static u64 create_pipe_budget(const xpar_options * o) {
  u64 phys, cap, budget = o->memory;
  if (budget) return budget;
  phys = xpar_physical_memory();
  cap = sizeof(void *) >= 8 ? ((u64) 1 << 30) : ((u64) 512 << 20);
  budget = phys ? phys / 4 : cap;
  if (budget > cap) budget = cap;
  if (budget < ((u64) 1 << 20)) budget = (u64) 1 << 20;
  return budget;
}

static u64 create_pipe_recovery(const xpar_options * o, u64 z) {
  u64 r = o->recovery.kind == XPAR_R_COUNT
            ? o->recovery.count : xpar_ceil_div(o->recovery.count, z);
  if (r < o->min_recovery) r = o->min_recovery;
  if (!r) r = 1;
  return r;
}

static void create_pipe_accumulate(xpar_codec * cd, xpar_pool * workers,
                                   u8 * data, u8 ** rec, u64 r, u64 slice,
                                   sz z) {
  matrix_batch batch;
  const u8 * source[1];
  xpar_memset(&batch, 0, sizeof batch);
  source[0] = data;
  batch.cd = cd;  batch.data = source;  batch.rec = rec;
  batch.data_first = slice;  batch.data_count = 1;
  batch.recovery_count = r;
  batch.jobs = MIN(r, (u64) xpar_pool_threads(workers));
  if (!batch.jobs) batch.jobs = 1;
  batch.bytes = z;  batch.clear = slice == 0;
  xpar_pool_run(workers, (sz) batch.jobs, matrix_accumulate_job, &batch);
}

static int create_from_pipe_direct(const xpar_options * o) {
  xpar_options nested = *o;
  pipe_ready ready;
  xpar_codec * cd;
  xpar_pool * workers;
  xpar_file * f;
  xpar_stat_t st;
  char * outdir = xpar_path_dir(o->output);
  char * final, * parent, * stage = NULL;
  char * one[1];
  u8 * data, ** rec;
  u64 z = o->slice_size ? o->slice_size : ((u64) 1 << 20);
  u64 r = create_pipe_recovery(o, z);
  u64 q, max_s, budget = create_pipe_budget(o), filled = 0, slices = 0;
  u8 field = o->field == 8 ? 8 : 16;
  int rc;

  xpar_memset(&ready, 0, sizeof ready);
  FATAL_UNLESS("Direct split pipe input supports one data volume; use "
               "--spool with --volumes=N.",
               o->layout != XPAR_LAYOUT_SPLIT ||
               o->volumes != XPAR_VOLS_FIXED || o->volume_count <= 1);
  FATAL_UNLESS("Direct pipe slices must be at least 4 KiB and 64-byte aligned.",
               z >= XPAR_SLICE_MIN && !(z & 63));
  q = (u64) 1 << field;
  FATAL_UNLESS("Recovery count exhausts GF(2^%" PRIu8 "); reduce -r or use "
               "--field=16.", r < q, field);
  FATAL_UNLESS("A one-pass matrix pipe needs %" PRIu64 " bytes; -m allows %"
               PRIu64 ".",
               r <= ((u64) -1) / z - 1 && (r + 1) * z <= budget,
               ((r + 1) * z),
               budget);
  FATAL_UNLESS("The pipe buffers exceed this host's address space.",
               r * z <= (u64) (sz) -1 && z <= (u64) (sz) -1);

  if (o->layout == XPAR_LAYOUT_SPLIT)
    xpar_asprintf(&final, "%s.d00", o->output);
  else
    final = xpar_path_join(outdir, o->stdin_name);
  parent = xpar_path_dir(final);
  if (xpar_mkdir_p(parent, 0777) != 0) FATAL_PERROR(parent);
  if (!o->force && xpar_lstat(final, &st) == 0)
    FATAL("'%s' exists; -f overwrites it.", final);
  f = create_stage_open(parent, &stage);

  ready.rs.z = z;  ready.rs.count = r;
  ready.rs.mem = (u8 *) xpar_calloc((sz) r, (sz) z);
  data = (u8 *) xpar_alloc_aligned((sz) z, 64);
  rec = (u8 **) xpar_alloc_raw((sz) r * sizeof(u8 *));
  for (u64 i = 0; i < r; i++) rec[i] = ready.rs.mem + i * z;
  max_s = q - r;
  cd = xpar_codec_new(XPAR_CODEC_MATRIX, field, max_s, r);
  workers = xpar_pool_create(o->jobs);

  for (;;) {
    sz got = xpar_read(xpar_stdin, data + filled, (sz) (z - filled));
    if (got) {
      xpar_xwrite(f, data + filled, got);
      filled += got;
    }
    if (filled == z) {
      FATAL_UNLESS("The pipe exceeds GF(2^%" PRIu8 ")'s %" PRIu64
                   "-slice data limit; "
                   "raise -s or use --spool.", slices < max_s, field,
                   max_s);
      create_pipe_accumulate(cd, workers, data, rec, r, slices, (sz) z);
      slices++;  filled = 0;
    }
    if (got == 0) {
      if (xpar_error(xpar_stdin)) FATAL_IO("Reading standard input failed.");
      break;
    }
  }
  if (filled) {
    FATAL_UNLESS("The pipe exceeds GF(2^%" PRIu8 ")'s %" PRIu64
                 "-slice data limit; "
                 "raise -s or use --spool.", slices < max_s, field,
                 max_s);
    xpar_memset(data + filled, 0, (sz) (z - filled));
    create_pipe_accumulate(cd, workers, data, rec, r, slices, (sz) z);
    slices++;
  }
  xpar_pool_destroy(workers);
  xpar_codec_free(cd);
  xpar_free(rec);
  xpar_free_aligned(data);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("Cannot flush direct pipe output '%s'.", stage);
  xpar_xclose(f);
  if (!slices) {
    xpar_free(ready.rs.mem);  ready.rs.mem = NULL;  ready.rs.count = 0;
  }
  ready.slices = slices;  ready.field_log2 = field;
  ready.encoded = true;
  ready.source_path = stage;
  ready.final_path = o->layout == XPAR_LAYOUT_SIDECAR ? final : NULL;
  one[0] = stage;
  nested.paths = one;  nested.path_count = 1;  nested.from_stdin = false;
  nested.base_dir = NULL;  nested.spool = false;
  nested.codec = XPAR_CODEC_MATRIX;  nested.field = field;
  nested.slice_size = z;  nested.slices = 0;
  rc = create_regular(&nested, &ready);

  if (xpar_lstat(stage, &st) == 0 && xpar_remove(stage) != 0 && o->verbose)
    xpar_fprintf(xpar_stderr, "xpar: warning: could not remove spool '%s'.\n",
                 stage);
  xpar_free(stage);  xpar_free(parent);  xpar_free(final);
  xpar_free(outdir);
  return rc;
}

static int create_from_pipe(const xpar_options * o) {
  return o->spool ? create_from_pipe_spooled(o)
                  : create_from_pipe_direct(o);
}

static int create_regular(const xpar_options * o, pipe_ready * ready) {
  ctx c;
  xpar_walk_opts w;
  xpar_geom_req gr;
  xpar_plan_req pr;
  xpar_plan_status ps;
  xpar_geom_status gs;
  recstore rs;
  recstore * rsp = ready && ready->encoded ? &ready->rs : &rs;
  vol_span * span = NULL;
  xpar_layt layt;
  xpar_crit cr;
  u64 budget, crit_bytes = 0, plan_z = 0;
  u32 nvol = 0, data_n = 0, name_count, i;
  int wf, wc;
  char ** names = NULL, ** write_names = NULL;
  char * output_stage = NULL;
  char * pipe_stage = NULL;

  if (o->from_stdin) return create_from_pipe(o);

  xpar_memset(&rs, 0, sizeof rs);
  xpar_memset(&c, 0, sizeof c);
  c.o = o;
  if (o->auth_key) {
    xpar_keyfile_status ks = xpar_keyfile_load(o->auth_key, &c.key, c.master);
    if (ks == XPAR_KEYFILE_OPEN) FATAL_PERROR(o->auth_key);
    if (ks == XPAR_KEYFILE_EMPTY)
      FATAL_CODE(XPAR_EXIT_AUTH, "The key file is empty.");
    if (ks != XPAR_KEYFILE_OK)
      FATAL_CODE(XPAR_EXIT_AUTH, "Reading key file '%s' failed.", o->auth_key);
    c.keyed = true;
    c.auth.kdf_id = 0;
    c.auth.slice_tag_keyed = 1;
    c.auth.packet_tag_keyed = 1;
    c.auth.unkeyed_retained = !o->auth_only;
    xpar_key_check(c.auth.key_check, c.master);
  }

  xpar_json_init(&c.js, o->json ? xpar_stdout : xpar_stderr, o->json);
  xpar_progress_init(&c.prog, xpar_progress_wanted(o), 0, "Creating");
  if (o->json) xpar_progress_sink(&c.prog, xpar_json_progress_sink, &c.js);

  c.wr.reproducible = o->reproducible;
  {
    u32 lit = o->preserve_explicit;
    c.wr.keep_mtime = (lit & XPAR_PRES_MTIME) != 0;
    c.wr.keep_atime = (lit & XPAR_PRES_ATIME) != 0;
    c.wr.keep_ctime = (lit & XPAR_PRES_CTIME) != 0;
    c.wr.keep_btime = (lit & XPAR_PRES_BTIME) != 0;
    c.wr.keep_posix = (lit & (XPAR_PRES_OWNER | XPAR_PRES_XATTR)) != 0;
  }
  c.base    = base_name(o);
  c.tag_len = c.keyed ? 16 : (u8) o->slice_tag;

  /*  Per-file alignment needs a provisional Z from the scanned lengths;
      recompute geometry from the packed stream afterwards.  */
  build_walk(o, &w, o->slice_size);
  if (o->dedup == XPAR_DEDUP_CHUNK) w.chunk_cache_out = &c.chunk_cache;
  else if (!o->auth_only) {
    u64 cache_budget = o->memory ? o->memory : xpar_plan_default_memory();
    w.stream_cache_out = &c.stream_cache;
    w.stream_cache_length_out = &c.stream_cache_length;
    w.stream_cache_limit = cache_budget - cache_budget / 4;
  }
  xpar_manifest_walk(&c.m, o->paths, o->path_count, &w);
  if (o->stdin_name && c.m.count == 1) {
    xpar_entry * e = &c.m.entry[0];
    xpar_free(e->name);
    e->name = xpar_strdup(o->stdin_name);
    e->name_len = (u32) xpar_strlen(o->stdin_name);
  }
  check_reachable(&c);
  if (o->align == XPAR_ALIGN_SLICE && !o->slice_size) {
    u64 sum = 0;
    xpar_memset(&gr, 0, sizeof gr);
    for (i = 0; i < c.m.count; i++) sum += c.m.entry[i].length;
    gr.stream_length = sum;  gr.field_log2 = 16;
    if (xpar_geom_choose(&gr, &c.geom) == XPAR_GEOM_OK)
      w.slice_size = c.geom.slice_size;
  }
  xpar_manifest_pack(&c.m, &w, &c.prog);
  if (o->auth_only) auth_only_hashes(&c);
  if (c.keyed)
    for (i = 0; i < c.m.count; i++)
      xpar_file_id(&c.m.entry[i], c.key.k_file, c.m.entry[i].file_id);
  xpar_occindex_build(&c.m, &c.ix);

  /*  Resolve R after S; the field bound depends on both.  */
  xpar_memset(&gr, 0, sizeof gr);
  gr.stream_length = c.m.stream_length;
  gr.slice_size    = o->slice_size;
  gr.slice_count   = o->slices;
  gr.field_log2    = o->field == 8 ? 8 : 16;
  gs = xpar_geom_choose(&gr, &c.geom);
  if (gs != XPAR_GEOM_OK) FATAL("%s.", xpar_geom_reason(gs));
  if (o->align == XPAR_ALIGN_1K) {
    FATAL_UNLESS("--align=1k needs slice tags; choose --slice-tag=8 or 16.",
                 c.tag_len != 0);
    if (c.geom.slice_size < XPAR_BLAKE3_CHUNK_LEN ||
        (c.geom.slice_size & (c.geom.slice_size - 1)) != 0) {
      FATAL_UNLESS("--align=1k needs a power-of-two slice size of at least "
                   "1 KiB; the explicit geometry does not provide one.",
                   !o->slice_size && !o->slices);
      plan_z = xpar_next_pow2(MAX(c.geom.slice_size,
                                  (u64) XPAR_BLAKE3_CHUNK_LEN));
      gr.slice_size = plan_z;
      gr.slice_count = 0;
      gs = xpar_geom_choose(&gr, &c.geom);
      if (gs != XPAR_GEOM_OK) FATAL("%s.", xpar_geom_reason(gs));
    }
  }
  c.recovery = c.geom.slice_count
                 ? resolve_recovery(&o->recovery, c.geom.slice_count,
                                    c.geom.slice_size, o->min_recovery)
                 : 0;

  budget = o->memory;

  /*  Armoured-layout frame geometry precedes planning because Y is an
      integral number of D*n*W-byte frames.  */
  if (o->layout == XPAR_LAYOUT_ARMOURED) {
    const char * bad;
    armour_params(o, c.geom.stream_length, false, &c.region_ap);
    armour_depth(o, budget ? budget : ((u64) 1 << 30), &c.region_ap);
    bad = xpar_armour_check(&c.region_ap);
    if (bad) FATAL("%s", bad);
  }

  xpar_memset(&pr, 0, sizeof pr);
  pr.stream_length   = c.m.stream_length;
  pr.memory_budget   = budget;
  pr.slice_size      = plan_z ? plan_z : o->slice_size;
  pr.slice_count     = plan_z ? 0 : o->slices;
  pr.recovery_slices = c.recovery;
  pr.cell_bytes      = (u32) o->cell_bytes;
  pr.column_chunk    = 0;
  pr.armour_frame    = o->layout == XPAR_LAYOUT_ARMOURED
                         ? (u32) (c.region_ap.depth * c.region_ap.n *
                                  (c.region_ap.symbol_bits / 8)) : 0;
  pr.field_log2      = o->field == 8 ? 8 : (o->field == 16 ? 16 : 0);
  pr.codec           = o->codec == XPAR_CLI_AUTO ? 0xFF : (u8) o->codec;
  pr.layout          = (u8) o->layout;
  pr.rotational      = xpar_is_rotational(c.base);
  pr.streaming       = o->from_stdin;
  pr.threads         = o->jobs;
  ps = xpar_plan_make(&pr, &c.plan);
  if (ps == XPAR_PLAN_NO_FIT) {
    char why[256];
    xpar_plan_explain_no_fit(&pr, why, sizeof why);
    FATAL_CODE(XPAR_EXIT_NOPLAN, "No plan fits: %s.", why);
  }
  if (ps != XPAR_PLAN_OK)
    /*  Format limits are usage errors, not planning failures.  */
    FATAL_CODE(ps == XPAR_PLAN_TOO_MANY_CELLS ? XPAR_EXIT_USAGE
                                              : XPAR_EXIT_NOPLAN,
               "%s.", xpar_plan_reason(ps));
  c.geom = c.plan.geom;
  if (!budget) budget = c.plan.mem_total;

  if (c.stream_cache) {
    u64 padded = 0, rec_bytes = 0;
    u64 encode_budget = budget > c.plan.mem_stage
                          ? budget - c.plan.mem_stage : 0;
    bool fits = c.plan.codec == XPAR_CODEC_MATRIX &&
                c.geom.slice_size &&
                c.geom.slice_count <= UINT64_MAX / c.geom.slice_size;
    if (fits) padded = c.geom.slice_count * c.geom.slice_size;
    if (fits && c.recovery > UINT64_MAX / c.geom.slice_size) fits = false;
    if (fits) rec_bytes = c.recovery * c.geom.slice_size;
    if (!fits || padded > (u64) (sz) -1 || rec_bytes > encode_budget ||
        padded > encode_budget - rec_bytes) {
      xpar_free(c.stream_cache);
      c.stream_cache = NULL;
      c.stream_cache_length = 0;
    } else {
      c.stream_cache = (u8 *) xpar_realloc(
        c.stream_cache, (sz) (padded ? padded : 1));
      if (padded > c.stream_cache_length)
        xpar_memset(c.stream_cache + c.stream_cache_length, 0,
                    (sz) (padded - c.stream_cache_length));
      c.stream_cache_length = padded;
    }
  }

  /*  The 32-byte whole-entry table overlaps the codec plan's lifetime.  */
  {
    u64 idx = 32 * (u64) c.m.count;
    if (o->memory && o->dedup != XPAR_DEDUP_NONE &&
        c.plan.mem_total + idx > budget)
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "The plan needs %" PRIu64 " bytes and the deduplication index for "
                 "%" PRIu32 " entries another %" PRIu64 ", against -m %" PRIu64 "; raise -m or use "
                 "--dedup=none.", c.plan.mem_total,
                 c.m.count, idx,
                 budget);
  }

  if (o->verbose && !o->json)
    xpar_plan_print(&c.plan, xpar_stderr, o->verbose > 1);

  if (o->armour != XPAR_ARMOUR_NONE && o->layout != XPAR_LAYOUT_ARMOURED) {
    xpar_armour_params ap;
    const char * bad;
    armour_params(o, 16384, true, &ap);
    armour_depth(o, budget, &ap);
    bad = xpar_armour_check(&ap);
    if (bad) FATAL("%s", bad);
    c.arm = xpar_armour_new(&ap);
  }

  xpar_memset(&c.sd, 0, sizeof c.sd);
  c.sd.slice_size       = c.geom.slice_size;
  c.sd.data_slice_count = c.geom.slice_count;
  c.sd.stream_length    = c.geom.stream_length;
  c.sd.file_count       = c.m.count;
  c.sd.field_log2       = c.plan.field_log2;
  c.sd.codec            = c.plan.codec;
  /*  FFT recovery is prefix-stable only inside its recorded power-of-two
      axis; --max-recovery may reserve a wider axis at no encode cost.  */
  {
    u64 axis = c.recovery;
    if (o->max_recovery.kind != XPAR_R_NONE) {
      u64 mx = resolve_recovery(&o->max_recovery, c.geom.slice_count,
                                c.geom.slice_size, 0);
      if (mx > axis) axis = mx;
    }
    if (c.plan.codec == XPAR_CODEC_MATRIX)
      c.sd.recovery_axis_log2 = c.plan.field_log2;
    else if (c.plan.codec == XPAR_CODEC_FFT_LOW)
      c.sd.recovery_axis_log2 = (u8) xpar_log2_floor(
                                  xpar_next_pow2(c.geom.slice_count));
    else
      c.sd.recovery_axis_log2 =
        (u8) xpar_log2_floor(xpar_next_pow2(axis));
    /* No recovery needs no codec axis. */
    FATAL_UNLESS("--max-recovery requires unsupported recovery axis 2^%"
                 PRIu32 "; lower it or raise --field.",
                 !c.recovery ||
                 xpar_codec_supports_axis(c.plan.codec, c.plan.field_log2,
                                          c.geom.slice_count, c.recovery,
                                          c.sd.recovery_axis_log2),
                 (u32) c.sd.recovery_axis_log2);
  }
  c.sd.layout           = (u8) o->layout;
  c.sd.align            = (u8) o->align;
  c.sd.slice_tag_len    = c.tag_len;
  c.sd.dedup_level      = c.m.dedup_level;
  if (o->align == XPAR_ALIGN_1K)
    c.sd.required_features |= XPAR_FEAT_B3_SUBTREE;
  c.sd.cell_bytes       = c.geom.cell_bytes;
  c.sd.generation       = 0;
  c.sd.posix_record_count = (o->reproducible && !c.wr.keep_posix)
                              ? 0 : c.m.posix_count;
  c.sd.stream_base      = 0;
  c.sd.file_id = (u8 (*)[XPAR_SET_ID_LEN])
                   xpar_alloc_raw((sz) (c.m.count ? c.m.count : 1) *
                                  XPAR_SET_ID_LEN);
  for (i = 0; i < c.m.count; i++)
    xpar_memcpy(c.sd.file_id[i], c.m.entry[i].file_id, XPAR_SET_ID_LEN);
  compute_set_id(&c);
  stage_chunk_cache(&c, &w);

  xpar_progress_init(&c.prog, xpar_progress_wanted(o),
                     c.geom.stream_length * (c.recovery ? 2 : 1), "Creating");
  if (o->json) xpar_progress_sink(&c.prog, xpar_json_progress_sink, &c.js);
  emit_json_set(&c);
  emit_json_files(&c);

  /*  Recovery storage uses discretionary read-ahead memory, then spills.  */
  if (ready && ready->encoded) {
    tag_pass(&c);
    FATAL_UNLESS("The direct pipe geometry changed after EOF.",
                 ready->slices == c.geom.slice_count &&
                 ready->rs.count == c.recovery &&
                 ready->rs.z == c.geom.slice_size &&
                 ready->field_log2 == c.plan.field_log2 &&
                 c.plan.codec == XPAR_CODEC_MATRIX);
  } else {
    rs_open(rsp, c.recovery, c.geom.slice_size,
            MAX(c.plan.mem_readahead,
                budget > c.plan.mem_total ? budget - c.plan.mem_total : 0),
            c.base);
    encode(&c, rsp, budget > c.plan.mem_stage ? budget - c.plan.mem_stage
                                              : budget);
  }
  xpar_progress_end(&c.prog);

  if (c.recovery && o->layout != XPAR_LAYOUT_ARMOURED) {
    u64 cap = o->volumes == XPAR_VOLS_FIXED
                ? MIN(c.recovery, (u64) o->volume_count) : 64;
    FATAL_UNLESS("The recovery-volume count is too large for this host.",
                 cap <= (u64) (sz) -1 / sizeof(*span));
    span = (vol_span *) xpar_alloc_raw((sz) cap * sizeof(*span));
    nvol = ladder(o, c.recovery, span, (u32) cap);
    FATAL_UNLESS("The recovery-volume layout is incomplete.",
                 nvol && span[nvol - 1].first + span[nvol - 1].count ==
                         c.recovery);
  }
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    data_n = o->volumes == XPAR_VOLS_FIXED ? o->volume_count : 1;
    if (!c.geom.slice_count) data_n = 1;
    else if (data_n > c.geom.slice_count) data_n = (u32) c.geom.slice_count;
  }
  name_count = nvol + 1 + data_n;
  { u64 widest = 0;
    for (i = 0; i < nvol; i++) widest = MAX(widest, span[i].count);
    xpar_vname_widths(c.recovery ? c.recovery - 1 : 0, widest, &wf, &wc); }
  names = (char **) xpar_calloc(name_count ? name_count : 1,
                                 sizeof(char *));
  names[0] = xpar_vname_index(c.base, 0);
  for (i = 0; i < nvol; i++)
    names[i + 1] = xpar_vname_recovery(c.base, 0, span[i].first,
                                       span[i].count, wf, wc);
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    int wd = MAX(xpar_digits10(data_n ? data_n - 1 : 0), 2);
    for (i = 0; i < data_n; i++)
      names[nvol + 1 + i] = xpar_vname_data(c.base, 0, i, wd);
  }
  /*  Preflight every output before the first write.  */
  if (!o->force)
    for (i = 0; i < name_count; i++) {
      xpar_file * probe;
      if (!names[i]) continue;
      probe = xpar_open(names[i], XPAR_O_RDONLY);
      if (probe) {
        xpar_close(probe);
        FATAL("'%s' exists; -f overwrites it.", names[i]);
      }
    }
  if (!o->force && o->labels && o->layout == XPAR_LAYOUT_SPLIT)
    for (i = 0; i < data_n; i++) {
      char * label;
      xpar_file * probe;
      label = xpar_vname_label(names[nvol + 1 + i]);
      probe = xpar_open(label, XPAR_O_RDONLY);
      if (probe) {
        xpar_close(probe);
        FATAL("'%s' exists; -f overwrites it.", label);
      }
      xpar_free(label);
    }

  xpar_memset(&layt, 0, sizeof layt);
  layt.count = o->layout == XPAR_LAYOUT_ARMOURED
                 ? 1 : nvol + 1 + data_n;
  layt.vol = (xpar_vol *) xpar_calloc(layt.count, sizeof(xpar_vol));
  layt.vol[0].kind   = XPAR_VOL_INDEX;
  layt.vol[0].vflags = c.arm != NULL;
  layt.vol[0].name   = xpar_strdup(xpar_path_base(names[0]));
  for (i = 0; i < nvol && o->layout != XPAR_LAYOUT_ARMOURED; i++) {
    layt.vol[i + 1].kind           = XPAR_VOL_RECOVERY;
    layt.vol[i + 1].vflags         = c.arm != NULL;
    layt.vol[i + 1].recovery_first = (u32) span[i].first;
    layt.vol[i + 1].byte_length    = span[i].count;
    layt.vol[i + 1].name           =
      xpar_strdup(xpar_path_base(names[i + 1]));
  }
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    /*  Spread the remainder across the leading volumes.  */
    u64 base = data_n ? c.geom.slice_count / data_n : 0;
    u64 rem  = data_n ? c.geom.slice_count % data_n : 0;
    u64 slice = 0;
    for (i = 0; i < data_n; i++) {
      u32 li = nvol + 1 + i;
      u64 count = base + (i < rem ? 1 : 0);
      u64 off = slice * c.geom.slice_size;
      u64 len = MIN(count * c.geom.slice_size,
                    c.geom.stream_length - off);
      layt.vol[li].kind          = XPAR_VOL_DATA;
      layt.vol[li].stream_offset = off;
      layt.vol[li].byte_length   = len;
      layt.vol[li].vol_tag       = create_stream_tag(&c, off, len);
      layt.vol[li].name = xpar_strdup(xpar_path_base(names[nvol + 1 + i]));
      slice += count;
    }
  }

  cr.setd        = &c.sd;
  cr.file        = c.m.entry;
  cr.file_count  = c.m.count;
  cr.posix       = c.m.posix;
  cr.posix_count = c.m.posix_count;
  cr.slice_crc   = o->auth_only ? NULL : c.slice_crc;
  cr.slice_count = c.geom.slice_count;
  cr.auth        = c.keyed ? &c.auth : NULL;
  cr.layt        = &layt;

  /*  Replication placement depends on the encoded critical-group size.  */
  {
    xpar_buf probe;
    xpar_buf_init(&probe);
    layt.this_volume = XPAR_VOL_STANDALONE;
    emit_crit(&c, &probe, &cr);
    crit_bytes = probe.len;
    xpar_buf_free(&probe);
  }

  output_stage = create_output_stage(c.base);
  write_names = (char **) xpar_calloc(name_count, sizeof(char *));
  for (i = 0; i < name_count; i++)
    write_names[i] = xpar_path_join(output_stage, xpar_path_base(names[i]));
  if (ready && ready->final_path) {
    for (i = 0; i < name_count; i++)
      FATAL_UNLESS("The pipe destination collides with set output '%s'.",
                   xpar_strcmp(ready->final_path, names[i]) != 0, names[i]);
    pipe_stage = xpar_path_join(output_stage, ".stdin-data");
    create_stage_input(ready, pipe_stage, &c.m);
  }

  if (o->layout == XPAR_LAYOUT_ARMOURED) {
    xpar_armour_params ap;
    xpar_armour * ra;
    xpar_buf head, tail;
    xpar_armsink sk;
    xpar_file * f;
    reader rd;
    u64 stream_at, plain_len, i2;
    u8 * slice;
    xpar_buf crtr;
    u64 strm_len = xpar_align_up(XPAR_PKT_HDR + 16 + c.geom.stream_length,
                                 XPAR_PKT_ALIGN);
    u64 rcvs_one = xpar_align_up(XPAR_PKT_HDR + 16 + c.geom.slice_size,
                                 XPAR_PKT_ALIGN);
    u64 rcvs_len;
    FATAL_UNLESS("The recovery payload is too large.",
                 !c.recovery || rcvs_one <= UINT64_MAX / c.recovery);
    rcvs_len = c.recovery * rcvs_one;
    ap = c.region_ap;
    ra = xpar_armour_new(&ap);
    xpar_buf_init(&head);
    layt.this_volume = XPAR_VOL_STANDALONE;
    emit_head(&c, &head, &cr, XPAR_VOL_STANDALONE, XPAR_VOL_INDEX, false,
              true);
    xpar_strm_write_header(&head, c.geom.stream_length, c.set_id,
                           create_key(&c));
    xpar_buf_init(&tail);
    if (c.tag_len)
      xpar_sltg_write_all(&tail, c.slice_tag, c.geom.slice_count, c.tag_len,
                          c.set_id, create_key(&c));
    if (c.cell_crc)
      xpar_slcl_write_all(&tail, c.cell_crc, c.geom.slice_count,
                          c.geom.cell_bytes, c.geom.cells_per_slice,
                          c.set_id, create_key(&c));
    xpar_buf_init(&crtr);
    xpar_crtr_write(&crtr, "xpar " PACKAGE_VERSION, c.set_id,
                    create_key(&c), &c.wr);
    /*  The prologue offset names the STRM body, not its packet header.  */
    stream_at = head.len;
    plain_len = head.len - (XPAR_PKT_HDR + 16) + strm_len + tail.len +
                rcvs_len + crtr.len;
    f = xpar_open(write_names[0], XPAR_O_WRONLY | XPAR_O_CREAT |
                                  XPAR_O_TRUNC);
    if (!f) FATAL_PERROR(write_names[0]);
    xpar_garm_write_prologue(f, xpar_armour_params_of(ra), plain_len,
                   xpar_armour_size(ra, plain_len), stream_at,
                   c.geom.stream_length);
    xpar_armsink_init(&sk, ra, f);
    xpar_armsink_put(&sk, head.data, head.len);
    slice = (u8 *) xpar_alloc_raw((sz) c.geom.slice_size);
    rd_init(&rd, &c);
    for (i2 = 0; i2 < c.geom.slice_count; i2++) {
      u64 have = xpar_slice_bytes(&c.geom, i2);
      if (!have) break;
      rd_bytes(&rd, xpar_slice_begin(&c.geom, i2), slice, have);
      xpar_armsink_put(&sk, slice, have);
    }
    rd_free(&rd);
    {
      u64 pad = strm_len - (XPAR_PKT_HDR + 16 + c.geom.stream_length);
      u8 zero[XPAR_PKT_ALIGN];
      xpar_memset(zero, 0, sizeof zero);
      if (pad) xpar_armsink_put(&sk, zero, pad);
    }
    xpar_armsink_put(&sk, tail.data, tail.len);
    for (i2 = 0; i2 < c.recovery; i2++) {
      static const u8 zero[XPAR_PKT_ALIGN] = { 0 };
      u8 rcvs_head[XPAR_PKT_HDR + 16];
      const u8 * p = rs_data(rsp, i2, slice);
      u32 pad = xpar_rcvs_stream_header(
                  rcvs_head, i2, p, (sz) c.geom.slice_size, c.set_id,
                  create_key(&c));
      xpar_armsink_put(&sk, rcvs_head, sizeof rcvs_head);
      xpar_armsink_put(&sk, p, (sz) c.geom.slice_size);
      if (pad) xpar_armsink_put(&sk, zero, pad);
    }
    xpar_free(slice);
    xpar_armsink_put(&sk, crtr.data, crtr.len);
    xpar_armsink_flush(&sk);
    xpar_armsink_free(&sk);
    xpar_xclose(f);
    xpar_buf_free(&head);
    xpar_buf_free(&tail);
    xpar_buf_free(&crtr);
    xpar_armour_free(ra);
  } else {
    xpar_buf b;
    xpar_buf_init(&b);
    layt.this_volume = XPAR_VOL_STANDALONE;
    emit_head(&c, &b, &cr, XPAR_VOL_STANDALONE, XPAR_VOL_INDEX, true, true);
    xpar_crtr_write(&b, "xpar " PACKAGE_VERSION, c.set_id, create_key(&c),
                    &c.wr);
    put_file(write_names[0], &b);
    xpar_buf_free(&b);

    for (i = 0; i < nvol; i++) {
      xpar_file * f;
      u64 payload = span[i].count * c.geom.slice_size, j;
      bool copy = xpar_replicate_here(crit_bytes, payload, i, nvol);
      u8 * scratch;
      xpar_buf_init(&b);
      layt.this_volume = i + 1;
      emit_head(&c, &b, &cr, i + 1, XPAR_VOL_RECOVERY, i == 0, copy);
      f = xpar_open(write_names[i + 1], XPAR_O_WRONLY | XPAR_O_CREAT |
                                        XPAR_O_TRUNC);
      if (!f) FATAL_PERROR(write_names[i + 1]);
      xpar_xwrite(f, b.data, b.len);
      xpar_buf_free(&b);
      scratch = rsp->mem ? NULL
                         : (u8 *) xpar_alloc_raw((sz) c.geom.slice_size);
      for (j = 0; j < span[i].count; j++) {
        static const u8 zero[XPAR_PKT_ALIGN] = { 0 };
        xpar_write_part part[3];
        u8 head[XPAR_PKT_HDR + 16];
        const u8 * p = rs_data(rsp, span[i].first + j, scratch);
        u32 pad = xpar_rcvs_stream_header(
                    head, span[i].first + j, p,
                    (sz) c.geom.slice_size, c.set_id, create_key(&c));
        part[0].data = head;  part[0].length = sizeof head;
        part[1].data = p;     part[1].length = (sz) c.geom.slice_size;
        part[2].data = zero;  part[2].length = pad;
        xpar_xwritev(f, part, pad ? 3 : 2);
      }
      xpar_free(scratch);
      xpar_buf_init(&b);
      xpar_crtr_write(&b, "xpar " PACKAGE_VERSION, c.set_id, create_key(&c),
                      &c.wr);
      xpar_xwrite(f, b.data, b.len);
      xpar_buf_free(&b);
      xpar_xclose(f);
    }

    if (o->layout == XPAR_LAYOUT_SPLIT) {
      /*  Split data volumes concatenate to the exact unpadded stream.  */
      reader rd;
      u8 * p;
      p = (u8 *) xpar_alloc_raw((sz) c.geom.slice_size);
      rd_init(&rd, &c);
      for (i = 0; i < data_n; i++) {
        u32 li = nvol + 1 + i;
        const xpar_vol * dv = &layt.vol[li];
        bool already = ready && ready->source_path && data_n == 1;
        xpar_file * f = already ? NULL : xpar_open(
                                  write_names[li], XPAR_O_WRONLY |
                                  XPAR_O_CREAT | XPAR_O_TRUNC);
        u64 at = dv->stream_offset, left = dv->byte_length;
        if (already)
          create_stage_input(ready, write_names[li], &c.m);
        if (!already && !f) FATAL_PERROR(write_names[li]);
        while (!already && left) {
          u64 take = MIN(left, c.geom.slice_size);
          rd_bytes(&rd, at, p, take);
          xpar_xwrite(f, p, (sz) take);
          at += take; left -= take;
        }
        if (f) xpar_xclose(f);
        if (o->labels) {
          char * label;
          xpar_buf lb;
          label = xpar_vname_label(write_names[li]);
          xpar_buf_init(&lb);
          layt.this_volume = li;
          emit_head(&c, &lb, &cr, li, XPAR_VOL_DATA, false, true);
          xpar_crtr_write(&lb, "xpar " PACKAGE_VERSION, c.set_id,
                          create_key(&c), &c.wr);
          put_file(label, &lb);
          xpar_buf_free(&lb);
          xpar_free(label);
        }
      }
      rd_free(&rd);
      xpar_free(p);
    }
  }

  /*  Bare split data and encoded armour regions are not packet volumes.  */
  if (!o->no_verify_after)
    verify_after(&c, write_names,
                 o->layout == XPAR_LAYOUT_ARMOURED ? 0 : nvol + 1);
  publish_outputs(o, write_names, names, name_count, nvol + 1,
                  o->labels && o->layout == XPAR_LAYOUT_SPLIT ? data_n : 0,
                  pipe_stage, ready ? ready->final_path : NULL,
                  output_stage);
  publish_chunk_cache(&c);

  if (!o->quiet && !o->json)
    xpar_fprintf(xpar_stderr, "xpar: %s: %" PRIu32 " %s, %" PRIu64
                 " slice%s of %" PRIu64 " "
                 "bytes, %" PRIu64 " recovery slice%s in %" PRIu32 " volume%s\n", c.base,
                 c.m.count,
                 c.m.count == 1 ? "entry" : "entries",
                 c.geom.slice_count,
                 PLURAL(c.geom.slice_count),
                 c.geom.slice_size,
                 c.recovery, PLURAL(c.recovery),
                 nvol, PLURAL(nvol));
  if (o->json) xpar_json_summary(&c.js, "ok", XPAR_EXIT_OK);

  rs_close(rsp);
  if (write_names != names) {
    for (i = 0; i < name_count; i++) xpar_free(write_names[i]);
    xpar_free(write_names);
  }
  xpar_free(output_stage);
  xpar_free(pipe_stage);
  xpar_free(span);
  for (i = 0; i < name_count; i++) xpar_free(names[i]);
  xpar_free(names);
  /* The layout owns its names. */
  xpar_layt_free(&layt);
  xpar_free(c.sd.file_id);
  xpar_free(c.slice_crc);
  xpar_free(c.slice_tag);
  xpar_free(c.cell_crc);
  xpar_chunk_index_free(&c.chunk_cache);
  xpar_free(c.stream_cache);
  xpar_free(c.chunk_cache_path);
  xpar_free(c.chunk_cache_stage);
  xpar_free(c.base);
  if (c.arm) xpar_armour_free(c.arm);
  xpar_occindex_free(&c.ix);
  xpar_manifest_free(&c.m);
  xpar_key_forget(&c.key, c.master);
  return XPAR_EXIT_OK;
}

int xpar_op_create(const xpar_options * o) {
  return o->from_stdin ? create_from_pipe(o) : create_regular(o, NULL);
}
