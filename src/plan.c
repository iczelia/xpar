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

/*  Codec, field, column, cell, and buffer planning.  */

#include "plan.h"
#include "kernel/codec.h"
#include "platform/port-thread.h"

/*  Heuristic machine constants affect only candidate ordering.  */

/*  Assumed last-level cache for candidate ranking.  */
#define PLAN_LLC        ((u64) 16 << 20)

/*  Below this a strided request stops amortising its own
    completion cost even on flash.  */
#define PLAN_MIN_REQ    ((u64) 64 << 10)

#define PLAN_STAGE_CAP  ((u64) 32 << 20)
#define PLAN_STAGE_SHR  32

#define PLAN_RA_QUANTUM ((u64) 64 << 20)
#define PLAN_RA_SMALL   ((u64) 1 << 20)

/*  GF(2^16) log and exponent tables.  */
#define PLAN_GF16_TABLES ((u64) 393212)

/*  Maximum armour frame batch.  */
#define PLAN_ARMOUR_CAP ((u64) 64 << 20)

/*  Pass count that triggers a warning.  */
#define PLAN_PASS_WARN  8

/*  Estimated MAC rates in MB/s by working-set class.  */
static u64 mac_rate(u8 codec, u8 field_log2, u64 ws) {
  static const u64 fft8 [3] = { 44000, 22600,  7500 };
  static const u64 fft16[3] = { 22000, 10400,  3550 };
  static const u64 mat8 [3] = { 62400, 46500, 40400 };
  static const u64 mat16[3] = { 22900, 22900, 20500 };
  int cls = ws <= PLAN_LLC ? 0 : (ws <= 8 * PLAN_LLC ? 1 : 2);
  if (XPAR_CODEC_IS_FFT(codec))
    return field_log2 == 8 ? fft8[cls] : fft16[cls];
  return field_log2 == 8 ? mat8[cls] : mat16[cls];
}

/*  Matrix decoding has a setup term; FFT decoding is flat in erasures.  */
static u64 decode_us(u8 codec, u8 field_log2, u64 work) {
  u64 rate;
  if (XPAR_CODEC_IS_FFT(codec))
    rate = field_log2 == 8 ? 22600 : 16850;
  else
    rate = field_log2 == 8 ? 46500 : 29000;
  return work / rate + (!XPAR_CODEC_IS_FFT(codec)
                         ? (field_log2 == 8 ? 250 : 7500) : 0);
}

#define PLAN_SEEK_US_ROT  10000ull
#define PLAN_SEEK_US_SSD     50ull

static u64 pow2_of(u64 v) { return xpar_next_pow2(v); }

static u64 floor64(u64 v) { return v & ~(u64) 63; }

static u64 sub_sat(u64 a, u64 b) { return a > b ? a - b : 0; }

static u64 mul_sat(u64 a, u64 b) {
  return (a && b > (u64) -1 / a) ? (u64) -1 : a * b;
}

static void group(char * buf, u64 v) {
  char tmp[24];
  int n = 0, i, k = 0;
  if (!v) { buf[0] = '0';  buf[1] = '\0';  return; }
  while (v) { tmp[n++] = (char) ('0' + (int) (v % 10));  v /= 10; }
  for (i = n - 1; i >= 0; i--) { buf[k++] = tmp[i];  if (i && i % 3 == 0) buf[k++] = ','; }
  buf[k] = '\0';
}

static void human(char * buf, sz cap, u64 v) {
  static const char * const u[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };
  u64 scale = 1;
  int i = 0;
  while (i < 5 && v >= scale * 1024) { scale *= 1024;  i++; }
  if (!i) { xpar_snprintf(buf, cap, "%" PRIu64 " B", v);  return; }
  xpar_snprintf(buf, cap, "%" PRIu64 ".%" PRIu64 " %s",
                (v / scale),
                (v % scale * 10 / scale), u[i]);
}

const char * xpar_plan_reason(xpar_plan_status s) {
  switch (s) {
    case XPAR_PLAN_OK:           return "ok";
    case XPAR_PLAN_NO_FIT:       return "no plan fits the memory budget";
    case XPAR_PLAN_BAD_GEOMETRY: return "Z, S or R outside what the field "
                                        "admits";
    case XPAR_PLAN_TOO_MANY_CELLS:
      return "slice exceeds 65536 cells; raise --cell or use more slices";
    case XPAR_PLAN_NO_CODEC:     return "no codec can express these "
                                        "parameters";
  }
  return "refused";
}

u64 xpar_plan_default_memory(void) {
  u64 phys = xpar_physical_memory();
  u64 cap  = sizeof(void *) >= 8 ? ((u64) 1 << 30) : ((u64) 512 << 20);
  u64 want = phys ? phys / 4 : cap;
  if (want > cap) want = cap;
  if (want < ((u64) 1 << 20)) want = (u64) 1 << 20;
  return want;
}

typedef u64 (* footprint_fn)(u8, u8, u64, u64, sz);

static u64 fits_in(footprint_fn fp, u8 codec, u8 field, u64 s, u64 r,
                   u64 limit, u64 hi) {
  u64 lo = 64, best = 0;
  if (hi < 64) hi = 64;
  hi = floor64(hi);
  while (lo <= hi) {
    u64 mid = floor64(lo + (hi - lo) / 2);
    if (mid < 64) mid = 64;
    if (fp(codec, field, s, r, (sz) mid) <= limit) {
      best = mid;
      if (mid == hi) break;
      lo = mid + 64;
    } else {
      if (mid == 64) break;
      hi = mid - 64;
    }
  }
  return best;
}

static u64 fits_decode_axis(u8 codec, u8 field, u64 s, u64 r, u8 axis,
                            u64 limit, u64 hi) {
  u64 lo = 64, best = 0;
  if (hi < 64) hi = 64;
  hi = floor64(hi);
  while (lo <= hi) {
    u64 mid = floor64(lo + (hi - lo) / 2);
    if (mid < 64) mid = 64;
    if (xpar_codec_decode_footprint_axis(codec, field, s, r, axis,
                                         (sz) mid) <= limit) {
      best = mid;
      if (mid == hi) break;
      lo = mid + 64;
    } else {
      if (mid == 64) break;
      hi = mid - 64;
    }
  }
  return best;
}

/*  Candidate evaluation.  */

typedef struct {
  u8   codec, field;
  u64  s, r, z;
  u64  budget;         /*  What is left for the codec after staging.  */
  bool rotational;
  u32  want_chunk;     /*  A requested column width, or 0 for none.  */
} cand_in;

typedef struct {
  bool feasible;
  const char * why;
  u32  chunk;
  u64  passes;
  u64  working_set;    /*  Resident bytes at that chunk.  */
  u64  encode_work;
  u64  repair_work;
  u64  requests;       /*  Extra I/O requests the pass structure implies.  */
  u64  decode_buffers;
  u64  decode_chunk;
  u64  decode_passes;
} cand_out;

static void eval(const cand_in * in, cand_out * out) {
  bool low = in->codec == XPAR_CODEC_FFT_LOW;
  u64 m = pow2_of(low ? in->s : in->r);
  u64 n = pow2_of(m + (low ? in->r : in->s));

  xpar_memset(out, 0, sizeof(*out));
  out->encode_work = xpar_codec_encode_work(in->codec, in->s, in->r, 1) *
                     in->z;

  if (in->codec == XPAR_CODEC_MATRIX) {
    u64 per, chunk = in->z;
    /*  One accumulator and one input slice is the floor; below it no
        partition of the recovery axis helps.  */
    if (xpar_codec_encode_footprint(in->codec, in->field, in->s, 1,
                                    (sz) chunk) > in->budget) {
      out->why = "R * Z does not fit even at one recovery slice per pass";
      return;
    }
    per = in->r;
    while (per > 1 &&
           xpar_codec_encode_footprint(in->codec, in->field, in->s, per,
                                       (sz) chunk) > in->budget)
      per = (per + 1) / 2;
    out->feasible    = true;
    out->chunk       = (u32) chunk;
    out->passes      = xpar_ceil_div(in->r, per);
    out->working_set = xpar_codec_encode_footprint(in->codec, in->field,
                                                   in->s, per, (sz) chunk);
    /*  Matrix work for one recovered slice.  */
    out->repair_work = in->s * in->z;
    out->requests    = out->passes;          /*  One sequential sweep each.  */
    return;
  }

  {
    u64 cache, chunk;
    cache = fits_in(xpar_codec_encode_footprint, in->codec, in->field,
                    in->s, in->r, PLAN_LLC, in->z);
    if (!cache) cache = 64;
    chunk = cache;
    /*  Rule (b). On a rotating device this rule is deliberately not
        applied: a wider column there buys nothing a seek does not eat, and
        the drop below usually sends the plan to the matrix codec anyway.  */
    if (!in->rotational && chunk < PLAN_MIN_REQ) chunk = PLAN_MIN_REQ;
    if (in->want_chunk) chunk = in->want_chunk;
    if (chunk > in->z) chunk = in->z;
    chunk = floor64(chunk);
    if (!chunk) chunk = MIN((u64) 64, in->z);
    /*  Rule (b) is a preference and -m is a bound, so the budget wins.  */
    if (xpar_codec_encode_footprint(in->codec, in->field, in->s, in->r,
                                    (sz) chunk) > in->budget) {
      chunk = fits_in(xpar_codec_encode_footprint, in->codec, in->field,
                      in->s, in->r, in->budget, in->z);
      if (!chunk) { out->why = "(S + 2m) buffers do not fit, even at a 64-byte column";  return; }
    }
    out->feasible    = true;
    out->chunk       = (u32) chunk;
    out->passes      = xpar_ceil_div(in->z, chunk);
    out->working_set = xpar_codec_encode_footprint(in->codec, in->field,
                                                   in->s, in->r, (sz) chunk);
    out->repair_work    = n * (u64) xpar_log2_floor(n) * in->z;
    out->requests       = out->passes * in->s;
    out->decode_buffers = in->s + in->r + n;
    out->decode_chunk   = fits_in(xpar_codec_decode_footprint, in->codec,
                                  in->field, in->s, in->r, PLAN_LLC, in->z);
    if (!out->decode_chunk) out->decode_chunk = 64;
    out->decode_passes  = xpar_ceil_div(in->z, out->decode_chunk);
  }
}

static u64 score_us(const cand_in * in, const cand_out * c) {
  u64 per_us = mac_rate(in->codec, in->field, c->working_set);
  u64 seek   = in->rotational ? PLAN_SEEK_US_ROT : PLAN_SEEK_US_SSD;
  u64 t;
  if (!per_us) per_us = 1;
  t = c->encode_work / per_us +
      decode_us(in->codec, in->field, c->repair_work);
  if (c->requests > ((u64) -1 - t) / seek) return (u64) -1;
  return t + c->requests * seek;
}

static bool fft_worth_it(u64 s, u64 r) {
  u64 m = pow2_of(r);
  u64 butter = ((s + m) / 2) * (u64) xpar_log2_floor(m);
  if (!butter) return false;
  return s * r >= 20 * butter;
}

u64 xpar_plan_repair_crossover(u8 fft_codec, u8 field, u64 s, u64 r,
                               u64 z) {
  u64 m = pow2_of(fft_codec == XPAR_CODEC_FFT_LOW ? s : r);
  u64 n = pow2_of(m + (fft_codec == XPAR_CODEC_FFT_LOW ? r : s));
  u64 fft_work, one, fft_time, e;
  u64 levels = (u64) xpar_log2_floor(n);
  if (!s || !r || !z || n > UINT64_MAX / MAX((u64) 1, levels))
    return 0;
  fft_work = n * (u64) xpar_log2_floor(n);
  if (fft_work > UINT64_MAX / z || s > UINT64_MAX / z) return 0;
  fft_work *= z; one = s * z;
  fft_time = decode_us(fft_codec, field, fft_work);
  for (e = 1; e <= r; e++) {
    u64 work = one > UINT64_MAX / e ? UINT64_MAX : one * e;
    if (decode_us(XPAR_CODEC_MATRIX, field, work) >= fft_time) return e;
  }
  return 0;
}

static u32 dedup_target(u64 stream_length, u64 budget, u64 payload,
                        u64 want) {
  u64 m_dedup = budget / 4;
  u64 floor_mem, floor_meta, thr, t = (u64) 1 << 20;
  if (want) return (u32) MIN(want, (u64) 0xFFFFFFFFu);
  if (!stream_length) return 0;
  floor_mem  = m_dedup ? xpar_ceil_div(32 * stream_length, m_dedup) : 0;
  thr        = payload / 40;
  if (thr < ((u64) 1 << 20)) thr = (u64) 1 << 20;
  floor_meta = xpar_ceil_div(16 * stream_length, thr);
  if (floor_mem  > t) t = floor_mem;
  if (floor_meta > t) t = floor_meta;
  return (u32) MIN(t, (u64) 0xFFFFFFFFu);
}

/*  Estimate process-lifetime tables and manifest state.  */
static u64 fixed_bytes(const xpar_geom * g, u8 field_log2, u8 tag, u32 files) {
  /*  The verifier's stream read-ahead is budgeted from -m, so it is not
      a fixed cost here.  */
  u64 s = g->slice_count, v = 0;
  if (field_log2 == 16) v += PLAN_GF16_TABLES;
  v += s * 4;                                   /*  Slice CRC32C.  */
  v += s * (u64) tag;                           /*  Strong slice tags.  */
  v += s * (u64) g->cells_per_slice * 4;        /*  Cell CRC32C.  */
  v += (u64) files * 512;                       /*  Manifest entries.  */
  return v;
}

/*  Estimate loaded volume-image bytes.  */
static u64 images_bytes(const xpar_setd * sd, u64 s, u64 r, u64 z) {
  u64 v = r * z;                                /*  Recovery payload.  */
  if (sd->layout != XPAR_LAYOUT_SIDECAR) v += s * z;
  return v;
}

/*  Allocate a capped, frame-aligned armour batch.  */
static u64 armour_batch_bytes(u64 budget, u64 free_now, u32 frame) {
  u64 want = budget / 4;
  if (!frame) return 0;
  if (want > PLAN_ARMOUR_CAP) want = PLAN_ARMOUR_CAP;
  if (want > free_now) want = free_now;
  if (want < frame) return frame;
  return want / frame * frame;
}

/*  Staging.  */

static u64 stage_bytes(u64 budget, u64 z, int threads) {
  u64 want = budget / PLAN_STAGE_SHR;
  u64 need = z + (u64) (threads > 0 ? threads : 1) * 4096;
  if (want > PLAN_STAGE_CAP) want = PLAN_STAGE_CAP;
  if (want < ((u64) 1 << 20)) want = (u64) 1 << 20;
  return MAX(want, need);
}

/*  The plan.  */

static void add_cand(xpar_plan * p, u8 codec, u8 field, const cand_out * c,
                     u64 work) {
  xpar_plan_cand * d;
  if (p->cand_count >= XPAR_PLAN_MAX_CAND) return;
  d = &p->cand[p->cand_count++];
  d->codec        = codec;
  d->field_log2   = field;
  d->feasible     = c->feasible;
  d->why          = c->why;
  d->encode_work  = work;
  d->working_set  = c->working_set;
  d->column_chunk = c->chunk;
  d->passes       = c->passes;
}

xpar_plan_status xpar_plan_make(const xpar_plan_req * req, xpar_plan * out) {
  xpar_geom_req gr;
  xpar_geom_status gs;
  u64 budget = req->memory_budget ? req->memory_budget
                                  : xpar_plan_default_memory();
  u64 r = req->recovery_slices, stage, best_score = 0;
  int i, best = -1, threads = req->threads > 0 ? req->threads
                                               : xpar_cpu_count();
  u8 probe = req->field_log2 ? req->field_log2 : 16;
  struct { u8 codec, field; } list[XPAR_PLAN_MAX_CAND];
  cand_out won;
  int n = 0;

  xpar_memset(out, 0, sizeof(*out));
  out->threads = threads;

  xpar_memset(&gr, 0, sizeof(gr));
  gr.stream_length = req->stream_length;
  gr.slice_size    = req->slice_size;
  gr.slice_count   = req->slice_count;
  gr.recovery      = r;
  gr.cell_bytes    = req->cell_bytes;
  gr.armour_frame  = req->layout == XPAR_LAYOUT_ARMOURED ? req->armour_frame
                                                         : 0;
  gr.field_log2    = probe;
  gs = xpar_geom_choose(&gr, &out->geom);
  if (gs == XPAR_GEOM_CELLS) return XPAR_PLAN_TOO_MANY_CELLS;
  if (gs != XPAR_GEOM_OK) return XPAR_PLAN_BAD_GEOMETRY;
  out->recovery_slices = r;

  /* Empty streams and parity-free sets need no codec. */
  if (!out->geom.slice_count || !r) {
    out->field_log2 = probe;
    out->codec      = XPAR_CODEC_MATRIX;
    out->mem_stage  = stage_bytes(budget, out->geom.slice_size, threads);
    out->mem_armour = armour_batch_bytes(budget,
                        sub_sat(budget, out->mem_stage), gr.armour_frame);
    out->mem_total  = out->mem_stage + out->mem_armour;
    out->mem_fixed  = fixed_bytes(&out->geom, out->field_log2,
                                  req->slice_tag, req->file_count);
    out->mem_peak   = out->mem_total + out->mem_fixed;
    return out->mem_total <= budget ? XPAR_PLAN_OK : XPAR_PLAN_NO_FIT;
  }

  if (xpar_codec_supports(XPAR_CODEC_MATRIX, 8, out->geom.slice_count, r)) {
    list[n].codec = XPAR_CODEC_MATRIX;  list[n++].field = 8;
  }
  { u8 fft = r > out->geom.slice_count ? XPAR_CODEC_FFT_LOW
                                       : XPAR_CODEC_FFT;
  if (xpar_codec_supports(fft, 8, out->geom.slice_count, r)) {
    list[n].codec = fft;  list[n++].field = 8;
  }
  if (xpar_codec_supports(XPAR_CODEC_MATRIX, 16, out->geom.slice_count, r)) {
    list[n].codec = XPAR_CODEC_MATRIX;  list[n++].field = 16;
  }
  if (xpar_codec_supports(fft, 16, out->geom.slice_count, r)) {
    list[n].codec = fft;  list[n++].field = 16;
  }
  }

  stage = stage_bytes(budget, out->geom.slice_size, threads);
  out->mem_stage = stage;

  Fi(n,
    cand_in in;
    cand_out c;
    u64 s;
    xpar_memset(&c, 0, sizeof(c));
    if (req->field_log2 && list[i].field != req->field_log2) continue;
    if (req->codec != 0xFF && list[i].codec != req->codec &&
        !(req->codec == XPAR_CODEC_FFT &&
          list[i].codec == XPAR_CODEC_FFT_LOW)) continue;
    in.codec      = list[i].codec;
    in.field      = list[i].field;
    in.s          = out->geom.slice_count;
    in.r          = r;
    in.z          = out->geom.slice_size;
    in.budget     = sub_sat(budget, stage);
    in.rotational = req->rotational;
    in.want_chunk = req->column_chunk;
    if (XPAR_CODEC_IS_FFT(in.codec) && req->streaming) {
      c.why = "the FFT codec needs every slice at once, and a pipe "
              "gives one at a time";
      add_cand(out, in.codec, in.field, &c, 0);
      continue;
    }
    if (XPAR_CODEC_IS_FFT(in.codec) && !fft_worth_it(in.s, r) &&
        req->codec == 0xFF) {
      c.why = "the arithmetic saving is under 20x";
      add_cand(out, in.codec, in.field, &c, 0);
      continue;
    }
    eval(&in, &c);
    add_cand(out, in.codec, in.field, &c, c.encode_work);
    if (!c.feasible) continue;
    s = score_us(&in, &c);
    if (best < 0 || s < best_score) {
      best_score = s;  best = i;  won = c;
    });

  if (best < 0) {
    int k;
    bool any = false;
    Fk(out->cand_count, if (out->cand[k].feasible || out->cand[k].why) any = true);
    return any && out->cand_count ? XPAR_PLAN_NO_FIT : XPAR_PLAN_NO_CODEC;
  }

  out->codec        = list[best].codec;
  out->field_log2   = list[best].field;
  out->column_chunk = won.chunk;
  out->passes       = won.passes;
  out->encode_work  = won.encode_work;
  out->mem_codec    = won.working_set;

  out->mem_armour = armour_batch_bytes(budget,
                      sub_sat(budget, out->mem_codec + out->mem_stage),
                      gr.armour_frame);

  {
    u64 left = sub_sat(budget, out->mem_codec + out->mem_stage +
                               out->mem_armour);
    u64 q;
    /*  Reading further ahead than the stream is long buys nothing, so the
        clamp comes before the quantum is chosen or a small set would round
        its whole read-ahead away.  */
    if (left > out->geom.stream_length) left = out->geom.stream_length;
    q = left >= PLAN_RA_QUANTUM ? PLAN_RA_QUANTUM : PLAN_RA_SMALL;
    out->mem_readahead = left / q * q;
  }

  out->dedup_target_chunk = dedup_target(out->geom.stream_length, budget,
                                         r * out->geom.slice_size,
                                         0);
  out->mem_total = out->mem_codec + out->mem_stage + out->mem_readahead +
                   out->mem_armour;
  out->mem_fixed = fixed_bytes(&out->geom, out->field_log2, req->slice_tag,
                               req->file_count);
  out->mem_peak  = out->mem_total + out->mem_fixed;
  if (out->mem_total > budget) return XPAR_PLAN_NO_FIT;
  return XPAR_PLAN_OK;
}

xpar_plan_status xpar_plan_for_repair(const xpar_setd * sd,
                                      u64 recovery_slices,
                                      u64 memory_budget, int threads,
                                      xpar_plan * out) {
  u64 budget = memory_budget ? memory_budget : xpar_plan_default_memory();
  u64 r, stage, chunk;

  xpar_memset(out, 0, sizeof(*out));
  if (!xpar_geom_from_setd(sd, &out->geom)) return XPAR_PLAN_BAD_GEOMETRY;
  out->threads    = threads > 0 ? threads : xpar_cpu_count();
  out->codec      = sd->codec;
  out->field_log2 = sd->field_log2;

  out->mem_images = images_bytes(sd, out->geom.slice_count,
                                 recovery_slices, out->geom.slice_size);

  if (!out->geom.slice_count) {
    out->mem_stage = stage_bytes(budget, out->geom.slice_size, out->threads);
    out->mem_total = out->mem_stage;
    out->mem_fixed = fixed_bytes(&out->geom, out->field_log2, sd->slice_tag_len,
                                 (u32) sd->file_count);
    out->mem_peak  = out->mem_total + out->mem_fixed + out->mem_images;
    return out->mem_total <= budget ? XPAR_PLAN_OK : XPAR_PLAN_NO_FIT;
  }

  r = recovery_slices;
  if (r > xpar_setd_recovery_limit(sd))
    r = xpar_setd_recovery_limit(sd);
  if (!r) return XPAR_PLAN_BAD_GEOMETRY;
  if (!xpar_codec_supports_axis(sd->codec, sd->field_log2,
                                out->geom.slice_count, r,
                                sd->recovery_axis_log2))
    return XPAR_PLAN_BAD_GEOMETRY;
  out->recovery_slices = r;

  stage = stage_bytes(budget, out->geom.slice_size, out->threads);
  out->mem_stage = stage;

  if (sd->codec == XPAR_CODEC_MATRIX) {
    /*  E slices recovered per pass, E = 1 being always affordable if
        anything is: the matrix decoder's footprint is linear in E and a
        repair splits along it freely.  */
    u64 per = r;
    if (xpar_codec_decode_footprint(sd->codec, sd->field_log2,
                                    out->geom.slice_count, 1,
                                    (sz) out->geom.slice_size) >
        sub_sat(budget, stage))
      return XPAR_PLAN_NO_FIT;
    while (per > 1 &&
           xpar_codec_decode_footprint(sd->codec, sd->field_log2,
                                       out->geom.slice_count, per,
                                       (sz) out->geom.slice_size) >
           sub_sat(budget, stage))
      per = (per + 1) / 2;
    out->column_chunk = (u32) out->geom.slice_size;
    out->passes       = xpar_ceil_div(r, per);
    out->mem_codec    = xpar_codec_decode_footprint(sd->codec,
                          sd->field_log2, out->geom.slice_count, per,
                          (sz) out->geom.slice_size);
  } else {
    chunk = fits_decode_axis(sd->codec, sd->field_log2,
                             out->geom.slice_count, r,
                             sd->recovery_axis_log2, PLAN_LLC,
                             out->geom.slice_size);
    if (!chunk) chunk = 64;
    if (xpar_codec_decode_footprint_axis(
          sd->codec, sd->field_log2, out->geom.slice_count, r,
          sd->recovery_axis_log2, (sz) chunk) >
        sub_sat(budget, stage))
      chunk = fits_decode_axis(sd->codec, sd->field_log2,
                               out->geom.slice_count, r,
                               sd->recovery_axis_log2,
                               sub_sat(budget, stage), out->geom.slice_size);
    if (!chunk) return XPAR_PLAN_NO_FIT;
    out->column_chunk = (u32) chunk;
    out->passes       = xpar_ceil_div(out->geom.slice_size, chunk);
    out->mem_codec = xpar_codec_decode_footprint_axis(
                       sd->codec, sd->field_log2, out->geom.slice_count, r,
                       sd->recovery_axis_log2, (sz) chunk);
  }
  out->encode_work = xpar_codec_encode_work(sd->codec,
                                            out->geom.slice_count, r, 1) *
                     out->geom.slice_size;
  out->mem_total = out->mem_codec + out->mem_stage;
  out->mem_fixed = fixed_bytes(&out->geom, out->field_log2, sd->slice_tag_len,
                               (u32) sd->file_count);
  out->mem_peak  = out->mem_total + out->mem_fixed + out->mem_images;
  if (out->mem_total > budget) return XPAR_PLAN_NO_FIT;
  return XPAR_PLAN_OK;
}

void xpar_plan_explain_no_fit(const xpar_plan_req * req, char * buf, sz cap) {
  xpar_plan p;
  xpar_plan_req q = *req;
  u64 need = (u64) -1, best_b = 0, budget;
  u64 lo, hi, mid;
  bool mat;
  int i;
  char a[32], b[32];

  budget = req->memory_budget ? req->memory_budget
                              : xpar_plan_default_memory();

  /*  The budget that would admit some plan at the requested geometry, by
      bisection on -m: the plan is monotone in the budget, so the smallest
      admitting value is well defined.  */
  lo = (u64) 1 << 20;
  hi = (u64) 1 << 20;
  Fi(44,
    q.memory_budget = hi;
    if (xpar_plan_make(&q, &p) == XPAR_PLAN_OK) break;
    lo = hi;
    if (hi > ((u64) 1 << 62)) break;
    hi *= 2);
  q.memory_budget = hi;
  if (xpar_plan_make(&q, &p) == XPAR_PLAN_OK) {
    while (lo + (1 << 20) < hi) {
      mid = lo + (hi - lo) / 2;
      q.memory_budget = mid;
      if (xpar_plan_make(&q, &p) == XPAR_PLAN_OK) hi = mid;  else lo = mid;
    }
    need = hi;
  }

  /*  Find a slice count that fits the budget.  */
  q = *req;
  q.memory_budget = budget;
  q.slice_size    = 0;
  for (i = 0; i <= 16; i++) {
    q.slice_count = (u64) 1 << i;
    if (q.slice_count > ((u64) 1 << 16) - 1) break;
    if (req->stream_length && q.slice_count > req->stream_length) break;
    if (xpar_plan_make(&q, &p) == XPAR_PLAN_OK) { best_b = q.slice_count; }
  }

  /*  Check whether the matrix codec fits, regardless of speed.  */
  q = *req;
  q.memory_budget = budget;
  q.codec         = XPAR_CODEC_MATRIX;
  mat = xpar_plan_make(&q, &p) == XPAR_PLAN_OK;

  human(a, sizeof a, need == (u64) -1 ? 0 : need);
  human(b, sizeof b, budget);
  if (need == (u64) -1) {
    xpar_snprintf(buf, cap,
                  "no addressable -m fits; split the set. At -m %s, %s; "
                  "--codec=matrix %s",
                  b, !best_b ? "no -b fits"
                    : req->slice_size ? "replacing -s with -b can fit"
                                      : "some -b does fit",
                  mat ? "does fit" : "does not fit");
    return;
  }
  {
    char fit[64];
    /*  -s and -b are mutually exclusive.  */
    if (best_b && req->slice_size)
      xpar_snprintf(fit, sizeof fit,
                    "replace -s with -b %" PRIu64, best_b);
    else if (best_b)
      xpar_snprintf(fit, sizeof fit, "-b %" PRIu64 " fits this -m",
                    best_b);
    else
      xpar_snprintf(fit, sizeof fit, "no -b fits this -m");
    xpar_snprintf(buf, cap,
                  "raise -m to %s; %s; --codec=matrix %s at -m %s",
                  a, fit, mat ? "does fit" : "does not fit either", b);
  }
}

/*  Return the smallest one-pass budget, or 0.  */
static u64 one_pass_budget(const xpar_plan_req * req) {
  xpar_plan_req q = *req;
  xpar_plan p;
  u64 lo = 0, hi = (u64) 1 << 20;
  int i;
  Fi(44,
    q.memory_budget = hi;
    if (xpar_plan_make(&q, &p) == XPAR_PLAN_OK && p.passes <= 1) break;
    lo = hi;
    if (hi > ((u64) 1 << 62)) return 0;
    hi *= 2);
  q.memory_budget = hi;
  if (xpar_plan_make(&q, &p) != XPAR_PLAN_OK || p.passes > 1) return 0;
  while (lo + ((u64) 1 << 20) < hi) {
    u64 mid = lo + (hi - lo) / 2;
    q.memory_budget = mid;
    if (xpar_plan_make(&q, &p) == XPAR_PLAN_OK && p.passes <= 1) hi = mid;
    else lo = mid;
  }
  return hi;
}

bool xpar_plan_pass_advice(const xpar_plan_req * req, const xpar_plan * p,
                           char * buf, sz cap) {
  u64 need;
  char a[32], b[32], g[28];
  if (p->passes <= PLAN_PASS_WARN) return false;
  need = one_pass_budget(req);
  group(g, p->passes);
  human(a, sizeof a, mul_sat(p->passes, p->geom.stream_length));
  if (!need) {
    xpar_snprintf(buf, cap, "%s passes read %s; increase -s or reduce -r",
                  g, a);
    return true;
  }
  /*  Round up to whole MiB for display.  */
  need = xpar_ceil_div(need, (u64) 1 << 20) << 20;
  human(b, sizeof b, need);
  xpar_snprintf(buf, cap, "%s passes read %s; -m %s makes one pass", g, a,
                b);
  return true;
}

static const char * codec_name(u8 c) {
  return c == XPAR_CODEC_FFT_LOW ? "fft-low"
       : c == XPAR_CODEC_FFT ? "fft" : "matrix";
}

static void expo(char * buf, sz cap, u64 v) {
  int e = 0;
  u64 t = v;
  if (!v) { xpar_snprintf(buf, cap, "0");  return; }
  while (t >= 1000) { t /= 10;  e++; }
  if (t < 100) { xpar_snprintf(buf, cap, "%" PRIu64, t);
                 return; }
  xpar_snprintf(buf, cap, "%" PRIu64 ".%02" PRIu64 "e%d", (t / 100),
                (t % 100), e + 2);
}

void xpar_plan_print(const xpar_plan * p, xpar_file * out, bool verbose) {
  char g1[28], g2[28], g3[28], g4[28], h1[32], h2[32], h3[32], h4[32];
  char w1[32];
  u64 m = pow2_of(p->recovery_slices);
  u64 n = pow2_of(m + p->geom.slice_count);
  int i;

  group(g1, p->geom.stream_length);
  group(g2, p->geom.slice_size);
  group(g3, p->geom.slice_count);
  group(g4, p->recovery_slices);
  xpar_fprintf(out, "  geometry   : L = %s  Z = %s  S = %s  R = %s\n",
               g1, g2, g3, g4);

  group(g1, p->geom.slice_count + p->recovery_slices);
  xpar_fprintf(out, "  field      : S + R = %s %s 256, so GF(2^%" PRIu8 ")\n", g1,
               p->geom.slice_count + p->recovery_slices > 256 ? ">" : "<=",
               p->field_log2);

  if (verbose) {
    Fi(p->cand_count,
      const xpar_plan_cand * c = &p->cand[i];
      xpar_fprintf(out, "  %-11s: %s/GF%" PRIu8 "  ", i ? "" : "candidates",
                   codec_name(c->codec), c->field_log2);
      if (!c->feasible) {
        xpar_fprintf(out, "dropped: %s\n", c->why ? c->why : "does not fit");
        continue;
      }
      expo(w1, sizeof w1, c->encode_work);
      human(h1, sizeof h1, c->working_set);
      group(g1, c->passes);
      group(g2, c->column_chunk);
      xpar_fprintf(out, "encode %s MAC-bytes, %s pass%s, C = %s B, %s\n",
                   w1, g1, c->passes == 1 ? "" : "es", g2, h1));
  }

  expo(w1, sizeof w1, p->encode_work);
  human(h1, sizeof h1, p->mem_codec);
  human(h2, sizeof h2, p->mem_readahead);
  human(h3, sizeof h3, p->mem_stage);
  human(h4, sizeof h4, p->mem_peak ? p->mem_peak : p->mem_total);
  group(g1, p->column_chunk);
  xpar_fprintf(out, "  codec      : %s  (GF(2^%" PRIu8 "), C = %s B)\n",
               codec_name(p->codec), p->field_log2, g1);
  xpar_fprintf(out, "  memory     : work buffers %s;  read-ahead %s;  "
               "stage + hash %s\n", h1, h2, h3);
  if (p->mem_armour || p->mem_fixed || p->mem_images) {
    char h5[32], h6[32], h7[32];
    human(h5, sizeof h5, p->mem_armour);
    human(h6, sizeof h6, p->mem_fixed);
    human(h7, sizeof h7, p->mem_images);
    xpar_fprintf(out, "               armour frames %s;  tables + buffers "
                 "%s;  volume images %s\n", h5, h6, h7);
    human(h5, sizeof h5, p->mem_total);
    xpar_fprintf(out, "               total %s, of which -m bounds %s\n",
                 h4, h5);
  } else
    xpar_fprintf(out, "               total %s\n", h4);

  if (p->geom.cell_bytes) {
    u64 last = p->geom.slice_size -
               (u64) (p->geom.cells_per_slice - 1) * p->geom.cell_bytes;
    group(g1, p->geom.cell_bytes);
    group(g2, p->geom.cells_per_slice);
    group(g3, last);
    group(g4, 4 * p->geom.slice_count * p->geom.cells_per_slice);
    xpar_fprintf(out, "  cells      : Y = %s B, K = %s per slice "
                 "(last cell %s B)\n"
                 "               erasure budget is %" PRIu64 " per column, not "
                 "%" PRIu64 " per set\n"
                 "               SLCL = %s B\n", g1, g2, g3,
                 p->recovery_slices,
                 p->recovery_slices, g4);
    /*  Equal C and Y remain independent choices.  */
    if (p->geom.cell_bytes == p->column_chunk)
      xpar_fputs("               note Y == C here by coincidence; C is "
                 "cache tiling and Y is\n"
                 "               erasure granularity, and neither is "
                 "derived from the other\n", out);
  } else
    xpar_fputs("  cells      : none; Z is below the format's 4096-byte "
               "cell floor\n", out);

  group(g1, p->passes);
  group(g2, p->geom.stream_length);
  xpar_fprintf(out, "  passes     : %s %s read%s totalling %s bytes\n", g1,
               XPAR_CODEC_IS_FFT(p->codec) ? "strided" : "sequential",
               p->passes == 1 ? "" : "s", g2);
  if (p->passes > PLAN_PASS_WARN) {
    group(g1, p->passes);
    human(h1, sizeof h1, mul_sat(p->passes, p->geom.stream_length));
    xpar_fprintf(out, "               %s passes read %s total\n", g1, h1);
  }

  if (verbose && XPAR_CODEC_IS_FFT(p->codec)) {
    u64 cross = xpar_plan_repair_crossover(p->codec, p->field_log2,
                                           p->geom.slice_count,
                                           p->recovery_slices,
                                           p->geom.slice_size);
    group(g1, p->geom.slice_count + p->recovery_slices + n);
    xpar_fprintf(out, "  repair     : FFT decode cost does not depend on "
                 "the loss count;\n"
                 "               decode needs %s buffers against the "
                 "encoder's %" PRIu64 "\n",
                 g1, (p->geom.slice_count + 2 * m));
    if (cross)
      xpar_fprintf(out,
                   "               calibrated matrix/FFT crossover is "
                   "about %" PRIu64 " lost slices (%.2f%% of S)\n",
                   cross,
                   p->geom.slice_count
                     ? 100.0 * (f64) cross / (f64) p->geom.slice_count : 0.0);
  }
  if (p->dedup_target_chunk) {
    group(g1, p->dedup_target_chunk);
    xpar_fprintf(out, "  dedup      : chunk target the memory rule admits "
                 "is %s B\n", g1);
  }
}
