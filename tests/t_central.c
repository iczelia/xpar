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

/* Central recovery tests: each column has an independent erasure budget. */

#include "t_harness.h"

#include "slice.h"
#include "kernel/codec.h"
#include "kernel/gf.h"

static const char * codec_name_of(u8 kind) {
  return kind == XPAR_CODEC_MATRIX  ? "matrix"
       : kind == XPAR_CODEC_FFT_LOW ? "fft-low" : "fft";
}

typedef struct {
  xpar_geom g;
  u64  r;
  u8   kind, field;
  u8 * stream;     /*  S by Z.  */
  u8 * pristine;
  u8 * recovery;   /*  R by Z.  */
  u8 ** data;      /*  Scratch pointer arrays, S and R long.  */
  u8 ** rec;
  const u8 ** cdata;
} ct;

static u64 col_off(const ct * c, u32 j) {
  return (u64) j * (c->g.cell_bytes ? c->g.cell_bytes : c->g.slice_size);
}

static void ct_init(ct * c, u8 kind, u8 field, u64 z, u32 y, u64 s, u64 r,
                    xt_rng * rng) {
  xpar_memset(c, 0, sizeof *c);
  c->kind = kind;  c->field = field;  c->r = r;
  c->g.slice_size    = z;
  c->g.slice_count   = s;
  c->g.stream_length = s * z;
  c->g.stream_base   = 0;
  c->g.cell_bytes    = y;
  c->g.cells_per_slice = y ? (u32) xpar_ceil_div(z, y) : 1;
  c->stream   = (u8 *) xpar_alloc_aligned((sz) (s * z), 64);
  c->pristine = (u8 *) xpar_alloc_aligned((sz) (s * z), 64);
  c->recovery = (u8 *) xpar_alloc_aligned((sz) (r * z), 64);
  c->data  = (u8 **) xpar_alloc_raw((sz) s * sizeof(u8 *));
  c->cdata = (const u8 **) xpar_alloc_raw((sz) s * sizeof(u8 *));
  c->rec   = (u8 **) xpar_alloc_raw((sz) r * sizeof(u8 *));
  xt_fill(rng, c->stream, (sz) (s * z));
  xpar_memcpy(c->pristine, c->stream, (sz) (s * z));
}

static void ct_free(ct * c) {
  xpar_free_aligned(c->stream);  xpar_free_aligned(c->pristine);
  xpar_free_aligned(c->recovery);
  xpar_free(c->data);  xpar_free(c->cdata);  xpar_free(c->rec);
}

static void ct_columns(ct * c, u32 j) {
  u64 i, off = col_off(c, j);
  for (i = 0; i < c->g.slice_count; i++) {
    c->data[i]  = c->stream + i * c->g.slice_size + off;
    c->cdata[i] = c->data[i];
  }
  for (i = 0; i < c->r; i++)
    c->rec[i] = c->recovery + i * c->g.slice_size + off;
}

static void ct_encode(ct * c) {
  xpar_codec * k = xpar_codec_new(c->kind, c->field, c->g.slice_count, c->r);
  u32 j;
  for (j = 0; j < c->g.cells_per_slice; j++) {
    sz bytes = (sz) xpar_cell_size(&c->g, j);
    ct_columns(c, j);
    CHECK(xpar_codec_encode(k, c->cdata, c->rec, bytes) == XPAR_CODEC_OK,
          "encoding column %" PRIu32 " failed", j);
  }
  xpar_codec_free(k);
}

typedef struct {
  u64 groups;        /*  Distinct column erasure patterns.  */
  u64 recovered;     /*  Columns decoded.  */
  u64 refused;       /*  Columns the recovery could not reach.  */
  u64 wrong;         /*  Columns that decoded to the wrong bytes.  */
} ct_result;

static void ct_repair(ct * c, const xpar_erasures * e, const u8 * rpres,
                      ct_result * out) {
  xpar_codec * k = xpar_codec_new(c->kind, c->field, c->g.slice_count, c->r);
  xpar_col_groups cg;
  u32 gi, n;
  u64 i;

  xpar_memset(out, 0, sizeof *out);
  xpar_col_groups_build(e, &cg);
  out->groups = cg.group_count;

  for (gi = 0; gi < cg.group_count; gi++) {
    xpar_codec_status st = XPAR_CODEC_OK;
    xpar_codec_plan * pl = xpar_codec_plan_new(k, cg.group[gi].present,
                                               rpres, &st);
    if (!pl) {
      out->refused += cg.group[gi].column_count;
      continue;
    }
    for (n = 0; n < cg.group[gi].column_count; n++) {
      u32 j = cg.group[gi].column[n];
      sz bytes = (sz) xpar_cell_size(&c->g, j);
      ct_columns(c, j);
      /* Do not let decoding depend on erased bytes. */
      for (i = 0; i < c->g.slice_count; i++)
        if (xpar_cell_bad(e, i, j)) xpar_memset(c->data[i], 0, bytes);
      if (xpar_codec_plan_apply(pl, c->data, (const u8 * const *) c->rec,
                                bytes) !=
          XPAR_CODEC_OK) {
        out->refused++;
        continue;
      }
      out->recovered++;
      for (i = 0; i < c->g.slice_count; i++) {
        u64 at = i * c->g.slice_size + col_off(c, j);
        if (!xpar_memcmp(c->stream + at, c->pristine + at, bytes)) continue;
        out->wrong++;
        break;
      }
    }
    xpar_codec_plan_free(pl);
  }
  xpar_col_groups_free(&cg);
  xpar_codec_free(k);
}

/* Erase depth[j] random cells in column j. */
static void mark_profile(xpar_erasures * e, const u64 * depth, xt_rng * rng) {
  u32 j;
  u64 i;
  xpar_erasures_clear(e);
  for (j = 0; j < e->cells_per_slice; j++) {
    u64 want = depth[j], done = 0;
    if (want > e->slice_count) want = e->slice_count;
    while (done < want) {
      i = xt_below(rng, (u32) e->slice_count);
      if (xpar_cell_bad(e, i, j)) continue;
      xpar_cell_mark(e, i, j);
      done++;
    }
  }
}

/* Each column has its own R-erasure budget. */
static void test_full_budget_every_column(u8 kind, u8 field, u64 z, u32 y,
                                          u64 s, u64 r, xt_rng * rng) {
  ct c;
  xpar_erasures e;
  ct_result res;
  u8 * rpres;
  u64 * depth;
  u32 j;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  ct_init(&c, kind, field, z, y, s, r, rng);
  ct_encode(&c);
  xpar_erasures_init(&e, s, c.g.cells_per_slice);
  depth = (u64 *) xpar_alloc_raw((sz) c.g.cells_per_slice * sizeof(u64));
  for (j = 0; j < c.g.cells_per_slice; j++) depth[j] = r;
  mark_profile(&e, depth, rng);

  CHECK_U64(e.bad_count, (u64) c.g.cells_per_slice * r,
            "K times R cells were erased");
  CHECK_U64(xpar_erasures_max_depth(&e), r, "and the deepest column holds R");

  rpres = (u8 *) xpar_alloc_raw((sz) r);
  xpar_memset(rpres, 1, (sz) r);
  ct_repair(&c, &e, rpres, &res);

  CHECK_U64(res.refused, 0, "%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64
            " K=%" PRIu32 ": no column was refused", codec_name_of(kind),
            (u32) field, s, r, c.g.cells_per_slice);
  CHECK_U64(res.wrong, 0, "no column decoded to the wrong bytes");
  CHECK_U64(res.recovered, c.g.cells_per_slice, "every column decoded");
  CHECK(xpar_memcmp(c.stream, c.pristine, (sz) (s * z)) == 0,
        "the whole stream came back byte for byte");

  xpar_free(depth);  xpar_free(rpres);
  xpar_erasures_free(&e);
  ct_free(&c);
}

/* Exceeding one column's budget must not lose other columns. */
static void test_one_column_over(u8 kind, u8 field, u64 z, u32 y, u64 s,
                                 u64 r, xt_rng * rng) {
  ct c;
  xpar_erasures e;
  ct_result res;
  u8 * rpres;
  u64 * depth;
  u32 j, victim;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  ct_init(&c, kind, field, z, y, s, r, rng);
  ct_encode(&c);
  xpar_erasures_init(&e, s, c.g.cells_per_slice);
  depth = (u64 *) xpar_alloc_raw((sz) c.g.cells_per_slice * sizeof(u64));
  victim = xt_below(rng, c.g.cells_per_slice);
  for (j = 0; j < c.g.cells_per_slice; j++)
    depth[j] = j == victim ? r + 1 : r;
  if (depth[victim] > s) { xpar_free(depth);  xpar_erasures_free(&e);
                           ct_free(&c);  return; }
  mark_profile(&e, depth, rng);

  CHECK_U64(xpar_erasures_max_depth(&e), r + 1,
            "the deepest column is one past the budget");

  rpres = (u8 *) xpar_alloc_raw((sz) r);
  xpar_memset(rpres, 1, (sz) r);
  ct_repair(&c, &e, rpres, &res);

  CHECK_U64(res.refused, 1, "exactly the over-budget column was refused");
  CHECK_U64(res.recovered, c.g.cells_per_slice - 1,
            "and every other column decoded");
  CHECK_U64(res.wrong, 0, "none of those decoded to the wrong bytes");

  /* Validate decoded columns where the refused column remains damaged. */
  {
    u64 i, bad = 0;
    for (i = 0; i < s; i++)
      for (j = 0; j < c.g.cells_per_slice; j++) {
        u64 at = i * z + col_off(&c, j);
        sz bytes = (sz) xpar_cell_size(&c.g, j);
        if (j == victim) continue;
        if (xpar_memcmp(c.stream + at, c.pristine + at, bytes)) bad++;
      }
    CHECK_U64(bad, 0, "every cell outside the refused column is intact");
  }

  xpar_free(depth);  xpar_free(rpres);
  xpar_erasures_free(&e);
  ct_free(&c);
}

static void test_missing_recovery(u8 kind, u8 field, u64 z, u32 y, u64 s,
                                  u64 r, u64 gone, xt_rng * rng) {
  ct c;
  xpar_erasures e;
  ct_result res;
  u8 * rpres;
  u64 * depth;
  u32 j;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  ct_init(&c, kind, field, z, y, s, r, rng);
  ct_encode(&c);
  xpar_erasures_init(&e, s, c.g.cells_per_slice);
  depth = (u64 *) xpar_alloc_raw((sz) c.g.cells_per_slice * sizeof(u64));
  rpres = (u8 *) xpar_alloc_raw((sz) r);
  xpar_memset(rpres, 1, (sz) r);
  for (j = 0; j < gone && j < r; j++) rpres[j] = 0;

  for (j = 0; j < c.g.cells_per_slice; j++) depth[j] = r - gone;
  mark_profile(&e, depth, rng);
  ct_repair(&c, &e, rpres, &res);
  CHECK_U64(res.refused, 0, "R minus the missing recovery still decodes");
  CHECK_U64(res.wrong, 0, "and decodes correctly");

  xpar_memcpy(c.stream, c.pristine, (sz) (s * z));
  for (j = 0; j < c.g.cells_per_slice; j++) depth[j] = r - gone + 1;
  if (depth[0] <= s) {
    mark_profile(&e, depth, rng);
    ct_repair(&c, &e, rpres, &res);
    CHECK_U64(res.refused, c.g.cells_per_slice,
              "one past the surviving recovery must be refused everywhere");
    CHECK_U64(res.recovered, 0, "and nothing must be handed back");
  }

  xpar_free(depth);  xpar_free(rpres);
  xpar_erasures_free(&e);
  ct_free(&c);
}

/* A lost slice consumes one erasure in every column. */
static void test_whole_slices(u8 kind, u8 field, u64 z, u32 y, u64 s, u64 r,
                              xt_rng * rng) {
  ct c;
  xpar_erasures e;
  ct_result res;
  u8 * rpres;
  u64 i, done = 0;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  ct_init(&c, kind, field, z, y, s, r, rng);
  ct_encode(&c);
  xpar_erasures_init(&e, s, c.g.cells_per_slice);
  while (done < r) {
    i = xt_below(rng, (u32) s);
    if (xpar_cell_bad(&e, i, 0)) continue;
    xpar_erasures_mark_slice(&e, i);
    done++;
  }
  CHECK_U64(e.bad_count, r * c.g.cells_per_slice,
            "R whole slices are R times K cells");
  CHECK_U64(xpar_erasures_max_depth(&e), r, "at depth R");

  rpres = (u8 *) xpar_alloc_raw((sz) r);
  xpar_memset(rpres, 1, (sz) r);
  ct_repair(&c, &e, rpres, &res);
  CHECK_U64(res.groups, 1, "every column shares one erasure pattern");
  CHECK_U64(res.refused, 0, "R whole slices are inside the budget");
  CHECK_U64(res.wrong, 0, "and come back exactly");
  CHECK(xpar_memcmp(c.stream, c.pristine, (sz) (s * z)) == 0,
        "R whole slices come back byte for byte");

  xpar_free(rpres);
  xpar_erasures_free(&e);
  ct_free(&c);
}

static void test_random_profiles(u8 kind, u8 field, u64 z, u32 y, u64 s,
                                 u64 r, u32 rounds, xt_rng * rng) {
  ct c;
  xpar_erasures e;
  u8 * rpres;
  u64 * depth;
  u32 round, j;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  ct_init(&c, kind, field, z, y, s, r, rng);
  ct_encode(&c);
  xpar_erasures_init(&e, s, c.g.cells_per_slice);
  depth = (u64 *) xpar_alloc_raw((sz) c.g.cells_per_slice * sizeof(u64));
  rpres = (u8 *) xpar_alloc_raw((sz) r);
  xpar_memset(rpres, 1, (sz) r);

  for (round = 0; round < rounds; round++) {
    ct_result res;
    u64 worst = 0, over = 0, total = 0;
    xpar_memcpy(c.stream, c.pristine, (sz) (s * z));
    for (j = 0; j < c.g.cells_per_slice; j++) {
      depth[j] = xt_below(rng, (u32) (r + 2));
      if (depth[j] > s) depth[j] = s;
      total += depth[j];
      if (depth[j] > worst) worst = depth[j];
      if (depth[j] > r) over++;
    }
    mark_profile(&e, depth, rng);
    CHECK_U64(xpar_erasures_max_depth(&e), worst,
              "max_depth follows the profile");
    ct_repair(&c, &e, rpres, &res);
    CHECK_U64(res.refused, over,
              "%s GF(2^%" PRIu32 "): %" PRIu64 " cells lost over %" PRIu32
              " columns, deepest %" PRIu64 " against R = %" PRIu64,
              codec_name_of(kind), (u32) field, total,
              c.g.cells_per_slice, worst, r);
    CHECK_U64(res.wrong, 0, "nothing decoded to the wrong bytes");
    if (!over)
      CHECK(xpar_memcmp(c.stream, c.pristine, (sz) (s * z)) == 0,
            "a profile inside the budget restores the whole stream");
  }

  xpar_free(depth);  xpar_free(rpres);
  xpar_erasures_free(&e);
  ct_free(&c);
}

void xt_run_central(void) {
  xt_rng rng;
  u32 i;

  /* Z, Y, S, R; the third case has a short final column. */
  static const struct { u64 z;  u32 y;  u64 s, r; } shapes[] = {
    { 65536, 8192,  32,  6 },
    { 65536, 16384, 24,  4 },
    { 65536, 24576, 20,  5 },
    { 16384, 4096,  40,  8 },
    { 65536, 0,     32,  6 }    /* K = 1. */
  };
  static const struct { u8 kind, field; } codecs[] = {
    { XPAR_CODEC_MATRIX,  8 },
    { XPAR_CODEC_MATRIX, 16 },
    { XPAR_CODEC_FFT,     8 },
    { XPAR_CODEC_FFT,    16 }
  };

  xt_seed(&rng, 0xCE27A1ull);

  for (i = 0; i < ARRAY_LEN(shapes); i++) {
    u32 k;
    for (k = 0; k < ARRAY_LEN(codecs); k++) {
      xt_trace("Z=%" PRIu64 " Y=%" PRIu32 " S=%" PRIu64 " R=%" PRIu64
               " %s GF(2^%" PRIu32 ")", shapes[i].z, shapes[i].y,
               shapes[i].s, shapes[i].r, codec_name_of(codecs[k].kind),
               (u32) codecs[k].field);

      xt_section_begin("full budget in every column");
      test_full_budget_every_column(codecs[k].kind, codecs[k].field,
                                    shapes[i].z, shapes[i].y, shapes[i].s,
                                    shapes[i].r, &rng);

      xt_section_begin("one column over budget");
      test_one_column_over(codecs[k].kind, codecs[k].field, shapes[i].z,
                           shapes[i].y, shapes[i].s, shapes[i].r, &rng);

      xt_section_begin("missing recovery slices");
      test_missing_recovery(codecs[k].kind, codecs[k].field, shapes[i].z,
                            shapes[i].y, shapes[i].s, shapes[i].r, 2, &rng);

      xt_section_begin("whole slices lost");
      test_whole_slices(codecs[k].kind, codecs[k].field, shapes[i].z,
                        shapes[i].y, shapes[i].s, shapes[i].r, &rng);

      xt_section_begin("random profiles");
      test_random_profiles(codecs[k].kind, codecs[k].field, shapes[i].z,
                           shapes[i].y, shapes[i].s, shapes[i].r,
                           xt_scale(3), &rng);
    }
  }

}
