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

/* Differential codec tests against scalar generation, Cauchy form and SIMD. */

#include "t_harness.h"

#include "kernel/codec.h"
#include "kernel/gf.h"

/* GF(2^16) symbols are little-endian pairs; GF(2^8) symbols are bytes. */

static u32 sym_get(const u8 * p, sz i, bool f16) {
  return f16 ? (u32) xpar_rd16(p + 2 * i) : (u32) p[i];
}

static void sym_put(u8 * p, sz i, u32 v, bool f16) {
  if (f16) xpar_wr16(p + 2 * i, (u16) v);
  else     p[i] = (u8) v;
}

static u32 fmul(u32 a, u32 b, bool f16) {
  return f16 ? (u32) xpar_gf16_mul((u16) a, (u16) b)
             : (u32) xpar_gf8_mul((u8) a, (u8) b);
}

static u32 finv(u32 a, bool f16) {
  return f16 ? (u32) xpar_gf16_inv((u16) a) : (u32) xpar_gf8_inv((u8) a);
}

typedef struct {
  u8  kind, field;
  bool f16;
  u64 s, r;
  sz  bytes, symbols;
  u8 ** data;
  u8 ** rec;
  u8 ** ref;          /*  Pristine copy of the data slices.  */
  const u8 ** cdata;
  u32 * gen;          /*  R by S, row major; NULL until extracted.  */
} cc;

/* Ensure parameter filtering does not skip every case. */
static u32 cc_ran[3];

static void cc_ran_note(u8 kind) {
  cc_ran[kind == XPAR_CODEC_MATRIX ? 1 : kind == XPAR_CODEC_FFT_LOW ? 2 : 0]++;
}

static const char * codec_name(u8 kind) {
  return kind == XPAR_CODEC_MATRIX  ? "matrix"
       : kind == XPAR_CODEC_FFT_LOW ? "fft-low" : "fft";
}

static void cc_init(cc * c, u8 kind, u8 field, u64 s, u64 r, sz bytes) {
  u64 i;
  c->kind = kind;  c->field = field;  c->f16 = field == 16;
  c->s = s;  c->r = r;
  c->bytes = bytes;
  c->symbols = c->f16 ? bytes / 2 : bytes;
  c->data  = (u8 **) xpar_alloc_raw((sz) s * sizeof(u8 *));
  c->ref   = (u8 **) xpar_alloc_raw((sz) s * sizeof(u8 *));
  c->cdata = (const u8 **) xpar_alloc_raw((sz) s * sizeof(u8 *));
  c->rec   = (u8 **) xpar_alloc_raw((sz) r * sizeof(u8 *));
  for (i = 0; i < s; i++) {
    c->data[i] = (u8 *) xpar_alloc_aligned(bytes, 64);
    c->ref [i] = (u8 *) xpar_alloc_aligned(bytes, 64);
    c->cdata[i] = c->data[i];
  }
  for (i = 0; i < r; i++) c->rec[i] = (u8 *) xpar_alloc_aligned(bytes, 64);
  c->gen = NULL;
}

static void cc_free(cc * c) {
  u64 i;
  for (i = 0; i < c->s; i++) {
    xpar_free_aligned(c->data[i]);  xpar_free_aligned(c->ref[i]);
  }
  for (i = 0; i < c->r; i++) xpar_free_aligned(c->rec[i]);
  xpar_free(c->data);  xpar_free(c->ref);  xpar_free(c->cdata);
  xpar_free(c->rec);   xpar_free(c->gen);
}

static void cc_random(cc * c, xt_rng * r) {
  u64 i;
  for (i = 0; i < c->s; i++) {
    xt_fill(r, c->data[i], c->bytes);
    xpar_memcpy(c->ref[i], c->data[i], c->bytes);
  }
}

static void cc_encode(cc * c) {
  xpar_codec * k = xpar_codec_new(c->kind, c->field, c->s, c->r);
  xpar_codec_status st = xpar_codec_encode(k, c->cdata, c->rec, c->bytes);
  CHECK(st == XPAR_CODEC_OK, "%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64
        ": encode returned %d", codec_name(c->kind), (u32) c->field,
        c->s, c->r, (int) st);
  xpar_codec_free(k);
}

/* Derive each generator column by encoding its unit vector. */
static void cc_extract(cc * c) {
  xpar_codec * k = xpar_codec_new(c->kind, c->field, c->s, c->r);
  u64 i, j;
  sz unit = 64;   /*  One aligned block, wide enough for either field.  */
  u8 ** d   = (u8 **) xpar_alloc_raw((sz) c->s * sizeof(u8 *));
  const u8 ** cd = (const u8 **) xpar_alloc_raw((sz) c->s * sizeof(u8 *));
  u8 ** rec = (u8 **) xpar_alloc_raw((sz) c->r * sizeof(u8 *));

  for (i = 0; i < c->s; i++) {
    d[i] = (u8 *) xpar_alloc_aligned(unit, 64);
    cd[i] = d[i];
  }
  for (j = 0; j < c->r; j++) rec[j] = (u8 *) xpar_alloc_aligned(unit, 64);

  c->gen = (u32 *) xpar_alloc_raw((sz) c->r * (sz) c->s * sizeof(u32));
  for (i = 0; i < c->s; i++) {
    for (j = 0; j < c->s; j++) xpar_memset(d[j], 0, unit);
    sym_put(d[i], 0, 1, c->f16);
    xpar_codec_encode(k, cd, rec, unit);
    for (j = 0; j < c->r; j++)
      c->gen[j * c->s + i] = sym_get(rec[j], 0, c->f16);
  }

  for (i = 0; i < c->s; i++) xpar_free_aligned(d[i]);
  for (j = 0; j < c->r; j++) xpar_free_aligned(rec[j]);
  xpar_free(d);  xpar_free(cd);  xpar_free(rec);
  xpar_codec_free(k);
}

/* Scalar encoding oracle. */
static void cc_oracle(const cc * c, u8 ** out) {
  u64 i, j;
  sz t;
  for (j = 0; j < c->r; j++) {
    xpar_memset(out[j], 0, c->bytes);
    for (i = 0; i < c->s; i++) {
      u32 g = c->gen[j * c->s + i];
      if (!g) continue;
      for (t = 0; t < c->symbols; t++)
        sym_put(out[j], t,
                sym_get(out[j], t, c->f16) ^
                fmul(g, sym_get(c->data[i], t, c->f16), c->f16),
                c->f16);
    }
  }
}

static void cc_check_generator(const cc * c) {
  u64 i, j;
  for (j = 0; j < c->r; j++)
    for (i = 0; i < c->s; i++)
      if (!c->gen[j * c->s + i]) {
        CHECK(false, "%s GF(2^%" PRIu32 "): generator entry (%" PRIu64
              ", %" PRIu64 ") is zero", codec_name(c->kind), (u32) c->field,
              j, i);
        return;
      }
  /* Check the Cauchy generator against its closed form. */
  if (c->kind == XPAR_CODEC_MATRIX) {
    u32 base = c->f16 ? 65535u : 255u;
    for (j = 0; j < c->r; j++)
      for (i = 0; i < c->s; i++) {
        u32 want = finv((base - (u32) j) ^ (u32) i, c->f16);
        if (c->gen[j * c->s + i] == want) continue;
        CHECK(false, "matrix GF(2^%" PRIu32 "): entry (%" PRIu64 ", %"
              PRIu64 ") is %" PRIu32 ", not the Cauchy value %" PRIu32,
              (u32) c->field, j, i, c->gen[j * c->s + i], want);
        return;
      }
  }
}

static void test_encode_differential(u8 kind, u8 field, u64 s, u64 r,
                                     xt_rng * rng) {
  cc c;
  u8 ** want;
  u64 j;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  cc_ran_note(kind);
  cc_init(&c, kind, field, s, r, 1024);
  cc_extract(&c);
  cc_check_generator(&c);
  cc_random(&c, rng);
  cc_encode(&c);

  want = (u8 **) xpar_alloc_raw((sz) r * sizeof(u8 *));
  for (j = 0; j < r; j++) want[j] = (u8 *) xpar_alloc_aligned(c.bytes, 64);
  cc_oracle(&c, want);
  for (j = 0; j < r; j++) {
    char label[96];
    xpar_snprintf(label, sizeof label,
                  "%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64
                  " recovery slice %" PRIu64,
                  codec_name(kind), (u32) field, s, r, j);
    if (!xt_bytes_equal(label, c.rec[j], want[j], c.bytes)) break;
  }
  for (j = 0; j < r; j++) xpar_free_aligned(want[j]);
  xpar_free(want);
  cc_free(&c);
}

/* Decoding succeeds exactly when keep >= e. */
static void decode_once(cc * c, const u8 * dpres, const u8 * rpres, u64 e,
                        u64 keep) {
  xpar_codec * k = xpar_codec_new(c->kind, c->field, c->s, c->r);
  xpar_codec_status st = XPAR_CODEC_OK;
  xpar_codec_plan * pl;
  u64 i;

  for (i = 0; i < c->s; i++)
    if (!dpres[i]) xpar_memset(c->data[i], 0, c->bytes);

  pl = xpar_codec_plan_new(k, dpres, rpres, &st);
  if (keep < e) {
    CHECK(!pl && st == XPAR_CODEC_TOO_MANY_LOST,
          "%s GF(2^%" PRIu32 "): %" PRIu64 " erasures against %" PRIu64
          " recovery slices must be refused", codec_name(c->kind),
          (u32) c->field, e, keep);
    if (pl) xpar_codec_plan_free(pl);
    xpar_codec_free(k);
    return;
  }
  if (!pl) {
    CHECK(false, "%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64
          ": no plan for %" PRIu64 " erasures with %" PRIu64 " recovery",
          codec_name(c->kind), (u32) c->field, c->s, c->r, e, keep);
    xpar_codec_free(k);
    return;
  }
  st = xpar_codec_plan_apply(pl, c->data, c->rec, c->bytes);
  CHECK(st == XPAR_CODEC_OK, "%s: plan_apply returned %d",
        codec_name(c->kind), (int) st);
  for (i = 0; i < c->s; i++) {
    char label[96];
    if (dpres[i]) continue;
    xpar_snprintf(label, sizeof label,
                  "%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64
                  ": decoded slice %" PRIu64, codec_name(c->kind),
                  (u32) c->field, c->s, c->r, i);
    if (!xt_bytes_equal(label, c->data[i], c->ref[i], c->bytes)) break;
  }
  xpar_codec_plan_free(pl);
  xpar_codec_free(k);
}

static void test_decode(u8 kind, u8 field, u64 s, u64 r, xt_rng * rng,
                        u32 rounds) {
  cc c;
  u8 * dpres, * rpres;
  u32 round;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  cc_ran_note(kind);
  cc_init(&c, kind, field, s, r, 512);
  cc_random(&c, rng);
  cc_encode(&c);
  dpres = (u8 *) xpar_alloc_raw((sz) s);
  rpres = (u8 *) xpar_alloc_raw((sz) r);

  for (round = 0; round < rounds; round++) {
    u64 cap = MIN(r, s);
    u64 i, e = 0, keep = 0;
    u64 want = round == 0 ? cap : 1 + xt_below(rng, (u32) cap);
    xpar_memset(dpres, 1, (sz) s);
    xpar_memset(rpres, 1, (sz) r);
    while (e < want) {
      u64 pick = xt_below(rng, (u32) s);
      if (!dpres[pick]) continue;
      dpres[pick] = 0;  e++;
    }
    /* Retain just enough recovery, or one too few. */
    if (round % 3 == 2) {
      u64 drop = r - e + (round % 6 == 5 ? 1 : 0);
      u64 done = 0;
      while (done < drop && done < r) {
        u64 pick = xt_below(rng, (u32) r);
        if (!rpres[pick]) continue;
        rpres[pick] = 0;  done++;
      }
    }
    for (i = 0; i < r; i++) keep += rpres[i] ? 1 : 0;
    for (i = 0; i < s; i++) xpar_memcpy(c.data[i], c.ref[i], c.bytes);
    decode_once(&c, dpres, rpres, e, keep);
  }

  xpar_memset(dpres, 1, (sz) s);
  xpar_memset(rpres, 1, (sz) r);
  decode_once(&c, dpres, rpres, 0, r);

  xpar_free(dpres);  xpar_free(rpres);
  cc_free(&c);
}

/* Compare codec-level output across runtime tiers. */
static void test_tiers(u8 kind, u8 field, u64 s, u64 r, xt_rng * rng) {
  cc c;
  u8 ** base;
  u8 * dpres, * rpres;
  int saved = xpar_gf_tier(), n = xpar_gf_tier_count(), t, scalar = -1;
  u64 i, j;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  for (t = 0; t < n; t++)
    if (!xpar_strcmp(xpar_gf_tier_name(t), "scalar")) scalar = t;
  if (scalar < 0) return;

  cc_init(&c, kind, field, s, r, 4096);
  cc_random(&c, rng);
  dpres = (u8 *) xpar_alloc_raw((sz) s);
  rpres = (u8 *) xpar_alloc_raw((sz) r);
  xpar_memset(dpres, 1, (sz) s);
  xpar_memset(rpres, 1, (sz) r);
  for (i = 0; i < r && i < s; i++) dpres[i] = 0;

  xpar_gf_use_tier(scalar);
  cc_encode(&c);
  base = (u8 **) xpar_alloc_raw((sz) r * sizeof(u8 *));
  for (j = 0; j < r; j++) {
    base[j] = (u8 *) xpar_alloc_aligned(c.bytes, 64);
    xpar_memcpy(base[j], c.rec[j], c.bytes);
  }

  for (t = 0; t < n; t++) {
    char label[96];
    if (t == scalar || !xpar_gf_tier_usable(t)) continue;
    if (!xpar_gf_use_tier(t)) continue;
    for (j = 0; j < r; j++) xpar_memset(c.rec[j], 0xCC, c.bytes);
    cc_encode(&c);
    for (j = 0; j < r; j++) {
      xpar_snprintf(label, sizeof label, "%s encode on tier %s, slice %"
                    PRIu64, codec_name(kind), xpar_gf_tier_name(t), j);
      if (!xt_bytes_equal(label, c.rec[j], base[j], c.bytes)) break;
    }
    for (i = 0; i < s; i++) xpar_memcpy(c.data[i], c.ref[i], c.bytes);
    decode_once(&c, dpres, rpres, r < s ? r : s, r);
  }

  xpar_gf_use_tier(saved);
  for (j = 0; j < r; j++) xpar_free_aligned(base[j]);
  xpar_free(base);  xpar_free(dpres);  xpar_free(rpres);
  cc_free(&c);
}

/* Compare streaming matrix encoding with batch encoding. */
static void test_matrix_streaming(u8 field, u64 s, u64 r, xt_rng * rng) {
  cc c;
  xpar_codec * k;
  u8 ** want;
  u64 i, j;

  if (!xpar_codec_supports(XPAR_CODEC_MATRIX, field, s, r)) return;
  cc_init(&c, XPAR_CODEC_MATRIX, field, s, r, 2048);
  cc_random(&c, rng);
  cc_encode(&c);
  want = (u8 **) xpar_alloc_raw((sz) r * sizeof(u8 *));
  for (j = 0; j < r; j++) {
    want[j] = (u8 *) xpar_alloc_aligned(c.bytes, 64);
    xpar_memcpy(want[j], c.rec[j], c.bytes);
  }

  k = xpar_codec_new(XPAR_CODEC_MATRIX, field, s, r);
  for (i = 0; i < s; i++)
    xpar_codec_matrix_accumulate(k, i, c.data[i], 0, c.rec, r, c.bytes,
                                 i == 0);
  for (j = 0; j < r; j++) {
    char label[96];
    xpar_snprintf(label, sizeof label,
                  "matrix streaming GF(2^%" PRIu32 ") slice %" PRIu64,
                  (u32) field, j);
    if (!xt_bytes_equal(label, c.rec[j], want[j], c.bytes)) break;
  }

  for (j = 0; j < r; j++) xpar_memset(c.rec[j], 0xA5, c.bytes);
  i = 0;
  while (i < s) {
    u64 take = 1 + xt_below(rng, 5);
    if (i + take > s) take = s - i;
    xpar_codec_matrix_accumulate_many(k, i, c.cdata + i, take, 0, c.rec, r,
                                      c.bytes, i == 0);
    i += take;
  }
  for (j = 0; j < r; j++) {
    char label[96];
    xpar_snprintf(label, sizeof label,
                  "matrix batched GF(2^%" PRIu32 ") slice %" PRIu64,
                  (u32) field, j);
    if (!xt_bytes_equal(label, c.rec[j], want[j], c.bytes)) break;
  }

  /* Partial recovery ranges must not alter excluded rows. */
  if (r >= 2) {
    for (j = 0; j < r; j++) xpar_memset(c.rec[j], 0x11, c.bytes);
    for (i = 0; i < s; i++)
      xpar_codec_matrix_accumulate(k, i, c.data[i], 1, c.rec + 1, r - 1,
                                   c.bytes, i == 0);
    for (j = 0; j < c.bytes; j++)
      if (c.rec[0][j] != 0x11) {
        CHECK(false, "a recovery range starting at 1 wrote row 0");
        break;
      }
    {
      char label[96];
      xpar_snprintf(label, sizeof label,
                    "matrix partial range GF(2^%" PRIu32 ")", (u32) field);
      xt_bytes_equal(label, c.rec[1], want[1], c.bytes);
    }
  }

  xpar_codec_free(k);
  for (j = 0; j < r; j++) xpar_free_aligned(want[j]);
  xpar_free(want);
  cc_free(&c);
}

/* Recovery remains stable when data grows within one transform axis. */
static void test_prefix_stable(u8 kind, u8 field, u64 s, u64 grown, u64 r,
                               xt_rng * rng) {
  cc a, b;
  u64 i, j;

  if (!xpar_codec_supports(kind, field, s, r)) return;
  if (!xpar_codec_supports(kind, field, grown, r)) return;
  cc_init(&a, kind, field, s, r, 512);
  cc_init(&b, kind, field, grown, r, 512);
  cc_random(&a, rng);
  for (i = 0; i < grown; i++)
    if (i < s) xpar_memcpy(b.data[i], a.data[i], a.bytes);
    else       xpar_memset(b.data[i], 0, b.bytes);
  cc_encode(&a);
  cc_encode(&b);
  for (j = 0; j < r; j++) {
    char label[96];
    xpar_snprintf(label, sizeof label,
                  "%s GF(2^%" PRIu32 "): S=%" PRIu64 " to %" PRIu64
                  ", recovery slice %" PRIu64,
                  codec_name(kind), (u32) field, s, grown, j);
    if (!xt_bytes_equal(label, b.rec[j], a.rec[j], a.bytes)) break;
  }
  cc_free(&a);  cc_free(&b);
}

static void test_supports(void) {
  xt_section_begin("codec admission");

  CHECK(!xpar_codec_supports(XPAR_CODEC_MATRIX, 8, 250, 10),
        "S + R past the field must be refused in GF(2^8)");
  CHECK(xpar_codec_supports(XPAR_CODEC_MATRIX, 8, 246, 10),
        "S + R filling the field exactly is admitted");
  CHECK(!xpar_codec_supports(XPAR_CODEC_MATRIX, 8, 1, 0),
        "zero recovery slices must be refused");
  CHECK(!xpar_codec_supports(XPAR_CODEC_MATRIX, 8, 0, 1),
        "zero data slices must be refused");
  CHECK(!xpar_codec_supports(XPAR_CODEC_MATRIX, 12, 4, 2),
        "a field width the code does not have must be refused");

  CHECK(!xpar_codec_supports(XPAR_CODEC_FFT, 8, 4, 8),
        "the FFT code must refuse R > S");
  CHECK(xpar_codec_supports(XPAR_CODEC_FFT, 8, 8, 4),
        "and admit R <= S");
  CHECK(!xpar_codec_supports(XPAR_CODEC_FFT, 8, 250, 9),
        "S plus the rounded axis must fit the field");

  CHECK(!xpar_codec_supports(XPAR_CODEC_FFT_LOW, 8, 32, 4),
        "the low-rate FFT code must refuse R <= S");
  CHECK(xpar_codec_supports(XPAR_CODEC_FFT_LOW, 8, 4, 32),
        "and admit R > S");
  CHECK(!xpar_codec_supports(XPAR_CODEC_FFT_LOW, 8, 4, 253),
        "R plus the rounded axis must fit the field");
}

typedef struct { u8 kind, field;  u64 s, r; } cc_case;

/* Boundary cases small enough to extract the generator. */
static const cc_case small_cases[] = {
  { XPAR_CODEC_MATRIX,  8,   1,   1 },
  { XPAR_CODEC_MATRIX,  8,   2,   1 },
  { XPAR_CODEC_MATRIX,  8,   1,  16 },
  { XPAR_CODEC_MATRIX,  8,  16,   1 },
  { XPAR_CODEC_MATRIX,  8,  32,  32 },
  { XPAR_CODEC_MATRIX,  8, 128, 128 },
  { XPAR_CODEC_MATRIX,  8, 240,  16 },
  { XPAR_CODEC_MATRIX, 16,   1,   1 },
  { XPAR_CODEC_MATRIX, 16,  17,   5 },
  { XPAR_CODEC_MATRIX, 16,  64,  64 },
  { XPAR_CODEC_FFT,     8,   1,   1 },
  { XPAR_CODEC_FFT,     8,   4,   4 },
  { XPAR_CODEC_FFT,     8,   7,   3 },
  { XPAR_CODEC_FFT,     8,  32,  16 },
  { XPAR_CODEC_FFT,     8, 128,  64 },
  { XPAR_CODEC_FFT,    16,   4,   4 },
  { XPAR_CODEC_FFT,    16,  33,  17 },
  { XPAR_CODEC_FFT_LOW, 8,   4,  32 },
  { XPAR_CODEC_FFT_LOW, 8,   3, 100 },
  { XPAR_CODEC_FFT_LOW,16,   8,  64 }
};

/* Larger round-trip and tier-agreement cases. */
static const cc_case big_cases[] = {
  { XPAR_CODEC_MATRIX,  8, 246,  10 },
  { XPAR_CODEC_MATRIX, 16, 600,  40 },
  { XPAR_CODEC_FFT,     8, 180,  64 },
  { XPAR_CODEC_FFT,    16, 700,  60 },
  { XPAR_CODEC_FFT_LOW,16,  16, 512 }
};

int xpar_main(int argc, char ** argv) {
  xt_rng rng;
  u32 i;

  (void) argc;  (void) argv;
  xt_level_from_env(xpar_getenv("XPAR_TEST_LEVEL"));
  xt_trace_from_env(xpar_getenv("XPAR_TEST_TRACE"));
  xpar_gf_init();
  xt_seed(&rng, 0xC0DEC0DEC0DEull);

  test_supports();

  xt_section_begin("encode differential");
  for (i = 0; i < ARRAY_LEN(small_cases); i++) {
    xt_trace("%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64,
             codec_name(small_cases[i].kind), (u32) small_cases[i].field,
             small_cases[i].s, small_cases[i].r);
    test_encode_differential(small_cases[i].kind, small_cases[i].field,
                             small_cases[i].s, small_cases[i].r, &rng);
  }

  xt_section_begin("decode");
  for (i = 0; i < ARRAY_LEN(small_cases); i++) {
    xt_trace("%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64,
             codec_name(small_cases[i].kind), (u32) small_cases[i].field,
             small_cases[i].s, small_cases[i].r);
    test_decode(small_cases[i].kind, small_cases[i].field, small_cases[i].s,
                small_cases[i].r, &rng, xt_scale(6));
  }
  for (i = 0; i < ARRAY_LEN(big_cases); i++) {
    xt_trace("%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64,
             codec_name(big_cases[i].kind), (u32) big_cases[i].field,
             big_cases[i].s, big_cases[i].r);
    test_decode(big_cases[i].kind, big_cases[i].field, big_cases[i].s,
                big_cases[i].r, &rng, xt_scale(2));
  }

  xt_section_begin("tier agreement");
  for (i = 0; i < ARRAY_LEN(small_cases); i++) {
    xt_trace("%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64,
             codec_name(small_cases[i].kind), (u32) small_cases[i].field,
             small_cases[i].s, small_cases[i].r);
    test_tiers(small_cases[i].kind, small_cases[i].field, small_cases[i].s,
               small_cases[i].r, &rng);
  }
  for (i = 0; i < ARRAY_LEN(big_cases); i++) {
    xt_trace("%s GF(2^%" PRIu32 ") S=%" PRIu64 " R=%" PRIu64,
             codec_name(big_cases[i].kind), (u32) big_cases[i].field,
             big_cases[i].s, big_cases[i].r);
    test_tiers(big_cases[i].kind, big_cases[i].field, big_cases[i].s,
               big_cases[i].r, &rng);
  }

  xt_section_begin("matrix streaming");
  test_matrix_streaming(8, 32, 8, &rng);
  test_matrix_streaming(8, 1, 1, &rng);
  test_matrix_streaming(16, 40, 7, &rng);

  xt_section_begin("prefix stability");
  test_prefix_stable(XPAR_CODEC_MATRIX, 8, 8, 20, 4, &rng);
  test_prefix_stable(XPAR_CODEC_MATRIX, 16, 30, 90, 6, &rng);
  test_prefix_stable(XPAR_CODEC_FFT, 8, 16, 32, 4, &rng);
  test_prefix_stable(XPAR_CODEC_FFT, 16, 40, 80, 8, &rng);
  test_prefix_stable(XPAR_CODEC_FFT_LOW, 8, 4, 4, 32, &rng);

  xt_section_begin("coverage");
  CHECK(cc_ran[0] >= 8, "the FFT cases must not all have been skipped "
        "(%" PRIu32 " ran)", cc_ran[0]);
  CHECK(cc_ran[1] >= 8, "the matrix cases must not all have been skipped "
        "(%" PRIu32 " ran)", cc_ran[1]);
  CHECK(cc_ran[2] >= 4, "the low-rate cases must not all have been skipped "
        "(%" PRIu32 " ran)", cc_ran[2]);

  return xt_finish("t_codec");
}
