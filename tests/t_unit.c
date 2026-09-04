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

/* Property tests for geometry, erasures, occurrences, paths and packets. */

#include "t_harness.h"
#include "container.h"
#include "manifest.h"
#include "pathname.h"
#include "slice.h"
#include "volname.h"
#include "platform/port-fs.h"
#include "kernel/blake3.h"
#include "kernel/crc32c.h"
#include "kernel/gf.h"

static void test_helpers(void) {
  xt_rng r;
  u32 i;
  u8 buf[8];
  char hex[17];

  xt_section_begin("helpers");
  xt_seed(&r, 0x1111);

  CHECK_U64(xpar_ceil_div(0, 7), 0, "ceil_div(0, 7)");
  CHECK_U64(xpar_ceil_div(1, 7), 1, "ceil_div(1, 7)");
  CHECK_U64(xpar_ceil_div(7, 7), 1, "ceil_div(7, 7)");
  CHECK_U64(xpar_ceil_div(8, 7), 2, "ceil_div(8, 7)");
  /* Avoid overflow in the common ceil-division formula. */
  CHECK_U64(xpar_ceil_div((u64) -1, 2), ((u64) -1) / 2 + 1,
            "ceil_div near 2^64");

  CHECK_U64(xpar_next_pow2(0), 1, "next_pow2(0)");
  CHECK_U64(xpar_next_pow2(1), 1, "next_pow2(1)");
  CHECK_U64(xpar_next_pow2(65), 128, "next_pow2(65)");
  CHECK_U64(xpar_next_pow2((u64) 1 << 62), (u64) 1 << 62, "next_pow2(2^62)");

  Fi(xt_scale(4096),
    u64 v = ((u64) xt_next(&r) << 32) | xt_next(&r);
    u64 a = (u64) 1 << (xt_below(&r, 20) + 1);
    u64 up;
    v &= ((u64) 1 << 48) - 1;
    up = xpar_align_up(v, a);
    if (up % a || up < v || up - v >= a) {
      CHECK(false, "align_up(%" PRIu64 ", %" PRIu64 ") = %" PRIu64,
            v, a, up);
      break;
    }
    if (!xpar_is_pow2(xpar_next_pow2(v & 0xFFFFFFu))) {
      CHECK(false, "next_pow2 produced a non-power of two");
      break;
    });
  CHECK(i == xt_scale(4096), "the alignment sweep ran to the end");

  Fi(xt_scale(2048),
    u64 v = ((u64) xt_next(&r) << 32) | xt_next(&r);
    xpar_wr16(buf, (u16) v);
    xpar_wr32(buf + 2, (u32) v);
    if (xpar_rd16(buf) != (u16) v || xpar_rd32(buf + 2) != (u32) v) {
      CHECK(false, "16 or 32 bit round trip failed");
      break;
    }
    xpar_wr64(buf, v);
    if (xpar_rd64(buf) != v) { CHECK(false, "64 bit round trip failed");
                               break; });
  CHECK(i == xt_scale(2048), "the byte-order sweep ran to the end");

  xt_fill(&r, buf, 8);
  xpar_hex(hex, buf, 8);
  CHECK(xpar_hex_prefix(buf, 8, ""), "the empty prefix must match");
  CHECK(xpar_hex_prefix(buf, 8, hex), "a full digest is its own prefix");
  hex[3] = hex[3] == 'a' ? 'b' : 'a';
  CHECK(!xpar_hex_prefix(buf, 8, hex), "a changed nibble must not match");

  CHECK_U64(xpar_digits10(0), 1, "digits10(0)");
  CHECK_U64(xpar_digits10(9), 1, "digits10(9)");
  CHECK_U64(xpar_digits10(10), 2, "digits10(10)");
  CHECK_U64(xpar_digits10((u64) -1), 20, "digits10(2^64 - 1)");

  xpar_memset(buf, 0x5A, 8);
  CHECK(!xpar_has_nul(buf, 8), "no NUL in a constant fill");
  buf[7] = 0;
  CHECK(xpar_has_nul(buf, 8), "a trailing NUL must be seen");
  CHECK(xpar_ct_equal(buf, buf, 8), "constant-time compare of equal buffers");
  {
    u8 other[8];
    xpar_memcpy(other, buf, 8);
    other[0] ^= 0x01;
    CHECK(!xpar_ct_equal(buf, other, 8), "a difference in the first byte");
    xpar_memcpy(other, buf, 8);
    other[7] ^= 0x80;
    CHECK(!xpar_ct_equal(buf, other, 8), "a difference in the last byte");
    CHECK(xpar_ct_equal(buf, other, 7), "a length that excludes it");
  }
}

/* Independent bitwise CRC-32C oracle. */
static u32 crc_reference(const u8 * p, sz n) {
  u32 c = 0xFFFFFFFFu;
  sz i;
  int k;
  Fi(n,
    c ^= p[i];
    Fk(8, c = (c >> 1) ^ (0x82F63B78u & (u32) -(i32) (c & 1))));
  return ~c;
}

static void test_crc32c(void) {
  xt_rng r;
  u8 * buf;
  u32 i;
  const sz cap = 4096;

  xt_section_begin("crc32c");
  xpar_crc32c_init();
  xt_seed(&r, 0x2222);
  buf = (u8 *) xpar_alloc_raw(cap);
  xt_fill(&r, buf, cap);

  /* Cover dispatch length classes. */
  for (i = 0; i <= 300; i++)
    CHECK_U64(xpar_crc32c(0, buf, i), crc_reference(buf, i),
              "crc32c of %" PRIu32 " bytes", i);
  CHECK_U64(xpar_crc32c(0, buf, cap), crc_reference(buf, cap),
            "crc32c of %" PRIu64 " bytes", (u64) cap);

  Fi(xt_scale(256),
    sz cut = (sz) xt_below(&r, (u32) cap + 1);
    u32 a = xpar_crc32c(0, buf, cut);
    u32 b = xpar_crc32c(0, buf + cut, cap - cut);
    u32 op[XPAR_CRC32C_OP_WORDS];
    CHECK_U64(xpar_crc32c_combine(a, b, cap - cut), xpar_crc32c(0, buf, cap),
              "combine at cut %" PRIu64, (u64) cut);
    /*  Check the reusable operator and a second CRC with the same operator.  */
    xpar_crc32c_shift_op(op, cap - cut);
    CHECK_U64(xpar_crc32c_combine_op(op, a, b), xpar_crc32c(0, buf, cap),
              "combine_op at cut %" PRIu64, (u64) cut);
    CHECK_U64(xpar_crc32c_combine_op(op, a ^ 0x5A5A5A5Au, b),
              xpar_crc32c_combine(a ^ 0x5A5A5A5Au, b, cap - cut),
              "reused operator at cut %" PRIu64, (u64) cut));

  /* Stored CRC shifting includes the zero suffix CRC. */
  Fi(xt_scale(64),
    u32 pad = xt_below(&r, 1024);
    u8 * tmp = (u8 *) xpar_calloc(2048 + pad, 1);
    xpar_memcpy(tmp, buf, 2048);
    CHECK_U64(xpar_crc32c_shift(xpar_crc32c(0, buf, 2048), pad) ^
              xpar_crc32c(0, tmp + 2048, pad),
              xpar_crc32c(0, tmp, 2048 + pad),
              "shift by %" PRIu32 " zero bytes", pad);
    xpar_free(tmp));

  xpar_free(buf);
}

static void test_blake3(void) {
  xt_rng r;
  u8 * buf;
  u8 a[32], b[32];
  u8 key[32];
  const sz cap = 8192;
  u32 i;

  xt_section_begin("blake3");
  xt_seed(&r, 0x3333);
  buf = (u8 *) xpar_alloc_raw(cap);
  xt_fill(&r, buf, cap);
  xt_fill(&r, key, 32);

  Fi(xt_scale(96),
    xpar_blake3_t h;
    sz cut = (sz) xt_below(&r, (u32) cap + 1);
    xpar_blake3_hash(buf, cap, a, 32);
    xpar_blake3_init(&h);
    xpar_blake3_update(&h, buf, cut);
    xpar_blake3_update(&h, buf + cut, cap - cut);
    xpar_blake3_final(&h, b, 32);
    if (!xt_bytes_equal("streamed digest", b, a, 32)) break);

  xpar_blake3_hash(buf, cap, a, 32);
  xpar_blake3_hash_keyed(key, buf, cap, b, 32);
  CHECK(xpar_memcmp(a, b, 32) != 0, "keying must change the digest");

  {
    u8 wide[128], part[128];
    xpar_blake3_t h;
    xpar_blake3_init(&h);
    xpar_blake3_update(&h, buf, cap);
    xpar_blake3_final(&h, wide, sizeof wide);
    xpar_blake3_final(&h, part, 32);
    xt_bytes_equal("XOF prefix", part, wide, 32);
    xpar_blake3_final_seek(&h, 64, part, 64);
    xt_bytes_equal("XOF seek", part, wide + 64, 64);
  }

  {
    const sz n = 4096;
    xpar_blake3_t h;
    xpar_blake3_subtree_tag(buf, n, 0, a, 32);
    xpar_blake3_hash(buf, n, b, 32);
    CHECK(xpar_memcmp(a, b, 32) != 0,
          "a subtree tag must not be the root digest");
    xpar_blake3_subtree_tag(buf, n, 7, b, 32);
    CHECK(xpar_memcmp(a, b, 32) != 0,
          "a subtree tag must depend on its chunk position");
    xpar_blake3_subtree_stream_init(&h, NULL, 7);
    xpar_blake3_update(&h, buf, n / 2);
    xpar_blake3_update(&h, buf + n / 2, n - n / 2);
    xpar_blake3_subtree_stream_final(&h, a, 32);
    xt_bytes_equal("streamed subtree tag", a, b, 32);
    xpar_blake3_subtree_tag_keyed(key, buf, n, 7, a, 32);
    CHECK(xpar_memcmp(a, b, 32) != 0,
          "keying must change a subtree tag");
    xpar_blake3_subtree_stream_init(&h, key, 7);
    xpar_blake3_update(&h, buf, n);
    xpar_blake3_subtree_stream_final(&h, b, 32);
    xt_bytes_equal("streamed keyed subtree tag", b, a, 32);
  }

  xpar_free(buf);
}

static void test_geometry(void) {
  xpar_geom_req q;
  xpar_geom g;
  xt_rng r;
  u32 i, accepted;

  xt_section_begin("geometry");
  xt_seed(&r, 0x4444);

  xpar_memset(&q, 0, sizeof q);
  q.field_log2 = 8;
  q.stream_length = 1 << 20;
  q.slice_size = 4096;
  q.slice_count = 16;
  CHECK(xpar_geom_choose(&q, &g) == XPAR_GEOM_EXCLUSIVE,
        "-s with -b must be refused");

  q.slice_count = 0;
  q.slice_size = 100;
  CHECK(xpar_geom_choose(&q, &g) == XPAR_GEOM_QUANTUM,
        "a slice size off the 64 byte quantum must be refused");

  q.slice_size = 32;
  CHECK(xpar_geom_choose(&q, &g) == XPAR_GEOM_QUANTUM,
        "a slice size under the floor must be refused");

  q.slice_size = XPAR_SLICE_REFUSE + 64;
  CHECK(xpar_geom_choose(&q, &g) == XPAR_GEOM_HUGE,
        "a slice size past the refusal point must be refused");

  q.slice_size = 0;
  q.slice_count = 0;
  q.recovery = 256;
  CHECK(xpar_geom_choose(&q, &g) == XPAR_GEOM_FIELD,
        "R filling GF(2^8) leaves no room for data");

  q.recovery = 2;
  q.slice_count = 4096;
  q.stream_length = 100;
  CHECK(xpar_geom_choose(&q, &g) == XPAR_GEOM_UNREACHABLE,
        "more slices than bytes must be refused");

  accepted = 0;
  Fi(xt_scale(512),
    u64 cap;
    xpar_memset(&q, 0, sizeof q);
    q.field_log2 = (xt_next(&r) & 1) ? 8 : 16;
    q.stream_length = xt_next(&r) % (u64) (32u << 20);
    q.recovery = 1 + xt_below(&r, 32);
    switch (xt_below(&r, 3)) {
      case 0: break;
      case 1: q.slice_size  = (u64) (1 + xt_below(&r, 64)) * 4096;  break;
      default: q.slice_count = 1 + xt_below(&r, 200);  break;
    }
    if (xpar_geom_choose(&q, &g) != XPAR_GEOM_OK) continue;
    accepted++;
    cap = (u64) 1 << q.field_log2;
    if (g.slice_count != xpar_ceil_div(q.stream_length, g.slice_size)) {
      CHECK(false, "S does not tile L: L = %" PRIu64 ", Z = %" PRIu64
            ", S = %" PRIu64, q.stream_length, g.slice_size, g.slice_count);
      break;
    }
    if (g.slice_count + q.recovery > cap) {
      CHECK(false, "S + R = %" PRIu64 " overflows the field",
            g.slice_count + q.recovery);
      break;
    }
    if (g.slice_size % 64) { CHECK(false, "Z off the quantum");  break; }
    if (g.cell_bytes && g.cells_per_slice !=
        (u32) xpar_ceil_div(g.slice_size, g.cell_bytes)) {
      CHECK(false, "K does not follow from Z and Y");  break;
    }
    if (g.cell_bytes > g.slice_size) { CHECK(false, "Y exceeds Z");  break; });
  /*  Ensure the sweep exercises accepted geometries.  */
  CHECK(accepted > xt_scale(512) / 4,
        "the sweep accepted %" PRIu32 " of %" PRIu32 " parameter sets",
        accepted, (u32) xt_scale(512));
}

/* Verify cell mapping and complete, non-overlapping slice coverage. */
static void test_cell_mapping(void) {
  xt_rng r;
  u32 i;

  xt_section_begin("cell mapping");
  xt_seed(&r, 0x5555);

  Fi(xt_scale(256),
    xpar_geom g;
    u64 s, covered = 0;
    u32 c;
    xpar_memset(&g, 0, sizeof g);
    g.slice_size    = (u64) (1 + xt_below(&r, 32)) * 4096;
    g.stream_base   = (u64) xt_below(&r, 4) * g.slice_size;
    g.stream_length = 1 + xt_next(&r) % (16u * g.slice_size);
    g.cell_bytes    = xpar_cell_choose(g.slice_size, 0, 0);
    g.cells_per_slice = g.cell_bytes
                        ? (u32) xpar_ceil_div(g.slice_size, g.cell_bytes) : 1;
    g.slice_count   = xpar_ceil_div(g.stream_length, g.slice_size);

    for (c = 0; c < g.cells_per_slice; c++) covered += xpar_cell_size(&g, c);
    if (covered != g.slice_size) {
      CHECK(false, "cells cover %" PRIu64 " of a %" PRIu64 " byte slice",
            covered, g.slice_size);
      break;
    }
    CHECK_U64(xpar_cell_size(&g, g.cells_per_slice), 0,
              "the column past the last must be empty");

    for (s = 0; s < g.slice_count; s++) {
      u64 begin = xpar_slice_begin(&g, s);
      u64 have  = xpar_slice_bytes(&g, s);
      u64 sum   = 0;
      if (xpar_slice_of(&g, begin) != s) {
        CHECK(false, "slice_of does not invert slice_begin at %" PRIu64, s);
        break;
      }
      if (have && xpar_slice_of(&g, begin + have - 1) != s) {
        CHECK(false, "the last byte of slice %" PRIu64 " lands elsewhere", s);
        break;
      }
      for (c = 0; c < g.cells_per_slice; c++) {
        u64 cb = xpar_cell_begin(&g, s, c);
        u64 cn = xpar_cell_bytes(&g, s, c);
        sum += cn;
        if (cn && xpar_cell_of(&g, cb) != c) {
          CHECK(false, "cell_of does not invert cell_begin at (%" PRIu64
                ", %" PRIu32 ")", s, c);
          break;
        }
        if (cn && xpar_cell_of(&g, cb + cn - 1) != c) {
          CHECK(false, "the last byte of cell (%" PRIu64 ", %" PRIu32
                ") lands elsewhere", s, c);
          break;
        }
      }
      if (sum != have) {
        CHECK(false, "the cells of slice %" PRIu64 " hold %" PRIu64
              " of %" PRIu64 " content bytes", s, sum, have);
        break;
      }
    });
}

static void test_erasures(void) {
  xpar_geom g;
  xpar_erasures e;
  xt_rng r;
  u32 i;

  xt_section_begin("erasures");
  xt_seed(&r, 0x6666);

  xpar_memset(&g, 0, sizeof g);
  g.slice_size      = 65536;
  g.cell_bytes      = 16384;
  g.cells_per_slice = 4;
  g.stream_length   = 65536 * 8;
  g.slice_count     = 8;

  xpar_erasures_init(&e, g.slice_count, g.cells_per_slice);

  xpar_erasures_mark_range(&e, &g, 65536 * 2 + 16384 + 5, 10);
  CHECK_U64(e.bad_count, 1, "a ten byte fault marks one cell");
  CHECK(xpar_cell_bad(&e, 2, 1), "it marks the cell it landed in");

  xpar_erasures_clear(&e);
  xpar_erasures_mark_range(&e, &g, 16384 - 1, 2);
  CHECK_U64(e.bad_count, 2, "a fault across a cell boundary marks two");

  xpar_erasures_clear(&e);
  xpar_erasures_mark_range(&e, &g, 65536 - 1, 2);
  CHECK(xpar_cell_bad(&e, 0, 3) && xpar_cell_bad(&e, 1, 0),
        "a fault across a slice boundary marks both sides");

  xpar_erasures_clear(&e);
  xpar_erasures_mark_range(&e, &g, g.slice_count * g.slice_size, 4096);
  CHECK_U64(e.bad_count, 0, "a fault past L marks nothing");

  xpar_erasures_clear(&e);
  for (u64 s = 0; s < (6); s++) { xpar_cell_mark(&e, s, 0); }
  for (u64 s = 0; s < (3); s++) { xpar_cell_mark(&e, s, 2); }
  CHECK_U64(e.bad_count, 9, "nine cells marked");
  CHECK_U64(xpar_erasures_max_depth(&e), 6, "the deepest column has six");

  xpar_erasures_clear(&e);
  xpar_erasures_mark_slice(&e, 3);
  CHECK_U64(e.bad_count, g.cells_per_slice, "a slice marks K cells");
  CHECK_U64(xpar_erasures_max_depth(&e), 1, "and one erasure per column");

  Fi(xt_scale(64),
    xpar_col_groups cg;
    u32 seen = 0, a, b;
    u64 s;
    xpar_erasures_clear(&e);
    for (s = 0; s < e.slice_count; s++)
      for (a = 0; a < e.cells_per_slice; a++)
        if (xt_below(&r, 4) == 0) xpar_cell_mark(&e, s, a);
    xpar_col_groups_build(&e, &cg);
    for (a = 0; a < cg.group_count; a++) seen += cg.group[a].column_count;
    CHECK_U64(seen, e.cells_per_slice, "the groups partition the columns");
    for (a = 0; a < cg.group_count; a++) {
      for (s = 0; s < e.slice_count; s++) {
        bool bad = xpar_cell_bad(&e, s, cg.group[a].column[0]);
        if (cg.group[a].present[s] == !bad) continue;
        CHECK(false, "group %" PRIu32 " misrecords slice %" PRIu64, a, s);
        break;
      }
      for (b = 1; b < cg.group[a].column_count; b++)
        for (s = 0; s < e.slice_count; s++)
          if (xpar_cell_bad(&e, s, cg.group[a].column[0]) !=
              xpar_cell_bad(&e, s, cg.group[a].column[b])) {
            CHECK(false, "columns %" PRIu32 " and %" PRIu32 " were grouped "
                  "with different patterns",
                  cg.group[a].column[0], cg.group[a].column[b]);
            b = cg.group[a].column_count;
            break;
          }
    }
    for (a = 0; a + 1 < cg.group_count; a++)
      for (b = a + 1; b < cg.group_count; b++) {
        bool same = true;
        for (s = 0; s < e.slice_count; s++)
          if (cg.group[a].present[s] != cg.group[b].present[s]) { same = false;  break; }
        if (same) {
          CHECK(false, "groups %" PRIu32 " and %" PRIu32 " are duplicates",
                a, b);
          b = cg.group_count;  a = cg.group_count;
        }
      }
    xpar_col_groups_free(&cg));

  xpar_erasures_free(&e);
}

/* Build single-extent entries separated by alignment gaps. */
static void build_gapped(xpar_manifest * m, u32 n, u64 len, u64 gap) {
  u64 off = 0;
  u32 i;
  xpar_memset(m, 0, sizeof *m);
  Fi(n,
    xpar_entry * e = xpar_manifest_append(m);
    char name[32];
    xpar_snprintf(name, sizeof name, "f%03" PRIu32 ".bin", i);
    e->name = (char *) xpar_alloc_raw(xpar_strlen(name) + 1);
    xpar_memcpy(e->name, name, xpar_strlen(name) + 1);
    e->name_len = (u32) xpar_strlen(name);
    e->entry_type = XPAR_ENTRY_REGULAR;
    e->length = len;
    e->extent_count = 1;
    e->extents = (xpar_extent *) xpar_alloc_raw(sizeof(xpar_extent));
    e->extents[0].stream_offset = off;
    e->extents[0].length = len;
    off += len + gap);
  m->stream_length = off ? off - gap : 0;
}

static void test_occindex(void) {
  xpar_manifest m;
  xpar_occindex ix;
  xpar_occurrence o;
  xpar_span sp;
  u64 run;
  const u64 len = 1000, gap = 24;

  xt_section_begin("occurrence index");

  build_gapped(&m, 4, len, gap);
  xpar_occindex_build(&m, &ix);

  CHECK(xpar_stream_locate(&ix, 0, &sp), "offset 0 is named");
  CHECK_U64(sp.entry, 0, "by the first entry");
  CHECK(xpar_stream_locate(&ix, len - 1, &sp),
        "the last byte of one is named");
  CHECK(!xpar_stream_locate(&ix, len, &sp), "the first gap byte is not");
  CHECK(!xpar_stream_locate(&ix, len + gap - 1, &sp),
        "nor the last gap byte");
  CHECK(xpar_stream_locate(&ix, len + gap, &sp), "the next entry is");
  CHECK_U64(sp.entry, 1, "and it is the second one");

  /* A gap resumes at the next extent, not the end of the stream. */
  CHECK_U64(xpar_occindex_next(&ix, 0, 4 * (len + gap)), 0,
            "a covered offset resumes at itself");
  CHECK_U64(xpar_occindex_next(&ix, len, 4 * (len + gap)), len + gap,
            "a gap resumes at the next extent");
  CHECK_U64(xpar_occindex_next(&ix, len + 1, 4 * (len + gap)), len + gap,
            "from anywhere inside the gap");
  CHECK_U64(xpar_occindex_next(&ix, len, len + 4), len + 4,
            "a limit inside the gap caps the answer");
  CHECK_U64(xpar_occindex_next(&ix, 3 * (len + gap) + len, 1ull << 40),
            1ull << 40, "past the last extent nothing resumes");
  CHECK_U64(xpar_occindex_next(&ix, 100, 50), 50,
            "an inverted range returns the limit");

  CHECK(xpar_occindex_canonical(&ix, 0, &o, &run),
        "the first byte has a canonical occurrence");
  CHECK_U64(run, len, "which runs to the end of its extent");
  CHECK(!xpar_occindex_canonical(&ix, len + 2, &o, &run),
        "a gap has no canonical occurrence");

  xpar_occindex_free(&ix);
  xpar_manifest_free(&m);

  build_gapped(&m, 3, len, 0);
  xpar_occindex_build(&m, &ix);
  {
    u64 p;
    for (p = 0; p < 3 * len; p += 97)
      if (xpar_occindex_next(&ix, p, 3 * len) != p) {
        CHECK(false, "a packed stream reported a gap at %" PRIu64, p);
        break;
      }
    CHECK_U64(xpar_occindex_next(&ix, 3 * len, 4 * len), 4 * len,
              "the tail past L is a gap to the limit");
  }
  xpar_occindex_free(&ix);
  xpar_manifest_free(&m);
}

static void test_extents(void) {
  xpar_extent * list = NULL;
  u32 count = 0, cap = 0;

  xt_section_begin("extents");

  xpar_extents_append(&list, &count, &cap, 0, 100);
  xpar_extents_append(&list, &count, &cap, 100, 100);
  CHECK_U64(count, 1, "abutting extents coalesce");
  CHECK_U64(list[0].length, 200, "into one of the summed length");

  xpar_extents_append(&list, &count, &cap, 300, 50);
  CHECK_U64(count, 2, "a gap starts a new extent");
  CHECK_U64(list[1].stream_offset, 300, "at the right offset");

  xpar_extents_append(&list, &count, &cap, 350, 1);
  CHECK_U64(count, 2, "and the new one keeps coalescing");
  CHECK_U64(list[1].length, 51, "with the right length");

  /* Backward references are aliases, not adjacent extents. */
  xpar_extents_append(&list, &count, &cap, 0, 100);
  CHECK_U64(count, 3, "a backward reference is its own extent");

  xpar_free(list);
}

/*  Manifest alias, extent, and alignment validation.  */

static xpar_entry * mf_add(xpar_manifest * m, const char * name,
                           u16 type, u64 length) {
  xpar_entry * e = xpar_manifest_append(m);
  e->name = xpar_strdup(name);
  e->name_len = (u32) xpar_strlen(name);
  e->entry_type = type;
  e->length = length;
  return e;
}

static void mf_extent(xpar_entry * e, u64 off, u64 len) {
  e->extents = (xpar_extent *) xpar_realloc(
                 e->extents, (sz) (e->extent_count + 1) * sizeof(xpar_extent));
  e->extents[e->extent_count].stream_offset = off;
  e->extents[e->extent_count].length = len;
  e->extent_count++;
}

static void mf_alias(xpar_entry * a, const xpar_entry * t,
                     const char * target) {
  a->extra = (u8 *) xpar_strdup(target);
  a->extra_len = (u32) xpar_strlen(target);
  a->length = t->length;
  xpar_memcpy(a->content_hash, t->content_hash, 32);
  xpar_memcpy(a->prefix_hash, t->prefix_hash, 16);
  a->mode = t->mode;  a->attrs = t->attrs;
  a->mtime_ns = t->mtime_ns;  a->atime_ns = t->atime_ns;
  a->ctime_ns = t->ctime_ns;  a->btime_ns = t->btime_ns;
}

static xpar_mf_status mf_run(xpar_manifest * m, u8 align, u64 z, u64 len) {
  xpar_mf_limits lim;
  xpar_mf_result res;
  xpar_mf_status s;
  xpar_memset(&lim, 0, sizeof lim);
  lim.stream_length = len;
  lim.slice_size = z;
  lim.align = align;
  lim.posix_record_count = XPAR_ABSENT_U32;
  s = xpar_manifest_validate(m, &lim, &res);
  return s;
}

static void test_manifest_rules(void) {
  xpar_manifest m;

  xt_section_begin("manifest rules");

  /*  Hard-link aliases must share metadata.  */
  { xpar_entry * f, * a;
    xpar_memset(&m, 0, sizeof m);
    f = mf_add(&m, "f", XPAR_ENTRY_REGULAR, 100);
    mf_extent(f, 0, 100);
    f->mtime_ns = 1234;
    a = mf_add(&m, "g", XPAR_ENTRY_HARDLINK, 0);
    mf_alias(a, f, "f");
    CHECK(mf_run(&m, XPAR_ALIGN_PACKED, 64, 100) == XPAR_MF_OK,
          "a faithful alias validates");
    a = &m.entry[1];
    a->mtime_ns = 5678;
    CHECK(mf_run(&m, XPAR_ALIGN_PACKED, 64, 100) == XPAR_MF_LINK_META,
          "an alias with another mtime is malformed");
    a->mtime_ns = 1234;  a->mode = 0700;  m.entry[0].mode = 0644;
    CHECK(mf_run(&m, XPAR_ALIGN_PACKED, 64, 100) == XPAR_MF_LINK_META,
          "an alias with another mode is malformed");
    xpar_manifest_free(&m); }

  /*  Shared extents must remain within canonical ranges.  */
  { xpar_entry * a, * b, * c;
    xpar_memset(&m, 0, sizeof m);
    a = mf_add(&m, "a", XPAR_ENTRY_REGULAR, 100);
    mf_extent(a, 0, 100);
    b = mf_add(&m, "b", XPAR_ENTRY_REGULAR, 64);
    mf_extent(b, 1024, 64);          /*  Canonical past a 1 KiB gap.  */
    c = mf_add(&m, "c", XPAR_ENTRY_REGULAR, 50);
    mf_extent(c, 10, 50);            /*  Inside a's range.  */
    CHECK(mf_run(&m, XPAR_ALIGN_1K, 64, 1088) == XPAR_MF_OK,
          "sharing bytes an earlier extent defined is allowed");
    xpar_free(m.entry[2].extents);
    m.entry[2].extents = NULL;  m.entry[2].extent_count = 0;
    mf_extent(&m.entry[2], 90, 50);  /*  Runs into the padding.  */
    CHECK(mf_run(&m, XPAR_ALIGN_1K, 64, 1088) == XPAR_MF_EXTENT_SHARE,
          "sharing alignment padding is refused");
    xpar_manifest_free(&m); }

  /*  Allow trailing alignment padding, but not plain gaps.  */
  { xpar_entry * a;
    xpar_memset(&m, 0, sizeof m);
    a = mf_add(&m, "a", XPAR_ENTRY_REGULAR, 100);
    mf_extent(a, 0, 100);
    CHECK(mf_run(&m, XPAR_ALIGN_1K, 64, 1024) == XPAR_MF_OK,
          "trailing alignment padding closes the generation");
    CHECK(mf_run(&m, XPAR_ALIGN_1K, 64, 2048) == XPAR_MF_STREAM_GAP,
          "padding past the next boundary is a gap");
    CHECK(mf_run(&m, XPAR_ALIGN_PACKED, 64, 1024) == XPAR_MF_STREAM_GAP,
          "without an alignment mode the same tail is a gap");
    CHECK(mf_run(&m, XPAR_ALIGN_PACKED, 64, 100) == XPAR_MF_OK,
          "an exact end still validates");
    xpar_manifest_free(&m); }
}

static void check_path(const char * name, xpar_path_status want,
                       u32 flags) {
  xpar_path_status got = xpar_path_check(name, (u32) xpar_strlen(name),
                                         flags);
  if (got == want) { xt_checks++;  return; }
  xt_report(false, "path \"%s\": got %s, want %s", name,
            xpar_path_reason(got), xpar_path_reason(want));
}

static void test_paths(void) {
  xt_section_begin("paths");

#if defined(XPAR_DOS) || defined(__MSDOS__)
  char * dos_abs;
  CHECK(xpar_path_same("/dev/c/work/set", "C:/WORK/SET"),
        "DJGPP and DOS drive paths compare equal");
  CHECK(xpar_path_same("/dev/c", "c:/"),
        "DJGPP and DOS drive roots compare equal");
  dos_abs = xpar_path_lex_abs("/dev/c/work/set");
  CHECK(dos_abs && xpar_path_same(dos_abs, "c:/work/set"),
        "lexical absolute paths accept DJGPP drive spelling");
  xpar_free(dos_abs);
  CHECK(xpar_vname_is_index("SAFE.XPA", "safe"),
        "the DOS index name is an index");
  CHECK(!xpar_vname_is_index("SAFE.V00", "safe"),
        "a DOS recovery volume is not an index");
  CHECK(xpar_vname_is_member("SAFE.V00", "safe"),
        "a DOS recovery volume is a set member");
#endif

  check_path("a/b/c.txt", XPAR_PATH_OK, 0);
  check_path("a", XPAR_PATH_OK, 0);
  check_path("", XPAR_PATH_EMPTY, 0);
  check_path("/etc/passwd", XPAR_PATH_ABSOLUTE, 0);
  check_path("a//b", XPAR_PATH_EMPTY_COMPONENT, 0);
  check_path("a/./b", XPAR_PATH_DOT, 0);
  check_path("../a", XPAR_PATH_DOTDOT, 0);
  check_path("a/../b", XPAR_PATH_DOTDOT, 0);
  check_path("a/b/", XPAR_PATH_TRAILING_SLASH, 0);
  check_path("a\tb", XPAR_PATH_CONTROL, 0);

  /* Drive and UNC prefixes are never relative. */
  check_path("C:/x", XPAR_PATH_DRIVE, 0);
  check_path("a:b", XPAR_PATH_DRIVE, 0);
  check_path("\\\\server\\share", XPAR_PATH_UNC, 0);

  /* Other Windows restrictions apply only with XPAR_PATH_WIN. */
  check_path("a?b", XPAR_PATH_OK, 0);
  check_path("a?b", XPAR_PATH_WINCHAR, XPAR_PATH_WIN);
  check_path("CON", XPAR_PATH_OK, 0);
  check_path("CON", XPAR_PATH_DEVICE, XPAR_PATH_WIN);
  check_path("con.txt", XPAR_PATH_DEVICE, XPAR_PATH_WIN);
  check_path("LPT9", XPAR_PATH_DEVICE, XPAR_PATH_WIN);
  check_path("a ", XPAR_PATH_WINTRAIL, XPAR_PATH_WIN);
  check_path("a.", XPAR_PATH_WINTRAIL, XPAR_PATH_WIN);

  {
    xpar_path_status got = xpar_path_check("a\0b", 3, 0);
    CHECK(got == XPAR_PATH_CONTROL, "an embedded NUL must be refused");
  }
}

static void test_packets(void) {
  xpar_buf b;
  xpar_scan sc;
  xpar_pkt hdr;
  const u8 * body;
  u64 off;
  u8 set_id[XPAR_SET_ID_LEN];
  u8 payload[200];
  xt_rng r;
  u32 seen = 0;

  xt_section_begin("packets");
  xt_seed(&r, 0x7777);
  xt_fill(&r, set_id, sizeof set_id);
  xt_fill(&r, payload, sizeof payload);

  xpar_buf_init(&b);
  xpar_pkt_write(&b, "SETD", 0, set_id, payload, 64, NULL);
  xpar_pkt_write(&b, "FILE", 0, set_id, payload + 64, 100, NULL);
  CHECK(b.len % 8 == 0, "packets leave the buffer eight byte aligned");

  xpar_scan_init(&sc, b.data, b.len, NULL, false);
  while (xpar_scan_next(&sc, &hdr, &body, &off)) {
    if (seen == 0) {
      CHECK(xpar_pkt_is(&hdr, "SETD"), "the first packet is SETD");
      CHECK_U64(hdr.length, XPAR_PKT_HDR + 64, "with its body length");
      xt_bytes_equal("SETD body", body, payload, 64);
    } else if (seen == 1) {
      CHECK(xpar_pkt_is(&hdr, "FILE"), "the second packet is FILE");
      xt_bytes_equal("FILE body", body, payload + 64, 100);
    }
    seen++;
  }
  CHECK_U64(seen, 2, "both packets were scanned");

  {
    u32 i, kept = 0;
    Fi(xt_scale(64),
      sz at = (sz) xt_below(&r, (u32) b.len);
      u8 save = b.data[at];
      b.data[at] ^= (u8) (1u << xt_below(&r, 8));
      seen = 0;
      xpar_scan_init(&sc, b.data, b.len, NULL, false);
      while (xpar_scan_next(&sc, &hdr, &body, &off)) seen++;
      if (seen == 2) kept++;
      b.data[at] = save);
    CHECK_U64(kept, 0, "a flipped bit must never leave both packets valid");
  }

  {
    sz full = b.len;
    b.len = full - 8;
    seen = 0;
    xpar_scan_init(&sc, b.data, b.len, NULL, false);
    while (xpar_scan_next(&sc, &hdr, &body, &off)) seen++;
    CHECK_U64(seen, 1, "a truncated packet is not emitted");
    b.len = full;
  }

  /*  Validate type bytes and reject unknown critical types.  */
  {
    xpar_pkt h2;
    xpar_buf_free(&b);
    xpar_buf_init(&b);
    xpar_pkt_write(&b, "xyz9", 0, set_id, payload, 8, NULL);
    CHECK(xpar_pkt_read(b.data, b.len, NULL, &h2) == XPAR_OK,
          "an unknown noncritical type is readable");
    xpar_buf_free(&b);
    xpar_buf_init(&b);
    xpar_pkt_write(&b, "ZZZZ", XPAR_PF_CRITICAL, set_id, payload, 8, NULL);
    CHECK(xpar_pkt_read(b.data, b.len, NULL, &h2) == XPAR_E_UNSUPPORTED,
          "an unknown critical type is refused");
    xpar_scan_init(&sc, b.data, b.len, NULL, false);
    seen = 0;
    while (xpar_scan_next(&sc, &hdr, &body, &off)) seen++;
    CHECK_U64(seen, 0, "the scan does not emit it");
    CHECK_U64(sc.skip_unsupported, 1, "the scan counts it apart");
    xpar_buf_free(&b);
    xpar_buf_init(&b);
    xpar_pkt_write(&b, "SE_D", 0, set_id, payload, 8, NULL);
    CHECK(xpar_pkt_read(b.data, b.len, NULL, &h2) == XPAR_E_MALFORMED,
          "a type byte outside the ASCII alphanumerics is refused");
  }

  /*  Reject reserved packet flags before further parsing.  */
  {
    xpar_pkt h2;
    u32 flags;
    xpar_buf_free(&b);
    xpar_buf_init(&b);
    xpar_pkt_write(&b, "SETD", 0, set_id, payload, 64, NULL);
    CHECK(xpar_pkt_read(b.data, b.len, NULL, &h2) == XPAR_OK,
          "packet reads before tampering");
    flags = xpar_rd32(b.data + 36);
    CHECK_U64(flags & ~(u32) XPAR_PF_KNOWN, 0,
              "writer clears reserved flags");
    xpar_wr32(b.data + 36, flags | (1u << 7));
    CHECK(xpar_pkt_read(b.data, b.len, NULL, &h2) == XPAR_E_MALFORMED,
          "reserved flag rejected");
    xpar_wr32(b.data + 36, flags | (1u << 31));
    CHECK(xpar_pkt_read(b.data, b.len, NULL, &h2) == XPAR_E_MALFORMED,
          "top reserved flag rejected");
    xpar_wr32(b.data + 36, flags);
    CHECK(xpar_pkt_read(b.data, b.len, NULL, &h2) == XPAR_OK,
          "untouched packet still reads");
  }

  xpar_buf_free(&b);
}

/*  VOLH replicas must claim the same volume index.  */
static void volh_scan(xpar_critset * cs, const xpar_buf * b) {
  xpar_scan sc;
  xpar_pkt hdr;
  const u8 * body;
  u64 off;
  xpar_scan_init(&sc, b->data, b->len, NULL, false);
  while (xpar_scan_next(&sc, &hdr, &body, &off))
    xpar_critset_add(cs, &hdr, body);
}

static void test_volume_headers(void) {
  xpar_critset cs;
  xpar_buf b, b2;
  xpar_volh v;
  u8 set_id[XPAR_SET_ID_LEN];
  xt_rng r;

  xt_section_begin("volume headers");
  xt_seed(&r, 0x0176);
  xt_fill(&r, set_id, sizeof set_id);

  /*  Keep packet backing buffers stable after insertion.  */
  xpar_buf_init(&b);
  xpar_buf_init(&b2);
  xpar_memset(&v, 0, sizeof v);
  v.volume_index = XPAR_VOL_STANDALONE;  v.volume_kind = XPAR_VOL_INDEX;
  xpar_volh_write(&b, &v, set_id, NULL);
  v.volume_index = 0;  v.volume_kind = XPAR_VOL_RECOVERY;
  xpar_volh_write(&b, &v, set_id, NULL);
  v.volume_index = 1;
  xpar_volh_write(&b, &v, set_id, NULL);
  /*  Two headers claiming one volume shall be byte-identical.  */
  v.volume_index = 1;  v.volume_kind = XPAR_VOL_DATA;
  xpar_volh_write(&b2, &v, set_id, NULL);

  xpar_critset_init(&cs);
  volh_scan(&cs, &b);
  CHECK_U64(cs.count, 3, "one entry per volume header");
  CHECK_U64(cs.conflicts, 0, "headers of different volumes do not conflict");

  volh_scan(&cs, &b2);
  CHECK_U64(cs.count, 3, "the repeat is not a fourth volume");
  CHECK(cs.conflicts > 0, "two headers for one volume disagree");

  {
    const u8 * borrowed = cs.pkt[0].body;
    u8 first = borrowed[0];
    xpar_critset_detach(&cs, b.data, b.len);
    CHECK(cs.pkt[0].body != borrowed,
          "detaching packet storage copies retained bodies");
    b.data[borrowed - b.data] ^= 0xFF;
    CHECK_U64(cs.pkt[0].body[0], first,
              "detached packet bodies outlive their backing storage");
  }
  xpar_critset_free(&cs);
  xpar_buf_free(&b);
  xpar_buf_free(&b2);
}

static void test_posx_bound(void) {
  u8 set_id[XPAR_SET_ID_LEN];
  xpar_posix_rec * rec;
  xpar_critset c;
  xpar_crit_pkt p;
  xt_rng r;

  xt_section_begin("posx bound");
  xt_seed(&r, 0x5150);
  xt_fill(&r, set_id, sizeof set_id);

  xpar_memset(&c, 0, sizeof c);
  rec = (xpar_posix_rec *) (void *) &c;
  CHECK(xpar_posx_collect(&c, set_id, 1, &rec) == XPAR_E_MALFORMED,
        "positive count without POSX data");
  CHECK(rec == NULL, "no table returned");
  CHECK(xpar_posx_collect(&c, set_id, 0xFFFFFFFFu, &rec) == XPAR_E_MALFORMED,
        "huge count rejected before allocation");
  CHECK(rec == NULL, "no table returned");

  CHECK(xpar_posx_collect(&c, set_id, 0, &rec) == XPAR_OK,
        "empty ownership table loads");

  xpar_memset(&p, 0, sizeof p);
  xpar_memcpy(p.hdr.type, XPAR_T_POSX, 4);
  xpar_memcpy(p.hdr.set_id, set_id, sizeof set_id);
  p.body = set_id;              /*  Bound rejects before parsing.  */
  p.body_len = 32;
  c.pkt = &p;  c.count = 1;
  CHECK(xpar_posx_collect(&c, set_id, 3, &rec) == XPAR_E_MALFORMED,
        "record count exceeds POSX data");
  CHECK(rec == NULL, "no table returned");

  p.hdr.set_id[0] ^= 0xFF;
  CHECK(xpar_posx_collect(&c, set_id, 1, &rec) == XPAR_E_MALFORMED,
        "other-set POSX data is ignored");
  CHECK(rec == NULL, "no table returned");
}

/* Every nonzero field element must have a unique logarithm. */
static void test_gf_tables(void) {
  u8 * seen8;
  u8 * seen16;
  u32 i, dup = 0, miss = 0, gap = 0;

  xt_section_begin("gf tables");

  seen8 = (u8 *) xpar_calloc(256, 1);
  Fi(255,
    u8 v = xpar_gf8_exp[i];
    if (!v) { gap++;  continue; }
    if (seen8[v]) dup++;
    seen8[v] = 1;
    if (xpar_gf8_log[v] != (u8) i) miss++);
  CHECK_U64(dup, 0, "GF(2^8) alpha^i repeats before the group order");
  CHECK_U64(miss, 0, "GF(2^8) log and exp disagree");
  for (i = 1; i < 256; i++) if (!seen8[i]) gap++;
  CHECK_U64(gap, 0, "GF(2^8) alpha does not reach every nonzero element");
  xpar_free(seen8);

  dup = 0;  miss = 0;  gap = 0;
  seen16 = (u8 *) xpar_calloc(65536, 1);
  Fi(65535u,
    u16 v = xpar_gf16_exp[i];
    if (!v) { gap++;  continue; }
    if (seen16[v]) dup++;
    seen16[v] = 1;
    if (xpar_gf16_log[v] != (u16) i) miss++);
  CHECK_U64(dup, 0, "GF(2^16) alpha^i repeats before the group order");
  CHECK_U64(miss, 0, "GF(2^16) log and exp disagree");
  for (i = 1; i < 65536u; i++) if (!seen16[i]) gap++;
  CHECK_U64(gap, 0, "GF(2^16) alpha does not reach every nonzero element");
  xpar_free(seen16);
}

/*  XPAR_O_APPEND has to seek to the end on every host. Win32 grants
    append-only access only when FILE_APPEND_DATA stands alone, so an
    implementation that ORs it into GENERIC_WRITE overwrites from zero
    and scrub --rebuild-cells eats the head of the volume it extends.  */
static void test_append_open(void) {
  static const char * path = "t_append.tmp";
  xpar_file * f;
  u8 got[8];
  sz n = 0;

  xt_section_begin("append open");
  xpar_remove(path);

  f = xpar_open(path, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_TRUNC);
  CHECK(f != NULL, "create the file");
  if (!f) return;
  xpar_xwrite(f, "AAAA", 4);
  xpar_xclose(f);

  f = xpar_open(path, XPAR_O_WRONLY | XPAR_O_APPEND);
  CHECK(f != NULL, "reopen for append");
  if (f) { xpar_xwrite(f, "BBBB", 4);  xpar_xclose(f); }

  f = xpar_open(path, XPAR_O_RDONLY);
  CHECK(f != NULL, "reopen for read");
  if (f) { n = xpar_xread(f, got, sizeof got);  xpar_xclose(f); }
  CHECK_U64(n, 8, "append extends rather than overwrites");
  if (n == 8) xt_bytes_equal("appended file", got, (const u8 *) "AAAABBBB", 8);

  xpar_remove(path);
}

/*  Readers must enforce the format's own bounds, not just the ones the
    body length happens to imply. Each case pairs a rejection with the
    largest conforming value, so a fix cannot pass by refusing both.  */

/*  A minimal generation-0 descriptor: one file, no slices, no stream.  */
static void hd_setd(u8 body[96], u64 slice_size, u32 cell_bytes) {
  xpar_memset(body, 0, 96);
  xpar_wr64(body, slice_size);
  xpar_wr32(body + 24, 1);          /*  file_count  */
  body[28] = 8;                     /*  field_log2  */
  body[30] = 8;                     /*  recovery_axis_log2  */
  xpar_wr32(body + 44, cell_bytes);
}

static void test_reader_bounds(void) {
  u8 setd[96];
  xpar_setd sd;
  xpar_slcr cr;
  xpar_sltg tg;
  xpar_slcl cl;
  xpar_layt lt;
  u8 * b;
  sz n;

  xt_section_begin("reader bounds");

  /*  SETD: a slice holds at most XPAR_CELLS_MAX cells.  */
  hd_setd(setd, XPAR_SLICE_REFUSE, XPAR_CELL_MIN);
  CHECK(xpar_setd_read(setd, sizeof setd, &sd) == XPAR_E_MALFORMED,
        "SETD with ceil(Z/Y) above XPAR_CELLS_MAX");
  xpar_setd_free(&sd);

  { u64 y = XPAR_SLICE_REFUSE / XPAR_CELLS_MAX;
    CHECK(y >= XPAR_CELL_MIN && y <= 0xFFFFFFFFu && y % 64 == 0,
          "the at-cap control needs a representable, legal cell size");
    hd_setd(setd, XPAR_SLICE_REFUSE, (u32) y);
    CHECK(xpar_setd_read(setd, sizeof setd, &sd) == XPAR_OK,
          "SETD with exactly XPAR_CELLS_MAX cells still loads");
    xpar_setd_free(&sd); }

  /*  Reject slice sizes beyond the writer limit.  */
  hd_setd(setd, XPAR_SLICE_REFUSE + 64, 0);
  CHECK(xpar_setd_read(setd, sizeof setd, &sd) == XPAR_E_MALFORMED,
        "SETD with Z above XPAR_SLICE_REFUSE");
  xpar_setd_free(&sd);
  hd_setd(setd, XPAR_SLICE_REFUSE, 0);
  CHECK(xpar_setd_read(setd, sizeof setd, &sd) == XPAR_OK,
        "SETD at exactly XPAR_SLICE_REFUSE still loads");
  xpar_setd_free(&sd);
  hd_setd(setd, XPAR_SLICE_MAX, 0);
  CHECK(xpar_setd_read(setd, sizeof setd, &sd) == XPAR_E_MALFORMED,
        "SETD at the field's arithmetic ceiling is refused");
  xpar_setd_free(&sd);

  /*  SLCR: at most XPAR_TABLE_SPLIT slices per packet.  */
  n = 16 + (XPAR_TABLE_SPLIT + 1) * 4;
  b = (u8 *) xpar_calloc(n, 1);
  xpar_wr64(b + 8, XPAR_TABLE_SPLIT + 1);
  CHECK(xpar_slcr_read(b, n, &cr) == XPAR_E_MALFORMED,
        "SLCR covering more than XPAR_TABLE_SPLIT slices");
  xpar_slcr_free(&cr);
  xpar_wr64(b + 8, XPAR_TABLE_SPLIT);
  CHECK(xpar_slcr_read(b, 16 + XPAR_TABLE_SPLIT * 4, &cr) == XPAR_OK,
        "SLCR at exactly XPAR_TABLE_SPLIT still loads");
  xpar_slcr_free(&cr);
  xpar_free(b);

  /*  SLTG: the same cap, counted in tags.  */
  n = 24 + (XPAR_TABLE_SPLIT + 1) * 8;
  b = (u8 *) xpar_calloc(n, 1);
  xpar_wr64(b + 8, XPAR_TABLE_SPLIT + 1);
  b[16] = 8;
  CHECK(xpar_sltg_read(b, n, &tg) == XPAR_E_MALFORMED,
        "SLTG covering more than XPAR_TABLE_SPLIT slices");
  xpar_sltg_free(&tg);
  xpar_wr64(b + 8, XPAR_TABLE_SPLIT);
  CHECK(xpar_sltg_read(b, 24 + XPAR_TABLE_SPLIT * 8, &tg) == XPAR_OK,
        "SLTG at exactly XPAR_TABLE_SPLIT still loads");
  xpar_sltg_free(&tg);
  xpar_free(b);

  /*  SLCL: n * ceil(Z/Y) is what the cap counts, so two cells per slice
      halve the permitted slice count.  */
  n = 24 + ((XPAR_TABLE_SPLIT / 2) + 1) * 2 * 4;
  b = (u8 *) xpar_calloc(n, 1);
  xpar_wr64(b + 8, (XPAR_TABLE_SPLIT / 2) + 1);
  xpar_wr32(b + 16, 4096);
  CHECK(xpar_slcl_read(b, n, 8192, &cl) == XPAR_E_MALFORMED,
        "SLCL covering more than XPAR_TABLE_SPLIT cells");
  xpar_slcl_free(&cl);
  xpar_wr64(b + 8, XPAR_TABLE_SPLIT / 2);
  CHECK(xpar_slcl_read(b, 24 + (XPAR_TABLE_SPLIT / 2) * 2 * 4, 8192, &cl) ==
          XPAR_OK,
        "SLCL at exactly XPAR_TABLE_SPLIT cells still loads");
  xpar_slcl_free(&cl);
  xpar_free(b);

  /*  LAYT: a volume name is one path component, not a relative path.  */
  { static const char * const nm[2] = { "s.r00.xpa", "sub/s.r00.xpa" };
    sz i;
    Fi(2,
      sz ln = xpar_strlen(nm[i]);
      sz e1 = xpar_align_up(32 + 5, XPAR_PKT_ALIGN);
      sz e2 = xpar_align_up(32 + ln, XPAR_PKT_ALIGN);
      n = 8 + e1 + e2;
      b = (u8 *) xpar_calloc(n, 1);
      xpar_wr32(b, 2);
      xpar_wr32(b + 4, XPAR_VOL_STANDALONE);
      b[8] = XPAR_VOL_INDEX;
      xpar_wr16(b + 10, 5);
      xpar_memcpy(b + 8 + 32, "s.xpa", 5);
      b[8 + e1] = XPAR_VOL_RECOVERY;
      xpar_wr16(b + 8 + e1 + 2, (u16) ln);
      xpar_wr64(b + 8 + e1 + 16, 1);         /*  byte_length  */
      xpar_memcpy(b + 8 + e1 + 32, nm[i], ln);
      if (i == 0)
        CHECK(xpar_layt_read(b, n, &lt) == XPAR_OK,
              "LAYT with a bare volume name loads");
      else
        CHECK(xpar_layt_read(b, n, &lt) == XPAR_E_MALFORMED,
              "LAYT volume name containing a path separator");
      xpar_layt_free(&lt);
      xpar_free(b)); }
}

void xt_run_unit(void) {
  test_helpers();
  test_crc32c();
  test_blake3();
  test_geometry();
  test_cell_mapping();
  test_erasures();
  test_occindex();
  test_extents();
  test_manifest_rules();
  test_paths();
  test_packets();
  test_volume_headers();
  test_posx_bound();
  test_gf_tables();
  test_append_open();
  test_reader_bounds();

}
