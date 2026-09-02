/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.  */

/*  End-to-end recovery geometry and fault profiles.  */

#include "t_system.h"

typedef struct {
  const char * title;
  const char * codec;
  const char * field;
  u64 bytes;
  u64 slice;
  u32 cell;
  u32 recovery;
} recovery_shape;

static int expect(xt_context * c, const char * dir, int want,
                  const char * const * args, const char * what) {
  int got = xt_run_xpar(c, dir, args);
  CHECK(got == want, "%s returned %d, expected %d", what, got, want);
  if (got != want) xt_dump_file(c->err, "child stderr follows");
  return got;
}

static const char * leaf_name(const char * path) {
  const char * leaf = path;
  while (*path) {
    if (*path == '/' || *path == '\\') leaf = path + 1;
    path++;
  }
  return leaf;
}

static bool damage_cell(const char * file, const recovery_shape * shape,
                        u32 slice, u32 column, u64 seed) {
  u64 offset = (u64) slice * shape->slice +
               (u64) column * shape->cell + 31;
  return xt_damage(file, offset, 37, seed);
}

static void check_json_geometry(xt_context * c, u64 cells, u64 depth,
                                const char * status) {
  u64 got;
  CHECK(xt_json_u64(c->out, "cells_bad", &got) && got == cells,
        "JSON reports %" PRIu64 " bad cells", cells);
  CHECK(xt_json_u64(c->out, "column_depth", &got) && got == depth,
        "JSON reports column depth %" PRIu64, depth);
  CHECK(xt_json_string(c->out, "status", status),
        "JSON reports status %s", status);
}

static void restore_file(const char * keep, const char * data,
                         const char * what) {
  CHECK(xt_copy_file(keep, data), "restore data before %s", what);
}

static void exact_capacity(xt_context * c, const char * dir,
                           const recovery_shape * shape,
                           const char * data, const char * keep) {
  static const char * const verify_json[] = {
    "verify", "--json", "SET.xpa", NULL
  };
  static const char * const verify[] = { "verify", "SET.xpa", NULL };
  static const char * const strong[] = {
    "verify", "--strong", "SET.xpa", NULL
  };
  static const char * const repair[] = {
    "repair", "--in-place", "SET.xpa", NULL
  };
  u32 columns = (u32) (shape->slice / shape->cell);
  u32 slice, column;

  restore_file(keep, data, "the exact capacity profile");
  for (column = 0; column < columns; column++)
    for (slice = 0; slice < shape->recovery; slice++)
      CHECK(damage_cell(data, shape, slice, column,
                        0xC100 + column * 17 + slice),
            "damage cell (%" PRIu32 ", %" PRIu32 ")", slice, column);
  expect(c, dir, 1, verify_json, "verify the exact capacity profile");
  check_json_geometry(c, (u64) columns * shape->recovery,
                      shape->recovery, "repairable");
  expect(c, dir, 1, verify, "classify the exact capacity profile");
  expect(c, dir, 0, repair, "repair the exact capacity profile");
  CHECK(xt_files_equal(data, keep), "exact-capacity repair is byte-exact");
  expect(c, dir, 0, strong, "strongly verify exact-capacity repair");
}

static void over_capacity(xt_context * c, const char * dir,
                          const recovery_shape * shape,
                          const char * data, const char * keep) {
  static const char * const verify_json[] = {
    "verify", "--json", "SET.xpa", NULL
  };
  static const char * const verify[] = { "verify", "SET.xpa", NULL };
  static const char * const repair[] = {
    "repair", "--in-place", "SET.xpa", NULL
  };
  u32 slice;
  restore_file(keep, data, "the over-capacity profile");
  for (slice = 0; slice <= shape->recovery; slice++)
    CHECK(damage_cell(data, shape, slice, 0, 0xC200 + slice),
          "damage over-budget cell %" PRIu32, slice);
  expect(c, dir, 2, verify_json, "verify the over-capacity profile");
  check_json_geometry(c, shape->recovery + 1, shape->recovery + 1,
                      "unrepairable");
  expect(c, dir, 2, verify, "classify the over-capacity profile");
  expect(c, dir, 2, repair, "refuse the over-capacity profile");
  CHECK(!xt_files_equal(data, keep),
        "refused repair does not claim to restore the bytes");
}

static void boundary_profiles(xt_context * c, const char * dir,
                              const recovery_shape * shape,
                              const char * data, const char * keep) {
  static const char * const verify_json[] = {
    "verify", "--json", "SET.xpa", NULL
  };
  static const char * const repair[] = {
    "repair", "--in-place", "SET.xpa", NULL
  };
  u32 columns = (u32) (shape->slice / shape->cell);
  u32 slice, column;

  restore_file(keep, data, "whole-slice loss");
  for (slice = 0; slice < shape->recovery; slice++)
    for (column = 0; column < columns; column++)
      CHECK(damage_cell(data, shape, slice, column,
                        0xC300 + slice * 19 + column),
            "damage whole-slice cell");
  expect(c, dir, 1, verify_json, "verify whole-slice loss");
  check_json_geometry(c, (u64) columns * shape->recovery,
                      shape->recovery, "repairable");
  expect(c, dir, 0, repair, "repair whole-slice loss");
  CHECK(xt_files_equal(data, keep), "whole-slice repair is byte-exact");

  restore_file(keep, data, "a cell-boundary burst");
  CHECK(xt_damage(data, shape->cell - 8, 16, 0xC401),
        "damage across a cell boundary");
  expect(c, dir, 1, verify_json, "verify a cell-boundary burst");
  check_json_geometry(c, 2, 1, "repairable");
  expect(c, dir, 0, repair, "repair a cell-boundary burst");
  CHECK(xt_files_equal(data, keep), "cell-boundary repair is byte-exact");

  restore_file(keep, data, "a slice-boundary burst");
  CHECK(xt_damage(data, shape->slice - 8, 16, 0xC402),
        "damage across a slice boundary");
  expect(c, dir, 1, verify_json, "verify a slice-boundary burst");
  check_json_geometry(c, 2, 1, "repairable");
  expect(c, dir, 0, repair, "repair a slice-boundary burst");
  CHECK(xt_files_equal(data, keep), "slice-boundary repair is byte-exact");
}

static void random_profiles(xt_context * c, const char * dir,
                            const recovery_shape * shape,
                            const char * data, const char * keep) {
  static const char * const verify_json[] = {
    "verify", "--json", "SET.xpa", NULL
  };
  static const char * const repair[] = {
    "repair", "--in-place", "SET.xpa", NULL
  };
  xt_rng rng;
  u32 columns = (u32) (shape->slice / shape->cell);
  u32 rounds = xt_scale(5), round;
  xt_seed(&rng, 0xFA017 + (u8) shape->field[0] + (u8) shape->codec[0]);
  for (round = 0; round < rounds; round++) {
    u32 column, cells = 0, depth = 0;
    bool any = false;
    restore_file(keep, data, "a random recovery profile");
    for (column = 0; column < columns; column++) {
      u32 want = xt_below(&rng, shape->recovery + 2), slice;
      if (want > 4) want = 4;
      if (want > depth) depth = want;
      cells += want;
      for (slice = 0; slice < want; slice++) {
        CHECK(damage_cell(data, shape, slice, column, xt_next(&rng)),
              "damage a random profile cell");
        any = true;
      }
    }
    if (!any) {
      static const char * const verify[] = { "verify", "SET.xpa", NULL };
      expect(c, dir, 0, verify, "verify an empty random profile");
      continue;
    }
    expect(c, dir, depth <= shape->recovery ? 1 : 2, verify_json,
           "verify a random recovery profile");
    check_json_geometry(c, cells, depth,
                        depth <= shape->recovery ? "repairable" :
                                                   "unrepairable");
    expect(c, dir, depth <= shape->recovery ? 0 : 2, repair,
           "repair a random recovery profile");
    CHECK(xt_files_equal(data, keep) == (depth <= shape->recovery),
          "random repair agrees with the column bound");
  }
}

static void recovery_volumes(xt_context * c, const char * dir,
                             const recovery_shape * shape,
                             const char * data, const char * keep) {
  static const char * const verify[] = { "verify", "SET.xpa", NULL };
  static const char * const scrub[] = { "scrub", "--deep", "SET.xpa", NULL };
  static const char * const repair[] = {
    "repair", "--in-place", "SET.xpa", NULL
  };
  char volume[XT_PATH_MAX], keepdir[XT_PATH_MAX], saved[XT_PATH_MAX], arg[64];
  const char * recover[] = { "recover", arg, "SET.xpa", NULL };
  u32 slice;
  restore_file(keep, data, "recovery-volume profiles");
#if defined(XPAR_DOS)
  if (!xt_find_file(dir, "SET_.V", "", volume, sizeof volume)) {
#else
  if (!xt_find_file(dir, "SET.V", "", volume, sizeof volume)) {
#endif
    CHECK(false, "find a separate recovery volume");
    return;
  }
  CHECK(xt_path(keepdir, sizeof keepdir, dir, "KEEP") &&
        xt_mkdir(keepdir) &&
        xt_path(saved, sizeof saved, keepdir, "VOL.BIN") &&
        xt_copy_file(volume, saved), "save a recovery volume");
  CHECK(xt_damage(volume, 600, 256, 0xC501), "damage recovery data");
  expect(c, dir, 0, verify, "ignore unused damaged recovery");
  {
    int got = xt_run_xpar(c, dir, scrub);
    CHECK(got == 1 || got == 2,
          "deep scrub detects damaged recovery, got %d", got);
  }
  CHECK(xt_copy_file(saved, volume), "restore the recovery volume");

  CHECK(xpar_remove(volume) == 0, "remove a recovery volume");
  expect(c, dir, 0, verify, "verify with a missing recovery volume");
  xpar_snprintf(arg, sizeof arg, "--volume=%s", leaf_name(volume));
  expect(c, dir, 0, recover, "recover a missing recovery volume");
  CHECK(xt_files_equal(volume, saved),
        "recovered volume is byte-identical to the writer's volume");

  restore_file(keep, data, "damaged data and recovery");
  for (slice = 0; slice < shape->recovery; slice++)
    CHECK(damage_cell(data, shape, slice, 0, 0xC600 + slice),
          "spend one column's recovery budget");
  CHECK(xt_damage(volume, 700, 512, 0xC601),
        "damage recovery before decoding");
  {
    int got = xt_run_xpar(c, dir, repair);
    CHECK(got != 0 || xt_files_equal(data, keep),
          "repair never succeeds with incorrect bytes");
  }
}

static void one_shape(xt_context * c, const recovery_shape * shape,
                      bool volumes) {
  char dir[XT_PATH_MAX], data[XT_PATH_MAX], keep[XT_PATH_MAX];
  char slice[32], cell[32], recovery[32];
  const char * create[] = {
    "create", "--reproducible", "--dedup=none", "--align=none",
    "-s", slice, "--cell", cell, "-r", recovery,
    "--codec", shape->codec, "--field", shape->field,
    "--volumes=equal", "-o", "SET", "DATA.BIN", NULL
  };
  static const char * const info[] = { "info", "SET.xpa", NULL };
  xpar_snprintf(slice, sizeof slice, "%" PRIu64, shape->slice);
  xpar_snprintf(cell, sizeof cell, "%" PRIu32, shape->cell);
  xpar_snprintf(recovery, sizeof recovery, "%" PRIu32, shape->recovery);
  if (!xt_case_begin(c, dir, sizeof dir, shape->title)) {
    CHECK(false, "create the recovery case directory");
    return;
  }
  CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
        xt_path(keep, sizeof keep, dir, "KEEP.BIN") &&
        xt_write_pattern(data, shape->bytes, 0xCE110 + (u8) shape->field[0]) &&
        xt_copy_file(data, keep), "prepare recovery geometry input");
  if (expect(c, dir, 0, create, "create recovery geometry") != 0) return;
  expect(c, dir, 0, info, "read recovery geometry");
  CHECK(xt_file_contains(c->out, "erasure unit") ||
        xt_file_contains(c->out, "per column"),
        "info explains the erasure geometry");
  exact_capacity(c, dir, shape, data, keep);
  over_capacity(c, dir, shape, data, keep);
  boundary_profiles(c, dir, shape, data, keep);
  random_profiles(c, dir, shape, data, keep);
  if (volumes) recovery_volumes(c, dir, shape, data, keep);
}

void xt_run_recovery(xt_context * c) {
  static const recovery_shape shapes[] = {
    { "matrix GF(2^8) recovery profiles",  "matrix", "8",
      65536, 16384, 4096, 2 },
    { "FFT GF(2^8) recovery profiles",     "fft",    "8",
      65536, 16384, 4096, 2 },
    { "matrix GF(2^16) recovery profiles", "matrix", "16",
      60000, 16384, 4096, 2 },
    { "FFT GF(2^16) recovery profiles",    "fft",    "16",
      65536, 16384, 4096, 2 }
  };
  u32 i;
  for (i = 0; i < ARRAY_LEN(shapes); i++) one_shape(c, &shapes[i], i == 0);
}
