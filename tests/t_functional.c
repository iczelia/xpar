/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.  */

#include "t_system.h"

static bool seed_file(xt_context * c, const char * name, u64 size, u64 seed) {
  char path[XT_PATH_MAX];
  return xt_path(path, sizeof path, c->seed, name) &&
         xt_write_pattern(path, size, seed);
}

static bool copy_seed(xt_context * c, const char * dir, const char * name) {
  char from[XT_PATH_MAX], to[XT_PATH_MAX];
  return xt_path(from, sizeof from, c->seed, name) &&
         xt_path(to, sizeof to, dir, name) && xt_copy_file(from, to);
}

static bool seed_tree(xt_context * c) {
  char tree[XT_PATH_MAX], sub[XT_PATH_MAX], path[XT_PATH_MAX], twin[XT_PATH_MAX];
  if (!xt_path(tree, sizeof tree, c->seed, "TREE") || !xt_mkdir(tree) ||
      !xt_path(sub, sizeof sub, tree, "SUB") || !xt_mkdir(sub)) return false;
  if (!xt_path(path, sizeof path, tree, "BIG.BIN") ||
      !xt_write_pattern(path, 131072, 701)) return false;
  if (!xt_path(path, sizeof path, tree, "SMALL.BIN") ||
      !xt_write_pattern(path, 32768, 37)) return false;
  if (!xt_path(path, sizeof path, tree, "BIG.BIN") ||
      !xt_path(twin, sizeof twin, sub, "TWIN.BIN") ||
      !xt_copy_file(path, twin)) return false;
  return xt_path(path, sizeof path, sub, "NOTE.TXT") &&
         xt_write_pattern(path, 127, 0xB07E);
}

static bool copy_seed_tree(xt_context * c, const char * dir,
                           const char * name) {
  char from[XT_PATH_MAX], to[XT_PATH_MAX];
  return xt_path(from, sizeof from, c->seed, "TREE") &&
         xt_path(to, sizeof to, dir, name) && xt_copy_tree(from, to);
}

static int run(xt_context * c, const char * dir, int want,
               const char * const * args, const char * what) {
  int got = xt_run_xpar(c, dir, args);
  CHECK(got == want, "%s returned %d, expected %d", what, got, want);
  if (got != want) xt_dump_file(c->err, "child stderr");
  return got;
}

static void test_basic_repair(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "--dedup=none", "--align=none",
    "-s", "4K", "-r", "4", "--codec=matrix", "-o", "set",
    "DATA.BIN", NULL
  };
  static const char * const verify[] = { "verify", "set.xpa", NULL };
  static const char * const verify_json[] = {
    "verify", "--json", "set.xpa", NULL
  };
  static const char * const repair[] = {
    "repair", "--in-place", "set.xpa", NULL
  };
  static const char * const info[] = { "info", "set.xpa", NULL };
  char dir[XT_PATH_MAX], data[XT_PATH_MAX], keep[XT_PATH_MAX];

  if (!xt_case_begin(c, dir, sizeof dir, "create, verify and repair"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
  CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
        xt_path(keep, sizeof keep, dir, "KEEP.BIN") &&
        xt_copy_file(data, keep), "save the pristine input");
  if (run(c, dir, 0, create, "create") != 0) return;
  run(c, dir, 0, verify, "verify intact data");
  run(c, dir, 0, info, "read set information");
  CHECK(xt_file_contains(c->out, "slices"), "info describes the slices");
  CHECK(xt_damage(data, 8123, 97, 0xD00D), "damage the input");
  run(c, dir, 1, verify_json, "verify damaged data");
  CHECK(xt_file_contains(c->out, "repairable"),
        "JSON reports repairable damage");
  run(c, dir, 0, repair, "repair damaged data");
  CHECK(xt_files_equal(data, keep), "repair restores every byte");
  run(c, dir, 0, verify, "verify repaired data");
}

static void test_extract(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "--dedup=none", "--align=none",
    "-s", "4K", "-r", "100%", "--layout=armoured", "-o", "set",
    "DATA.BIN", NULL
  };
  static const char * const extract[] = {
    "extract", "--to=OUT", "set.xpa", NULL
  };
  char dir[XT_PATH_MAX], data[XT_PATH_MAX], keep[XT_PATH_MAX];
  char outdir[XT_PATH_MAX], restored[XT_PATH_MAX];

  if (!xt_case_begin(c, dir, sizeof dir, "extract missing data"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
  CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
        xt_path(keep, sizeof keep, dir, "KEEP.BIN") &&
        xt_copy_file(data, keep), "save the pristine input");
  if (run(c, dir, 0, create, "create a self-contained set") != 0) return;
  CHECK(xpar_remove(data) == 0, "remove the input");
  run(c, dir, 0, extract, "extract missing data");
  CHECK(xt_path(outdir, sizeof outdir, dir, "OUT") &&
        xt_path(restored, sizeof restored, outdir, "DATA.BIN") &&
        xt_files_equal(restored, keep), "extraction restores every byte");
}

typedef struct {
  const char * name;
  const char * layout;
  const char * codec;
  const char * field;
} layout_case;

static void test_layouts(xt_context * c) {
  static const layout_case cases[] = {
    { "sidecar matrix GF(2^8)", "sidecar", "matrix", "8" },
    { "split FFT GF(2^8)", "split", "fft", "8" },
    { "armoured matrix GF(2^16)", "armoured", "matrix", "16" }
  };
  u32 i;
  Fi(ARRAY_LEN(cases),
    const char * create[] = {
      "create", "--reproducible", "--dedup=none", "--align=none",
      "-s", "4K", "-r", "4", "--layout", cases[i].layout,
      "--codec", cases[i].codec, "--field", cases[i].field,
      "--volumes=3", "-o", "set", "DATA.BIN", NULL
    };
    static const char * const verify[] = { "verify", "set.xpa", NULL };
    static const char * const repair[] = {
      "repair", "--in-place", "set.xpa", NULL
    };
    char dir[XT_PATH_MAX], data[XT_PATH_MAX], keep[XT_PATH_MAX];
    char target[XT_PATH_MAX], clean[XT_PATH_MAX];
    char outdir[XT_PATH_MAX], restored[XT_PATH_MAX];
    if (!xt_case_begin(c, dir, sizeof dir, cases[i].name))
      { CHECK(false, "create the case directory");  continue; }
    CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
    CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
          xt_path(keep, sizeof keep, dir, "KEEP.BIN") &&
          xt_copy_file(data, keep), "save the pristine input");
    if (run(c, dir, 0, create, "create") != 0) continue;
    if (!xpar_strcmp(cases[i].layout, "sidecar")) {
      CHECK(xt_damage(data, 4100, 211, 0xCA5E + i), "damage the input");
      run(c, dir, 1, verify, "verify damaged data");
      run(c, dir, 0, repair, "repair damaged data");
      CHECK(xt_files_equal(data, keep), "the layout repairs every byte");
    } else if (!xpar_strcmp(cases[i].layout, "split")) {
      CHECK(xt_find_file(dir, "set", ".d00", target, sizeof target) &&
            xt_path(clean, sizeof clean, dir, "CLEAN.BIN") &&
            xt_copy_file(target, clean), "save the split data volume");
      CHECK(xt_damage(target, 4100, 211, 0xCA5E + i),
            "damage the split data volume");
      run(c, dir, 1, verify, "verify the damaged volume");
      run(c, dir, 0, repair, "repair the damaged volume");
      CHECK(xt_files_equal(target, clean),
            "split repair reproduces the volume exactly");
    } else {
      static const char * const extract[] = {
        "extract", "--to=OUT", "set.xpa", NULL
      };
      CHECK(xt_find_file(dir, "set", ".xpa", target, sizeof target) &&
            xt_damage(target, 600, 4, 0xCA5E + i),
            "damage the armoured archive");
      run(c, dir, 0, verify, "verify correctable inner damage");
      run(c, dir, 0, extract, "extract through inner correction");
      CHECK(xt_path(outdir, sizeof outdir, dir, "OUT") &&
            xt_path(restored, sizeof restored, outdir, "DATA.BIN") &&
            xt_files_equal(restored, keep),
            "inner correction restores every byte");
    });
}

static void test_reproducible(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "--no-verify-after", "--dedup=none",
    "--align=none", "-s", "4K", "-r", "3", "-o", "same",
    "DATA.BIN", NULL
  };
  char dir[XT_PATH_MAX], a[XT_PATH_MAX], b[XT_PATH_MAX];
  char adata[XT_PATH_MAX], bdata[XT_PATH_MAX];
  char aa[XT_PATH_MAX], ba[XT_PATH_MAX];

  if (!xt_case_begin(c, dir, sizeof dir, "reproducible writer"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(xt_path(a, sizeof a, dir, "A") && xt_mkdir(a) &&
        xt_path(b, sizeof b, dir, "B") && xt_mkdir(b),
        "create matching workspaces");
  CHECK(xt_path(adata, sizeof adata, a, "DATA.BIN") &&
        xt_path(bdata, sizeof bdata, b, "DATA.BIN") &&
        copy_seed(c, a, "DATA.BIN") && copy_seed(c, b, "DATA.BIN"),
        "copy matching inputs");
  (void) adata;  (void) bdata;
  if (run(c, a, 0, create, "first reproducible create") != 0) return;
  if (run(c, b, 0, create, "second reproducible create") != 0) return;
  CHECK(xt_path(aa, sizeof aa, a, "same.xpa") &&
        xt_path(ba, sizeof ba, b, "same.xpa") && xt_files_equal(aa, ba),
        "reproducible output is byte-identical");
}

static void test_statuses(xt_context * c) {
  static const char * const no_verb[] = { NULL };
  static const char * const bad_option[] = { "--definitely-not-an-option", NULL };
  static const char * const absent[] = { "verify", "ABSENT.XPA", NULL };
  char dir[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "documented command statuses"))
    { CHECK(false, "create the case directory");  return; }
  run(c, dir, 4, no_verb, "an absent verb");
  run(c, dir, 4, bad_option, "an unknown option");
  run(c, dir, 3, absent, "an absent set");
}

static const char * path_leaf(const char * path) {
  const char * leaf = path;
  while (*path) { if (*path == '/' || *path == '\\') leaf = path + 1;  path++; }
  return leaf;
}

static void test_tree_workflow(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "-R", "-r", "25%", "--dedup=file",
    "-s", "8K", "-o", "side", "TREE", NULL
  };
  static const char * const verify[] = { "verify", "side.xpa", NULL };
  static const char * const strong[] = {
    "verify", "--strong", "side.xpa", NULL
  };
  static const char * const scrub[] = { "scrub", "--deep", "side.xpa", NULL };
  static const char * const list[] = { "list", "side.xpa", NULL };
  static const char * const list_json[] = {
    "list", "--json", "side.xpa", NULL
  };
  static const char * const info[] = { "info", "side.xpa", NULL };
  static const char * const explain[] = { "explain", "side.xpa", NULL };
  static const char * const repair[] = {
    "repair", "--in-place", "--paranoid", "side.xpa", NULL
  };
  static const char * const add[] = {
    "add", "-R", "-r", "25%", "side.xpa", "TREE", NULL
  };
  static const char * const chain[] = {
    "verify", "--chain", "side.xpa", NULL
  };
  char dir[XT_PATH_MAX], tree[XT_PATH_MAX], keep[XT_PATH_MAX];
  char big[XT_PATH_MAX], small[XT_PATH_MAX], kbig[XT_PATH_MAX], ksmall[XT_PATH_MAX];
  char volume[XT_PATH_MAX], volume_arg[64], late[XT_PATH_MAX];
  const char * recover[4];

  if (!xt_case_begin(c, dir, sizeof dir, "recursive sidecar workflow"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed_tree(c, dir, "TREE") && copy_seed_tree(c, dir, "KEEP"),
        "copy the shared tree seed");
  CHECK(xt_path(tree, sizeof tree, dir, "TREE") &&
        xt_path(keep, sizeof keep, dir, "KEEP") &&
        xt_path(big, sizeof big, tree, "BIG.BIN") &&
        xt_path(small, sizeof small, tree, "SMALL.BIN") &&
        xt_path(kbig, sizeof kbig, keep, "BIG.BIN") &&
        xt_path(ksmall, sizeof ksmall, keep, "SMALL.BIN"),
        "name the tree files");
  if (run(c, dir, 0, create, "create a recursive set") != 0) return;
  run(c, dir, 0, verify, "verify the recursive set");
  run(c, dir, 0, strong, "strongly verify the recursive set");
  run(c, dir, 0, scrub, "deep-scrub the recursive set");
  run(c, dir, 0, list, "list the recursive set");
  CHECK(xt_file_contains_ci(c->out, "BIG.BIN"),
        "list names the large file");
  run(c, dir, 0, list_json, "list the recursive set as JSON");
  CHECK(xt_file_contains(c->out, "entry"), "JSON contains entry records");
  run(c, dir, 0, info, "inspect the recursive set");
  run(c, dir, 0, explain, "explain the recursive set");

  CHECK(xt_damage(big, 60000, 31, 0x7101) &&
        xt_damage(small, 4096, 17, 0x7102), "damage two tree entries");
  run(c, dir, 1, verify, "detect scattered tree damage");
  run(c, dir, 0, repair, "repair scattered tree damage");
  CHECK(xt_files_equal(big, kbig) && xt_files_equal(small, ksmall),
        "scattered repair restores both files");

  CHECK(xt_truncate(small, 12000), "truncate a tree entry");
  run(c, dir, 1, verify, "detect a truncated entry");
  run(c, dir, 0, repair, "rebuild a truncated entry");
  CHECK(xt_files_equal(small, ksmall), "rebuild restores every byte");

  CHECK(xt_find_file(dir, "side.v", "", volume, sizeof volume),
        "find a recovery volume");
  xpar_snprintf(volume_arg, sizeof volume_arg, "--volume=%s",
                path_leaf(volume));
  CHECK(xpar_remove(volume) == 0, "remove one recovery volume");
  run(c, dir, 0, verify, "verify with one missing recovery volume");
  recover[0] = "recover";  recover[1] = volume_arg;
  recover[2] = "side.xpa";  recover[3] = NULL;
  run(c, dir, 0, recover, "recover the missing volume");
  CHECK(xpar_lstat(volume, &(xpar_stat_t) { 0 }) == 0,
        "the recovery volume exists again");

  CHECK(xt_path(late, sizeof late, tree, "LATE.BIN") &&
        xt_write_pattern(late, 8192, 64), "add a late file");
  run(c, dir, 0, add, "add a generation");
  run(c, dir, 0, chain, "verify the whole generation chain");
}

static void test_geometry_and_limits(xt_context * c) {
  static const char * const verify_g1[] = { "verify", "mgeo.xpa", NULL };
  static const char * const verify_g2[] = { "verify", "fgeo.xpa", NULL };
  static const char * const verify_g3[] = { "verify", "tiny.xpa", NULL };
  static const char * const create_g1[] = {
    "create", "--reproducible", "-s", "4K", "--codec=matrix",
    "--field=16", "-r", "30%", "-o", "mgeo", "DATA.BIN", NULL
  };
  static const char * const create_g2[] = {
    "create", "--reproducible", "-b", "16", "--codec=fft",
    "--depth=4", "-r", "30%", "-o", "fgeo", "DATA.BIN", NULL
  };
  static const char * const create_g3[] = {
    "create", "--reproducible", "-r", "50%", "-o", "tiny",
    "EMPTY.BIN", "ONE.BIN", NULL
  };
  static const char * const create_doom[] = {
    "create", "--reproducible", "-s", "4K", "-r", "1", "-o", "doom",
    "DATA.BIN", NULL
  };
  static const char * const verify_doom[] = { "verify", "doom.xpa", NULL };
  char dir[XT_PATH_MAX], data[XT_PATH_MAX], empty[XT_PATH_MAX], one[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "geometry and recovery limits"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN") &&
        xt_path(empty, sizeof empty, dir, "EMPTY.BIN") &&
        xt_write_pattern(empty, 0, 1) &&
        xt_path(one, sizeof one, dir, "ONE.BIN") &&
        xt_write_pattern(one, 1, 2), "prepare geometry inputs");
  run(c, dir, 0, create_g1, "create explicit matrix geometry");
  run(c, dir, 0, verify_g1, "verify explicit matrix geometry");
  run(c, dir, 0, create_g2, "create explicit FFT geometry");
  run(c, dir, 0, verify_g2, "verify explicit FFT geometry");
  run(c, dir, 0, create_g3, "create empty and one-byte entries");
  run(c, dir, 0, verify_g3, "verify empty and one-byte entries");
  if (run(c, dir, 0, create_doom, "create a one-slice budget") != 0) return;
  CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
        xt_damage(data, 100, 17, 0xD001) &&
        xt_damage(data, 4200, 17, 0xD002), "damage two slices");
  run(c, dir, 2, verify_doom, "report damage beyond one recovery slice");
}

static void test_authentication(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "--auth-key=KEY.BIN", "-s", "4K",
    "-r", "3", "-o", "auth", "DATA.BIN", NULL
  };
  static const char * const verify[] = {
    "verify", "--auth-key=KEY.BIN", "auth.xpa", NULL
  };
  static const char * const no_key[] = { "verify", "auth.xpa", NULL };
  static const char * const wrong_key[] = {
    "verify", "--auth-key=BAD.BIN", "auth.xpa", NULL
  };
  static const char * const repair[] = {
    "repair", "--auth-key=KEY.BIN", "--in-place", "auth.xpa", NULL
  };
  char dir[XT_PATH_MAX], key[XT_PATH_MAX], bad[XT_PATH_MAX];
  char data[XT_PATH_MAX], keep[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "authenticated repair"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN") &&
        xt_path(key, sizeof key, dir, "KEY.BIN") &&
        xt_write_pattern(key, 32, 0xA071) &&
        xt_path(bad, sizeof bad, dir, "BAD.BIN") &&
        xt_write_pattern(bad, 32, 0xBAD), "prepare authentication keys");
  CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
        xt_path(keep, sizeof keep, dir, "KEEP.BIN") &&
        xt_copy_file(data, keep), "save authenticated source bytes");
  if (run(c, dir, 0, create, "create an authenticated set") != 0) return;
  run(c, dir, 0, verify, "verify with the correct key");
  run(c, dir, 6, no_key, "refuse a missing authentication key");
  run(c, dir, 6, wrong_key, "refuse a wrong authentication key");
  CHECK(xt_damage(data, 10000, 29, 0xA072), "damage authenticated data");
  run(c, dir, 1, verify, "classify authenticated damage");
  run(c, dir, 0, repair, "repair authenticated damage");
  CHECK(xt_files_equal(data, keep), "authenticated repair is exact");
}

static void test_stdout_safety(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "--layout=armoured", "-s", "4K",
    "-r", "4", "-o", "pack", "DATA.BIN", NULL
  };
  static const char * const extract[] = {
    "extract", "--stdout", "pack.xpa", NULL
  };
  char dir[XT_PATH_MAX], data[XT_PATH_MAX], keep[XT_PATH_MAX];
  char archive[XT_PATH_MAX];
  u64 size = 1;
  u32 i;
  int status;
  if (!xt_case_begin(c, dir, sizeof dir, "verified standard output"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
  CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
        xt_path(keep, sizeof keep, dir, "KEEP.BIN") &&
        xt_copy_file(data, keep), "save the standard-output bytes");
  if (run(c, dir, 0, create, "create an armoured stream") != 0) return;
  CHECK(xpar_remove(data) == 0, "remove the external source");
  run(c, dir, 0, extract, "extract an intact stream to stdout");
  CHECK(xt_files_equal(c->out, keep), "stdout carries the exact stream");
  CHECK(xt_path(archive, sizeof archive, dir, "pack.xpa"),
        "name the armoured stream");
  CHECK(xt_damage(archive, 50000, 16, 0x5701),
        "add correctable inner-code damage");
  run(c, dir, 0, extract, "correct while extracting to stdout");
  CHECK(xt_files_equal(c->out, keep), "corrected stdout is exact");
  Fi(12,
    CHECK(xt_damage(archive, 2048 + (u64) i * 8192, 64, 0x5800 + i),
      "damage armoured slice %" PRIu32, i));
  status = xt_run_xpar(c, dir, extract);
  if (status == 0)
    CHECK(xt_files_equal(c->out, keep),
          "successful stdout output is exact");
  else
    CHECK(xt_file_size(c->out, &size) && size == 0,
          "refused stdout emits no bytes");
}

static void test_journal_spellings(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "-s", "4K", "-r", "300%", "-o", "base",
    "DATA.BIN", NULL
  };
  static const char * const repair_relative[] = {
    "repair", "--in-place", "--keep-journal", "base.xpa", NULL
  };
  static const char * const undo_relative[] = { "undo", "base.xpa", NULL };
  const char * repair_absolute[] = {
    "repair", "--in-place", "--keep-journal", NULL, NULL
  };
  const char * undo_absolute[] = { "undo", NULL, NULL };
  char dir[XT_PATH_MAX], data[XT_PATH_MAX], damaged[XT_PATH_MAX];
  char archive[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "journal path spellings"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
  CHECK(xt_path(data, sizeof data, dir, "DATA.BIN") &&
        xt_path(damaged, sizeof damaged, dir, "DAMAGED.BIN") &&
        xt_path(archive, sizeof archive, dir, "base.xpa"),
        "name the journal files");
  if (run(c, dir, 0, create, "create the journal set") != 0) return;
  CHECK(xt_damage(data, 4096, 64, 0x5E11) &&
        xt_copy_file(data, damaged), "save damaged bytes");

  repair_absolute[3] = archive;
  run(c, dir, 0, repair_absolute, "repair an absolute set path");
  CHECK(!xt_files_equal(data, damaged), "repair changed the damaged bytes");
  run(c, dir, 0, undo_relative, "undo through a relative set path");
  CHECK(xt_files_equal(data, damaged),
        "relative undo restores damaged bytes");

  run(c, dir, 0, repair_relative, "repair through a relative set path");
  undo_absolute[1] = archive;
  run(c, dir, 0, undo_absolute, "undo through an absolute set path");
  CHECK(xt_files_equal(data, damaged),
        "absolute undo restores damaged bytes");
}

static bool same_link(const char * a, const char * b) {
  xpar_stat_t as, bs;
  return xpar_lstat(a, &as) == 0 && xpar_lstat(b, &bs) == 0 &&
         as.dev == bs.dev && as.ino == bs.ino;
}

static void test_hardlink_journal(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "--dedup=none", "-R", "-s", "4K",
    "-r", "30%", "-o", "safe", "TREE", NULL
  };
  static const char * const verify[] = { "verify", "safe.xpa", NULL };
  static const char * const repair[] = {
    "repair", "--in-place", "--keep-journal", "safe.xpa", NULL
  };
  static const char * const undo[] = { "undo", "safe.xpa", NULL };
  char dir[XT_PATH_MAX], tree[XT_PATH_MAX], a[XT_PATH_MAX], b[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "hard-link repair and undo"))
    { CHECK(false, "create the case directory");  return; }
  if (!(xpar_fs_caps(dir) & XPAR_FS_HARDLINK)) return;
  CHECK(xt_path(tree, sizeof tree, dir, "TREE") && xt_mkdir(tree) &&
        xt_path(a, sizeof a, tree, "A.BIN") &&
        xt_path(b, sizeof b, tree, "B.BIN") &&
        xt_write_pattern(a, 40000, 5), "prepare the hard-link tree");
  CHECK(xpar_link(a, b) == 0, "create the hard link");
  if (!same_link(a, b)) { CHECK(false, "created names share an identity");  return; }
  CHECK(true, "created names share an identity");
  if (run(c, dir, 0, create, "protect the hard link") != 0) return;
  CHECK(xpar_remove(b) == 0 && xt_copy_file(a, b),
        "replace one link with an identical copy");
  CHECK(!same_link(a, b), "the replacement starts as a separate file");

  if (!c->target_link_identity) {
    run(c, dir, 0, verify, "accept identity that the target cannot observe");
    CHECK(xt_damage(b, 4096, 64, 0x1A11),
          "damage the independent alias");
    run(c, dir, 1, verify, "detect damage to the independent alias");
    run(c, dir, 0, repair, "repair the independent alias");
    CHECK(!same_link(a, b), "repair does not invent an unsupported link");
    CHECK(xt_files_equal(a, b), "repair restores the alias bytes");
    run(c, dir, 0, undo, "undo the alias repair");
    CHECK(!same_link(a, b), "undo keeps the aliases independent");
    CHECK(!xt_files_equal(a, b), "undo restores the damaged alias bytes");
    return;
  }

  run(c, dir, 1, verify, "detect the replaced hard link");
  run(c, dir, 0, repair, "restore the hard link");
  CHECK(same_link(a, b), "repair restores the shared identity");
  run(c, dir, 0, undo, "undo the relinking");
  CHECK(!same_link(a, b), "undo restores independent identities");
  CHECK(xt_files_equal(a, b), "undo preserves both files' bytes");
}

static void test_nonconforming_volume(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "-r", "20%", "-o", "back",
    "DATA.BIN", NULL
  };
  static const char * const verify[] = { "verify", "back.xpa", NULL };
  static const char * const repair[] = {
    "repair", "--in-place", "back.xpa", NULL
  };
  char dir[XT_PATH_MAX], archive[XT_PATH_MAX], keepdir[XT_PATH_MAX];
  char clean[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "trim a nonconforming volume"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
  if (run(c, dir, 0, create, "create the volume") != 0) return;
  CHECK(xt_path(archive, sizeof archive, dir, "back.xpa") &&
        xt_path(keepdir, sizeof keepdir, dir, "KEEP") && xt_mkdir(keepdir) &&
        xt_path(clean, sizeof clean, keepdir, "BACK.XPA") &&
        xt_copy_file(archive, clean), "save the conforming volume");
  CHECK(xt_truncate(archive, 1000), "truncate the index volume");
  run(c, dir, 1, verify, "report a nonconforming volume");
  CHECK(xt_file_contains(c->err, "nonconforming"),
        "diagnostic names the conformance failure");
  run(c, dir, 0, repair, "restore the nonconforming volume");
  CHECK(xt_files_equal(archive, clean),
        "volume repair reproduces the original bytes");
  run(c, dir, 0, verify, "verify the restored volume");
}

static void test_renamed_volume(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "-r", "20%", "-o", "back",
    "DATA.BIN", NULL
  };
  static const char * const verify[] = { "verify", "back.xpa", NULL };
  static const char * const repair[] = {
    "repair", "--in-place", "back.xpa", NULL
  };
  char dir[XT_PATH_MAX], volume[XT_PATH_MAX], renamed[XT_PATH_MAX];
  char keepdir[XT_PATH_MAX], clean[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "put back a renamed volume"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
  if (run(c, dir, 0, create, "create the volume set") != 0) return;
  CHECK(xt_find_file(dir, "back.v", "", volume, sizeof volume),
        "find a recovery volume");
  CHECK(xt_path(keepdir, sizeof keepdir, dir, "KEEP") && xt_mkdir(keepdir) &&
        xt_path(clean, sizeof clean, keepdir, "VOLUME.BIN") &&
        xt_copy_file(volume, clean) &&
        xt_path(renamed, sizeof renamed, dir, "ZZ") &&
        xpar_rename(volume, renamed) == 0, "rename and save the volume");
  run(c, dir, 1, verify, "find the volume under another name");
  CHECK(xt_file_contains(c->err, "using"),
        "verify reports the substitute name");
  run(c, dir, 0, repair, "put the renamed volume back");
  CHECK(xt_files_equal(volume, clean),
        "restored volume is byte-identical");
  run(c, dir, 0, verify, "verify the restored volume name");
}

static void test_prologue_recovery(xt_context * c) {
  static const char * const create[] = {
    "create", "--reproducible", "--layout=armoured", "-r", "20%",
    "-o", "arch", "DATA.BIN", NULL
  };
  static const char * const verify[] = { "verify", "arch.xpa", NULL };
  static const char * const recover[] = {
    "recover-prologue", "arch.xpa", NULL
  };
  char dir[XT_PATH_MAX], archive[XT_PATH_MAX], clean[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, "recover an armoured prologue"))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_seed(c, dir, "DATA.BIN"), "copy the seeded input");
  if (run(c, dir, 0, create, "create the armoured archive") != 0) return;
  CHECK(xt_path(archive, sizeof archive, dir, "arch.xpa") &&
        xt_path(clean, sizeof clean, dir, "CLEAN.BIN") &&
        xt_copy_file(archive, clean), "save the armoured archive");
  CHECK(xt_damage(archive, 0, 384, 0xA11C), "destroy the prologue");
  run(c, dir, 2, verify, "refuse an archive without its prologue");
  CHECK(xt_file_contains(c->err, "recover-prologue"),
        "the diagnostic gives the recovery command");
  run(c, dir, 0, recover, "recover the prologue");
  CHECK(xt_files_equal(archive, clean),
        "prologue recovery reproduces the archive");
  run(c, dir, 0, verify, "verify the recovered archive");
}

static bool fixture_path(xt_context * c, char * out, sz cap,
                         const char * name) {
  const char * fixtures = xpar_getenv("XPAR_TEST_FIXTURES");
  const char * src = xpar_getenv("srcdir");
  char base[XT_PATH_MAX], format[XT_PATH_MAX];
  if (fixtures && *fixtures) {
    if (fixtures[0] == '/' || fixtures[0] == '\\' ||
        (fixtures[0] && fixtures[1] == ':'))
      return xt_path(out, cap, fixtures, name);
    return xt_path(base, sizeof base, c->cwd, fixtures) &&
           xt_path(out, cap, base, name);
  }
  if (!src || !*src) src = "tests";
  if (src[0] == '/' || src[0] == '\\' || (src[0] && src[1] == ':')) {
    if (xpar_strlen(src) + 1 > sizeof base) return false;
    xpar_memcpy(base, src, xpar_strlen(src) + 1);
  } else if (!xt_path(base, sizeof base, c->cwd, src)) return false;
  return xt_path(format, sizeof format, base, "format") &&
         xt_path(out, cap, format, name);
}

static bool copy_fixture(xt_context * c, const char * dir,
                         const char * name, const char * as) {
  char from[XT_PATH_MAX], to[XT_PATH_MAX];
  return fixture_path(c, from, sizeof from, name) &&
         xt_path(to, sizeof to, dir, as ? as : name) &&
         xt_copy_file(from, to);
}

static void test_format_archive(xt_context * c, const char * file,
                                const char * title, u64 damage_seed) {
  const char * verify[] = { "verify", file, NULL };
  const char * extract[] = { "extract", "--to=OUT", file, NULL };
  char dir[XT_PATH_MAX], archive[XT_PATH_MAX], keep[XT_PATH_MAX];
  char outdir[XT_PATH_MAX], restored[XT_PATH_MAX];
  if (!xt_case_begin(c, dir, sizeof dir, title))
    { CHECK(false, "create the case directory");  return; }
  CHECK(copy_fixture(c, dir, file, NULL), "copy %s", file);
  CHECK(copy_fixture(c, dir, "DATA.BIN", "KEEP.BIN"),
        "copy expected output");
  CHECK(xt_path(archive, sizeof archive, dir, file) &&
        xt_path(keep, sizeof keep, dir, "KEEP.BIN"),
        "name the fixture files");
  run(c, dir, 0, verify, "verify the checked-in archive");
  run(c, dir, 0, extract, "extract the checked-in archive");
  CHECK(xt_path(outdir, sizeof outdir, dir, "OUT") &&
        xt_path(restored, sizeof restored, outdir, "DATA.BIN") &&
        xt_files_equal(restored, keep),
        "archive decodes to expected bytes");
  CHECK(xt_remove_tree(outdir), "clear the extraction directory");
  CHECK(xt_damage(archive, 600, 8, damage_seed),
        "damage the old armoured archive");
  run(c, dir, 0, verify, "correct old inner-code damage");
  CHECK(xt_file_contains(c->err, "1 corrected") ||
        xt_file_contains(c->out, "1 corrected"),
        "old inner code corrects damage");
  run(c, dir, 0, extract, "extract through old inner-code damage");
  CHECK(xt_path(restored, sizeof restored, outdir, "DATA.BIN") &&
        xt_files_equal(restored, keep),
        "inner code restores expected bytes");
}

static void test_format_stability(xt_context * c) {
  test_format_archive(c, "m8.xpa", "stable matrix GF(2^8) archive", 0xA080);
  test_format_archive(c, "a16.xpa", "stable matrix GF(2^16) archive", 0xA160);
  test_format_archive(c, "f16a.xpa", "stable FFT GF(2^16) archive", 0xF160);
}

void xt_run_functional(int argc, char ** argv) {
  xt_context c;
  xt_section_begin("functional harness");
  if (!xt_context_init(&c, argc, argv)) {
    CHECK(false, "initialise the C test workspace");
    return;
  }
  CHECK(seed_file(&c, "DATA.BIN", 65536, 0x512CED),
        "seed the shared random input");
  CHECK(seed_file(&c, "SMALL.BIN", 2048, 0x5A11),
        "seed the shared small input");
  CHECK(seed_tree(&c), "seed the shared directory tree");

  test_basic_repair(&c);
  test_extract(&c);
  test_layouts(&c);
  test_reproducible(&c);
  test_statuses(&c);
  test_tree_workflow(&c);
  test_geometry_and_limits(&c);
  test_authentication(&c);
  test_stdout_safety(&c);
  test_journal_spellings(&c);
  test_hardlink_journal(&c);
  test_nonconforming_volume(&c);
  test_renamed_volume(&c);
  test_prologue_recovery(&c);
  test_format_stability(&c);
  xt_run_recovery(&c);
  xt_context_free(&c);
}
