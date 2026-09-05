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

/*  Generation-chain operations and the differential self-test.  */

#include "ops.h"
#include "chain.h"
#include "auth.h"
#include "vset.h"
#include "volimg.h"
#include "armour.h"
#include "blake3.h"
#include "chunk.h"
#include "codec.h"
#include "container.h"
#include "crc32c.h"
#include "gf.h"
#include "json.h"
#include "manifest.h"
#include "pathname.h"
#include "plan.h"
#include "port-cpu.h"
#include "port-fs.h"
#include "undo.h"
#include "slice.h"
#include "volname.h"

static xpar_file * gen_hout(const xpar_options * o) {
  return o->json ? xpar_stderr : xpar_stdout;
}

static void gen_json_result_field(const xpar_options * o, const char * verb,
                                  const u8 * set_id, u32 generation,
                                  const char * status, int rc,
                                  const char * field, u64 value) {
  xpar_json js;
  if (!o->json) return;
  xpar_json_init(&js, xpar_stdout, true);
  if (set_id) {
    xpar_json_begin(&js, "set");
    xpar_json_hex(&js, "set_id", set_id, XPAR_SET_ID_LEN);
    xpar_json_u64(&js, "generation", generation);
    xpar_json_end(&js);
  }
  xpar_json_begin(&js, "operation");
  xpar_json_str(&js, "verb", verb);
  xpar_json_u64(&js, "generation", generation);
  if (field) xpar_json_u64(&js, field, value);
  xpar_json_end(&js);
  xpar_json_summary(&js, status, rc);
}

static void gen_json_result(const xpar_options * o, const char * verb,
                            const u8 * set_id, u32 generation,
                            const char * status, int rc) {
  gen_json_result_field(o, verb, set_id, generation, status, rc, NULL, 0);
}

static u8 * gen_read_whole(const char * path, sz * out_len, bool fatal) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  i64 n;  u8 * p;
  *out_len = 0;
  if (!f) {
    if (fatal) FATAL_IO("cannot open '%s': %s", path,
                        xpar_strerror(xpar_errno()));
    return NULL;
  }
  n = xpar_size(f);
  if (n < 0 || (u64) n >= (u64) (sz) -1) {
    xpar_close(f);
    if (fatal) FATAL_IO("cannot size '%s'", path);
    return NULL;
  }
  p = xpar_alloc_raw((sz) n + 1);
  if (n) xpar_xread(f, p, (sz) n);
  xpar_xclose(f);
  *out_len = (sz) n;
  return p;
}

/*  Anything at the path, including a dangling symbolic link and a name
    the caller cannot open. Opening for read answered a narrower question:
    a staging or rollback name chosen against it would be handed a name
    that lstat can see, and renaming onto it destroys what is there.  */
static bool gen_exists(const char * path) {
  xpar_stat_t st;
  return xpar_lstat(path, &st) == 0;
}

/*  A volume is never assembled in its final pathname. Besides avoiding a
    truncated volume after a crash, O_EXCL makes a stale or hostile temporary
    name harmless: another suffix is tried rather than followed.  */
static xpar_file * gen_stage_mode(const char * path, int access,
                                  char ** out_tmp) {
  xpar_file * f = NULL;
  char * tmp = NULL;
  u32 i;
  for (i = 0; i < 1000; i++) {
#if defined(XPAR_DOS) || defined(__MSDOS__)
    tmp = xpar_dos_numbered(path, "GST", "TMP", i);
#else
    xpar_asprintf(&tmp, "%s.xpar-tmp-%03" PRIu32, path, i);
#endif
    f = xpar_open(tmp, access | XPAR_O_CREAT | XPAR_O_EXCL);
    if (f) break;
    xpar_free(tmp);  tmp = NULL;
  }
  if (!f) FATAL_IO("cannot create a temporary file beside '%s': %s", path,
                   xpar_strerror(xpar_errno()));
  *out_tmp = tmp;
  return f;
}

static xpar_file * gen_stage_open(const char * path, char ** out_tmp) {
  return gen_stage_mode(path, XPAR_O_WRONLY, out_tmp);
}

static xpar_file * gen_stage_open_rw(const char * path, char ** out_tmp) {
  return gen_stage_mode(path, XPAR_O_RDWR, out_tmp);
}

static char * gen_stage_whole(const char * path, const void * p, sz n) {
  char * tmp;
  xpar_file * f = gen_stage_open(path, &tmp);
  xpar_xwrite(f, p, n);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("cannot flush temporary volume '%s'", tmp);
  xpar_xclose(f);
  return tmp;
}

static void gen_publish_whole(char * tmp, const char * path, bool replace) {
  if (!replace && gen_exists(path)) {
    xpar_remove(tmp);
    xpar_free(tmp);
    FATAL("'%s' exists; use -f to overwrite it", path);
  }
  if (xpar_rename(tmp, path) != 0) {
    int e = xpar_errno();
    xpar_remove(tmp);
    xpar_free(tmp);
    FATAL_IO("cannot publish '%s': %s", path, xpar_strerror(e));
  }
  if (xpar_fsync_dir(path) != 0) {
    xpar_free(tmp);
    FATAL_IO("cannot make the published volume '%s' durable: %s", path,
             xpar_strerror(xpar_errno()));
  }
  xpar_free(tmp);
}

static void gen_write_whole(const char * path, const void * p, sz n,
                            bool replace) {
  char * tmp;
  if (!replace && gen_exists(path))
    FATAL("'%s' exists; use -f to overwrite it", path);
  tmp = gen_stage_whole(path, p, n);
  gen_publish_whole(tmp, path, replace);
}

/*  The directory part of a path and the part after it, both freshly
    allocated. A path with no separator has an empty directory, which
    every caller joins back with the current directory implicitly.  */
/*  The directory part keeps its trailing separator here, unlike
    xpar_path_dir: the two halves must concatenate back to `path` for a
    caller that only wants to substitute the leaf.  */
static void gen_split_path(const char * path, char ** dir, char ** name) {
  const char * leaf = xpar_path_base(path);
  *dir  = xpar_strndup(path, (sz) (leaf - path));
  *name = xpar_strdup(leaf);
}

static char * gen_unused_base(const char * base, const char * label) {
  u32 i;
  for (i = 0; i < 1000; i++) {
    char * candidate, * index;
#if defined(XPAR_DOS) || defined(__MSDOS__)
    (void) label;
    candidate = xpar_dos_numbered(base, "S", "", i);
#else
    xpar_asprintf(&candidate, "%s.%s-%03" PRIu32, base, label, i);
#endif
    index = xpar_vname_index(candidate, 0);
    if (!gen_exists(index)) { xpar_free(index);  return candidate; }
    xpar_free(index);  xpar_free(candidate);
  }
  FATAL_IO("cannot choose a staging name beside '%s'", base);
  return NULL;
}

static char * gen_unused_path(const char * path, const char * label,
                              const char * dos_tag, const char * dos_ext,
                              u32 lane) {
  u32 i;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  static const char digits[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  char tag[6];
  sz tl = xpar_strlen(dos_tag);
  FATAL_UNLESS(tl <= 3 && lane < 36 * 36,
               "internal DOS rollback lane is out of range");
  xpar_memcpy(tag, dos_tag, tl);
  tag[tl] = digits[lane / 36];
  tag[tl + 1] = digits[lane % 36];
  tag[tl + 2] = 0;
#else
  (void) lane;
#endif
  for (i = 0; i < 1000; i++) {
    char * candidate;
    xpar_stat_t st;
#if defined(XPAR_DOS) || defined(__MSDOS__)
    (void) label;
    candidate = xpar_dos_numbered(path, tag, dos_ext, i);
#else
    (void) dos_tag;  (void) dos_ext;
    xpar_asprintf(&candidate, "%s.%s-%03" PRIu32, path, label, i);
#endif
    if (xpar_lstat(candidate, &st) != 0) return candidate;
    xpar_free(candidate);
  }
  return NULL;
}

/*  Maintenance journals make multi-volume publication recoverable.  */

#define XPAR_MAINT_MAGIC   "XPARMNTJ"
#define XPAR_MAINT_END     "XPARMNTN"
#define XPAR_MAINT_VER     1U
#define XPAR_MAINT_HDR     64U
#define XPAR_MAINT_REC     24U
#define XPAR_MAINT_FOOT    24U
#if defined(XPAR_DOS) || defined(__MSDOS__)
#define XPAR_MAINT_EXT     ".XPM"
#else
#define XPAR_MAINT_EXT     ".xparmaint"
#endif
#define XPAR_MAINT_CONSOL  1U
#define XPAR_MAINT_PRUNE   2U
#define XPAR_MAINT_MOVE    1U        /*  from -> to  */
#define XPAR_MAINT_PUBLISH 2U        /*  stage -> final.  */
#define XPAR_MAINT_DISCARD 3U        /*  remove after commit  */
#define XPAR_MAINT_COMMIT  1U        /*  commit point  */
#define XPAR_MAINT_KEEP    2U        /*  stage holds an original  */

typedef struct {
  char * from, * to;
  u32 kind, flags;
} gen_maint_rec;

typedef struct {
  gen_maint_rec * rec;
  u32 count, cap, op;
  char * path;
  bool written;
} gen_maint;

typedef enum {
  GEN_MAINT_ABSENT = 0,
  GEN_MAINT_VALID,
  GEN_MAINT_INVALID,
  GEN_MAINT_IO
} gen_maint_status;

static void gen_maint_free(gen_maint * j) {
  u32 i;
  Fi(j->count, xpar_free(j->rec[i].from);  xpar_free(j->rec[i].to));
  xpar_free(j->rec);  xpar_free(j->path);
  xpar_memset(j, 0, sizeof *j);
}

static void gen_maint_add(gen_maint * j, u32 kind, const char * from,
                          const char * to, u32 flags) {
  gen_maint_rec * r;
  if (j->count == j->cap) {
    j->cap = j->cap ? j->cap * 2 : 16;
    j->rec = xpar_realloc(j->rec, (sz) j->cap * sizeof *j->rec);
  }
  r = &j->rec[j->count++];
  xpar_memset(r, 0, sizeof *r);
  r->kind = kind;  r->flags = flags;
  r->from = xpar_strdup(from);
  r->to   = xpar_strdup(to ? to : "");
}

static bool gen_maint_is_stage(const gen_maint * j, const char * path) {
  u32 i;
  Fi(j->count,
    if (j->rec[i].kind == XPAR_MAINT_PUBLISH &&
        xpar_path_same(j->rec[i].from, path)) return true);
  return false;
}

/*  Only rollback copies not reused as stages prove all moves finished.  */
static bool gen_maint_moves_done(const gen_maint * j) {
  bool any = false;
  u32 i;
  Fi(j->count,
    if (j->rec[i].kind != XPAR_MAINT_MOVE ||
        gen_maint_is_stage(j, j->rec[i].to)) continue;
    any = true;
    if (!gen_exists(j->rec[i].to)) return false);
  return any;
}

static gen_maint_status gen_maint_load(const char * path, gen_maint * j) {
  sz n = 0;
  u8 * b = gen_read_whole(path, &n, false);
  u64 count = 0, i, at = XPAR_MAINT_HDR, avail;
  bool ok = false;
  xpar_memset(j, 0, sizeof *j);
  if (!b) {
    xpar_stat_t st;
    if (xpar_lstat(path, &st) == 0) return GEN_MAINT_IO;
    return xpar_errno_absent(xpar_errno()) ? GEN_MAINT_ABSENT
                                           : GEN_MAINT_IO;
  }
  if ((u64) n < (u64) XPAR_MAINT_HDR + XPAR_MAINT_FOOT ||
      xpar_memcmp(b, XPAR_MAINT_MAGIC, 8) ||
      xpar_rd32(b + 8) != XPAR_MAINT_VER || xpar_rd32(b + 12) ||
      xpar_rd32(b + 56) || xpar_crc32c(0, b, 60) != xpar_rd32(b + 60))
    goto out;
  avail = (u64) n - XPAR_MAINT_FOOT;
  count = xpar_rd64(b + 32);
  { const u8 * foot = b + n - XPAR_MAINT_FOOT;
    if (xpar_memcmp(foot, XPAR_MAINT_END, 8) || xpar_rd64(foot + 8) != count ||
        xpar_rd32(foot + 20) ||
        xpar_crc32c(0, b, (sz) avail) != xpar_rd32(foot + 16))
      goto out; }
  j->op = xpar_rd32(b + 16);
  j->path = xpar_strdup(path);
  Fi(count,
    u32 fl, tl, kind, flags, tail, k;
    if (at > avail || avail - at < XPAR_MAINT_REC) goto out;
    kind  = xpar_rd32(b + at);      flags = xpar_rd32(b + at + 4);
    fl    = xpar_rd32(b + at + 8);  tl    = xpar_rd32(b + at + 12);
    if (xpar_rd32(b + at + 16) ||
        xpar_crc32c(0, b + at, 20) != xpar_rd32(b + at + 20)) goto out;
    at += XPAR_MAINT_REC;
    if (avail - at < (u64) fl + tl || !fl) goto out;
    Fk(fl + tl, if (!b[at + k]) goto out);
    { char * from = xpar_strndup((const char *) b + at, fl);
      char * to   = xpar_strndup((const char *) b + at + fl, tl);
      gen_maint_add(j, kind, from, to, flags);
      xpar_free(from);  xpar_free(to); }
    at += (u64) fl + tl;
    tail = (u32) ((8 - ((XPAR_MAINT_REC + fl + tl) & 7)) & 7);
    if (avail - at < tail) goto out;
    at += tail);
  ok = at == avail;
out:
  xpar_free(b);
  if (!ok) gen_maint_free(j);
  return ok ? GEN_MAINT_VALID : GEN_MAINT_INVALID;
}

static gen_maint_status gen_maint_describe(const char * path,
                                           const char ** op) {
  gen_maint j;
  gen_maint_status st = gen_maint_load(path, &j);
  if (st == GEN_MAINT_VALID) {
    if (op) *op = j.op == XPAR_MAINT_PRUNE ? "prune" : "consolidate";
    gen_maint_free(&j);
  } else if (st != GEN_MAINT_ABSENT && op)
    *op = "maintenance operation with a damaged journal";
  return st;
}

/*  Commit at the final publish.  */
static void gen_maint_commit_point(gen_maint * j) {
  u32 i = j->count;
  while (i--) if (j->rec[i].kind == XPAR_MAINT_PUBLISH)
    { j->rec[i].flags |= XPAR_MAINT_COMMIT;  return; }
}

/*  Persist the plan before moving files.  */
static bool gen_maint_write(gen_maint * j, const char * base) {
  xpar_file * f;
  u8 hdr[XPAR_MAINT_HDR], rec[XPAR_MAINT_REC], foot[XPAR_MAINT_FOOT], pad[8];
  u32 all, i;
  u64 bytes = 0;
  xpar_memset(pad, 0, sizeof pad);
  gen_maint_commit_point(j);
  Fi(j->count, bytes += xpar_strlen(j->rec[i].from) + xpar_strlen(j->rec[i].to));
  j->path = xpar_vname_maint(base);
  if (gen_exists(j->path)) {
    xpar_fprintf(xpar_stderr, "xpar: journal '%s' is pending%s; run "
                 "'xpar repair'\n", j->path,
                 gen_maint_describe(j->path, NULL) == GEN_MAINT_VALID
                   ? "" : " but cannot be validated");
    return false;
  }
  xpar_memset(hdr, 0, sizeof hdr);
  xpar_memcpy(hdr, XPAR_MAINT_MAGIC, 8);
  xpar_wr32(hdr + 8, XPAR_MAINT_VER);
  xpar_wr32(hdr + 16, j->op);
  xpar_wr64(hdr + 32, j->count);
  xpar_wr64(hdr + 40, bytes);
  xpar_wr64(hdr + 48, (u64) xpar_wall_ns());
  xpar_wr32(hdr + 60, xpar_crc32c(0, hdr, 60));
  f = xpar_open(j->path, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_EXCL |
                         XPAR_O_NOFOLLOW | XPAR_O_PRIVATE);
  if (!f) {
    xpar_fprintf(xpar_stderr, "xpar: cannot write journal '%s': %s\n",
                 j->path, xpar_strerror(xpar_errno()));
    return false;
  }
  xpar_xwrite(f, hdr, sizeof hdr);
  all = xpar_crc32c(0, hdr, sizeof hdr);
  Fi(j->count,
    u32 fl = (u32) xpar_strlen(j->rec[i].from);
    u32 tl = (u32) xpar_strlen(j->rec[i].to);
    u32 tail = (u32) ((8 - ((XPAR_MAINT_REC + fl + tl) & 7)) & 7);
    xpar_memset(rec, 0, sizeof rec);
    xpar_wr32(rec, j->rec[i].kind);      xpar_wr32(rec + 4, j->rec[i].flags);
    xpar_wr32(rec + 8, fl);              xpar_wr32(rec + 12, tl);
    xpar_wr32(rec + 20, xpar_crc32c(0, rec, 20));
    xpar_xwrite(f, rec, sizeof rec);
    xpar_xwrite(f, j->rec[i].from, fl);
    xpar_xwrite(f, j->rec[i].to, tl);
    if (tail) xpar_xwrite(f, pad, tail);
    all = xpar_crc32c(all, rec, sizeof rec);
    all = xpar_crc32c(all, j->rec[i].from, fl);
    all = xpar_crc32c(all, j->rec[i].to, tl);
    if (tail) all = xpar_crc32c(all, pad, tail));
  /*  The footer CRC marks a complete journal.  */
  xpar_memset(foot, 0, sizeof foot);
  xpar_memcpy(foot, XPAR_MAINT_END, 8);
  xpar_wr64(foot + 8, j->count);
  xpar_wr32(foot + 16, all);
  xpar_xwrite(f, foot, sizeof foot);
  if (xpar_fsync(f) != 0) {
    int err = xpar_errno();
    xpar_close(f);  xpar_remove(j->path);
    xpar_fprintf(xpar_stderr, "xpar: cannot flush journal '%s': %s\n",
                 j->path, xpar_strerror(err));
    return false;
  }
  xpar_xclose(f);
  if (xpar_fsync_dir(j->path) != 0) {
    int err = xpar_errno();
    xpar_remove(j->path);
    xpar_fprintf(xpar_stderr, "xpar: cannot persist journal '%s': %s\n",
                 j->path, xpar_strerror(err));
    return false;
  }
  j->written = true;
  return true;
}

static void gen_maint_done(gen_maint * j) {
  if (j->written) (void) xpar_journal_drop(j->path);
  gen_maint_free(j);
}

static void gen_maint_stuck(const char * verb, const char * from,
                            const char * to) {
  if (to) xpar_fprintf(xpar_stderr, "xpar: cannot %s '%s' to '%s': %s\n",
                       verb, from, to, xpar_strerror(xpar_errno()));
  else xpar_fprintf(xpar_stderr, "xpar: cannot %s '%s': %s\n", verb, from,
                    xpar_strerror(xpar_errno()));
}

/*  Finish a committed operation; otherwise restore the original files.  */
int xpar_maint_recover(const char * path, bool quiet) {
  gen_maint j;
  gen_maint_status status;
  u32 i;
  bool committed = false, stuck = false;
  const char * op;

  status = gen_maint_load(path, &j);
  if (status != GEN_MAINT_VALID) {
    if (!quiet)
      xpar_fprintf(xpar_stderr,
                   "xpar: cannot validate pending maintenance journal "
                   "'%s'; it was kept\n", path);
    return status == GEN_MAINT_IO ? XPAR_EXIT_IO : XPAR_EXIT_UNREPAIRABLE;
  }
  op = j.op == XPAR_MAINT_PRUNE ? "prune" : "consolidate";
  Fi(j.count,
    if ((j.rec[i].flags & XPAR_MAINT_COMMIT) && gen_exists(j.rec[i].to) &&
        !gen_exists(j.rec[i].from))
      committed = true);
  if (!quiet)
    xpar_fprintf(xpar_stderr, "xpar: %s interrupted %s from '%s'\n",
                 committed ? "completing" : "rolling back", op, path);

  if (committed) {
    Fi(j.count,
      gen_maint_rec * r = &j.rec[i];
      if (r->kind != XPAR_MAINT_PUBLISH || gen_exists(r->to)) continue;
      if (!gen_exists(r->from)) {
        xpar_fprintf(xpar_stderr, "xpar: missing both '%s' and '%s'\n",
                     r->from, r->to);
        stuck = true;  continue;
      }
      if (xpar_rename(r->from, r->to) != 0)
        { gen_maint_stuck("publish", r->from, r->to);  stuck = true; });
    if (!stuck && xpar_fsync_dir(path) != 0)
      xpar_fprintf(xpar_stderr, "xpar: warning: cannot sync maintenance "
                   "directory: %s\n", xpar_strerror(xpar_errno()));
    for (i = 0; !stuck && i < j.count; i++) {
      gen_maint_rec * r = &j.rec[i];
      const char * gone = r->kind == XPAR_MAINT_MOVE ? r->to
                        : r->kind == XPAR_MAINT_DISCARD ? r->from : NULL;
      if (!gone || !gen_exists(gone)) continue;
      if (r->kind == XPAR_MAINT_MOVE && gen_maint_is_stage(&j, gone)) continue;
      if (xpar_remove(gone) != 0) { gen_maint_stuck("remove", gone, NULL);  stuck = true; }
    }
  } else {
    /*  Until all moves finish, canonical names still hold originals.  */
    bool moved = gen_maint_moves_done(&j);
    for (i = j.count; i-- > 0;) {
      gen_maint_rec * r = &j.rec[i];
      if (r->kind != XPAR_MAINT_PUBLISH || !moved || !gen_exists(r->to))
        continue;
      if (r->flags & XPAR_MAINT_KEEP) {
        if (!gen_exists(r->from) && xpar_rename(r->to, r->from) != 0)
          { gen_maint_stuck("move", r->to, r->from);  stuck = true; }
      } else if (xpar_remove(r->to) != 0)
        { gen_maint_stuck("remove", r->to, NULL);  stuck = true; }
    }
    for (i = j.count; !stuck && i-- > 0;) {
      gen_maint_rec * r = &j.rec[i];
      if (r->kind != XPAR_MAINT_MOVE || !gen_exists(r->to)) continue;
      if (gen_exists(r->from)) {
        xpar_fprintf(xpar_stderr, "xpar: cannot restore '%s': '%s' exists\n",
                     r->to, r->from);
        stuck = true;  continue;
      }
      if (xpar_rename(r->to, r->from) != 0)
        { gen_maint_stuck("restore", r->to, r->from);  stuck = true; }
    }
    /*  Discard stages last so failed rollbacks remain recognizable.  */
    for (i = 0; !stuck && i < j.count; i++) {
      gen_maint_rec * r = &j.rec[i];
      if (r->kind != XPAR_MAINT_PUBLISH || (r->flags & XPAR_MAINT_KEEP))
        continue;
      if (gen_exists(r->from) && xpar_remove(r->from) != 0)
        { gen_maint_stuck("remove", r->from, NULL);  stuck = true; }
    }
  }
  if (xpar_fsync_dir(path) != 0)
    xpar_fprintf(xpar_stderr, "xpar: warning: cannot sync maintenance "
                 "directory: %s\n", xpar_strerror(xpar_errno()));
  if (stuck) {
    xpar_fprintf(xpar_stderr, "xpar: kept journal '%s' for retry\n", path);
    gen_maint_free(&j);
    return XPAR_EXIT_IO;
  }
  if (!xpar_journal_drop(path)) { gen_maint_free(&j);  return XPAR_EXIT_IO; }
  gen_maint_free(&j);
  return XPAR_EXIT_OK;
}

char * xpar_maint_pending(const char * arg, const char ** op) {
  char * cand[2];
  char * found = NULL;
  u32 n = 0, i;
  sz al = xpar_strlen(arg);
  xpar_dir * d = xpar_opendir(arg);
  if (d) {
    const xpar_dirent * e;
    while (!found && (e = xpar_readdir(d)) != NULL) {
      char * p;
      if (e->is_dir || !xpar_path_ends_with(e->name, XPAR_MAINT_EXT)) continue;
      p = xpar_path_join(arg, e->name);
      if (gen_maint_describe(p, op) != GEN_MAINT_ABSENT) found = p;
      else xpar_free(p);
    }
    xpar_closedir(d);
    return found;
  }
  cand[n++] = xpar_vname_maint(arg);
  if (al > XPAR_EXT_LEN && xpar_vname_has_ext(arg)) {
    char * stem = xpar_strndup(arg, al - XPAR_EXT_LEN);
    cand[n++] = xpar_vname_maint(stem);
    xpar_free(stem);
  }
  Fi(n,
    if (!found && gen_maint_describe(cand[i], op) != GEN_MAINT_ABSENT)
      found = cand[i];
    else xpar_free(cand[i]));
  return found;
}

/*  The cache is regenerable, so failure to publish it does not invalidate
    an already durable set. It is nevertheless staged before encoding so
    the dedup index does not remain resident alongside the codec plan.  */
static bool gen_publish_cache(char * stage, const char * final) {
  if (xpar_rename(stage, final) != 0) return false;
  return xpar_fsync_dir(final) == 0;
}

/*  The armoured prologue.  */

#define ARM_PLAIN_LEN  96
#define ARM_COPY_LEN   128    /*  96 plaintext plus 32 RS(255,223) parity.  */
#define ARM_HDR_LEN    384    /*  Three copies.  */

static bool arm_checksum_ok(const u8 * p) {
  xpar_blake3_t h;  u8 want[8];
  if (xpar_memcmp(p, "XPAR2ARM", 8)) return false;
  xpar_blake3_init(&h);
  xpar_blake3_update(&h, "xpar2 armour prologue v1", 24);
  xpar_blake3_update(&h, p, 88);
  xpar_blake3_final(&h, want, 8);
  return xpar_memcmp(want, p + 88, 8) == 0;
}

static void arm_prologue_decode(const u8 * p, xpar_arm_prologue * o) {
  o->symbol_bits     = p[10];
  o->poly            = xpar_rd32(p + 12);
  o->n               = xpar_rd32(p + 16);
  o->k               = xpar_rd32(p + 20);
  o->fcr             = xpar_rd32(p + 24);
  o->prim            = xpar_rd32(p + 28);
  o->depth           = xpar_rd64(p + 32);
  o->plain_length    = xpar_rd64(p + 40);
  o->armoured_length = xpar_rd64(p + 48);
  o->stream_offset   = xpar_rd64(p + 56);
  o->stream_length   = xpar_rd64(p + 64);
}

static void arm_params_of(const xpar_arm_prologue *, xpar_armour_params *);

static bool arm_prologue_valid(const u8 * p, sz len,
                               xpar_arm_prologue * out) {
  xpar_armour_params ap;
  u32 i;
  if (p[8] != XPAR_FORMAT_MAJOR ||
      (p[10] != 8 && p[10] != 16) || p[11])
    return false;
  Fi(16, if (p[72 + i]) return false);
  arm_prologue_decode(p, out);
  arm_params_of(out, &ap);
  if (xpar_armour_check(&ap)) return false;
  /*  Validation limits symbol_bits to 8 or 16.  */
  if (out->armoured_length != xpar_armg_length(
                                  (u8) ap.symbol_bits, ap.n, ap.k,
                                  ap.depth, out->plain_length))
    return false;
  /*  The outer code handles a short tail; a long tail is junk.  */
  (void) len;
  if (!out->armoured_length) return false;
  if (out->stream_offset > out->plain_length ||
      out->stream_length > out->plain_length - out->stream_offset)
    return false;
  return true;
}

/*  Recognize whole-file armour from any valid prologue copy.  */
bool xpar_garm_is_archive(const u8 * file, sz len) {
  u32 j;
  if (len < ARM_HDR_LEN) return false;
  Fj(3,
    if (!xpar_memcmp(file + (sz) j * ARM_COPY_LEN, "XPAR2ARM", 8))
      return true);
  return false;
}

/*  How many of the three stored prologue copies still check out.  */
u32 xpar_garm_prologue_copies(const u8 * file, sz len) {
  u32 n = 0;
  u32 j;
  if (len < ARM_HDR_LEN) return 0;
  Fj(3, if (arm_checksum_ok(file + (sz) j * ARM_COPY_LEN)) n++);
  return n;
}

/*  Recover the prologue from stored copies, corrected copies, then byte
    majority. The first stage with an agreed result wins.  */

static bool arm_agreed(const u8 * const * copy, int n, int * first) {
  int j;
  *first = -1;
  Fj(n,
    if (!arm_checksum_ok(copy[j])) continue;
    if (*first < 0) *first = j;
    else if (xpar_memcmp(copy[*first], copy[j], ARM_PLAIN_LEN) != 0)
      return false);
  return true;
}

bool xpar_garm_prologue(const u8 * file, sz len, xpar_arm_prologue * out,
                        int * which) {
  u8 corrected[3][ARM_PLAIN_LEN], vote[ARM_PLAIN_LEN];
  const u8 * stored[3], * fixed[3];
  int j, first;

  if (len < ARM_HDR_LEN) return false;
  Fj(3, stored[j] = file + (sz) j * ARM_COPY_LEN);

  if (!arm_agreed(stored, 3, &first)) return false;
  if (first >= 0) {
    if (!arm_prologue_valid(stored[first], len, out)) return false;
    if (which) *which = first;
    return true;
  }

  /*  Reinsert the implicit zero data symbols before RS(255,223) decoding;
      attempt this only after checksum failure.  */
  { xpar_armour_params ap;
    xpar_armour * a;
    xpar_gf_init();
    xpar_armour_defaults(&ap, 8);
    ap.n = 255;  ap.k = 223;  ap.depth = 1;
    a = xpar_armour_new(&ap);
    if (!a) return false;
    Fj(3,
      u8 frame[255];
      xpar_memset(frame, 0, sizeof frame);
      xpar_memcpy(frame, stored[j], ARM_PLAIN_LEN);
      xpar_memcpy(frame + ap.k, stored[j] + ARM_PLAIN_LEN, 32);
      if (xpar_armour_decode_frame(a, frame, NULL) != XPAR_ARMOUR_FAILED)
        xpar_memcpy(corrected[j], frame, ARM_PLAIN_LEN);
      else
        xpar_memcpy(corrected[j], stored[j], ARM_PLAIN_LEN);
      fixed[j] = corrected[j]);
    xpar_armour_free(a);
  }

  if (!arm_agreed(fixed, 3, &first)) return false;
  if (first >= 0) {
    if (!arm_prologue_valid(fixed[first], len, out)) return false;
    if (which) *which = first;
    return true;
  }

  /*  A position where all three differ has no majority, and the format
      says the procedure fails there rather than picking one.  */
  Fj(ARM_PLAIN_LEN,
    u8 a = corrected[0][j], b = corrected[1][j], c = corrected[2][j];
    if (a == b || a == c)   vote[j] = a;
    else if (b == c)        vote[j] = b;
    else                    return false);
  if (arm_checksum_ok(vote) && arm_prologue_valid(vote, len, out)) {
    if (which) *which = 3;
    return true;
  }
  return false;
}

static void arm_prologue_encode(u8 * p, const xpar_arm_prologue * o) {
  xpar_blake3_t h;
  xpar_memset(p, 0, ARM_PLAIN_LEN);
  xpar_memcpy(p, "XPAR2ARM", 8);
  p[8] = XPAR_FORMAT_MAJOR;  p[9] = XPAR_FORMAT_MINOR;
  p[10] = o->symbol_bits;
  xpar_wr32(p + 12, o->poly);
  xpar_wr32(p + 16, o->n);
  xpar_wr32(p + 20, o->k);
  xpar_wr32(p + 24, o->fcr);
  xpar_wr32(p + 28, o->prim);
  xpar_wr64(p + 32, o->depth);
  xpar_wr64(p + 40, o->plain_length);
  xpar_wr64(p + 48, o->armoured_length);
  xpar_wr64(p + 56, o->stream_offset);
  xpar_wr64(p + 64, o->stream_length);
  xpar_blake3_init(&h);
  xpar_blake3_update(&h, "xpar2 armour prologue v1", 24);
  xpar_blake3_update(&h, p, 88);
  xpar_blake3_final(&h, p + 88, 8);
}

static void arm_params_of(const xpar_arm_prologue * pr,
                          xpar_armour_params * p) {
  p->symbol_bits = pr->symbol_bits;  p->poly = pr->poly;
  p->n           = pr->n;            p->k    = pr->k;
  p->fcr         = pr->fcr;          p->prim = pr->prim;
  p->depth       = pr->depth;
}

static bool chain_arm_check(const void * key, const u8 * plain, u64 len) {
  return xpar_verify_packets_ok(plain, len, (const xpar_key *) key);
}

static u8 * arm_extract(const xpar_armour_params * p, const u8 * region,
                        u64 region_len, u64 plain_len, sz * out_len,
                        const xpar_key * key) {
  xpar_armour * a;  u8 * plain;
  const char * why = xpar_armour_check(p);
  *out_len = 0;
  if (why) return NULL;
  a = xpar_armour_new(p);
  if (!a) return NULL;
  if (xpar_armour_size(a, plain_len) > region_len ||
      plain_len >= (u64) (sz) -1) {
    xpar_armour_free(a);  return NULL;
  }
  plain = xpar_alloc_raw((sz) plain_len + 1);
  xpar_armour_extract(a, plain, plain_len, region);
  if (!xpar_verify_packets_ok(plain, plain_len, key)) {
    u64 encoded = xpar_armour_size(a, plain_len);
    u8 * copy = xpar_alloc_raw((sz) encoded);
    xpar_armour_status st;
    xpar_memcpy(copy, region, (sz) encoded);
    st = xpar_armour_decode(a, copy, encoded, plain, plain_len,
                            chain_arm_check, key, NULL);
    xpar_free(copy);
    if (st == XPAR_ARMOUR_FAILED) { xpar_free(plain);  xpar_armour_free(a);  return NULL; }
  }
  xpar_armour_free(a);
  *out_len = (sz) plain_len;
  return plain;
}

/*  Loading a chain.  */

static void chain_blob(xpar_chain * c, u8 * p) {
  c->blob = xpar_realloc(c->blob,
                         (sz) (c->blob_count + 1) * sizeof *c->blob);
  c->blob[c->blob_count++] = p;
}

/*  Scan one packet buffer, unwrapping at most one ARMG nesting level.  */
static void chain_scan(xpar_chain * c, xpar_chain_vol * v, const u8 * buf,
                       u64 len, bool nested) {
  xpar_scan sc;  xpar_pkt hdr;  const u8 * body;  u64 off;
  xpar_scan_init(&sc, buf, len, c->key_loaded ? &c->key : NULL, false);
  sc.accept_unverified_keyed = !c->key_loaded;
  while (xpar_scan_next(&sc, &hdr, &body, &off)) {
    u64 blen = hdr.length - XPAR_PKT_HDR;
    if (!c->key_loaded && (hdr.flags & XPAR_PF_KEYED)) {
      xpar_auth a;
      if (xpar_pkt_is(&hdr, XPAR_T_AUTH) &&
          xpar_auth_read(body, (sz) blen, &a) == XPAR_OK)
        FATAL_CODE(XPAR_EXIT_AUTH,
                   "this set is authenticated; supply --auth-key=FILE");
      continue;
    }
    if (c->key_loaded && !(hdr.flags & XPAR_PF_KEYED))
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "a keyed operation encountered an unkeyed packet");
    if (!xpar_pkt_is(&hdr, XPAR_T_VOLH) && v->has_volh &&
        xpar_memcmp(hdr.set_id, v->set_id, XPAR_SET_ID_LEN))
      FATAL_FORMAT("a volume contains a packet owned by another generation");
    if (xpar_pkt_is(&hdr, XPAR_T_ARMG)) continue;
    if (xpar_pkt_is(&hdr, XPAR_T_VOLH)) {
      xpar_volh vh;
      if (xpar_volh_read(body, (sz) blen, &vh) == XPAR_OK) {
        v->has_volh     = true;
        v->volume_index = vh.volume_index;
        v->volume_kind  = vh.volume_kind;
        xpar_memcpy(v->set_id, hdr.set_id, XPAR_SET_ID_LEN);
        c->crit.rank = vh.volume_kind == XPAR_VOL_INDEX ? 2 : 1;
      }
    }
    if (xpar_pkt_is(&hdr, XPAR_T_RCVS)) {
      xpar_rcvs rc;
      u64 e;
      /*  Validate structure before advertising recovery.  */
      if (xpar_rcvs_read(body, (sz) blen, 0, &rc) != XPAR_OK) continue;
      e = rc.exponent;
      if (!v->recovery_count || e < v->recovery_first) v->recovery_first = e;
      v->recovery_count++;
    }
    if (xpar_pkt_is(&hdr, XPAR_T_LAYT) &&
        !v->layt_body) { v->layt_body = body;  v->layt_len = (sz) blen; }
    if (xpar_pkt_is(&hdr, XPAR_T_VOLH)) continue;
    { u32 stale = c->crit.stale;
      xpar_critset_add(&c->crit, &hdr, body);
      if (c->crit.stale != stale) v->stale_packets++; }
  }
  xpar_reject_unknown_critical(&sc);
  /*  Authentication failures are not format errors.  */
  if (c->key_loaded) c->auth_failed += sc.skip_checksum + sc.skip_keyed;
  if (!nested) {
    u64 pos = 0, blen = 0;
    while (xpar_verify_next_armg(buf, len,
                                 c->key_loaded ? &c->key : NULL,
                                 &pos, &body, &blen)) {
      xpar_armg ag;
      xpar_armour_params ap;
      u8 * plain;
      sz plen;
      char wt[4];
      if (xpar_armg_read(body, (sz) blen, &ag) != XPAR_OK) continue;
      ap.symbol_bits = ag.symbol_bits;  ap.poly = ag.poly;
      ap.n = ag.n;  ap.k = ag.k;  ap.fcr = ag.fcr;  ap.prim = ag.prim;
      ap.depth = ag.depth;
      v->armg_disk  += blen + XPAR_PKT_HDR;
      v->armg_plain += ag.plain_length;
      /*  Read the wrapped type from the frame prefix.  */
      if (xpar_armg_wrapped_type(body, (sz) blen, wt)) {
        if (!xpar_memcmp(wt, XPAR_T_RCVS, 4))
          { v->wrap_rcvs = true;  v->wrap_rcvs_ap = ap; }
        else if (!xpar_memcmp(wt, XPAR_T_SLTG, 4) ||
                 !xpar_memcmp(wt, XPAR_T_SLCL, 4))
          { v->wrap_tab = true;  v->wrap_tab_ap = ap; }
      }
      plain = arm_extract(&ap, ag.data, ag.armoured_length, ag.plain_length,
                          &plen, c->key_loaded ? &c->key : NULL);
      if (!plain) continue;
      chain_blob(c, plain);
      v->armoured_crit = true;
      chain_scan(c, v, plain, plen, true);
    }
  }
}

static void chain_add_vol(xpar_chain * c, char * path) {
  xpar_chain_vol * v;  u8 * data;  sz len;  u32 i;

  Fi(c->vol_count,
    if (xpar_path_same(c->vol[i].path, path)) { xpar_free(path);  return; });

  data = gen_read_whole(path, &len, true);

  c->vol = xpar_realloc(c->vol,
                        (sz) (c->vol_count + 1) * sizeof *c->vol);
  v = &c->vol[c->vol_count++];
  xpar_memset(v, 0, sizeof *v);
  v->path = path;  v->data = data;  v->len = len;
  v->gen = XPAR_GEN_NONE;  v->volume_index = XPAR_VOL_STANDALONE;

  if (len >= ARM_HDR_LEN) {
    xpar_arm_prologue pr;
    int copy = -1;
    if (xpar_garm_prologue(data, len, &pr, &copy)) {
      v->armoured_file = true;  v->scan_first = true;
      return;
    }
  }
  { xpar_pkt h;
    xpar_volh vh;
    if (len >= XPAR_PKT_HDR && !xpar_memcmp(data, XPAR_PKT_MAGIC, 8) &&
        xpar_pkt_read(data, len, c->key_loaded ? &c->key : NULL, &h) ==
          XPAR_OK &&
        xpar_pkt_is(&h, XPAR_T_VOLH) &&
        xpar_volh_read(data + XPAR_PKT_HDR, (sz) (h.length - XPAR_PKT_HDR),
                       &vh) == XPAR_OK)
      v->scan_first = vh.volume_kind == XPAR_VOL_INDEX; }
}

static void chain_scan_vol(xpar_chain * c, xpar_chain_vol * v) {
  if (v->armoured_file) {
    xpar_arm_prologue pr;  xpar_armour_params ap;  u8 * plain;  sz plen;
    int copy = -1;
    if (!xpar_garm_prologue(v->data, v->len, &pr, &copy)) return;
    arm_params_of(&pr, &ap);
    plain = arm_extract(&ap, v->data + ARM_HDR_LEN,
                        (u64) v->len - ARM_HDR_LEN, pr.plain_length, &plen,
                        c->key_loaded ? &c->key : NULL);
    if (!plain) {
      xpar_fprintf(xpar_stderr,
                   "xpar: '%s': uncorrectable armoured-region damage\n",
                   v->path);
      return;
    }
    chain_blob(c, plain);
    chain_scan(c, v, plain, plen, false);
    return;
  }
  chain_scan(c, v, v->data, v->len, false);
}

/*  Scan index volumes first.  */
static void chain_scan_all(xpar_chain * c) {
  u32 i, pass;
  for (pass = 0; pass < 2; pass++)
    Fi(c->vol_count,
      if ((pass == 0) != c->vol[i].scan_first) continue;
      c->crit.rank = c->vol[i].scan_first ? 2 : 1;
      chain_scan_vol(c, &c->vol[i]));
}

static void chain_strip_gen(char * stem) {
  sz n = xpar_strlen(stem);
  if (n < 5) return;
  if (stem[n - 5] != '.' || stem[n - 4] != 'g') return;
  if (stem[n - 3] < '0' || stem[n - 3] > '9') return;
  if (stem[n - 2] < '0' || stem[n - 2] > '9') return;
  if (stem[n - 1] < '0' || stem[n - 1] > '9') return;
  stem[n - 5] = 0;
}

/*  Volumes of one set named `stem`, from one shelf.  */
static void chain_gather_dir(const xpar_options * o, xpar_chain * c,
                             const char * dir, const char * stem) {
  xpar_dir * d = xpar_opendir(*dir ? dir : ".");
  const xpar_dirent * e;
  if (!d) {
    int err = xpar_errno();
    if (!xpar_errno_absent(err))
      FATAL_IO("cannot list '%s': %s", *dir ? dir : ".",
               xpar_strerror(err));
    return;
  }
  while ((e = xpar_readdir(d)) != NULL)
    if (!e->is_dir &&
        (o->chain_metadata_only
           ? xpar_vname_is_index(e->name, stem)
           : xpar_vname_is_member(e->name, stem)))
      chain_add_vol(c, xpar_path_join(dir, e->name));
  xpar_closedir(d);
}

static void chain_gather(const xpar_options * o, xpar_chain * c) {
  u32 i;
  Fi(o->set_ref.count,
    chain_add_vol(c, xpar_strdup(o->set_ref.vol[i])));

  if (o->set_ref.base) {
    char * dir;  char * stem;
    gen_split_path(o->set_ref.base, &dir, &stem);
    chain_strip_gen(stem);
    c->base = xpar_path_join(dir, stem);
    c->dir  = xpar_strdup(dir);
    chain_gather_dir(o, c, dir, stem);
    /*  Gather matching volumes from --scan.  */
    if (o->scan_dir) chain_gather_dir(o, c, o->scan_dir, stem);
    xpar_free(dir);  xpar_free(stem);
  } else if (o->set_ref.dir) c->dir = xpar_strdup(o->set_ref.dir);

  /*  A directory-only reference has no stem to filter by.  */
  if (o->scan_dir && !o->set_ref.base) {
    xpar_dir * d = xpar_opendir(o->scan_dir);
    if (d) {
      const xpar_dirent * e;
      while ((e = xpar_readdir(d)) != NULL)
        if (!e->is_dir && xpar_vname_has_ext(e->name))
          chain_add_vol(c, xpar_path_join(o->scan_dir, e->name));
      xpar_closedir(d);
    }
  }
  chain_scan_all(c);
}

/*  Report authentication failures before derivative format errors.  */
static void chain_auth_or_die(const xpar_chain * c) {
  if (!c->key_loaded || !c->auth_failed) return;
  FATAL_CODE(XPAR_EXIT_AUTH,
             "%" PRIu64 " packet%s failed authentication; wrong key or "
             "tampered data",
             c->auth_failed, c->auth_failed == 1 ? "" : "s");
}

static void chain_link(xpar_chain * c) {
  u32 i, j, n = 0, heads = 0;
  Fi(c->crit.count, if (xpar_pkt_is(&c->crit.pkt[i].hdr, XPAR_T_SETD)) n++);
  if (!n) {
    chain_auth_or_die(c);
    FATAL_FORMAT("no set descriptor found; this is not an xpar 2 set");
  }

  c->gen = xpar_calloc(n, sizeof *c->gen);
  Fi(c->crit.count,
    const xpar_crit_pkt * p = &c->crit.pkt[i];
    xpar_chain_gen * g;
    xpar_status st;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_SETD)) continue;
    g = &c->gen[c->gen_count];
    st = xpar_setd_read(p->body, (sz) p->body_len, &g->sd);
    if (st != XPAR_OK && st != XPAR_E_UNSUPPORTED) {
      xpar_fprintf(xpar_stderr,
                   "xpar: a set descriptor is unreadable (%s); ignored\n",
                   xpar_status_str(st));
      continue;
    }
    if (st == XPAR_E_UNSUPPORTED)
      xpar_fprintf(xpar_stderr, "xpar: generation %" PRIu32
                   " requires features this build does not implement "
                   "(0x%08" PRIx32 ")\n", g->sd.generation,
                   g->sd.required_features);
    xpar_memcpy(g->set_id, p->hdr.set_id, XPAR_SET_ID_LEN);
    g->parent = XPAR_GEN_NONE;
    c->gen_count++);
  if (!c->gen_count) { chain_auth_or_die(c);  FATAL_FORMAT("every set descriptor is malformed"); }

  /*  Insertion sort by generation number: a chain is short and the order
      is what every later walk assumes.  */
  for (i = 1; i < c->gen_count; i++) {
    xpar_chain_gen t = c->gen[i];
    j = i;
    while (j && c->gen[j - 1].sd.generation > t.sd.generation) { c->gen[j] = c->gen[j - 1];  j--; }
    c->gen[j] = t;
  }

  Fi(c->gen_count,
    xpar_chain_gen * g = &c->gen[i];
    if (!g->sd.generation) continue;
    Fj(c->gen_count,
      if (j != i && !xpar_memcmp(g->sd.parent_set_id, c->gen[j].set_id,
         XPAR_SET_ID_LEN)) { g->parent = j;  break; });
    if (g->parent == XPAR_GEN_NONE) g->parent_missing = true;
    else {
      xpar_status st = xpar_setd_check_parent(&g->sd,
                                              c->gen[g->parent].set_id,
                                              &c->gen[g->parent].sd);
      if (st != XPAR_OK)
        FATAL_FORMAT("generation %" PRIu32 " does not follow generation %"
                     PRIu32 ": %s",
                     g->sd.generation, c->gen[g->parent].sd.generation,
                     xpar_status_str(st));
    });

  c->head = XPAR_GEN_NONE;
  Fi(c->gen_count,
    bool named = false;
    Fj(c->gen_count,
      if (c->gen[j].parent == i)
        { if (named) c->forked = true;  named = true; });
    if (!named) { c->head = i;  heads++; });
  if (heads > 1) c->forked = true;
  if (c->head == XPAR_GEN_NONE) c->head = c->gen_count - 1;
}

static void chain_map_volumes(xpar_chain * c) {
  u32 i, j;
  Fi(c->vol_count,
    if (!c->vol[i].has_volh) continue;
    Fj(c->gen_count,
      if (!xpar_memcmp(c->vol[i].set_id, c->gen[j].set_id, XPAR_SET_ID_LEN)) {
        c->vol[i].gen = j;
        c->gen[j].vol_count++;
        if (!c->gen[j].layt_body && c->vol[i].layt_body) {
          c->gen[j].layt_body = c->vol[i].layt_body;
          c->gen[j].layt_len  = c->vol[i].layt_len;
        }
        break;
      }));
  Fj(c->gen_count,
    const xpar_crit_pkt * p =
      xpar_critset_find(&c->crit, c->gen[j].set_id, XPAR_T_LAYT, 0);
    if (p) { c->gen[j].layt_body = p->body;
             c->gen[j].layt_len = (sz) p->body_len; });
  Fi(c->crit.count,
    const xpar_crit_pkt * p = &c->crit.pkt[i];
    if (!xpar_pkt_is(&p->hdr, XPAR_T_RCVS)) continue;
    Fj(c->gen_count,
      if (!xpar_memcmp(p->hdr.set_id, c->gen[j].set_id, XPAR_SET_ID_LEN)) {
        xpar_rcvs rc;
        u64 e;
        /*  Validate against the generation's slice size.  */
        if (xpar_rcvs_read(p->body, (sz) p->body_len,
                           c->gen[j].sd.slice_size, &rc) != XPAR_OK) break;
        e = rc.exponent;
        c->gen[j].recovery_count++;
        if (e + 1 > c->gen[j].recovery_top) c->gen[j].recovery_top = e + 1;
        break;
      }););
}

/*  Count generations named by volumes but lacking descriptors.  */
u32 xpar_gen_unreadable(const xpar_setref * ref, const u32 * have,
                        u32 have_count, char * const * read, u32 read_count,
                        u32 * first) {
  const char * stem;
  u32 i, j, lost = 0;
  u32 * seen = NULL, seen_count = 0;
  *first = 0;
  if (!ref || !ref->base) return 0;
  stem = xpar_path_base(ref->base);
  Fi(ref->count,
    i64 g = xpar_vname_gen_of(xpar_path_base(ref->vol[i]), stem);
    bool known = false;
    if (g < 0) continue;
    /*  A volume that parsed holds the generation it says it does.  */
    for (j = 0; j < read_count && !known; j++)
      if (!xpar_strcmp(read[j], ref->vol[i])) known = true;
    if (known) continue;
    for (j = 0; j < have_count && !known; j++)
      if (have[j] == (u32) g) known = true;
    for (j = 0; j < seen_count && !known; j++)
      if (seen[j] == (u32) g) known = true;
    if (known) continue;
    seen = xpar_realloc(seen, (sz) (seen_count + 1) * sizeof *seen);
    seen[seen_count++] = (u32) g;
    if (!lost || (u32) g < *first) *first = (u32) g;
    lost++);
  xpar_free(seen);
  return lost;
}

void xpar_gchain_load(const xpar_options * o, xpar_chain * c) {
  u32 j;
  xpar_memset(c, 0, sizeof *c);
  xpar_critset_init(&c->crit);
  if (o->auth_key) {
    xpar_keyfile_load_or_die(o->auth_key, &c->key, c->master);
    c->key_loaded = true;
  }
  chain_gather(o, c);
  if (!c->vol_count) FATAL_FORMAT("no readable volume of this set");
  chain_link(c);
  chain_map_volumes(c);
  /*  Derive a writable base from the oldest index when input was a
      directory.  */
  if (!c->base) {
    Fj(c->vol_count,
      char * stem;  char * dir;
      if (c->vol[j].volume_kind != XPAR_VOL_INDEX) continue;
      if (!xpar_vname_has_ext(c->vol[j].path)) continue;
      gen_split_path(c->vol[j].path, &dir, &stem);
      stem[xpar_strlen(stem) - XPAR_EXT_LEN] = 0;
      chain_strip_gen(stem);
      c->base = xpar_path_join(dir, stem);
      if (!c->dir) c->dir = xpar_strdup(dir);
      xpar_free(dir);  xpar_free(stem);
      break);
  }
  if (c->crit.conflicts)
    FATAL_FORMAT("replicated packets verify but disagree");
  Fj(c->crit.count,
    const xpar_crit_pkt * p = &c->crit.pkt[j];
    xpar_auth a;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_AUTH) ||
        xpar_auth_read(p->body, (sz) p->body_len, &a) != XPAR_OK) continue;
    if (c->key_loaded && !xpar_auth_key_ok(&a, c->master))
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "the authentication key is wrong for this set");
    c->authenticated = true;
    c->auth_only = !a.unkeyed_retained);
  if (c->key_loaded && !c->authenticated)
    FATAL_CODE(XPAR_EXIT_AUTH, "this set is not authenticated");
  { u32 * have = xpar_calloc(c->gen_count ? c->gen_count : 1, sizeof *have);
    char ** read = xpar_calloc(c->vol_count ? c->vol_count : 1, sizeof *read);
    u32 nread = 0;
    Fj(c->gen_count, have[j] = c->gen[j].sd.generation);
    Fj(c->vol_count, if (c->vol[j].gen != XPAR_GEN_NONE) read[nread++] = c->vol[j].path);
    c->lost_count = xpar_gen_unreadable(&o->set_ref, have, c->gen_count,
                                        read, nread, &c->lost_first);
    xpar_free(have);  xpar_free(read);
  }
  if (c->lost_count && !o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: generation %" PRIu32 " is unreadable: no descriptor "
                 "survives%s\n", c->lost_first,
                 c->lost_count > 1 ? "; others are also unreadable" : "");
}

void xpar_gchain_free(xpar_chain * c) {
  u32 i;
  Fi(c->vol_count,
    xpar_free(c->vol[i].path);  xpar_free(c->vol[i].data));
  Fi(c->blob_count, xpar_free(c->blob[i]));
  Fi(c->gen_count, xpar_setd_free(&c->gen[i].sd));
  xpar_free(c->vol);  xpar_free(c->blob);  xpar_free(c->gen);
  xpar_free(c->base);  xpar_free(c->dir);
  xpar_critset_free(&c->crit);
  xpar_key_forget(&c->key, c->master);
  xpar_memset(c, 0, sizeof *c);
}

static const xpar_key * gen_chain_key(const xpar_chain * c) {
  return c->key_loaded ? &c->key : NULL;
}

/*  Refuse rewrites that would discard an unreadable generation.  */
static void gen_refuse_unreadable(const xpar_chain * c, const char * verb) {
  if (!c->lost_count) return;
  FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
             "generation %" PRIu32 " is unreadable; repair or recover it "
             "before %s", c->lost_first, verb);
}

static void gen_require_write_key(const xpar_chain * c, const char * verb) {
  if (c->authenticated && !c->key_loaded)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "%s requires --auth-key=FILE for this set", verb);
}

/*  Existing chains inherit their layout. Only consolidate may change it.  */
static int gen_chain_layout(const xpar_options * o, const xpar_chain * c,
                            u32 head, bool extending) {
  int chain = (int) c->gen[head].sd.layout;
  if (!o->layout_given) return chain;
  if (extending && o->layout != chain)
    FATAL("layout is '%s'; use consolidate to change it",
          xpar_layout_name((u8) chain));
  return o->layout;
}

/*  Inherit field and tag strength unless explicitly overridden.  */
static void gen_chain_integrity(const xpar_options * o, const xpar_chain * c,
                                u32 head, xpar_options * eff) {
  const xpar_setd * sd = &c->gen[head].sd;
  if (o->field == XPAR_CLI_AUTO)
    eff->field = sd->field_log2 == 8 ? 8 : 16;
  if (!o->slice_tag_given) eff->slice_tag = sd->slice_tag_len;
}

/*  Print prefixes accepted by --generation.  */
static void gen_list_branches(const xpar_chain * c) {
  u32 i, j;
  Fi(c->gen_count,
    char id[XPAR_SET_ID_LEN * 2 + 1];
    bool named = false;
    Fj(c->gen_count, if (c->gen[j].parent == i) { named = true;  break; });
    if (named) continue;
    xpar_hex(id, c->gen[i].set_id, XPAR_SET_ID_LEN);
    xpar_fprintf(xpar_stderr,
                 "xpar:   branch head generation %" PRIu32 ": --generation=%s\n",
                 c->gen[i].sd.generation, id));
}

u32 xpar_gchain_select(const xpar_chain * c, const xpar_genref * g) {
  u32 i, found = XPAR_GEN_NONE, matches = 0;
  if (!g) {
    if (c->forked) {
      gen_list_branches(c);
      FATAL("this chain has forked; select a branch with --generation");
    }
    return c->head;
  }
  if (g->by_id) {
    Fi(c->gen_count,
      if (xpar_hex_prefix(c->gen[i].set_id, XPAR_SET_ID_LEN, g->id_prefix))
        { found = i;  matches++; });
    if (matches > 1)
      FATAL("set-id prefix '%s' is ambiguous; add digits", g->id_prefix);
    if (matches == 1) return found;
    FATAL("no generation has set-id prefix '%s'", g->id_prefix);
  }
  Fi(c->gen_count,
    if (c->gen[i].sd.generation == (u32) g->number)
      { found = i;  matches++; });
  if (matches > 1)
    FATAL("generation %" PRIu64
          " is ambiguous across fork branches; select it "
          "by set-id prefix", g->number);
  if (matches == 1) return found;
  FATAL("this set has no generation %" PRIu64, g->number);
  return 0;
}

void xpar_gchain_genref(const xpar_chain * c, u32 g, xpar_genref * ref,
                        char text[XPAR_SET_ID_LEN * 2 + 1]) {
  xpar_memset(ref, 0, sizeof *ref);
  xpar_hex(text, c->gen[g].set_id, XPAR_SET_ID_LEN);
  ref->by_id = true;
  ref->id_prefix = text;
}

/*  Resolve the nearest FILE owner, rechecking the full file_id after the
    collector's eight-byte discriminator.  */
static const xpar_crit_pkt * chain_file_pkt(const xpar_chain * c, u32 g,
                                            const u8 * file_id, u32 * owner) {
  u32 h = g;
  for (;;) {
    const xpar_crit_pkt * p = xpar_critset_find_file(
                                &c->crit, c->gen[h].set_id, file_id);
    if (p && p->body_len >= XPAR_SET_ID_LEN &&
        !xpar_memcmp(p->body, file_id, XPAR_SET_ID_LEN)) {
      *owner = h;  return p;
    }
    /*  Every FILE packet is indexed, so a miss is definitive.  */
    if (c->gen[h].parent == XPAR_GEN_NONE) return NULL;
    h = c->gen[h].parent;
  }
}

void xpar_gchain_manifest(const xpar_chain * c, u32 g, xpar_manifest * m,
                          u32 ** owner) {
  const xpar_setd * sd = &c->gen[g].sd;
  u32 i, * own;

  xpar_memset(m, 0, sizeof *m);
  own = xpar_calloc(sd->file_count ? sd->file_count : 1, sizeof *own);
  Fi(sd->file_count,
    u32 h = XPAR_GEN_NONE;
    const xpar_crit_pkt * p = chain_file_pkt(c, g, sd->file_id[i], &h);
    xpar_entry * e;
    xpar_status st;
    if (!p) {
      chain_auth_or_die(c);
      FATAL_FORMAT("generation %" PRIu32
                   " names a manifest entry no generation "
                   "owns; the chain is incomplete", sd->generation);
    }
    e  = xpar_manifest_append(m);
    st = xpar_entry_read(p->body, (sz) p->body_len,
                         c->gen[h].sd.posix_record_count, e);
    if (st != XPAR_OK)
      FATAL_FORMAT("a manifest entry of generation %" PRIu32
                   " is unreadable (%s)",
                   c->gen[h].sd.generation, xpar_status_str(st));
    if (e->posix_index != XPAR_ABSENT_U32 &&
        e->posix_index >= c->gen[h].sd.posix_record_count)
      FATAL_FORMAT("manifest entry %" PRIu32
                   " names a POSX record outside generation "
                   "%" PRIu32 "'s table", i, c->gen[h].sd.generation);
    own[i] = h);
  m->stream_base   = sd->stream_base;
  m->stream_length = sd->stream_length;
  m->dedup_level   = sd->dedup_level;
  m->align         = sd->align;
  m->slice_size    = sd->slice_size;
  { xpar_mf_limits lim;
    xpar_mf_result res;
    xpar_gen_range * anc;
    u32 * lineage;
    u32 na = 0, h = c->gen[g].parent;
    anc = xpar_calloc(c->gen_count ? c->gen_count : 1, sizeof *anc);
    lineage = xpar_calloc(c->gen_count ? c->gen_count : 1, sizeof *lineage);
    while (h != XPAR_GEN_NONE && na < c->gen_count) { lineage[na++] = h;  h = c->gen[h].parent; }
    FATAL_UNLESS(h == XPAR_GEN_NONE,
                 "the selected generation's ancestry is cyclic");
    Fi(na,
      u32 a = lineage[na - i - 1];
      anc[i].base = c->gen[a].sd.stream_base;
      anc[i].length = c->gen[a].sd.stream_length);
    xpar_free(lineage);
    xpar_memset(&lim, 0, sizeof lim);
    lim.stream_base        = sd->stream_base;
    lim.stream_length      = sd->stream_length;
    lim.slice_size         = sd->slice_size;
    lim.align              = sd->align;
    /*  POSX indices were checked against the generation that owns each
        FILE packet above; the effective manifest has no single POSX table.  */
    lim.posix_record_count = XPAR_ABSENT_U32;
    lim.ancestor           = na ? anc : NULL;
    lim.ancestor_count     = na;
    if (xpar_manifest_validate(m, &lim, &res) != XPAR_MF_OK)
      FATAL_FORMAT("manifest entry %" PRIu32 " is invalid: %s", res.entry,
                   xpar_mf_reason(res.status));
    xpar_free(anc);
  }
  { xpar_nameidx ix;
    xpar_posix_rec ** tab;
    u32 * tabn, posix_mismatch = 0;
    xpar_nameidx_build(m, &ix);
    tab = xpar_calloc(c->gen_count, sizeof *tab);
    tabn = xpar_calloc(c->gen_count, sizeof *tabn);
    Fi(c->gen_count, tabn[i] = xpar_gchain_posix(c, i, &tab[i]));
    Fi(m->count,
      xpar_entry * e = &m->entry[i];
      i64 target;
      const xpar_entry * t;
      if (e->entry_type != XPAR_ENTRY_HARDLINK) continue;
      target = xpar_link_target(m, &ix, i);
      if (target < 0) continue;       /*  Validation above rejects it.  */
      t = &m->entry[target];
      if ((e->posix_index == XPAR_ABSENT_U32) !=
          (t->posix_index == XPAR_ABSENT_U32) ||
          (e->posix_index != XPAR_ABSENT_U32 &&
           !xpar_posix_equal(&tab[own[i]][e->posix_index],
                             &tab[own[target]][t->posix_index])))
        posix_mismatch++;
      e->length = t->length;
      xpar_memcpy(e->content_hash, t->content_hash, sizeof e->content_hash);
      xpar_memcpy(e->prefix_hash, t->prefix_hash, sizeof e->prefix_hash);
      e->mode = t->mode;  e->attrs = t->attrs;
      e->mtime_ns = t->mtime_ns;  e->atime_ns = t->atime_ns;
      e->ctime_ns = t->ctime_ns;  e->btime_ns = t->btime_ns;
      e->posix_index = t->posix_index;
      own[i] = own[target]);
    Fi(c->gen_count, xpar_gchain_posix_free(tab[i], tabn[i]));
    xpar_free(tab);  xpar_free(tabn);
    xpar_nameidx_free(&ix);
    /*  Hard-link aliases must share one POSX record.  */
    if (posix_mismatch)
      FATAL_FORMAT("%" PRIu32 " hard-link alias%s disagree%s with the "
                   "canonical entry's POSX record", posix_mismatch,
                   posix_mismatch == 1 ? "" : "es",
                   posix_mismatch == 1 ? "s" : "");
  }
  *owner = own;
}

u32 xpar_gchain_posix(const xpar_chain * c, u32 g, xpar_posix_rec ** out) {
  u32 count = c->gen[g].sd.posix_record_count;
  *out = NULL;
  if (xpar_posx_collect(&c->crit, c->gen[g].set_id, count, out) != XPAR_OK)
    FATAL_FORMAT("generation %" PRIu32
                 "'s POSX table has gaps, overlaps, or invalid "
                 "ranges", c->gen[g].sd.generation);
  return count;
}

void xpar_gchain_posix_free(xpar_posix_rec * rec, u32 count) {
  xpar_posix_records_free(rec, count);
}

/*  Read critical-group armour from ARMG, not the region-code prologue.  */
bool xpar_gchain_crit_armour(const xpar_chain * c, u32 g,
                             xpar_armour_params * p) {
  u32 i;
  Fi(c->vol_count,
    xpar_scan sc;  xpar_pkt hdr;  const u8 * body;  u64 off;
    if (c->vol[i].gen != g) continue;
    xpar_scan_init(&sc, c->vol[i].data, c->vol[i].len, NULL, false);
    while (xpar_scan_next(&sc, &hdr, &body, &off)) {
      xpar_armg ag;
      if (!xpar_pkt_is(&hdr, XPAR_T_ARMG)) continue;
      if (xpar_armg_read(body, (sz) (hdr.length - XPAR_PKT_HDR), &ag) !=
          XPAR_OK) continue;
      p->symbol_bits = ag.symbol_bits;  p->poly = ag.poly;
      p->n = ag.n;  p->k = ag.k;  p->fcr = ag.fcr;  p->prim = ag.prim;
      p->depth = ag.depth;
      return xpar_armour_check(p) == NULL;
    });
  return false;
}

bool xpar_gchain_wrap_armour(const xpar_chain * c, u32 g, bool rcvs,
                             xpar_armour_params * p) {
  u32 i;
  /*  Prefer parameters stored for the requested packet kind.  */
  Fi(c->vol_count,
    if (c->vol[i].gen != g) continue;
    if (rcvs && c->vol[i].wrap_rcvs) { *p = c->vol[i].wrap_rcvs_ap;
                                       return true; }
    if (!rcvs && c->vol[i].wrap_tab) { *p = c->vol[i].wrap_tab_ap;
                                       return true; });
  return false;
}

void xpar_gchain_armour_bytes(const xpar_chain * c, u32 g, u64 * disk,
                              u64 * plain) {
  u32 i;
  *disk = *plain = 0;
  Fi(c->vol_count,
    if (c->vol[i].gen != g) continue;
    *disk  += c->vol[i].armg_disk;
    *plain += c->vol[i].armg_plain);
}

i64 xpar_gchain_gen_of(const xpar_chain * c, u64 off, u64 len) {
  u32 i;
  Fi(c->gen_count,
    u64 lo = c->gen[i].sd.stream_base;
    u64 hi = lo + c->gen[i].sd.stream_length;
    if (off >= lo && off < hi && len <= hi - off) return (i64) i);
  return -1;
}

/*  Count slices touched by superseded bytes.  */
void xpar_gchain_superseded(const xpar_chain * c, const xpar_manifest * m,
                            u64 * out) {
  xpar_occindex ix;
  u32 g;
  xpar_occindex_build(m, &ix);
  for (g = 0; g < c->gen_count; g++) {
    u64 z = c->gen[g].sd.slice_size;
    u64 s = c->gen[g].sd.data_slice_count;
    u64 base = c->gen[g].sd.stream_base;
    u64 span = c->gen[g].sd.stream_length;
    u64 q = c->gen[g].sd.align == XPAR_ALIGN_SLICE ? z
          : c->gen[g].sd.align == XPAR_ALIGN_1K
              ? (u64) XPAR_BLAKE3_CHUNK_LEN : 0;
    u64 at, end, dead = 0;
    u8 * gone;
    out[g] = 0;
    if (!z || !s || !span) continue;
    end = base + span;
    gone = xpar_calloc((sz) s, 1);
    for (at = base; at < end; ) {
      u64 nxt = xpar_occindex_next(&ix, at, end);
      if (nxt > at) {
        /*  Alignment padding is not superseded data.  */
        if (!(q && nxt - at < q && (nxt - base) % q == 0)) {
          u64 lo = (at - base) / z, hi = (nxt - 1 - base) / z, k;
          for (k = lo; k <= hi && k < s; k++) gone[k] = 1;
        }
        at = nxt;
        continue;
      }
      { xpar_occurrence oc;  u64 run = 0;
        if (xpar_occindex_canonical(&ix, at, &oc, &run) && run) at += run;
        else at++; }
    }
    for (at = 0; at < s; at++) if (gone[at]) dead++;
    out[g] = dead;
    xpar_free(gone);
  }
  xpar_occindex_free(&ix);
}

/*  Entries of MANIFEST whose bytes still live in generation G.  */
u64 xpar_gchain_users(const xpar_chain * c, const xpar_manifest * m, u32 g) {
  u64 users = 0;
  u32 i, k;
  Fi(m->count,
    bool hit = false;
    for (k = 0; k < m->entry[i].extent_count && !hit; k++)
      if (xpar_gchain_gen_of(c, m->entry[i].extents[k].stream_offset,
                             m->entry[i].extents[k].length) == (i64) g)
        hit = true;
    if (hit) users++);
  return users;
}

void xpar_gchain_deps(const xpar_chain * c, const xpar_manifest * m,
                      const u32 * owner, u64 * by_extent, u64 * by_packet) {
  u32 i, j, k;
  Fi(c->gen_count, by_extent[i] = 0;  by_packet[i] = 0);
  Fi(m->count,
    const xpar_entry * e = &m->entry[i];
    by_packet[owner[i]]++;
    Fj(c->gen_count,
      bool hit = false;
      for (k = 0; k < e->extent_count && !hit; k++) {
        i64 g = xpar_gchain_gen_of(c, e->extents[k].stream_offset,
                                   e->extents[k].length);
        if (g == (i64) j) hit = true;
      }
      if (hit) by_extent[j]++));
}

/*  Resolve absolute chain offsets through canonical occurrences.  */

typedef bool (*gen_read_fn)(void *, u64, u8 *, u64);

typedef struct {
  const xpar_manifest * m;
  xpar_occindex ix;
  xpar_file *   open_file;
  u32           open_entry;
  u64           limit;          /*  stream_base + stream_length.  */
  gen_read_fn   read;
  void *        read_ctx;
} gen_src;

static void gen_src_init(gen_src * s, const xpar_manifest * m, u64 limit) {
  s->m = m;  s->open_file = NULL;  s->open_entry = XPAR_GEN_NONE;
  s->limit = limit;  s->read = NULL;  s->read_ctx = NULL;
  xpar_occindex_build(m, &s->ix);
}

static void gen_src_use_reader(gen_src * s, gen_read_fn read, void * ctx) {
  s->read = read;
  s->read_ctx = ctx;
}

static void gen_src_free(gen_src * s) {
  if (s->open_file) xpar_close(s->open_file);
  xpar_occindex_free(&s->ix);
}

static void gen_src_read(gen_src * s, u64 off, u64 len, u8 * out) {
  if (s->read) {
    u64 take;
    if (off >= s->limit) { xpar_memset(out, 0, (sz) len);  return; }
    take = MIN(len, s->limit - off);
    if (take && !s->read(s->read_ctx, off, out, take))
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "the selected generation's stored stream became "
                 "unreadable; nothing was written");
    if (take < len) xpar_memset(out + take, 0, (sz) (len - take));
    return;
  }
  while (len) {
    xpar_occurrence occ;  u64 run = 0, take, at;
    if (off >= s->limit) {
      /*  Past L: the final data slice's zero padding, which is never
          stored and is regenerated here for the coder and the tags.  */
      xpar_memset(out, 0, (sz) len);
      return;
    }
    if (!xpar_occindex_canonical(&s->ix, off, &occ, &run)) {
      /*  Zero only to the next extent; gaps may be interior.  */
      u64 gap = xpar_occindex_next(&s->ix, off,
                                   MIN(off + len, s->limit)) - off;
      if (!gap) gap = len;
      xpar_memset(out, 0, (sz) gap);
      out += gap;  off += gap;  len -= gap;
      continue;
    }
    take = MIN(run, len);
    if (occ.entry != s->open_entry) {
      const char * path = s->m->source ? s->m->source[occ.entry] : NULL;
      if (s->open_file) xpar_close(s->open_file);
      s->open_file  = NULL;
      s->open_entry = occ.entry;
      if (!path)
        FATAL("entry %" PRIu32 " has no readable source; the stream cannot be "
              "rebuilt", occ.entry);
      s->open_file = xpar_open(path, XPAR_O_RDONLY);
      if (!s->open_file)
        FATAL_IO("cannot open '%s': %s", path, xpar_strerror(xpar_errno()));
    }
    /*  Offset the read within the occurrence, not merely its entry.  */
    at = occ.file_offset + (off - occ.stream_offset);
    if (xpar_pread(s->open_file, out, (sz) take, at) != (sz) take)
      FATAL_IO("short read from '%s' at %" PRIu64,
               s->m->source[occ.entry], at);
    out += take;  off += take;  len -= take;
  }
}

/*  Geometry, redundancy and the codec.  */

static u64 gen_resolve_r(const xpar_rspec * r, u64 s, u64 z) {
  f64 d;
  u64 v;
  switch (r->kind) {
    case XPAR_R_COUNT:   return r->count;
    case XPAR_R_PERCENT:
      d = (f64) s * r->factor / 100.0 + 0.5;
      v = d >= (f64) UINT64_MAX ? UINT64_MAX : (u64) d;
      return v ? v : 1;
    case XPAR_R_BYTES:   return z ? xpar_ceil_div(r->count, z) : 0;
    case XPAR_R_TIMES:
      d = (f64) s * r->factor + 0.5;
      v = d >= (f64) UINT64_MAX ? UINT64_MAX : (u64) d;
      return v ? v : 1;
    default: break;
  }
  return 0;
}

typedef struct {
  xpar_geom geom;
  u64 recovery;
  u64 encode_r;
  u8  field_log2, codec, axis;
} gen_plan;

/*  Report stale consolidate staging directories.  */
static void gen_report_stale_stage(const xpar_options * o, const char * base) {
  char * dir = xpar_path_dir(base);
  xpar_dir * d = xpar_opendir(*dir ? dir : ".");
  const xpar_dirent * e;
  if (!d && !xpar_errno_absent(xpar_errno()))
    xpar_fprintf(xpar_stderr, "xpar: cannot list '%s' for stale staging "
                 "trees: %s\n", *dir ? dir : ".",
                 xpar_strerror(xpar_errno()));
  if (d) {
    while ((e = xpar_readdir(d)) != NULL) {
      char * p;
#if defined(XPAR_DOS) || defined(__MSDOS__)
      sz n = xpar_strlen(e->name), i;
      bool stage = n == 8 && !xpar_strncmp(e->name, "GCO", 3);
      for (i = 3; stage && i < n; i++)
        stage = (e->name[i] >= '0' && e->name[i] <= '9') ||
                (e->name[i] >= 'a' && e->name[i] <= 'f') ||
                (e->name[i] >= 'A' && e->name[i] <= 'F');
      if (!e->is_dir || !stage) continue;
#else
      if (!e->is_dir || xpar_strncmp(e->name, ".xpar-consolidate-", 18))
        continue;
#endif
      p = xpar_path_join(dir, e->name);
      xpar_fprintf(xpar_stderr, "xpar: stale staging tree '%s'; safe to "
                   "remove\n", p);
      xpar_free(p);
    }
    xpar_closedir(d);
  }
  xpar_free(dir);
}

/*  Return the redundancy recorded by a generation's layout.  */
static u64 gen_gen_recovery(const xpar_chain * c, u32 g) {
  u64 top = c->gen[g].recovery_top;
  if (c->gen[g].layt_body) {
    xpar_layt l;
    if (xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, &l) ==
        XPAR_OK) {
      u64 n = 0;
      u32 i;
      Fi(l.count, if (l.vol[i].kind == XPAR_VOL_RECOVERY) n += l.vol[i].byte_length);
      xpar_layt_free(&l);
      if (n > top) top = n;
    }
  }
  return top;
}

/*  The redundancy one generation actually carries, as a percentage of S.  */
static f64 gen_gen_ratio(const xpar_chain * c, u32 g) {
  u64 s = c->gen[g].sd.data_slice_count, r = gen_gen_recovery(c, g);
  return s && r ? 100.0 * (f64) r / (f64) s : 0.0;
}

/*  Inherit the widest chain redundancy when consolidating.  */
static bool gen_inherit_recovery(xpar_options * eff, const xpar_chain * c,
                                 u32 g, bool whole_chain, f64 * ratio,
                                 u32 * from) {
  *ratio = gen_gen_ratio(c, g);
  *from  = g;
  if (whole_chain) {
    u32 at = g, walked = 0;
    while (at != XPAR_GEN_NONE && walked++ < c->gen_count) {
      f64 r = gen_gen_ratio(c, at);
      if (r > *ratio) { *ratio = r;  *from = at; }
      at = c->gen[at].parent;
    }
  }
  if (eff->recovery.kind != XPAR_R_NONE || *ratio <= 0.0) return false;
  eff->recovery.kind   = XPAR_R_PERCENT;
  eff->recovery.factor = *ratio;
  eff->recovery.count  = 0;
  return true;
}

/*  Report recovery spent on superseded ancestor slices.  */
static void gen_report_superseded(const xpar_options * o,
                                  const xpar_chain * c,
                                  const xpar_manifest * m, u32 head) {
  u64 * sup;
  u32 g, at, walked = 0;
  u8 * anc;
  if (o->quiet) return;
  sup = xpar_calloc(c->gen_count, sizeof *sup);
  anc = xpar_calloc(c->gen_count, 1);
  for (at = head; at != XPAR_GEN_NONE && walked++ < c->gen_count;
       at = c->gen[at].parent) anc[at] = 1;
  xpar_gchain_superseded(c, m, sup);
  for (g = 0; g < c->gen_count; g++) {
    u64 rec = gen_gen_recovery(c, g), users;
    if (!anc[g] || !sup[g]) continue;
    users = xpar_gchain_users(c, m, g);
    if (!users) continue;
    if (sup[g] >= rec)
      xpar_fprintf(xpar_stderr,
                   "xpar: warning: generation %" PRIu32 ": %" PRIu64 " of its %"
                   PRIu64 " slices are superseded and count as erasures, past "
                   "its %" PRIu64 " recovery slice%s; its %" PRIu64
                   " inherited entr%s no longer repairable; run 'xpar "
                   "consolidate' to restore protection\n",
                   c->gen[g].sd.generation, sup[g],
                   c->gen[g].sd.data_slice_count, rec, PLURAL(rec), users,
                   users == 1 ? "y is" : "ies are");
    else
      xpar_fprintf(xpar_stderr,
                   "xpar: warning: generation %" PRIu32 ": %" PRIu64 " of its %"
                   PRIu64 " slices are superseded and count as erasures; only %"
                   PRIu64 " of %" PRIu64 " recovery slices remain for its %"
                   PRIu64 " inherited entr%s\n",
                   c->gen[g].sd.generation, sup[g],
                   c->gen[g].sd.data_slice_count, rec - sup[g], rec, users,
                   users == 1 ? "y" : "ies");
  }
  xpar_free(sup);  xpar_free(anc);
}

/*  Say so when the new generation protects less than the old one did.  */
static void gen_warn_thinner(const xpar_options * o, f64 was, u64 r, u64 s) {
  f64 now = s ? 100.0 * (f64) r / (f64) s : 0.0;
  if (o->quiet || !s || was <= 0.0 || now >= was - 0.05) return;
  xpar_fprintf(xpar_stderr,
               "xpar: warning: redundancy falls from %.1f%% to %.1f%%; pass "
               "-r to keep the old ratio\n", was, now);
}

/*  Auto selects matrix coding for stable, field-wide recovery rows. Honour
    explicit FFT requests where their capacity rules allow.  */
static void gen_choose(const xpar_options * o, u64 stream_length,
                       gen_plan * p) {
  xpar_geom_req rq;
  xpar_geom_status st;
  u64 r = 0, maxr;
  int pass;

  xpar_memset(p, 0, sizeof *p);
  p->codec = (o->codec == XPAR_CLI_AUTO) ? XPAR_CODEC_MATRIX
                                         : (u8) o->codec;
  p->field_log2 = (o->field == XPAR_CLI_AUTO) ? 8 : (u8) o->field;

  /*  Two passes: three of -r's four forms are a function of S, and S
      depends on the field bound, which depends on R (slice.c).  */
  for (pass = 0; pass < 2; pass++) {
    xpar_memset(&rq, 0, sizeof rq);
    rq.stream_length = stream_length;
    rq.slice_size    = o->slice_size;
    rq.slice_count   = o->slices;
    rq.recovery      = r;
    rq.cell_bytes    = (u32) o->cell_bytes;
    rq.field_log2    = p->field_log2;
    st = xpar_geom_choose(&rq, &p->geom);
    if (st == XPAR_GEOM_FIELD && o->field == XPAR_CLI_AUTO &&
        p->field_log2 == 8) {
      p->field_log2 = 16;
      st = xpar_geom_choose(&rq, &p->geom);
    }
    if (st != XPAR_GEOM_OK)
      FATAL("cannot choose a geometry: %s", xpar_geom_reason(st));
    if (o->align == XPAR_ALIGN_1K &&
        (p->geom.slice_size < XPAR_BLAKE3_CHUNK_LEN ||
         (p->geom.slice_size & (p->geom.slice_size - 1)) != 0)) {
      FATAL_UNLESS(!o->slice_size && !o->slices,
                   "--align=1k needs a power-of-two slice size of at least "
                   "1 KiB; the explicit geometry does not provide one");
      rq.slice_size = xpar_next_pow2(MAX(p->geom.slice_size,
                                         (u64) XPAR_BLAKE3_CHUNK_LEN));
      rq.slice_count = 0;
      st = xpar_geom_choose(&rq, &p->geom);
      if (st != XPAR_GEOM_OK)
        FATAL("cannot choose a geometry: %s", xpar_geom_reason(st));
    }
    r = gen_resolve_r(&o->recovery, p->geom.slice_count, p->geom.slice_size);
    if (!r && o->recovery.kind == XPAR_R_NONE && p->geom.slice_count) {
      /*  An omitted -r uses create's five-percent default.  */
      r = (p->geom.slice_count * 5 + 99) / 100;
      if (!r) r = 1;
    }
    if (o->min_recovery && r < o->min_recovery) r = o->min_recovery;
    if (!p->geom.slice_count) r = 0;
  }
  p->recovery = r;
  p->encode_r = r;

  if (p->codec == XPAR_CODEC_FFT && r > p->geom.slice_count)
    p->codec = XPAR_CODEC_FFT_LOW;

  if (r && !xpar_codec_supports(p->codec, p->field_log2,
                                p->geom.slice_count, r)) {
    if (o->field == XPAR_CLI_AUTO && p->field_log2 == 8 &&
        xpar_codec_supports(p->codec, 16, p->geom.slice_count, r))
      p->field_log2 = 16;
    else
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "the %s codec cannot express S=%" PRIu64 ", R=%" PRIu64 " over GF(2^%" PRIu8 "); "
                 "try --codec=matrix",
                 xpar_codec_name(p->codec),
                 p->geom.slice_count,
                 r, p->field_log2);
  }

  /*  Reserve capacity promised by --max-recovery.  */
  maxr = gen_resolve_r(&o->max_recovery, p->geom.slice_count,
                       p->geom.slice_size);
  if (maxr < r) maxr = r;
  if (p->codec == XPAR_CODEC_MATRIX) {
    if (maxr && !xpar_codec_supports(p->codec, p->field_log2,
                                     p->geom.slice_count, maxr)) {
      if (o->field == XPAR_CLI_AUTO && p->field_log2 == 8 &&
          xpar_codec_supports(p->codec, 16, p->geom.slice_count, maxr))
        p->field_log2 = 16;
      else
        FATAL_CODE(XPAR_EXIT_NOPLAN,
                   "--max-recovery=%" PRIu64 " does not fit beside S=%" PRIu64
                   " in GF(2^%" PRIu8 ")",
                   maxr, p->geom.slice_count, p->field_log2);
    }
    p->axis = p->field_log2;
  } else if (p->codec == XPAR_CODEC_FFT_LOW) {
    u64 m = xpar_next_pow2(p->geom.slice_count);
    if (maxr > m) m = xpar_next_pow2(maxr);
    p->axis = (u8) xpar_log2_floor(m);
  } else {
    u64 m = xpar_next_pow2(r ? r : 1);
    if (maxr > m) {
      u64 wide = xpar_next_pow2(maxr);
      if (!xpar_codec_supports(p->codec, p->field_log2, p->geom.slice_count,
                               wide))
        FATAL_CODE(XPAR_EXIT_NOPLAN,
                   "--max-recovery=%" PRIu64 " needs a recovery axis of %" PRIu64 ", "
                   "which this field and S cannot express",
                   maxr, wide);
      m = wide;
    }
    p->axis = (u8) xpar_log2_floor(m);
  }
}

/*  Encoding one generation's stream.  */

typedef struct {
  u32 * slice_crc;
  u8  * slice_tag;
  u8    tag_len;
  u32 * cell_crc;
  u8  * rec;
  xpar_file * rec_spill;
  char * rec_path;
  u64 rec_z;
} gen_tables;

static void gen_tables_free(gen_tables * t) {
  xpar_free(t->slice_crc);  xpar_free(t->slice_tag);
  xpar_free(t->cell_crc);   xpar_free(t->rec);
  if (t->rec_spill) xpar_close(t->rec_spill);
  if (t->rec_path) xpar_remove(t->rec_path);
  xpar_free(t->rec_path);
  xpar_memset(t, 0, sizeof *t);
}

static void gen_rec_spill_open(gen_tables * t, const char * base) {
  u32 i;
  for (i = 0; i < 1000; i++) {
#if defined(XPAR_DOS) || defined(__MSDOS__)
    t->rec_path = xpar_dos_numbered(base, "GEN", "TMP", i);
#else
    xpar_asprintf(&t->rec_path, "%s.xpar-encode-tmp-%03" PRIu32, base, i);
#endif
    t->rec_spill = xpar_open(t->rec_path, XPAR_O_RDWR | XPAR_O_CREAT |
                                          XPAR_O_EXCL);
    if (t->rec_spill) return;
    xpar_free(t->rec_path);  t->rec_path = NULL;
  }
  FATAL_IO("cannot create an encoding scratch file beside '%s': %s", base,
           xpar_strerror(xpar_errno()));
}

static void gen_rec_put(gen_tables * t, u64 e, u64 off,
                        const u8 * p, u64 n) {
  if (t->rec) { xpar_memcpy(t->rec + e * t->rec_z + off, p, (sz) n);  return; }
  if (xpar_pwrite(t->rec_spill, p, (sz) n, e * t->rec_z + off) != (sz) n)
    FATAL_IO("cannot write encoding scratch '%s'", t->rec_path);
}

static const u8 * gen_rec_get(gen_tables * t, u64 e, u8 * scratch) {
  if (t->rec) return t->rec + e * t->rec_z;
  if (xpar_pread(t->rec_spill, scratch, (sz) t->rec_z,
                 e * t->rec_z) != (sz) t->rec_z)
    FATAL_IO("cannot read encoding scratch '%s'", t->rec_path);
  return scratch;
}

/*  Encoding uses bounded columns and spills recovery output beyond -m.  */
static void gen_encode(const xpar_manifest * m, const gen_plan * p,
                       u8 tag_len, u64 memory, const char * scratch_base,
                       const xpar_key * key, gen_read_fn read, void * read_ctx,
                       gen_tables * t,
                       xpar_progress_t * prog) {
  u64 S = p->geom.slice_count, Z = p->geom.slice_size, R = p->encode_r;
  u32 K = p->geom.cells_per_slice, Y = p->geom.cell_bytes;
  u64 c, i, j, chunk, budget = memory ? memory : xpar_plan_default_memory();
  u64 cells = 0, meta;
  u8 * data;
  gen_src src;
  xpar_codec * codec = NULL;
  xpar_setd tag_sd;

  xpar_memset(t, 0, sizeof *t);
  xpar_memset(&tag_sd, 0, sizeof tag_sd);
  tag_sd.slice_size = Z;
  tag_sd.stream_base = m->stream_base;
  if (m->align == XPAR_ALIGN_1K)
    tag_sd.required_features = XPAR_FEAT_B3_SUBTREE;
  t->tag_len = tag_len;
  t->rec_z = Z;
  if (!S) return;

  if (Y && K && S > (u64) -1 / K)
    FATAL_CODE(XPAR_EXIT_NOPLAN, "the cell checksum table is too large");
  if (Y) cells = S * K;
  if (S > ((u64) -1) / (4 + tag_len) ||
      cells > (((u64) -1) - S * (4 + tag_len)) / 4)
    FATAL_CODE(XPAR_EXIT_NOPLAN, "the checksum tables are too large");
  meta = S * (4 + tag_len) + cells * 4;

  if (S > ((u64) (sz) -1) / 4 ||
      (tag_len && S > ((u64) (sz) -1) / tag_len) ||
      cells > ((u64) (sz) -1) / 4 || meta > budget || Z > budget - meta)
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "the checksum tables plus one %" PRIu64 "-byte slice need more than "
               "-m %" PRIu64 "; raise -m or choose a smaller slice size",
               Z, budget);

  t->slice_crc = xpar_calloc((sz) S, 4);
  if (tag_len)
    t->slice_tag = xpar_calloc((sz) S, tag_len);
  if (Y) t->cell_crc = xpar_calloc((sz) cells, 4);

  data = xpar_alloc_raw((sz) Z);
  gen_src_init(&src, m, m->stream_base + m->stream_length);
  gen_src_use_reader(&src, read, read_ctx);
  Fi(S,
    gen_src_read(&src, m->stream_base + i * Z, Z, data);
    t->slice_crc[i] = xpar_crc32c(0, data, (sz) Z);
    if (tag_len) {
      if (key)
        xpar_slice_tag_keyed(&tag_sd, i, data, key->k_slice,
                             t->slice_tag + i * tag_len, tag_len);
      else
        xpar_slice_tag(&tag_sd, i, data, t->slice_tag + i * tag_len,
                       tag_len);
    }
    if (Y) {
      u32 col;
      for (col = 0; col < K; col++) {
        u64 at = (u64) col * Y;
        t->cell_crc[i * K + col] = xpar_crc32c(
          0, data + at, (sz) xpar_cell_size(&p->geom, col));
      }
    }
    if (prog) xpar_progress_tick(prog, Z));
  gen_src_free(&src);
  if (!R) { xpar_free(data);  return; }

  codec = xpar_codec_new_axis(p->codec, p->field_log2, S, R, p->axis);
  if (meta + xpar_codec_encode_footprint_axis(
               p->codec, p->field_log2, S, R, p->axis, (sz) Z) <= budget &&
      R <= ((u64) (sz) -1) / Z)
    t->rec = xpar_alloc_raw((sz) (R * Z));
  else gen_rec_spill_open(t, scratch_base);

  if (p->codec == XPAR_CODEC_MATRIX) {
    u64 first = 0, batch = R;
    u8 * pool;
    u8 ** rptr;
    while (batch > 1 &&
           meta + xpar_codec_encode_footprint_axis(
                    p->codec, p->field_log2, S, batch, p->axis,
                    (sz) Z) > budget)
      batch = (batch + 1) / 2;
    if (meta + xpar_codec_encode_footprint_axis(
                 p->codec, p->field_log2, S, batch, p->axis,
                 (sz) Z) > budget)
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "the matrix encoder needs one data slice and one recovery "
                 "accumulator (%" PRIu64 " bytes each), which do not fit -m %" PRIu64,
                 Z, budget);
    pool = t->rec ? NULL : xpar_alloc_aligned((sz) (batch * Z), 64);
    rptr = xpar_alloc_raw((sz) batch * sizeof *rptr);
    while (first < R) {
      u64 nr = MIN(batch, R - first);
      Fj(nr, rptr[j] = t->rec ? t->rec + (first + j) * Z : pool + j * Z);
      gen_src_init(&src, m, m->stream_base + m->stream_length);
      gen_src_use_reader(&src, read, read_ctx);
      Fi(S,
        gen_src_read(&src, m->stream_base + i * Z, Z, data);
        if (xpar_codec_matrix_accumulate(codec, i, data, first, rptr, nr,
                                         (sz) Z, i == 0) != XPAR_CODEC_OK)
          FATAL_CODE(XPAR_EXIT_INTERNAL, "internal: matrix streaming encode "
                     "refused a supported range"));
      gen_src_free(&src);
      if (!t->rec)
        Fj(nr, gen_rec_put(t, first + j, 0, rptr[j], Z));
      first += nr;
    }
    xpar_free(rptr);
    if (pool) xpar_free_aligned(pool);
  } else {
    u8 ** dptr;
    u8 ** rptr;
    u8 * pool;
    chunk = Z;
    while (chunk >= 64 &&
           meta + xpar_codec_encode_footprint_axis(
                    p->codec, p->field_log2, S, R, p->axis,
                    (sz) chunk) > budget)
      chunk = (chunk / 2) & ~(u64) 63;
    if (chunk < 64)
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "the FFT encoder's minimum 64-byte column does not fit "
                 "-m %" PRIu64, budget);
    xpar_free(data);  data = NULL;
    pool = xpar_alloc_aligned(
             (sz) ((S + (t->rec ? 0 : R)) * chunk), 64);
    dptr = xpar_alloc_raw((sz) S * sizeof *dptr);
    rptr = xpar_alloc_raw((sz) R * sizeof *rptr);
    Fi(S, dptr[i] = pool + i * chunk);
    if (!t->rec)
      Fj(R, rptr[j] = pool + (S + j) * chunk);
    gen_src_init(&src, m, m->stream_base + m->stream_length);
    gen_src_use_reader(&src, read, read_ctx);
    for (c = 0; c < Z; c += chunk) {
      u64 len = MIN(chunk, Z - c);
      Fi(S, gen_src_read(&src, m->stream_base + i * Z + c, len, dptr[i]));
      if (t->rec)
        Fj(R, rptr[j] = t->rec + j * Z + c);
      if (xpar_codec_encode(codec, (const u8 * const *) dptr, rptr,
                            (sz) len) != XPAR_CODEC_OK)
        FATAL_CODE(XPAR_EXIT_INTERNAL, "internal: FFT encode refused a "
                   "supported geometry");
      if (!t->rec)
        Fj(R, gen_rec_put(t, j, c, rptr[j], len));
    }
    gen_src_free(&src);
    xpar_free(dptr);  xpar_free(rptr);  xpar_free_aligned(pool);
  }
  xpar_codec_free(codec);
  xpar_free(data);
}

/*  Critical metadata uses the selected armour field and t=16 by default.  */
static void gen_armour_params(const xpar_options * o,
                              xpar_armour_params * p) {
  u32 t = 16;
  xpar_armour_defaults(p, o->armour_field == 16 ? 16 : 8);
  if (o->armour_t) t = o->armour_t;
  else if (o->armour_pct > 0.0)
    /*  P is overhead over the data, so 2t/(n - 2t) = P/100.  */
    t = (u32) (o->armour_pct * (f64) p->n / (100.0 + o->armour_pct)
               / 2.0 + 0.5);
  if (!t) t = 1;
  if (t > (p->n - 1) / 2) t = (p->n - 1) / 2;
  p->k = p->n - 2 * t;
  if (o->depth) p->depth = o->depth;
  else if (o->burst) {
    /*  Saturate the increment to prevent wraparound.  */
    u64 want = o->burst == (u64) -1 ? (u64) -1 : o->burst + 1;
    p->depth = xpar_ceil_div(want, (u64) t * (p->symbol_bits / 8));
    if (!p->depth) p->depth = 1;
  }
}

static void gen_armour_pack_ap(xpar_buf * out, const xpar_armour_params * p,
                               const u8 * plain, sz plain_len,
                               const u8 * set_id, const xpar_key * key) {
  xpar_armour_params ap = *p;  xpar_armour * a;  xpar_armg ag;  u8 * arm;
  const char * why;
  why = xpar_armour_check(&ap);
  if (why) FATAL("invalid armour parameters: %s", why);
  a = xpar_armour_new(&ap);
  xpar_memset(&ag, 0, sizeof ag);
  ag.symbol_bits     = (u8) ap.symbol_bits;
  ag.poly            = ap.poly;   ag.n = ap.n;  ag.k = ap.k;
  ag.fcr             = ap.fcr;    ag.prim = ap.prim;
  ag.depth           = ap.depth;
  ag.plain_length    = plain_len;
  ag.armoured_length = xpar_armour_size(a, plain_len);
  arm = xpar_calloc((sz) ag.armoured_length, 1);
  xpar_armour_encode(a, arm, plain, plain_len);
  xpar_armg_write(out, &ag, arm, set_id, key);
  xpar_free(arm);
  xpar_armour_free(a);
}

static void gen_armour_pack(xpar_buf * out, const xpar_options * o,
                            const u8 * plain, sz plain_len,
                            const u8 * set_id, const xpar_key * key) {
  xpar_armour_params ap;
  gen_armour_params(o, &ap);
  gen_armour_pack_ap(out, &ap, plain, plain_len, set_id, key);
}

/*  Build armour for one wrapped RCVS packet.  */
static xpar_armour * gen_rcvs_armour(const xpar_options * o, u64 z,
                                     const xpar_armour_params * stored) {
  xpar_armour_params ap;
  const char * why;
  if (stored) ap = *stored;
  else
    xpar_armour_wrap_params(o, xpar_align_up(XPAR_PKT_HDR + 16 + z,
                                             XPAR_PKT_ALIGN), &ap);
  why = xpar_armour_check(&ap);
  if (why) FATAL("invalid armour parameters: %s", why);
  xpar_gf_init();
  return xpar_armour_new(&ap);
}

/*  Emit one recovery slice, optionally wrapped.  */
static void gen_rcvs_emit(xpar_buf * out, const xpar_armour * ra, u64 e,
                          const u8 * rec, sz z, const u8 * set_id,
                          const xpar_key * key) {
  xpar_buf pkt;
  if (!ra) { xpar_rcvs_write(out, e, rec, z, set_id, key);  return; }
  xpar_buf_init(&pkt);
  xpar_rcvs_write(&pkt, e, rec, z, set_id, key);
  xpar_armg_wrap_with(out, ra, pkt.data, pkt.len, set_id, key);
  xpar_buf_free(&pkt);
}

/*  Emit slice tables, optionally wrapped.  */
static void gen_emit_tables(xpar_buf * out, const xpar_options * o,
                            bool wrap, const gen_tables * t,
                            const gen_plan * pl, u8 tag_len,
                            const u8 * set_id, const xpar_key * key) {
  xpar_buf b;
  xpar_buf * d = out;
  if (wrap) { xpar_buf_init(&b);  d = &b; }
  if (t->slice_tag)
    xpar_sltg_write_all(d, t->slice_tag, pl->geom.slice_count, tag_len,
                        set_id, key);
  if (t->cell_crc)
    xpar_slcl_write_all(d, t->cell_crc, pl->geom.slice_count,
                        pl->geom.cell_bytes, pl->geom.cells_per_slice,
                        set_id, key);
  if (wrap) { xpar_armg_wrap_each(out, o, b.data, b.len, set_id, key);  xpar_buf_free(&b); }
}

/*  Convert a stored region code to wrapping options.  */
static void gen_wrap_options(xpar_options * wo,
                             const xpar_armour_params * ap) {
  wo->armour_field = (int) ap->symbol_bits;
  wo->armour_t     = (ap->n - ap->k) / 2;
  wo->armour_pct   = 0.0;
  wo->depth        = ap->depth <= 0xFFFFFFFFU ? (u32) ap->depth : 1;
  wo->burst        = 0;
}

/*  Resolve a generation's inherited or explicit armour level.  */
static int gen_chain_armour(const xpar_options * o, const xpar_chain * c,
                            u32 g, int layout) {
  xpar_armour_params ap;
  if (layout == XPAR_LAYOUT_ARMOURED) return XPAR_ARMOUR_ALL;
  if (o->armour_given) return o->armour;
  if (xpar_gchain_wrap_armour(c, g, true, &ap) ||
      xpar_gchain_wrap_armour(c, g, false, &ap))
    return XPAR_ARMOUR_ALL;
  return XPAR_ARMOUR_METADATA;
}

/*  Return the generation's RCVS armour, or NULL for plain packets.  */
static xpar_armour * gen_wrap_rcvs_armour(const xpar_options * o,
                                          const xpar_chain * c, u32 g,
                                          int level, u64 z) {
  xpar_armour_params ap;
  xpar_options wo = *o;
  if (level != XPAR_ARMOUR_ALL) return NULL;
  if (xpar_gchain_wrap_armour(c, g, true, &ap))
    return gen_rcvs_armour(o, 0, &ap);
  if (xpar_gchain_wrap_armour(c, g, false, &ap)) gen_wrap_options(&wo, &ap);
  return gen_rcvs_armour(&wo, z, NULL);
}

void xpar_garm_write_prologue(xpar_file * f,
                              const xpar_armour_params * ap,
                              u64 plain_len, u64 armoured_len,
                              u64 stream_offset, u64 stream_len) {
  xpar_arm_prologue pr;
  xpar_armour_params pp;
  xpar_armour * pa;
  u8 copy[ARM_COPY_LEN];
  u8 frame[255];
  u32 i;
  xpar_memset(&pr, 0, sizeof pr);
  pr.symbol_bits = (u8) ap->symbol_bits;  pr.poly = ap->poly;
  pr.n = ap->n;  pr.k = ap->k;  pr.fcr = ap->fcr;  pr.prim = ap->prim;
  pr.depth = ap->depth;  pr.plain_length = plain_len;
  pr.armoured_length = armoured_len;  pr.stream_offset = stream_offset;
  pr.stream_length = stream_len;
  xpar_memset(copy, 0, sizeof copy);
  arm_prologue_encode(copy, &pr);
  xpar_armour_defaults(&pp, 8);
  pp.n = 255;  pp.k = 223;  pp.depth = 1;
  pa = xpar_armour_new(&pp);
  xpar_memset(frame, 0, sizeof frame);
  xpar_memcpy(frame, copy, ARM_PLAIN_LEN);
  xpar_armour_encode_frame(pa, frame);
  xpar_memcpy(copy + ARM_PLAIN_LEN, frame + pp.k, 32);
  xpar_armour_free(pa);
  Fi(3, xpar_xwrite(f, copy, sizeof copy));
}

/*  Re-armour mapped plaintext with sparse repaired-slice replacements.  */
void xpar_garm_write_patched(const char * path,
                             const xpar_armour_params * ap,
                             const u8 * plain, u64 plain_len,
                             u64 stream_offset, u64 stream_len,
                             xpar_file * staged, const u64 * slot,
                             u64 slice_count, u64 slice_size) {
  xpar_armour * a = xpar_armour_new(ap);
  xpar_armsink sink;
  xpar_file * f;
  char * tmp;
  u8 * io = NULL;
  u64 at = 0, stream_end;
  FATAL_UNLESS(a != NULL, "the armoured maintenance parameters are invalid");
  FATAL_UNLESS(stream_offset <= plain_len && stream_len <= plain_len -
                 stream_offset,
               "the armoured protected stream lies outside its plaintext");
  stream_end = stream_offset + stream_len;
  if (staged) io = xpar_alloc_raw(1U << 16);
  f = gen_stage_open(path, &tmp);
  xpar_garm_write_prologue(f, ap, plain_len, xpar_armour_size(a, plain_len),
                         stream_offset, stream_len);
  xpar_armsink_init(&sink, a, f);
  while (at < plain_len) {
    u64 take;
    if (!staged || !slot || at < stream_offset || at >= stream_end) {
      u64 boundary = at < stream_offset ? stream_offset : plain_len;
      if (at >= stream_end) boundary = plain_len;
      take = MIN(boundary - at, (u64) 1 << 20);
      xpar_armsink_put(&sink, plain + at, take);
    } else {
      u64 rel = at - stream_offset;
      u64 slice = slice_size ? rel / slice_size : 0;
      u64 in = slice_size ? rel % slice_size : 0;
      FATAL_UNLESS(slice < slice_count && slice_size != 0,
                   "a staged armoured slice lies outside the stream");
      take = MIN(MIN(stream_end - at, slice_size - in),
                 (u64) 1 << 16);
      if (slot[slice] == UINT64_MAX)
        xpar_armsink_put(&sink, plain + at, take);
      else {
        if (xpar_pread(staged, io, (sz) take,
                       slot[slice] * slice_size + in) != (sz) take)
          FATAL_IO("cannot read staged armoured repair bytes");
        xpar_armsink_put(&sink, io, take);
      }
    }
    at += take;
  }
  xpar_armsink_flush(&sink);
  xpar_armsink_free(&sink);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("cannot flush rebuilt armoured archive '%s'", tmp);
  xpar_xclose(f);
  gen_publish_whole(tmp, path, true);
  xpar_free(io);
  xpar_armour_free(a);
}

/*  Insert rebuilt metadata while streaming; adjust the prologue offset only
    for insertion before STRM.  */
void xpar_garm_write_inserted(const char * path,
                              const xpar_armour_params * ap,
                              const u8 * plain, u64 plain_len, u64 insert,
                              const u8 * extra, u64 extra_len,
                              u64 stream_offset, u64 stream_len) {
  xpar_armour * a = xpar_armour_new(ap);
  xpar_armsink sink;
  xpar_file * f;
  char * tmp;
  FATAL_UNLESS(a != NULL, "the armoured maintenance parameters are invalid");
  FATAL_UNLESS(insert <= plain_len &&
               stream_len <= UINT64_MAX - stream_offset &&
               (insert <= stream_offset ||
                insert >= stream_offset + stream_len) &&
               extra_len <= UINT64_MAX - plain_len,
               "the armoured metadata insertion lies outside its plaintext");
  f = gen_stage_open(path, &tmp);
  xpar_garm_write_prologue(f, ap, plain_len + extra_len,
                         xpar_armour_size(a, plain_len + extra_len),
                         stream_offset + (insert <= stream_offset
                                            ? extra_len : 0),
                         stream_len);
  xpar_armsink_init(&sink, a, f);
  xpar_armsink_put(&sink, plain, insert);
  xpar_armsink_put(&sink, extra, extra_len);
  xpar_armsink_put(&sink, plain + insert, plain_len - insert);
  xpar_armsink_flush(&sink);
  xpar_armsink_free(&sink);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("cannot flush rebuilt armoured archive '%s'", tmp);
  xpar_xclose(f);
  gen_publish_whole(tmp, path, true);
  xpar_armour_free(a);
}

static u64 gen_stream_tag(const xpar_manifest * m, u64 local_offset,
                          u64 length) {
  gen_src src;
  xpar_blake3_t h;
  u8 * buf = xpar_alloc_raw(1U << 16);
  u64 at = m->stream_base + local_offset, left = length;
  xpar_vol_tag_begin(&h);
  gen_src_init(&src, m, m->stream_base + m->stream_length);
  while (left) {
    u64 take = MIN(left, (u64) 1 << 16);
    gen_src_read(&src, at, take, buf);
    xpar_blake3_update(&h, buf, (sz) take);
    at += take;  left -= take;
  }
  gen_src_free(&src);
  xpar_free(buf);
  return xpar_vol_tag_final(&h);
}

static char * gen_stage_data_range(const xpar_manifest * m,
                                   const char * path, u64 local_offset,
                                   u64 length, bool replace) {
  gen_src src;
  char * tmp;
  xpar_file * f;
  u8 * buf = xpar_alloc_raw(1U << 16);
  u64 at, left = length;
  if (local_offset > m->stream_length ||
      length > m->stream_length - local_offset)
    FATAL_CODE(XPAR_EXIT_INTERNAL,
               "internal: a data-volume range is outside its generation");
  at = m->stream_base + local_offset;
  if (!replace && gen_exists(path))
    FATAL("'%s' exists; use -f to overwrite it", path);
  f = gen_stage_open(path, &tmp);
  gen_src_init(&src, m, m->stream_base + m->stream_length);
  while (left) {
    u64 take = MIN(left, (u64) 1 << 16);
    gen_src_read(&src, at, take, buf);
    xpar_xwrite(f, buf, (sz) take);
    at += take;  left -= take;
  }
  gen_src_free(&src);
  xpar_free(buf);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("cannot flush temporary data volume '%s'", tmp);
  xpar_xclose(f);
  return tmp;
}

static char * gen_stage_arm_archive(const char * path,
                                    const xpar_armour_params * ap,
                                    const xpar_manifest * m,
                                    const gen_plan * plan,
                                    gen_tables * tables,
                                    u8 * rec_scratch, xpar_buf * head,
                                    const u8 * set_id, const xpar_key * key,
                                    const xpar_wropt * w,
                                    gen_read_fn read, void * read_ctx) {
  xpar_armour * a = xpar_armour_new(ap);
  xpar_armsink sink;
  gen_src src;
  xpar_buf tail, crtr;
  xpar_file * f;
  char * tmp;
  u8 * buf;
  u64 stream_packet, stream_at, plain_len, at, left, e;
  xpar_strm_write_header(head, m->stream_length, set_id, key);
  stream_at = head->len;
  xpar_buf_init(&tail);
  if (tables->slice_tag)
    xpar_sltg_write_all(&tail, tables->slice_tag, plan->geom.slice_count,
                        tables->tag_len, set_id, key);
  if (tables->cell_crc)
    xpar_slcl_write_all(&tail, tables->cell_crc, plan->geom.slice_count,
                        plan->geom.cell_bytes, plan->geom.cells_per_slice,
                        set_id, key);
  xpar_buf_init(&crtr);
  xpar_crtr_write(&crtr, "xpar " PACKAGE_VERSION, set_id, key, w);
  stream_packet = xpar_align_up(XPAR_PKT_HDR + 16 + m->stream_length,
                                XPAR_PKT_ALIGN);
  plain_len = head->len - (XPAR_PKT_HDR + 16) + stream_packet + tail.len +
              plan->recovery * (XPAR_PKT_HDR + 16 + plan->geom.slice_size) +
              crtr.len;
  f = gen_stage_open(path, &tmp);
  xpar_garm_write_prologue(f, ap, plain_len, xpar_armour_size(a, plain_len),
                         stream_at, m->stream_length);
  xpar_armsink_init(&sink, a, f);
  xpar_armsink_put(&sink, head->data, head->len);
  buf = xpar_alloc_raw(1U << 16);
  gen_src_init(&src, m, m->stream_base + m->stream_length);
  gen_src_use_reader(&src, read, read_ctx);
  at = m->stream_base;  left = m->stream_length;
  while (left) {
    u64 take = MIN(left, (u64) 1 << 16);
    gen_src_read(&src, at, take, buf);
    xpar_armsink_put(&sink, buf, take);
    at += take;  left -= take;
  }
  gen_src_free(&src);
  { u8 zero[XPAR_PKT_ALIGN] = { 0 };
    u64 pad = stream_packet - (XPAR_PKT_HDR + 16 + m->stream_length);
    if (pad) xpar_armsink_put(&sink, zero, pad);
  }
  xpar_armsink_put(&sink, tail.data, tail.len);
  for (e = 0; e < plan->recovery; e++) {
    xpar_buf pkt;
    const u8 * rec = gen_rec_get(tables, e, rec_scratch);
    xpar_buf_init(&pkt);
    xpar_rcvs_write(&pkt, e, rec, (sz) plan->geom.slice_size, set_id, key);
    xpar_armsink_put(&sink, pkt.data, pkt.len);
    xpar_buf_free(&pkt);
  }
  xpar_armsink_put(&sink, crtr.data, crtr.len);
  xpar_armsink_flush(&sink);
  xpar_armsink_free(&sink);
  xpar_free(buf);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("cannot flush temporary armoured archive '%s'", tmp);
  xpar_xclose(f);
  xpar_buf_free(&tail);  xpar_buf_free(&crtr);
  xpar_armour_free(a);
  return tmp;
}

/*  Writing one generation.  */

typedef struct {
  u64    first, count;          /*  Recovery exponents carried here.  */
  char * name;
  bool   is_index;
} gen_vol;

static gen_vol * gen_volumes(const xpar_options * o, u64 r, const char * base,
                             u32 gen, u32 * count) {
  gen_vol * v = NULL;
  u32 n = 0, i;
  u64 left = r, step = 1, first = 0;
  int wf, wc;

  v = xpar_calloc(1, sizeof *v);
  v[0].is_index = true;
  n = 1;

  /*  An armoured layout is one archive. Recovery packets live inside its
      protected region, so exposing sidecar recovery names in LAYT would
      describe files the writer never creates.  */
  if (o->layout == XPAR_LAYOUT_ARMOURED) r = 0;

  if (r) {
    u64 fixed = 0;
    if (o->volumes != XPAR_VOLS_LADDER)
      fixed = o->volume_count ? o->volume_count : 1;
    while (left) {
      u64 take = fixed ? xpar_ceil_div(left, fixed - (n - 1)) : MIN(step, left);
      if (take > left) take = left;
      v = xpar_realloc(v, (sz) (n + 1) * sizeof *v);
      xpar_memset(&v[n], 0, sizeof v[n]);
      v[n].first = first;  v[n].count = take;
      n++;
      first += take;  left -= take;  step *= 2;
      if (fixed && n - 1 == fixed && left) {
        /*  Rounding left a tail: it belongs to the last volume rather
            than to a volume the user did not ask for.  */
        v[n - 1].count += left;  left = 0;
      }
    }
  }

  { u64 max_first = 0, max_count = 1;
    for (i = 1; i < n; i++) {
      if (v[i].first > max_first) max_first = v[i].first;
      if (v[i].count > max_count) max_count = v[i].count;
    }
    xpar_vname_widths(max_first, max_count, &wf, &wc);
  }
  v[0].name = xpar_vname_index(base, gen);
  for (i = 1; i < n; i++)
    v[i].name = xpar_vname_recovery(base, gen, v[i].first, v[i].count,
                                    wf, wc, i - 1);
  *count = n;
  return v;
}

static void gen_volumes_free(gen_vol * v, u32 n) {
  u32 i;
  Fi(n, xpar_free(v[i].name));
  xpar_free(v);
}

static bool gen_chain_names(const xpar_chain * c, const char * path) {
  u32 i;
  Fi(c->vol_count,
    if (xpar_path_same(c->vol[i].path, path)) return true);
  return false;
}

static bool gen_path_equal(const char * a, const char * b) {
  return xpar_path_same(a, b);
}

static bool gen_chain_data_names(const xpar_chain * c, const char * path) {
  u32 g;
  for (g = 0; g < c->gen_count; g++) {
    xpar_layt l;
    u32 i;
    if (!c->gen[g].layt_body ||
        xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, &l) !=
          XPAR_OK) continue;
    Fi(l.count,
      if (l.vol[i].kind == XPAR_VOL_DATA) {
            char * p = xpar_path_join(c->dir, l.vol[i].name);
            bool same = gen_path_equal(p, path);
            xpar_free(p);
            if (same) { xpar_layt_free(&l);  return true; }
          });
    xpar_layt_free(&l);
  }
  return false;
}

/*  Record superseded bare data volumes for removal after commit.  */
static void gen_maint_superseded(gen_maint * j, const xpar_chain * c,
                                 char * const * final_data, u32 data_n) {
  u32 g, k, d;
  for (g = 0; g < c->gen_count; g++) {
    xpar_layt l;
    if (!c->gen[g].layt_body ||
        xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, &l) !=
          XPAR_OK) continue;
    Fk(l.count,
      if (l.vol[k].kind == XPAR_VOL_DATA) {
            char * old = xpar_path_join(c->dir, l.vol[k].name);
            for (d = 0; d < data_n; d++)
              if (!xpar_strcmp(old, final_data[d])) break;
            if (d == data_n && gen_exists(old))
              gen_maint_add(j, XPAR_MAINT_DISCARD, old, NULL, 0);
            xpar_free(old);
          });
    xpar_layt_free(&l);
  }
}

/*  Consolidation rollback state.  */
typedef struct {
  const xpar_chain * c;
  const xpar_options * o;
  char ** backup;
  char ** data_backup;
  char ** final_data;
  char * final_base;
  u32 vol_count, data_n;
  gen_maint j;
} gen_consol_tx;

static void gen_commit_consolidation(const xpar_chain * c,
                                     const xpar_options * o,
                                     const char * stage_base,
                                     const char * final_base,
                                     const gen_plan * p,
                                     gen_consol_tx * tx) {
  gen_vol * stage, * final;
  char ** backup, ** stage_data = NULL, ** final_data = NULL;
  char ** data_backup = NULL;
  char ** stage_label = NULL, ** final_label = NULL;
  bool * published, * data_published = NULL, * data_moved = NULL;
  bool * label_published = NULL;
  gen_maint j;
  u32 ns, nf, data_n = 0, i, d, moved = 0;
  int saved = 0;
  const char * failed_from = NULL, * failed_to = NULL;

  stage = gen_volumes(o, p->recovery, stage_base, 0, &ns);
  final = gen_volumes(o, p->recovery, final_base, 0, &nf);
  xpar_assert(ns == nf);
  backup = xpar_calloc(c->vol_count ? c->vol_count : 1, sizeof *backup);
  published = xpar_calloc(nf ? nf : 1, sizeof *published);
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    int width;
    data_n = o->volumes == XPAR_VOLS_FIXED ? o->volume_count : 1;
    if (!p->geom.slice_count) data_n = 1;
    else if (data_n > p->geom.slice_count) data_n = (u32) p->geom.slice_count;
    width = xpar_digits10(data_n ? data_n - 1 : 0);
    if (width < 2) width = 2;
    stage_data = xpar_calloc(data_n, sizeof *stage_data);
    final_data = xpar_calloc(data_n, sizeof *final_data);
    data_backup = xpar_calloc(data_n, sizeof *data_backup);
    data_published = xpar_calloc(data_n, sizeof *data_published);
    data_moved = xpar_calloc(data_n, sizeof *data_moved);
    if (o->labels) {
      stage_label = xpar_calloc(data_n, sizeof *stage_label);
      final_label = xpar_calloc(data_n, sizeof *final_label);
      label_published = xpar_calloc(data_n, sizeof *label_published);
    }
    Fi(data_n,
      stage_data[i] = xpar_vname_data(stage_base, 0, i, width);
      final_data[i] = xpar_vname_data(final_base, 0, i, width);
      if (o->labels) {
        stage_label[i] = xpar_vname_label(stage_data[i]);
        final_label[i] = xpar_vname_label(final_data[i]);
      });
  }

  Fi(nf,
    if (gen_exists(final[i].name) && !gen_chain_names(c, final[i].name) &&
        !o->force) {
      u32 k;
      Fk(ns, xpar_remove(stage[k].name));
      FATAL("'%s' is not a volume of this chain; use -f to overwrite it",
            final[i].name);
    });
  Fi(data_n,
    if (gen_exists(final_data[i]) &&
        !gen_chain_data_names(c, final_data[i]) && !o->force) {
      u32 k;
      Fk(ns, xpar_remove(stage[k].name));
      Fk(data_n, xpar_remove(stage_data[k]));
      FATAL("'%s' is not a data volume of this chain; use -f to overwrite it",
            final_data[i]);
    });
  if (o->labels)
    Fi(data_n,
      if (gen_exists(final_label[i]) &&
          !gen_chain_names(c, final_label[i]) && !o->force) {
        u32 k;
        Fk(ns, xpar_remove(stage[k].name));
        Fk(data_n, xpar_remove(stage_data[k]);  xpar_remove(stage_label[k]));
        FATAL("'%s' is not a label of this chain; use -f to overwrite it",
              final_label[i]);
      });

  Fi(c->vol_count,
    backup[i] = gen_unused_path(c->vol[i].path, "xpar-old", "GCO", "BAK", i);
    if (!backup[i]) {
      u32 k;
      Fk(ns, xpar_remove(stage[k].name));
      FATAL("cannot choose a rollback name for '%s'", c->vol[i].path);
    });
  Fi(data_n,
    if (gen_exists(final_data[i])) {
        data_backup[i] = gen_unused_path(final_data[i], "xpar-old", "GCD",
                                         "BAK", i);
        if (!data_backup[i])
          FATAL("cannot choose a rollback name for '%s'", final_data[i]);
      });

  /*  Journal the transaction before its first move.  */
  xpar_memset(&j, 0, sizeof j);
  j.op = XPAR_MAINT_CONSOL;
  Fi(c->vol_count, gen_maint_add(&j, XPAR_MAINT_MOVE, c->vol[i].path, backup[i], 0));
  Fi(data_n,
    if (data_backup[i])
      gen_maint_add(&j, XPAR_MAINT_MOVE, final_data[i], data_backup[i], 0));
  Fi(nf,
    u32 k = i + 1 < nf ? i + 1 : 0;
    if (k == 0) for (d = 0; d < data_n; d++) {
      gen_maint_add(&j, XPAR_MAINT_PUBLISH, stage_data[d], final_data[d], 0);
      if (o->labels)
        gen_maint_add(&j, XPAR_MAINT_PUBLISH, stage_label[d], final_label[d],
                      0);
    }
    gen_maint_add(&j, XPAR_MAINT_PUBLISH, stage[k].name, final[k].name, 0));
  gen_maint_superseded(&j, c, final_data, data_n);
  if (!gen_maint_write(&j, final_base)) {
    gen_maint_free(&j);
    Fi(ns, xpar_remove(stage[i].name));
    Fi(data_n,
      xpar_remove(stage_data[i]);
      if (o->labels) xpar_remove(stage_label[i]));
    FATAL_IO("cannot journal consolidation of '%s'; no files moved",
             final_base);
  }

  Fi(c->vol_count,
    if (xpar_rename(c->vol[i].path, backup[i]) != 0) {
      saved = xpar_errno();
      failed_from = c->vol[i].path;  failed_to = backup[i];
      break;
    }
    moved++);
  if (moved != c->vol_count) goto rollback;
  Fi(data_n,
    if (data_backup[i]) {
        if (xpar_rename(final_data[i], data_backup[i]) != 0) {
          saved = xpar_errno();
          failed_from = final_data[i];  failed_to = data_backup[i];
          goto rollback;
        }
        data_moved[i] = true;
      });
  if (xpar_fsync_dir(final_base) != 0) { saved = xpar_errno();  goto rollback; }

  /*  As elsewhere, make the index visible last.  */
  Fi(nf,
    u32 k = i + 1 < nf ? i + 1 : 0;
    if (k == 0) {
      for (d = 0; d < data_n; d++) {
        if (xpar_rename(stage_data[d], final_data[d]) != 0) {
          saved = xpar_errno();
          failed_from = stage_data[d];  failed_to = final_data[d];
          goto rollback;
        }
        data_published[d] = true;
        if (o->labels) {
          if (xpar_rename(stage_label[d], final_label[d]) != 0) {
            saved = xpar_errno();
            failed_from = stage_label[d];  failed_to = final_label[d];
            goto rollback;
          }
          label_published[d] = true;
        }
      }
    }
    if (xpar_rename(stage[k].name, final[k].name) != 0) {
      saved = xpar_errno();
      failed_from = stage[k].name;  failed_to = final[k].name;
      goto rollback;
    }
    published[k] = true);
  if (xpar_fsync_dir(final_base) != 0) { saved = xpar_errno();  goto rollback; }

  /*  Keep backups through read-back.  */
  tx->c = c;  tx->o = o;
  tx->backup = backup;  tx->vol_count = c->vol_count;
  tx->data_backup = data_backup;  tx->final_data = final_data;
  tx->data_n = data_n;
  tx->final_base = xpar_strdup(final_base);
  tx->j = j;  xpar_memset(&j, 0, sizeof j);
  backup = NULL;  data_backup = NULL;  final_data = NULL;
  goto done;

rollback:
  Fi(nf, if (published[i]) xpar_remove(final[i].name));
  Fi(ns, xpar_remove(stage[i].name));
  Fi(data_n,
    if (data_published[i]) xpar_remove(final_data[i]);
    else xpar_remove(stage_data[i]);
    if (o->labels) {
      if (label_published[i]) xpar_remove(final_label[i]);
      else xpar_remove(stage_label[i]);
    });
  for (i = data_n; i-- > 0;)
    if (data_moved[i] &&
        xpar_rename(data_backup[i], final_data[i]) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot restore '%s': %s; the "
                   "original remains at '%s'\n", final_data[i],
                   xpar_strerror(xpar_errno()), data_backup[i]);
  while (moved) {
    moved--;
    if (xpar_rename(backup[moved], c->vol[moved].path) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot restore '%s': %s; the "
                   "original remains at '%s'\n", c->vol[moved].path,
                   xpar_strerror(xpar_errno()), backup[moved]);
  }
  if (xpar_fsync_dir(final_base) != 0)
    xpar_fprintf(xpar_stderr, "xpar: warning: cannot sync the directory "
                 "after rollback: %s\n", xpar_strerror(xpar_errno()));
  gen_maint_done(&j);
  if (failed_from)
    FATAL_IO("cannot rename '%s' to '%s' while publishing the consolidated "
             "set: %s", failed_from, failed_to, xpar_strerror(saved));
  FATAL_IO("cannot publish the consolidated set: %s", xpar_strerror(saved));

done:
  Fi(data_n,
    xpar_free(stage_data[i]);
    if (o->labels)
      { xpar_free(stage_label[i]);  xpar_free(final_label[i]); });
  xpar_free(published);
  xpar_free(stage_data);
  xpar_free(data_published);  xpar_free(data_moved);
  xpar_free(stage_label);  xpar_free(final_label);  xpar_free(label_published);
  gen_volumes_free(stage, ns);  gen_volumes_free(final, nf);
}

/*  Drop the old chain once the consolidated set has been read back.  */
static void gen_consol_finish(gen_consol_tx * tx) {
  u32 i;
  if (!tx->final_base) return;
  Fi(tx->vol_count,
    if (tx->backup[i] && xpar_remove(tx->backup[i]) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot remove rollback volume '%s': "
                   "%s\n", tx->backup[i], xpar_strerror(xpar_errno()));
    xpar_free(tx->backup[i]));
  Fi(tx->data_n,
    if (tx->data_backup[i] && xpar_remove(tx->data_backup[i]) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot remove rollback volume '%s': "
                   "%s\n", tx->data_backup[i], xpar_strerror(xpar_errno()));
    xpar_free(tx->data_backup[i]));
  /*  Remove superseded bare data volumes, which are absent from c->vol.  */
  Fi(tx->c->gen_count,
    xpar_layt l;
    u32 k;
    if (!tx->c->gen[i].layt_body ||
        xpar_layt_read(tx->c->gen[i].layt_body, tx->c->gen[i].layt_len,
                       &l) != XPAR_OK) continue;
    Fk(l.count,
      if (l.vol[k].kind == XPAR_VOL_DATA) {
            char * old = xpar_path_join(tx->c->dir, l.vol[k].name);
            u32 d;
            for (d = 0; d < tx->data_n; d++)
              if (!xpar_strcmp(old, tx->final_data[d])) break;
            if (d == tx->data_n && gen_exists(old) && xpar_remove(old) != 0)
              xpar_fprintf(xpar_stderr, "xpar: cannot remove the superseded data "
                           "volume '%s': %s\n", old, xpar_strerror(xpar_errno()));
            xpar_free(old);
          });
    xpar_layt_free(&l));
  if (xpar_fsync_dir(tx->final_base) != 0)
    xpar_fprintf(xpar_stderr, "xpar: cannot sync the directory after "
                 "removing rollback volumes: %s\n",
                 xpar_strerror(xpar_errno()));
  Fi(tx->data_n, xpar_free(tx->final_data[i]));
  xpar_free(tx->backup);  xpar_free(tx->data_backup);
  xpar_free(tx->final_data);  xpar_free(tx->final_base);
  gen_maint_done(&tx->j);
  xpar_memset(tx, 0, sizeof *tx);
}

static void gen_layt_build(xpar_layt * l, const gen_vol * v, u32 n) {
  u32 i;
  l->this_volume = XPAR_VOL_STANDALONE;
  l->count = n;
  l->vol = xpar_calloc(n, sizeof *l->vol);
  Fi(n,
    char * dir;  char * name;
    l->vol[i].kind = v[i].is_index ? (u8) XPAR_VOL_INDEX
                                   : (u8) XPAR_VOL_RECOVERY;
    l->vol[i].recovery_first = (u32) v[i].first;
    l->vol[i].byte_length    = v[i].count;
    gen_split_path(v[i].name, &dir, &name);
    l->vol[i].name = name;
    xpar_free(dir));
}

typedef struct {
  const xpar_options * o;
  xpar_manifest *      m;
  const bool *         owned;     /*  Per entry: owned by this generation.  */
  const u8 **          inh_body;  /*  Per entry: the ancestor's FILE body.  */
  const sz *           inh_len;
  u32                  generation;
  u64                  stream_base;
  const u8 *           parent_set_id;
  const char *         base;      /*  Output base name.  */
  const char *         layout_base; /*  Final names when output is staged.  */
  bool                 quiet;
  bool                 auth_only;
  u8                   set_id[XPAR_SET_ID_LEN];
  gen_plan             plan;
  u32                  volumes;
  char *               index_path;
} gen_write_req;

static void gen_wropt(const xpar_options * o, xpar_wropt * w) {
  xpar_memset(w, 0, sizeof *w);
  w->reproducible = o->reproducible;
  w->keep_mtime = (o->preserve_explicit & XPAR_PRES_MTIME) != 0;
  w->keep_atime = (o->preserve_explicit & XPAR_PRES_ATIME) != 0;
  w->keep_ctime = (o->preserve_explicit & XPAR_PRES_CTIME) != 0;
  w->keep_btime = (o->preserve_explicit & XPAR_PRES_BTIME) != 0;
  w->keep_posix = (o->preserve_explicit &
                   (XPAR_PRES_OWNER | XPAR_PRES_XATTR)) != 0;
}

static void gen_auth_only_hashes(xpar_manifest * m, const bool * owned,
                                 const xpar_key * key) {
  xpar_nameidx ix;
  u32 i;
  xpar_nameidx_build(m, &ix);
  Fi(m->count,
    xpar_entry * e = &m->entry[i];
    xpar_blake3_t h;
    if (!owned[i] || e->entry_type == XPAR_ENTRY_HARDLINK) continue;
    xpar_blake3_init_keyed(&h, key->k_file);
    if (e->entry_type == XPAR_ENTRY_SYMLINK)
      xpar_blake3_update(&h, e->extra, e->extra_len);
    else if (e->entry_type == XPAR_ENTRY_REGULAR) {
      xpar_file * f = xpar_open(m->source[i], XPAR_O_RDONLY);
      u8 buf[16384];
      if (!f) FATAL_PERROR(m->source[i]);
      for (;;) {
        sz n = xpar_read(f, buf, sizeof buf);
        if (n) xpar_blake3_update(&h, buf, n);
        if (n < sizeof buf) {
          if (xpar_error(f)) FATAL_IO("cannot read '%s'", m->source[i]);
          if (xpar_eof(f) || !n) break;
        }
      }
      xpar_xclose(f);
      xpar_secure_zero(buf, sizeof buf);
    }
    xpar_blake3_final(&h, e->content_hash, 32);
    xpar_secure_zero(&h, sizeof h));
  Fi(m->count,
    if (owned[i]) {
      xpar_entry * e = &m->entry[i];
      if (e->entry_type == XPAR_ENTRY_HARDLINK) {
        i64 t = xpar_link_target(m, &ix, i);
        FATAL_UNLESS(t >= 0,
                     "hard-link entry '%.*s' has no canonical target",
                     (int) e->name_len, e->name);
        xpar_memcpy(e->content_hash, m->entry[t].content_hash, 32);
      }
    });
  xpar_nameidx_free(&ix);
}

/*  Hash SETD and FILE bodies exactly as written, including packet padding.  */
static void gen_set_id(const xpar_setd * sd, const xpar_manifest * m,
                       const bool * owned, const u8 ** inh_body,
                       const sz * inh_len, const xpar_wropt * w,
                       const xpar_key * key, u8 * out) {
  static const u8 zero[XPAR_SET_ID_LEN] = { 0 };
  xpar_set_id_ctx ctx;  xpar_buf b;  u32 i;
  xpar_buf_init(&b);
  xpar_setd_write(&b, sd, zero, NULL);
  xpar_set_id_begin(&ctx, key ? key->k_set : NULL, b.data + XPAR_PKT_HDR,
                    b.len - XPAR_PKT_HDR);
  xpar_buf_free(&b);
  Fi(m->count,
    if (owned[i]) {
      xpar_buf e;
      xpar_buf_init(&e);
      xpar_entry_write(&e, &m->entry[i], zero, NULL, w);
      xpar_set_id_update(&ctx, e.data + XPAR_PKT_HDR, e.len - XPAR_PKT_HDR);
      xpar_buf_free(&e);
    } else xpar_set_id_update(&ctx, inh_body[i], inh_len[i]));
  xpar_set_id_final(&ctx, out);
}

static void gen_crit_group(xpar_buf * out, const xpar_setd * sd,
                           const xpar_manifest * m, const bool * owned,
                           const gen_tables * t, const xpar_layt * layt,
                           u32 this_vol, const u8 * set_id,
                           const xpar_wropt * w, const xpar_key * key,
                           const xpar_auth * auth, bool auth_only) {
  xpar_layt l = *layt;  u32 i;
  l.this_volume = this_vol;
  xpar_setd_write(out, sd, set_id, key);
  Fi(m->count, if (owned[i]) xpar_entry_write(out, &m->entry[i], set_id, key, w));
  if (m->posix_count && !w->reproducible)
    xpar_posx_write_all(out, m->posix, m->posix_count, set_id, key);
  if (!auth_only && sd->data_slice_count && t->slice_crc)
    xpar_slcr_write_all(out, t->slice_crc, sd->data_slice_count, set_id,
                        key);
  if (auth) xpar_auth_write(out, auth, set_id, key);
  xpar_layt_write(out, &l, set_id, key);
}

typedef struct {
  char * stage;
  const char * final;
  bool replace;
} gen_addrec_file;

static void gen_addrec_discard(gen_addrec_file * files, u32 count) {
  u32 i;
  Fi(count,
    if (files[i].stage) xpar_remove(files[i].stage);
    xpar_free(files[i].stage));
  xpar_free(files);
}

/*  Publish staged files with rollback.  */
static void gen_addrec_publish(gen_addrec_file * files, u32 count) {
  char ** backup = xpar_calloc(count ? count : 1, sizeof *backup);
  const char * failed = NULL;
  int err = 0;
  u32 i, done = 0;
  Fi(count,
    if (!files[i].replace && gen_exists(files[i].final)) {
      const char * name = files[i].final;
      xpar_free(backup);
      gen_addrec_discard(files, count);
      FATAL("'%s' exists; use -f to overwrite it", name);
    });
  Fi(count,
    if (gen_exists(files[i].final)) {
      backup[i] = gen_unused_path(files[i].final, "xpar-old", "GAD", "BAK",
                                  i);
      if (!backup[i] || xpar_keep_aside(files[i].final, backup[i]) != 0) {
        err = xpar_errno();  failed = files[i].final;
        xpar_free(backup[i]);  backup[i] = NULL;
        break;
      }
    }
    if (xpar_rename(files[i].stage, files[i].final) != 0) {
      err = xpar_errno();  failed = files[i].final;
      break;
    }
    xpar_free(files[i].stage);  files[i].stage = NULL;
    done = i + 1);
  if (!failed)
    Fi(count,
      if (xpar_fsync_dir(files[i].final) != 0) {
        err = xpar_errno();  failed = files[i].final;
        break;
      });
  if (failed) {
    char * name = xpar_strdup(failed);
    /*  Restore backups in reverse order.  */
    for (i = count; i > 0; i--) {
      u32 k = i - 1;
      if (backup[k]) {
        if (xpar_put_back(files[k].final, backup[k]) != 0) {
          int e = xpar_errno();
          if (gen_exists(files[k].final))
            xpar_fprintf(xpar_stderr, "xpar: warning: rollback copy of '%s' "
                         "remains at '%s': %s\n", files[k].final, backup[k],
                         xpar_strerror(e));
          else
            xpar_fprintf(xpar_stderr, "xpar: cannot restore '%s': %s; the "
                         "original remains at '%s'\n", files[k].final,
                         xpar_strerror(e), backup[k]);
        }
      } else if (k < done && xpar_remove(files[k].final) != 0)
        xpar_fprintf(xpar_stderr, "xpar: warning: unreferenced '%s' "
                     "remains: %s\n", files[k].final,
                     xpar_strerror(xpar_errno()));
      xpar_free(backup[k]);
    }
    if (count && xpar_fsync_dir(files[0].final) != 0)
      xpar_fprintf(xpar_stderr, "xpar: warning: cannot sync the directory "
                   "after rollback: %s\n", xpar_strerror(xpar_errno()));
    xpar_free(backup);
    gen_addrec_discard(files, count);
    FATAL_IO("cannot publish '%s': %s", name,
             xpar_strerror(err));
  }
  Fi(count,
    if (backup[i] && xpar_remove(backup[i]) != 0)
      xpar_fprintf(xpar_stderr, "xpar: warning: old '%s' remains "
                   "at '%s'\n", files[i].final, backup[i]);
    xpar_free(backup[i]));
  xpar_free(backup);
  xpar_free(files);
}

/*  Order replacements first, the index last, then new files.  */
static void gen_addrec_order(gen_addrec_file * files, u32 count, u32 fresh,
                             const char * index_path) {
  gen_addrec_file * out = xpar_calloc(count ? count : 1, sizeof *out);
  u32 i, n = 0;
  for (i = fresh; i < count; i++)
    if (!index_path || !xpar_path_same(files[i].final, index_path))
      out[n++] = files[i];
  for (i = fresh; i < count; i++)
    if (index_path && xpar_path_same(files[i].final, index_path))
      out[n++] = files[i];
  Fi(fresh, out[n++] = files[i]);
  xpar_memcpy(files, out, (sz) count * sizeof *out);
  xpar_free(out);
}

static void gen_write_set(gen_write_req * rq) {
  const xpar_options * o = rq->o;
  xpar_manifest * m = rq->m;
  gen_tables t;
  xpar_setd sd;
  xpar_wropt w;
  xpar_layt layt;
  gen_vol * vol;
  gen_vol * layout_vol = NULL;
  char ** data_name = NULL;
  char ** layout_data_name = NULL;
  char ** label_name = NULL;
  xpar_buf probe;
  xpar_progress_t prog;
  xpar_json pjs;
  gen_addrec_file * pub = NULL;
  u32 nvol, data_n = 0, i, j, pub_n = 0;
  u64 crit_bytes;
  u8 * rec_scratch = NULL;
  xpar_key key;
  u8 master[XPAR_BLAKE3_KEY_LEN];
  xpar_auth auth;
  bool keyed = false;
  const xpar_key * kp = NULL;
  u8 tag_len = (u8) (o->slice_tag < 0 ? 8 : o->slice_tag);
  bool wrap = o->armour == XPAR_ARMOUR_ALL &&
              o->layout != XPAR_LAYOUT_ARMOURED;
  xpar_armour * ra = NULL;

  xpar_memset(&key, 0, sizeof key);
  xpar_memset(master, 0, sizeof master);
  xpar_memset(&auth, 0, sizeof auth);
  if (o->auth_key) {
    xpar_keyfile_load_or_die(o->auth_key, &key, master);
    keyed = true;  kp = &key;  tag_len = 16;
    auth.kdf_id = 0;  auth.slice_tag_keyed = 1;
    auth.packet_tag_keyed = 1;
    auth.unkeyed_retained = !rq->auth_only;
    xpar_key_check(auth.key_check, master);
    if (rq->auth_only) gen_auth_only_hashes(m, rq->owned, kp);
    Fi(m->count,
      if (rq->owned[i])
        xpar_file_id(&m->entry[i], key.k_file, m->entry[i].file_id));
  }
  FATAL_UNLESS(o->align != XPAR_ALIGN_1K || tag_len != 0,
               "--align=1k needs slice tags; choose --slice-tag=8 or 16");
  /*  Subtree tags require a 1 KiB-aligned generation base.  */
  FATAL_UNLESS(o->align != XPAR_ALIGN_1K ||
               rq->stream_base % XPAR_BLAKE3_CHUNK_LEN == 0,
               "--align=1k requires a 1 KiB-aligned stream base; offset "
               "%" PRIu64 " is %" PRIu64 " bytes past one",
               rq->stream_base,
               rq->stream_base % XPAR_BLAKE3_CHUNK_LEN);

  gen_wropt(o, &w);
  gen_choose(o, m->stream_length, &rq->plan);

  vol = gen_volumes(o, rq->plan.recovery, rq->base, rq->generation, &nvol);
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    int width;
    data_n = o->volumes == XPAR_VOLS_FIXED ? o->volume_count : 1;
    if (!rq->plan.geom.slice_count) data_n = 1;
    else if (data_n > rq->plan.geom.slice_count)
      data_n = (u32) rq->plan.geom.slice_count;
    width = xpar_digits10(data_n ? data_n - 1 : 0);
    if (width < 2) width = 2;
    data_name = xpar_calloc(data_n, sizeof *data_name);
    layout_data_name = xpar_calloc(data_n, sizeof *layout_data_name);
    label_name = xpar_calloc(data_n, sizeof *label_name);
    Fi(data_n,
      data_name[i] = xpar_vname_data(rq->base, rq->generation, i, width);
      layout_data_name[i] = xpar_vname_data(
        rq->layout_base ? rq->layout_base : rq->base,
        rq->generation, i, width);
      if (o->labels)
        label_name[i] = xpar_vname_label(data_name[i]));
  }
  if (!o->force)
    Fi(nvol,
      if (gen_exists(vol[i].name))
        FATAL("'%s' exists; use -f to overwrite it; nothing was written",
              vol[i].name));
  if (!o->force)
    Fi(data_n,
      if (gen_exists(data_name[i]))
        FATAL("'%s' exists; use -f to overwrite it; nothing was written",
              data_name[i]);
      if (label_name[i] && gen_exists(label_name[i]))
        FATAL("'%s' exists; use -f to overwrite it; nothing was written",
              label_name[i]));

  xpar_progress_init(&prog, xpar_progress_wanted(o),
                     rq->plan.geom.slice_count * rq->plan.geom.slice_size,
                     "encoding");
  xpar_json_init(&pjs, xpar_stdout, o->json);
  if (o->json) xpar_progress_sink(&prog, xpar_json_progress_sink, &pjs);
  gen_encode(m, &rq->plan, tag_len, o->memory, rq->base, kp, NULL, NULL,
             &t, &prog);
  if (t.rec_spill) rec_scratch = xpar_alloc_raw((sz) t.rec_z);
  xpar_progress_end(&prog);

  xpar_memset(&sd, 0, sizeof sd);
  sd.slice_size         = rq->plan.geom.slice_size;
  sd.data_slice_count   = rq->plan.geom.slice_count;
  sd.stream_length      = m->stream_length;
  sd.file_count         = m->count;
  sd.field_log2         = rq->plan.field_log2;
  sd.codec              = rq->plan.codec;
  sd.recovery_axis_log2 = rq->plan.axis;
  sd.layout             = (u8) o->layout;
  sd.align              = (u8) o->align;
  sd.slice_tag_len      = tag_len;
  sd.dedup_level        = m->dedup_level;
  if (o->align == XPAR_ALIGN_1K)
    sd.required_features |= XPAR_FEAT_B3_SUBTREE;
  sd.cell_bytes         = rq->plan.geom.cell_bytes;
  sd.generation         = rq->generation;
  sd.posix_record_count = w.reproducible && !w.keep_posix
                            ? 0 : m->posix_count;
  sd.stream_base        = rq->stream_base;
  if (rq->parent_set_id)
    xpar_memcpy(sd.parent_set_id, rq->parent_set_id, XPAR_SET_ID_LEN);
  sd.file_id = (u8 (*)[XPAR_SET_ID_LEN])
                 xpar_calloc(m->count ? m->count : 1, XPAR_SET_ID_LEN);
  Fi(m->count, xpar_memcpy(sd.file_id[i], m->entry[i].file_id, XPAR_SET_ID_LEN));

  gen_set_id(&sd, m, rq->owned, rq->inh_body, rq->inh_len, &w, kp,
             rq->set_id);

  if (rq->layout_base) {
    u32 layout_n;
    layout_vol = gen_volumes(o, rq->plan.recovery, rq->layout_base,
                             rq->generation, &layout_n);
    xpar_assert(layout_n == nvol);
    gen_layt_build(&layt, layout_vol, layout_n);
  } else gen_layt_build(&layt, vol, nvol);
  /*  Bit 0 identifies the armoured index entry.  */
  Fi(layt.count,
    layt.vol[i].vflags = layt.vol[i].kind == XPAR_VOL_INDEX &&
     o->layout == XPAR_LAYOUT_ARMOURED);
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    /*  Spread the remainder across the leading volumes.  */
    u64 base = data_n ? rq->plan.geom.slice_count / data_n : 0;
    u64 rem  = data_n ? rq->plan.geom.slice_count % data_n : 0;
    u64 slice = 0;
    layt.vol = xpar_realloc(
      layt.vol, (sz) (layt.count + data_n) * sizeof *layt.vol);
    Fi(data_n,
      char * dir, * name;
      u64 count = base + (i < rem ? 1 : 0);
      u64 off = slice * rq->plan.geom.slice_size;
      u64 len = MIN(count * rq->plan.geom.slice_size,
                    m->stream_length - off);
      xpar_memset(&layt.vol[layt.count], 0, sizeof layt.vol[layt.count]);
      layt.vol[layt.count].kind = XPAR_VOL_DATA;
      /*  LAYT tiles this generation's local [0,L) stream; SETD.stream_base
         places that stream in the chain-wide address space.  */
      layt.vol[layt.count].stream_offset = off;
      layt.vol[layt.count].byte_length = len;
      layt.vol[layt.count].vol_tag = gen_stream_tag(m, off, len);
      gen_split_path(layout_data_name[i], &dir, &name);
      layt.vol[layt.count].name = name;
      xpar_free(dir);
      layt.count++;  slice += count);
  }

  xpar_buf_init(&probe);
  gen_crit_group(&probe, &sd, m, rq->owned, &t, &layt, 0, rq->set_id, &w,
                 kp, keyed ? &auth : NULL, rq->auth_only);
  crit_bytes = probe.len;
  if (o->armour != XPAR_ARMOUR_NONE) {
    xpar_buf a;
    xpar_buf_init(&a);
    gen_armour_pack(&a, o, probe.data, probe.len, rq->set_id, kp);
    crit_bytes = a.len;
    xpar_buf_free(&a);
  }
  xpar_buf_free(&probe);

  /*  Stage the generation before publishing its index last.  */
  pub = xpar_calloc(nvol + 2 * data_n + 1, sizeof *pub);

  if (o->layout == XPAR_LAYOUT_ARMOURED) {
    xpar_armour_params ap;
    xpar_armour * a;
    xpar_buf head;
    xpar_volh vh;
    char * tmp;
    const char * why;

    gen_armour_params(o, &ap);
    why = xpar_armour_check(&ap);
    if (why) FATAL("invalid armour parameters: %s", why);
    a = xpar_armour_new(&ap);
    xpar_buf_init(&head);
    xpar_memset(&vh, 0, sizeof vh);
    vh.volume_index = XPAR_VOL_STANDALONE;
    vh.volume_kind = XPAR_VOL_INDEX;
    xpar_volh_write(&head, &vh, rq->set_id, kp);
    layt.this_volume = XPAR_VOL_STANDALONE;
    gen_crit_group(&head, &sd, m, rq->owned, &t, &layt,
                   XPAR_VOL_STANDALONE, rq->set_id, &w, kp,
                   keyed ? &auth : NULL, rq->auth_only);
    tmp = gen_stage_arm_archive(vol[0].name, &ap, m, &rq->plan, &t,
                                rec_scratch, &head, rq->set_id, kp, &w,
                                NULL, NULL);
    pub[pub_n].stage = tmp;  pub[pub_n].final = vol[0].name;
    pub[pub_n].replace = o->force;  pub_n++;
    xpar_buf_free(&head);
    xpar_armour_free(a);
  } else {
    if (wrap) ra = gen_rcvs_armour(o, rq->plan.geom.slice_size, NULL);
    /*  Recovery volumes are published before the index. A reader discovers a
        generation through its index, so interruption leaves only unreferenced
        recovery volumes rather than a visible index naming absent volumes.  */
    Fj(nvol,
      xpar_buf out, group;
      xpar_volh vh;
      u64 payload, k;
      bool carry;
      i = j + 1 < nvol ? j + 1 : 0;
      payload = vol[i].count * rq->plan.geom.slice_size;
      carry = vol[i].is_index ||
              xpar_replicate_here(crit_bytes, payload, i - 1, nvol - 1);

      xpar_buf_init(&out);
      xpar_memset(&vh, 0, sizeof vh);
      /*  LAYT/VOLH indices include the index volume; replication indices do
          not.  */
      vh.volume_index = vol[i].is_index ? XPAR_VOL_STANDALONE : i;
      vh.volume_kind  = vol[i].is_index ? XPAR_VOL_INDEX : XPAR_VOL_RECOVERY;
      xpar_volh_write(&out, &vh, rq->set_id, kp);

      if (carry) {
        xpar_buf_init(&group);
        gen_crit_group(&group, &sd, m, rq->owned, &t, &layt,
                       vol[i].is_index ? XPAR_VOL_STANDALONE : i, rq->set_id,
                       &w, kp, keyed ? &auth : NULL, rq->auth_only);
        if (o->armour != XPAR_ARMOUR_NONE)
          gen_armour_pack(&out, o, group.data, group.len, rq->set_id, kp);
        else
          xpar_buf_put(&out, group.data, group.len);
        xpar_buf_free(&group);
      }

      if (vol[i].is_index || i == 1)
        gen_emit_tables(&out, o, wrap, &t, &rq->plan, tag_len, rq->set_id, kp);
      if (vol[i].count) {
        xpar_buf tail;
        char * tmp;
        xpar_file * f = gen_stage_open(vol[i].name, &tmp);
        xpar_xwrite(f, out.data, out.len);
        xpar_buf_free(&out);
        Fk(vol[i].count,
          xpar_buf pkt;
          u64 e = vol[i].first + k;
          const u8 * rec = gen_rec_get(&t, e, rec_scratch);
          xpar_buf_init(&pkt);
          gen_rcvs_emit(&pkt, ra, e, rec, (sz) rq->plan.geom.slice_size,
                        rq->set_id, kp);
          xpar_xwrite(f, pkt.data, pkt.len);
          xpar_buf_free(&pkt));
        xpar_buf_init(&tail);
        xpar_crtr_write(&tail, "xpar " PACKAGE_VERSION, rq->set_id, kp, &w);
        xpar_xwrite(f, tail.data, tail.len);
        xpar_buf_free(&tail);
        if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
          FATAL_IO("cannot flush temporary volume '%s'", tmp);
        xpar_xclose(f);
        pub[pub_n].stage = tmp;  pub[pub_n].final = vol[i].name;
        pub[pub_n].replace = o->force;  pub_n++;
      } else {
        xpar_crtr_write(&out, "xpar " PACKAGE_VERSION, rq->set_id, kp, &w);
        pub[pub_n].stage = gen_stage_whole(vol[i].name, out.data, out.len);
        pub[pub_n].final = vol[i].name;
        pub[pub_n].replace = o->force;  pub_n++;
        xpar_buf_free(&out);
      }
      if (!rq->quiet && o->verbose)
        xpar_fprintf(xpar_stderr, "xpar: wrote %s\n", vol[i].name));
    if (o->layout == XPAR_LAYOUT_SPLIT) {
      u32 data_first = layt.count - data_n;
      Fi(data_n,
        const xpar_vol * dv = &layt.vol[data_first + i];
        pub[pub_n].stage = gen_stage_data_range(m, data_name[i],
                                                dv->stream_offset,
                                                dv->byte_length, o->force);
        pub[pub_n].final = data_name[i];
        pub[pub_n].replace = o->force;  pub_n++;
        if (label_name[i]) {
          xpar_buf out, group;
          xpar_volh vh;
          xpar_buf_init(&out);
          xpar_memset(&vh, 0, sizeof vh);
          vh.volume_index = data_first + i;
          vh.volume_kind = XPAR_VOL_DATA;
          xpar_volh_write(&out, &vh, rq->set_id, kp);
          xpar_buf_init(&group);
          gen_crit_group(&group, &sd, m, rq->owned, &t, &layt,
                         data_first + i, rq->set_id, &w, kp,
                         keyed ? &auth : NULL, rq->auth_only);
          if (o->armour != XPAR_ARMOUR_NONE)
            gen_armour_pack(&out, o, group.data, group.len, rq->set_id, kp);
          else
            xpar_buf_put(&out, group.data, group.len);
          xpar_buf_free(&group);
          xpar_crtr_write(&out, "xpar " PACKAGE_VERSION, rq->set_id, kp, &w);
          pub[pub_n].stage = gen_stage_whole(label_name[i], out.data, out.len);
          pub[pub_n].final = label_name[i];
          pub[pub_n].replace = o->force;  pub_n++;
          xpar_buf_free(&out);
        });
    }
  }
  /*  Publish data volumes and labels before the index that names them.  */
  gen_addrec_order(pub, pub_n, 0, vol[0].name);
  gen_addrec_publish(pub, pub_n);
  pub = NULL;
  rq->volumes = nvol;
  rq->index_path = xpar_strdup(vol[0].name);

  xpar_layt_free(&layt);
  if (layout_vol) gen_volumes_free(layout_vol, nvol);
  gen_volumes_free(vol, nvol);
  gen_tables_free(&t);
  Fi(data_n,
    xpar_free(data_name[i]);  xpar_free(layout_data_name[i]);
    xpar_free(label_name[i]));
  xpar_free(data_name);  xpar_free(layout_data_name);  xpar_free(label_name);
  xpar_free(rec_scratch);
  xpar_free(sd.file_id);
  if (ra) xpar_armour_free(ra);
  xpar_key_forget(&key, master);
}

/*  Comparing an entry against the disk.  */

static void gen_entry_copy(xpar_entry * d, const xpar_entry * s) {
  *d = *s;
  d->name    = s->name_len ? xpar_malloc(s->name_len) : NULL;
  if (s->name_len) xpar_memcpy(d->name, s->name, s->name_len);
  d->extra   = s->extra_len ? xpar_malloc(s->extra_len) : NULL;
  if (s->extra_len) xpar_memcpy(d->extra, s->extra, s->extra_len);
  d->extents = s->extent_count
                 ? xpar_malloc((sz) s->extent_count *
                               sizeof *d->extents) : NULL;
  if (s->extent_count)
    xpar_memcpy(d->extents, s->extents,
                (sz) s->extent_count * sizeof *d->extents);
}

/*  Same name, same bytes: only the metadata moved.  */
static bool gen_content_same(const xpar_entry * a, const xpar_entry * b) {
  return a->entry_type == b->entry_type && a->length == b->length &&
         !xpar_memcmp(a->content_hash, b->content_hash, 32) &&
         !xpar_memcmp(a->prefix_hash, b->prefix_hash, 16);
}

/*  Entry equality covers every FILE field, including metadata changes.  */
static bool gen_entry_same(const xpar_entry * a, const xpar_entry * b,
                           const xpar_posix_rec * ta, u32 na,
                           const xpar_posix_rec * tb, u32 nb) {
  if (a->entry_type != b->entry_type || a->length != b->length) return false;
  if (xpar_memcmp(a->content_hash, b->content_hash, 32)) return false;
  if (a->mode != b->mode || a->attrs != b->attrs) return false;
  if (a->mtime_ns != b->mtime_ns || a->atime_ns != b->atime_ns) return false;
  if (a->ctime_ns != b->ctime_ns || a->btime_ns != b->btime_ns) return false;
  if (a->extra_len != b->extra_len) return false;
  if (a->extra_len && xpar_memcmp(a->extra, b->extra, a->extra_len))
    return false;
  if ((a->posix_index == XPAR_ABSENT_U32) !=
      (b->posix_index == XPAR_ABSENT_U32)) return false;
  if (a->posix_index != XPAR_ABSENT_U32) {
    if (a->posix_index >= na || b->posix_index >= nb) return false;
    if (!xpar_posix_equal(&ta[a->posix_index], &tb[b->posix_index]))
      return false;
  }
  return true;
}

/*  Resolve stored names against --base or the set directory.  */
static char * gen_entry_path(const xpar_options * o, const xpar_entry * e) {
  char * p, * dir, * leaf, * name;
  name = xpar_strndup(e->name, e->name_len);
  if (o->base_dir) { p = xpar_path_join(o->base_dir, name);  xpar_free(name);  return p; }
  gen_split_path(o->set, &dir, &leaf);
  p = xpar_path_join(dir, name);
  xpar_free(dir);
  xpar_free(leaf);
  xpar_free(name);
  return p;
}

/*  Rebuild a rescanned inherited entry with manifest.c's canonical fields.  */
static bool gen_refresh(xpar_entry * e, const char * path,
                        const xpar_options * o, u32 caps, bool * warn_posix,
                        const xpar_key * key, bool auth_only) {
  xpar_stat_t st;
  xpar_blake3_t h, prefix;
  u32 keep = o->reproducible
               ? o->preserve & o->preserve_explicit : o->preserve;

  if (xpar_lstat(path, &st) != 0) return false;
  xpar_free(e->extents);  e->extents = NULL;  e->extent_count = 0;
  e->mode = XPAR_ABSENT_U32;
  e->mtime_ns = e->atime_ns = e->ctime_ns = e->btime_ns = XPAR_ABSENT_TIME;
  if (st.mode != XPAR_MODE_NONE && (keep & XPAR_PRES_MODE))
    e->mode = st.mode & XPAR_MODE_PERM;
  if (keep & XPAR_PRES_MTIME) e->mtime_ns = st.mtime_ns;
  if (keep & XPAR_PRES_ATIME) e->atime_ns = st.atime_ns;
  if (keep & XPAR_PRES_CTIME) e->ctime_ns = st.ctime_ns;
  if (keep & XPAR_PRES_BTIME) e->btime_ns = st.btime_ns;
  e->attrs = (u16) ((caps & XPAR_FS_FATATTR) ? st.attrs : 0);
  if (st.mode != XPAR_MODE_NONE) {
    if (st.mode & 0111U) e->attrs |= XPAR_ATTR_EXEC;
    if (st.mode & (XPAR_MODE_SETUID | XPAR_MODE_SETGID | XPAR_MODE_STICKY))
      e->attrs |= XPAR_ATTR_SETID;
  }
  if (!xpar_utf8_valid((const u8 *) e->name, e->name_len))
    e->attrs |= XPAR_ATTR_RAWNAME;
  if ((keep & (XPAR_PRES_OWNER | XPAR_PRES_XATTR | XPAR_PRES_XATTR_ALL)) &&
      e->posix_index != XPAR_ABSENT_U32)
    *warn_posix = true;
  e->posix_index = XPAR_ABSENT_U32;

  if (st.is_dir) {
    e->entry_type = XPAR_ENTRY_DIR;  e->length = 0;
    if (auth_only) xpar_blake3_init_keyed(&h, key->k_file);
    else           xpar_blake3_init(&h);
    xpar_blake3_final(&h, e->content_hash, 32);
    xpar_blake3_init(&prefix);
    xpar_blake3_final(&prefix, e->prefix_hash, 16);
  } else if (st.is_symlink) {
    u32 n;
    char * buf = xpar_read_symlink(path, &n);
    if (!buf) return false;
    e->entry_type = XPAR_ENTRY_SYMLINK;  e->length = 0;
    xpar_free(e->extra);
    e->extra     = (u8 *) buf;
    e->extra_len = n;
    if (auth_only) xpar_blake3_init_keyed(&h, key->k_file);
    else           xpar_blake3_init(&h);
    if (n) xpar_blake3_update(&h, buf, n);
    xpar_blake3_final(&h, e->content_hash, 32);
    xpar_blake3_init(&prefix);
    xpar_blake3_final(&prefix, e->prefix_hash, 16);
  } else if (st.is_regular) {
    xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
    u8 * buf;  sz got;  u64 total;
    if (!f) return false;
    buf = xpar_alloc_raw(1U << 16);
    if (auth_only) xpar_blake3_init_keyed(&h, key->k_file);
    else           xpar_blake3_init(&h);
    xpar_blake3_init(&prefix);
    got = xpar_xread(f, buf, 16384);
    if (got) { xpar_blake3_update(&h, buf, got);  xpar_blake3_update(&prefix, buf, got); }
    total = got;
    xpar_blake3_final(&prefix, e->prefix_hash, 16);
    while ((got = xpar_xread(f, buf, 1U << 16)) > 0)
      { xpar_blake3_update(&h, buf, got);  total += got; }
    xpar_blake3_final(&h, e->content_hash, 32);
    xpar_free(buf);
    xpar_xclose(f);
    e->entry_type = XPAR_ENTRY_REGULAR;
    e->length     = total;
  } else return false;
  xpar_file_id(e, key ? key->k_file : NULL, e->file_id);
  return true;
}

/*  Whether a manifest fault must be rejected before publishing.  */
static bool gen_name_fault(xpar_mf_status s) {
  switch (s) {
    case XPAR_MF_PATH:
    case XPAR_MF_DUP_NAME:
    case XPAR_MF_TYPE:
    case XPAR_MF_TYPE_LENGTH:
    case XPAR_MF_TYPE_EXTENTS:
    case XPAR_MF_TYPE_EXTRA:
    case XPAR_MF_LINK_MISSING:
    case XPAR_MF_LINK_CHAIN:
    case XPAR_MF_LINK_SELF:  return true;
    default:                 return false;
  }
}

/*  Reject unstorable manifest names before writing volumes.  */
static void gen_check_manifest(const xpar_manifest * m) {
  xpar_mf_limits lim;
  xpar_mf_result res;
  const xpar_entry * e;
  xpar_mf_status st;
  xpar_memset(&lim, 0, sizeof lim);
  lim.stream_base        = m->stream_base;
  lim.stream_length      = m->stream_length;
  lim.slice_size         = m->slice_size;
  lim.align              = m->align;
  lim.posix_record_count = XPAR_ABSENT_U32;
  st = xpar_manifest_validate(m, &lim, &res);
  if (!gen_name_fault(st)) return;
  e = &m->entry[res.entry];
  if (st == XPAR_MF_DUP_NAME)
    FATAL("two entries map to '%.*s'; use --base to disambiguate",
          (int) e->name_len, e->name);
  FATAL("entry '%.*s' cannot be stored: %s", (int) e->name_len, e->name,
        xpar_mf_reason(st));
}

/*  The merged manifest.  */

typedef struct {
  xpar_manifest m;
  bool *        owned;      /*  A new FILE packet is written for it.  */
  bool *        reuse;      /*  Its extents came from an ancestor.  */
  const u8 **   body;       /*  The ancestor's FILE body, when inherited.  */
  sz *          blen;
  u32           cap;
} gen_merge;

typedef struct {
  gen_merge * g;
  const xpar_options * o;
  xpar_chunk_index * ix;
  xpar_extent * ext;
  u32 count, capacity;
  u64 * high;
  xpar_vset * ancestor;
  bool aligned, full;
} gen_chunk_pack;

static bool gen_pack_chunk(void * user, u64 file_offset, u32 len,
                           const u8 hash[16]) {
  gen_chunk_pack * c = user;
  xpar_chunk_slot * hit = xpar_chunk_index_find(c->ix, hash, len);
  u64 off;
  (void) file_offset;
  if (hit && !hit->trust) {
    xpar_blake3_t h;
    u8 got[1 << 16], check[16];
    u64 at = 0;
    bool ok = c->ancestor != NULL;
    xpar_blake3_init(&h);
    while (ok && at < len) {
      u64 take = MIN((u64) sizeof got, (u64) len - at);
      ok = xpar_vset_read(c->ancestor, hit->stream_offset + at, got, take);
      if (ok) xpar_blake3_update(&h, got, (sz) take);
      at += take;
    }
    xpar_blake3_final(&h, check, sizeof check);
    hit->trust = ok && !xpar_memcmp(check, hash, sizeof check) ? 1 : 2;
  }
  if (hit && hit->trust == 2) hit = NULL;
  if (hit && c->o->dedup_max_refs &&
      hit->refs + 1 > c->o->dedup_max_refs) hit = NULL;
  if (hit) {
    off = hit->stream_offset;
    hit->refs++;
    c->g->m.shared_bytes += len;
    c->g->m.alias_extents++;
  } else {
    u64 q = c->o->align == XPAR_ALIGN_SLICE ? c->g->m.slice_size
          : c->o->align == XPAR_ALIGN_1K ? XPAR_BLAKE3_CHUNK_LEN : 0;
    if (q && (!c->aligned || c->o->align == XPAR_ALIGN_1K)) {
      u64 pad = (*c->high - c->g->m.stream_base) % q;
      if (pad) *c->high += q - pad;
    }
    c->aligned = true;
    off = *c->high;
    *c->high += len;
    if (!xpar_chunk_index_put(c->ix, hash, len, off)) { c->full = true;  return false; }
  }
  xpar_extents_append(&c->ext, &c->count, &c->capacity, off, len);
  return true;
}

static void gen_chunk_entry(gen_merge * g, const xpar_options * o,
                            xpar_chunk_index * ix, u32 entry, u64 * high,
                            xpar_vset * ancestor) {
  gen_chunk_pack c;
  xpar_entry * e = &g->m.entry[entry];
  xpar_memset(&c, 0, sizeof c);
  c.g = g;  c.o = o;  c.ix = ix;  c.high = high;
  c.ancestor = ancestor;
  if (!g->m.source[entry] ||
      !xpar_chunk_file(g->m.source[entry],
                       o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20,
                       gen_pack_chunk, &c, NULL, NULL)) {
    xpar_free(c.ext);
    if (c.full)
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "the chunk fingerprint index exceeded --dedup-memory; "
                 "raise it or --dedup-chunk");
    FATAL("cannot read '%s' while chunking it",
          g->m.source[entry] ? g->m.source[entry] : "(unknown)");
  }
  xpar_free(e->extents);
  e->extents = c.ext;
  e->extent_count = c.count;
}

static xpar_entry * merge_append(gen_merge * g) {
  xpar_entry * e = xpar_manifest_append(&g->m);
  u32 n = g->m.count;
  if (n > g->cap) {
    u32 cap = g->cap ? g->cap * 2 : 32;
    while (cap < n) cap *= 2;
    g->owned = xpar_realloc(g->owned, cap * sizeof *g->owned);
    g->reuse = xpar_realloc(g->reuse, cap * sizeof *g->reuse);
    g->body  = xpar_realloc(g->body, cap * sizeof *g->body);
    g->blen  = xpar_realloc(g->blen, cap * sizeof *g->blen);
    g->cap   = cap;
  }
  g->owned[n - 1] = true;   g->reuse[n - 1] = false;
  g->body[n - 1]  = NULL;   g->blen[n - 1]  = 0;
  return e;
}

static void merge_free(gen_merge * g) {
  xpar_manifest_free(&g->m);
  xpar_free(g->owned);  xpar_free(g->reuse);
  xpar_free(g->body);   xpar_free(g->blen);
  xpar_memset(g, 0, sizeof *g);
}

/*  Map (content hash, length) to its first manifest entry.  */

typedef struct { u32 * slot;  u32 mask; } gen_dmap;   /*  slot = index + 1  */

static u64 dmap_hash(const u8 * h, u64 len) {
  return xpar_rd64(h) ^ (len * 0x9E3779B97F4A7C15ULL);
}

static void dmap_init(gen_dmap * d, u32 n) {
  u64 cap = 16;
  while (cap < (u64) n * 2 && cap < ((u64) 1 << 31)) cap <<= 1;
  d->mask = (u32) (cap - 1);
  d->slot = xpar_calloc((sz) cap, sizeof *d->slot);
}

static void dmap_free(gen_dmap * d) { xpar_free(d->slot);  d->slot = NULL; }

/*  Returns the stored index for a matching (hash, length), else -1.  */
static i64 dmap_probe(const gen_dmap * d, const xpar_manifest * m,
                      const xpar_entry * e) {
  u32 j;
  if (!d->slot) return -1;
  j = (u32) (dmap_hash(e->content_hash, e->length) & d->mask);
  while (d->slot[j]) {
    const xpar_entry * c = &m->entry[d->slot[j] - 1];
    if (c->length == e->length &&
        !xpar_memcmp(c->content_hash, e->content_hash, 32))
      return (i64) (d->slot[j] - 1);
    j = (j + 1) & d->mask;
  }
  return -1;
}

static void dmap_add(gen_dmap * d, const xpar_manifest * m, u32 i) {
  const xpar_entry * e = &m->entry[i];
  if (!d->slot) return;
  if (e->entry_type != XPAR_ENTRY_REGULAR || !e->extent_count) return;
  if (dmap_probe(d, m, e) >= 0) return;            /*  Keep the first.  */
  { u32 j = (u32) (dmap_hash(e->content_hash, e->length) & d->mask);
    while (d->slot[j]) j = (j + 1) & d->mask;
    d->slot[j] = i + 1; }
}

/*  Preserve inherited extents; append new bytes at the monotone high-water
    mark in manifest order.  */
/*  Whether every byte an entry names lives in the generation being built.  */
static bool gen_extents_local(const xpar_entry * e, u64 base) {
  u32 k;
  Fk(e->extent_count, if (e->extents[k].stream_offset < base) return false);
  return true;
}

static void gen_repack(gen_merge * g, const xpar_options * o,
                       const char * cache_path, const u8 * ancestor_id,
                       u64 base, xpar_chunk_index * cache_out) {
  u64 H = base;
  u32 i;
  xpar_chunk_index chunks;
  gen_dmap dmap;
  xpar_vset * ancestor = NULL;
  bool have_chunks = false;
  xpar_memset(&dmap, 0, sizeof dmap);
  if (o->dedup != XPAR_DEDUP_NONE) dmap_init(&dmap, g->m.count);
  g->m.stream_base   = base;
  g->m.entry_bytes   = 0;
  g->m.shared_bytes  = 0;
  g->m.alias_extents = 0;
  xpar_memset(&chunks, 0, sizeof chunks);
  if (o->dedup == XPAR_DEDUP_CHUNK) {
    u64 budget = o->dedup_memory ? o->dedup_memory :
                 (o->memory ? o->memory : xpar_plan_default_memory()) / 4;
    if (!xpar_chunk_index_init(&chunks, budget))
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "--dedup-memory=%" PRIu64 " is too small for a chunk index",
                 budget);
    have_chunks = true;
    if (o->dedup_scope == XPAR_SCOPE_CHAIN) {
      u64 average = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
      bool available = cache_path && ancestor_id &&
        xpar_chunk_cache_load(cache_path, ancestor_id, average, &chunks);
      if (available)
        Fi(chunks.capacity,
          const xpar_chunk_slot * s = &chunks.slot[i];
          if (s->length &&
              (s->stream_offset >= base || s->length > base -
                                                    s->stream_offset)) {
            available = false;
            break;
          });
      if (!available) {
        xpar_chunk_index_free(&chunks);
        if (!o->quiet)
          xpar_fprintf(xpar_stderr, "xpar: chain dedup is unavailable "
                       "because its verified cache is absent or stale; "
                       "using generation scope\n");
        xpar_assert(xpar_chunk_index_init(&chunks, budget));
      } else ancestor = xpar_vset_open(o);
    }
  }
  Fi(g->m.count,
    xpar_entry * e = &g->m.entry[i];
    xpar_extent x;
    if (e->entry_type == XPAR_ENTRY_REGULAR) g->m.entry_bytes += e->length;
    if (!g->owned[i] || g->reuse[i]) {
      /*  Honor dedup scope for retained extents.  */
      if (o->dedup_scope == XPAR_SCOPE_CHAIN || gen_extents_local(e, base))
        dmap_add(&dmap, &g->m, i);
      continue;
    }
    if (e->entry_type != XPAR_ENTRY_REGULAR || !e->length) {
      xpar_free(e->extents);  e->extents = NULL;  e->extent_count = 0;
      continue;
    }
    if (o->dedup != XPAR_DEDUP_NONE) {
      i64 hit = dmap_probe(&dmap, &g->m, e);
      if (hit >= 0) {
        const xpar_entry * c = &g->m.entry[hit];
        xpar_free(e->extents);
        e->extent_count = c->extent_count;
        e->extents = xpar_malloc((sz) c->extent_count * sizeof *e->extents);
        xpar_memcpy(e->extents, c->extents,
                    (sz) c->extent_count * sizeof *e->extents);
        g->m.shared_bytes  += e->length;
        g->m.alias_extents += c->extent_count;
        continue;
      }
    }
    if (have_chunks && o->dedup == XPAR_DEDUP_CHUNK) {
      gen_chunk_entry(g, o, &chunks, i, &H, ancestor);
      dmap_add(&dmap, &g->m, i);
      continue;
    }
    { u64 q = o->align == XPAR_ALIGN_SLICE ? g->m.slice_size
            : o->align == XPAR_ALIGN_1K ? XPAR_BLAKE3_CHUNK_LEN : 0;
      u64 pad = q ? (H - base) % q : 0;
      if (pad) H += q - pad;
    }
    x.stream_offset = H;  x.length = e->length;
    xpar_free(e->extents);
    e->extents = xpar_malloc(sizeof *e->extents);
    e->extents[0] = x;  e->extent_count = 1;
    H += e->length;
    dmap_add(&dmap, &g->m, i));
  dmap_free(&dmap);
  /*  Pad the generation to the selected alignment.  */
  { u64 q = o->align == XPAR_ALIGN_SLICE ? g->m.slice_size
          : o->align == XPAR_ALIGN_1K ? (u64) XPAR_BLAKE3_CHUNK_LEN : 0;
    if (q) H = base + xpar_align_up(H - base, q); }
  /*  Slice-tag tables read alignment from the manifest.  */
  g->m.align         = (u8) o->align;
  g->m.stream_length = H - base;
  g->m.dedup_level   = g->m.alias_extents ? (u8) o->dedup : XPAR_DEDUP_NONE;
  if (have_chunks && cache_out) { *cache_out = chunks;  xpar_memset(&chunks, 0, sizeof chunks); }
  xpar_vset_close(ancestor);
  if (have_chunks) xpar_chunk_index_free(&chunks);
}

/*  Whole-entry references copy ancestor extents at every deduplication
    scope; the dependency already existed under the prior manifest.  */
/*  Find reusable ancestor content through the shared hash index.  */
static const xpar_entry * gen_find_content(const xpar_manifest * anc,
                                           const gen_dmap * map,
                                           const xpar_entry * e) {
  i64 hit;
  if (e->entry_type != XPAR_ENTRY_REGULAR || !e->length) return NULL;
  hit = dmap_probe(map, anc, e);
  return hit < 0 ? NULL : &anc->entry[hit];
}

static void gen_take_extents(xpar_entry * d, const xpar_entry * s) {
  xpar_free(d->extents);
  d->extent_count = s->extent_count;
  d->extents = s->extent_count
                 ? xpar_malloc((sz) s->extent_count *
                               sizeof *d->extents) : NULL;
  if (s->extent_count)
    xpar_memcpy(d->extents, s->extents,
                (sz) s->extent_count * sizeof *d->extents);
}

/*  Rewrite changed critical packets while preserving every retained RCVS
    body; restate only its header set_id.  */

typedef struct {
  const u8 *        group;      /*  Replacement critical group, or NULL.  */
  sz                group_len;
  const xpar_layt * layt;       /*  Replacement LAYT alone, or NULL.  */
  u32               this_vol;
  const u8 *        set_id;     /*  Header identity for every packet.  */
  const xpar_key *  key;        /*  Packet/slice authentication, if any.  */
} gen_rewrite;

static bool gen_is_critical(const xpar_pkt * h) {
  return xpar_pkt_is(h, XPAR_T_SETD) || xpar_pkt_is(h, XPAR_T_FILE) ||
         xpar_pkt_is(h, XPAR_T_POSX) || xpar_pkt_is(h, XPAR_T_SLCR) ||
         xpar_pkt_is(h, XPAR_T_AUTH) || xpar_pkt_is(h, XPAR_T_LAYT);
}

static void gen_rebuild(xpar_buf * out, const xpar_options * o,
                        const u8 * data, u64 len, const gen_rewrite * rw,
                        bool nested) {
  xpar_scan sc;  xpar_pkt hdr;  const u8 * body;  u64 off;
  bool group_done = false;
  xpar_scan_init(&sc, data, len, rw->key, false);
  while (xpar_scan_next(&sc, &hdr, &body, &off)) {
    u64 blen = hdr.length - XPAR_PKT_HDR;
    if (xpar_pkt_is(&hdr, XPAR_T_ARMG) && !nested) {
      xpar_armg ag;  xpar_armour_params ap;  u8 * plain;  sz plen;
      xpar_buf inner;
      char wt[4];
      /*  Copy unchanged wrapped recovery and table packets.  */
      if (xpar_armg_wrapped_type(body, (sz) blen, wt) &&
          (!xpar_memcmp(wt, XPAR_T_RCVS, 4) ||
           !xpar_memcmp(wt, XPAR_T_SLTG, 4) ||
           !xpar_memcmp(wt, XPAR_T_SLCL, 4))) {
        xpar_pkt_write(out, hdr.type, hdr.flags, rw->set_id, body,
                       (sz) blen, rw->key);
        continue;
      }
      if (xpar_armg_read(body, (sz) blen, &ag) != XPAR_OK) continue;
      ap.symbol_bits = ag.symbol_bits;  ap.poly = ag.poly;
      ap.n = ag.n;  ap.k = ag.k;  ap.fcr = ag.fcr;  ap.prim = ag.prim;
      ap.depth = ag.depth;
      plain = arm_extract(&ap, ag.data, ag.armoured_length, ag.plain_length,
                          &plen, rw->key);
      if (!plain) FATAL_FORMAT("an armoured critical group will not extract");
      xpar_buf_init(&inner);
      gen_rebuild(&inner, o, plain, plen, rw, true);
      /*  Preserve the volume's armour parameters.  */
      gen_armour_pack_ap(out, &ap, inner.data, inner.len, rw->set_id,
                         rw->key);
      xpar_buf_free(&inner);
      xpar_free(plain);
      continue;
    }
    if (rw->group && gen_is_critical(&hdr)) {
      if (!group_done) { xpar_buf_put(out, rw->group, rw->group_len);  group_done = true; }
      continue;
    }
    if (rw->layt && xpar_pkt_is(&hdr, XPAR_T_LAYT)) {
      xpar_layt l = *rw->layt;
      l.this_volume = rw->this_vol;
      xpar_layt_write(out, &l, rw->set_id, rw->key);
      continue;
    }
    xpar_pkt_write(out, hdr.type, hdr.flags, rw->set_id, body, (sz) blen,
                   rw->key);
  }
  if (rw->group && !group_done && !nested)
    xpar_buf_put(out, rw->group, rw->group_len);
}

/*  Rebuild critical groups from stored packet bodies to preserve set_id
    inputs exactly.  */

static const xpar_crit_pkt * gen_owned_file(const xpar_chain * c, u32 g,
                                            const u8 * file_id) {
  return xpar_critset_find_file(&c->crit, c->gen[g].set_id, file_id);
}

static void gen_emit_stored(xpar_buf * out, const xpar_chain * c, u32 g,
                            const char * type, const u8 * set_id) {
  u64 want = 0;
  u32 emitted;
  /*  Bound passes so hostile keys cannot prevent termination.  */
  for (emitted = 0; emitted < c->crit.count; emitted++) {
    const xpar_crit_pkt * best = NULL;
    u64 best_key = 0;
    u32 i;
    Fi(c->crit.count,
      const xpar_crit_pkt * p = &c->crit.pkt[i];
      u64 key;
      if (!xpar_pkt_is(&p->hdr, type)) continue;
      if (xpar_memcmp(p->hdr.set_id, c->gen[g].set_id, XPAR_SET_ID_LEN))
        continue;
      key = p->body_len >= 8 ? xpar_rd64(p->body) : 0;
      if (!xpar_memcmp(type, XPAR_T_POSX, 4))
        key = p->body_len >= 4 ? xpar_rd32(p->body) : 0;
      if (key < want) continue;
      if (!best || key < best_key) { best = p;  best_key = key; });
    if (!best) return;
    xpar_pkt_write(out, best->hdr.type, best->hdr.flags, set_id, best->body,
                   (sz) best->body_len, gen_chain_key(c));
    /*  These keys come from unvalidated packet bodies. At the maximum the
        increment wraps to zero, the same packet is chosen again and the
        loop never ends, so stop instead of wrapping.  */
    if (best_key == (u64) -1) return;
    want = best_key + 1;
  }
}

/*  Reuse the set's replicated CRTR so added volumes cannot conflict.  */
static void gen_crtr_stored(xpar_buf * out, const xpar_chain * c, u32 g,
                            const u8 * set_id) {
  u32 i;
  Fi(c->crit.count,
    const xpar_crit_pkt * p = &c->crit.pkt[i];
    if (!xpar_pkt_is(&p->hdr, XPAR_T_CRTR)) continue;
    if (xpar_memcmp(p->hdr.set_id, c->gen[g].set_id, XPAR_SET_ID_LEN))
      continue;
    xpar_pkt_write(out, XPAR_T_CRTR, p->hdr.flags, set_id, p->body,
                   (sz) p->body_len, gen_chain_key(c));
    return);
  xpar_crtr_write(out, "xpar " PACKAGE_VERSION, set_id, gen_chain_key(c),
                  NULL);
}

/*  Rebuild a critical group from stored packet bodies.  */
static void gen_group_stored(xpar_buf * out, const xpar_chain * c, u32 g,
                             const xpar_layt * layt, u32 this_vol,
                             const u8 * set_id) {
  u32 j;
  xpar_setd_write(out, &c->gen[g].sd, set_id, gen_chain_key(c));
  Fj(c->gen[g].sd.file_count,
    const xpar_crit_pkt * q = gen_owned_file(c, g, c->gen[g].sd.file_id[j]);
    if (q) xpar_pkt_write(out, XPAR_T_FILE, q->hdr.flags, set_id, q->body,
                          (sz) q->body_len, gen_chain_key(c)));
  gen_emit_stored(out, c, g, XPAR_T_POSX, set_id);
  gen_emit_stored(out, c, g, XPAR_T_SLCR, set_id);
  gen_emit_stored(out, c, g, XPAR_T_AUTH, set_id);
  if (layt) {
    xpar_layt l = *layt;
    l.this_volume = this_vol;
    xpar_layt_write(out, &l, set_id, gen_chain_key(c));
  }
}

/*  The entries of generation `g` as they are on disk, with a source path
    per entry so the stream can be read again. Only the entries whose
    canonical bytes lie in this generation's range are ever read.  */
static void gen_manifest_on_disk(const xpar_chain * c, u32 g,
                                 const xpar_options * o, xpar_manifest * m,
                                 u32 ** owner) {
  u32 i;
  xpar_gchain_manifest(c, g, m, owner);
  Fi(m->count, m->source[i] = gen_entry_path(o, &m->entry[i]));
}

static bool gen_read_vset(void * ctx, u64 off, u8 * out, u64 len) {
  xpar_vset * s = ctx;
  if (xpar_vset_read(s, off, out, len)) return true;
  if (xpar_vset_io_errors(s))
    FATAL_IO("cannot read stored stream at offset %" PRIu64,
             off);
  return false;
}

static void gen_require_source_tables(const xpar_vset * set,
                                      const gen_tables * made,
                                      const gen_plan * plan) {
  const xpar_tags * stored = xpar_vset_tags(set);
  u32 have = xpar_vset_have_tables(set);
  u64 i;
  if ((have & XPAR_TAGS_CRC) && stored->slice_crc)
    Fi(plan->geom.slice_count,
      if (stored->slice_crc[i] != made->slice_crc[i])
        FATAL_CODE(XPAR_EXIT_REPAIRABLE,
                   "slice %" PRIu64 " changed while recovery data was prepared; "
                   "nothing was written", i));
  if ((have & XPAR_TAGS_TAG) && stored->slice_tag && made->slice_tag) {
    if (stored->tag_len != made->tag_len)
      FATAL_FORMAT("the selected generation's slice-tag table has the "
                   "wrong tag length");
    Fi(plan->geom.slice_count,
      if (!xpar_blake3_tag_equal(stored->slice_tag + i * stored->tag_len,
                                made->slice_tag + i * made->tag_len,
                                made->tag_len))
        FATAL_CODE(XPAR_EXIT_REPAIRABLE,
                   "slice %" PRIu64 " changed while recovery data was prepared; "
                   "nothing was written", i));
  }
}

int xpar_op_addrecovery(const xpar_options * o) {
  xpar_chain c;
  xpar_manifest m;
  gen_plan p;
  gen_tables t;
  u32 * owner = NULL;
  u32 g, i, nvol, base_vol, base_rec = 0;
  u64 have, want, axis, e;
  gen_vol * vol;
  xpar_layt layt;
  xpar_layt old;
  u8 * rec_scratch = NULL;
  const char * verify_path = NULL;
  xpar_genref verify_ref;
  char verify_id[XPAR_SET_ID_LEN * 2 + 1];
  xpar_vset * source_set;
  int source_rc;
  gen_addrec_file * staged = NULL;
  u32 staged_count = 0, staged_cap = 0;
  xpar_armour * ra = NULL;
  int level;

  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "addrecovery");
  g = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  level = gen_chain_armour(o, &c, g, c.gen[g].sd.layout);
  xpar_gchain_genref(&c, g, &verify_ref, verify_id);
  have = c.gen[g].recovery_top;
  axis = xpar_setd_recovery_limit(&c.gen[g].sd);

  if (!c.gen[g].sd.data_slice_count)
    FATAL("generation %" PRIu32 " has no stream to protect",
          c.gen[g].sd.generation);
  if (!c.gen[g].layt_body)
    FATAL_FORMAT("generation %" PRIu32 " carries no volume layout",
                 c.gen[g].sd.generation);
  if (xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &old) != XPAR_OK)
    FATAL_FORMAT("generation %" PRIu32 "'s volume layout is malformed",
                 c.gen[g].sd.generation);

  want = gen_resolve_r(&o->recovery, c.gen[g].sd.data_slice_count,
                       c.gen[g].sd.slice_size);
  if (!want)
    FATAL("use --recovery=SPEC to raise the current total of %" PRIu64,
          have);
  if (want <= have) {
    xpar_fprintf(xpar_stderr, "xpar: generation %" PRIu32 " already has %"
                 PRIu64 " "
                 "recovery slice%s; nothing to do\n",
                 c.gen[g].sd.generation, have,
                 PLURAL(have));
    gen_json_result(o, "addrecovery", c.gen[g].set_id,
                    c.gen[g].sd.generation, "unchanged", XPAR_EXIT_OK);
    xpar_layt_free(&old);
    xpar_gchain_free(&c);
    return XPAR_EXIT_OK;
  }

  /*  Refuse FFT growth beyond its recorded prefix-stable bracket.  */
  if (want > axis) {
    if (c.gen[g].sd.codec == XPAR_CODEC_FFT)
      FATAL("generation %" PRIu32 " allows at most %" PRIu64
            " FFT recovery slices, not %" PRIu64 "; re-encode with "
            "'xpar consolidate --max-recovery=%" PRIu64 "'",
            c.gen[g].sd.generation, axis, want, want);
    FATAL("generation %" PRIu32 " allows at most %" PRIu64
          " recovery slices, not %" PRIu64 "; re-encode with "
          "'xpar consolidate --max-recovery=%" PRIu64 "'",
          c.gen[g].sd.generation, axis, want, want);
  }
  if (!xpar_codec_supports_axis(c.gen[g].sd.codec,
                                c.gen[g].sd.field_log2,
                                c.gen[g].sd.data_slice_count, want,
                                c.gen[g].sd.recovery_axis_log2))
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "the %s codec cannot express S=%" PRIu64 " with R=%" PRIu64 " over "
               "GF(2^%" PRIu8 ")", xpar_codec_name(c.gen[g].sd.codec),
               c.gen[g].sd.data_slice_count,
               want, c.gen[g].sd.field_log2);

  /*  Strongly verify the stored generation stream before encoding new
      recovery.  */
  source_set = xpar_vset_open(o);
  source_rc = xpar_vset_check(source_set, o, NULL);
  /*  A volume that only wants rewriting still hands back the whole stream,
      and that is all this encode reads.  */
  if (!xpar_vset_stream_intact(source_set, source_rc))
    FATAL_CODE(source_rc,
               "recovery was not added because generation %" PRIu32
               " is damaged", c.gen[g].sd.generation);

  gen_manifest_on_disk(&c, g, o, &m, &owner);
  xpar_memset(&p, 0, sizeof p);
  if (!xpar_geom_from_setd(&c.gen[g].sd, &p.geom))
    FATAL_FORMAT("generation %" PRIu32 "'s geometry is malformed",
                 c.gen[g].sd.generation);
  p.recovery   = want;
  p.encode_r   = want;
  p.field_log2 = c.gen[g].sd.field_log2;
  p.codec      = c.gen[g].sd.codec;
  p.axis       = c.gen[g].sd.recovery_axis_log2;
  gen_encode(&m, &p, c.gen[g].sd.slice_tag_len, o->memory,
             c.base ? c.base : o->set,
             gen_chain_key(&c), gen_read_vset, source_set, &t, NULL);
  gen_require_source_tables(source_set, &t, &p);
  if (t.rec_spill) rec_scratch = xpar_alloc_raw((sz) t.rec_z);

  /*  Re-encoding must reproduce every existing recovery exponent exactly.  */
  Fi(c.crit.count,
    const xpar_crit_pkt * q = &c.crit.pkt[i];
    xpar_rcvs r;
    if (!xpar_pkt_is(&q->hdr, XPAR_T_RCVS)) continue;
    if (xpar_memcmp(q->hdr.set_id, c.gen[g].set_id, XPAR_SET_ID_LEN))
      continue;
    if (xpar_rcvs_read(q->body, (sz) q->body_len, p.geom.slice_size, &r) !=
        XPAR_OK) continue;
    if (r.exponent >= want) continue;
    if (xpar_memcmp(r.data, gen_rec_get(&t, r.exponent, rec_scratch),
                    (sz) p.geom.slice_size))
      FATAL_CODE(XPAR_EXIT_INTERNAL,
                 "internal: re-encoding at R=%" PRIu64 " changed recovery slice "
                 "%" PRIu64 "; nothing was written",
                 want, r.exponent));

  if (c.gen[g].sd.layout == XPAR_LAYOUT_ARMOURED) {
    const xpar_chain_vol * source = NULL;
    xpar_arm_prologue pr;
    xpar_armour_params ap;
    xpar_buf head, group;
    xpar_volh vh;
    xpar_wropt w;
    char * arm_stage;
    Fi(c.vol_count,
      if (c.vol[i].gen == g && c.vol[i].armoured_file)
        { source = &c.vol[i];  break; });
    if (!source || !xpar_garm_prologue(source->data, source->len, &pr, NULL))
      FATAL_FORMAT("generation %" PRIu32 "'s armoured archive is unavailable",
                   c.gen[g].sd.generation);
    arm_params_of(&pr, &ap);
    xpar_buf_init(&head);
    xpar_memset(&vh, 0, sizeof vh);
    vh.volume_index = XPAR_VOL_STANDALONE;
    vh.volume_kind = XPAR_VOL_INDEX;
    xpar_volh_write(&head, &vh, c.gen[g].set_id, gen_chain_key(&c));
    xpar_buf_init(&group);
    gen_group_stored(&group, &c, g, &old, XPAR_VOL_STANDALONE,
                     c.gen[g].set_id);
    xpar_buf_put(&head, group.data, group.len);
    xpar_buf_free(&group);
    gen_wropt(o, &w);
    arm_stage = gen_stage_arm_archive(
      source->path, &ap, &m, &p, &t, rec_scratch, &head,
      c.gen[g].set_id, gen_chain_key(&c), &w, gen_read_vset, source_set);
    /*  A whole-file layout has an especially simple transaction: the staged
        archive is itself a complete set, so run the public reader over it
        before the single atomic rename replaces the old archive.  */
    xpar_verify_written_archive_at(o, arm_stage, &verify_ref);
    xpar_vset_close(source_set);
    gen_publish_whole(arm_stage, source->path, true);
    xpar_buf_free(&head);
    xpar_verify_written_set_at(o, source->path, &verify_ref);
    if (!o->quiet)
      xpar_fprintf(xpar_stderr,
                   "xpar: generation %" PRIu32 " now carries %" PRIu64 " recovery slice%s "
                   "inside its armoured archive (%" PRIu64 " added)\n",
                   c.gen[g].sd.generation, want,
                   PLURAL(want), want - have);
    gen_json_result(o, "addrecovery", c.gen[g].set_id,
                    c.gen[g].sd.generation, "ok", XPAR_EXIT_OK);
    xpar_layt_free(&old);
    gen_tables_free(&t);
    xpar_free(rec_scratch);
    xpar_free(owner);
    xpar_manifest_free(&m);
    xpar_gchain_free(&c);
    return XPAR_EXIT_OK;
  }

  xpar_vset_close(source_set);

  /*  The new volumes continue the ladder, so the existing ones keep
      their names and their bytes; only the layout packet learns about
      the new ones.  */
  { u64 left = want - have, first = have, step = 1;
    u32 n = old.count;
    layt.this_volume = XPAR_VOL_STANDALONE;
    layt.count = n;
    layt.vol = xpar_calloc(n + 64, sizeof *layt.vol);
    Fi(n,
      layt.vol[i] = old.vol[i];
      layt.vol[i].name = xpar_strdup(old.vol[i].name);
      if (old.vol[i].kind == XPAR_VOL_RECOVERY) base_rec++);
    base_vol = n;
    nvol = 0;
    vol = NULL;
    while (left) {
      u64 take = MIN(step, left);
      vol = xpar_realloc(vol, (sz) (nvol + 1) * sizeof *vol);
      xpar_memset(&vol[nvol], 0, sizeof vol[nvol]);
      vol[nvol].first = first;  vol[nvol].count = take;
      layt.vol[layt.count].kind           = XPAR_VOL_RECOVERY;
      layt.vol[layt.count].recovery_first = (u32) first;
      layt.vol[layt.count].byte_length    = take;
      layt.count++;
      nvol++;  first += take;  left -= take;  step *= 2;
      if (layt.count >= n + 64) break;
    }
    /*  The widths span the old volumes too: the names already on disk
        keep theirs, and the new ones must not come out narrower.  */
    { u64 max_first = 0, max_count = 1;
      int wf, wc;
      Fi(layt.count,
        if (layt.vol[i].kind != XPAR_VOL_RECOVERY) continue;
        if (layt.vol[i].recovery_first > max_first)
          max_first = layt.vol[i].recovery_first;
        if (layt.vol[i].byte_length > max_count)
          max_count = layt.vol[i].byte_length);
      xpar_vname_widths(max_first, max_count, &wf, &wc);
      Fi(nvol,
        char * nd, * nn;
        vol[i].name = xpar_vname_recovery(c.base ? c.base : o->set,
                                        c.gen[g].sd.generation, vol[i].first,
                                        vol[i].count, wf, wc,
                                        base_rec + i);
        gen_split_path(vol[i].name, &nd, &nn);
        layt.vol[base_vol + i].name = nn;
        xpar_free(nd));
    }
  }

  if (!o->force)
    Fi(nvol,
      if (gen_exists(vol[i].name))
        FATAL("'%s' exists; use -f to overwrite it; nothing was written",
              vol[i].name));

  /*  Strictly parse all staged files, then publish new volumes before their
      referring index.  */
  staged_cap = nvol + c.vol_count;
  staged = xpar_calloc(staged_cap ? staged_cap : 1, sizeof *staged);
  Fi(nvol,
    xpar_buf out;
    xpar_volh vh;
    xpar_buf_init(&out);
    xpar_memset(&vh, 0, sizeof vh);
    vh.volume_index = base_vol + i;
    vh.volume_kind  = XPAR_VOL_RECOVERY;
    xpar_volh_write(&out, &vh, c.gen[g].set_id, gen_chain_key(&c));
    { xpar_buf group;
      xpar_buf_init(&group);
      gen_group_stored(&group, &c, g, &layt, base_vol + i,
                       c.gen[g].set_id);
      if (o->armour != XPAR_ARMOUR_NONE)
        gen_armour_pack(&out, o, group.data, group.len, c.gen[g].set_id,
                        gen_chain_key(&c));
      else
        xpar_buf_put(&out, group.data, group.len);
      xpar_buf_free(&group);
    }
    if (!ra) ra = gen_wrap_rcvs_armour(o, &c, g, level, p.geom.slice_size);
    for (e = vol[i].first; e < vol[i].first + vol[i].count; e++) {
      const u8 * rec = gen_rec_get(&t, e, rec_scratch);
      gen_rcvs_emit(&out, ra, e, rec, (sz) p.geom.slice_size,
                    c.gen[g].set_id, gen_chain_key(&c));
    }
    gen_crtr_stored(&out, &c, g, c.gen[g].set_id);
    staged[staged_count].stage = gen_stage_whole(vol[i].name, out.data,
                                                  out.len);
    staged[staged_count].final = vol[i].name;
    staged[staged_count].replace = o->force;
    if (!xpar_verify_written_volume(staged[staged_count].stage,
                                    gen_chain_key(&c), c.gen[g].set_id,
                                    base_vol + i, XPAR_VOL_RECOVERY,
                                    vol[i].first, vol[i].count,
                                    p.geom.slice_size)) {
      xpar_buf_free(&out);
      gen_addrec_discard(staged, staged_count + 1);
      FATAL_CODE(XPAR_EXIT_INTERNAL,
                 "internal: staged addrecovery volume failed verification");
    }
    staged_count++;
    xpar_buf_free(&out));

  /*  Now every existing volume of this generation learns the new layout.
      Its recovery slices are copied byte for byte, which is what makes
      this cheap.  */
  Fi(c.vol_count,
    xpar_buf out;
    gen_rewrite rw;
    u32 this_vol = XPAR_VOL_STANDALONE;
    if (c.vol[i].gen != g) continue;
    if (c.vol[i].volume_kind == XPAR_VOL_INDEX) verify_path = c.vol[i].path;
    if (c.vol[i].volume_kind == XPAR_VOL_RECOVERY)
      this_vol = c.vol[i].volume_index;
    xpar_memset(&rw, 0, sizeof rw);
    rw.layt = &layt;  rw.this_vol = this_vol;  rw.set_id = c.gen[g].set_id;
    rw.key = gen_chain_key(&c);
    xpar_buf_init(&out);
    gen_rebuild(&out, o, c.vol[i].data, c.vol[i].len, &rw, false);
    staged[staged_count].stage = gen_stage_whole(c.vol[i].path, out.data,
                                                  out.len);
    staged[staged_count].final = c.vol[i].path;
    staged[staged_count].replace = true;
    if (!xpar_verify_written_volume(staged[staged_count].stage,
                                    gen_chain_key(&c), c.gen[g].set_id,
                                    c.vol[i].volume_index,
                                    c.vol[i].volume_kind,
                                    c.vol[i].recovery_first,
                                    c.vol[i].recovery_count,
                                    p.geom.slice_size)) {
      xpar_buf_free(&out);
      gen_addrec_discard(staged, staged_count + 1);
      FATAL_CODE(XPAR_EXIT_INTERNAL,
                 "internal: staged layout update failed verification");
    }
    staged_count++;
    xpar_buf_free(&out));

  FATAL_UNLESS(verify_path != NULL,
               "generation %" PRIu32
               " has no index volume to verify after writing", c.gen[g].sd.generation);
  /*  Publish the index between replacements and new volumes.  */
  gen_addrec_order(staged, staged_count, nvol, verify_path);
  gen_addrec_publish(staged, staged_count);
  xpar_verify_written_set_at(o, verify_path, &verify_ref);

  if (!o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: generation %" PRIu32 " now carries %" PRIu64 " recovery slice%s "
                 "(%" PRIu64 " added in %" PRIu32 " volume%s); every existing slice is "
                 "unchanged\n", c.gen[g].sd.generation,
                 want, PLURAL(want),
                 want - have, nvol, PLURAL(nvol));
  gen_json_result(o, "addrecovery", c.gen[g].set_id,
                  c.gen[g].sd.generation, "ok", XPAR_EXIT_OK);

  gen_volumes_free(vol, nvol);
  xpar_layt_free(&layt);
  xpar_layt_free(&old);
  gen_tables_free(&t);
  if (ra) xpar_armour_free(ra);
  xpar_free(rec_scratch);
  xpar_free(owner);
  xpar_manifest_free(&m);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

int xpar_op_add(const xpar_options * caller) {
  xpar_options eff = *caller;
  const xpar_options * o = &eff;
  xpar_chain c;
  xpar_manifest inh, fresh;
  gen_dmap inh_map;

  gen_merge g;
  gen_write_req rq;
  xpar_posix_rec ** tab;
  u32 * tabn;
  u32 * owner = NULL;
  u32 head, i, ia = 0, ib = 0, caps, ratio_gen = 0;
  u32 added = 0, changed = 0, kept = 0, dropped = 0;
  bool warn_posix = false, inherited_r;
  /*  Older-generation references follow --dedup-scope.  */
  bool chain_dedup;
  f64 old_ratio = 0.0;
  char idbuf[XPAR_SET_ID_LEN * 2 + 1];
  char * input_cache = NULL, * output_cache = NULL, * stage_cache = NULL;
  char * stdin_stage = NULL, * stdin_final = NULL;
  xpar_chunk_index chunk_cache;

  xpar_memset(&fresh, 0, sizeof fresh);
  xpar_memset(&g, 0, sizeof g);
  xpar_memset(&chunk_cache, 0, sizeof chunk_cache);
  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "add");
  gen_refuse_unreadable(&c, "add");
  if (c.authenticated && o->auth_only && !c.auth_only)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "a generation must keep its chain's authentication mode; "
               "this chain retains public verification hashes");
  head = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  eff.layout = gen_chain_layout(o, &c, head, true);
  xpar_cli_armour_for_layout(&eff, eff.layout);
  eff.armour = gen_chain_armour(&eff, &c, head, eff.layout);
  inherited_r = gen_inherit_recovery(&eff, &c, head, false, &old_ratio,
                                     &ratio_gen);
  chain_dedup = o->dedup != XPAR_DEDUP_NONE &&
                o->dedup_scope == XPAR_SCOPE_CHAIN;
  gen_chain_integrity(o, &c, head, &eff);

  for (i = head; i != XPAR_GEN_NONE; i = c.gen[i].parent)
    if (c.gen[i].parent_missing) {
      xpar_hex(idbuf, c.gen[i].sd.parent_set_id, XPAR_SET_ID_LEN);
      FATAL_FORMAT("generation %" PRIu32
                   " names parent %s, which is not here; an "
                   "incomplete chain cannot be extended",
                   c.gen[i].sd.generation, idbuf);
    }
  if (o->align == XPAR_ALIGN_SLICE && !o->slice_size)
    FATAL("--align=slice on an existing set requires explicit -s");
  /*  Whole-file chain deduplication searches the effective ancestor
      manifest; only chunk deduplication needs the disk index.  */
  if (o->dedup == XPAR_DEDUP_NONE && o->verbose)
    xpar_fprintf(xpar_stderr,
                 "xpar: --dedup=none: renamed or metadata-only changes "
                 "append and re-encode file data\n");

  xpar_gchain_manifest(&c, head, &inh, &owner);
  /*  Share one ancestor-content index across all lookups.  */
  xpar_memset(&inh_map, 0, sizeof inh_map);
  if (chain_dedup) {
    dmap_init(&inh_map, inh.count);
    Fi(inh.count, dmap_add(&inh_map, &inh, i));
  }
  tab  = xpar_calloc(c.gen_count, sizeof *tab);
  tabn = xpar_calloc(c.gen_count, sizeof *tabn);
  Fi(c.gen_count, tabn[i] = xpar_gchain_posix(&c, i, &tab[i]));

  if (o->path_count) {
    xpar_walk_opts wo;
    xpar_progress_t prog;
    xpar_json pjs;
    char * staged_path[1];
    char * const * walk_paths = o->paths;
    u32 walk_count = o->path_count;
    if (o->from_stdin) {
      stdin_stage = xpar_spool_stdin(o);
      staged_path[0] = stdin_stage;
      walk_paths = staged_path;
      walk_count = 1;
    }
    xpar_walk_opts_default(&wo);
    /*  gen_repack owns cross-generation chunk placement. The walk still
        hashes every entry, but running its root-set chunker too would read
        every changed file twice for an extent list immediately discarded.  */
    wo.dedup           = (u8) (o->dedup == XPAR_DEDUP_CHUNK
                                 ? XPAR_DEDUP_FILE : o->dedup);
    wo.align           = (u8) o->align;
    wo.slice_size      = o->slice_size;
    wo.stream_base     = c.gen[head].sd.stream_base +
                         c.gen[head].sd.stream_length;
    wo.dedup_max_refs  = o->dedup_max_refs;
    wo.dedup_chunk     = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
    wo.dedup_memory    = o->dedup_memory ? o->dedup_memory
                           : (o->memory ? o->memory
                                        : xpar_plan_default_memory()) / 4;
    wo.preserve        = o->preserve;
    wo.preserve_explicit = o->preserve_explicit;
    wo.base_dir        = o->base_dir;
    wo.exclude         = o->exclude;
    wo.exclude_count   = o->exclude_count;
    wo.include         = o->include;
    wo.include_count   = o->include_count;
    wo.recurse         = o->recurse;
    wo.follow_symlinks = o->follow_symlinks;
    wo.reproducible    = o->reproducible;
    wo.strict          = o->strict;
    wo.self_base       = o->set_ref.base;
    /*  A secure spool has a random private basename. Selection is defined
        over --stdin-name, not that implementation detail, and is therefore
        applied immediately after the manifest entry is renamed.  */
    if (stdin_stage) {
      wo.exclude = NULL;  wo.exclude_count = 0;
      wo.include = NULL;  wo.include_count = 0;
    }
    xpar_manifest_walk(&fresh, walk_paths, walk_count, &wo);
    if (stdin_stage) {
      FATAL_UNLESS(fresh.count == 1 &&
                   fresh.entry[0].entry_type == XPAR_ENTRY_REGULAR,
                   "the staged pipe did not produce one regular entry");
      xpar_free(fresh.entry[0].name);
      fresh.entry[0].name = xpar_strdup(o->stdin_name);
      fresh.entry[0].name_len = (u32) xpar_strlen(o->stdin_name);
      { xpar_walk_opts select = wo;
        select.exclude = o->exclude;  select.exclude_count = o->exclude_count;
        select.include = o->include;  select.include_count = o->include_count;
        if (!xpar_manifest_name_selected(&select, o->stdin_name)) {
          xpar_manifest_free(&fresh);
          xpar_memset(&fresh, 0, sizeof fresh);
        }
      }
    }
    xpar_progress_init(&prog, xpar_progress_wanted(o),
                       0, "hashing");
    xpar_json_init(&pjs, xpar_stdout, o->json);
    if (o->json) xpar_progress_sink(&prog, xpar_json_progress_sink, &pjs);
    xpar_manifest_pack(&fresh, &wo, &prog);
    xpar_progress_end(&prog);
    if (c.auth_only) {
      bool * all = xpar_calloc(fresh.count ? fresh.count : 1, sizeof *all);
      Fi(fresh.count, all[i] = true);
      gen_auth_only_hashes(&fresh, all, &c.key);
      xpar_free(all);
    }
    if (c.authenticated)
      Fi(fresh.count,
        xpar_file_id(&fresh.entry[i], c.key.k_file,
             fresh.entry[i].file_id));
  }

  caps = xpar_fs_caps(o->base_dir ? o->base_dir : ".");

  g.m.slice_size = o->slice_size;
  while (ia < inh.count || ib < fresh.count) {
    int cmp;
    xpar_entry * e;
    if      (ia >= inh.count)   cmp =  1;
    else if (ib >= fresh.count) cmp = -1;
    else cmp = xpar_name_cmp(inh.entry[ia].name, inh.entry[ia].name_len,
                             fresh.entry[ib].name, fresh.entry[ib].name_len);

    if (cmp > 0) {
      const xpar_entry * anc;
      e = merge_append(&g);
      gen_entry_copy(e, &fresh.entry[ib]);
      if (fresh.source && fresh.source[ib])
        g.m.source[g.m.count - 1] = xpar_strdup(fresh.source[ib]);
      anc = o->dedup != XPAR_DEDUP_NONE ? gen_find_content(&inh, &inh_map, e) : NULL;
      if (anc) { gen_take_extents(e, anc);  g.reuse[g.m.count - 1] = true; }
      added++;  ib++;
      continue;
    }
    if (cmp == 0) {
      bool same = gen_entry_same(&inh.entry[ia], &fresh.entry[ib],
                                 tab[owner[ia]], tabn[owner[ia]],
                                 fresh.posix, fresh.posix_count);
      if (same) {
        u32 h = 0;
        const xpar_crit_pkt * p = chain_file_pkt(&c, head,
                                                 inh.entry[ia].file_id, &h);
        e = merge_append(&g);
        gen_entry_copy(e, &inh.entry[ia]);
        g.owned[g.m.count - 1] = false;
        g.body[g.m.count - 1]  = p ? p->body : NULL;
        g.blen[g.m.count - 1]  = p ? (sz) p->body_len : 0;
        kept++;
      } else {
        const xpar_entry * anc = NULL;
        e = merge_append(&g);
        gen_entry_copy(e, &fresh.entry[ib]);
        if (fresh.source && fresh.source[ib])
          g.m.source[g.m.count - 1] = xpar_strdup(fresh.source[ib]);
        /*  Reuse bytes for metadata-only changes.  */
        if (o->rescan == XPAR_RESCAN_HASH &&
            gen_content_same(&inh.entry[ia], e))
          anc = &inh.entry[ia];
        else if (o->dedup != XPAR_DEDUP_NONE)
          anc = gen_find_content(&inh, &inh_map, e);
        if (anc) { gen_take_extents(e, anc);  g.reuse[g.m.count - 1] = true; }
        changed++;
      }
      ia++;  ib++;
      continue;
    }

    /*  Inherited and not named by <paths...>: --rescan decides whether
        the copy on disk is still the one the chain describes.  */
    { const xpar_entry * old = &inh.entry[ia];
      char * path;
      xpar_stat_t st;
      bool gone, stale;
      if (o->rescan == XPAR_RESCAN_NONE) {
        u32 h = 0;
        const xpar_crit_pkt * p = chain_file_pkt(&c, head, old->file_id, &h);
        e = merge_append(&g);
        gen_entry_copy(e, old);
        g.owned[g.m.count - 1] = false;
        g.body[g.m.count - 1]  = p ? p->body : NULL;
        g.blen[g.m.count - 1]  = p ? (sz) p->body_len : 0;
        kept++;  ia++;
        continue;
      }
      path = gen_entry_path(o, old);
      gone = xpar_lstat(path, &st) != 0;
      stale = false;

      if (gone) {
        if (!o->allow_missing) {
          xpar_free(path);
          FATAL("'%.*s' is in the set but not on disk; pass "
                "--allow-missing to record the deletion",
                (int) old->name_len, old->name);
        }
        dropped++;  xpar_free(path);  ia++;
        continue;
      }
      if (o->rescan == XPAR_RESCAN_STAT) {
        if (old->entry_type == XPAR_ENTRY_REGULAR &&
            (st.size != old->length ||
             (old->mtime_ns != XPAR_ABSENT_TIME &&
              st.mtime_ns != old->mtime_ns))) stale = true;
      } else if (o->rescan == XPAR_RESCAN_HASH) {
        xpar_entry probe;
        gen_entry_copy(&probe, old);
        if (!gen_refresh(&probe, path, o, caps, &warn_posix,
                         gen_chain_key(&c), c.auth_only)) {
          int err = xpar_errno();
          if (st.is_regular || st.is_dir || st.is_symlink)
            FATAL_IO("cannot re-read '%s': %s", path, xpar_strerror(err));
          stale = true;
        } else if (xpar_memcmp(probe.content_hash, old->content_hash, 32))
          stale = true;
        xpar_entry_free(&probe);
      }

      e = merge_append(&g);
      gen_entry_copy(e, old);
      if (stale) {
        const xpar_entry * anc;
        if (!gen_refresh(e, path, o, caps, &warn_posix,
                         gen_chain_key(&c), c.auth_only))
          FATAL_IO("cannot re-read '%s'", path);
        g.m.source[g.m.count - 1] = xpar_strdup(path);
        anc = o->dedup != XPAR_DEDUP_NONE ? gen_find_content(&inh, &inh_map, e) : NULL;
        if (anc) { gen_take_extents(e, anc);  g.reuse[g.m.count - 1] = true; }
        changed++;
      } else {
        u32 h = 0;
        const xpar_crit_pkt * p = chain_file_pkt(&c, head, old->file_id, &h);
        g.owned[g.m.count - 1] = false;
        g.body[g.m.count - 1]  = p ? p->body : NULL;
        g.blen[g.m.count - 1]  = p ? (sz) p->body_len : 0;
        kept++;
      }
      xpar_free(path);
      ia++;
    }
  }

  /*  Resolve hard-link targets by indexed name lookup.  */
  { xpar_nameidx ix;
    xpar_nameidx_build(&g.m, &ix);
    Fi(g.m.count,
      xpar_entry * e = &g.m.entry[i];
      const xpar_entry * t;
      i64 j;
      if (e->entry_type != XPAR_ENTRY_HARDLINK || g.owned[i]) continue;
      j = xpar_nameidx_find(&g.m, &ix, (const char *) e->extra,
                            e->extra_len);
      if (j < 0 || (u32) j == i || !g.owned[j]) continue;
      t = &g.m.entry[j];
      xpar_memcpy(e->content_hash, t->content_hash, 32);
      xpar_memcpy(e->prefix_hash, t->prefix_hash, 16);
      e->length = t->length;
      xpar_file_id(e, c.authenticated ? c.key.k_file : NULL, e->file_id);
      g.owned[i] = true);
    xpar_nameidx_free(&ix); }

  if (warn_posix)
    xpar_fprintf(xpar_stderr,
                 "xpar: rescanning an unnamed inherited entry dropped its "
                 "ownership and extended attributes\n");

  if (fresh.posix_count) {
    g.m.posix       = fresh.posix;
    g.m.posix_count = fresh.posix_count;
    g.m.posix_cap   = fresh.posix_cap;
    fresh.posix     = NULL;
    fresh.posix_count = fresh.posix_cap = 0;
  }
  Fi(g.m.count, if (!g.m.source[i]) g.m.source[i] = gen_entry_path(o, &g.m.entry[i]));
  /*  Reject sidecar entries the chain cannot resolve.  */
  if (o->layout == XPAR_LAYOUT_SIDECAR) {
    char * dir = xpar_path_dir(c.base ? c.base : o->set);
    const xpar_entry * lost =
      xpar_manifest_unreachable(&g.m, dir, o->stdin_name);
    xpar_free(dir);
    if (lost)
      FATAL("sidecar entry '%.*s' is unreachable; use paths relative to the "
            "set directory or use --base",
            (int) lost->name_len, lost->name);
  }
  if (c.base) input_cache = xpar_vname_cache(c.base);
  gen_repack(&g, o, input_cache, c.gen[head].set_id,
             c.gen[head].sd.stream_base + c.gen[head].sd.stream_length,
             o->dedup == XPAR_DEDUP_CHUNK &&
             o->dedup_scope == XPAR_SCOPE_CHAIN ? &chunk_cache : NULL);

  gen_check_manifest(&g.m);

  xpar_memset(&rq, 0, sizeof rq);
  rq.o             = o;
  rq.m             = &g.m;
  rq.owned         = g.owned;
  rq.inh_body      = g.body;
  rq.inh_len       = g.blen;
  rq.generation    = c.gen[head].sd.generation + 1;
  rq.stream_base   = g.m.stream_base;
  rq.parent_set_id = c.gen[head].set_id;
  rq.base          = o->output ? o->output : c.base;
  rq.quiet         = o->quiet;
  rq.auth_only     = c.authenticated ? c.auth_only : o->auth_only;
  if (!rq.base) FATAL("this set has no base name; pass --output");
  if (chunk_cache.slot) {
    u64 average = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
    output_cache = xpar_vname_cache(rq.base);
    stage_cache = gen_unused_path(output_cache, "xpar-cache", "GCA", "TMP",
                                  0);
    if (!stage_cache ||
        !xpar_chunk_cache_write(stage_cache, c.gen[head].set_id, average,
                                &chunk_cache)) {
      if (o->verbose)
        xpar_fprintf(xpar_stderr, "xpar: cannot stage chunk cache '%s'\n",
                     output_cache);
      xpar_free(stage_cache);  stage_cache = NULL;
    }
    xpar_chunk_index_free(&chunk_cache);
  }
  gen_write_set(&rq);
  if (!o->no_verify_after)
    xpar_verify_written_set_sources(o, rq.index_path, &g.m);
  if (stdin_stage && o->layout == XPAR_LAYOUT_SIDECAR) {
    stdin_final = xpar_publish_spooled_stdin(o, stdin_stage);
    xpar_free(stdin_stage);
    stdin_stage = NULL;
  }
  if (stage_cache &&
      (!xpar_chunk_cache_rebind(stage_cache, rq.set_id) ||
       !gen_publish_cache(stage_cache, output_cache))) {
    if (o->verbose)
      xpar_fprintf(xpar_stderr, "xpar: cannot update chunk cache '%s'\n",
                   output_cache);
    xpar_remove(stage_cache);
  }

  xpar_hex(idbuf, rq.set_id, XPAR_SET_ID_LEN);
  if (!o->quiet) {
    xpar_fprintf(xpar_stderr,
                 "xpar: generation %" PRIu32 ", set %s: %" PRIu32 " %s "
                 "(%" PRIu32 " added, %" PRIu32 " changed, %" PRIu32
                 " inherited, %" PRIu32 " dropped), %" PRIu64
                 " new stream bytes, %" PRIu64 " recovery slice%s in %" PRIu32 " "
                 "volume%s%s\n", rq.generation, idbuf, g.m.count,
                 g.m.count == 1 ? "entry" : "entries", added,
                 changed, kept, dropped,
                 g.m.stream_length,
                 rq.plan.recovery,
                 PLURAL(rq.plan.recovery), rq.volumes - 1,
                 PLURAL(rq.volumes - 1),
                 inherited_r ? " at the redundancy this chain already had"
                             : "");
    if (inherited_r)
      xpar_fprintf(xpar_stderr,
                   "xpar: without -r, inherited generation %" PRIu32
                   "'s %.1f%%\n",
                   c.gen[head].sd.generation, old_ratio);
  }
  gen_warn_thinner(o, old_ratio, rq.plan.recovery,
                   rq.plan.geom.slice_count);
  gen_report_superseded(o, &c, &g.m, head);
  gen_json_result(o, "add", rq.set_id, rq.generation, "ok", XPAR_EXIT_OK);

  Fi(c.gen_count, if (tab[i]) xpar_gchain_posix_free(tab[i], tabn[i]));
  xpar_free(tab);  xpar_free(tabn);  xpar_free(owner);
  xpar_free(input_cache);  xpar_free(output_cache);  xpar_free(stage_cache);
  if (stdin_stage && xpar_remove(stdin_stage) != 0 && o->verbose)
    xpar_fprintf(xpar_stderr, "xpar: warning: cannot remove spool '%s'\n",
                 stdin_stage);
  xpar_free(stdin_stage);
  xpar_free(stdin_final);
  xpar_free(rq.index_path);
  xpar_chunk_index_free(&chunk_cache);
  merge_free(&g);
  xpar_manifest_free(&fresh);
  dmap_free(&inh_map);
  xpar_manifest_free(&inh);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

/*  prune.  */

/*  Removing a generation destroys entries whose extents or owning FILE
    packets live there.  */
static bool gen_orphaned(const xpar_chain * c, const xpar_entry * e,
                         u32 owner, const bool * removed) {
  u32 k;
  if (removed[owner]) return true;
  Fk(e->extent_count,
    i64 h = xpar_gchain_gen_of(c, e->extents[k].stream_offset,
                               e->extents[k].length);
    if (h >= 0 && removed[h]) return true);
  return false;
}

static u64 gen_volume_bytes(const xpar_chain * c, u32 g) {
  u64 n = 0;  u32 i;
  Fi(c->vol_count, if (c->vol[i].gen == g) n += c->vol[i].len);
  if (c->gen[g].layt_body) {
    xpar_layt l;
    if (xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, &l) ==
        XPAR_OK) {
      Fi(l.count,
        if (l.vol[i].kind == XPAR_VOL_DATA) {
                char * p = xpar_path_join(c->dir, l.vol[i].name);
                xpar_stat_t st;
                if (xpar_lstat(p, &st) == 0 && !st.is_dir) n += st.size;
                xpar_free(p);
              });
      xpar_layt_free(&l);
    }
  }
  return n;
}

/*  Ancestor pruning compacts stream coordinates and generation numbers;
    recovery packet bodies remain generation-local and unchanged.  */
static void gen_prune_rebase(const xpar_chain * c, xpar_manifest * m,
                             const bool * keep, const bool * removed,
                             const u64 * base) {
  u32 i, k;
  Fi(m->count,
    if (keep[i])
      Fk(m->entry[i].extent_count,
        xpar_extent * x = &m->entry[i].extents[k];
        i64 h = xpar_gchain_gen_of(c, x->stream_offset, x->length);
        FATAL_UNLESS(h >= 0 && !removed[h],
                     "internal: a surviving extent has no surviving generation");
        x->stream_offset = base[h] +
                           (x->stream_offset - c->gen[h].sd.stream_base)));
}

static void gen_prune_name(xpar_vol * v, const xpar_chain * c, u32 generation,
                           u32 data_index, u32 data_count, u32 recovery_index,
                           int wf, int wc) {
  char * full = NULL, * dir, * name;
  int width;
  xpar_free(v->name);  v->name = NULL;
  if (v->kind == XPAR_VOL_INDEX)
    full = xpar_vname_index(c->base, generation);
  else if (v->kind == XPAR_VOL_RECOVERY)
    full = xpar_vname_recovery(c->base, generation, v->recovery_first,
                             v->byte_length, wf, wc, recovery_index);
  else {
    width = xpar_digits10(data_count ? data_count - 1 : 0);
    if (width < 2) width = 2;
    full = xpar_vname_data(c->base, generation, data_index, width);
  }
  gen_split_path(full, &dir, &name);
  v->name = name;
  xpar_free(dir);  xpar_free(full);
}

static bool gen_prune_layout(const xpar_chain * c, u32 g, u32 generation,
                             xpar_layt * old, xpar_layt * now) {
  u32 i, di = 0, ri = 0, dn = 0;
  u64 max_first = 0, max_count = 1;
  int wf, wc;
  if (!c->gen[g].layt_body ||
      xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, old) !=
        XPAR_OK) return false;
  xpar_memset(now, 0, sizeof *now);
  now->this_volume = old->this_volume;
  now->count = old->count;
  now->vol = xpar_calloc(now->count, sizeof *now->vol);
  Fi(old->count,
    if (old->vol[i].kind == XPAR_VOL_DATA) dn++;
    if (old->vol[i].kind != XPAR_VOL_RECOVERY) continue;
    if (old->vol[i].recovery_first > max_first)
      max_first = old->vol[i].recovery_first;
    if (old->vol[i].byte_length > max_count)
      max_count = old->vol[i].byte_length);
  xpar_vname_widths(max_first, max_count, &wf, &wc);
  Fi(old->count,
    now->vol[i] = old->vol[i];
    now->vol[i].name = NULL;
    gen_prune_name(&now->vol[i], c, generation, di, dn, ri, wf, wc);
    if (old->vol[i].kind == XPAR_VOL_DATA) di++;
    if (old->vol[i].kind == XPAR_VOL_RECOVERY) ri++);
  return true;
}

static void gen_prune_id(const xpar_setd * sd, const xpar_manifest * m,
                         const bool * keep, const xpar_key * key, u8 * id) {
  static const u8 zero[XPAR_SET_ID_LEN];
  xpar_set_id_ctx ctx;
  xpar_wropt w;
  xpar_buf b;
  u32 i;
  xpar_memset(&w, 0, sizeof w);
  xpar_buf_init(&b);
  xpar_setd_write(&b, sd, zero, NULL);
  xpar_set_id_begin(&ctx, key ? key->k_set : NULL,
                    b.data + XPAR_PKT_HDR, b.len - XPAR_PKT_HDR);
  xpar_buf_free(&b);
  Fi(m->count,
    if (keep[i]) {
        xpar_buf e;
        xpar_buf_init(&e);
        xpar_entry_write(&e, &m->entry[i], zero, NULL, &w);
        xpar_set_id_update(&ctx, e.data + XPAR_PKT_HDR, e.len - XPAR_PKT_HDR);
        xpar_buf_free(&e);
      });
  xpar_set_id_final(&ctx, id);
}

static void gen_prune_group(xpar_buf * group, const xpar_chain * c, u32 g,
                            const xpar_setd * sd, const xpar_manifest * m,
                            const u32 * owner, const bool * keep,
                            const xpar_layt * layout, u32 this_volume,
                            const u8 * id) {
  xpar_layt l = *layout;
  xpar_wropt w;
  u32 i;
  xpar_memset(&w, 0, sizeof w);
  l.this_volume = this_volume;
  xpar_buf_init(group);
  xpar_setd_write(group, sd, id, gen_chain_key(c));
  Fi(m->count,
    if (keep[i] && owner[i] == g)
      xpar_entry_write(group, &m->entry[i], id, gen_chain_key(c), &w));
  gen_emit_stored(group, c, g, XPAR_T_POSX, id);
  gen_emit_stored(group, c, g, XPAR_T_SLCR, id);
  gen_emit_stored(group, c, g, XPAR_T_AUTH, id);
  xpar_layt_write(group, &l, id, gen_chain_key(c));
}

static void gen_prune_armoured(xpar_buf * out, const xpar_chain * c,
                               const xpar_chain_vol * v,
                               const xpar_options * o,
                               const gen_rewrite * rw) {
  xpar_arm_prologue pr;
  xpar_armour_params ap, pp;
  xpar_armour * a, * pa;
  xpar_buf plain_out;
  u8 * plain, * region, copy[ARM_COPY_LEN], frame[255];
  sz plen;
  u64 stream_at = 0, stream_len = 0;
  xpar_scan sc;
  xpar_pkt hdr;
  const u8 * body;
  u64 off;
  u32 i;
  int which;
  FATAL_UNLESS(xpar_garm_prologue(v->data, v->len, &pr, &which),
               "internal: an armoured generation has no recoverable "
               "prologue");
  arm_params_of(&pr, &ap);
  plain = arm_extract(&ap, v->data + ARM_HDR_LEN,
                      (u64) v->len - ARM_HDR_LEN, pr.plain_length, &plen,
                      gen_chain_key(c));
  FATAL_UNLESS(plain != NULL, "internal: an armoured generation has no recoverable "
               "packet stream");
  xpar_buf_init(&plain_out);
  gen_rebuild(&plain_out, o, plain, plen, rw, false);
  xpar_free(plain);
  xpar_scan_init(&sc, plain_out.data, plain_out.len, gen_chain_key(c), false);
  while (xpar_scan_next(&sc, &hdr, &body, &off))
    if (xpar_pkt_is(&hdr, XPAR_T_STRM)) {
      stream_at = off + XPAR_PKT_HDR + 16;
      stream_len = hdr.length - XPAR_PKT_HDR - 16;
      break;
    }
  FATAL_UNLESS(stream_at != 0, "internal: a pruned armoured generation no longer carries "
               "its stream");
  xpar_gf_init();
  a = xpar_armour_new(&ap);
  region = xpar_alloc_raw((sz) xpar_armour_size(a, plain_out.len));
  xpar_armour_encode(a, region, plain_out.data, plain_out.len);

  pr.plain_length = plain_out.len;
  pr.armoured_length = xpar_armour_size(a, plain_out.len);
  pr.stream_offset = stream_at;
  pr.stream_length = stream_len;
  xpar_memset(copy, 0, sizeof copy);
  arm_prologue_encode(copy, &pr);
  xpar_armour_defaults(&pp, 8);  pp.n = 255;  pp.k = 223;  pp.depth = 1;
  pa = xpar_armour_new(&pp);
  xpar_memset(frame, 0, sizeof frame);
  xpar_memcpy(frame, copy, ARM_PLAIN_LEN);
  xpar_armour_encode_frame(pa, frame);
  xpar_memcpy(copy + ARM_PLAIN_LEN, frame + pp.k, 32);
  Fi(3, xpar_buf_put(out, copy, sizeof copy));
  xpar_buf_put(out, region, (sz) pr.armoured_length);
  xpar_armour_free(pa);  xpar_armour_free(a);
  xpar_free(region);  xpar_buf_free(&plain_out);
}

typedef struct {
  char * old_path;
  char * new_path;
  char * stage;
  char * backup;
  u32 order;
  bool move, index, published;
} gen_prune_file;

typedef struct {
  gen_prune_file * f;
  u32 count, cap;
} gen_prune_tx;

static i64 gen_prune_find(const gen_prune_tx * t, const char * path) {
  u32 i;
  Fi(t->count, if (gen_path_equal(t->f[i].old_path, path)) return i);
  return -1;
}

static gen_prune_file * gen_prune_add(gen_prune_tx * t,
                                      const char * old_path) {
  gen_prune_file * f;
  i64 found = gen_prune_find(t, old_path);
  if (found >= 0) return &t->f[found];
  if (t->count == t->cap) {
    t->cap = t->cap ? t->cap * 2 : 16;
    t->f = xpar_realloc(t->f, (sz) t->cap * sizeof *t->f);
  }
  f = &t->f[t->count++];
  xpar_memset(f, 0, sizeof *f);
  f->old_path = xpar_strdup(old_path);
  return f;
}

static void gen_prune_output(gen_prune_tx * t, const char * old_path,
                             const char * new_path, char * stage, bool move,
                             bool index, u32 order) {
  gen_prune_file * f = gen_prune_add(t, old_path);
  FATAL_UNLESS(f->new_path == NULL,
               "internal: a pruned volume was given two replacements");
  f->new_path = xpar_strdup(new_path);
  f->stage = stage;  f->move = move;  f->index = index;  f->order = order;
}

static void gen_prune_tx_free(gen_prune_tx * t) {
  u32 i;
  Fi(t->count,
    xpar_free(t->f[i].old_path);  xpar_free(t->f[i].new_path);
    xpar_free(t->f[i].stage);     xpar_free(t->f[i].backup));
  xpar_free(t->f);  xpar_memset(t, 0, sizeof *t);
}

static void gen_prune_discard_stages(gen_prune_tx * t) {
  u32 i;
  Fi(t->count, if (t->f[i].stage) xpar_remove(t->f[i].stage));
}

static void gen_prune_commit(gen_prune_tx * t, const char * sync_path) {
  u32 i, j;
  int saved = 0;
  gen_maint m;
  /*  If a crash lands between two index renames, every visible child must
      already have a visible parent. Keep index publication oldest first
      regardless of readdir order in the chain collector.  */
  for (i = 1; i < t->count; i++) {
    gen_prune_file f = t->f[i];
    u32 key = f.index ? f.order + 1 : 0;
    j = i;
    while (j && (t->f[j - 1].index ? t->f[j - 1].order + 1 : 0) > key)
      { t->f[j] = t->f[j - 1];  j--; }
    t->f[j] = f;
  }
  /*  A canonical target may currently be one of the old chain's names. Any
     other occupant is unrelated and is never overwritten by rotation.  */
  Fi(t->count,
    if (t->f[i].new_path) {
      for (j = i + 1; j < t->count; j++)
        FATAL_UNLESS(!t->f[j].new_path ||
                     !gen_path_equal(t->f[i].new_path, t->f[j].new_path),
                     "internal: two surviving volumes share a final name");
      if (gen_exists(t->f[i].new_path) &&
          gen_prune_find(t, t->f[i].new_path) < 0) {
        gen_prune_discard_stages(t);
        FATAL("'%s' is not part of this chain; prune will not overwrite it",
              t->f[i].new_path);
      }
    });
  Fi(t->count,
    if (gen_exists(t->f[i].old_path)) {
      t->f[i].backup = gen_unused_path(t->f[i].old_path, "xpar-prune-old",
                                       "GPR", "BAK", i);
      if (!t->f[i].backup) {
        gen_prune_discard_stages(t);
        FATAL("cannot choose a rollback name for '%s'", t->f[i].old_path);
      }
    });
  /*  Journal the transaction before its first move.  */
  xpar_memset(&m, 0, sizeof m);
  m.op = XPAR_MAINT_PRUNE;
  Fi(t->count,
    if (t->f[i].backup)
      gen_maint_add(&m, XPAR_MAINT_MOVE, t->f[i].old_path, t->f[i].backup, 0));
  Fj(2,
    Fi(t->count,
      if (t->f[i].new_path) {
        if ((j == 0 && t->f[i].index) || (j == 1 && !t->f[i].index)) continue;
        /*  A rotated volume is also its rollback stage.  */
        if (t->f[i].move) {
          if (t->f[i].backup)
            gen_maint_add(&m, XPAR_MAINT_PUBLISH, t->f[i].backup,
                          t->f[i].new_path, XPAR_MAINT_KEEP);
        } else if (t->f[i].stage)
          gen_maint_add(&m, XPAR_MAINT_PUBLISH, t->f[i].stage,
                        t->f[i].new_path, 0);
      }));
  if (!gen_maint_write(&m, sync_path)) {
    gen_maint_free(&m);
    gen_prune_discard_stages(t);
    FATAL_IO("cannot journal pruning of '%s'; no files moved",
             sync_path);
  }
  Fi(t->count,
    if (t->f[i].backup &&
        xpar_rename(t->f[i].old_path, t->f[i].backup) != 0) {
      saved = xpar_errno();  goto rollback;
    });
  if (xpar_fsync_dir(sync_path) != 0) { saved = xpar_errno();  goto rollback; }
  /*  Publish non-index volumes first.  */
  Fj(2,
    Fi(t->count,
      if (t->f[i].new_path) {
        if ((j == 0 && t->f[i].index) || (j == 1 && !t->f[i].index)) continue;
        if (t->f[i].move) {
          if (!t->f[i].backup ||
              xpar_rename(t->f[i].backup, t->f[i].new_path) != 0) {
            saved = xpar_errno();  goto rollback;
          }
        } else if (!t->f[i].stage ||
                   xpar_rename(t->f[i].stage, t->f[i].new_path) != 0) {
          saved = xpar_errno();  goto rollback;
        }
        t->f[i].published = true;
      }));
  if (xpar_fsync_dir(sync_path) != 0) { saved = xpar_errno();  goto rollback; }
  Fi(t->count,
    if (t->f[i].backup && !t->f[i].move &&
        xpar_remove(t->f[i].backup) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot remove rollback volume '%s': "
                   "%s\n", t->f[i].backup, xpar_strerror(xpar_errno())));
  if (xpar_fsync_dir(sync_path) != 0)
    xpar_fprintf(xpar_stderr, "xpar: cannot sync the directory after "
                 "removing rollback volumes: %s\n",
                 xpar_strerror(xpar_errno()));
  gen_maint_done(&m);
  return;

rollback:
  /*  Restore rollback names.  */
  Fi(t->count,
    if (t->f[i].published && !t->f[i].move &&
        xpar_remove(t->f[i].new_path) != 0)
      xpar_fprintf(xpar_stderr, "xpar: warning: unreferenced '%s' remains: "
                   "%s\n", t->f[i].new_path, xpar_strerror(xpar_errno())));
  Fi(t->count,
    if (t->f[i].published && t->f[i].move &&
        xpar_rename(t->f[i].new_path, t->f[i].backup) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot move '%s' back to '%s': %s\n",
                   t->f[i].new_path, t->f[i].backup,
                   xpar_strerror(xpar_errno())));
  Fi(t->count,
    if (t->f[i].backup &&
        xpar_rename(t->f[i].backup, t->f[i].old_path) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot restore '%s': %s; the "
                   "original remains at '%s'\n", t->f[i].old_path,
                   xpar_strerror(xpar_errno()), t->f[i].backup));
  Fi(t->count, if (t->f[i].stage && !t->f[i].published) xpar_remove(t->f[i].stage));
  if (xpar_fsync_dir(sync_path) != 0)
    xpar_fprintf(xpar_stderr, "xpar: warning: cannot sync the directory "
                 "after rollback: %s\n", xpar_strerror(xpar_errno()));
  gen_maint_done(&m);
  FATAL_IO("cannot publish the pruned chain: %s", xpar_strerror(saved));
}

int xpar_op_prune(const xpar_options * o) {
  xpar_chain c;
  xpar_manifest m;
  u32 * owner = NULL;
  bool * removed;
  u32 head, g, i, k, survivors = 0;
  u64 orphans = 0, reclaim = 0;
  u8 (* new_id)[XPAR_SET_ID_LEN];
  u32 * new_generation;
  u64 * new_base;
  gen_prune_tx tx;

  xpar_gchain_load(o, &c);
  gen_refuse_unreadable(&c, "prune");
  head = xpar_gchain_select(&c, NULL);
  removed   = xpar_calloc(c.gen_count, sizeof *removed);
  new_id    = (u8 (*)[XPAR_SET_ID_LEN]) xpar_calloc(c.gen_count,
                                                    XPAR_SET_ID_LEN);
  new_generation = xpar_calloc(c.gen_count, sizeof *new_generation);
  new_base = xpar_calloc(c.gen_count, sizeof *new_base);
  xpar_memset(&tx, 0, sizeof tx);

  if (o->have_before) {
    u32 lim = (u32) o->before.number;
    if (o->before.by_id) lim = c.gen[xpar_gchain_select(&c, &o->before)].
                                 sd.generation;
    for (g = 0; g < c.gen_count; g++)
      if (c.gen[g].sd.generation < lim) removed[g] = true;
  }
  Fi(o->gen_count,
    removed[xpar_gchain_select(&c,
    &o->gens[i])] = true);
  for (g = 0; g < c.gen_count; g++) if (!removed[g]) survivors++;
  if (survivors == c.gen_count)
    FATAL("specify --before=G or --generation=G");
  if (!survivors)
    FATAL("cannot remove every generation; delete the volumes instead");
  if (removed[head])
    FATAL("cannot prune newest generation %" PRIu32
          "; prune removes older snapshots",
          c.gen[head].sd.generation);

  xpar_gchain_manifest(&c, head, &m, &owner);

  for (g = 0; g < c.gen_count; g++) {
    u64 dep = 0, vbytes;
    /*  Dependencies matter only for removed generations.  */
    if (!removed[g]) continue;
    Fi(m.count,
      bool hit = owner[i] == g;
      for (k = 0; k < m.entry[i].extent_count && !hit; k++) {
        i64 h = xpar_gchain_gen_of(&c, m.entry[i].extents[k].stream_offset,
                                   m.entry[i].extents[k].length);
        if (h == (i64) g) hit = true;
      }
      if (hit) dep++);
    /*  Avoid rescanning and restatting the generation's volumes.  */
    vbytes = gen_volume_bytes(&c, g);
    reclaim += vbytes;
    xpar_fprintf(gen_hout(o),
                 "  gen %-3" PRIu32 ": %" PRIu64 " bytes of stream, %" PRIu64 " bytes of volumes, "
                 "%" PRIu32 " entries owned\n", c.gen[g].sd.generation,
                 c.gen[g].sd.stream_length,
                 vbytes,
                 c.gen[g].sd.file_count);
    xpar_fprintf(gen_hout(o),
                 "           %" PRIu64 " of generation %" PRIu32 "'s %"
                 PRIu32 " entries still depend "
                 "on it\n", dep,
                 c.gen[head].sd.generation, m.count);
  }

  Fi(m.count, if (gen_orphaned(&c, &m.entry[i], owner[i], removed)) orphans++);

  if (orphans && !o->force) {
    xpar_fprintf(gen_hout(o),
                 "refusing: %" PRIu64 " entries in generation %" PRIu32
                 " would become unrecoverable.\n", orphans,
                 c.gen[head].sd.generation);
    xpar_fprintf(gen_hout(o),
                 "run 'xpar consolidate' first, or use --force to accept "
                 "the loss.\n");
    xpar_free(owner);  xpar_manifest_free(&m);
    xpar_free(removed);  xpar_free(new_id);
    xpar_free(new_generation);  xpar_free(new_base);
    xpar_gchain_free(&c);
    gen_json_result(o, "prune", NULL, 0, "refused", XPAR_EXIT_USAGE);
    return XPAR_EXIT_USAGE;
  }
  if (orphans) {
    xpar_fprintf(gen_hout(o), "these entries will be dropped from every "
                 "surviving manifest:\n");
    Fi(m.count,
      if (gen_orphaned(&c, &m.entry[i], owner[i], removed))
        xpar_fprintf(gen_hout(o), "  %.*s\n", (int) m.entry[i].name_len,
                     m.entry[i].name));
  }
  if (o->dry_run) {
    xpar_fprintf(gen_hout(o), "would reclaim %" PRIu64 " bytes of volumes.\n",
                 reclaim);
    xpar_free(owner);  xpar_manifest_free(&m);
    xpar_free(removed);  xpar_free(new_id);
    xpar_free(new_generation);  xpar_free(new_base);
    gen_json_result(o, "prune", c.gen[head].set_id,
                    c.gen[head].sd.generation, "dry-run", XPAR_EXIT_OK);
    xpar_gchain_free(&c);
    return XPAR_EXIT_OK;
  }
  if (orphans && o->force && xpar_is_tty(xpar_stdin)) {
    char answer[8];
    sz n;
    xpar_fprintf(xpar_stderr,
                 "xpar: permanently drop these entries and prune the "
                 "generations? [y/N] ");
    n = xpar_read(xpar_stdin, answer, sizeof answer);
    if (!n || (answer[0] != 'y' && answer[0] != 'Y')) {
      xpar_fprintf(xpar_stderr,
                   "xpar: prune cancelled; nothing was written\n");
      xpar_free(owner);  xpar_manifest_free(&m);
      xpar_free(removed);  xpar_free(new_id);
      xpar_free(new_generation);  xpar_free(new_base);
      xpar_gchain_free(&c);
      return XPAR_EXIT_USAGE;
    }
  }
  gen_require_write_key(&c, "prune");

  { u32 rank = 0;
    u64 base = 0;
    for (g = 0; g < c.gen_count; g++) if (!removed[g]) {
      new_generation[g] = rank++;
      new_base[g] = base;
      base += c.gen[g].sd.stream_length;
    }
  }

  /*  The transaction owns every old packet, bare data and label pathname,
      including removed generations. They are all moved to rollback names
      before a replacement becomes visible.  */
  Fi(c.vol_count, gen_prune_add(&tx, c.vol[i].path));
  for (g = 0; g < c.gen_count; g++) if (c.gen[g].layt_body) {
    xpar_layt l;
    if (xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &l) != XPAR_OK)
      FATAL_FORMAT("generation %" PRIu32 " has a malformed volume layout",
                   c.gen[g].sd.generation);
    Fi(l.count,
      if (l.vol[i].kind == XPAR_VOL_DATA) {
            char * data = xpar_path_join(c.dir, l.vol[i].name);
            char * label;
            gen_prune_add(&tx, data);
            label = xpar_vname_label(data);
            if (gen_exists(label)) gen_prune_add(&tx, label);
            xpar_free(label);  xpar_free(data);
          });
    xpar_layt_free(&l);
  }

  /*  Rebuild survivors oldest-first, restamping FILE extents while retaining
      recovery and tag bodies.  */
  for (g = 0; g < c.gen_count; g++) {
    xpar_manifest gm;
    u32 * gown = NULL;
    bool * keep;
    xpar_setd sd;
    xpar_layt old_layt, layt;
    u32 kept = 0;

    if (removed[g]) continue;
    xpar_gchain_manifest(&c, g, &gm, &gown);
    keep = xpar_calloc(gm.count ? gm.count : 1, sizeof *keep);
    Fi(gm.count,
      keep[i] = !gen_orphaned(&c, &gm.entry[i], gown[i], removed);
      if (keep[i]) kept++);
    if (!kept)
      FATAL("generation %" PRIu32
            " would be left with no entries at all; that is a "
            "chain with nothing in it, so nothing was written",
            c.gen[g].sd.generation);

    gen_prune_rebase(&c, &gm, keep, removed, new_base);
    FATAL_UNLESS(gen_prune_layout(&c, g, new_generation[g],
                                  &old_layt, &layt),
                 "internal: a surviving generation has no volume layout");

    sd = c.gen[g].sd;
    sd.generation = new_generation[g];
    sd.stream_base = new_base[g];
    sd.file_count = kept;
    sd.file_id = (u8 (*)[XPAR_SET_ID_LEN]) xpar_calloc(kept,
                                                       XPAR_SET_ID_LEN);
    for (i = 0, k = 0; i < gm.count; i++)
      if (keep[i]) xpar_memcpy(sd.file_id[k++], gm.entry[i].file_id,
                               XPAR_SET_ID_LEN);
    xpar_memset(sd.parent_set_id, 0, XPAR_SET_ID_LEN);
    if (sd.generation) {
      u32 parent = g;
      while (parent && removed[--parent]) { }
      FATAL_UNLESS(parent < g && !removed[parent],
                   "internal: a pruned generation above zero has no "
                   "surviving parent");
      xpar_memcpy(sd.parent_set_id, new_id[parent], XPAR_SET_ID_LEN);
    }
    gen_prune_id(&sd, &gm, keep, gen_chain_key(&c), new_id[g]);

    Fi(c.vol_count,
      xpar_buf out, group;
      gen_rewrite rw;
      u32 this_vol;
      char * target;
      char * stage;
      if (c.vol[i].gen != g) continue;
      this_vol = c.vol[i].volume_kind == XPAR_VOL_RECOVERY
                   ? c.vol[i].volume_index : XPAR_VOL_STANDALONE;
      gen_prune_group(&group, &c, g, &sd, &gm, gown, keep, &layt,
                      this_vol, new_id[g]);
      xpar_memset(&rw, 0, sizeof rw);
      rw.group = group.data;  rw.group_len = group.len;
      rw.set_id = new_id[g];  rw.this_vol = this_vol;
      rw.key = gen_chain_key(&c);
      xpar_buf_init(&out);
      if (c.vol[i].armoured_file)
        gen_prune_armoured(&out, &c, &c.vol[i], o, &rw);
      else
        gen_rebuild(&out, o, c.vol[i].data, c.vol[i].len, &rw, false);
      if (c.vol[i].volume_kind == XPAR_VOL_RECOVERY &&
          c.vol[i].volume_index < layt.count)
        target = xpar_path_join(c.dir, layt.vol[c.vol[i].volume_index].name);
      else
        target = xpar_vname_index(c.base, new_generation[g]);
      stage = gen_stage_whole(target, out.data, out.len);
      gen_prune_output(&tx, c.vol[i].path, target, stage, false,
                       c.vol[i].volume_kind == XPAR_VOL_INDEX ||
                       c.vol[i].armoured_file, new_generation[g]);
      xpar_free(target);
      xpar_buf_free(&out);
      xpar_buf_free(&group));

    /*  Bare data is already byte-correct and is moved without copying. An
       optional label is packet-bearing and therefore gets the same new
       critical group and set_id as the other volume copies.  */
    { u32 di;
      for (di = 0; di < old_layt.count; di++)
        if (old_layt.vol[di].kind == XPAR_VOL_DATA) {
          char * old_data = xpar_path_join(c.dir, old_layt.vol[di].name);
          char * new_data = xpar_path_join(c.dir, layt.vol[di].name);
          char * old_label, * new_label;
          if (gen_exists(old_data))
            gen_prune_output(&tx, old_data, new_data, NULL, true, false, 0);
          old_label = xpar_vname_label(old_data);
          new_label = xpar_vname_label(new_data);
          if (gen_exists(old_label)) {
            i64 ti = gen_prune_find(&tx, old_label);
            if (ti < 0 || !tx.f[ti].new_path) {
              u8 * data;
              sz len;
              xpar_buf group, out;
              gen_rewrite rw;
              char * stage;
              data = gen_read_whole(old_label, &len, true);
              gen_prune_group(&group, &c, g, &sd, &gm, gown, keep, &layt,
                              di, new_id[g]);
              xpar_memset(&rw, 0, sizeof rw);
              rw.group = group.data;  rw.group_len = group.len;
              rw.set_id = new_id[g];  rw.this_vol = di;
              rw.key = gen_chain_key(&c);
              xpar_buf_init(&out);
              gen_rebuild(&out, o, data, len, &rw, false);
              stage = gen_stage_whole(new_label, out.data, out.len);
              gen_prune_output(&tx, old_label, new_label, stage, false,
                               false, 0);
              xpar_buf_free(&out);  xpar_buf_free(&group);  xpar_free(data);
            }
          }
          xpar_free(old_label);  xpar_free(new_label);
          xpar_free(old_data);  xpar_free(new_data);
        }
    }

    xpar_free(sd.file_id);
    xpar_free(keep);  xpar_free(gown);
    xpar_manifest_free(&gm);
    xpar_layt_free(&old_layt);  xpar_layt_free(&layt);
  }

  gen_prune_commit(&tx, c.base);
  xpar_fprintf(gen_hout(o),
               "pruned %" PRIu32 " generation%s, reclaimed %" PRIu64 " bytes; %" PRIu64 " %s "
               "dropped from the surviving manifests.\n",
               c.gen_count - survivors, PLURAL(c.gen_count - survivors),
               reclaim, orphans,
               orphans == 1 ? "entry was" : "entries were");
  gen_json_result(o, "prune", c.gen[head].set_id,
                  c.gen[head].sd.generation, "ok", XPAR_EXIT_OK);

  xpar_free(owner);  xpar_manifest_free(&m);
  gen_prune_tx_free(&tx);
  xpar_free(removed);  xpar_free(new_id);
  xpar_free(new_generation);  xpar_free(new_base);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

/*  Remove only staged manifest paths, deepest-first, then their directory.  */
static void gen_unstage_owned(const xpar_manifest * m, const char * dir) {
  u32 * order, i, j;
  if (!dir) return;
  order = xpar_alloc_raw((m->count ? m->count : 1) * sizeof *order);
  Fi(m->count, order[i] = i);
  /*  Reverse name order puts a child before the parent it lives in.  */
  for (i = 1; i < m->count; i++) {
    u32 t = order[i];
    j = i;
    while (j && xpar_name_cmp(m->entry[order[j - 1]].name,
                              m->entry[order[j - 1]].name_len,
                              m->entry[t].name, m->entry[t].name_len) < 0) {
      order[j] = order[j - 1];  j--;
    }
    order[j] = t;
  }
  Fi(m->count,
    const xpar_entry * e = &m->entry[order[i]];
    xpar_path_status why;
    char * path = xpar_path_resolve(dir, e->name, e->name_len,
                                    XPAR_PATH_LEAF_LINK, &why);
    if (!path) continue;
    if (e->entry_type == XPAR_ENTRY_DIR) (void) xpar_rmdir(path);
    else                                 (void) xpar_remove(path);
    xpar_free(path));
  xpar_free(order);
  (void) xpar_rmdir(dir);
}

/*  Extract owned data to a temporary tree for consolidation.  */
static char * gen_stage_owned(const xpar_options * o, const char * base,
                              const xpar_manifest * m) {
  xpar_options ex = *o;
  char * parent = xpar_path_dir(base);
  char * stem = xpar_path_join(parent, ".xpar-consolidate-");
  char * dir = xpar_stage_dir(stem, "GCO");
  xpar_free(parent);  xpar_free(stem);
  if (!dir)
    FATAL_IO("cannot create a staging directory beside '%s'", base);
  ex.verb        = XPAR_VERB_EXTRACT;
  ex.to_dir      = dir;
  ex.force       = true;
  ex.dry_run     = false;
  ex.replace     = false;
  ex.json        = false;
  ex.chain       = false;
  ex.exit_on_change = false;
  if (xpar_op_extract(&ex) != XPAR_EXIT_OK) {
    /*  Remove partial staging.  */
    gen_unstage_owned(m, dir);
    xpar_free(dir);
    FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
               "cannot stage this archive for consolidation; repair it first");
  }
  return dir;
}

int xpar_op_consolidate(const xpar_options * caller) {
  xpar_options eff = *caller;
  const xpar_options * o = &eff;
  xpar_chain c;
  xpar_manifest m;
  xpar_posix_rec ** tab;
  u32 * tabn;
  u32 * owner = NULL;
  gen_write_req rq;
  bool * owned = NULL;
  u32 head, i, caps, bad = 0, unreadable = 0, io_bad = 0, ratio_gen = 0;
  bool owned_layout;
  char * stage_tree = NULL;
  u64 live = 0, total = 0;
  bool warn_posix = false, inherited_r;
  f64 old_ratio = 0.0;
  const char * base;
  char * stage_base = NULL, * cache_path = NULL, * stage_cache = NULL;
  xpar_chunk_index chunk_cache;
  gen_consol_tx ctx;

  xpar_memset(&ctx, 0, sizeof ctx);
  xpar_memset(&chunk_cache, 0, sizeof chunk_cache);
  xpar_memset(&rq, 0, sizeof rq);
  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "consolidate");
  gen_refuse_unreadable(&c, "consolidate");
  if (c.authenticated && o->auth_only && !c.auth_only)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "a consolidated set must keep its chain's authentication "
               "mode; this chain retains public verification hashes");
  head = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  eff.layout = gen_chain_layout(o, &c, head, false);
  xpar_cli_armour_for_layout(&eff, eff.layout);
  eff.armour = gen_chain_armour(&eff, &c, head, eff.layout);
  inherited_r = gen_inherit_recovery(&eff, &c, head, true, &old_ratio,
                                     &ratio_gen);
  owned_layout = c.gen[head].sd.layout != XPAR_LAYOUT_SIDECAR;
  if (!o->quiet && c.base) gen_report_stale_stage(o, c.base);
  base = o->output ? o->output : c.base;
  if (!base) FATAL("this set has no base name; pass --output");
  /*  Dry runs need no destination.  */
  if (!o->output && !o->replace && !o->dry_run)
    FATAL("consolidate writes a new generation-0 set; use --output=BASE or "
          "--replace");

  xpar_gchain_manifest(&c, head, &m, &owner);
  tab  = xpar_calloc(c.gen_count, sizeof *tab);
  tabn = xpar_calloc(c.gen_count, sizeof *tabn);
  Fi(c.gen_count,
    tabn[i] = xpar_gchain_posix(&c, i,
    &tab[i]));
  Fi(c.gen_count, total += c.gen[i].sd.stream_length);
  /*  Count shared extents once.  */
  { xpar_occindex ix;
    u64 at;
    xpar_occindex_build(&m, &ix);
    for (at = 0; at < total; ) {
      xpar_occurrence oc;
      u64 run = 0, nxt = xpar_occindex_next(&ix, at, total);
      if (nxt > at) { at = nxt;  continue; }
      if (xpar_occindex_canonical(&ix, at, &oc, &run) && run) {
        if (run > total - at) run = total - at;
        live += run;  at += run;
      } else at++;
    }
    xpar_occindex_free(&ix);
  }

  /*  Dry runs need no staging or archive extraction.  */
  if (o->dry_run) {
    xpar_fprintf(gen_hout(o),
                 "  chain      : %" PRIu32 " generations, %" PRIu32 " entries\n"
                 "  stream     : %" PRIu64 " bytes across the chain, %" PRIu64 " still "
                 "referenced (%.1f%%)\n"
                 "  reclaim    : %" PRIu64 " bytes of stream\n"
                 "  cost       : read %" PRIu64 " bytes, one full encode\n",
                 c.gen_count, m.count, total,
                 live,
                 total ? 100.0 * (f64) live / (f64) total : 100.0,
                 (total - live),
                 live);
    goto done;
  }

  /*  Stage owned data for the refresh pass.  */
  if (owned_layout) {
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: staging %" PRIu64
                   " self-contained bytes for consolidation\n", live);
    stage_tree = gen_stage_owned(caller, base, &m);
    eff.base_dir = stage_tree;
  }

  caps = xpar_fs_caps(o->base_dir ? o->base_dir : ".");
  owned = xpar_calloc(m.count ? m.count : 1, sizeof *owned);
  Fi(m.count,
    char * path = gen_entry_path(o, &m.entry[i]);
    u8 want[32];
    u32 pi = m.entry[i].posix_index;
    u32 og = owner[i];
    owned[i] = true;
    xpar_memcpy(want, m.entry[i].content_hash, 32);
    if (m.entry[i].entry_type == XPAR_ENTRY_HARDLINK) { xpar_free(path);  continue; }
    if (!gen_refresh(&m.entry[i], path, o, caps, &warn_posix,
                     gen_chain_key(&c), c.auth_only)) {
      int err = xpar_errno();
      xpar_stat_t rst;
      bool refused = xpar_lstat(path, &rst) == 0 &&
                     (rst.is_regular || rst.is_dir || rst.is_symlink);
      if (owned_layout) { bad++;  unreadable++; }
      else if (refused) {
        xpar_fprintf(xpar_stderr, "xpar: cannot read '%s': %s\n", path,
                     xpar_strerror(err));
        io_bad++;
      } else {
        xpar_fprintf(xpar_stderr, "xpar: cannot read '%s'\n", path);
        bad++;
      }
      xpar_free(path);  continue;
    }
    if (xpar_memcmp(want, m.entry[i].content_hash, 32)) {
      xpar_fprintf(xpar_stderr,
                   "xpar: '%.*s' does not match the content the chain "
                   "records for it\n", (int) m.entry[i].name_len,
                   m.entry[i].name);
      bad++;
    }
    if (pi != XPAR_ABSENT_U32 && pi < tabn[og])
      m.entry[i].posix_index = xpar_posix_intern(&m, &tab[og][pi]);
    m.source[i] = path);

  /*  Missing staged entries indicate unrecoverable archive damage.  */
  if (owned_layout && unreadable)
    FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
               "%" PRIu32 " of %" PRIu32 " entries cannot be extracted "
               "from this self-contained chain; repair it first",
               unreadable, m.count);
  if (io_bad)
    FATAL_CODE(XPAR_EXIT_IO,
               "%" PRIu32 "/%" PRIu32 " entries unreadable; nothing written",
               io_bad, m.count);
  if (bad && !o->force)
    FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
               "%" PRIu32 " entries do not match the chain; repair or pass "
               "--force", bad);
  if (bad)
    xpar_fprintf(xpar_stderr, "xpar: --force recorded %" PRIu32
                 " mismatched entr%s as found\n", bad,
                 bad == 1 ? "y" : "ies");

  { gen_merge g;
    xpar_memset(&g, 0, sizeof g);
    g.m         = m;
    g.owned     = owned;
    g.reuse     = xpar_calloc(m.count ? m.count : 1, sizeof *g.reuse);
    g.m.slice_size = o->slice_size;
    gen_repack(&g, o, NULL, NULL, 0,
               o->dedup == XPAR_DEDUP_CHUNK ? &chunk_cache : NULL);
    m = g.m;
    xpar_free(g.reuse);
  }

  gen_check_manifest(&m);

  xpar_memset(&rq, 0, sizeof rq);
  rq.o = o;  rq.m = &m;  rq.owned = owned;
  rq.generation = 0;  rq.stream_base = 0;  rq.parent_set_id = NULL;
  if (o->replace) {
    stage_base = gen_unused_base(base, "xpar-consolidate");
    rq.base = stage_base;
    rq.layout_base = base;
  } else rq.base = base;
  rq.quiet = o->quiet;
  rq.auth_only = c.authenticated ? c.auth_only : o->auth_only;
  if (chunk_cache.slot) {
    static const u8 unbound[XPAR_SET_ID_LEN];
    u64 average = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
    cache_path = xpar_vname_cache(base);
    stage_cache = gen_unused_path(cache_path, "xpar-cache", "GCC", "TMP",
                                  0);
    if (!stage_cache ||
        !xpar_chunk_cache_write(stage_cache, unbound, average,
                                &chunk_cache)) {
      if (o->verbose)
        xpar_fprintf(xpar_stderr, "xpar: cannot stage chunk cache '%s'\n",
                     cache_path);
      xpar_free(stage_cache);  stage_cache = NULL;
    }
    xpar_chunk_index_free(&chunk_cache);
  }
  gen_write_set(&rq);
  if (o->replace)
    gen_commit_consolidation(&c, o, stage_base, base, &rq.plan, &ctx);
  /*  Verify before removing rollback names.  */
  if (!o->no_verify_after) {
    if (o->replace) {
      gen_vol * final;
      u32 final_n;
      final = gen_volumes(o, rq.plan.recovery, base, 0, &final_n);
      xpar_verify_written_set(o, final[0].name);
      gen_volumes_free(final, final_n);
    } else xpar_verify_written_set(o, rq.index_path);
  }
  gen_consol_finish(&ctx);
  if (stage_cache &&
      (!xpar_chunk_cache_rebind(stage_cache, rq.set_id) ||
       !gen_publish_cache(stage_cache, cache_path))) {
    if (o->verbose)
      xpar_fprintf(xpar_stderr, "xpar: cannot publish chunk cache '%s'\n",
                   cache_path);
    xpar_remove(stage_cache);
  }
  if (!o->quiet) {
    xpar_fprintf(xpar_stderr,
                 "xpar: collapsed %" PRIu32 " generations into one: %" PRIu32 " %s, "
                 "%" PRIu64 " stream bytes, %" PRIu64 " recovery slice%s "
                 "(R was %" PRIu64 " at %.1f%%, now %" PRIu64 " at %.1f%%)\n",
                 c.gen_count,
                 m.count, m.count == 1 ? "entry" : "entries",
                 m.stream_length,
                 rq.plan.recovery,
                 PLURAL(rq.plan.recovery),
                 gen_gen_recovery(&c, ratio_gen), old_ratio,
                 rq.plan.recovery,
                 rq.plan.geom.slice_count
                   ? 100.0 * (f64) rq.plan.recovery /
                     (f64) rq.plan.geom.slice_count : 0.0);
    if (inherited_r)
      xpar_fprintf(xpar_stderr,
                   "xpar: without -r, inherited generation %" PRIu32
                   "'s %.1f%%, the chain's widest\n",
                   c.gen[ratio_gen].sd.generation, old_ratio);
  }
  gen_warn_thinner(o, old_ratio, rq.plan.recovery,
                   rq.plan.geom.slice_count);

done:
  gen_json_result_field(o, "consolidate",
                        rq.index_path ? rq.set_id : c.gen[head].set_id,
                        rq.index_path ? 0 : c.gen[head].sd.generation,
                        o->dry_run ? "dry-run" : "ok", XPAR_EXIT_OK,
                        "entries_damaged", bad);
  Fi(c.gen_count, if (tab[i]) xpar_gchain_posix_free(tab[i], tabn[i]));
  xpar_free(tab);  xpar_free(tabn);  xpar_free(owned);  xpar_free(owner);
  if (stage_tree) { gen_unstage_owned(&m, stage_tree);  xpar_free(stage_tree); }
  xpar_free(stage_base);  xpar_free(cache_path);  xpar_free(stage_cache);
  xpar_free(rq.index_path);
  xpar_chunk_index_free(&chunk_cache);
  xpar_manifest_free(&m);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

/*  recover.  */

/*  Lay out one index or recovery volume from freshly encoded tables, so
    that `recover` and `repair` publish exactly the same bytes.  */
static void gen_recovery_volume(xpar_buf * out, const xpar_options * o,
                                const xpar_chain * c, u32 g,
                                const xpar_layt * layt, u32 target,
                                const gen_plan * pl, gen_tables * t,
                                u8 * rec_scratch) {
  xpar_volh vh;
  u64 e;
  /*  Preserve stored wrapping parameters.  */
  xpar_armour_params rap, tap;
  bool wrap_r = xpar_gchain_wrap_armour(c, g, true, &rap);
  bool wrap_t = xpar_gchain_wrap_armour(c, g, false, &tap);
  xpar_armour * ra = wrap_r ? gen_rcvs_armour(o, 0, &rap) : NULL;
  xpar_options wo = *o;
  if (wrap_t) gen_wrap_options(&wo, &tap);
  xpar_buf_init(out);
  xpar_memset(&vh, 0, sizeof vh);
  vh.volume_index = layt->vol[target].kind == XPAR_VOL_INDEX
                      ? XPAR_VOL_STANDALONE : target;
  vh.volume_kind  = layt->vol[target].kind;
  xpar_volh_write(out, &vh, c->gen[g].set_id, gen_chain_key(c));
  { xpar_buf group;
    u64 payload = layt->vol[target].byte_length * pl->geom.slice_size;
    u64 crit_bytes;
    u32 rec_count = 0, rec_index = 0, q;
    xpar_armour_params ap;
    bool armoured, carry;
    xpar_buf_init(&group);
    gen_group_stored(&group, c, g, layt,
                     layt->vol[target].kind == XPAR_VOL_INDEX
                       ? XPAR_VOL_STANDALONE : target, c->gen[g].set_id);
    /*  Recover armour parameters from the set, not CLI defaults.  */
    armoured = xpar_gchain_crit_armour(c, g, &ap);
    /*  Replication uses stored size and the recovery-volume index.  */
    crit_bytes = group.len;
    if (armoured) {
      xpar_buf probe;
      xpar_buf_init(&probe);
      gen_armour_pack_ap(&probe, &ap, group.data, group.len, c->gen[g].set_id,
                         gen_chain_key(c));
      crit_bytes = probe.len;
      xpar_buf_free(&probe);
    }
    for (q = 0; q < layt->count; q++) {
      if (layt->vol[q].kind != XPAR_VOL_RECOVERY) continue;
      if (q < target) rec_index++;
      rec_count++;
    }
    carry = layt->vol[target].kind == XPAR_VOL_INDEX ||
            xpar_replicate_here(crit_bytes, payload, rec_index, rec_count);
    if (carry) {
      if (armoured)
        gen_armour_pack_ap(out, &ap, group.data, group.len, c->gen[g].set_id,
                           gen_chain_key(c));
      else
        xpar_buf_put(out, group.data, group.len);
    }
    xpar_buf_free(&group);
  }
  if (layt->vol[target].kind == XPAR_VOL_INDEX || target == 1)
    gen_emit_tables(out, &wo, wrap_t, t, pl, t->tag_len, c->gen[g].set_id,
                    gen_chain_key(c));
  for (e = layt->vol[target].recovery_first;
       e < layt->vol[target].recovery_first + layt->vol[target].byte_length;
       e++) {
    const u8 * rec = gen_rec_get(t, e, rec_scratch);
    gen_rcvs_emit(out, ra, e, rec, (sz) pl->geom.slice_size,
                  c->gen[g].set_id, gen_chain_key(c));
  }
  gen_crtr_stored(out, c, g, c->gen[g].set_id);
  if (ra) xpar_armour_free(ra);
}

/*  Re-encode volumes containing missing recovery exponents.  */
u64 xpar_gen_regen_recovery(const xpar_options * o, u64 * volumes,
                            const char ** reason, bool dry) {
  xpar_chain c;
  xpar_manifest m;
  xpar_layt layt;
  gen_plan p;
  gen_tables t;
  gen_addrec_file * staged;
  char ** paths;
  u32 * owner = NULL;
  u32 * hit;
  u32 g, i, hits = 0;
  u64 r_total = 0, slices = 0, e;
  u8 * want;
  u8 * rec_scratch = NULL;
  xpar_vset * src;
  int src_rc;

  if (reason) *reason = NULL;
  if (volumes) *volumes = 0;
  xpar_gchain_load(o, &c);
  g = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  if (c.gen[g].sd.layout == XPAR_LAYOUT_ARMOURED || !c.gen[g].layt_body ||
      xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &layt) !=
        XPAR_OK) {
    /*  These layouts name no separate recovery volumes.  */
    xpar_gchain_free(&c);
    return 0;
  }
  gen_require_write_key(&c, "repair");

  Fi(layt.count,
    if (layt.vol[i].kind == XPAR_VOL_RECOVERY)
      r_total = MAX(r_total, layt.vol[i].recovery_first +
                             layt.vol[i].byte_length));
  if (!r_total) { xpar_layt_free(&layt);  xpar_gchain_free(&c);  return 0; }

  /*  Rebuild exponents with no surviving RCVS packet.  */
  src = xpar_vset_open(o);
  want = xpar_calloc((sz) r_total, 1);
  for (e = 0; e < r_total; e++) { u64 n;  want[e] = !xpar_vset_rcvs(src, e, &n); }
  /*  Rebuild each affected volume in full.  */
  hit = xpar_calloc(layt.count ? layt.count : 1, sizeof *hit);
  Fi(layt.count,
    const xpar_vol * v = &layt.vol[i];
    if (v->kind != XPAR_VOL_RECOVERY || !v->name) continue;
    for (e = v->recovery_first; e < v->recovery_first + v->byte_length; e++)
      if (want[e]) { hit[hits++] = i;  slices += v->byte_length;  break; });
  xpar_free(want);
  if (!hits || dry) {
    xpar_vset_close(src);
    if (volumes) *volumes = hits;
    xpar_free(hit);  xpar_layt_free(&layt);  xpar_gchain_free(&c);
    return hits ? slices : 0;
  }
  slices = 0;

  gen_manifest_on_disk(&c, g, o, &m, &owner);
  xpar_memset(&p, 0, sizeof p);
  if (!xpar_geom_from_setd(&c.gen[g].sd, &p.geom))
    FATAL_FORMAT("generation %" PRIu32 "'s geometry is malformed",
                 c.gen[g].sd.generation);
  p.recovery   = r_total;
  p.encode_r   = r_total;
  p.field_log2 = c.gen[g].sd.field_log2;
  p.codec      = c.gen[g].sd.codec;
  p.axis       = c.gen[g].sd.recovery_axis_log2;

  /*  Never derive parity from damaged data.  */
  src_rc = xpar_vset_check(src, o, NULL);
  if (!xpar_vset_stream_intact(src, src_rc)) {
    if (reason) *reason = "the protected data is not intact";
    xpar_vset_close(src);
    xpar_free(hit);  xpar_free(owner);  xpar_manifest_free(&m);
    xpar_layt_free(&layt);  xpar_gchain_free(&c);
    return 0;
  }
  gen_encode(&m, &p, c.gen[g].sd.slice_tag_len, o->memory,
             c.base ? c.base : o->set, gen_chain_key(&c), gen_read_vset,
             src, &t, NULL);
  gen_require_source_tables(src, &t, &p);
  xpar_vset_close(src);
  if (t.rec_spill) rec_scratch = xpar_alloc_raw((sz) t.rec_z);

  /*  Stage all volumes before publishing.  */
  staged = xpar_calloc(hits, sizeof *staged);
  paths  = xpar_calloc(hits, sizeof *paths);
  Fi(hits,
    const xpar_vol * v = &layt.vol[hit[i]];
    xpar_buf out;
    gen_recovery_volume(&out, o, &c, g, &layt, hit[i], &p, &t,
                        rec_scratch);
    paths[i] = xpar_path_join(c.dir, v->name);
    staged[i].stage = gen_stage_whole(paths[i], out.data, out.len);
    staged[i].final = paths[i];
    staged[i].replace = true;
    if (!xpar_verify_written_volume(staged[i].stage, gen_chain_key(&c),
                                    c.gen[g].set_id, hit[i], v->kind,
                                    v->recovery_first, v->byte_length,
                                    p.geom.slice_size)) {
      gen_addrec_discard(staged, i + 1);
      FATAL_CODE(XPAR_EXIT_INTERNAL,
                 "internal: regenerated recovery volume failed verification");
    }
    xpar_buf_free(&out);
    slices += v->byte_length);
  gen_addrec_publish(staged, hits);
  Fi(hits, xpar_free(paths[i]));
  xpar_free(paths);
  if (volumes) *volumes = hits;

  xpar_free(hit);
  gen_tables_free(&t);
  xpar_free(rec_scratch);
  xpar_free(owner);
  xpar_manifest_free(&m);
  xpar_layt_free(&layt);
  xpar_gchain_free(&c);
  return slices;
}

/*  Recreate index volumes the layout names but the directory no longer
    holds; every packet they carry survives in the recovery volumes.  */
u64 xpar_gen_regen_index(const xpar_options * o, u64 * volumes,
                         const char ** reason, bool dry) {
  xpar_chain c;
  xpar_layt layt;
  gen_plan p;
  gen_tables t;
  gen_addrec_file * staged;
  char ** paths;
  const xpar_tags * st;
  xpar_vset * src;
  u32 * hit;
  u32 g, i, hits = 0;

  if (reason) *reason = NULL;
  if (volumes) *volumes = 0;
  xpar_gchain_load(o, &c);
  g = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  if (c.gen[g].sd.layout == XPAR_LAYOUT_ARMOURED || !c.gen[g].layt_body ||
      xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &layt) !=
        XPAR_OK) {
    xpar_gchain_free(&c);
    return 0;
  }
  hit = xpar_calloc(layt.count ? layt.count : 1, sizeof *hit);
  Fi(layt.count,
    const xpar_vol * v = &layt.vol[i];
    xpar_stat_t vst;
    char * path;
    if (v->kind != XPAR_VOL_INDEX || !v->name) continue;
    path = xpar_path_join(c.dir, v->name);
    /*  Lost, or no longer exactly the packets it should hold.  */
    if (xpar_lstat(path, &vst) != 0 || !vst.is_regular ||
        !xpar_verify_volume_tiles(path, gen_chain_key(&c))) hit[hits++] = i;
    xpar_free(path));
  if (!hits || dry) {
    if (volumes) *volumes = hits;
    xpar_free(hit);  xpar_layt_free(&layt);  xpar_gchain_free(&c);
    return hits;
  }
  gen_require_write_key(&c, "repair");
  xpar_memset(&p, 0, sizeof p);
  if (!xpar_geom_from_setd(&c.gen[g].sd, &p.geom))
    FATAL_FORMAT("generation %" PRIu32 "'s geometry is malformed",
                 c.gen[g].sd.generation);
  p.field_log2 = c.gen[g].sd.field_log2;
  p.codec      = c.gen[g].sd.codec;
  p.axis       = c.gen[g].sd.recovery_axis_log2;

  /*  An index volume carries no recovery slices, so the stored tag tables
      are all the payload it needs.  */
  src = xpar_vset_open(o);
  st  = xpar_vset_tags(src);
  xpar_memset(&t, 0, sizeof t);
  t.slice_crc = st->slice_crc;
  t.slice_tag = st->slice_tag;
  t.cell_crc  = st->cell_crc;
  t.tag_len   = st->tag_len;
  staged = xpar_calloc(hits, sizeof *staged);
  paths  = xpar_calloc(hits, sizeof *paths);
  Fi(hits,
    const xpar_vol * v = &layt.vol[hit[i]];
    xpar_buf out;
    gen_recovery_volume(&out, o, &c, g, &layt, hit[i], &p, &t, NULL);
    paths[i] = xpar_path_join(c.dir, v->name);
    staged[i].stage = gen_stage_whole(paths[i], out.data, out.len);
    staged[i].final = paths[i];
    staged[i].replace = true;
    if (!xpar_verify_written_volume(staged[i].stage, gen_chain_key(&c),
                                    c.gen[g].set_id, XPAR_VOL_STANDALONE,
                                    v->kind, v->recovery_first, 0,
                                    p.geom.slice_size)) {
      gen_addrec_discard(staged, i + 1);
      xpar_vset_close(src);
      FATAL_CODE(XPAR_EXIT_INTERNAL,
                 "internal: regenerated index volume failed verification");
    }
    xpar_buf_free(&out));
  /*  Nothing may hold a mapping over a file about to be renamed over.  */
  xpar_vset_close(src);
  gen_addrec_publish(staged, hits);
  Fi(hits, xpar_free(paths[i]));
  xpar_free(paths);
  if (volumes) *volumes = hits;
  xpar_free(hit);
  xpar_layt_free(&layt);
  xpar_gchain_free(&c);
  return hits;
}

/*  Rewrite stale critical packets from the index; `dry` only counts.  */
u64 xpar_gen_rewrite_stale(const xpar_options * o, u64 * volumes,
                           const char ** reason, bool dry) {
  xpar_chain c;
  gen_addrec_file * staged;
  u32 i, hits = 0, n = 0;
  if (reason) *reason = NULL;
  if (volumes) *volumes = 0;
  xpar_gchain_load(o, &c);
  Fi(c.vol_count,
    if (c.vol[i].stale_packets && c.vol[i].gen != XPAR_GEN_NONE &&
        !c.vol[i].armoured_file) hits++);
  if (!hits || dry) { if (volumes) *volumes = hits;  xpar_gchain_free(&c);  return hits; }
  gen_require_write_key(&c, "repair");
  staged = xpar_calloc(hits, sizeof *staged);
  Fi(c.vol_count,
    const xpar_chain_vol * v = &c.vol[i];
    u32 g = v->gen;
    xpar_layt layt;
    bool have_layt;
    xpar_buf group, out;
    gen_rewrite rw;
    if (!v->stale_packets || g == XPAR_GEN_NONE || v->armoured_file)
      continue;
    have_layt = c.gen[g].layt_body &&
      xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &layt) == XPAR_OK;
    xpar_buf_init(&group);
    gen_group_stored(&group, &c, g, have_layt ? &layt : NULL,
                     v->volume_index, c.gen[g].set_id);
    xpar_memset(&rw, 0, sizeof rw);
    rw.group = group.data;  rw.group_len = group.len;
    rw.this_vol = v->volume_index;  rw.set_id = c.gen[g].set_id;
    rw.key = gen_chain_key(&c);
    xpar_buf_init(&out);
    gen_rebuild(&out, o, v->data, v->len, &rw, false);
    staged[n].stage = gen_stage_whole(v->path, out.data, out.len);
    staged[n].final = v->path;
    staged[n].replace = true;
    if (!xpar_verify_written_volume(staged[n].stage, gen_chain_key(&c),
                                    c.gen[g].set_id, v->volume_index,
                                    v->volume_kind, v->recovery_first,
                                    v->recovery_count,
                                    c.gen[g].sd.slice_size)) {
      xpar_buf_free(&out);  xpar_buf_free(&group);
      if (have_layt) xpar_layt_free(&layt);
      gen_addrec_discard(staged, n + 1);
      FATAL_CODE(XPAR_EXIT_INTERNAL,
                 "internal: rewritten stale volume failed verification");
    }
    n++;
    xpar_buf_free(&out);  xpar_buf_free(&group);
    if (have_layt) xpar_layt_free(&layt));
  gen_addrec_publish(staged, n);
  if (volumes) *volumes = n;
  xpar_gchain_free(&c);
  return n;
}

int xpar_op_recover(const xpar_options * o) {
  xpar_chain c;
  xpar_manifest m;
  xpar_layt layt;
  gen_plan p;
  gen_tables t;
  u32 * owner = NULL;
  u32 g, i, target = 0xFFFFFFFFU;
  u64 r_total = 0;
  xpar_buf out;
  char * path;
  u8 * rec_scratch = NULL;
  xpar_vset * source_set;
  int source_rc;

  /*  --to names an output directory, which extract and repair create.  */
  if (o->to_dir && xpar_strlen(o->to_dir) && xpar_mkdir_p(o->to_dir, 0777) != 0) {
    xpar_stat_t dst;
    if (xpar_lstat(o->to_dir, &dst) != 0 || !dst.is_dir)
      FATAL_IO("cannot create '%s': %s", o->to_dir,
               xpar_strerror(xpar_errno()));
  }
  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "recover");
  g = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  if (!c.gen[g].layt_body)
    FATAL_FORMAT("generation %" PRIu32
                 " carries no volume layout, so there is "
                 "nothing to say what the lost volume held",
                 c.gen[g].sd.generation);
  if (xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &layt) != XPAR_OK)
    FATAL_FORMAT("generation %" PRIu32 "'s volume layout is malformed",
                 c.gen[g].sd.generation);

  /*  Reject recovery ranges outside the generation's axis.  */
  { u64 axis = xpar_setd_recovery_limit(&c.gen[g].sd);
    Fi(layt.count,
      const xpar_vol * v = &layt.vol[i];
      if (v->kind != XPAR_VOL_RECOVERY) continue;
      if (!v->byte_length || v->byte_length > axis ||
          v->recovery_first > axis - v->byte_length)
        FATAL_FORMAT("generation %" PRIu32 "'s recovery range exceeds its "
                     "axis", c.gen[g].sd.generation)); }

  Fi(layt.count,
    if (layt.vol[i].kind == XPAR_VOL_RECOVERY)
      r_total = MAX(r_total, layt.vol[i].recovery_first +
                             layt.vol[i].byte_length);
    if (o->volume_name) {
      if (layt.vol[i].name && xpar_path_same(layt.vol[i].name,
                                             o->volume_name))
        target = i;
    } else if (i == (u32) o->volume_index) target = i);
  if (target == 0xFFFFFFFFU) {
    u32 h;
    if (o->volume_name)
      for (h = 0; h < c.gen_count; h++) {
        xpar_layt other;
        if (h == g || !c.gen[h].layt_body) continue;
        if (xpar_layt_read(c.gen[h].layt_body, c.gen[h].layt_len, &other) !=
            XPAR_OK) continue;
        Fi(other.count,
          if (other.vol[i].name &&
              xpar_path_same(other.vol[i].name, o->volume_name)) {
            u32 num = c.gen[h].sd.generation;
            xpar_layt_free(&other);
            FATAL("'%s' belongs to generation %" PRIu32 ", not to generation %"
                  PRIu32 "; "
                  "pass --generation=%" PRIu32, o->volume_name, num,
                  c.gen[g].sd.generation, num);
          });
        xpar_layt_free(&other);
      }
    if (o->volume_name)
      FATAL("generation %" PRIu32 "'s layout names no volume '%s'",
            c.gen[g].sd.generation, o->volume_name);
    FATAL("generation %" PRIu32 "'s layout has %" PRIu32
          " volumes, so there is no volume "
          "%" PRIu64, c.gen[g].sd.generation, layt.count,
          o->volume_index);
  }
  if (c.gen[g].sd.layout == XPAR_LAYOUT_ARMOURED) {
    const xpar_chain_vol * source = NULL;
    Fi(c.vol_count,
      if (c.vol[i].gen == g && c.vol[i].armoured_file)
        { source = &c.vol[i];  break; });
    if (!source)
      FATAL_FORMAT("generation %" PRIu32 "'s armoured archive is unavailable",
                   c.gen[g].sd.generation);
    if (o->to_dir && xpar_strlen(o->to_dir))
      xpar_asprintf(&path, "%s/%s", o->to_dir, layt.vol[target].name);
    else
      path = xpar_path_join(c.dir, layt.vol[target].name);
    gen_write_whole(path, source->data, source->len, o->force);
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: regenerated %s (%zu armoured "
                   "bytes)\n", path, source->len);
    xpar_free(path);
    xpar_layt_free(&layt);
    xpar_gchain_free(&c);
    return XPAR_EXIT_OK;
  }
  if (layt.vol[target].kind == XPAR_VOL_DATA) {
    xpar_vset * set;
    xpar_file * dst;
    char * tmp;
    const char * why = NULL;
    if (o->to_dir && xpar_strlen(o->to_dir))
      xpar_asprintf(&path, "%s/%s", o->to_dir, layt.vol[target].name);
    else
      path = xpar_path_join(c.dir, layt.vol[target].name);
    if (!o->force && gen_exists(path))
      FATAL("'%s' exists; use -f to overwrite it", path);
    set = xpar_vset_open(o);
    dst = gen_stage_open_rw(path, &tmp);
    if (!xpar_vset_recover_data(set, layt.vol[target].stream_offset,
                                layt.vol[target].byte_length, o->memory,
                                dst, &why)) {
      xpar_xclose(dst);  xpar_remove(tmp);  xpar_free(tmp);
      xpar_vset_close(set);
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "volume '%s' cannot be reconstructed from the surviving "
                 "data and recovery slices: %s", layt.vol[target].name,
                 why ? why : "unknown decoder failure");
    }
    if (xpar_flush(dst) != 0 || xpar_fsync(dst) != 0) {
      xpar_xclose(dst);  xpar_remove(tmp);  xpar_free(tmp);
      xpar_vset_close(set);
      FATAL_IO("cannot flush reconstructed volume '%s'", path);
    }
    xpar_xclose(dst);
    xpar_vset_close(set);
    gen_publish_whole(tmp, path, o->force);
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: recovered %s from survivor and "
                   "parity slices (%" PRIu64 " bare stream bytes)\n", path,
                   layt.vol[target].byte_length);
    xpar_free(path);
    xpar_layt_free(&layt);
    xpar_gchain_free(&c);
    return XPAR_EXIT_OK;
  }
  gen_manifest_on_disk(&c, g, o, &m, &owner);
  xpar_memset(&p, 0, sizeof p);
  if (!xpar_geom_from_setd(&c.gen[g].sd, &p.geom))
    FATAL_FORMAT("generation %" PRIu32 "'s geometry is malformed",
                 c.gen[g].sd.generation);
  p.recovery   = r_total;
  p.encode_r   = r_total;
  p.field_log2 = c.gen[g].sd.field_log2;
  p.codec      = c.gen[g].sd.codec;
  p.axis       = c.gen[g].sd.recovery_axis_log2;
  /*  Recover one volume in one pass over surviving data.  */
  source_set = xpar_vset_open(o);
  source_rc = xpar_vset_check(source_set, o, NULL);
  if (!xpar_vset_stream_intact(source_set, source_rc))
    FATAL_CODE(source_rc,
               "generation %" PRIu32 "'s protected stream is not clean; refusing to "
               "derive a replacement recovery volume from it",
               c.gen[g].sd.generation);
  gen_encode(&m, &p, c.gen[g].sd.slice_tag_len, o->memory,
             c.base ? c.base : o->set, gen_chain_key(&c), gen_read_vset,
             source_set, &t, NULL);
  gen_require_source_tables(source_set, &t, &p);
  xpar_vset_close(source_set);
  if (t.rec_spill) rec_scratch = xpar_alloc_raw((sz) t.rec_z);

  gen_recovery_volume(&out, o, &c, g, &layt, target, &p, &t, rec_scratch);

  if (o->to_dir && xpar_strlen(o->to_dir))
    xpar_asprintf(&path, "%s/%s", o->to_dir, layt.vol[target].name);
  else
    path = xpar_path_join(c.dir, layt.vol[target].name);
  gen_write_whole(path, out.data, out.len, o->force);
  if (!o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: regenerated %s (%zu bytes, %" PRIu64 " "
                 "recovery slices)\n", path, out.len,
                 layt.vol[target].byte_length);
  gen_json_result(o, "recover", c.gen[g].set_id,
                  c.gen[g].sd.generation, "ok", XPAR_EXIT_OK);

  xpar_free(path);
  xpar_buf_free(&out);
  gen_tables_free(&t);
  xpar_free(rec_scratch);
  xpar_layt_free(&layt);
  xpar_free(owner);
  xpar_manifest_free(&m);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

static bool gen_path_equal_n(const char * a, sz an,
                             const char * b, sz bn) {
  sz i;
  if (an != bn) return false;
  for (i = 0; i < an; i++) {
    char ca = a[i], cb = b[i];
    if (xpar_path_sep(ca)) ca = '/';
    if (xpar_path_sep(cb)) cb = '/';
#if defined(XPAR_WIN32) || defined(XPAR_DOS) || defined(__MSDOS__)
    if (ca >= 'A' && ca <= 'Z') ca = (char) (ca - 'A' + 'a');
    if (cb >= 'A' && cb <= 'Z') cb = (char) (cb - 'A' + 'a');
#endif
    if (ca != cb) return false;
  }
  return true;
}

static bool gen_same_dir(const char * a, const char * bpath,
                         const xpar_stat_t * b) {
  xpar_stat_t sa;
  char * aa, * bb;
  bool same;
  if (!b->is_dir) return false;
  if (xpar_lstat(a, &sa) != 0 || !sa.is_dir) return false;
  if ((sa.dev | sa.ino) && (b->dev | b->ino))
    return sa.dev == b->dev && sa.ino == b->ino;

  /*  Hosts without stable file IDs cannot prove aliases equivalent. Exact
      lexical equivalence is still safe and accepts relative spellings.  */
  aa = xpar_path_lex_abs(a);
  bb = xpar_path_lex_abs(bpath);
  if (!aa || !bb) { xpar_free(aa);  xpar_free(bb);  return false; }
  same = xpar_path_same(aa, bb);
  xpar_free(aa);  xpar_free(bb);
  return same;
}

/*  Match a journal path to generation G's volume in the set directory.  */
static const char * gen_undo_volume(const xpar_chain * c, u32 g,
                                    const char * path, u32 plen,
                                    const char * setdir_path,
                                    const xpar_stat_t * setdir) {
  xpar_layt l;
  const char * base;
  char * head;
  u32 i, cut;
  bool named = false, here;
  if (!c->gen[g].layt_body ||
      xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, &l) != XPAR_OK)
    return NULL;
  for (base = path + plen; base > path && !xpar_path_sep(base[-1]); base--);
  cut = (u32) (base - path);
  for (i = 0; i < l.count && !named; i++)
    if (l.vol[i].name && xpar_strlen(l.vol[i].name) == (sz) (plen - cut) &&
        gen_path_equal_n(l.vol[i].name, (sz) (plen - cut), base,
                         (sz) (plen - cut))) named = true;
  xpar_layt_free(&l);
  if (!named) return NULL;
  while (cut && xpar_path_sep(path[cut - 1])) cut--;
  head = cut ? xpar_strndup(path, cut)
             : xpar_strdup(plen && xpar_path_sep(path[0]) ? "/" : ".");
  here = gen_same_dir(head, setdir_path, setdir);
  xpar_free(head);
  return here ? base : NULL;
}

/*  Resolve a journal path to an entry in setdir, or return -1.  */
static i64 gen_undo_entry(const xpar_chain * c, const xpar_manifest * m,
                          const char * path, u32 plen,
                          const char * setdir_path,
                          const xpar_stat_t * setdir) {
  const char * dir = c->dir && *c->dir ? c->dir : ".";
  i64 best = -1;
  u32 bestlen = 0, i, cut;
  char * head;
  bool ok;
  Fi(m->count,
    const xpar_entry * e = &m->entry[i];
    char * allowed;
    bool same;
    /*  Every manifest object can be created by repair and journalled.  */
    allowed = xpar_path_join_n(dir, e->name, e->name_len);
    same = gen_path_equal_n(allowed, xpar_strlen(allowed), path, plen);
    xpar_free(allowed);
    if (same) return (i64) i;
    /*  The longest matching tail wins, so 'a/x' beats 'x' for 'd/a/x'.  */
    if (e->name_len <= bestlen || plen < e->name_len) continue;
    if (plen > e->name_len && !xpar_path_sep(path[plen - e->name_len - 1]))
      continue;
    if (!gen_path_equal_n(path + plen - e->name_len, e->name_len,
                          e->name, e->name_len)) continue;
    best = (i64) i;  bestlen = e->name_len);
  if (best < 0) return -1;
  cut = plen - bestlen;
  while (cut && xpar_path_sep(path[cut - 1])) cut--;
  /*  An empty absolute prefix means root.  */
  head = cut ? xpar_strndup(path, cut)
             : xpar_strdup(plen && xpar_path_sep(path[0]) ? "/" : ".");
  ok = gen_same_dir(head, setdir_path, setdir);
  xpar_free(head);
  return ok ? best : -1;
}

/*  Remove or empty a spent journal.  */
bool xpar_journal_drop(const char * path) {
  xpar_file * f;
  bool gone = xpar_remove(path) == 0;
  if (!gone) {
    f = xpar_open(path, XPAR_O_WRONLY | XPAR_O_NOFOLLOW);
    if (f && xpar_ftruncate(f, 0) == 0 && xpar_fsync(f) == 0) gone = true;
    if (f) xpar_close(f);
    if (gone) xpar_fprintf(xpar_stderr, "xpar: warning: spent journal '%s': "
                           "removal failed; emptied instead\n", path);
    else xpar_fprintf(xpar_stderr, "xpar: cannot remove spent journal '%s': "
                      "%s\n", path, xpar_strerror(xpar_errno()));
  }
  if (xpar_fsync_dir(path) != 0)
    xpar_fprintf(xpar_stderr, "xpar: warning: dropping the journal '%s' is "
                 "not durable: %s\n", path, xpar_strerror(xpar_errno()));
  return gone;
}

static char * gen_undo_path(const xpar_options * o, const xpar_chain * c,
                            u32 g) {
  return xpar_vname_undo(o->set_ref.base, c->gen[g].sd.generation);
}

/*  Return 0 for success, 1 for absence, or 2 for an I/O error.  */
static int gen_journal_read(const char * path, u8 ** out, sz * out_len,
                            int * err) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  i64 n;
  *out = NULL;  *out_len = 0;  *err = 0;
  if (!f) { *err = xpar_errno();  return xpar_errno_absent(*err) ? 1 : 2; }
  n = xpar_size(f);
  if (n < 0 || (u64) n >= (u64) (sz) -1) { *err = xpar_errno();  xpar_close(f);  return 2; }
  *out = xpar_alloc_raw((sz) n + 1);
  if (n) xpar_xread(f, *out, (sz) n);
  xpar_xclose(f);
  *out_len = (sz) n;
  return 0;
}

static int gen_undo_worse(int a, int b) {
  if (a == XPAR_EXIT_IO || b == XPAR_EXIT_IO) return XPAR_EXIT_IO;
  return a > b ? a : b;
}

/*  Replay one generation's journal.  */
static int gen_undo_one(const xpar_options * o, xpar_chain * chain,
                        u32 generation, const char * path, bool * present) {
  u8 * j;  sz n = 0;  u64 at, count, i;
  u32 applied = 0, skipped = 0;
  xpar_manifest manifest;
  xpar_stat_t setdir;
  u32 * owner = NULL;
  int err = 0;

  *present = false;
  switch (gen_journal_read(path, &j, &n, &err)) {
    case 1: return XPAR_EXIT_NOTFOUND;
    case 2:
      *present = true;
      xpar_fprintf(xpar_stderr, "xpar: cannot read the journal '%s': %s\n",
                   path, err ? xpar_strerror(err)
                             : "the host would not size it");
      return XPAR_EXIT_IO;
    default: break;
  }
  *present = true;
  /*  A short or footer-damaged artifact cannot prove whether it was torn
      before protected writes or corrupted afterwards.  Preserve it.  */
  if (n < XPAR_UNDO_HDR + XPAR_UNDO_FOOT || xpar_memcmp(j, XPAR_UNDO_MAGIC, 8))
    FATAL_FORMAT("'%s' is not an xpar repair journal", path);
  if (xpar_rd32(j + 8) != XPAR_UNDO_VER)
    FATAL_FORMAT("'%s' is a version %" PRIu32 " journal; this build reads %"
                 PRIu32, path, xpar_rd32(j + 8),
                 (u32) XPAR_UNDO_VER);
  if (xpar_crc32c(0, j, 60) != xpar_rd32(j + 60))
    FATAL_FORMAT("the header of '%s' does not verify", path);
  if (xpar_rd32(j + 12) || xpar_rd32(j + 56))
    FATAL_FORMAT("the header of '%s' has non-zero reserved fields", path);

  count = xpar_rd64(j + 32);
  { const u8 * foot = j + n - XPAR_UNDO_FOOT;
    bool complete = !xpar_memcmp(foot, XPAR_UNDO_END, 8) &&
                    xpar_rd64(foot + 8) == count &&
                    !xpar_rd32(foot + 20) &&
                    xpar_crc32c(0, j, (sz) (n - XPAR_UNDO_FOOT)) ==
                      xpar_rd32(foot + 16);
    if (!complete)
      FATAL_FORMAT("the journal '%s' is incomplete or corrupt; it was kept",
                   path);
  }

  if (xpar_memcmp(j + 16, chain->gen[generation].set_id, XPAR_SET_ID_LEN))
    FATAL_FORMAT("the journal '%s' belongs to a different set or generation",
                 path);
  xpar_gchain_manifest(chain, generation, &manifest, &owner);
  { const char * d = chain->dir && *chain->dir ? chain->dir : ".";
    if (xpar_lstat(d, &setdir) != 0) xpar_memset(&setdir, 0, sizeof setdir); }

  /*  Validate all record bounds and paths before replay.  */
  { u64 payload = 0;
    at = XPAR_UNDO_HDR;
    Fi(count,
      const u8 * rec;
      u32 plen, rflags;
      u64 off, len, raw, step, remain;
      const u8 * old;
      u64 k;
      if (at > (u64) n - XPAR_UNDO_FOOT ||
          (u64) n - XPAR_UNDO_FOOT - at < XPAR_UNDO_REC)
        FATAL_FORMAT("journal '%s' ends before record %" PRIu64, path,
                     i);
      rec = j + at;
      plen = xpar_rd32(rec);
      rflags = xpar_rd32(rec + 4);
      off = xpar_rd64(rec + 8);
      len = xpar_rd64(rec + 16);
      remain = (u64) n - XPAR_UNDO_FOOT - at;
      if ((rflags & ~XPAR_UNDO_FLAGS) ||
          ((rflags & XPAR_UNDO_CREATED) &&
           (rflags & XPAR_UNDO_REPLACED)) ||
          ((rflags & XPAR_UNDO_DIRECTORY) &&
           !(rflags & XPAR_UNDO_CREATED)) ||
          !plen || off + len < off ||
          (u64) XPAR_UNDO_REC + plen > remain ||
          len > remain - XPAR_UNDO_REC - plen)
        FATAL_FORMAT("journal '%s' has invalid framing in record %" PRIu64, path,
                     i);
      raw = (u64) XPAR_UNDO_REC + plen + len;
      if (raw > (u64) -1 - 7)
        FATAL_FORMAT("journal '%s' overflows record %" PRIu64 "'s length", path,
                     i);
      step = xpar_align_up(raw, 8);
      if (step > remain)
        FATAL_FORMAT("journal '%s' truncates record %" PRIu64, path,
                     i);
      if (payload + len < payload)
        FATAL_FORMAT("journal '%s' overflows its payload count", path);
      payload += len;
      { const char * rp = (const char *) rec + XPAR_UNDO_REC;
        const char * d = chain->dir && *chain->dir ? chain->dir : ".";
        if (xpar_has_nul(rec + XPAR_UNDO_REC, plen) ||
            (gen_undo_entry(chain, &manifest, rp, plen, d, &setdir) < 0 &&
             !gen_undo_volume(chain, generation, rp, plen, d, &setdir)))
          FATAL_FORMAT("journal record %" PRIu64
                       " names '%.*s' outside this set directory",
                       i, (int) plen, rp);
      }
      old = rec + XPAR_UNDO_REC + plen;
      if (xpar_crc32c(0, rec, 36) != xpar_rd32(rec + 36) ||
          xpar_crc32c(0, old, (sz) len) != xpar_rd32(rec + 32))
        FATAL_FORMAT("journal record %" PRIu64 " does not verify",
                     i);
      for (k = raw; k < step; k++)
        if (rec[k]) FATAL_FORMAT("journal record %" PRIu64
                                 " has non-zero padding",
                                 i);
      at += step);
    if (at != (u64) n - XPAR_UNDO_FOOT || payload != xpar_rd64(j + 40))
      FATAL_FORMAT("journal '%s' has inconsistent record or payload counts",
                   path);
  }

  at = XPAR_UNDO_HDR;
  Fi(count,
    const u8 * rec = j + at;
    u32 plen, rflags;
    u64 off, len, orig, step;
    const char * rp;  const u8 * old;
    const char * d;
    char * full;
    xpar_file * f;
    i64 ix;

    /*  Framing and CRC were validated above.  */
    plen   = xpar_rd32(rec);
    rflags = xpar_rd32(rec + 4);
    off    = xpar_rd64(rec + 8);
    len    = xpar_rd64(rec + 16);
    orig   = xpar_rd64(rec + 24);
    step   = xpar_align_up((u64) XPAR_UNDO_REC + plen + len, 8);
    rp  = (const char *) rec + XPAR_UNDO_REC;
    old = rec + XPAR_UNDO_REC + plen;
    at += step;

    d = chain->dir && *chain->dir ? chain->dir : ".";
    ix = gen_undo_entry(chain, &manifest, rp, plen, d, &setdir);
    if (ix >= 0)
      full = xpar_path_join_n(d, manifest.entry[ix].name,
                              manifest.entry[ix].name_len);
    else {
      const char * vol = gen_undo_volume(chain, generation, rp, plen,
                                         d, &setdir);
      /*  VOL is not NUL-terminated.  */
      if (!vol) { skipped++;  continue; }
      full = xpar_path_join_n(d, vol, plen - (u32) (vol - rp));
    }
    if (rflags & XPAR_UNDO_CREATED) {
      /*  Undo newly created names.  Absence means an interrupted earlier
          replay already completed this idempotent record.  */
      int removed = (rflags & XPAR_UNDO_DIRECTORY)
                      ? xpar_rmdir(full) : xpar_remove(full);
      if (removed != 0) {
        if (xpar_errno_absent(xpar_errno())) applied++;
        else {
          xpar_fprintf(xpar_stderr, "xpar: cannot remove '%s': %s\n", full,
                       xpar_strerror(xpar_errno()));
          skipped++;
        }
      }
      else applied++;
      xpar_free(full);
      continue;
    }
    if (rflags & XPAR_UNDO_REPLACED) {
      char * stage = NULL;
      xpar_file * out = xpar_stage_open(full, "XUN", XPAR_O_WRONLY |
                                              XPAR_O_NOFOLLOW, 1, &stage);
      bool ok = out != NULL;
      if (ok && len && xpar_pwrite(out, old, (sz) len, 0) != (sz) len)
        ok = false;
      if (ok && xpar_ftruncate(out, orig) != 0) ok = false;
      if (ok && xpar_fsync(out) != 0) ok = false;
      if (out) xpar_close(out);
      if (ok && xpar_rename(stage, full) != 0) ok = false;
      if (ok && xpar_fsync_dir(full) != 0) ok = false;
      if (!ok) {
        xpar_fprintf(xpar_stderr,
                     "xpar: cannot restore independent file '%s': %s\n",
                     full, xpar_strerror(xpar_errno()));
        if (stage) xpar_remove(stage);
        skipped++;
      } else applied++;
      xpar_free(stage);  xpar_free(full);
      continue;
    }
    /*  Refuse links created after the journal was written.  */
    f = xpar_open(full, XPAR_O_RDWR | XPAR_O_NOFOLLOW);
    if (!f) {
      xpar_fprintf(xpar_stderr, "xpar: cannot open '%s': %s\n", full,
                   xpar_strerror(xpar_errno()));
      skipped++;  xpar_free(full);  continue;
    }
    if (len && xpar_pwrite(f, old, (sz) len, off) != (sz) len) {
      xpar_fprintf(xpar_stderr, "xpar: short write to '%s'\n", full);
      skipped++;
    } else if (xpar_size(f) != (i64) orig && xpar_ftruncate(f, orig) != 0) {
      /*  Keep the journal if only the bytes were restored.  */
      xpar_fprintf(xpar_stderr, "xpar: cannot resize '%s': %s\n",
                   full, xpar_strerror(xpar_errno()));
      skipped++;
    } else applied++;
    xpar_fsync(f);
    xpar_xclose(f);
    xpar_free(full));

  xpar_fprintf(xpar_stderr,
               "xpar: replayed %" PRIu32 "/%" PRIu64 " journal records "
               "from '%s'%s\n",
               applied, count, path, skipped ? "; some failed" : "");
  if (!skipped && !o->keep_journal && !xpar_journal_drop(path))
    FATAL_IO("cannot retire journal '%s'; it remains replayable", path);
  xpar_free(j);
  xpar_free(owner);  xpar_manifest_free(&manifest);
  return skipped ? XPAR_EXIT_UNREPAIRABLE : XPAR_EXIT_OK;
}

int xpar_op_undo(const xpar_options * o) {
  xpar_chain chain;
  u32 * order;  char ** paths;
  u32 order_n = 0, played = 0, i, k;
  int worst = XPAR_EXIT_OK;

  if (o->set && xpar_vname_is_undo(o->set))
    FATAL("undo requires a set path, not a journal");
  xpar_gchain_load(o, &chain);
  if (!o->set_ref.base) FATAL("undo needs a set with a resolvable base name");
  FATAL_UNLESS(chain.gen_count > 0,
               "this set has no readable generation to undo");

  order = xpar_calloc(chain.gen_count, sizeof *order);
  if (o->gen_count)
    order[order_n++] = xpar_gchain_select(&chain, &o->gens[0]);
  else {
    /*  Replay each generation's journal newest first.  */
    bool * taken = xpar_calloc(chain.gen_count, sizeof *taken);
    Fk(chain.gen_count,
      u32 best = XPAR_GEN_NONE;
      Fi(chain.gen_count,
        if (!taken[i] && (best == XPAR_GEN_NONE ||
                          chain.gen[i].sd.generation >
                            chain.gen[best].sd.generation))
          best = i);
      taken[best] = true;  order[order_n++] = best);
    xpar_free(taken);
  }

  paths = xpar_calloc(order_n, sizeof *paths);
  Fk(order_n,
    bool present = false, seen = false;
    int rc;
    paths[k] = gen_undo_path(o, &chain, order[k]);
    /*  Fork branches can share a generation number, hence a journal.  */
    Fi(k, if (xpar_path_same(paths[i], paths[k])) seen = true);
    if (seen) continue;
    rc = gen_undo_one(o, &chain, order[k], paths[k], &present);
    if (!present) continue;
    played++;
    worst = gen_undo_worse(worst, rc));

  if (!played) {
    xpar_fprintf(xpar_stderr, "xpar: no journal at '%s'; nothing to undo\n",
                 paths[0]);
    worst = XPAR_EXIT_NOTFOUND;
  }
  gen_json_result(o, "undo", chain.gen[order[0]].set_id,
                  chain.gen[order[0]].sd.generation,
                  worst == XPAR_EXIT_OK ? "ok"
                    : worst == XPAR_EXIT_NOTFOUND ? "not-found"
                    : worst == XPAR_EXIT_IO ? "failed" : "unrepairable",
                  worst);
  Fk(order_n, xpar_free(paths[k]));
  xpar_free(paths);  xpar_free(order);
  xpar_gchain_free(&chain);
  return worst;
}

/*  Recover armour parameters by requiring all-zero frame syndromes. Field
    choice fixes n and the polynomial, bounding the search.  */

int xpar_op_recover_prologue(const xpar_options * o) {
  u8 * f;  sz n = 0;  u64 region;
  xpar_arm_prologue pr;
  xpar_key key;
  u8 master[XPAR_BLAKE3_KEY_LEN];
  bool key_loaded = false;
  int bits, bit_order[2] = { 8, 16 }, bi, found = 0, which = -1;
  u32 t;
  u64 d;
  u8 * frame = NULL;

  xpar_memset(&key, 0, sizeof key);
  xpar_memset(master, 0, sizeof master);
  if (o->auth_key) { xpar_keyfile_load_or_die(o->auth_key, &key, master);  key_loaded = true; }

  f = gen_read_whole(o->set, &n, true);
  if (n <= ARM_HDR_LEN)
    FATAL_FORMAT("'%s' is too short to be an armoured archive", o->set);
  region = (u64) n - ARM_HDR_LEN;
  xpar_memset(&pr, 0, sizeof pr);

  /*  Search parameters only when no stored prologue decodes.  */
  if (xpar_garm_prologue(f, n, &pr, &which)) {
    found = 1;
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: prologue copy %d of 3 verifies\n",
                   which + 1);
  }

  /*  Try symbol widths whose frame size divides the input first; syndromes
      remain the authority.  */
  if (region >= 2ULL * 65535 && region % (2ULL * 65535) == 0)
    { bit_order[0] = 16;  bit_order[1] = 8; }

  /*  Try depths 1..64, then powers of two. Search t downward because a
      codeword also satisfies smaller t values.  */
  for (bi = 0; bi < 2 && !found; bi++) {
    u32 di;
    bits = bit_order[bi];
    if (o->verbose)
      xpar_fprintf(xpar_stderr, "xpar: searching GF(2^%d) framings\n", bits);
    for (di = 0; di < 82 && !found; di++) {
      u32 nmax, ncand;
      u64 symbols;
      d = di < 64 ? (u64) di + 1 : ((u64) 1 << (di - 64 + 7));
      if (d > XPAR_ARMG_DEPTH_MAX) break;
      if (region % (d * (u64) (bits / 8))) continue;
      symbols = region / (d * (u64) (bits / 8));
      nmax = bits == 8 ? 255U : 65535U;
      if (symbols < nmax) nmax = (u32) symbols;
      /*  Shortened n must divide the region into integral D*n*W frames.  */
      for (ncand = nmax; ncand >= 3 && !found; ncand--) {
        xpar_armour_params ap;
        xpar_armour * a;
        u64 fx;
        xpar_armour_defaults(&ap, (u32) bits);
        if (symbols % ncand) continue;
        ap.n = ncand;  ap.depth = d;
        for (t = MIN(128U, (ncand - 1) / 2); t >= 1 && !found; t--) {
          ap.k = ap.n - 2 * t;
          if (xpar_armour_check(&ap)) continue;
          fx = d * ap.n * (ap.symbol_bits / 8);
          a = xpar_armour_new(&ap);
          frame = xpar_realloc(frame, (sz) fx);
          xpar_memcpy(frame, f + ARM_HDR_LEN, (sz) fx);
          if (xpar_armour_decode_frame(a, frame, NULL) !=
              XPAR_ARMOUR_FAILED) {
            u64 frames = region / fx;
            u64 q, probe_frames = MIN(frames, (u64) 8);
            bool all_clean = true;
            /*  A short false framing can occasionally satisfy one syndrome
               check because the actual region is itself coded. Require the
               candidate to frame every byte before accepting it; a bounded
               prefix throws most false framings out first. Correctable
               damage is still a decode, so CLEAN is not the bar.  */
            for (q = 1; q < probe_frames && all_clean; q++) {
              xpar_memcpy(frame, f + ARM_HDR_LEN + q * fx, (sz) fx);
              if (xpar_armour_decode_frame(a, frame, NULL) ==
                  XPAR_ARMOUR_FAILED) all_clean = false;
            }
            for (q = probe_frames; q < frames && all_clean; q++) {
              xpar_memcpy(frame, f + ARM_HDR_LEN + q * fx, (sz) fx);
              if (xpar_armour_decode_frame(a, frame, NULL) ==
                  XPAR_ARMOUR_FAILED) all_clean = false;
            }
            if (all_clean) {
              u64 maxplain = frames * d * ap.k * (ap.symbol_bits / 8);
              u64 rlen = frames * fx;
              u8 * probe = xpar_alloc_raw((sz) maxplain);
              u8 * fixed = xpar_alloc_raw((sz) rlen);
              xpar_scan ps;
              xpar_pkt ph;
              const u8 * pb;
              u64 po;
              bool have_setd = false, have_strm = false;
              /*  Confirm packets in the corrected plaintext.  */
              xpar_memcpy(fixed, f + ARM_HDR_LEN, (sz) rlen);
              xpar_armour_decode_frames(a, fixed, frames, NULL);
              xpar_armour_extract(a, probe, maxplain, fixed);
              xpar_free(fixed);
              /*  Framing recovery must not depend on whether a supplied key
                 is right. The authenticated preflight below distinguishes
                 missing and wrong credentials with exit 6.  */
              xpar_scan_init(&ps, probe, maxplain, NULL, true);
              ps.accept_unverified_keyed = true;
              while (xpar_scan_next(&ps, &ph, &pb, &po)) {
                if (xpar_pkt_is(&ph, XPAR_T_SETD)) have_setd = true;
                if (xpar_pkt_is(&ph, XPAR_T_STRM) &&
                    ph.length >= XPAR_PKT_HDR + 16) have_strm = true;
              }
              xpar_free(probe);
              if (have_setd && have_strm) {
                pr.symbol_bits     = (u8) ap.symbol_bits;
                pr.poly            = ap.poly;
                pr.n               = ap.n;
                pr.k               = ap.k;
                pr.fcr             = ap.fcr;
                pr.prim            = ap.prim;
                pr.depth           = ap.depth;
                pr.plain_length    = maxplain;
                pr.armoured_length = frames * fx;
                found = 1;
              }
            }
          }
          xpar_armour_free(a);
        }
      }
    }
  }
  xpar_free(frame);
  if (!found)
    FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
               "no supported armour parameters decode the first frame of "
               "'%s'; it may be damaged or use non-default field "
               "parameters", o->set);

  /*  The parameters demodulate the region; the stream range comes out of
      the packets inside it, which is where SETD says how long the
      protected stream is.  */
  { xpar_armour_params ap;
    u8 * plain;  sz plen;
    xpar_scan sc;  xpar_pkt hdr;  const u8 * body;
    u64 off, last = 0, declared_stream = 0;
    bool authenticated = false;
    arm_params_of(&pr, &ap);
    plain = arm_extract(&ap, f + ARM_HDR_LEN, region, pr.plain_length, &plen,
                        NULL);
    if (!plain) FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                           "the recovered parameters do not frame the "
                           "region");
    /*  AUTH itself is discoverable without trusting its packet tag. It is
        inspected first solely to distinguish missing/wrong credentials from
        damage; then the whole plaintext is scanned again with K_pkt.  */
    xpar_scan_init(&sc, plain, plen, NULL, true);
    sc.accept_unverified_keyed = true;
    while (xpar_scan_next(&sc, &hdr, &body, &off)) {
      xpar_auth auth;
      if (!xpar_pkt_is(&hdr, XPAR_T_AUTH) ||
          xpar_auth_read(body, (sz) (hdr.length - XPAR_PKT_HDR), &auth) !=
            XPAR_OK) continue;
      authenticated = true;
      if (!key_loaded) {
        xpar_free(plain);
        FATAL_CODE(XPAR_EXIT_AUTH,
                   "recovering this authenticated archive's prologue "
                   "requires --auth-key=FILE");
      }
      if (!xpar_auth_key_ok(&auth, master)) {
        xpar_free(plain);
        FATAL_CODE(XPAR_EXIT_AUTH,
                   "the authentication key is wrong for this archive");
      }
      break;
    }
    xpar_scan_init(&sc, plain, plen,
                   authenticated && key_loaded ? &key : NULL, true);
    while (xpar_scan_next(&sc, &hdr, &body, &off)) {
      if (off + hdr.length > last) last = off + hdr.length;
      if (xpar_pkt_is(&hdr, XPAR_T_STRM) &&
          hdr.length >= XPAR_PKT_HDR + 16) {
        pr.stream_offset = off + XPAR_PKT_HDR + 16;
        pr.stream_length = hdr.length - XPAR_PKT_HDR - 16;
      }
      if (xpar_pkt_is(&hdr, XPAR_T_SETD)) {
        xpar_setd sd;
        xpar_status sst = xpar_setd_read(body,
                                         (sz) (hdr.length - XPAR_PKT_HDR),
                                         &sd);
        if (sst == XPAR_OK) {
          if (o->verbose > 1)
            xpar_fprintf(xpar_stderr,
                         "xpar: recovered SETD stream length %" PRIu64 "\n",
                         sd.stream_length);
          declared_stream = sd.stream_length;
        }
        if (sst == XPAR_OK || sst == XPAR_E_UNSUPPORTED) xpar_setd_free(&sd);
      }
    }
    if (declared_stream &&
        (!pr.stream_length || declared_stream <= pr.stream_length))
      pr.stream_length = declared_stream;
    if (!last || !pr.stream_length) {
      xpar_free(plain);
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "the recovered framing does not contain a validating "
                 "protected stream");
    }
    pr.plain_length = last;
    xpar_free(plain);
  }

  xpar_fprintf(gen_hout(o),
               "recovered prologue for %s:\n"
               "  symbol_bits     %" PRIu8 "\n  poly            0x%" PRIX32 "\n"
               "  n               %" PRIu32 "\n  k               %" PRIu32 "   (t = %" PRIu32 ")\n"
               "  fcr             %" PRIu32 "\n  prim            %" PRIu32 "\n"
               "  depth D         %" PRIu64 "\n  plain_length    %" PRIu64 "\n"
               "  armoured_length %" PRIu64 "\n  stream_offset   %" PRIu64 "\n"
               "  stream_length   %" PRIu64 "\n",
               o->set, pr.symbol_bits, pr.poly,
               pr.n, pr.k,
               ((pr.n - pr.k) / 2), pr.fcr,
               pr.prim, pr.depth,
               pr.plain_length,
               pr.armoured_length,
               pr.stream_offset,
               pr.stream_length);

  if (!o->dry_run) {
    xpar_armour_params ap;
    xpar_armour * a;
    gen_addrec_file * pub;
    u8 copy[ARM_COPY_LEN];
    int i;
    /*  Each 96-byte prologue is zero-extended to RS(255,223) data and
        followed by its 32 parity bytes.  */
    xpar_armour_defaults(&ap, 8);
    a = xpar_armour_new(&ap);
    xpar_memset(copy, 0, sizeof copy);
    arm_prologue_encode(copy, &pr);
    { u8 * fr = xpar_calloc((sz) ap.n, 1);
      xpar_memcpy(fr, copy, ARM_PLAIN_LEN);
      xpar_armour_encode_frame(a, fr);
      xpar_memcpy(copy + ARM_PLAIN_LEN, fr + ap.k, 32);
      xpar_free(fr);
    }
    xpar_armour_free(a);
    /*  Publish all three prologue copies atomically.  */
    Fi(3, xpar_memcpy(f + (sz) i * ARM_COPY_LEN, copy, sizeof copy));
    pub = xpar_calloc(1, sizeof *pub);
    pub[0].stage = gen_stage_whole(o->set, f, n);
    pub[0].final = o->set;
    pub[0].replace = true;
    gen_addrec_publish(pub, 1);
    xpar_fprintf(xpar_stderr, "xpar: wrote three repaired prologue copies "
                 "to %s\n", o->set);
  }
  xpar_free(f);
  gen_json_result(o, "recover-prologue", NULL, 0, "ok", XPAR_EXIT_OK);
  xpar_key_forget(&key, master);
  return XPAR_EXIT_OK;
}

/*  Require each selected kernel tier to match scalar before timing it.  */

typedef struct { u64 s; } bm_rng;

static u32 bm_next(bm_rng * r) {
  r->s ^= r->s << 13;  r->s ^= r->s >> 7;  r->s ^= r->s << 17;
  return (u32) (r->s >> 32);
}

static void bm_fill(bm_rng * r, u8 * p, sz n) {
  sz i;
  Fi(n, p[i] = (u8) bm_next(r));
}

static u32 bm_cmp(const char * tier, const char * what, const u8 * a,
                  const u8 * b, sz n) {
  sz i;
  Fi(n,
    if (a[i] != b[i]) {
      xpar_fprintf(xpar_stderr,
                   "xpar: benchmark: %s disagrees with scalar in %s at byte "
                   "%zu of %zu (%02" PRIX8 " against %02" PRIX8 ")\n", tier, what,
                   i, n, a[i], b[i]);
      return 1;
    });
  return 0;
}

/*  Exercise each GF region entry point across vector-width boundaries.  */
static u32 bm_check_gf(const xpar_gf_kernels * k, const char * tier) {
  static const sz len[] = { 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
                            65, 127, 128, 255, 256, 1023, 4096 };
  const sz big = 4096;
  u8 * src = xpar_alloc_raw(big);
  u8 * d0  = xpar_alloc_raw(big);
  u8 * a   = xpar_alloc_raw(big);
  u8 * b   = xpar_alloc_raw(big);
  u8 * a2  = xpar_alloc_raw(big);
  u8 * b2  = xpar_alloc_raw(big);
  bm_rng r;
  u32 bad = 0, i;

  r.s = 0x9E3779B97F4A7C15ULL;
  bm_fill(&r, src, big);
  bm_fill(&r, d0, big);

  for (i = 0; i < ARRAY_LEN(len) && !bad; i++) {
    sz n = len[i], n2 = n & ~(sz) 1;
    u8  c8  = (u8)  (1 + (bm_next(&r) & 0xFE));
    u16 c16 = (u16) (1 + (bm_next(&r) & 0xFFFE));
    xpar_gf8_coef  m8;
    xpar_gf16_coef m16;
    xpar_gf8_prepare(&m8, c8);
    xpar_gf16_prepare(&m16, c16);

    xpar_memcpy(a, d0, n);  k->mac8(a, src, n, &m8);
    xpar_memcpy(b, d0, n);  xpar_gf8_mac_ref(b, src, n, c8);
    bad += bm_cmp(tier, "mac8", a, b, n);

    /*  Check paired matrix-kernel slots.  */
    { u8 c8b = (u8) (1 + (bm_next(&r) & 0xFE));
      xpar_gf8_coef pm[2];
      u8 * pd[2];
      xpar_gf8_prepare(&pm[0], c8);
      xpar_gf8_prepare(&pm[1], c8b);
      xpar_memcpy(a, d0, n);   xpar_memcpy(a2, d0, n);
      pd[0] = a;  pd[1] = a2;
      k->mac8x2(pd, src, n, pm);
      xpar_memcpy(b, d0, n);   xpar_gf8_mac_ref(b, src, n, c8);
      xpar_memcpy(b2, d0, n);  xpar_gf8_mac_ref(b2, src, n, c8b);
      bad += bm_cmp(tier, "mac8x2 lo", a, b, n);
      bad += bm_cmp(tier, "mac8x2 hi", a2, b2, n);
    }

    k->mul8(a, src, n, &m8);
    xpar_gf8_mul_ref(b, src, n, c8);
    bad += bm_cmp(tier, "mul8", a, b, n);

    xpar_memcpy(a, d0, n);  k->xor2(a, src, n);
    xpar_memcpy(b, d0, n);  xpar_xor2_ref(b, src, n);
    bad += bm_cmp(tier, "xor2", a, b, n);

    k->xor3(a, d0, src, n);
    xpar_xor3_ref(b, d0, src, n);
    bad += bm_cmp(tier, "xor3", a, b, n);

    xpar_memcpy(a, d0, n);  xpar_memcpy(a2, src, n);
    k->fft8(a, a2, n, &m8);
    xpar_memcpy(b, d0, n);  xpar_memcpy(b2, src, n);
    xpar_gf8_fft2_ref(b, b2, n, c8);
    bad += bm_cmp(tier, "fft8 x", a, b, n);
    bad += bm_cmp(tier, "fft8 y", a2, b2, n);

    xpar_memcpy(a, d0, n);  xpar_memcpy(a2, src, n);
    k->ifft8(a, a2, n, &m8);
    xpar_memcpy(b, d0, n);  xpar_memcpy(b2, src, n);
    xpar_gf8_ifft2_ref(b, b2, n, c8);
    bad += bm_cmp(tier, "ifft8 x", a, b, n);
    bad += bm_cmp(tier, "ifft8 y", a2, b2, n);

    if (!n2) continue;
    xpar_memcpy(a, d0, n2);  k->mac16(a, src, n2, &m16);
    xpar_memcpy(b, d0, n2);  xpar_gf16_mac_ref(b, src, n2, c16);
    bad += bm_cmp(tier, "mac16", a, b, n2);

    { u16 c16b = (u16) (1 + (bm_next(&r) & 0xFFFE));
      xpar_gf16_coef pm[2];
      u8 * pd[2];
      xpar_gf16_prepare(&pm[0], c16);
      xpar_gf16_prepare(&pm[1], c16b);
      xpar_memcpy(a, d0, n2);   xpar_memcpy(a2, d0, n2);
      pd[0] = a;  pd[1] = a2;
      k->mac16x2(pd, src, n2, pm);
      xpar_memcpy(b, d0, n2);   xpar_gf16_mac_ref(b, src, n2, c16);
      xpar_memcpy(b2, d0, n2);  xpar_gf16_mac_ref(b2, src, n2, c16b);
      bad += bm_cmp(tier, "mac16x2 lo", a, b, n2);
      bad += bm_cmp(tier, "mac16x2 hi", a2, b2, n2);
    }

    k->mul16(a, src, n2, &m16);
    xpar_gf16_mul_ref(b, src, n2, c16);
    bad += bm_cmp(tier, "mul16", a, b, n2);

    xpar_memcpy(a, d0, n2);  xpar_memcpy(a2, src, n2);
    k->fft16(a, a2, n2, &m16);
    xpar_memcpy(b, d0, n2);  xpar_memcpy(b2, src, n2);
    xpar_gf16_fft2_ref(b, b2, n2, c16);
    bad += bm_cmp(tier, "fft16 x", a, b, n2);
    bad += bm_cmp(tier, "fft16 y", a2, b2, n2);

    xpar_memcpy(a, d0, n2);  xpar_memcpy(a2, src, n2);
    k->ifft16(a, a2, n2, &m16);
    xpar_memcpy(b, d0, n2);  xpar_memcpy(b2, src, n2);
    xpar_gf16_ifft2_ref(b, b2, n2, c16);
    bad += bm_cmp(tier, "ifft16 x", a, b, n2);
    bad += bm_cmp(tier, "ifft16 y", a2, b2, n2);
  }
  xpar_free(src);  xpar_free(d0);  xpar_free(a);  xpar_free(b);
  xpar_free(a2);   xpar_free(b2);
  return bad;
}

/*  Exercise armour tiers through encode, t-symbol damage and decode.  */
static u32 bm_armour_frame(const xpar_armour_params * p, u8 * frame,
                           bm_rng * r) {
  xpar_armour * a = xpar_armour_new(p);
  u64 fd = xpar_armour_frame_plain(a), fx = xpar_armour_frame_disk(a);
  bm_fill(r, frame, (sz) fx);
  xpar_memset(frame + fd, 0, (sz) (fx - fd));
  xpar_armour_encode_frame(a, frame);
  xpar_armour_free(a);
  return (u32) fx;
}

static u32 bm_check_armour(const char * tier, const xpar_armour_params * p,
                           const u8 * ref, u64 fx, const char * what) {
  u8 * frame = xpar_alloc_raw((sz) fx);
  xpar_armour * a = xpar_armour_new(p);
  bm_rng r;
  u32 bad = 0, i, t = (p->n - p->k) / 2, w = p->symbol_bits / 8;

  r.s = 0x243F6A8885A308D3ULL;
  xpar_memcpy(frame, ref, (sz) fx);
  xpar_armour_encode_frame(a, frame);
  bad += bm_cmp(tier, what, frame, ref, (sz) fx);

  /*  Damage each codeword to capacity before decoding.  */
  Fi(t,
    u32 s = bm_next(&r) % p->n;
    u64 at = ((u64) s * p->depth) * w;
    frame[at] ^= 0xA5;
    if (w == 2) frame[at + 1] ^= 0x5A);
  if (xpar_armour_decode_frame(a, frame, NULL) == XPAR_ARMOUR_FAILED) {
    xpar_fprintf(xpar_stderr, "xpar: benchmark: %s failed to decode a frame "
                 "at capacity (%s)\n", tier, what);
    bad++;
  } else bad += bm_cmp(tier, what, frame, ref, (sz) fx);
  xpar_armour_free(a);
  xpar_free(frame);
  return bad;
}

#if defined(HAVE_SSE42) || defined(HAVE_ARM_CRC32)
/*  Compare hardware CRC tiers with the scalar reference.  */
static u32 bm_crc_one(const char * name, u32 got, u32 want, sz n) {
  if (got == want) return 0;
  xpar_fprintf(xpar_stderr, "xpar: benchmark: crc32c %s gives %08" PRIX32
               " at %zu bytes, scalar gives %08" PRIX32 "\n",
               name, got, n, want);
  return 1;
}
#endif

static u32 bm_check_crc32c(void) {
  static const sz len[] = { 1, 7, 8, 63, 64, 255, 256, 1024, 8192, 24577,
                            65536 };
  u32 bad = 0, i;
  u8 * buf;
  bm_rng r;
#if defined(HAVE_SSE42) || defined(HAVE_ARM_CRC32)
  u32 feat = xpar_cpu_features();
#endif
  r.s = 0xB5026F5AA96619EULL;
  buf = xpar_alloc_raw(65536);
  bm_fill(&r, buf, 65536);
  /*  Compare every dispatchable kernel with the scalar result.  */
  for (i = 0; i < ARRAY_LEN(len); i++) {
    u32 want = xpar_crc32c_scalar(0x1234U, buf, len[i]);
    (void) want;
#ifdef HAVE_SSE42
    if (feat & XPAR_CPU_SSE42)
      bad += bm_crc_one("sse42", xpar_crc32c_sse42(0x1234U, buf, len[i]),
                        want, len[i]);
#endif
#ifdef HAVE_ARM_CRC32
    if (feat & XPAR_CPU_ARMCRC)
      bad += bm_crc_one("armcrc", xpar_crc32c_arm(0x1234U, buf, len[i]),
                        want, len[i]);
#endif
  }
  xpar_free(buf);
  return bad;
}

static u32 bm_check_blake3(void) {
  enum { LANES = XPAR_BLAKE3_MAX_DEGREE, BLOCKS = 3 };
  u8 * in = xpar_alloc_raw(LANES * BLOCKS * XPAR_BLAKE3_BLOCK_LEN);
  const u8 * ptr[LANES];
  u8 want[LANES * XPAR_BLAKE3_OUT_LEN], got[LANES * XPAR_BLAKE3_OUT_LEN];
  bm_rng r;
  u32 bad = 0, i;
  const char * name = NULL;
#if defined(HAVE_AVX2) || defined(HAVE_NEON)
  u32 feat = xpar_cpu_features();
#endif

  r.s = 0x452821E638D01377ULL;
  bm_fill(&r, in, LANES * BLOCKS * XPAR_BLAKE3_BLOCK_LEN);
  Fi(LANES, ptr[i] = in + (sz) i * BLOCKS * XPAR_BLAKE3_BLOCK_LEN);
  xpar_blake3_hash_many_scalar(ptr, LANES, BLOCKS, xpar_blake3_iv, 7, true,
                               0, 1, 2, want);
  xpar_memcpy(got, want, sizeof got);
#ifdef HAVE_AVX2
  if (feat & XPAR_CPU_AVX2) {
    xpar_blake3_hash_many_avx2(ptr, LANES, BLOCKS, xpar_blake3_iv, 7, true,
                               0, 1, 2, got);
    name = "avx2";
  }
#endif
#ifdef HAVE_NEON
  if (feat & XPAR_CPU_NEON) {
    xpar_blake3_hash_many_neon(ptr, LANES, BLOCKS, xpar_blake3_iv, 7, true,
                               0, 1, 2, got);
    name = "neon";
  }
#endif
  if (name) bad += bm_cmp(name, "blake3 hash_many", got, want, sizeof got);
  xpar_free(in);
  return bad;
}

static u8 bm_hexdigit(char c) {
  if (c >= '0' && c <= '9') return (u8) (c - '0');
  if (c >= 'a' && c <= 'f') return (u8) (c - 'a' + 10);
  return (u8) (c - 'A' + 10);
}

static u32 bm_kat_hex(const char * what, const u8 * got, sz n,
                      const char * hex) {
  sz i;
  Fi(n,
    u8 want = (u8) ((bm_hexdigit(hex[2 * i]) << 4) |
                    bm_hexdigit(hex[2 * i + 1]));
    if (got[i] != want) {
      xpar_fprintf(xpar_stderr,
                   "xpar: benchmark: conformance KAT %s differs at byte %zu "
                   "(%02" PRIX8 " against %02" PRIX8 ")\n", what, i,
                   got[i], want);
      return 1;
    });
  return 0;
}

/*  Frozen installed KATs pin the published hash, CRC and generation bytes.  */
static u32 bm_check_kats(void) {
  static const u32 roll_want[] = {
    0xcf762298U, 0x96fce802U, 0x35c5ff48U, 0x73771252U,
    0xdff7f330U, 0x6f56d7b1U, 0xa1d86dc4U, 0x4ca80c42U,
    0xa2d3bb0cU, 0xd14a6a4cU, 0x2ee14d0bU, 0x8b5fd88eU,
    0xef1bba18U, 0x6b86a270U, 0x3ebe57efU, 0x0363e9c7U,
    0xbf835fabU
  };
  u8 data[20000], setd_body[96], file_body[160];
  u8 content[32], prefix[16], file_id[16], set_id[16], master[32], check[16];
  xpar_entry entry;
  char name[] = "tree/fixed.bin";
  xpar_set_id_ctx set_hash;
  xpar_crc32c_roll roll;
  xpar_armour_params ap;
  xpar_armour * armour;
  u8 frame[255];
  u32 generator[3], crc, bad = 0, i;

  Fi(sizeof data, data[i] = (u8) (i * 29 + 7));
  Fi(sizeof setd_body, setd_body[i] = (u8) (i * 3 + 1));
  Fi(sizeof file_body, file_body[i] = (u8) (i * 5 + 9));
  xpar_blake3_hash(data, sizeof data, content, sizeof content);
  xpar_blake3_hash(data, 16384, prefix, sizeof prefix);
  bad += bm_kat_hex("V-HASH content_hash", content, sizeof content,
                    "f9d161476303e9b8a45d8a4403d6bd5b"
                    "6649ae5a333b1d1787334fcf603f0011");
  bad += bm_kat_hex("V-HASH prefix_hash", prefix, sizeof prefix,
                    "a24032354ddaf4559e32caf4f14ba510");
  xpar_memset(&entry, 0, sizeof entry);
  entry.name = name;
  entry.name_len = (u32) (sizeof name - 1);
  entry.length = sizeof data;
  xpar_memcpy(entry.prefix_hash, prefix, sizeof prefix);
  xpar_file_id(&entry, NULL, file_id);
  bad += bm_kat_hex("V-HASH file_id", file_id, sizeof file_id,
                    "0144119834d4eefb811fb9935c3f7523");
  xpar_set_id_begin(&set_hash, NULL, setd_body, sizeof setd_body);
  xpar_set_id_update(&set_hash, file_body, sizeof file_body);
  xpar_set_id_final(&set_hash, set_id);
  bad += bm_kat_hex("V-HASH set_id", set_id, sizeof set_id,
                    "cf2b9c0a22b17377f7873c716ad20c97");
  xpar_key_master(master, "xpar2 conformance key\n", 22);
  xpar_key_check(check, master);
  bad += bm_kat_hex("V-HASH key_check", check, sizeof check,
                    "485ae68f1442ed7c0aead7358b86a037");
  xpar_memset(master, 0, sizeof master);

  crc = xpar_crc32c(0, data, 4096);
  if (crc != 0x752b349cU) {
    xpar_fprintf(xpar_stderr,
                 "xpar: benchmark: conformance KAT V-CRC slice is %08" PRIX32 ", "
                 "expected 752B349C\n", crc);
    bad++;
  }
  xpar_crc32c_roll_init(&roll, 64);
  crc = xpar_crc32c(0, data, 64);
  Fi(ARRAY_LEN(roll_want),
    if (i) crc = xpar_crc32c_roll_step(&roll, crc, data[i - 1],
                                       data[i + 63]);
    if (crc != roll_want[i]) {
      xpar_fprintf(xpar_stderr,
                   "xpar: benchmark: conformance KAT V-CRC rolling state "
                   "%" PRIu32 " is %08" PRIX32 ", expected %08" PRIX32 "\n", i,
                   crc, roll_want[i]);
      bad++;
      break;
    });

  xpar_armour_defaults(&ap, 8);
  ap.n = 255;  ap.k = 253;  ap.depth = 1;
  armour = xpar_armour_new(&ap);
  Fi(253, frame[i] = (u8) (11 + i * 37));
  frame[253] = frame[254] = 0;
  xpar_armour_generator(armour, generator);
  if (generator[0] != 0x96 || generator[1] != 0x70 || generator[2] != 1) {
    xpar_fprintf(xpar_stderr,
                 "xpar: benchmark: conformance KAT V-GEN generator differs\n");
    bad++;
  }
  xpar_armour_encode_frame(armour, frame);
  bad += bm_kat_hex("V-GEN codeword parity", frame + 253, 2, "03fc");
  xpar_armour_free(armour);
  return bad;
}

/*  Emit raw JSON measurements so callers can compute rates.  */
static xpar_json * bm_js;
static bool        bm_quiet;

static void bm_rate(const char * tier, const char * operation,
                    u64 bytes, u64 usec) {
  f64 mib_s;
  if (!usec) usec = 1;
  mib_s = ((f64) bytes * 1000000.0) / ((f64) usec * 1048576.0);
  if (bm_js) {
    xpar_json_begin(bm_js, "kernel");
    xpar_json_str(bm_js, "tier", tier);
    xpar_json_str(bm_js, "operation", operation);
    xpar_json_u64(bm_js, "bytes", bytes);
    xpar_json_u64(bm_js, "usec", usec);
    xpar_json_end(bm_js);
  }
  if (bm_quiet) return;
  xpar_fprintf(xpar_stderr,
               "xpar: benchmark: %-12s %-12s %10" PRIu64 " bytes %8" PRIu64 " us "
               "%9.2f MiB/s\n", tier, operation,
               bytes, usec, mib_s);
}

static void bm_measure_gf(const xpar_gf_kernels * k, const char * tier) {
  const sz n = (sz) 1 << 20;
  const u32 repeat = 8;
  u8 * src = xpar_alloc_aligned(n, 64);
  u8 * dst = xpar_alloc_aligned(n, 64);
  xpar_gf8_coef coef;
  bm_rng rng;
  u64 begin, elapsed;
  u32 i;
  rng.s = 0x6A09E667F3BCC909ULL;
  bm_fill(&rng, src, n);
  xpar_memset(dst, 0, n);
  xpar_gf8_prepare(&coef, 173);
  begin = xpar_usec_now();
  Fi(repeat, k->mac8(dst, src, n, &coef));
  elapsed = xpar_usec_now() - begin;
  bm_rate(tier, "gf8-mac", (u64) n * repeat, elapsed);
  xpar_free_aligned(src);  xpar_free_aligned(dst);
}

static void bm_measure_armour(const xpar_armour_params * p,
                              const char * tier, const char * operation) {
  const u32 repeat = 32;
  xpar_armour * a = xpar_armour_new(p);
  u64 plain = xpar_armour_frame_plain(a), disk = xpar_armour_frame_disk(a);
  u8 * frame = xpar_alloc_raw((sz) disk);
  bm_rng rng;
  u64 begin, elapsed;
  u32 i;
  rng.s = 0xBB67AE8584CAA73BULL;
  bm_fill(&rng, frame, (sz) plain);
  begin = xpar_usec_now();
  Fi(repeat, xpar_armour_encode_frame(a, frame));
  elapsed = xpar_usec_now() - begin;
  bm_rate(tier, operation, plain * repeat, elapsed);
  xpar_free(frame);  xpar_armour_free(a);
}

int xpar_op_benchmark(const xpar_options * o) {
  u32 bad = 0, tiers = 0;
  int n, i, saved_gf, saved_arm;
  xpar_armour_params p8, p16;
  u8 * ref8;  u8 * ref16;
  u64 fx8, fx16;
  bm_rng r;
  xpar_json js;

  xpar_json_init(&js, xpar_stdout, o->json);
  bm_js = o->json ? &js : NULL;
  bm_quiet = o->quiet;
  xpar_gf_init();
  xpar_crc32c_init();
  saved_gf  = xpar_gf_tier();
  saved_arm = xpar_armour_tier();

  bad += bm_check_kats();
  if (!o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: benchmark: V-HASH, V-CRC and V-GEN KATs %s\n",
                 bad ? "failed" : "ok");
  if (o->benchmark_tiers && !o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: benchmark: tier         operation         bytes"
                 "     time       rate\n");

  n = xpar_gf_tier_count();
  Fi(n,
    if (!o->benchmark_tiers && i != saved_gf) continue;
    if (!xpar_gf_tier_usable(i)) {
      if (o->verbose)
        xpar_fprintf(xpar_stderr, "xpar: benchmark: gf tier %s is unavailable; "
                     "skipped\n",
                     xpar_gf_tier_name(i));
      continue;
    }
    if (!xpar_gf_use_tier(i)) continue;
    bad += bm_check_gf(xpar_gf_active(), xpar_gf_tier_name(i));
    if (o->benchmark_tiers && (!o->quiet || o->json))
      bm_measure_gf(xpar_gf_active(), xpar_gf_tier_name(i));
    tiers++;
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: benchmark: gf tier %-8s ok\n",
                   xpar_gf_tier_name(i)));
  xpar_gf_use_tier(saved_gf);

  /*  A shortened GF(2^16) frame keeps the differential test bounded.  */
  xpar_armour_defaults(&p8, 8);
  /*  Depth sets the lane width the kernels see: below one vector the
      vector body is skipped entirely and every tier runs the scalar
      reference. These depths cover a whole body plus a partial tail.  */
  p8.k = p8.n - 32;  p8.depth = 40;
  xpar_armour_defaults(&p16, 16);
  p16.n = 4096;  p16.k = 4096 - 16;  p16.depth = 33;
  r.s = 0xC0FFEE123456789ULL;
  { xpar_armour * a = xpar_armour_new(&p8);
    fx8 = xpar_armour_frame_disk(a);
    xpar_armour_free(a);
    a = xpar_armour_new(&p16);
    fx16 = xpar_armour_frame_disk(a);
    xpar_armour_free(a);
  }
  ref8  = xpar_alloc_raw((sz) fx8);
  ref16 = xpar_alloc_raw((sz) fx16);
  xpar_armour_use_tier(xpar_armour_tier_count() - 1);   /*  Scalar.  */
  bm_armour_frame(&p8, ref8, &r);
  bm_armour_frame(&p16, ref16, &r);

  n = xpar_armour_tier_count();
  Fi(n,
    if (!o->benchmark_tiers && i != saved_arm) continue;
    if (!xpar_armour_tier_usable(i)) continue;
    if (!xpar_armour_use_tier(i)) continue;
    bad += bm_check_armour(xpar_armour_tier_name(i), &p8, ref8, fx8,
                           "armour GF(2^8)");
    bad += bm_check_armour(xpar_armour_tier_name(i), &p16, ref16, fx16,
                           "armour GF(2^16)");
    if (o->benchmark_tiers && (!o->quiet || o->json)) {
      bm_measure_armour(&p8, xpar_armour_tier_name(i), "armour-gf8");
      bm_measure_armour(&p16, xpar_armour_tier_name(i), "armour-gf16");
    }
    tiers++;
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: benchmark: armour tier %-8s ok\n",
                   xpar_armour_tier_name(i)));
  xpar_armour_use_tier(saved_arm);
  xpar_free(ref8);  xpar_free(ref16);

  bad += bm_check_crc32c();
  bad += bm_check_blake3();
  if (!o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: benchmark: crc32c %s, blake3 %s\n",
                 xpar_crc32c_variant(), xpar_blake3_variant());

  if (bm_js) {
    xpar_json_begin(bm_js, "hashes");
    xpar_json_str(bm_js, "crc32c", xpar_crc32c_variant());
    xpar_json_str(bm_js, "blake3", xpar_blake3_variant());
    xpar_json_u64(bm_js, "tiers_checked", tiers);
    xpar_json_end(bm_js);
  }

  if (bad) {
    xpar_fprintf(xpar_stderr, "xpar: benchmark: %" PRIu32
                 " differences across %" PRIu32 " tiers\n", bad, tiers);
    gen_json_result(o, "benchmark", NULL, 0, "failed", XPAR_EXIT_INTERNAL);
    bm_js = NULL;
    return XPAR_EXIT_INTERNAL;
  }
  if (!o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: benchmark: %" PRIu32
                 " tiers agree with scalar\n", tiers);
  gen_json_result(o, "benchmark", NULL, 0, "ok", XPAR_EXIT_OK);
  bm_js = NULL;
  return XPAR_EXIT_OK;
}
