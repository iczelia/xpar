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

/*  Four-pass repair: plan, solve, journal, then apply.  */

#include "ops.h"
#include "auth.h"
#include "chain.h"
#include "vset.h"
#include "volimg.h"

#include "armour.h"
#include "blake3.h"
#include "codec.h"
#include "container.h"
#include "crc32c.h"
#include "json.h"
#include "manifest.h"
#include "pathname.h"
#include "plan.h"
#include "port-fs.h"
#include "resync.h"
#include "slice.h"
#include "undo.h"

/*  Small descriptor cache for sequential and column-strided reads.  */
#define RP_FD_CACHE  8

/*  One cell of the work set: a cell that is damaged somewhere on disk,
    with the bytes that will repair every damaged occurrence of it.  */
typedef struct {
  u64  slice;
  u32  col;
  u64  begin;                 /*  Absolute stream offset.  */
  u64  size;                  /*  Cell width, zero padding included.  */
  u8 * bytes;                 /*  size bytes once solved.  */
  u8   solved;                /*  Verified against the cell CRC.  */
  u8   decode;                /*  No intact occurrence: spends recovery.  */
} rp_cell;

/*  A byte range of one entry that is to be overwritten. `cell` and
    `cell_off` name where the replacement bytes live until pass 3 copies
    them, so nothing is duplicated per edit before it has to be.  */
typedef struct {
  u32 entry;
  u64 off;                    /*  Within the entry.  */
  u64 len;
  u32 cell;
  u64 cell_off;
} rp_edit;

typedef struct {
  u32   entry;
  u64   off, len;
  u8 *  data;                 /*  Owned; the coalesced replacement.  */
  u8 *  old;                  /*  Owned; what was there, for the journal.  */
  u8    trunc;                /*  Remove [off, off+len) instead.  */
  u8    link;                 /*  Link from canon[entry] instead.  */
  u8    relink;               /*  A copy stands where the link belongs.  */
  u8    shadow;               /*  Applied only if the copy could not be linked.  */
} rp_write;

/*  Metadata a run could not restore, counted whatever the output mode.  */
typedef enum {
  RP_META_OWNER, RP_META_MODE, RP_META_SETID, RP_META_TIMES, RP_META_ATIME,
  RP_META_BTIME, RP_META_CTIME, RP_META_ATTRS, RP_META_XATTR,
  RP_META_XATTR_NS, RP_META_SYMLINK, RP_META_COPY, RP_META_CLASSES
} rp_meta_cls;

static const char * const rp_meta_name[RP_META_CLASSES] = {
  "owner", "mode", "setid", "times", "atime", "btime", "ctime", "attrs",
  "xattr", "xattr-namespace", "symlink-unsafe", "materialised-as-copy"
};

typedef struct {
  const xpar_options * o;
  xpar_json  js;
  int        verbose;
  bool       quiet;

  xpar_volimg * vol;
  u32        vol_count, vol_cap;
  u8 **      plain;           /*  Unwrapped ARMG plaintexts, owned.  */
  u32        plain_count, plain_cap;

  xpar_critset crit;
  xpar_key     key;
  u8           master[XPAR_BLAKE3_KEY_LEN];
  bool         key_loaded, keyed, auth_only;
  xpar_setd    sd;
  bool         have_setd;
  u8           set_id[XPAR_SET_ID_LEN];

  xpar_manifest  mf;
  xpar_nameidx   nix;
  xpar_occindex  ox;
  u32            scan_entry_count;
  xpar_geom      geom;
  xpar_tagset    tags;
  u32            tag_have;
  xpar_layt      layt;
  bool           have_layt;
  u32 *          owner;
  xpar_posix_rec ** posix_tab;
  u32 *          posix_tab_count;
  u32            posix_gen_count;

  u64          rec_total;     /*  R, the recovery axis width.  */
  const u8 **  rec;           /*  R pointers into mapped volumes.  */
  u8 *         rec_present;
  u64          rec_avail;

  char *       dir;           /*  Where the protected tree lives.  */
  char *       journal;       /*  base.xparundo.  */
  char **      path;          /*  Per entry, dir + '/' + name.  */
  u8 *         alias;         /*  Per entry: shares an inode with another.  */
  u32 *        canon;         /*  Per entry: the entry it aliases.  */
  u64 *        fsize;         /*  Per entry, as stat found it.  */
  u8 *         fstate;        /*  Bit 0: the file exists. Bit 1: too long.  */
  xpar_resync_map * resync;   /*  Strongly confirmed displaced slices.  */

  u64          armg_corrected;/*  Inner-code corrections while reading.  */
  u64          unrecovered;   /*  Entries repair could not reproduce.  */
  u64          overlong;      /*  Entries with bytes past the recorded end.  */

  xpar_erasures er;           /*  Cells with no intact occurrence.  */
  u8 *          susp;         /*  Cells damaged in at least one place.  */

  rp_cell *  cell;   u32 cell_count, cell_cap;
  rp_edit *  edit;   u32 edit_count, edit_cap;
  rp_write * wr;     u32 wr_count, wr_cap;

  struct { u32 entry;  xpar_file * f;  bool used; } fd[RP_FD_CACHE];
  u32 fd_next;

  u64 bytes_written, writes, cells_copied, cells_decoded;
  u64 entries_repaired, links_repaired, links_missing, links_made;
  u64 opaque;                 /*  Hash fails and nothing can be written.  */
  u64 structure_bad;          /*  Recorded type or link target differs.  */
  u64 names_made;             /*  Empty names the manifest fully describes.  */
  u64 names_failed;           /*  Names the run could not recreate.  */
  u64 links_failed;           /*  Hard-link names that stayed unlinked.  */
  u64 rec_regen, rec_regen_vols;  /*  Recovery slices re-encoded, and where. */
  u64 index_regen;            /*  Index volumes rebuilt from replicas.  */
  u64 stale_regen;            /*  Volumes rewritten to the index's copies.  */
  u64 names_restored;         /*  Volumes put back under their name.  */
  u64 ragged_trimmed;         /*  Volumes cut back to their last packet.  */
  u64 vols_dropped;           /*  Volumes rewritten from packet replicas.  */
  u64 meta_skip[RP_META_CLASSES];  /*  See rp_meta_name.  */
  u8 * hash_bad;              /*  Per entry; owned.  */
  u8 * link_failed;           /*  Per entry; the copy keeps its name.  */
  u8 * io_bad;                /*  Per entry; the host refused its bytes.  */
  u64  io_errors;             /*  Host refusals; they force exit 5.  */
  bool changed;
} rp;

static void rp_note(rp * r, const char * fmt, ...) XPAR_PRINTF(2, 3);

static void rp_note(rp * r, const char * fmt, ...) {
  va_list ap;
  if (r->quiet) return;
  va_start(ap, fmt);
  xpar_vfprintf(r->o->json ? xpar_stderr : xpar_stdout, fmt, ap);
  va_end(ap);
}

static void rp_io_error(rp * r, u32 entry, int err) {
  if (r->io_bad && r->io_bad[entry]) return;
  if (r->io_bad) r->io_bad[entry] = 1;
  r->io_errors++;
  xpar_fprintf(xpar_stderr, "xpar: cannot read '%s': %s\n", r->path[entry],
               xpar_strerror(err));
}

static const struct { u32 bit;  rp_meta_cls cls; } rp_require_map[] = {
  { XPAR_PRES_OWNER, RP_META_OWNER }, { XPAR_PRES_MODE,  RP_META_MODE  },
  { XPAR_PRES_SETID, RP_META_SETID }, { XPAR_PRES_MTIME, RP_META_TIMES },
  { XPAR_PRES_ATIME, RP_META_ATIME }, { XPAR_PRES_BTIME, RP_META_BTIME },
  { XPAR_PRES_CTIME, RP_META_CTIME }, { XPAR_PRES_ATTRS, RP_META_ATTRS },
  { XPAR_PRES_XATTR, RP_META_XATTR },
  { XPAR_PRES_XATTR_ALL, RP_META_XATTR_NS },
  { XPAR_PRES_LINKS, RP_META_SYMLINK }, { XPAR_PRES_LINKS, RP_META_COPY }
};

static bool rp_require_lost(const xpar_options * o, const u64 * counts,
                            bool say) {
  u32 q;
  bool lost = false;
  for (q = 0; q < ARRAY_LEN(rp_require_map); q++) {
    rp_meta_cls cls = rp_require_map[q].cls;
    if (!(o->require & rp_require_map[q].bit) || !counts[cls]) continue;
    if (say)
      xpar_fprintf(xpar_stderr, "xpar: --require: %" PRIu64 " entr%s lost "
                   "%s.\n", counts[cls], counts[cls] == 1 ? "y" : "ies",
                   rp_meta_name[cls]);
    lost = true;
  }
  return lost;
}

static int rp_code(const rp * r, int code) {
  if (r->io_errors) return XPAR_EXIT_IO;
  if (rp_require_lost(r->o, r->meta_skip, false)) return XPAR_EXIT_IO;
  return code;
}

static void rp_meta_skip(u64 * counts, const xpar_options * o,
                         const xpar_entry * e, rp_meta_cls cls,
                         const char * reason) {
  xpar_json js;
  counts[cls]++;
  if (!o->json) return;
  xpar_json_init(&js, xpar_stdout, true);
  xpar_json_begin(&js, "metadata_skipped");
  xpar_json_name(&js, "entry", e->name, e->name_len);
  xpar_json_str(&js, "class", rp_meta_name[cls]);
  xpar_json_str(&js, "reason", reason);
  xpar_json_end(&js);
}

static void rp_meta_report(const xpar_options * o, const u64 * counts,
                           xpar_json * js) {
  u64 total = 0;
  u32 k, shown = 0;
  for (k = 0; k < RP_META_CLASSES; k++) total += counts[k];
  if (o->json) {
    xpar_json_begin(js, "metadata_skipped_total");
    xpar_json_u64(js, "skipped", total);
    for (k = 0; k < RP_META_CLASSES; k++)
      xpar_json_u64(js, rp_meta_name[k], counts[k]);
    xpar_json_end(js);
    return;
  }
  if (!total) return;
  xpar_fprintf(xpar_stderr, "xpar: %" PRIu64 " metadata restoration%s "
               "skipped: ", total, PLURAL(total));
  for (k = 0; k < RP_META_CLASSES; k++) {
    if (!counts[k]) continue;
    xpar_fprintf(xpar_stderr, "%s%s=%" PRIu64, shown++ ? ", " : "",
                 rp_meta_name[k], counts[k]);
  }
  xpar_fputs("\n", xpar_stderr);
}

static void rp_tag(const rp * r, u64 slice, const u8 * bytes,
                   u8 * out, sz n) {
  if (r->keyed)
    xpar_slice_tag_keyed(&r->sd, slice, bytes, r->key.k_slice, out, n);
  else
    xpar_slice_tag(&r->sd, slice, bytes, out, n);
}

static char * rp_tree_path(const char * dir, const xpar_entry * e,
                           xpar_path_status * why) {
  u32 cut = e->name_len;
  char * parent, * out;
  *why = xpar_path_check(e->name, e->name_len, xpar_host_path_flags());
  if (*why != XPAR_PATH_OK) return NULL;
  while (cut && e->name[cut - 1] != '/') cut--;
  if (!cut) return xpar_path_join_n(dir, e->name, e->name_len);
  parent = xpar_path_resolve(dir, e->name, cut - 1, 0, why);
  if (!parent) return NULL;
  out = xpar_path_join_n(parent, e->name + cut, e->name_len - cut);
  xpar_free(parent);
  return out;
}

static void rp_tree_preflight(const xpar_options * o, const xpar_manifest * m,
                              const char * dir) {
  u32 i;
  for (i = 0; i < m->count; i++) {
    const xpar_entry * e = &m->entry[i];
    xpar_path_status why;
    xpar_stat_t st;
    char * p = rp_tree_path(dir, e, &why);
    FATAL_UNLESS("Refusing repair output '%.*s': %s.", p != NULL,
                 (int) e->name_len, e->name, xpar_path_reason(why));
    if (xpar_lstat(p, &st) == 0) {
      if (e->entry_type == XPAR_ENTRY_DIR)
        FATAL_UNLESS("Destination '%s' is not a directory.", st.is_dir, p);
      else {
        FATAL_UNLESS("Destination '%s' exists; -f overwrites it.",
                     o->force, p);
        FATAL_UNLESS("Refusing to replace destination directory '%s'.",
                     !st.is_dir, p);
      }
    }
    xpar_free(p);
  }
}

static bool rp_vol_open(rp * r, const char * path) {
  xpar_volimg v;
  int err = 0;
  xpar_volimg_status st = xpar_volimg_read(&v, path, &err);
  if (st == XPAR_VOLIMG_IO)
    FATAL_IO("Cannot read volume '%s': %s.", path,
             xpar_strerror(err ? err : xpar_errno()));
  if (st != XPAR_VOLIMG_OK) return false;
  if (r->vol_count == r->vol_cap) {
    r->vol_cap = r->vol_cap ? r->vol_cap * 2 : 8;
    r->vol = (xpar_volimg *)
               xpar_realloc(r->vol, r->vol_cap * sizeof(xpar_volimg));
  }
  r->vol[r->vol_count++] = v;
  return true;
}

static void rp_keep_plain(rp * r, u8 * p) {
  if (r->plain_count == r->plain_cap) {
    r->plain_cap = r->plain_cap ? r->plain_cap * 2 : 4;
    r->plain = (u8 **) xpar_realloc(r->plain, r->plain_cap * sizeof(u8 *));
  }
  r->plain[r->plain_count++] = p;
}

static void rp_collect_at(rp * r, const u8 * buf, u64 size, bool resync,
                          bool nested);

/*  Retain decoded plaintext backing collected packet pointers.  */
static void rp_plain(void * user, u8 * plain, u64 len) {
  rp * r = (rp *) user;
  rp_collect_at(r, plain, len, false, true);
  rp_keep_plain(r, plain);
}

static void rp_collect_at(rp * r, const u8 * buf, u64 size, bool resync,
                          bool nested) {
  const xpar_key * key = r->key_loaded ? &r->key : NULL;
  xpar_scan sc;
  xpar_pkt h;
  const u8 * body;
  u64 off;
  xpar_scan_init(&sc, buf, size, key, resync);
  sc.accept_unverified_keyed = !r->key_loaded;
  while (xpar_scan_next(&sc, &h, &body, &off)) {
    /*  Defer ARMG validation to the inner-code sweep.  */
    if (xpar_pkt_is(&h, XPAR_T_ARMG)) continue;
    xpar_critset_add(&r->crit, &h, body);
  }
  xpar_reject_unknown_critical(&sc);
  if (r->verbose > 2)
    rp_note(r, "xpar: scan: %" PRIu64 " packets, %" PRIu64 " bad tags, %"
            PRIu64 " need key.\n",
            sc.emitted,
            sc.skip_checksum,
            sc.skip_keyed);
  /*  ARMG nesting is exactly one level.  */
  if (nested) return;
  { u64 pos = 0, blen = 0;
    while (xpar_verify_next_armg(buf, size, key, &pos, &body, &blen)) {
      u64 plen = 0;
      bool corrected = false;
      u8 * plain = xpar_verify_armg_plain(body, (sz) blen, key, false, &plen,
                                          NULL, &corrected);
      if (!plain) continue;
      if (corrected) r->armg_corrected++;
      rp_keep_plain(r, plain);
      rp_collect_at(r, plain, plen, false, true);
    } }
}

static void rp_collect(rp * r, const u8 * buf, u64 size, bool resync) {
  rp_collect_at(r, buf, size, resync, false);
}

static bool rp_have_setd(const rp * r) {
  For(u32, i, r->crit.count,
      if (xpar_pkt_is(&r->crit.pkt[i].hdr, XPAR_T_SETD)) return true)
  return false;
}

/*  Recover a failed ARMG only when the ordinary scan found no SETD; its
    failed packet tag is the prerequisite for inner decoding.  */
static void rp_salvage(rp * r, const u8 * buf, u64 size) {
  xpar_armg_salvage(buf, size, r->key_loaded ? &r->key : NULL, rp_plain, r);
}

static void rp_authenticate(rp * r) {
  const xpar_crit_pkt * p = NULL;
  xpar_auth a;
  u32 i;
  for (i = 0; i < r->crit.count; i++)
    if (xpar_pkt_is(&r->crit.pkt[i].hdr, XPAR_T_AUTH) &&
        !xpar_memcmp(r->crit.pkt[i].hdr.set_id, r->set_id,
                     XPAR_SET_ID_LEN)) { p = &r->crit.pkt[i]; break; }
  if (!p) return;
  if (xpar_auth_read(p->body, (sz) p->body_len, &a) != XPAR_OK)
    FATAL_CODE(XPAR_EXIT_AUTH, "The AUTH packet is malformed.");
  if (!r->key_loaded)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "Repairing an authenticated set requires --auth-key=FILE; "
               "keyless access is read-only.");
  if (!xpar_auth_key_ok(&a, r->master))
    FATAL_CODE(XPAR_EXIT_AUTH, "The authentication key is wrong for this set.");
  r->keyed = true;
  r->auth_only = !a.unkeyed_retained;
}

static void rp_key_preflight(rp * r) {
  u32 i;
  if (!r->key_loaded) return;
  r->key_loaded = false;
  for (i = 0; i < r->vol_count; i++)
    rp_collect(r, r->vol[i].data, r->vol[i].size, true);
  if (!rp_have_setd(r))
    for (i = 0; i < r->vol_count; i++)
      rp_salvage(r, r->vol[i].data, r->vol[i].size);
  for (i = 0; i < r->crit.count; i++) {
    xpar_auth a;
    const xpar_crit_pkt * p = &r->crit.pkt[i];
    if (!xpar_pkt_is(&p->hdr, XPAR_T_AUTH) ||
        xpar_auth_read(p->body, (sz) p->body_len, &a) != XPAR_OK) continue;
    if (!xpar_auth_key_ok(&a, r->master))
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "The authentication key is wrong for this set.");
    break;
  }
  for (i = 0; i < r->plain_count; i++) xpar_free(r->plain[i]);
  xpar_free(r->plain);
  r->plain = NULL;  r->plain_count = r->plain_cap = 0;
  xpar_critset_free(&r->crit);
  xpar_critset_init(&r->crit);
  r->key_loaded = true;
  /*  Without AUTH, defer key validation to packet and slice tags.  */
}

/*  Resolve chain heads and generation selectors from SETD identities, not
    filenames.  */
static void rp_pick_setd(rp * r) {
  u32 i, j;
  const xpar_crit_pkt * want = NULL;
  xpar_setd sd;
  for (i = 0; i < r->crit.count; i++) {
    const xpar_crit_pkt * p = &r->crit.pkt[i];
    bool named = false, head = true;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_SETD)) continue;
    if (xpar_setd_read(p->body, (sz) p->body_len, &sd) != XPAR_OK) {
      xpar_setd_free(&sd);
      continue;
    }
    if (r->o->gen_count) {
      const xpar_genref * g = &r->o->gens[0];
      named = g->by_id
                ? xpar_hex_prefix(p->hdr.set_id, XPAR_SET_ID_LEN,
                                  g->id_prefix)
                : sd.generation == (u32) g->number;
    }
    for (j = 0; j < r->crit.count && head; j++) {
      const xpar_crit_pkt * q = &r->crit.pkt[j];
      xpar_setd other;
      if (j == i || !xpar_pkt_is(&q->hdr, XPAR_T_SETD)) continue;
      if (xpar_setd_read(q->body, (sz) q->body_len, &other) != XPAR_OK) {
        xpar_setd_free(&other);
        continue;
      }
      if (!xpar_memcmp(other.parent_set_id, p->hdr.set_id,
                       XPAR_SET_ID_LEN)) head = false;
      xpar_setd_free(&other);
    }
    xpar_setd_free(&sd);
    if (r->o->gen_count ? named : head) { want = p;  break; }
  }
  FATAL_UNLESS("No set descriptor survived in '%s'; the critical group is "
               "gone. Try another volume of the set, or "
               "`xpar recover-prologue`.", want != NULL, r->o->set);
  if (xpar_setd_read(want->body, (sz) want->body_len, &r->sd) != XPAR_OK)
    FATAL_FORMAT("The set descriptor is malformed.");
  xpar_memcpy(r->set_id, want->hdr.set_id, XPAR_SET_ID_LEN);
  r->have_setd = true;
}

static void rp_read_manifest(rp * r) {
  xpar_chain c;
  u32 * owner = NULL;
  u32 g;
  xpar_gchain_load(r->o, &c);
  g = xpar_gchain_select(&c, r->o->gen_count ? &r->o->gens[0] : NULL);
  if (xpar_memcmp(c.gen[g].set_id, r->set_id, XPAR_SET_ID_LEN))
    FATAL_FORMAT("The repair reader and chain reader selected different "
                 "generations.");
  xpar_gchain_manifest(&c, g, &r->mf, &owner);
  r->owner = owner;
  r->scan_entry_count = r->mf.count;
  if (r->o->repair_head_set) {
    xpar_genref ref;
    xpar_manifest head;
    u32 * head_owner = NULL;
    u32 h;
    xpar_memset(&ref, 0, sizeof ref);
    ref.by_id = true;
    ref.id_prefix = r->o->repair_head_id;
    h = xpar_gchain_select(&c, &ref);
    if (h != g) {
      xpar_manifest selected = r->mf;
      u64 count;
      /*  Prefer head occurrences; retain the selected generation as a
          resynchronisation source for bytes the head no longer names.  */
      xpar_memset(&r->mf, 0, sizeof r->mf);
      xpar_gchain_manifest(&c, h, &head, &head_owner);
      count = (u64) head.count + selected.count;
      FATAL_UNLESS("The repair manifest has too many entries.",
                   count <= UINT32_MAX &&
                   count <= (u64) (sz) -1 / sizeof(xpar_entry));
      r->mf.entry = (xpar_entry *) xpar_alloc_raw(
                      (sz) MAX(count, 1) * sizeof(xpar_entry));
      if (head.count)
        xpar_memcpy(r->mf.entry, head.entry,
                    (sz) head.count * sizeof(xpar_entry));
      if (selected.count)
        xpar_memcpy(r->mf.entry + head.count, selected.entry,
                    (sz) selected.count * sizeof(xpar_entry));
      r->mf.count = r->mf.cap = (u32) count;
      r->mf.stream_base = selected.stream_base;
      r->mf.stream_length = selected.stream_length;
      r->scan_entry_count = head.count;
      /*  Free source arrays after moving their entries.  */
      { u32 q;
        for (q = 0; head.source && q < head.count; q++)
          xpar_free(head.source[q]);
        for (q = 0; selected.source && q < selected.count; q++)
          xpar_free(selected.source[q]); }
      xpar_free(head.entry);   xpar_free(head.source);
      head.entry = NULL;  head.source = NULL;  head.count = 0;
      xpar_free(selected.entry);  xpar_free(selected.source);
      selected.entry = NULL;  selected.source = NULL;
      selected.count = 0;
      xpar_free(r->owner);  r->owner = NULL;
      xpar_free(head_owner);
    }
  }
  r->posix_gen_count = c.gen_count;
  r->posix_tab = (xpar_posix_rec **) xpar_calloc(c.gen_count,
                                                  sizeof *r->posix_tab);
  r->posix_tab_count = (u32 *) xpar_calloc(c.gen_count, sizeof(u32));
  for (g = 0; g < c.gen_count; g++)
    r->posix_tab_count[g] = xpar_gchain_posix(&c, g, &r->posix_tab[g]);
  xpar_gchain_free(&c);

  xpar_nameidx_build(&r->mf, &r->nix);
  xpar_occindex_build(&r->mf, &r->ox);
}

static void rp_read_tags(rp * r) {
  u32 i, cps = r->sd.cell_bytes ? r->geom.cells_per_slice : 0;
  u64 input = 0;
  for (i = 0; i < r->vol_count; i++) input += r->vol[i].size;
  if (!xpar_tagset_init(&r->tags, r->geom.slice_count,
                        r->sd.slice_tag_len, cps, !r->auth_only, input))
    FATAL_FORMAT("The slice tables claim more bytes than the volumes "
                 "hold.");
  for (i = 0; i < r->crit.count; i++) {
    const xpar_crit_pkt * p = &r->crit.pkt[i];
    if (xpar_memcmp(p->hdr.set_id, r->set_id, XPAR_SET_ID_LEN)) continue;
    if (xpar_pkt_is(&p->hdr, XPAR_T_SLCR)) {
      xpar_slcr t;
      if (xpar_slcr_read(p->body, (sz) p->body_len, &t) == XPAR_OK) {
        if (xpar_tagset_slcr(&r->tags, &t) != XPAR_OK)
          FATAL_FORMAT("Slice CRC table coverage overlaps or is out of range.");
        xpar_slcr_free(&t);
      }
    } else if (xpar_pkt_is(&p->hdr, XPAR_T_SLTG)) {
      xpar_sltg t;
      if (xpar_sltg_read(p->body, (sz) p->body_len, &t) == XPAR_OK) {
        if (xpar_tagset_sltg(&r->tags, &t) != XPAR_OK)
          FATAL_FORMAT("Slice tag table coverage overlaps or is out of range.");
        xpar_sltg_free(&t);
      }
    } else if (xpar_pkt_is(&p->hdr, XPAR_T_SLCL)) {
      xpar_slcl t;
      if (xpar_slcl_read(p->body, (sz) p->body_len, r->sd.slice_size,
                         &t) == XPAR_OK) {
        if (t.cell_bytes != r->sd.cell_bytes ||
            xpar_tagset_slcl(&r->tags, &t) != XPAR_OK)
          FATAL_FORMAT("Cell table geometry or coverage is malformed.");
        xpar_slcl_free(&t);
      }
    }
  }
  r->tag_have = xpar_tagset_complete(&r->tags);
  FATAL_UNLESS("Slice integrity tables are incomplete; damage cannot be "
               "located.",
               (r->tag_have & XPAR_TAGS_CRC) ||
               (r->keyed && (r->tag_have & XPAR_TAGS_TAG)) ||
               !r->geom.slice_count);
  if (r->keyed && (!(r->tag_have & XPAR_TAGS_TAG) ||
                   r->tags.t.tag_len != 16))
    FATAL_CODE(XPAR_EXIT_AUTH,
               "An authenticated set must carry complete 16-byte slice "
               "tags before it can be repaired.");
  if (r->sd.cell_bytes && !(r->tag_have & XPAR_TAGS_CELL))
    rp_note(r, "xpar: the cell table is missing or damaged; falling back "
               "to slice-granular erasures, which this set can survive "
               "much less of. `scrub --rebuild-cells` restores it.\n");
}

static void rp_open_recovery(rp * r) {
  u32 i;
  u64 limit = xpar_setd_recovery_limit(&r->sd);
  u64 e;
  const xpar_crit_pkt * p = NULL;
  for (i = 0; i < r->crit.count && !p; i++)
    if (xpar_pkt_is(&r->crit.pkt[i].hdr, XPAR_T_LAYT) &&
        !xpar_memcmp(r->crit.pkt[i].hdr.set_id, r->set_id, XPAR_SET_ID_LEN))
      p = &r->crit.pkt[i];
  if (p && xpar_layt_read(p->body, (sz) p->body_len, &r->layt) == XPAR_OK) {
    r->have_layt = true;
    for (i = 0; i < r->layt.count; i++) {
      const xpar_vol * v = &r->layt.vol[i];
      char * path;
      u32 k;
      bool seen = false;
      if (v->kind == XPAR_VOL_RECOVERY) {
        u64 first = v->recovery_first, count = v->byte_length;
        if (!count || first >= limit || count > limit - first)
          FATAL_FORMAT("Recovery volume range exceeds the declared axis.");
        if (first + count > r->rec_total) r->rec_total = first + count;
      }
      if (v->kind != XPAR_VOL_RECOVERY || !v->name) continue;
      path = xpar_path_vol(r->dir, v->name);
      for (k = 0; k < r->vol_count && !seen; k++)
        if (!xpar_strcmp(r->vol[k].path, path)) seen = true;
      if (!seen && rp_vol_open(r, path))
        rp_collect(r, r->vol[r->vol_count - 1].data,
                   r->vol[r->vol_count - 1].size, false);
      xpar_free(path);
    }
  }
  /*  Without LAYT, infer the recovery axis from packets actually found.  */
  for (i = 0; i < r->crit.count; i++) {
    const xpar_crit_pkt * q = &r->crit.pkt[i];
    xpar_rcvs rc;
    if (!xpar_pkt_is(&q->hdr, XPAR_T_RCVS)) continue;
    if (xpar_memcmp(q->hdr.set_id, r->set_id, XPAR_SET_ID_LEN)) continue;
    if (xpar_rcvs_read(q->body, (sz) q->body_len, r->sd.slice_size,
                       &rc) != XPAR_OK) continue;
    if (rc.exponent >= limit)
      FATAL_FORMAT("Recovery exponent exceeds the declared axis.");
    if (rc.exponent + 1 > r->rec_total) r->rec_total = rc.exponent + 1;
  }
  r->rec = (const u8 **) xpar_calloc(r->rec_total ? (sz) r->rec_total : 1,
                                     sizeof(const u8 *));
  r->rec_present = (u8 *) xpar_calloc(r->rec_total ? (sz) r->rec_total : 1,
                                      1);
  for (e = 0; e < r->rec_total; e++) {
    const xpar_crit_pkt * q = xpar_critset_find(&r->crit, r->set_id,
                                                XPAR_T_RCVS, e);
    xpar_rcvs rc;
    if (!q) continue;
    if (xpar_rcvs_read(q->body, (sz) q->body_len, r->sd.slice_size,
                       &rc) != XPAR_OK) continue;
    if (rc.length != r->sd.slice_size) continue;
    r->rec[e] = rc.data;  r->rec_present[e] = 1;  r->rec_avail++;
  }
}

static xpar_file * rp_entry_file(rp * r, u32 entry) {
  u32 i;
  xpar_file * f;
  for (i = 0; i < RP_FD_CACHE; i++)
    if (r->fd[i].used && r->fd[i].entry == entry) return r->fd[i].f;
  f = xpar_open(r->path[entry], XPAR_O_RDONLY | XPAR_O_NOFOLLOW);
  if (!f) {
    if (!xpar_errno_absent(xpar_errno())) rp_io_error(r, entry, xpar_errno());
    return NULL;
  }
  i = r->fd_next++ % RP_FD_CACHE;
  if (r->fd[i].used) xpar_close(r->fd[i].f);
  r->fd[i].used = true;  r->fd[i].entry = entry;  r->fd[i].f = f;
  return f;
}

static void rp_close_files(rp * r) {
  u32 i;
  for (i = 0; i < RP_FD_CACHE; i++)
    if (r->fd[i].used) { xpar_close(r->fd[i].f);  r->fd[i].used = false; }
}

/*  Close cached descriptors before replacing an entry.  */
static void rp_close_entry(rp * r, u32 entry) {
  u32 i;
  for (i = 0; i < RP_FD_CACHE; i++)
    if (r->fd[i].used && r->fd[i].entry == entry) {
      xpar_close(r->fd[i].f);  r->fd[i].used = false;
    }
}

/*  Read an entry range, zero-filling gaps; false distinguishes missing
    bytes from stored zeros.  */
static bool rp_read_entry_raw(rp * r, u32 entry, u64 off, u64 len,
                              u8 * dst) {
  xpar_file * f = rp_entry_file(r, entry);
  sz got;
  xpar_memset(dst, 0, (sz) len);
  if (!f) return false;
  if (len > (u64) (sz) -1) return false;
  got = xpar_pread(f, dst, (sz) len, off);
  if (got != (sz) len && xpar_error(f)) rp_io_error(r, entry, xpar_error(f));
  return got == (sz) len;
}


static bool rp_read_entry_resynced(rp * r, u32 entry, u64 off, u64 len,
                                   u8 * dst) {
  xpar_file * f;
  u64 physical;
  sz got;
  if (!xpar_resync_map_shift(&r->resync[entry], off, &physical))
    return rp_read_entry_raw(r, entry, off, len, dst);
  xpar_memset(dst, 0, (sz) len);
  if (len > (u64) (sz) -1 || UINT64_MAX - physical < len) return false;
  f = rp_entry_file(r, entry);
  if (!f) return false;
  got = xpar_pread(f, dst, (sz) len, physical);
  if (got != (sz) len && xpar_error(f)) rp_io_error(r, entry, xpar_error(f));
  return got == (sz) len;
}

typedef struct {
  rp * r;
  xpar_file * f;
  const xpar_resync_probe * probe;
  u8 * buf;
  u32 entry;
} rp_confirm;

static bool rp_confirm_at(void * user, u32 at, u64 physical) {
  rp_confirm * c = (rp_confirm *) user;
  const xpar_resync_probe * p = &c->probe[at];
  u8 got[XPAR_BLAKE3_OUT_LEN];
  u64 z = c->r->geom.slice_size;
  if (physical > UINT64_MAX - z) return false;
  if (xpar_pread(c->f, c->buf, (sz) z, physical) != (sz) z) {
    if (xpar_error(c->f)) rp_io_error(c->r, c->entry, xpar_error(c->f));
    return false;
  }
  rp_tag(c->r, p->slice, c->buf, got, c->r->tags.t.tag_len);
  return xpar_blake3_tag_equal(
    got, c->r->tags.t.slice_tag + p->slice * c->r->tags.t.tag_len,
    c->r->tags.t.tag_len);
}

static xpar_resync_probe * rp_entry_probes(rp * r, u32 entry,
                                            u32 * count) {
  const xpar_entry * e = &r->mf.entry[entry];
  xpar_resync_probe * p = NULL;
  u32 cap = 0, n = 0, k;
  u64 file_off = 0, z = r->geom.slice_size;
  u64 gen_begin = r->geom.stream_base;
  u64 gen_end = gen_begin + r->geom.stream_length;
  for (k = 0; k < e->extent_count; k++) {
    const xpar_extent * x = &e->extents[k];
    u64 begin = MAX(x->stream_offset, gen_begin), end;
    if (x->stream_offset > UINT64_MAX - x->length) goto next;
    end = MIN(x->stream_offset + x->length, gen_end);
    if (begin < end) {
      u64 rem = (begin - gen_begin) % z;
      u64 at = rem ? begin + (z - rem) : begin;
      for (; at <= end && end - at >= z; at += z) {
        xpar_occurrence o;
        u64 run, slice = (at - gen_begin) / z;
        if (!xpar_occindex_canonical(&r->ox, at, &o, &run) ||
            o.entry != entry || o.extent != k || run < z) continue;
        if (n == cap) {
          cap = cap ? cap * 2 : 16;
          p = (xpar_resync_probe *)
                xpar_realloc(p, cap * sizeof(xpar_resync_probe));
        }
        p[n].crc = r->tags.t.slice_crc[slice];
        p[n].expected = file_off + at - x->stream_offset;
        p[n].slice = slice;
        n++;
      }
    }
next:
    file_off += x->length;
  }
  *count = n;
  return p;
}

static void rp_resync_entry(rp * r, u32 entry) {
  const xpar_entry * e = &r->mf.entry[entry];
  xpar_resync_probe * p;
  xpar_resync_opts opt;
  xpar_resync_outcome got;
  rp_confirm confirm;
  xpar_file * f;
  xpar_stat_t st;
  u64 * located;
  u64 z = r->geom.slice_size;
  u32 n, i, displaced = 0;
  if (e->entry_type != XPAR_ENTRY_REGULAR || !e->extent_count ||
      r->alias[entry] || xpar_lstat(r->path[entry], &st) != 0 ||
      !st.is_regular) return;
  p = rp_entry_probes(r, entry, &n);
  if (!n) { xpar_free(p);  return; }
  f = rp_entry_file(r, entry);
  if (!f) { xpar_free(p);  return; }
  confirm.r = r;  confirm.f = f;  confirm.probe = p;
  confirm.entry = entry;
  confirm.buf = (u8 *) xpar_alloc_raw((sz) z);
  located = (u64 *) xpar_alloc_raw((sz) n * sizeof(u64));

  opt.mode       = (u32) r->o->resync;
  opt.step       = r->o->resync_step;
  opt.window     = r->o->resync_window;
  opt.exhaustive = r->o->resync_exhaustive;
  opt.have_tags  = (r->tag_have & XPAR_TAGS_TAG) != 0 &&
                   r->tags.t.slice_tag != NULL;
  xpar_resync_entry(f, st.size, z, e->length, p, n, &opt,
                    rp_confirm_at, &confirm, confirm.buf, located, &got);
  /*  Retry unresolved chain displacement exhaustively.  */
  if (got.candidates && r->o->chain_member && !opt.exhaustive) {
    opt.mode = XPAR_RESYNC_ALWAYS;
    opt.exhaustive = true;
    xpar_resync_entry(f, st.size, z, e->length, p, n, &opt,
                      rp_confirm_at, &confirm, confirm.buf, located, &got);
  }

  if (xpar_error(f)) rp_io_error(r, entry, xpar_error(f));
  if (got.need_tags && r->verbose)
    rp_note(r, "xpar: %s: resync needs strong slice tags; using erasures.\n",
            r->path[entry]);
  if (got.candidates)
    rp_note(r, "xpar: %s: no dominant displacement among %" PRIu64
               " candidates; use --resync=always --resync-exhaustive.\n",
            r->path[entry], got.candidates);
  for (i = 0; i < n; i++)
    if (located[i] != UINT64_MAX) {
      /*  Delta zero confirms a slice in place.  */
      if (located[i] != p[i].expected) displaced++;
      xpar_resync_map_add(&r->resync[entry], p[i].expected, located[i]);
    }
  if (got.clipped)
    rp_note(r, "xpar: %s: %" PRIu64 " matches lie outside --resync-window; "
               "raise it to consider them.\n", r->path[entry], got.clipped);
  if (displaced)
    rp_note(r, "xpar: %s: found %" PRIu32 " displaced slices (%" PRIu64
               " confirmations).\n", r->path[entry], displaced,
            got.confirmations);
  else if (r->resync[entry].count && r->verbose)
    rp_note(r, "xpar: %s: resync: no displacement.\n", r->path[entry]);
  xpar_free(located);
  xpar_free(confirm.buf);  xpar_free(p);
}

static void rp_resync_tree(rp * r) {
  if (r->o->resync == XPAR_RESYNC_OFF) return;
  For(u32, i, r->mf.count, rp_resync_entry(r, i))
}

/*  Sorted by (slice, column); stable after classification.  */
static rp_cell * rp_cell_at(rp * r, u64 slice, u32 col) {
  u64 key = slice * r->geom.cells_per_slice + col;
  u32 lo = 0, hi = r->cell_count;
  while (lo < hi) {
    u32 mid = lo + (hi - lo) / 2;
    u64 k = r->cell[mid].slice * r->geom.cells_per_slice + r->cell[mid].col;
    if (k < key) lo = mid + 1;
    else if (k > key) hi = mid;
    else return &r->cell[mid];
  }
  return NULL;
}

/*  Substitute solved cells before decoding another erasure in the same
    column. Journal reads bypass this patch so they retain the old bytes.  */
static void rp_patch(rp * r, u64 off, u64 len, u8 * dst) {
  u64 axis = r->geom.stream_base + r->geom.slice_count * r->geom.slice_size;
  u64 p = off, end = MIN(off + len, axis);
  if (!r->cell_count) return;
  while (p < end) {
    u64 s  = xpar_slice_of(&r->geom, p);
    u32 c  = xpar_cell_of(&r->geom, p);
    u64 cb = xpar_cell_begin(&r->geom, s, c);
    u64 cs = xpar_cell_size(&r->geom, c);
    rp_cell * w = rp_cell_at(r, s, c);
    u64 lo = MAX(cb, p), hi = MIN(cb + cs, end);
    if (w && w->solved && w->bytes && lo < hi)
      xpar_memcpy(dst + (lo - off), w->bytes + (lo - cb), (sz) (hi - lo));
    p = cb + cs;
  }
}

/*  Read canonical stream bytes with solved-cell substitutions and zero
    padding beyond the stored length.  */
static bool rp_read_stream(rp * r, u64 off, u64 len, u8 * dst) {
  u64 base = off, end = off + len;
  u64 l_end = r->geom.stream_base + r->geom.stream_length;
  bool ok = true;
  xpar_memset(dst, 0, (sz) len);
  while (off < end && off < l_end) {
    xpar_occurrence o;
    u64 run, take;
    if (!xpar_occindex_canonical(&r->ox, off, &o, &run)) {
      /*  Skip alignment padding to the next extent.  */
      u64 gap = xpar_occindex_next(&r->ox, off, MIN(end, l_end)) - off;
      if (!gap) { ok = false;  break; }
      off += gap;
      continue;
    }
    take = MIN(run, end - off);
    if (off + take > l_end) take = l_end - off;
    if (!rp_read_entry_raw(r, o.entry,
                           o.file_offset + (off - o.stream_offset), take,
                           dst + (off - base)))
      ok = false;
    off += take;
  }
  rp_patch(r, base, len, dst);
  return ok;
}

static u8 * rp_cell_key(rp * r, u64 slice, u32 col) {
  return &r->susp[slice * r->geom.cells_per_slice + col];
}

/*  A failed strong slice tag overrides clean cell CRCs; erase the whole
    slice rather than accept a CRC collision.  */
static void rp_scan_stream(rp * r, xpar_progress_t * pg) {
  u64 s, z = r->geom.slice_size;
  u32 k = r->geom.cells_per_slice;
  u8 * buf = (u8 *) xpar_alloc_raw((sz) z);
  /*  Reuse the CRC shift operator until the cell length changes.  */
  u32 comb_op[XPAR_CRC32C_OP_WORDS];
  u64 comb_len = 0;
  for (s = 0; s < r->geom.slice_count; s++) {
    u32 c;
    bool cells_ok = true;
    u32 slice_crc = 0;
    rp_read_stream(r, xpar_slice_begin(&r->geom, s), z, buf);
    if (r->keyed) {
      u8 tag[16];
      bool local = false;
      rp_tag(r, s, buf, tag, r->tags.t.tag_len);
      if (xpar_blake3_tag_equal(tag, r->tags.t.slice_tag +
                                    s * r->tags.t.tag_len,
                                r->tags.t.tag_len)) {
        xpar_progress_tick(pg, z);
        continue;
      }
      if ((r->tag_have & XPAR_TAGS_CELL) && r->tags.t.cell_crc)
        for (c = 0; c < k; c++) {
          u64 sz_c = xpar_cell_size(&r->geom, c);
          u64 at = (u64) c * (r->geom.cell_bytes ? r->geom.cell_bytes : z);
          if (xpar_crc32c(0, buf + at, (sz) sz_c) !=
              r->tags.t.cell_crc[s * k + c]) {
            xpar_cell_mark(&r->er, s, c);
            *rp_cell_key(r, s, c) = 1;
            local = true;
          }
        }
      if (!local)
        for (c = 0; c < k; c++) {
          xpar_cell_mark(&r->er, s, c);
          *rp_cell_key(r, s, c) = 1;
        }
      xpar_progress_tick(pg, z);
      continue;
    }
    for (c = 0; c < k; c++) {
      u64 sz_c = xpar_cell_size(&r->geom, c);
      u64 at   = (u64) c * (r->geom.cell_bytes ? r->geom.cell_bytes : z);
      u32 crc  = xpar_crc32c(0, buf + at, (sz) sz_c);
      if (!c) slice_crc = crc;
      else {
        if (sz_c != comb_len) {
          xpar_crc32c_shift_op(comb_op, sz_c);
          comb_len = sz_c;
        }
        slice_crc = xpar_crc32c_combine_op(comb_op, slice_crc, crc);
      }
      if (!(r->tag_have & XPAR_TAGS_CELL) || !r->tags.t.cell_crc) continue;
      if (crc != r->tags.t.cell_crc[s * k + c]) {
        xpar_cell_mark(&r->er, s, c);
        *rp_cell_key(r, s, c) = 1;
        cells_ok = false;
      }
    }
    if (!(r->tag_have & XPAR_TAGS_CELL) || !r->tags.t.cell_crc) {
      if (slice_crc != r->tags.t.slice_crc[s]) {
        xpar_erasures_mark_slice(&r->er, s);
        for (c = 0; c < k; c++) *rp_cell_key(r, s, c) = 1;
        cells_ok = false;
      }
    }
    if (cells_ok && (r->tag_have & XPAR_TAGS_TAG) && r->tags.t.slice_tag) {
      u8 tag[16];
      rp_tag(r, s, buf, tag, r->tags.t.tag_len);
      if (!xpar_blake3_tag_equal(tag, r->tags.t.slice_tag +
                                 s * r->tags.t.tag_len, r->tags.t.tag_len)) {
        for (c = 0; c < k; c++) {
          xpar_cell_mark(&r->er, s, c);
          *rp_cell_key(r, s, c) = 1;
        }
      }
    }
    xpar_progress_tick(pg, z);
  }
  xpar_free(buf);
}

/*  Check an occurrence's part of a cell, filling the rest from the
    canonical stream before comparing with SLCL.  */
typedef struct { rp * r;  u64 slice;  u32 col;  u8 * buf; } rp_probe;

static bool rp_occ_raw_intact(const xpar_occurrence * o, rp_probe * pr) {
  rp * r = pr->r;
  u64 begin = xpar_cell_begin(&r->geom, pr->slice, pr->col);
  u64 size = xpar_cell_size(&r->geom, pr->col);
  u64 lo = MAX(begin, o->stream_offset);
  u64 hi = MIN(begin + size, o->stream_offset + o->length);
  u64 l_end = r->geom.stream_base + r->geom.stream_length;
  u32 crc;
  if (hi > l_end) hi = l_end;
  if (lo >= hi) return false;
  rp_read_stream(r, begin, size, pr->buf);
  if (!rp_read_entry_raw(r, o->entry,
                         o->file_offset + (lo - o->stream_offset), hi - lo,
                         pr->buf + (lo - begin))) return false;
  crc = xpar_crc32c(0, pr->buf, (sz) size);
  if ((r->tag_have & XPAR_TAGS_CELL) && r->tags.t.cell_crc)
    return crc == r->tags.t.cell_crc[pr->slice * r->geom.cells_per_slice +
                                     pr->col];
  /* Fall back to a slice CRC only for single-cell slices. */
  if (r->geom.cells_per_slice > 1) return false;
  if (!(r->tag_have & XPAR_TAGS_CRC) || !r->tags.t.slice_crc) return false;
  return crc == r->tags.t.slice_crc[pr->slice];
}

static bool rp_occ_intact(const xpar_occurrence * o, void * user) {
  rp_probe * pr = (rp_probe *) user;
  rp * r = pr->r;
  u64 begin = xpar_cell_begin(&r->geom, pr->slice, pr->col);
  u64 size  = xpar_cell_size(&r->geom, pr->col);
  u64 lo    = MAX(begin, o->stream_offset);
  u64 hi    = MIN(begin + size, o->stream_offset + o->length);
  u64 l_end = r->geom.stream_base + r->geom.stream_length;
  u32 crc;
  if (hi > l_end) hi = l_end;
  if (lo >= hi) return false;
  /*  Canonical failures outside this occurrence do not classify it.  */
  if (rp_occ_raw_intact(o, pr)) return true;
  if (!r->resync[o->entry].count) return false;
  rp_read_stream(r, begin, size, pr->buf);
  if (!rp_read_entry_resynced(r, o->entry,
                              o->file_offset + (lo - o->stream_offset),
                              hi - lo, pr->buf + (lo - begin))) return false;
  crc = xpar_crc32c(0, pr->buf, (sz) size);
  if ((r->tag_have & XPAR_TAGS_CELL) && r->tags.t.cell_crc)
    return crc == r->tags.t.cell_crc[pr->slice * r->geom.cells_per_slice +
                                     pr->col];
  /* Fall back to a slice CRC only for single-cell slices. */
  if (r->geom.cells_per_slice > 1) return false;
  if (!(r->tag_have & XPAR_TAGS_CRC) || !r->tags.t.slice_crc) return false;
  return crc == r->tags.t.slice_crc[pr->slice];
}

/*  Skip entries already certified through canonical verified cells.  */
static void rp_scan_entries(rp * r, xpar_progress_t * pg) {
  u32 i;
  u64 z = r->geom.slice_size;
  u8 * buf  = (u8 *) xpar_alloc_raw((sz) z);
  u8 * cell = (u8 *) xpar_alloc_raw((sz) (r->geom.cell_bytes ?
                                          r->geom.cell_bytes : z));
  for (i = 0; i < r->scan_entry_count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    u64 gen_begin = r->geom.stream_base;
    u64 gen_end = gen_begin + r->geom.stream_length;
    xpar_stat_t st;
    bool certified = true, exists, touches = false;
    u32 k;
    if (e->entry_type == XPAR_ENTRY_HARDLINK || r->alias[i]) continue;
    if (e->entry_type != XPAR_ENTRY_REGULAR || !e->extent_count) continue;
    exists = xpar_lstat(r->path[i], &st) == 0;
    r->fsize[i]  = exists ? st.size : 0;
    r->fstate[i] = exists;
    if (!exists || st.size != e->length) certified = false;
    /*  Whether the entry lies in this generation is independent of whether
        it is certified, so it is settled before the cell walk, whose
        `certified` guard is only an early exit.  */
    for (k = 0; k < e->extent_count; k++)
      if (e->extents[k].stream_offset >= gen_begin &&
          e->extents[k].stream_offset < gen_end) { touches = true;  break; }
    for (k = 0; k < e->extent_count && certified; k++) {
      const xpar_extent * x = &e->extents[k];
      u64 p;
      if (x->stream_offset < gen_begin || x->stream_offset >= gen_end)
        continue;
      for (p = x->stream_offset; p < x->stream_offset + x->length;) {
        u64 s = xpar_slice_of(&r->geom, p);
        u32 c = xpar_cell_of(&r->geom, p);
        xpar_occurrence o;
        u64 run;
        if (xpar_cell_bad(&r->er, s, c)) { certified = false;  break; }
        if (!xpar_occindex_canonical(&r->ox, p, &o, &run) || o.entry != i) {
          certified = false;  break;
        }
        p = xpar_cell_begin(&r->geom, s, c) + xpar_cell_size(&r->geom, c);
      }
    }
    /*  Mark overlong owned entries for truncation.  */
    if (exists && touches && st.size > e->length) r->fstate[i] |= 2;
    if (!touches) continue;
    /*  Clean CRCs certify an entry only where a strong slice tag backs
        them; without one the entry hash is the only forgery check.  */
    if (certified && (r->tag_have & XPAR_TAGS_TAG) && r->tags.t.slice_tag)
      continue;
    /*  The entry hash strongly covers aliases; cell CRCs localise a
        failing occurrence.  */
    { xpar_blake3_t h;
      u8 got[32];
      u64 fo = 0;
      bool short_file = false;
      if (r->auth_only) xpar_blake3_init_keyed(&h, r->key.k_file);
      else              xpar_blake3_init(&h);
      for (k = 0; k < e->extent_count; k++) {
        u64 left = e->extents[k].length, at = fo;
        while (left) {
          u64 take = MIN(left, z);
          if (!rp_read_entry_raw(r, i, at, take, buf)) short_file = true;
          xpar_blake3_update(&h, buf, (sz) take);
          at += take;  left -= take;
        }
        fo += e->extents[k].length;
      }
      xpar_blake3_final(&h, got, 32);
      xpar_progress_tick(pg, e->length);
      if (!short_file && !xpar_memcmp(got, e->content_hash, 32)) continue;
      /*  Mark only cells that fail with this occurrence substituted.  */
      fo = 0;
      for (k = 0; k < e->extent_count; k++) {
        xpar_occurrence o;
        u64 p, end = e->extents[k].stream_offset + e->extents[k].length;
        o.entry = i;  o.extent = k;
        o.stream_offset = e->extents[k].stream_offset;
        o.length = e->extents[k].length;
        o.file_offset = fo;
        fo += e->extents[k].length;
        if (o.stream_offset < gen_begin || o.stream_offset >= gen_end)
          continue;
        for (p = o.stream_offset; p < end;) {
          rp_probe pr;
          pr.r = r;
          pr.slice = xpar_slice_of(&r->geom, p);
          pr.col   = xpar_cell_of(&r->geom, p);
          pr.buf   = cell;
          if (!rp_occ_intact(&o, &pr))
            *rp_cell_key(r, pr.slice, pr.col) = 1;
          p = xpar_cell_begin(&r->geom, pr.slice, pr.col) +
              xpar_cell_size(&r->geom, pr.col);
        }
      }
      /*  Write construction determines whether this is repairable.  */
      r->hash_bad[i] = 1;
    }
  }
  xpar_free(buf);  xpar_free(cell);
}

/*  Find object-kind and link-target mismatches.  */
static void rp_scan_structure(rp * r) {
  u32 i;
  for (i = 0; i < r->scan_entry_count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    const char * want = NULL;
    xpar_stat_t st;
    bool bad = false;
    if (xpar_lstat(r->path[i], &st) != 0) continue;
    switch (e->entry_type) {
      case XPAR_ENTRY_DIR:
        want = "a directory";  bad = !st.is_dir;  break;
      case XPAR_ENTRY_SYMLINK: {
        u32 n = 0;
        char * tgt = xpar_read_symlink(r->path[i], &n);
        want = "a symbolic link";
        bad = !tgt || n != e->extra_len ||
              xpar_memcmp(tgt, e->extra, e->extra_len) != 0;
        xpar_free(tgt);
        break;
      }
      default:
        want = "a regular file";  bad = !st.is_regular;  break;
    }
    if (!bad) continue;
    r->structure_bad++;
    /*  --to restores structure from the manifest.  */
    if (r->o->dest != XPAR_DEST_TO)
      rp_note(r, "xpar: %.*s: recorded as %s; the object on disk differs and "
              "recovery cannot localise it\n",
              (int) e->name_len, e->name, want);
  }
}

static rp_cell * rp_add_cell(rp * r, u64 slice, u32 col) {
  rp_cell * c;
  if (r->cell_count == r->cell_cap) {
    r->cell_cap = r->cell_cap ? r->cell_cap * 2 : 32;
    r->cell = (rp_cell *) xpar_realloc(r->cell,
                                       r->cell_cap * sizeof(rp_cell));
  }
  c = &r->cell[r->cell_count++];
  xpar_memset(c, 0, sizeof *c);
  c->slice = slice;  c->col = col;
  c->begin = xpar_cell_begin(&r->geom, slice, col);
  c->size  = xpar_cell_size(&r->geom, col);
  return c;
}

/*  Any intact occurrence solves the cell without consuming recovery.  */
static void rp_classify(rp * r) {
  u64 s;
  u32 c, k = r->geom.cells_per_slice;
  u8 * buf = (u8 *) xpar_alloc_raw((sz) (r->geom.cell_bytes ?
                                         r->geom.cell_bytes :
                                         r->geom.slice_size));
  for (s = 0; s < r->geom.slice_count; s++)
    for (c = 0; c < k; c++) {
      rp_cell * cell;
      u64 begin, size, l_end, p, real;
      bool all = true;
      if (!*rp_cell_key(r, s, c)) continue;
      cell  = rp_add_cell(r, s, c);
      begin = cell->begin;  size = cell->size;
      l_end = r->geom.stream_base + r->geom.stream_length;
      real  = begin >= l_end ? 0 : MIN(size, l_end - begin);
      /*  Choose sources per run; two entries may each hold part intact.  */
      for (p = begin; p < begin + real && all;) {
        xpar_occurrence o, src;
        rp_probe pr;
        u64 run;
        if (!xpar_occindex_canonical(&r->ox, p, &o, &run)) {
          all = false;  break;
        }
        if (run > begin + real - p) run = begin + real - p;
        pr.r = r;  pr.slice = s;  pr.col = c;  pr.buf = buf;
        if (!xpar_occindex_repair_source(&r->ox, p, run, rp_occ_intact,
                                         &pr, &src)) all = false;
        p += run;
      }
      if (all) {
        cell->decode = 0;
        if (xpar_cell_bad(&r->er, s, c)) {
          r->er.bad[s * k + c] = 0;  r->er.bad_count--;
        }
      } else {
        cell->decode = 1;
        xpar_cell_mark(&r->er, s, c);
      }
    }
  xpar_free(buf);
}

static bool rp_cell_verify(rp * r, rp_cell * c) {
  u32 crc = xpar_crc32c(0, c->bytes, (sz) c->size);
  if ((r->tag_have & XPAR_TAGS_CELL) && r->tags.t.cell_crc)
    return crc == r->tags.t.cell_crc[c->slice * r->geom.cells_per_slice +
                                     c->col];
  /*  Slice CRCs verify cells only when a slice contains one cell.  */
  if (r->geom.cells_per_slice > 1) return true;
  if (!(r->tag_have & XPAR_TAGS_CRC) || !r->tags.t.slice_crc) return true;
  return crc == r->tags.t.slice_crc[c->slice];
}

/*  Recheck copy sources so a source that rotted after planning is never
    written over merely damaged evidence.  */
static void rp_solve_copies(rp * r) {
  u32 i;
  u8 * buf = (u8 *) xpar_alloc_raw((sz) (r->geom.cell_bytes ?
                                         r->geom.cell_bytes :
                                         r->geom.slice_size));
  for (i = 0; i < r->cell_count; i++) {
    rp_cell * c = &r->cell[i];
    u64 l_end = r->geom.stream_base + r->geom.stream_length;
    u64 real  = c->begin >= l_end ? 0 : MIN(c->size, l_end - c->begin);
    u64 p;
    bool ok = true;
    if (c->decode) continue;
    c->bytes = (u8 *) xpar_calloc((sz) c->size, 1);
    for (p = c->begin; p < c->begin + real && ok;) {
      xpar_occurrence o, src;
      rp_probe pr;
      u64 run;
      if (!xpar_occindex_canonical(&r->ox, p, &o, &run)) { ok = false; break; }
      if (run > c->begin + real - p) run = c->begin + real - p;
      pr.r = r;  pr.slice = c->slice;  pr.col = c->col;  pr.buf = buf;
      if (!xpar_occindex_repair_source(&r->ox, p, run, rp_occ_intact, &pr,
                                       &src)) { ok = false;  break; }
      if (!rp_read_entry_resynced(
            r, src.entry, src.file_offset + (p - src.stream_offset), run,
            c->bytes + (p - c->begin))) ok = false;
      p += run;
    }
    if (ok && rp_cell_verify(r, c)) { c->solved = 1;  r->cells_copied++; }
    else {
      /*  A changed source falls back to outer decoding.  */
      xpar_free(c->bytes);  c->bytes = NULL;
      c->decode = 1;
      xpar_cell_mark(&r->er, c->slice, c->col);
    }
  }
  xpar_free(buf);
}

/*  Reuse one decode plan for every column with the same erasure pattern.  */
static bool rp_solve_decode(rp * r, u32 chunk, bool partial) {
  xpar_col_groups g;
  xpar_codec * cd;
  u64 s = r->geom.slice_count;
  u8 ** dptr;  u8 ** rptr;  u8 * pool;
  bool ok = true;
  u32 gi;
  if (!r->er.bad_count) return true;
  if (!xpar_codec_supports_axis(r->sd.codec, r->sd.field_log2, s,
                                r->rec_total,
                                r->sd.recovery_axis_log2))
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "The recorded codec cannot express S = %" PRIu64 " with R = %" PRIu64 " "
               "in GF(2^%" PRIu8 ").", s,
               r->rec_total, r->sd.field_log2);
  cd = xpar_codec_new_axis(r->sd.codec, r->sd.field_log2, s, r->rec_total,
                           r->sd.recovery_axis_log2);
  /* Guard the size_t allocation on 32-bit hosts. */
  FATAL_UNLESS("Repair column allocation is too large for this host.",
               !chunk || s + r->rec_total <= (u64) (sz) -1 / chunk);
  pool = (u8 *) xpar_alloc_raw((sz) ((s + r->rec_total) * chunk));
  dptr = (u8 **) xpar_alloc_raw((sz) s * sizeof(u8 *));
  rptr = (u8 **) xpar_alloc_raw((sz) (r->rec_total ? r->rec_total : 1) *
                                sizeof(u8 *));
  { u64 i;
    for (i = 0; i < s; i++) dptr[i] = pool + i * chunk;
    for (i = 0; i < r->rec_total; i++) rptr[i] = pool + (s + i) * chunk;
  }
  xpar_col_groups_build(&r->er, &g);
  for (gi = 0; gi < g.group_count && ok; gi++) {
    xpar_col_group * grp = &g.group[gi];
    xpar_codec_status st;
    xpar_codec_plan * pl;
    u32 ci;
    if (!grp->erased) continue;
    pl = xpar_codec_plan_new(cd, grp->present, r->rec_present, &st);
    /*  A column past the recovery leaves its cells unsolved; --to still
        writes every entry that does not depend on them.  */
    if (!pl) { if (partial) continue;  ok = false;  break; }
    for (ci = 0; ci < grp->column_count && ok; ci++) {
      u32 col = grp->column[ci];
      u64 width = xpar_cell_size(&r->geom, col);
      u64 base  = (u64) col * (r->geom.cell_bytes ? r->geom.cell_bytes
                                                  : r->geom.slice_size);
      u64 at;
      for (at = 0; at < width; at += chunk) {
        u64 n = MIN((u64) chunk, width - at), i;
        for (i = 0; i < s; i++) {
          if (!grp->present[i]) { xpar_memset(dptr[i], 0, (sz) n);  continue; }
          rp_read_stream(r, xpar_slice_begin(&r->geom, i) + base + at, n,
                         dptr[i]);
        }
        for (i = 0; i < r->rec_total; i++) {
          if (!r->rec_present[i]) { xpar_memset(rptr[i], 0, (sz) n); continue; }
          xpar_memcpy(rptr[i], r->rec[i] + base + at, (sz) n);
        }
        if (xpar_codec_plan_apply(pl, dptr, (const u8 * const *) rptr,
                                  (sz) n) != XPAR_CODEC_OK) {
          ok = false;  break;
        }
        for (i = 0; i < s; i++) {
          rp_cell * c;
          if (grp->present[i]) continue;
          c = rp_cell_at(r, i, col);
          if (!c) continue;
          if (!c->bytes) c->bytes = (u8 *) xpar_calloc((sz) c->size, 1);
          xpar_memcpy(c->bytes + at, dptr[i], (sz) n);
        }
      }
    }
    xpar_codec_plan_free(pl);
  }
  xpar_col_groups_free(&g);
  xpar_free(pool);  xpar_free(dptr);  xpar_free(rptr);
  xpar_codec_free(cd);
  if (ok) {
    u32 i;
    for (i = 0; i < r->cell_count; i++) {
      rp_cell * c = &r->cell[i];
      if (!c->decode || !c->bytes || c->solved) continue;
      if (rp_cell_verify(r, c)) { c->solved = 1;  r->cells_decoded++; }
      else ok = false;
    }
  }
  return ok;
}

/*  A slice whose strong tag still fails holds damage no cell checksum can
    see, so every cell of it becomes an erasure and the outer code rebuilds
    the whole slice. The work set stays ordered by (slice, column).  */
static bool rp_widen_slice(rp * r, u64 slice) {
  u32 k = r->geom.cells_per_slice, c, i, lo, hi, n;
  rp_cell * out;
  bool changed = false;
  lo = 0;
  while (lo < r->cell_count && r->cell[lo].slice < slice) lo++;
  hi = lo;
  while (hi < r->cell_count && r->cell[hi].slice == slice) hi++;
  n = r->cell_count - (hi - lo) + k;
  out = (rp_cell *) xpar_alloc_raw((sz) n * sizeof(rp_cell));
  xpar_memcpy(out, r->cell, (sz) lo * sizeof(rp_cell));
  for (c = 0; c < k; c++) {
    rp_cell * d = &out[lo + c];
    const rp_cell * src = NULL;
    for (i = lo; i < hi; i++)
      if (r->cell[i].col == c) { src = &r->cell[i];  break; }
    if (src) *d = *src;
    else {
      xpar_memset(d, 0, sizeof *d);
      d->slice = slice;  d->col = c;
      d->begin = xpar_cell_begin(&r->geom, slice, c);
      d->size  = xpar_cell_size(&r->geom, c);
    }
    if (!src || !d->decode) changed = true;
    if (d->solved && !d->decode && r->cells_copied) r->cells_copied--;
    if (d->solved && d->decode && r->cells_decoded) r->cells_decoded--;
    xpar_free(d->bytes);  d->bytes = NULL;
    d->decode = 1;  d->solved = 0;
    *rp_cell_key(r, slice, c) = 1;
    xpar_cell_mark(&r->er, slice, c);
  }
  xpar_memcpy(out + lo + k, r->cell + hi,
              (sz) (r->cell_count - hi) * sizeof(rp_cell));
  xpar_free(r->cell);
  r->cell = out;  r->cell_count = n;  r->cell_cap = n;
  return changed;
}

/*  Verify repaired slices when strong tags exist; -f gates their absence.
    With `widened`, a failing slice is erased whole and retried through the
    decoder instead of being called unrepairable.  */
static bool rp_slice_gate(rp * r, bool * widened) {
  u32 i;
  u64 z = r->geom.slice_size;
  u8 * buf;
  bool ok = true;
  if (!(r->tag_have & XPAR_TAGS_TAG) || !r->tags.t.slice_tag) return true;
  buf = (u8 *) xpar_alloc_raw((sz) z);
  for (i = 0; i < r->cell_count && ok; i++) {
    u64 s = r->cell[i].slice;
    u32 c;
    u8 tag[16];
    bool done = true;
    if (i && r->cell[i - 1].slice == s) continue;
    for (c = 0; c < r->geom.cells_per_slice; c++) {
      rp_cell * cc = rp_cell_at(r, s, c);
      if (cc && !cc->solved) done = false;
    }
    if (!done) continue;
    rp_read_stream(r, xpar_slice_begin(&r->geom, s), z, buf);
    rp_tag(r, s, buf, tag, r->tags.t.tag_len);
    if (xpar_blake3_tag_equal(tag, r->tags.t.slice_tag +
                              s * r->tags.t.tag_len, r->tags.t.tag_len))
      continue;
    if (widened && rp_widen_slice(r, s)) *widened = true;
    else ok = false;
  }
  xpar_free(buf);
  return ok;
}

static void rp_add_edit(rp * r, u32 entry, u64 off, u64 len, u32 cell,
                        u64 cell_off) {
  rp_edit * e;
  if (!len) return;
  if (r->edit_count == r->edit_cap) {
    r->edit_cap = r->edit_cap ? r->edit_cap * 2 : 32;
    r->edit = (rp_edit *) xpar_realloc(r->edit,
                                       r->edit_cap * sizeof(rp_edit));
  }
  e = &r->edit[r->edit_count++];
  e->entry = entry;  e->off = off;  e->len = len;
  e->cell = cell;  e->cell_off = cell_off;
}

static int rp_edit_cmp(const rp_edit * a, const rp_edit * b) {
  if (a->entry != b->entry) return a->entry < b->entry ? -1 : 1;
  if (a->off != b->off) return a->off < b->off ? -1 : 1;
  return 0;
}

/*  Stable heap sort by (entry, file offset).  */

static int rp_edit_key(const rp * r, u32 a, u32 b) {
  int c = rp_edit_cmp(&r->edit[a], &r->edit[b]);
  if (c) return c;
  return a < b ? -1 : (a > b ? 1 : 0);
}

static void rp_edit_sift(const rp * r, u32 * a, u32 root, u32 n) {
  while (1) {
    u32 c = 2 * root + 1, big;
    if (c >= n) return;
    big = c;
    if (c + 1 < n && rp_edit_key(r, a[c], a[c + 1]) < 0) big = c + 1;
    if (rp_edit_key(r, a[root], a[big]) >= 0) return;
    { u32 t = a[root];  a[root] = a[big];  a[big] = t; }
    root = big;
  }
}

static void rp_sort_edits(rp * r) {
  u32 n = r->edit_count, i, * ord;
  rp_edit * out;
  if (n < 2) return;
  ord = (u32 *) xpar_alloc_raw((sz) n * sizeof(u32));
  For(u32, q, n, ord[q] = q)
  for (i = n / 2; i-- > 0;) rp_edit_sift(r, ord, i, n);
  for (i = n; i-- > 1;) {
    u32 t = ord[0];  ord[0] = ord[i];  ord[i] = t;
    rp_edit_sift(r, ord, 0, i);
  }
  out = (rp_edit *) xpar_alloc_raw((sz) n * sizeof(rp_edit));
  For(u32, q, n, out[q] = r->edit[ord[q]])
  xpar_free(ord);  xpar_free(r->edit);
  r->edit = out;  r->edit_cap = n;
}

/*  Whether two names already hold the same bytes, so linking one over the
    other loses nothing.  */
static bool rp_same_content(rp * r, u32 a, u32 b) {
  u64 len = r->mf.entry[a].length, off = 0;
  u32 z = (u32) MIN((u64) (r->geom.slice_size ? r->geom.slice_size : 4096),
                    (u64) (1u << 20));
  u8 * pa, * pb;
  bool same = true;
  if (r->mf.entry[b].length != len) return false;
  pa = (u8 *) xpar_alloc_raw(z);  pb = (u8 *) xpar_alloc_raw(z);
  while (off < len && same) {
    u64 take = MIN((u64) z, len - off);
    if (!rp_read_entry_raw(r, a, off, take, pa) ||
        !rp_read_entry_raw(r, b, off, take, pb) ||
        xpar_memcmp(pa, pb, (sz) take)) same = false;
    off += take;
  }
  xpar_free(pa);  xpar_free(pb);
  return same;
}

/*  Emit one write per damaged inode, not per hard-link alias.  */

typedef struct {
  rp *  r;
  u32   cell;
  u64   real;
  u8 *  buf;
} rp_wctx;

static void rp_write_occ(const xpar_occurrence * o, void * user) {
  rp_wctx * w = (rp_wctx *) user;
  rp * r = w->r;
  rp_cell * c = &r->cell[w->cell];
  u64 lo = MAX(c->begin, o->stream_offset);
  u64 hi = MIN(c->begin + w->real, o->stream_offset + o->length);
  if (lo >= hi) return;
  if (r->alias[o->entry] ||
      r->mf.entry[o->entry].entry_type != XPAR_ENTRY_REGULAR) return;
  /*  The solved bytes have passed the cell tag and, where one exists, the
      slice tag, so only they decide whether this occurrence must change; a
      CRC probe would call a CRC-preserving forgery intact.  */
  { u64 n = hi - lo;
    if (rp_read_entry_raw(r, o->entry,
                          o->file_offset + (lo - o->stream_offset), n,
                          w->buf) &&
        !xpar_memcmp(w->buf, c->bytes + (lo - c->begin), (sz) n))
      return;
  }
  rp_add_edit(r, o->entry, o->file_offset + (lo - o->stream_offset),
              hi - lo, w->cell, lo - c->begin);
}

static void rp_build_writes(rp * r) {
  u32 i;
  u8 * buf = (u8 *) xpar_alloc_raw((sz) (r->geom.cell_bytes ?
                                         r->geom.cell_bytes :
                                         r->geom.slice_size));
  for (i = 0; i < r->cell_count; i++) {
    rp_cell * c = &r->cell[i];
    u64 l_end = r->geom.stream_base + r->geom.stream_length;
    rp_wctx w;
    w.r = r;  w.cell = i;  w.buf = buf;
    w.real = c->begin >= l_end ? 0 : MIN(c->size, l_end - c->begin);
    if (!c->solved || !w.real) continue;
    xpar_occindex_overlaps(&r->ox, c->begin, w.real, rp_write_occ, &w);
  }
  xpar_free(buf);
  rp_sort_edits(r);
  /*  Coalesce contiguous edits into one pwrite.  */
  for (i = 0; i < r->edit_count;) {
    u32 j = i;
    u64 end = r->edit[i].off + r->edit[i].len, k;
    rp_write * w;
    while (j + 1 < r->edit_count && r->edit[j + 1].entry == r->edit[i].entry &&
           r->edit[j + 1].off <= end) {
      j++;
      if (r->edit[j].off + r->edit[j].len > end)
        end = r->edit[j].off + r->edit[j].len;
    }
    if (r->wr_count == r->wr_cap) {
      r->wr_cap = r->wr_cap ? r->wr_cap * 2 : 16;
      r->wr = (rp_write *) xpar_realloc(r->wr,
                                        r->wr_cap * sizeof(rp_write));
    }
    w = &r->wr[r->wr_count++];
    xpar_memset(w, 0, sizeof *w);
    w->entry = r->edit[i].entry;  w->off = r->edit[i].off;
    w->len = end - w->off;
    w->data = (u8 *) xpar_calloc((sz) w->len, 1);
    for (k = i; k <= j; k++) {
      const rp_edit * e = &r->edit[k];
      xpar_memcpy(w->data + (e->off - w->off),
                  r->cell[e->cell].bytes + e->cell_off, (sz) e->len);
    }
    i = j + 1;
  }
  /*  Recreate missing hard-link names, and relink a copy that replaced
      one, from their canonical entries.  */
  for (i = 0; i < r->mf.count; i++) {
    rp_write * w;
    xpar_stat_t st, ct;
    u32 t = r->canon[i];
    bool relink = false;
    if (r->mf.entry[i].entry_type != XPAR_ENTRY_HARDLINK || t == i) continue;
    if (xpar_lstat(r->path[t], &ct) != 0 || !ct.is_regular) continue;
    if (xpar_lstat(r->path[i], &st) == 0) {
      if (!st.is_regular || !(st.dev | st.ino) || !(ct.dev | ct.ino)) continue;
      if (st.dev == ct.dev && st.ino == ct.ino) continue;
      /*  Only an identical copy may be replaced by the link.  */
      if (!rp_same_content(r, i, t)) {
        r->structure_bad++;
        rp_note(r, "xpar: %.*s differs from hard-link target '%.*s'; copy "
                "left unchanged\n",
                (int) r->mf.entry[i].name_len, r->mf.entry[i].name,
                (int) r->mf.entry[t].name_len, r->mf.entry[t].name);
        continue;
      }
      relink = true;
      r->fstate[i] |= 1;  r->fsize[i] = st.size;
    }
    if (r->wr_count == r->wr_cap) {
      r->wr_cap = r->wr_cap ? r->wr_cap * 2 : 16;
      r->wr = (rp_write *) xpar_realloc(r->wr,
                                        r->wr_cap * sizeof(rp_write));
    }
    w = &r->wr[r->wr_count++];
    xpar_memset(w, 0, sizeof *w);
    /*  Record the absent name; rp_read_old allocates its empty payload.  */
    w->entry = i;  w->link = 1;  w->relink = relink ? 1 : 0;
    /*  Repair the copy if relinking fails.  */
    if (relink) {
      u32 n = r->wr_count, k;
      for (k = 0; k < n; k++) {
        rp_write d = r->wr[k];
        if (d.entry != t || d.link || d.shadow) continue;
        d.entry = i;  d.shadow = 1;  d.old = NULL;
        if (d.data) {
          d.data = (u8 *) xpar_alloc_raw((sz) d.len);
          xpar_memcpy(d.data, r->wr[k].data, (sz) d.len);
        }
        if (r->wr_count == r->wr_cap) {
          r->wr_cap *= 2;
          r->wr = (rp_write *) xpar_realloc(r->wr,
                                            r->wr_cap * sizeof(rp_write));
        }
        r->wr[r->wr_count++] = d;
      }
    }
  }

  /*  A hash failure with no corresponding write is unrepairable.  */
  for (i = 0; i < r->mf.count; i++) {
    u32 w;
    bool written = false;
    if (!r->hash_bad[i] || r->alias[i]) continue;
    for (w = 0; w < r->wr_count && !written; w++)
      if (r->wr[w].entry == i) written = true;
    if (written) continue;
    r->opaque++;
    rp_note(r, "xpar: %.*s: content damage cannot be located\n",
            (int) r->mf.entry[i].name_len, r->mf.entry[i].name);
  }

  /*  Journal an overlong tail before truncating it.  */
  for (i = 0; i < r->mf.count; i++) {
    rp_write * w;
    if (!(r->fstate[i] & 2) || r->alias[i]) continue;
    if (r->wr_count == r->wr_cap) {
      r->wr_cap = r->wr_cap ? r->wr_cap * 2 : 16;
      r->wr = (rp_write *) xpar_realloc(r->wr,
                                        r->wr_cap * sizeof(rp_write));
    }
    w = &r->wr[r->wr_count++];
    xpar_memset(w, 0, sizeof *w);
    w->entry = i;  w->off = r->mf.entry[i].length;
    w->len   = r->fsize[i] - r->mf.entry[i].length;
    w->trunc = 1;
  }
}

static void rp_journal(rp * r) {
  xpar_file * f;
  u8 hdr[XPAR_UNDO_HDR], rec[XPAR_UNDO_REC], foot[XPAR_UNDO_FOOT], pad[8];
  u32 i, all = 0;
  u64 payload = 0;
  xpar_memset(pad, 0, sizeof pad);
  for (i = 0; i < r->wr_count; i++) payload += r->wr[i].len;
  xpar_memset(hdr, 0, sizeof hdr);
  xpar_memcpy(hdr, XPAR_UNDO_MAGIC, 8);
  xpar_wr32(hdr + 8, XPAR_UNDO_VER);
  xpar_memcpy(hdr + 16, r->set_id, XPAR_SET_ID_LEN);
  xpar_wr64(hdr + 32, r->wr_count);
  xpar_wr64(hdr + 40, payload);
  xpar_wr64(hdr + 48, (u64) xpar_wall_ns());
  xpar_wr32(hdr + 60, xpar_crc32c(0, hdr, 60));
  /*  Create the journal privately without following links.  */
  f = xpar_open(r->journal, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_EXCL |
                            XPAR_O_NOFOLLOW | XPAR_O_PRIVATE);
  if (!f) FATAL_IO("Cannot write journal '%s': %s; use --no-journal to "
                   "continue.", r->journal,
                   xpar_strerror(xpar_errno()));
  if (xpar_set_mode(r->journal, 1, 0600) != 0)
    xpar_fprintf(xpar_stderr, "xpar: warning: cannot restrict journal '%s' "
                 "to its owner: %s\n", r->journal,
                 xpar_strerror(xpar_errno()));
  xpar_xwrite(f, hdr, sizeof hdr);
  all = xpar_crc32c(0, hdr, sizeof hdr);
  for (i = 0; i < r->wr_count; i++) {
    rp_write * w = &r->wr[i];
    const char * path = r->path[w->entry];
    u32 plen = (u32) xpar_strlen(path);
    u32 tail = (u32) ((8 - ((XPAR_UNDO_REC + plen + w->len) & 7)) & 7);
    xpar_memset(rec, 0, sizeof rec);
    xpar_wr32(rec, plen);
    xpar_wr32(rec + 4, (r->fstate[w->entry] & 1) ? 0 : XPAR_UNDO_CREATED);
    xpar_wr64(rec + 8, w->off);
    xpar_wr64(rec + 16, w->len);
    xpar_wr64(rec + 24, r->fsize[w->entry]);
    xpar_wr32(rec + 32, xpar_crc32c(0, w->old, (sz) w->len));
    xpar_wr32(rec + 36, xpar_crc32c(0, rec, 36));
    xpar_xwrite(f, rec, sizeof rec);
    xpar_xwrite(f, path, plen);
    xpar_xwrite(f, w->old, (sz) w->len);
    if (tail) xpar_xwrite(f, pad, tail);
    all = xpar_crc32c(all, rec, sizeof rec);
    all = xpar_crc32c(all, path, plen);
    all = xpar_crc32c(all, w->old, (sz) w->len);
    if (tail) all = xpar_crc32c(all, pad, tail);
  }
  /*  The footer CRC makes incomplete journals non-replayable.  */
  xpar_memset(foot, 0, sizeof foot);
  xpar_memcpy(foot, XPAR_UNDO_END, 8);
  xpar_wr64(foot + 8, r->wr_count);
  xpar_wr32(foot + 16, all);
  xpar_xwrite(f, foot, sizeof foot);
  if (xpar_fsync(f) != 0) FATAL_IO("Cannot flush the undo journal.");
  xpar_xclose(f);
  /*  Make the journal directory entry durable before data writes.  */
  if (xpar_fsync_dir(r->journal) != 0) {
    int err = xpar_errno();
    xpar_remove(r->journal);
    FATAL_IO("Cannot persist journal '%s': %s; use --no-journal to continue.",
             r->journal, xpar_strerror(err));
  }
  if (r->verbose)
    rp_note(r, "xpar: journalled %" PRIu32 " ranges (%" PRIu64
            " bytes) to '%s'.\n",
            r->wr_count,
            payload, r->journal);
}

/*  Refuse journals that would replace unread existing bytes with zeroes.  */
static void rp_read_old(rp * r) {
  u32 i;
  for (i = 0; i < r->wr_count; i++) {
    rp_write * w = &r->wr[i];
    u64 have;
    w->old = (u8 *) xpar_calloc((sz) w->len, 1);
    if (rp_read_entry_raw(r, w->entry, w->off, w->len, w->old)) continue;
    if (r->o->no_journal || !(r->fstate[w->entry] & 1)) continue;
    have = r->fsize[w->entry] > w->off
             ? MIN(w->len, r->fsize[w->entry] - w->off) : 0;
    if (have && !rp_read_entry_raw(r, w->entry, w->off, have, w->old))
      FATAL_IO("Cannot journal '%.*s' at offset %" PRIu64
               ": read failed; use --no-journal to continue.",
               (int) r->mf.entry[w->entry].name_len,
               r->mf.entry[w->entry].name, w->off);
  }
}

static void rp_basic_meta(rp * r, u32 idx, const char * path, bool link);
static bool rp_foreign(const rp * r, u32 i);

/*  Whether a missing manifest entry can be recreated without recovery.  */
static bool rp_recreatable(const rp * r, u32 i) {
  const xpar_entry * e = &r->mf.entry[i];
  xpar_stat_t st;
  if (xpar_lstat(r->path[i], &st) == 0) return false;
  if (!xpar_errno_absent(xpar_errno())) return false;
  if (e->entry_type == XPAR_ENTRY_HARDLINK) return false;
  if (e->entry_type == XPAR_ENTRY_REGULAR &&
      (e->length || rp_foreign(r, i))) return false;
  return true;
}

/*  Count recreatable names without writing.  */
static u64 rp_missing_names(const rp * r) {
  u64 n = 0;
  u32 i;
  for (i = 0; i < r->scan_entry_count; i++) if (rp_recreatable(r, i)) n++;
  return n;
}

static void rp_restore_missing(rp * r) {
  u32 i;
  for (i = 0; i < r->scan_entry_count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    char * dir;
    int made, err = 0;
    if (!rp_recreatable(r, i)) continue;
    dir = xpar_path_dir(r->path[i]);
    xpar_mkdir_p(dir, 0777);
    xpar_free(dir);
    if (e->entry_type == XPAR_ENTRY_DIR)
      made = xpar_mkdir_p(r->path[i], 0777);
    else if (e->entry_type == XPAR_ENTRY_SYMLINK) {
      char * tgt = xpar_strndup((const char *) e->extra, e->extra_len);
      made = xpar_symlink(tgt, r->path[i]);
      if (made != 0) err = xpar_errno();
      xpar_free(tgt);
    } else {
      xpar_file * f = xpar_open(r->path[i], XPAR_O_WRONLY | XPAR_O_CREAT |
                                            XPAR_O_EXCL | XPAR_O_NOFOLLOW);
      made = f ? 0 : -1;
      if (f) xpar_xclose(f);
    }
    if (made != 0) {
      rp_note(r, "xpar: cannot recreate '%s': %s\n", r->path[i],
              xpar_strerror(err ? err : xpar_errno()));
      r->names_failed++;
      continue;
    }
    rp_basic_meta(r, i, r->path[i],
                  e->entry_type == XPAR_ENTRY_SYMLINK);
    r->writes++;  r->names_made++;
  }
}

static char * rp_link_stage(const char * path) {
  xpar_stat_t st;
  char * out = NULL;
  u32 n;
  for (n = 1; ; n++) {
    xpar_free(out);
    xpar_asprintf(&out, "%s.xpar-link-%" PRIu32, path, n);
    if (xpar_lstat(out, &st) != 0) return out;
    FATAL_UNLESS("Too many link stages beside '%s'.", n != 1000, path);
  }
}

/*  Stage a hard link beside PATH.  */
static char * rp_link_aside(const char * canon, const char * path,
                            int * err) {
  u32 n;
  *err = 0;
  for (n = 0; n < 64; n++) {
    xpar_stat_t st;
    char * stage = rp_link_stage(path);
    if (xpar_link(canon, stage) == 0) return stage;
    *err = xpar_errno();
    if (xpar_lstat(stage, &st) != 0) { xpar_free(stage);  return NULL; }
    xpar_free(stage);
  }
  return NULL;
}

/*  Stage a symbolic link beside PATH.  */
static char * rp_symlink_aside(const char * target, const char * path,
                               int * err) {
  u32 n;
  *err = 0;
  for (n = 0; n < 64; n++) {
    xpar_stat_t st;
    char * stage = rp_link_stage(path);
    if (xpar_symlink(target, stage) == 0) return stage;
    *err = xpar_errno();
    if (xpar_lstat(stage, &st) != 0) { xpar_free(stage);  return NULL; }
    xpar_free(stage);
  }
  return NULL;
}

/*  Create or atomically replace an alias with a hard link.  */
static int rp_relink(rp * r, const rp_write * w) {
  const char * path = r->path[w->entry];
  const char * canon = r->path[r->canon[w->entry]];
  char * d = xpar_path_dir(path);
  int ok, err = 0;
  xpar_mkdir_p(d, 0777);
  xpar_free(d);
  if (!w->relink) {
    ok = xpar_link(canon, path);
    if (ok != 0) err = xpar_errno();
  } else {
    char * stage = rp_link_aside(canon, path, &err);
    ok = stage ? 0 : -1;
    if (stage) {
      rp_close_entry(r, w->entry);
      ok = xpar_rename(stage, path);
      if (ok != 0) { err = xpar_errno();  xpar_remove(stage); }
      xpar_free(stage);
    }
  }
  if (ok != 0)
    rp_note(r, "xpar: cannot link '%s' to '%s': %s\n", path, canon,
            xpar_strerror(err));
  return ok;
}

static void rp_apply(rp * r) {
  u32 i;
  bool warned = false;
  rp_restore_missing(r);
  for (i = 0; i < r->wr_count;) {
    u32 entry = r->wr[i].entry, j;
    xpar_file * f;
    bool made = false;
    if (r->io_bad && r->io_bad[entry]) {
      for (j = i; j < r->wr_count && r->wr[j].entry == entry; j++) {}
      rp_note(r, "xpar: %s: not rewritten; its bytes could not be read\n",
              r->path[entry]);
      i = j;
      continue;
    }
    if (r->wr[i].link) {
      if (rp_relink(r, &r->wr[i]) == 0) { r->writes++;  r->links_made++; }
      else if (r->wr[i].relink) {
        if (!r->link_failed)
          r->link_failed = (u8 *) xpar_calloc(r->mf.count ? r->mf.count : 1, 1);
        r->link_failed[entry] = 1;
      }
      i++;
      continue;
    }
    if (r->wr[i].shadow && !(r->link_failed && r->link_failed[entry])) {
      for (j = i; j < r->wr_count && r->wr[j].entry == entry; j++) {}
      i = j;
      continue;
    }
    f = xpar_open(r->path[entry], XPAR_O_RDWR | XPAR_O_NOFOLLOW);
    if (!f) {
      /*  Recreate missing parent directories.  */
      char * d = xpar_path_dir(r->path[entry]);
      xpar_mkdir_p(d, 0777);
      xpar_free(d);
      f = xpar_open(r->path[entry], XPAR_O_RDWR | XPAR_O_CREAT |
                                         XPAR_O_NOFOLLOW);
      if (!f) {
        int err = xpar_errno();
        /*  Remove an unused journal.  */
        if (!r->writes && !r->o->no_journal && !r->o->keep_journal) {
          xpar_remove(r->journal);
          xpar_fsync_dir(r->journal);
        }
        FATAL_IO("Cannot open '%s' for repair: %s.", r->path[entry],
                 xpar_strerror(err));
      }
      made = true;
    }
    if (!xpar_lock_supported() && !warned) {
      warned = true;
      rp_note(r, "xpar: this host has no file locking, so a concurrent "
                 "writer would race this repair.\n");
    } else if (xpar_lock_supported() && xpar_lock(f, true) != 0)
      FATAL_IO("'%s' is locked by another process; repair would race it.",
               r->path[entry]);
    for (j = i; j < r->wr_count && r->wr[j].entry == entry; j++) {
      rp_write * w = &r->wr[j];
      (void) made;
      if (w->trunc) {
        if (xpar_ftruncate(f, w->off) != 0)
          FATAL_IO("Cannot remove the %" PRIu64
                   " bytes past the end of '%s': %s.",
                   w->len, r->path[entry],
                   xpar_strerror(xpar_errno()));
        r->writes++;
        continue;
      }
      if (xpar_pwrite(f, w->data, (sz) w->len, w->off) != (sz) w->len)
        FATAL_IO("Write failed at offset %" PRIu64 " in '%s': %s (journal: "
                 "'%s').", w->off, r->path[entry],
                 xpar_strerror(xpar_errno()),
                 r->journal);
      r->bytes_written += w->len;  r->writes++;
    }
    if (xpar_fsync(f) != 0)
      FATAL_IO("Cannot flush '%s' after repair.", r->path[entry]);
    if (xpar_lock_supported()) xpar_unlock(f);
    xpar_xclose(f);
    /*  Restore metadata lost to the umask.  */
    if (made) rp_basic_meta(r, entry, r->path[entry], false);
    i = j;
  }
  rp_close_files(r);
}

/*  Compare alias and canonical inodes where available.  */
static bool rp_linked(const rp * r, u32 i) {
  xpar_stat_t lst, cst;
  if (xpar_lstat(r->path[i], &lst) != 0) return false;
  if (xpar_lstat(r->path[r->canon[i]], &cst) != 0) return false;
  if (!(lst.dev | lst.ino) || !(cst.dev | cst.ino)) return true;
  return lst.dev == cst.dev && lst.ino == cst.ino;
}

/*  Hash file `file` over `e`'s extents against its certificate.  */
static bool rp_hash_ok(rp * r, u32 file, const xpar_entry * e, u8 * buf) {
  xpar_blake3_t h;
  u8 got[32];
  u64 fo = 0, z = r->geom.slice_size;
  u32 k;
  if (r->auth_only) xpar_blake3_init_keyed(&h, r->key.k_file);
  else              xpar_blake3_init(&h);
  for (k = 0; k < e->extent_count; k++) {
    u64 left = e->extents[k].length, at = fo;
    while (left) {
      u64 take = MIN(left, z);
      rp_read_entry_raw(r, file, at, take, buf);
      xpar_blake3_update(&h, buf, (sz) take);
      at += take;  left -= take;
    }
    fo += e->extents[k].length;
  }
  xpar_blake3_final(&h, got, 32);
  return xpar_memcmp(got, e->content_hash, 32) == 0;
}

/*  Re-read every changed entry and alias under its 256-bit certificate.  */
static bool rp_reverify(rp * r) {
  u32 i;
  u8 * seen = (u8 *) xpar_calloc(r->mf.count ? r->mf.count : 1, 1);
  u64 z = r->geom.slice_size;
  u8 * buf = (u8 *) xpar_alloc_raw((sz) z);
  bool ok = true;
  /*  Bit 1: bytes were written; bit 2: a link was attempted.  */
  for (i = 0; i < r->wr_count; i++)
    seen[r->wr[i].entry] |= r->wr[i].link ? 2 : 1;
  for (i = 0; i < r->mf.count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    if (r->alias[i]) {
      if (seen[r->canon[i]] & 1) r->links_repaired++;
      if ((seen[i] & 2) && !rp_linked(r, i)) r->links_failed++;
      /*  A copy given the canonical writes must certify like it.  */
      if ((seen[i] & 1) && r->link_failed && r->link_failed[i] &&
          !rp_hash_ok(r, i, &r->mf.entry[r->canon[i]], buf)) {
        rp_note(r, "xpar: %.*s: the copy does not verify after repair\n",
                (int) e->name_len, e->name);
        ok = false;
      }
      continue;
    }
    if (!(seen[i] & 1)) continue;
    if (!rp_hash_ok(r, i, e, buf)) {
      ok = false;
      rp_note(r, "xpar: '%s' still fails its recorded hash\n", r->path[i]);
    } else r->entries_repaired++;
  }
  xpar_free(seen);  xpar_free(buf);
  rp_close_files(r);
  return ok;
}

/*  --paranoid re-encodes the repaired stream and compares every present
    recovery slice, independently detecting a clean-cell CRC collision.  */
static bool rp_paranoid(rp * r, u32 chunk) {
  xpar_codec * cd;
  u64 s = r->geom.slice_count, i, col;
  u8 ** dptr;  u8 ** rptr;  u8 * pool;
  bool ok = true;
  if (!r->rec_avail || !s) return true;
  if (!xpar_codec_supports_axis(r->sd.codec, r->sd.field_log2, s,
                                r->rec_total,
                                r->sd.recovery_axis_log2))
    return true;
  cd = xpar_codec_new_axis(r->sd.codec, r->sd.field_log2, s, r->rec_total,
                           r->sd.recovery_axis_log2);
  /* Guard the size_t allocation on 32-bit hosts. */
  FATAL_UNLESS("Repair column allocation is too large for this host.",
               !chunk || s + r->rec_total <= (u64) (sz) -1 / chunk);
  pool = (u8 *) xpar_alloc_raw((sz) ((s + r->rec_total) * chunk));
  dptr = (u8 **) xpar_alloc_raw((sz) s * sizeof(u8 *));
  rptr = (u8 **) xpar_alloc_raw((sz) (r->rec_total ? r->rec_total : 1) *
                                sizeof(u8 *));
  for (i = 0; i < s; i++) dptr[i] = pool + i * chunk;
  for (i = 0; i < r->rec_total; i++) rptr[i] = pool + (s + i) * chunk;
  for (col = 0; col < r->geom.cells_per_slice && ok; col++) {
    u64 width = xpar_cell_size(&r->geom, (u32) col);
    u64 base  = col * (r->geom.cell_bytes ? r->geom.cell_bytes
                                          : r->geom.slice_size);
    u64 at;
    for (at = 0; at < width && ok; at += chunk) {
      u64 n = MIN((u64) chunk, width - at);
      for (i = 0; i < s; i++)
        rp_read_stream(r, xpar_slice_begin(&r->geom, i) + base + at, n,
                       dptr[i]);
      if (xpar_codec_encode(cd, (const u8 * const *) dptr, rptr,
                            (sz) n) != XPAR_CODEC_OK) { ok = false;  break; }
      for (i = 0; i < r->rec_total; i++) {
        if (!r->rec_present[i]) continue;
        if (xpar_memcmp(rptr[i], r->rec[i] + base + at, (sz) n)) {
          rp_note(r, "xpar: recovery slice %" PRIu64 " disagrees with the "
                     "repaired data at column %" PRIu64 ".\n",
                  i, col);
          ok = false;
        }
      }
    }
  }
  xpar_free(pool);  xpar_free(dptr);  xpar_free(rptr);
  xpar_codec_free(cd);
  rp_close_files(r);
  return ok;
}

typedef struct { u64 dev, ino;  u32 entry; } rp_link;

static int rp_link_cmp(const rp_link * a, const rp_link * b) {
  if (a->dev != b->dev) return a->dev < b->dev ? -1 : 1;
  if (a->ino != b->ino) return a->ino < b->ino ? -1 : 1;
  return a->entry < b->entry ? -1 : (a->entry > b->entry);
}

static void rp_link_sift(rp_link * a, u32 root, u32 n) {
  while (1) {
    u32 c = 2 * root + 1, big;
    if (c >= n) return;
    big = c;
    if (c + 1 < n && rp_link_cmp(&a[c], &a[c + 1]) < 0) big = c + 1;
    if (rp_link_cmp(&a[root], &a[big]) >= 0) return;
    { rp_link t = a[root];  a[root] = a[big];  a[big] = t; }
    root = big;
  }
}

static void rp_find_aliases(rp * r) {
  u32 caps = xpar_fs_caps(r->dir), i;
  for (i = 0; i < r->mf.count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    r->path[i] = xpar_path_join_n(r->dir, e->name, e->name_len);
    r->canon[i] = i;
  }
  for (i = 0; i < r->mf.count; i++) {
    if (r->mf.entry[i].entry_type != XPAR_ENTRY_HARDLINK) continue;
    { i64 t = xpar_link_target(&r->mf, &r->nix, i);
      r->alias[i] = 1;
      if (t >= 0) r->canon[i] = (u32) t;
    }
  }
  if (!(caps & XPAR_FS_LINKID)) return;
  /*  Group names by inode, then merge those with matching extents.  */
  { rp_link * lk = (rp_link *) xpar_alloc_raw(
                     (sz) MAX(r->mf.count, 1) * sizeof(rp_link));
    u32 n = 0, g0;
    for (i = 0; i < r->mf.count; i++) {
      xpar_stat_t a;
      if (r->alias[i] || r->mf.entry[i].entry_type != XPAR_ENTRY_REGULAR)
        continue;
      if (xpar_lstat(r->path[i], &a) != 0 || a.nlink < 2) continue;
      lk[n].dev = a.dev;  lk[n].ino = a.ino;  lk[n].entry = i;  n++;
    }
    if (n > 1) {
      for (i = n / 2; i-- > 0;) rp_link_sift(lk, i, n);
      for (i = n; i-- > 1;) {
        rp_link t = lk[0];  lk[0] = lk[i];  lk[i] = t;
        rp_link_sift(lk, 0, i);
      }
    }
    /*  Groups are contiguous and retain manifest order.  */
    for (g0 = 0; g0 < n;) {
      u32 g1 = g0 + 1, a, b;
      while (g1 < n && lk[g1].dev == lk[g0].dev && lk[g1].ino == lk[g0].ino)
        g1++;
      for (a = g0 + 1; a < g1; a++) {
        u32 ai = lk[a].entry;
        if (r->alias[ai]) continue;
        for (b = g0; b < a; b++) {
          u32 bi = lk[b].entry;
          const xpar_entry * ea = &r->mf.entry[ai], * eb = &r->mf.entry[bi];
          u32 k;
          bool same = ea->extent_count == eb->extent_count;
          if (r->alias[bi]) continue;
          for (k = 0; k < ea->extent_count && same; k++)
            same = ea->extents[k].stream_offset ==
                     eb->extents[k].stream_offset &&
                   ea->extents[k].length == eb->extents[k].length;
          if (!same) {
            /*  One name in two generations is not two linked names.  */
            if (ea->name_len == eb->name_len &&
                !xpar_memcmp(ea->name, eb->name, ea->name_len)) continue;
            rp_note(r, "xpar: '%s' and '%s' share an inode but differ; "
                       "kept separate.\n",
                    r->path[ai], r->path[bi]);
            continue;
          }
          r->alias[ai] = 1;  r->canon[ai] = bi;
          break;
        }
      }
      g0 = g1;
    }
    xpar_free(lk); }
}

/*  Whether the entry has no extents in this generation.  */
static bool rp_foreign(const rp * r, u32 i) {
  const xpar_entry * e = &r->mf.entry[i];
  u64 lo = r->geom.stream_base, hi = lo + r->geom.stream_length;
  u32 k;
  if (!e->extent_count) return false;
  for (k = 0; k < e->extent_count; k++)
    if (e->extents[k].stream_offset >= lo &&
        e->extents[k].stream_offset < hi) return false;
  return true;
}

static void rp_entry_state_alloc(rp * r) {
  u32 i;
  u32 n = r->mf.count ? r->mf.count : 1;
  r->path = (char **) xpar_calloc(n, sizeof(char *));
  r->alias = (u8 *) xpar_calloc(n, 1);
  r->canon = (u32 *) xpar_calloc(n, sizeof(u32));
  r->fsize = (u64 *) xpar_calloc(n, sizeof(u64));
  r->fstate = (u8 *) xpar_calloc(n, 1);
  r->hash_bad = (u8 *) xpar_calloc(n, 1);
  r->io_bad = (u8 *) xpar_calloc(n, 1);
  r->resync = (xpar_resync_map *)
                xpar_calloc(n, sizeof(xpar_resync_map));
  rp_find_aliases(r);
  for (i = 0; i < r->mf.count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    xpar_stat_t st;
    if (e->entry_type != XPAR_ENTRY_REGULAR || r->alias[i]) continue;
    if (xpar_lstat(r->path[i], &st) == 0) {
      if (!st.is_regular) continue;
      r->fstate[i] = 1;
      r->fsize[i] = st.size;
      if (st.size > e->length && !rp_foreign(r, i)) r->fstate[i] |= 2;
    } else if (!xpar_errno_absent(xpar_errno())) {
      rp_io_error(r, i, xpar_errno());
    }
  }
}

static void rp_entry_state_free(rp * r) {
  u32 i;
  rp_close_files(r);
  for (i = 0; i < r->mf.count; i++) {
    xpar_free(r->path[i]);
    xpar_resync_map_free(&r->resync[i]);
  }
  xpar_free(r->path);  xpar_free(r->alias);  xpar_free(r->canon);
  xpar_free(r->fsize);  xpar_free(r->fstate);  xpar_free(r->resync);
  xpar_free(r->hash_bad);  xpar_free(r->io_bad);
  r->path = NULL;  r->alias = NULL;  r->canon = NULL;
  r->fsize = NULL;  r->fstate = NULL;  r->resync = NULL;
  r->hash_bad = NULL;  r->io_bad = NULL;
  xpar_occindex_free(&r->ox);
  xpar_nameidx_free(&r->nix);
  xpar_free(r->owner);  r->owner = NULL;
  xpar_manifest_free(&r->mf);
}

static void rp_select_head_output(rp * r) {
  xpar_chain c;
  xpar_genref ref;
  u32 g;
  rp_entry_state_free(r);
  xpar_gchain_load(r->o, &c);
  xpar_memset(&ref, 0, sizeof ref);
  ref.by_id = true;
  ref.id_prefix = r->o->repair_head_id;
  g = xpar_gchain_select(&c, &ref);
  xpar_gchain_manifest(&c, g, &r->mf, &r->owner);
  r->scan_entry_count = r->mf.count;
  xpar_gchain_free(&c);
  xpar_nameidx_build(&r->mf, &r->nix);
  xpar_occindex_build(&r->mf, &r->ox);
  rp_entry_state_alloc(r);
}

/*  --to and --backup materialise whole entries through solved-cell stream
    reads.  */
static bool rp_read_repaired(rp * r, u32 entry, u64 off, u64 len, u8 * dst) {
  const xpar_entry * e = &r->mf.entry[entry];
  u64 fo = 0, want = off + len;
  u32 k;
  bool ok = true;
  xpar_memset(dst, 0, (sz) len);
  for (k = 0; k < e->extent_count; k++) {
    u64 xl = e->extents[k].length;
    u64 lo = MAX(off, fo), hi = MIN(want, fo + xl);
    if (lo < hi) {
      u64 so = e->extents[k].stream_offset + (lo - fo);
      if (!rp_read_stream(r, so, hi - lo, dst + (lo - off))) ok = false;
    }
    fo += xl;
  }
  return ok;
}

static xpar_file * rp_tree_stage(const char * path, char ** stage) {
  char * stem = NULL;
  xpar_file * f;
  int err;
  xpar_asprintf(&stem, "%s.xpar-repair-", path);
  f = xpar_stage_open(stem, XPAR_O_WRONLY | XPAR_O_NOFOLLOW, 1, stage);
  err = xpar_errno();
  xpar_free(stem);
  if (!f)
    FATAL_IO("Cannot stage repair beside '%s': %s.", path,
             xpar_strerror(err));
  return f;
}

static char * rp_backup_name(const char * path) {
  xpar_stat_t st;
  char * out = NULL;
  u32 n;
  for (n = 1; ; n++) {
    xpar_free(out);
    xpar_asprintf(&out, "%s.%" PRIu32, path, n);
    if (xpar_lstat(out, &st) != 0) return out;
    FATAL_UNLESS("Too many backups exist for '%s'.", n != UINT32_MAX, path);
  }
}

/*  Keep PATH reachable under a fresh adjacent name.  */
static char * rp_keep_aside_name(const char * path, int * err) {
  u32 n;
  *err = 0;
  for (n = 0; n < 64; n++) {
    xpar_stat_t st;
    char * bak = rp_backup_name(path);
    if (xpar_link(path, bak) == 0) return bak;
    *err = xpar_errno();
    if (xpar_lstat(bak, &st) == 0) { xpar_free(bak);  continue; }
    if (xpar_keep_aside(path, bak) == 0) return bak;
    *err = xpar_errno();
    xpar_free(bak);
    return NULL;
  }
  return NULL;
}


static void rp_publish_tree_stage(const xpar_options * o, char * stage,
                                  const char * path) {
  xpar_stat_t st;
  char * backup = NULL;
  bool had = xpar_lstat(path, &st) == 0;
  if (had) {
    int err = 0;
    FATAL_UNLESS("Refusing to replace destination directory '%s'.",
                 !st.is_dir, path);
    FATAL_UNLESS("Destination '%s' exists; -f overwrites it.",
                 o->force, path);
    backup = rp_keep_aside_name(path, &err);
    if (!backup)
      FATAL_IO("Cannot stage old destination '%s': %s.", path,
               xpar_strerror(err));
  }
  if (xpar_rename(stage, path) != 0 || xpar_fsync_dir(path) != 0) {
    int saved = xpar_errno();
    (void) xpar_remove(stage);
    if (backup && xpar_put_back(path, backup) != 0)
      xpar_fprintf(xpar_stderr, "xpar: original '%s' remains at '%s'.\n",
                   path, backup);
    FATAL_IO("Cannot publish repaired '%s': %s.", path,
             xpar_strerror(saved));
  }
  if (backup && xpar_remove(backup) != 0)
    xpar_fprintf(xpar_stderr, "xpar: old destination remains at '%s'.\n",
                 backup);
  xpar_free(backup);
}

static void rp_apply_meta(rp * r, const xpar_entry * e,
                          const xpar_posix_rec * pr, const char * path,
                          bool link, u32 caps) {
  if (link && !(caps & XPAR_FS_NOFOLLOW)) {
    rp_meta_skip(r->meta_skip, r->o, e, RP_META_SYMLINK,
                 "the host has no symlink-safe metadata call");
    rp_note(r, "xpar: %.*s: symlink metadata skipped; no safe host API\n",
            (int) e->name_len, e->name);
    return;
  }
  /*  Set ownership before mode because chown clears set-ID bits.  */
  if (pr && (pr->uid != XPAR_ID_NONE || pr->gid != XPAR_ID_NONE ||
             pr->owner || pr->group)) {
    if (!(r->o->preserve & XPAR_PRES_OWNER))
      rp_meta_skip(r->meta_skip, r->o, e, RP_META_OWNER,
                   "--preserve=owner was not given");
    else if (xpar_set_owner(
               path, 1, pr->uid, pr->gid,
               r->o->owner_map == XPAR_OWNERMAP_NAME ? pr->owner : NULL,
               r->o->owner_map == XPAR_OWNERMAP_NAME ? pr->group : NULL)
             != 0) {
      rp_meta_skip(r->meta_skip, r->o, e, RP_META_OWNER,
                   xpar_strerror(xpar_errno()));
      rp_note(r, "xpar: %.*s: owner restoration skipped: %s.\n",
              (int) e->name_len, e->name, xpar_strerror(xpar_errno()));
    }
  }

  /*  Do not follow symlinks to set their targets' modes. */
  if ((r->o->preserve & XPAR_PRES_MODE) && !link &&
      e->mode != XPAR_ABSENT_U32) {
    u32 mode = e->mode & XPAR_MODE_PERM;
    if (!(r->o->preserve & XPAR_PRES_SETID)) {
      mode &= ~(u32) (XPAR_MODE_SETUID | XPAR_MODE_SETGID |
                      XPAR_MODE_STICKY);
      if (e->mode & (XPAR_MODE_SETUID | XPAR_MODE_SETGID |
                     XPAR_MODE_STICKY))
        rp_meta_skip(r->meta_skip, r->o, e, RP_META_SETID,
                     "setuid, setgid and sticky bits cleared");
    }
    if (xpar_set_mode(path, 1, mode) != 0) {
      rp_meta_skip(r->meta_skip, r->o, e, RP_META_MODE,
                   xpar_strerror(xpar_errno()));
      rp_note(r, "xpar: %.*s: mode restoration skipped: %s.\n",
              (int) e->name_len, e->name, xpar_strerror(xpar_errno()));
    }
  }
  {
    i64 at = XPAR_TIME_NONE, mt = XPAR_TIME_NONE, bt = XPAR_TIME_NONE;
    if ((r->o->preserve & XPAR_PRES_ATIME) &&
        e->atime_ns != XPAR_ABSENT_TIME) at = e->atime_ns;
    else if (e->atime_ns != XPAR_ABSENT_TIME)
      rp_meta_skip(r->meta_skip, r->o, e, RP_META_ATIME,
                   "--preserve=atime was not given");
    if ((r->o->preserve & XPAR_PRES_MTIME) &&
        e->mtime_ns != XPAR_ABSENT_TIME) mt = e->mtime_ns;
    if ((r->o->preserve & XPAR_PRES_BTIME) &&
        e->btime_ns != XPAR_ABSENT_TIME) {
      if (caps & XPAR_FS_BTIME) bt = e->btime_ns;
      else rp_meta_skip(r->meta_skip, r->o, e, RP_META_BTIME,
                        "this host cannot set a birth time");
    }
    if ((at != XPAR_TIME_NONE || mt != XPAR_TIME_NONE ||
         bt != XPAR_TIME_NONE) && xpar_set_times(path, 1, at, mt, bt) != 0) {
      rp_meta_skip(r->meta_skip, r->o, e, RP_META_TIMES,
                   xpar_strerror(xpar_errno()));
      rp_note(r, "xpar: %.*s: time restoration skipped: %s.\n",
              (int) e->name_len, e->name, xpar_strerror(xpar_errno()));
    }
    if (e->ctime_ns != XPAR_ABSENT_TIME) {
      rp_meta_skip(r->meta_skip, r->o, e, RP_META_CTIME,
                   "ctime cannot be set on any host");
      if (r->o->preserve & XPAR_PRES_CTIME)
      rp_note(r, "xpar: %.*s: ctime cannot be restored.\n",
              (int) e->name_len, e->name);
    }
  }
  if ((r->o->preserve & XPAR_PRES_ATTRS) &&
      (e->attrs & XPAR_ATTR_SETTABLE) &&
      xpar_set_attrs(path, 1, (u16) (e->attrs & XPAR_ATTR_SETTABLE)) != 0) {
    rp_meta_skip(r->meta_skip, r->o, e, RP_META_ATTRS,
                 xpar_strerror(xpar_errno()));
    rp_note(r, "xpar: %.*s: attribute restoration skipped: %s.\n",
            (int) e->name_len, e->name, xpar_strerror(xpar_errno()));
  }
  if (pr && pr->xattr_count && !(r->o->preserve & XPAR_PRES_XATTR))
    rp_meta_skip(r->meta_skip, r->o, e, RP_META_XATTR,
                 "--preserve=xattr was not given");
  if (pr && (r->o->preserve & XPAR_PRES_XATTR)) {
    u32 k;
    for (k = 0; k < pr->xattr_count; k++) {
      const xpar_xattr * a = &pr->xattrs[k];
      bool user = a->name && !xpar_strncmp(a->name, "user.", 5);
      if (!user && !(r->o->preserve & XPAR_PRES_XATTR_ALL)) {
        rp_meta_skip(r->meta_skip, r->o, e, RP_META_XATTR_NS,
                     "outside user. and --preserve=xattr-all was not given");
        continue;
      }
      if (xpar_setxattr(path, 1, a->name, a->value, a->value_len) != 0) {
        rp_meta_skip(r->meta_skip, r->o, e, RP_META_XATTR,
                     xpar_strerror(xpar_errno()));
        rp_note(r, "xpar: %.*s: xattr restoration skipped: %s.\n",
                (int) e->name_len, e->name, xpar_strerror(xpar_errno()));
      }
    }
  }
}

static void rp_basic_meta(rp * r, u32 idx, const char * path, bool link) {
  const xpar_entry * e = &r->mf.entry[idx];
  const xpar_posix_rec * pr = NULL;
  if (r->owner && r->owner[idx] < r->posix_gen_count &&
      e->posix_index != XPAR_ABSENT_U32 &&
      e->posix_index < r->posix_tab_count[r->owner[idx]])
    pr = &r->posix_tab[r->owner[idx]][e->posix_index];
  rp_apply_meta(r, e, pr, path, link, xpar_fs_caps(path));
}

static void rp_write_tree(rp * r, const char * dir, bool backup) {
  u32 i;
  const u64 chunk = (u64) 1 << 16;
  u8 * buf = (u8 *) xpar_alloc_raw((sz) chunk);
  u8 * hit = (u8 *) xpar_calloc(r->mf.count ? r->mf.count : 1, 1);
  if (!backup) rp_tree_preflight(r->o, &r->mf, dir);
  /*  Back up damaged cells and overlong tails only.  */
  for (i = 0; i < r->edit_count; i++) hit[r->edit[i].entry] = 1;
  for (i = 0; i < r->mf.count; i++)
    if ((r->fstate[i] & 2) && !r->alias[i]) hit[i] = 1;
  for (i = 0; i < r->mf.count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    xpar_path_status why;
    char * out;
    xpar_file * f;
    u64 at = 0;
    if (e->entry_type == XPAR_ENTRY_DIR) {
      if (backup) continue;
      out = xpar_path_resolve(dir, e->name, e->name_len, 0, &why);
      FATAL_UNLESS("Refusing repair output '%.*s': %s.", out != NULL,
                   (int) e->name_len, e->name, xpar_path_reason(why));
      if (xpar_mkdir_p(out, 0777) != 0) {
        int err = xpar_errno();
        xpar_stat_t st;
        if (xpar_lstat(out, &st) != 0 || !st.is_dir)
          FATAL_IO("Cannot create '%s': %s.", out, xpar_strerror(err));
      }
      xpar_free(out);
      continue;
    }
    if (e->entry_type != XPAR_ENTRY_REGULAR) continue;
    if (backup && !hit[i]) continue;
    /*  --backup leaves unreadable originals untouched.  */
    if (backup && r->io_bad && r->io_bad[i]) {
      r->unrecovered++;
      rp_note(r, "xpar: %s: not rewritten; its bytes could not be read\n",
              r->path[i]);
      continue;
    }
    if (backup) {
      out = xpar_strdup(r->path[i]);
    } else {
      out = xpar_path_resolve(dir, e->name, e->name_len, 0, &why);
      FATAL_UNLESS("Refusing repair output '%.*s': %s.", out != NULL,
                   (int) e->name_len, e->name, xpar_path_reason(why));
      { char * d = xpar_path_dir(out);
        if (xpar_mkdir_p(d, 0777) != 0) {
          int err = xpar_errno();
          xpar_stat_t st;
          if (xpar_lstat(d, &st) != 0 || !st.is_dir)
            FATAL_IO("Cannot create '%s': %s.", d, xpar_strerror(err));
        }
        xpar_free(d); }
    }
    {
      char * stage = NULL;
      char * bak = NULL;
      xpar_blake3_t h;
      u8 got[32];
      f = rp_tree_stage(out, &stage);
      if (r->auth_only) xpar_blake3_init_keyed(&h, r->key.k_file);
      else xpar_blake3_init(&h);
    while (at < e->length) {
      u64 take = MIN(chunk, e->length - at);
      /*  The content hash, not a short read, validates rebuilt data.  */
      (void) rp_read_repaired(r, i, at, take, buf);
      xpar_xwrite(f, buf, (sz) take);
      xpar_blake3_update(&h, buf, (sz) take);
      at += take;
      r->bytes_written += take;
    }
    if (xpar_fsync(f) != 0)
      FATAL_IO("Flushing staged repair of '%s' failed.", out);
    xpar_xclose(f);
      xpar_blake3_final(&h, got, sizeof got);
      /*  Skip unrecoverable entries without abandoning the tree.  */
      if (!xpar_ct_equal(got, e->content_hash, sizeof got)) {
        xpar_remove(stage);
        xpar_free(stage);  xpar_free(bak);  xpar_free(out);
        r->unrecovered++;
        rp_note(r, "xpar: '%.*s' could not be reproduced; omitted from "
                "the repaired tree\n",
                (int) e->name_len, e->name);
        continue;
      }
      if (backup) {
        xpar_stat_t bst;
        /*  Missing entries have no original to back up.  */
        if (xpar_lstat(out, &bst) == 0) {
          int err = 0;
          bak = rp_keep_aside_name(out, &err);
          if (!bak)
            FATAL_IO("Cannot back up '%s': %s.", out,
                     xpar_strerror(err));
        }
        if (xpar_rename(stage, out) != 0) {
          int saved = xpar_errno();
          (void) xpar_remove(stage);
          if (bak && xpar_put_back(out, bak) != 0)
            xpar_fprintf(xpar_stderr, "xpar: original '%s' remains at "
                         "'%s'.\n", out, bak);
          FATAL_IO("Cannot publish repaired '%s': %s.", out,
                   xpar_strerror(saved));
        }
        if (xpar_fsync_dir(out) != 0)
          FATAL_IO("Flushing repaired '%s' failed: %s.", out,
                   xpar_strerror(xpar_errno()));
      } else {
        rp_publish_tree_stage(r->o, stage, out);
      }
      xpar_free(stage); xpar_free(bak);
    }
    r->writes++;
    /*  Count the entries this run actually put right.  */
    if (hit[i]) r->entries_repaired++;
    /*  Never apply metadata through a symlink.  */
    rp_basic_meta(r, i, out, false);
    xpar_free(out);
  }
  /*  Create aliases after regular entries; fall back to a reported copy.  */
  for (i = 0; i < r->mf.count; i++) {
    const xpar_entry * e = &r->mf.entry[i];
    i64 t;
    char * out;  char * src;
    xpar_path_status why, src_why = XPAR_PATH_OK;
    if (backup) continue;
    if (e->entry_type == XPAR_ENTRY_SYMLINK) {
      out = xpar_path_resolve(dir, e->name, e->name_len, 0, &why);
      FATAL_UNLESS("Refusing repair output '%.*s': %s.", out != NULL,
                   (int) e->name_len, e->name, xpar_path_reason(why));
      if (e->extra) {
        char * tgt = xpar_strndup((const char *) e->extra, e->extra_len);
        int err = 0;
        char * stage = rp_symlink_aside(tgt, out, &err);
        if (!stage)
          FATAL_IO("Cannot stage symbolic link '%s': %s.", out,
                   xpar_strerror(err));
        rp_publish_tree_stage(r->o, stage, out);
        xpar_free(stage);
        xpar_free(tgt);
      }
      xpar_free(out);
      continue;
    }
    if (e->entry_type != XPAR_ENTRY_HARDLINK) continue;
    t = xpar_link_target(&r->mf, &r->nix, i);
    if (t < 0) continue;
    /* Keep output and target rejection reasons separate. */
    out = xpar_path_resolve(dir, e->name, e->name_len, 0, &why);
    FATAL_UNLESS("Refusing repair output '%.*s': %s.", out != NULL,
                 (int) e->name_len, e->name, xpar_path_reason(why));
    src = backup ? xpar_strdup(r->path[t])
                 : xpar_path_resolve(dir, r->mf.entry[t].name,
                                     r->mf.entry[t].name_len, 0, &src_why);
    FATAL_UNLESS("Unsafe hard-link target '%.*s': %s.",
                 src != NULL, (int) r->mf.entry[t].name_len,
                 r->mf.entry[t].name, xpar_path_reason(src_why));
    {
      int err = 0;
      char * link_stage = rp_link_aside(src, out, &err);
      if (link_stage) {
        rp_publish_tree_stage(r->o, link_stage, out);
        xpar_free(link_stage);
        xpar_free(out);  xpar_free(src);
        continue;
      }
    }
    {
      u64 at = 0;
      char * stage = NULL;
      xpar_file * f = rp_tree_stage(out, &stage);
        while (at < e->length) {
          u64 take = MIN(chunk, e->length - at);
          if (!rp_read_repaired(r, (u32) t, at, take, buf))
            FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                       "The hard-link copy '%.*s' became unreadable.",
                       (int) e->name_len, e->name);
          xpar_xwrite(f, buf, (sz) take);
          at += take;
        }
      if (xpar_fsync(f) != 0)
        FATAL_IO("Flushing hard-link copy '%s' failed.", out);
      xpar_xclose(f);
      rp_publish_tree_stage(r->o, stage, out);
      xpar_free(stage);
      rp_meta_skip(r->meta_skip, r->o, e, RP_META_COPY,
                   "the destination cannot create the hard link");
      rp_note(r, "xpar: %.*s: materialised-as-copy.\n", (int) e->name_len,
              e->name);
    }
    xpar_free(out);  xpar_free(src);
  }
  if (!backup) for (i = r->mf.count; i-- > 0; ) {
    const xpar_entry * e = &r->mf.entry[i];
    xpar_path_status why;
    char * out;
    bool link = e->entry_type == XPAR_ENTRY_SYMLINK;
    if (e->entry_type != XPAR_ENTRY_DIR && !link) continue;
    /*  Allow the leaf symlink, but reject symlinked parents.  */
    out = xpar_path_resolve(dir, e->name, e->name_len,
                            link ? XPAR_PATH_LEAF_LINK : 0, &why);
    if (out) { rp_basic_meta(r, i, out, link);  xpar_free(out); }
  }
  xpar_free(buf);  xpar_free(hit);
}

static void rp_report(rp * r, const char * status, int code) {
  if (r->o->json) {
    xpar_json_begin(&r->js, "repair");
    xpar_json_u64(&r->js, "cells_damaged", r->cell_count);
    xpar_json_u64(&r->js, "cells_copied", r->cells_copied);
    xpar_json_u64(&r->js, "cells_decoded", r->cells_decoded);
    xpar_json_u64(&r->js, "writes", r->writes);
    xpar_json_u64(&r->js, "bytes_written", r->bytes_written);
    xpar_json_u64(&r->js, "entries_repaired", r->entries_repaired);
    xpar_json_u64(&r->js, "entries_overlong", r->overlong);
    xpar_json_u64(&r->js, "links_repaired", r->links_repaired);
    xpar_json_u64(&r->js, "links_relinked", r->links_made);
    xpar_json_u64(&r->js, "entries_opaque", r->opaque);
    xpar_json_u64(&r->js, "names_recreated", r->names_made);
    xpar_json_u64(&r->js, "names_failed", r->names_failed);
    xpar_json_u64(&r->js, "links_failed", r->links_failed);
    xpar_json_u64(&r->js, "recovery_regenerated", r->rec_regen);
    xpar_json_u64(&r->js, "index_volumes_recreated", r->index_regen);
    xpar_json_u64(&r->js, "volumes_stale_rewritten", r->stale_regen);
    xpar_json_u64(&r->js, "volumes_restored", r->names_restored);
    xpar_json_u64(&r->js, "volumes_trimmed", r->ragged_trimmed);
    xpar_json_u64(&r->js, "volumes_dropped_rewritten", r->vols_dropped);
    xpar_json_u64(&r->js, "inner_corrected", r->armg_corrected);
    xpar_json_end(&r->js);
    rp_meta_report(r->o, r->meta_skip, &r->js);
    xpar_json_summary(&r->js, status, code);
    return;
  }
  rp_meta_report(r->o, r->meta_skip, NULL);
  if (!r->quiet) {
    if (r->links_made)
      rp_note(r, "xpar: relinked %" PRIu64 " hard-link name%s.\n",
              r->links_made, r->links_made == 1 ? "" : "s");
    if (r->overlong)
      rp_note(r, xpar_strcmp(status, "dry-run")
                   ? "xpar: restored %" PRIu64 " overlong entr%s.\n"
                   : "xpar: found %" PRIu64 " overlong entr%s.\n",
              r->overlong, r->overlong == 1 ? "y" : "ies");
    if (r->names_made)
      rp_note(r, r->o->dry_run
                   ? "xpar: would recreate %" PRIu64 " missing name%s from "
                     "the manifest.\n"
                   : "xpar: recreated %" PRIu64 " missing name%s from the "
                     "manifest.\n", r->names_made, PLURAL(r->names_made));
    if (r->names_failed)
      rp_note(r, "xpar: failed to recreate %" PRIu64 " missing name%s.\n",
              r->names_failed, PLURAL(r->names_failed));
    if (r->links_failed)
      rp_note(r, "xpar: failed to link %" PRIu64 " hard-link name%s.\n",
              r->links_failed, PLURAL(r->links_failed));
    if (r->index_regen)
      rp_note(r, r->o->dry_run
                   ? "xpar: %" PRIu64 " index volume%s would be recreated "
                     "from packet replicas\n"
                   : "xpar: recreated %" PRIu64 " index volume%s from packet "
                     "replicas\n", r->index_regen, PLURAL(r->index_regen));
    if (r->rec_regen)
      rp_note(r, r->o->dry_run
                   ? "xpar: %" PRIu64 " recovery slice%s would be regenerated "
                     "in %" PRIu64 " volume%s\n"
                   : "xpar: %" PRIu64 " recovery slice%s regenerated in %"
                     PRIu64 " volume%s\n",
              r->rec_regen, PLURAL(r->rec_regen),
              r->rec_regen_vols, PLURAL(r->rec_regen_vols));
    if (r->stale_regen)
      rp_note(r, r->o->dry_run
                   ? "xpar: %" PRIu64 " stale volume%s would be rewritten\n"
                   : "xpar: rewrote %" PRIu64 " stale volume%s\n",
              r->stale_regen, PLURAL(r->stale_regen));
    if (r->armg_corrected)
      rp_note(r, "xpar: the inner code corrected %" PRIu64
              " armoured region%s\n", r->armg_corrected,
              PLURAL(r->armg_corrected));
    if (!r->cell_count && !r->overlong && !r->links_made && !r->opaque &&
        !r->names_made && !r->rec_regen && !r->index_regen &&
        !r->stale_regen && !r->names_failed && !r->links_failed &&
        !r->names_restored && !r->ragged_trimmed && !r->vols_dropped &&
        !r->o->chain_member)
      rp_note(r, "xpar: no damage found.\n");
    else if (r->cell_count)
      rp_note(r, "xpar: %" PRIu32 " cell%s damaged, %" PRIu64 " copied, %"
              PRIu64 " decoded; "
                 "%" PRIu64 " write%s, %" PRIu64 " bytes; %" PRIu64 " %s repaired"
                 " (%" PRIu64 " further %s a repaired inode).\n",
              r->cell_count, PLURAL(r->cell_count),
              r->cells_copied,
              r->cells_decoded,
              r->writes, PLURAL(r->writes),
              r->bytes_written,
              r->entries_repaired,
              r->entries_repaired == 1 ? "entry" : "entries",
              r->links_repaired,
              r->links_repaired == 1 ? "name shares" : "names share");
  }
}

static void rp_free(rp * r) {
  u32 i;
  rp_close_files(r);
  for (i = 0; i < r->vol_count; i++) xpar_volimg_close(&r->vol[i]);
  xpar_free(r->vol);
  for (i = 0; i < r->plain_count; i++) xpar_free(r->plain[i]);
  xpar_free(r->plain);
  for (i = 0; i < r->cell_count; i++) xpar_free(r->cell[i].bytes);
  xpar_free(r->cell);  xpar_free(r->edit);
  for (i = 0; i < r->wr_count; i++)
    { xpar_free(r->wr[i].data);  xpar_free(r->wr[i].old); }
  xpar_free(r->wr);
  rp_entry_state_free(r);
  xpar_free(r->link_failed);
  xpar_free(r->susp);
  xpar_free((void *) r->rec);  xpar_free(r->rec_present);
  if (r->have_layt) xpar_layt_free(&r->layt);
  xpar_erasures_free(&r->er);
  xpar_tagset_free(&r->tags);
  for (i = 0; i < r->posix_gen_count; i++)
    xpar_gchain_posix_free(r->posix_tab[i], r->posix_tab_count[i]);
  xpar_free(r->posix_tab);  xpar_free(r->posix_tab_count);
  if (r->have_setd) xpar_setd_free(&r->sd);
  xpar_critset_free(&r->crit);
  xpar_free(r->dir);  xpar_free(r->journal);
  xpar_key_forget(&r->key, r->master);
}

/*  Owned layouts repair the logical stream. Reconstructed slices pass their
    strong tags before any volume is published.  */

typedef struct {
  xpar_mmap map;
  u8 * heap;
  const u8 * data;
  u64 len;
  char * path;
} owned_vol;

static bool owned_open(owned_vol * v, const char * path) {
  xpar_file * f;
  i64 n;
  xpar_memset(v, 0, sizeof *v);
  v->path = xpar_strdup(path);
  v->map = xpar_map(path);
  if (v->map.valid) { v->data = v->map.map; v->len = v->map.size; return true; }
  f = xpar_open(path, XPAR_O_RDONLY);
  if (!f) {
    if (!xpar_errno_absent(xpar_errno()))
      FATAL_IO("Cannot read recovery volume '%s': %s.", path,
               xpar_strerror(xpar_errno()));
    xpar_free(v->path); v->path = NULL; return false;
  }
  n = xpar_size(f);
  if (n < 0) FATAL_IO("Cannot size '%s': %s.", path,
                      xpar_strerror(xpar_error(f)));
  if ((u64) n > (u64) (sz) -1) {
    xpar_close(f); xpar_free(v->path); v->path = NULL; return false;
  }
  v->heap = (u8 *) xpar_alloc_raw(n ? (sz) n : 1);
  if (n) {
    sz got = xpar_read(f, v->heap, (sz) n);
    if (got != (sz) n) {
      /*  Accept a short read only if the file shrank.  */
      i64 now = xpar_size(f);
      if (now < 0 || (u64) now != (u64) got)
        FATAL_IO("Reading '%s' stopped after %" PRIu64 " of %" PRIu64
                 " bytes.", path, (u64) got, (u64) n);
      n = (i64) got;
    }
  }
  xpar_close(f); v->data = v->heap; v->len = (u64) n; return true;
}

static void owned_close(owned_vol * v) {
  if (v->map.valid) xpar_unmap(&v->map); else xpar_free(v->heap);
  xpar_free(v->path);
}

static char * owned_chain_gen_dir(const char * root, const u8 * id) {
  char text[XPAR_SET_ID_LEN * 2 + 1], * name = NULL, * out;
  xpar_hex(text, id, XPAR_SET_ID_LEN);
  xpar_asprintf(&name, "g-%s", text);
  out = xpar_path_join(root, name);
  xpar_free(name);
  return out;
}

/*  Find ancestor bytes by global extent coordinate; renames and chunk
    deduplication make current pathnames unreliable.  */
static bool owned_chain_read(const xpar_chain * c, xpar_manifest * mf,
                             u8 * loaded, const char * root, u64 at,
                             u8 * out, u64 * length) {
  i64 owner = xpar_gchain_gen_of(c, at, *length);
  char * dir;
  u32 i;
  if (owner < 0) return false;
  if (!loaded[owner]) {
    u32 * entry_owner = NULL;
    xpar_gchain_manifest(c, (u32) owner, &mf[owner], &entry_owner);
    xpar_free(entry_owner);
    loaded[owner] = 1;
  }
  dir = owned_chain_gen_dir(root, c->gen[owner].set_id);
  for (i = 0; i < mf[owner].count; i++) {
    const xpar_entry * e = &mf[owner].entry[i];
    u64 file_off = 0;
    u32 k;
    if (e->entry_type != XPAR_ENTRY_REGULAR) continue;
    for (k = 0; k < e->extent_count; k++) {
      const xpar_extent * x = &e->extents[k];
      if (at >= x->stream_offset && at - x->stream_offset < x->length) {
        xpar_path_status why;
        char * path = xpar_path_resolve(dir, e->name, e->name_len, 0, &why);
        xpar_file * f;
        u64 take = MIN(*length, x->length - (at - x->stream_offset));
        if (!path) { xpar_free(dir); return false; }
        f = xpar_open(path, XPAR_O_RDONLY);
        if (!f || xpar_pread(f, out, (sz) take,
                             file_off + at - x->stream_offset) != (sz) take) {
          if (f) xpar_close(f);
          xpar_free(path); xpar_free(dir);
          return false;
        }
        xpar_close(f);
        xpar_free(path); xpar_free(dir);
        *length = take;
        return true;
      }
      file_off += x->length;
    }
  }
  xpar_free(dir);
  return false;
}

static char * owned_chain_stage_new(const char * destination) {
  xpar_stat_t st;
  char * stem, * path;
  if (xpar_mkdir_p(destination, 0777) != 0) {
    int err = xpar_errno();
    if (xpar_lstat(destination, &st) != 0)
      FATAL_IO("Cannot create '%s': %s.", destination, xpar_strerror(err));
  }
  FATAL_UNLESS("The repair destination '%s' is a symbolic link; refusing "
               "to write through it.",
               xpar_lstat(destination, &st) == 0 && !st.is_symlink,
               destination);
  stem = xpar_path_join(destination, ".xpar-chain-");
  path = xpar_stage_dir(stem);
  xpar_free(stem);
  if (path) return path;
  FATAL_IO("Cannot create a private chain-repair stage in '%s': %s.",
           destination, xpar_strerror(xpar_errno()));
  return NULL;
}

static void owned_chain_stage_remove(const xpar_chain * c,
                                     const u32 * order, u32 count,
                                     u32 selected, const char * root) {
  u32 q;
  for (q = 0; q < count; q++) {
    u32 g = order[q], i;
    xpar_manifest m;
    u32 * owner = NULL;
    char * dir;
    if (g == selected) continue;
    dir = owned_chain_gen_dir(root, c->gen[g].set_id);
    xpar_gchain_manifest(c, g, &m, &owner);
    xpar_free(owner);
    for (i = 0; i < m.count; i++)
      if (m.entry[i].entry_type != XPAR_ENTRY_DIR) {
        xpar_path_status why;
        char * p = xpar_path_resolve(dir, m.entry[i].name,
                                     m.entry[i].name_len, 0, &why);
        if (p) { (void) xpar_remove(p); xpar_free(p); }
      }
    /*  Parents precede children, so reverse order removes directories.  */
    for (i = m.count; i-- > 0; )
      if (m.entry[i].entry_type == XPAR_ENTRY_DIR) {
        xpar_path_status why;
        char * p = xpar_path_resolve(dir, m.entry[i].name,
                                     m.entry[i].name_len, 0, &why);
        if (p) { (void) xpar_rmdir(p); xpar_free(p); }
      }
    xpar_manifest_free(&m);
    (void) xpar_rmdir(dir);
    xpar_free(dir);
  }
  (void) xpar_rmdir(root);
}

/*  Stage and atomically publish owned-volume backups before changing any
    archive byte.  */
static void owned_backup_path(const char * path) {
  xpar_stat_t st, dstst;
  xpar_file * src, * dst;
  char * target = NULL, * stage = NULL;
  u8 rnd[8], * buf;
  char hex[17];
  u64 at = 0;
  u32 n;
  FATAL_UNLESS("Cannot back up missing owned volume '%s'.",
               xpar_lstat(path, &st) == 0 && st.is_regular, path);
  for (n = 1; ; n++) {
    xpar_free(target);
    xpar_asprintf(&target, "%s.%" PRIu32, path, n);
    if (xpar_lstat(target, &dstst) != 0) break;
    FATAL_UNLESS("Too many backups exist for '%s'.", n != UINT32_MAX, path);
  }
  xpar_random_bytes(rnd, sizeof rnd);
  xpar_hex(hex, rnd, sizeof rnd);
  xpar_asprintf(&stage, "%s.backup-%s.tmp", path, hex);
  src = xpar_open(path, XPAR_O_RDONLY | XPAR_O_NOFOLLOW);
  { int err = src ? 0 : xpar_errno();
    dst = xpar_open(stage, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_EXCL |
                           XPAR_O_NOFOLLOW);
    if (!dst && !err) err = xpar_errno();
    if (!src || !dst) FATAL_IO("Cannot stage backup of '%s': %s.", path,
                               xpar_strerror(err)); }
  buf = (u8 *) xpar_alloc_raw(1u << 20);
  while (at < st.size) {
    sz take = (sz) MIN(st.size - at, (u64) 1 << 20);
    if (xpar_pread(src, buf, take, at) != take)
      FATAL_IO("Reading '%s' for backup failed.", path);
    xpar_xwrite(dst, buf, take);
    at += take;
  }
  xpar_free(buf);
  if (xpar_fsync(dst) != 0) FATAL_IO("Flushing backup of '%s' failed.", path);
  xpar_xclose(src); xpar_xclose(dst);
  (void) xpar_set_mode(stage, 1, st.mode & 07777u);
  (void) xpar_set_times(stage, 1, XPAR_TIME_NONE, st.mtime_ns,
                        XPAR_TIME_NONE);
  if (xpar_rename(stage, target) != 0 || xpar_fsync_dir(target) != 0)
    FATAL_IO("Publishing backup '%s' failed: %s.", target,
             xpar_strerror(xpar_errno()));
  xpar_free(stage); xpar_free(target);
}

static void owned_scan_recovery(const xpar_vset * s, const owned_vol * v,
                                const u8 ** rec, u64 rtop) {
  xpar_scan sc;
  xpar_pkt h;
  const u8 * body;
  u64 off;
  xpar_scan_init(&sc, v->data, v->len, xpar_vset_key(s), false);
  sc.accept_unverified_keyed = false;
  while (xpar_scan_next(&sc, &h, &body, &off)) {
    xpar_rcvs r;
    if (!xpar_pkt_is(&h, XPAR_T_RCVS) ||
        xpar_memcmp(h.set_id, xpar_vset_id(s), XPAR_SET_ID_LEN)) continue;
    if (xpar_rcvs_read(body, (sz) (h.length - XPAR_PKT_HDR),
                       xpar_vset_setd(s)->slice_size, &r) != XPAR_OK)
      continue;
    if (r.exponent < rtop && !rec[r.exponent]) rec[r.exponent] = r.data;
  }
}

bool xpar_vset_recover_data(xpar_vset * s, u64 stream_offset, u64 length,
                            u64 memory, xpar_file * dst,
                            const char ** reason) {
  const xpar_setd * sd = xpar_vset_setd(s);
  const xpar_geom * g = xpar_vset_geom(s);
  const xpar_layt * l = xpar_vset_layt(s);
  const xpar_tags * tags = xpar_vset_tags(s);
  const xpar_key * key = xpar_vset_key(s);
  const u8 ** rec;
  owned_vol * rv = NULL;
  u32 rv_count = 0;
  u8 ** data, ** rptr, * present, * rpresent, * storage, * gate;
  xpar_codec * codec;
  xpar_codec_plan * plan;
  xpar_codec_status st;
  u64 end, first, last, rtop = 0, available = 0, i, at;
  u64 budget, rows, chunk;
  bool ok = true;

  if (reason) *reason = "invalid data-volume range";

  if (!l || sd->layout != XPAR_LAYOUT_SPLIT || !length ||
      stream_offset >= g->stream_length ||
      length > g->stream_length - stream_offset) return false;
  end = stream_offset + length;
  first = stream_offset / g->slice_size;
  last = xpar_ceil_div(end, g->slice_size);
  if (last > g->slice_count) last = g->slice_count;
  for (u32 q = 0; q < l->count; q++)
    if (l->vol[q].kind == XPAR_VOL_RECOVERY)
      rtop = MAX(rtop, l->vol[q].recovery_first + l->vol[q].byte_length);
  if (!rtop) {
    if (reason) *reason = "this set carries no recovery slices";
    return false;
  }
  if (last - first > rtop) {
    /*  The caller consumes this before the next call.  */
    static char shortfall[96];
    if (reason) {
      xpar_snprintf(shortfall, sizeof shortfall,
                    "too few recovery slices: %" PRIu64 " needed, %" PRIu64
                    " exist, %" PRIu64 " short",
                    last - first, rtop, last - first - rtop);
      *reason = shortfall;
    }
    return false;
  }
  if (!xpar_codec_supports_axis(sd->codec, sd->field_log2, g->slice_count,
                                rtop, sd->recovery_axis_log2)) {
    if (reason) *reason = "this codec has no plan for that recovery axis";
    return false;
  }

  rec = (const u8 **) xpar_calloc((sz) rtop, sizeof(u8 *));
  rpresent = (u8 *) xpar_calloc((sz) rtop, 1);
  rv = (owned_vol *) xpar_calloc(l->count ? l->count : 1, sizeof(*rv));
  for (u32 q = 0; q < l->count; q++)
    if (l->vol[q].kind == XPAR_VOL_RECOVERY && l->vol[q].name) {
      char * path = xpar_path_vol(xpar_vset_dir(s), l->vol[q].name);
      if (owned_open(&rv[rv_count], path)) {
        owned_scan_recovery(s, &rv[rv_count], rec, rtop);
        rv_count++;
      }
      xpar_free(path);
    }
  for (i = 0; i < rtop; i++) {
    rpresent[i] = rec[i] != NULL;
    if (rpresent[i]) available++;
  }
  if (available < last - first) {
    static char shortfall[96];
    if (reason) {
      xpar_snprintf(shortfall, sizeof shortfall,
                    "too few intact recovery slices: %" PRIu64 " needed, %"
                    PRIu64 " readable, %" PRIu64 " short",
                    last - first, available, last - first - available);
      *reason = shortfall;
    }
    ok = false; goto no_codec;
  }

  present = (u8 *) xpar_calloc((sz) g->slice_count, 1);
  for (i = 0; i < g->slice_count; i++)
    present[i] = i < first || i >= last;
  codec = xpar_codec_new_axis(sd->codec, sd->field_log2, g->slice_count,
                              rtop, sd->recovery_axis_log2);
  plan = xpar_codec_plan_new(codec, present, rpresent, &st);
  if (!plan || st != XPAR_CODEC_OK) {
    if (reason) *reason = "the erasure pattern has no decode plan";
    xpar_codec_plan_free(plan); xpar_codec_free(codec);
    xpar_free(present); ok = false; goto no_codec;
  }

  rows = g->slice_count + rtop;
  budget = memory ? memory : xpar_plan_default_memory();
  if (!rows || budget / rows < 64) {
    if (reason) *reason = "the memory limit cannot hold one codec column";
    ok = false; goto done;
  }
  chunk = MIN(g->slice_size, budget / rows);
  chunk -= chunk % 64;
  if (!chunk) chunk = 64;
  FATAL_UNLESS("Repair column allocation is too large for this host.",
               !chunk || rows <= (u64) (sz) -1 / chunk);
  storage = (u8 *) xpar_alloc_raw((sz) (rows * chunk));
  data = (u8 **) xpar_alloc_raw((sz) g->slice_count * sizeof(u8 *));
  rptr = (u8 **) xpar_alloc_raw((sz) rtop * sizeof(u8 *));
  for (i = 0; i < g->slice_count; i++) data[i] = storage + i * chunk;
  for (i = 0; i < rtop; i++) rptr[i] = storage + (g->slice_count + i) * chunk;

  for (at = 0; at < g->slice_size && ok; at += chunk) {
    u64 n = MIN(chunk, g->slice_size - at);
    for (i = 0; i < g->slice_count; i++) {
      if (!present[i]) xpar_memset(data[i], 0, (sz) n);
      else if (!xpar_vset_read(s, xpar_slice_begin(g, i) + at, data[i], n)) {
        if (reason) *reason = "another data volume is unavailable";
        ok = false; break;
      }
    }
    if (!ok) break;
    for (i = 0; i < rtop; i++) {
      if (rpresent[i]) xpar_memcpy(rptr[i], rec[i] + at, (sz) n);
      else xpar_memset(rptr[i], 0, (sz) n);
    }
    if (xpar_codec_plan_apply(plan, data, (const u8 * const *) rptr,
                              (sz) n) != XPAR_CODEC_OK) {
      if (reason) *reason = "the codec rejected the decode";
      ok = false; break;
    }
    for (i = first; i < last; i++) {
      u64 lo = MAX(stream_offset, i * g->slice_size + at);
      u64 hi = MIN(end, i * g->slice_size + at + n);
      if (lo < hi && xpar_pwrite(dst, data[i] + lo - (i * g->slice_size + at),
                                 (sz) (hi - lo), lo - stream_offset) !=
                     (sz) (hi - lo)) {
        if (reason) *reason = "writing the staged volume failed";
        ok = false; break;
      }
    }
  }
  xpar_free(data); xpar_free(rptr); xpar_free(storage);
  if (!ok) goto done;

  /*  Reassemble and strong-gate every touched slice before publication.  */
  gate = (u8 *) xpar_alloc_raw((sz) g->slice_size);
  for (i = first; i < last && ok; i++) {
    u64 sb = i * g->slice_size;
    u64 lo = MAX(stream_offset, sb), hi = MIN(end, sb + g->slice_size);
    u8 got[32];
    xpar_vset_read(s, xpar_slice_begin(g, i), gate, g->slice_size);
    if (lo < hi && xpar_pread(dst, gate + lo - sb, (sz) (hi - lo),
                              lo - stream_offset) != (sz) (hi - lo)) {
      if (reason) *reason = "reading the staged volume back failed";
      ok = false; break;
    }
    if (i + 1 == g->slice_count && xpar_slice_bytes(g, i) < g->slice_size)
      xpar_memset(gate + xpar_slice_bytes(g, i), 0,
                  (sz) (g->slice_size - xpar_slice_bytes(g, i)));
    if (tags->tag_len && tags->slice_tag) {
      if (key) xpar_slice_tag_keyed(sd, i, gate, key->k_slice, got,
                                    tags->tag_len);
      else     xpar_slice_tag(sd, i, gate, got, tags->tag_len);
      if (!xpar_blake3_tag_equal(got,
            tags->slice_tag + i * tags->tag_len, tags->tag_len)) {
        if (reason) *reason = "a reconstructed slice failed its strong tag";
        ok = false;
      }
    } else if (!tags->slice_crc ||
               xpar_crc32c(0, gate, (sz) g->slice_size) !=
                 tags->slice_crc[i]) {
      if (reason) *reason = "a reconstructed slice failed its CRC";
      ok = false;
    }
  }
  xpar_free(gate);

done:
  xpar_codec_plan_free(plan); xpar_codec_free(codec); xpar_free(present);
no_codec:
  for (u32 q = 0; q < rv_count; q++) owned_close(&rv[q]);
  xpar_free(rv);
  xpar_free((void *) rec); xpar_free(rpresent);
  if (ok && reason) *reason = NULL;
  return ok;
}

static xpar_file * owned_stage_open(const char * dir, char ** path) {
  char * stem = xpar_path_join(dir, ".xpar-repair-");
  xpar_file * f = xpar_stage_open(stem, XPAR_O_RDWR | XPAR_O_NOFOLLOW, 1,
                                  path);
  int err = xpar_errno();
  xpar_free(stem);
  if (!f)
    FATAL_IO("Cannot create a secure repair stage in '%s': %s.", dir,
             xpar_strerror(err));
  return f;
}

static void owned_publish_split_stage(const xpar_vset * s, xpar_file * stage,
                                      u64 stage_off, u64 stream_off,
                                      u64 len, u8 * buf, u64 cap) {
  const xpar_layt * l = xpar_vset_layt(s);
  const char * dir = xpar_vset_dir(s);
  u64 copied = 0;
  while (copied < len) {
    const xpar_vol * v = NULL;
    xpar_file * dst;
    char * path;
    u64 in_stream = stream_off + copied, part, done = 0;
    u32 q;
    for (q = 0; q < l->count; q++)
      if (l->vol[q].kind == XPAR_VOL_DATA &&
          in_stream >= l->vol[q].stream_offset &&
          in_stream - l->vol[q].stream_offset < l->vol[q].byte_length) {
        v = &l->vol[q]; break;
      }
    FATAL_UNLESS("The split layout has no data volume for stream offset "
                 "%" PRIu64 ".", v != NULL, in_stream);
    part = MIN(len - copied,
               v->byte_length - (in_stream - v->stream_offset));
    path = xpar_path_join(dir, v->name);
    dst = xpar_open(path, XPAR_O_RDWR | XPAR_O_CREAT | XPAR_O_NOFOLLOW);
    if (!dst) FATAL_PERROR(path);
    while (done < part) {
      u64 take = MIN(part - done, cap);
      if (xpar_pread(stage, buf, (sz) take,
                     stage_off + copied + done) != (sz) take ||
          xpar_pwrite(dst, buf, (sz) take,
                      in_stream - v->stream_offset + done) != (sz) take)
        FATAL_IO("Publishing repaired split data to '%s' failed.", path);
      done += take;
    }
    if (xpar_fsync(dst) != 0)
      FATAL_IO("Flushing repaired split data volume '%s' failed.", path);
    xpar_xclose(dst); xpar_free(path);
    copied += part;
  }
}

/*  Owned-layout repair summary.  */
typedef struct {
  u64  cells_bad;
  u64  slices_rebuilt;
  u64  bytes_rebuilt;
  u64  volumes_rebuilt;      /*  Data volumes written back in place.  */
  u64  volumes_rewritten;    /*  Damaged volumes replaced by a substitute.  */
  u64  volumes_relengthed;   /*  Same-name volumes trimmed or extended.  */
  u64  recovery_regen;       /*  Recovery slices re-encoded from the data.  */
  u64  recovery_volumes;     /*  Recovery volumes rewritten to carry them.  */
  u64  index_regen;          /*  Index volumes rebuilt from replicas.  */
  u64  names_restored;       /*  Volumes put back under their name.  */
  u64  stale_regen;          /*  Volumes rewritten to the index's copies.  */
  u64  ragged_trimmed;       /*  Volumes cut back to their last packet.  */
  u64  volumes_dropped;      /*  Volumes rewritten from packet replicas.  */
  u64  meta_skip[RP_META_CLASSES];  /*  See rp_meta_name.  */
  bool inner_corrected;
} owned_acct;

static void owned_write_tree(const xpar_options *, xpar_vset *,
                             xpar_file *, const u64 *, owned_acct *);

/*  Decode owned cells into a sparse stage. Recovery rows stay mapped and
    column width shrinks until the complete footprint fits -m.  */
static int owned_repair_stream(const xpar_options * o, xpar_vset * s,
                               int checked, const u8 * const * rec,
                               const u8 * rpresent, u64 rtop,
                               owned_acct * acct) {
  const xpar_setd * sd = xpar_vset_setd(s);
  const xpar_geom * g = xpar_vset_geom(s);
  const xpar_erasures * er = xpar_vset_erasures(s);
  const xpar_tags * tags = xpar_vset_tags(s);
  const xpar_layt * l = xpar_vset_layt(s);
  u64 * slot, touched = 0, i, col, sub;
  u8 * present, * storage, ** data, * io;
  /* Read-only views into mapped recovery volumes. */
  const u8 ** rptr;
  xpar_codec * codec;
  xpar_file * stage;
  char * stage_path = NULL;
  u64 depth = xpar_erasures_max_depth(er);
  u64 budget = o->memory ? o->memory : xpar_plan_default_memory();
  u64 chunk, max_cell, pool_bytes;
  const u8 * arm_plain = NULL;
  u64 arm_len = 0, strm_off = 0;
  xpar_armour_params ap;
  const char * arm_path = NULL;
  bool armoured = sd->layout == XPAR_LAYOUT_ARMOURED;
  bool ok = true;

  if (checked == XPAR_EXIT_UNREPAIRABLE || depth > rtop)
    return XPAR_EXIT_UNREPAIRABLE;
  if (armoured)
    FATAL_UNLESS("The armoured archive has no recoverable plaintext.",
                 xpar_vset_armoured(s, &arm_plain, &arm_len, &strm_off,
                                    &ap, &arm_path));
  slot = (u64 *) xpar_alloc_raw((sz) g->slice_count * sizeof(u64));
  for (i = 0; i < g->slice_count; i++) {
    bool any = false;
    for (col = 0; col < g->cells_per_slice; col++)
      if (xpar_cell_bad(er, i, (u32) col)) { any = true; break; }
    slot[i] = any ? touched++ : UINT64_MAX;
  }
  acct->cells_bad = er->bad_count;
  acct->slices_rebuilt = touched;
  for (i = 0; i < g->slice_count; i++)
    if (slot[i] != UINT64_MAX) acct->bytes_rebuilt += xpar_slice_bytes(g, i);
  if (!touched) {
    if (armoured && (xpar_vset_inner_corrected(s) ||
                     xpar_vset_archive_stale(s))) {
      if (o->dest == XPAR_DEST_BACKUP) owned_backup_path(arm_path);
      /*  The rebuilt archive is renamed over this name; a host that locks
          a mapped file needs the image gone first.  */
      xpar_vset_release_volume(s, arm_path);
      xpar_garm_write_patched(arm_path, &ap, arm_plain, arm_len, strm_off,
                              g->stream_length, NULL, NULL,
                              g->slice_count, g->slice_size);
      acct->inner_corrected = true;
    }
    xpar_free(slot);
    return XPAR_EXIT_OK;
  }
  FATAL_UNLESS("The staged split repair is too large for file offsets.",
               !g->slice_size || touched <= UINT64_MAX / g->slice_size);
  stage = owned_stage_open(xpar_vset_dir(s), &stage_path);

  max_cell = g->cell_bytes ? g->cell_bytes : g->slice_size;
  chunk = MIN(max_cell, (u64) 1 << 20);
  chunk &= ~(u64) 63;
  if (!chunk) chunk = 64;
  while (chunk >= 64 &&
         xpar_codec_decode_footprint_axis(
           sd->codec, sd->field_log2, g->slice_count, rtop,
           sd->recovery_axis_log2, (sz) chunk) >
           budget)
    chunk = (chunk / 2) & ~(u64) 63;
  if (chunk < 64)
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "Owned-layout repair needs at least %" PRIu64 " bytes for one "
               "64-byte decode column; raise -m.",
               xpar_codec_decode_footprint_axis(
                 sd->codec, sd->field_log2, g->slice_count, rtop,
                 sd->recovery_axis_log2, 64));
  FATAL_UNLESS("The repair column allocation overflows this host.",
               !g->slice_count || chunk <= (u64) (sz) -1 / g->slice_count);
  pool_bytes = g->slice_count * chunk;
  storage = (u8 *) xpar_alloc_aligned((sz) pool_bytes, 64);
  data = (u8 **) xpar_alloc_raw((sz) g->slice_count * sizeof(u8 *));
  rptr = (const u8 **) xpar_alloc_raw((sz) MAX(rtop, 1) * sizeof(u8 *));
  present = (u8 *) xpar_calloc((sz) g->slice_count, 1);
  io = (u8 *) xpar_alloc_raw((sz) chunk);
  for (i = 0; i < g->slice_count; i++) data[i] = storage + i * chunk;

  /*  The final strong gate catches changes to cells copied into the stage.  */
  for (i = 0; i < g->slice_count; i++) if (slot[i] != UINT64_MAX)
    for (sub = 0; sub < g->slice_size; sub += chunk) {
      u64 n = MIN(chunk, g->slice_size - sub);
      (void) xpar_vset_read(s, xpar_slice_begin(g, i) + sub, io, n);
      if (xpar_pwrite(stage, io, (sz) n,
                      slot[i] * g->slice_size + sub) != (sz) n)
        FATAL_IO("Writing the owned repair stage failed.");
    }

  codec = xpar_codec_new_axis(sd->codec, sd->field_log2, g->slice_count,
                              rtop, sd->recovery_axis_log2);
  for (col = 0; col < g->cells_per_slice && ok; col++) {
    xpar_codec_plan * plan;
    xpar_codec_status st;
    u64 y = xpar_cell_size(g, (u32) col);
    u64 at = col * (g->cell_bytes ? g->cell_bytes : g->slice_size);
    bool any = false;
    for (i = 0; i < g->slice_count; i++) {
      present[i] = !xpar_cell_bad(er, i, (u32) col);
      if (!present[i]) any = true;
    }
    if (!any) continue;
    plan = xpar_codec_plan_new(codec, present, rpresent, &st);
    if (!plan || st != XPAR_CODEC_OK) {
      xpar_codec_plan_free(plan); ok = false; break;
    }
    for (sub = 0; sub < y && ok; sub += chunk) {
      u64 n = MIN(chunk, y - sub);
      for (i = 0; i < g->slice_count; i++) {
        if (!present[i]) xpar_memset(data[i], 0, (sz) n);
        else if (!xpar_vset_read(s, xpar_slice_begin(g, i) + at + sub,
                                 data[i], n)) { ok = false; break; }
      }
      if (!ok) break;
      for (i = 0; i < rtop; i++)
        rptr[i] = rpresent[i] ? rec[i] + at + sub : NULL;
      if (xpar_codec_plan_apply(plan, data, rptr, (sz) n) != XPAR_CODEC_OK) {
        ok = false; break;
      }
      for (i = 0; i < g->slice_count; i++) if (!present[i] &&
                                                xpar_pwrite(stage, data[i],
                                                  (sz) n,
                                                  slot[i] * g->slice_size +
                                                  at + sub) != (sz) n) {
        ok = false; break;
      }
    }
    xpar_codec_plan_free(plan);
  }
  xpar_codec_free(codec);
  if (ok && xpar_fsync(stage) != 0) ok = false;

  /*  Stream the slice gate; B3-subtree sets require chaining-value output.  */
  for (i = 0; i < g->slice_count && ok; i++) if (slot[i] != UINT64_MAX) {
    xpar_blake3_t h;
    u32 crc = 0;
    u8 got[32];
    bool subtree = tags->tag_len &&
      (sd->required_features & XPAR_FEAT_B3_SUBTREE) != 0;
    if (tags->tag_len) {
      if (subtree)
        xpar_blake3_subtree_stream_init(&h,
          xpar_vset_key(s) ? xpar_vset_key(s)->k_slice : NULL,
          xpar_slice_begin(g, i) / XPAR_BLAKE3_CHUNK_LEN);
      else if (xpar_vset_key(s))
        xpar_blake3_init_keyed(&h, xpar_vset_key(s)->k_slice);
      else xpar_blake3_init(&h);
    }
    for (sub = 0; sub < g->slice_size; sub += chunk) {
      u64 n = MIN(chunk, g->slice_size - sub);
      if (xpar_pread(stage, io, (sz) n,
                     slot[i] * g->slice_size + sub) != (sz) n) {
        ok = false; break;
      }
      if (tags->tag_len) xpar_blake3_update(&h, io, (sz) n);
      else crc = xpar_crc32c(crc, io, (sz) n);
    }
    if (!ok) break;
    if (tags->tag_len) {
      if (subtree) xpar_blake3_subtree_stream_final(&h, got, tags->tag_len);
      else xpar_blake3_final(&h, got, tags->tag_len);
      if (!xpar_blake3_tag_equal(got,
            tags->slice_tag + i * tags->tag_len, tags->tag_len)) ok = false;
    } else if (crc != tags->slice_crc[i]) ok = false;
  }

  if (ok && o->dest == XPAR_DEST_BACKUP) {
    if (armoured) {
      owned_backup_path(arm_path);
    } else for (u32 q = 0; q < l->count; q++)
        if (l->vol[q].kind == XPAR_VOL_DATA && l->vol[q].name) {
          char * path = xpar_path_vol(xpar_vset_dir(s), l->vol[q].name);
          owned_backup_path(path); xpar_free(path);
        }
  }
  if (ok && o->dest == XPAR_DEST_TO) {
    owned_write_tree(o, s, stage, slot, acct);
  } else if (ok && armoured) {
    /*  As above: release the image before the rebuilt archive replaces it.  */
    xpar_vset_release_volume(s, arm_path);
    xpar_garm_write_patched(arm_path, &ap, arm_plain, arm_len, strm_off,
                            g->stream_length, stage, slot,
                            g->slice_count, g->slice_size);
  } else if (ok) {
    for (i = 0; i < g->slice_count; i++) if (slot[i] != UINT64_MAX)
      owned_publish_split_stage(s, stage, slot[i] * g->slice_size,
                                i * g->slice_size,
                                xpar_slice_bytes(g, i), io, chunk);
    /*  Count every volume receiving rebuilt slices.  */
    if (l) for (u32 q = 0; q < l->count; q++) {
      if (l->vol[q].kind != XPAR_VOL_DATA) continue;
      for (i = 0; i < g->slice_count; i++) {
        u64 at = i * g->slice_size;
        if (slot[i] == UINT64_MAX) continue;
        if (at + xpar_slice_bytes(g, i) > l->vol[q].stream_offset &&
            at < l->vol[q].stream_offset + l->vol[q].byte_length) {
          acct->volumes_rebuilt++;  break;
        }
      }
    }
  }
  xpar_xclose(stage);
  xpar_remove(stage_path);
  xpar_free(stage_path); xpar_free(slot); xpar_free(present);
  xpar_free(io); xpar_free(rptr); xpar_free(data);
  xpar_free_aligned(storage);
  if (!ok && xpar_vset_io_errors(s)) return XPAR_EXIT_IO;
  return ok ? XPAR_EXIT_OK : XPAR_EXIT_UNREPAIRABLE;
}

/*  `repair --to` materialises the protected tree without publishing the
    reconstructed owned stream.  */
static void owned_write_tree(const xpar_options * o, xpar_vset * s,
                             xpar_file * staged, const u64 * slot,
                             owned_acct * acct) {
  const xpar_manifest * m = xpar_vset_manifest(s);
  const xpar_geom * g = xpar_vset_geom(s);
  xpar_chain ancestry;
  xpar_chain metadata;
  xpar_manifest * ancestor_mf = NULL;
  xpar_manifest metadata_mf;
  u8 * ancestor_loaded = NULL;
  u32 * metadata_owner = NULL;
  xpar_posix_rec ** metadata_posix;
  u32 * metadata_posix_count;
  u32 metadata_g;
  bool have_ancestry = o->repair_chain_stage != NULL;
  xpar_nameidx nix;
  xpar_stat_t st;
  rp meta;
  u32 i;
  xpar_memset(&meta, 0, sizeof meta);
  meta.o = o;  meta.quiet = o->quiet;
  if (have_ancestry) {
    xpar_gchain_load(o, &ancestry);
    ancestor_mf = (xpar_manifest *)
      xpar_calloc(ancestry.gen_count ? ancestry.gen_count : 1,
                  sizeof(*ancestor_mf));
    ancestor_loaded = (u8 *)
      xpar_calloc(ancestry.gen_count ? ancestry.gen_count : 1, 1);
  }
  FATAL_UNLESS("repair --to needs a destination directory.", o->to_dir);
  if (xpar_mkdir_p(o->to_dir, 0777) != 0) {
    int err = xpar_errno();
    if (xpar_lstat(o->to_dir, &st) != 0)
      FATAL_IO("Cannot create '%s': %s.", o->to_dir, xpar_strerror(err));
  }
  FATAL_UNLESS("The repair destination '%s' is a symbolic link; refusing "
               "to write through it.",
               xpar_lstat(o->to_dir, &st) == 0 && !st.is_symlink, o->to_dir);
  xpar_gchain_load(o, &metadata);
  metadata_g = xpar_gchain_select(&metadata,
                                  o->gen_count ? &o->gens[0] : NULL);
  xpar_gchain_manifest(&metadata, metadata_g, &metadata_mf,
                       &metadata_owner);
  FATAL_UNLESS("The repair and metadata readers selected different trees.",
               metadata_mf.count == m->count);
  metadata_posix = (xpar_posix_rec **) xpar_calloc(
    metadata.gen_count, sizeof *metadata_posix);
  metadata_posix_count = (u32 *) xpar_calloc(metadata.gen_count,
                                              sizeof(u32));
  for (i = 0; i < metadata.gen_count; i++)
    metadata_posix_count[i] = xpar_gchain_posix(&metadata, i,
                                                &metadata_posix[i]);
  rp_tree_preflight(o, m, o->to_dir);

  for (i = 0; i < m->count; i++) if (m->entry[i].entry_type ==
                                      XPAR_ENTRY_DIR) {
    xpar_path_status why;
    char * p = xpar_path_resolve(o->to_dir, m->entry[i].name,
                                 m->entry[i].name_len, 0, &why);
    FATAL_UNLESS("Refusing repair output '%.*s': %s.", p != NULL,
                 (int) m->entry[i].name_len, m->entry[i].name,
                 xpar_path_reason(why));
    if (xpar_mkdir_p(p, 0777) != 0) {
      int err = xpar_errno();
      if (xpar_lstat(p, &st) != 0)
        FATAL_IO("Cannot create '%s': %s.", p, xpar_strerror(err));
    }
    xpar_free(p);
  }
  for (i = 0; i < m->count; i++) if (m->entry[i].entry_type ==
                                      XPAR_ENTRY_REGULAR) {
    const xpar_entry * e = &m->entry[i];
    xpar_path_status why;
    char * p = xpar_path_resolve(o->to_dir, e->name, e->name_len, 0, &why);
    xpar_file * f;
    char * stage = NULL;
    xpar_blake3_t h;
    u8 got[32];
    u8 * io = (u8 *) xpar_alloc_raw(1u << 16);
    u32 k;
    FATAL_UNLESS("Refusing repair output '%.*s': %s.", p != NULL,
                 (int) e->name_len, e->name, xpar_path_reason(why));
    { char * d = xpar_path_dir(p);
      if (xpar_mkdir_p(d, 0777) != 0) {
        int err = xpar_errno();
        if (xpar_lstat(d, &st) != 0)
          FATAL_IO("Cannot create '%s': %s.", d, xpar_strerror(err));
      }
      xpar_free(d); }
    f = rp_tree_stage(p, &stage);
    if (xpar_vset_key(s)) xpar_blake3_init_keyed(&h,
                                                  xpar_vset_key(s)->k_file);
    else xpar_blake3_init(&h);
    for (k = 0; k < e->extent_count; k++) {
      u64 at = e->extents[k].stream_offset;
      u64 left = e->extents[k].length;
      while (left) {
        bool local = at >= g->stream_base &&
                     at - g->stream_base < g->stream_length;
        u64 rel = local ? at - g->stream_base : 0;
        u64 slice = local ? rel / g->slice_size : 0;
        u64 in = local ? rel % g->slice_size : 0;
        u64 take = MIN(left, (u64) 1 << 16);
        const u8 * bytes;
        if (local) take = MIN(take, g->slice_size - in);
        if (!local) {
          FATAL_UNLESS("Entry '%.*s' depends on an ancestor stream; "
                       "repair it with --chain --to so every owning "
                       "generation is available.",
                       have_ancestry &&
                       owned_chain_read(&ancestry, ancestor_mf,
                                        ancestor_loaded,
                                        o->repair_chain_stage, at, io,
                                        &take),
                       (int) e->name_len, e->name);
          bytes = io;
        } else if (slot && slot[slice] != UINT64_MAX) {
          if (xpar_pread(staged, io, (sz) take,
                         slot[slice] * g->slice_size + in) != (sz) take)
            FATAL_IO("Reading staged repair data for '%.*s' failed.",
                     (int) e->name_len, e->name);
          bytes = io;
        } else {
          if (!xpar_vset_read(s, at, io, take))
            FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                       "An unchanged extent of '%.*s' became unreadable "
                       "during repair.", (int) e->name_len, e->name);
          bytes = io;
        }
        xpar_xwrite(f, bytes, (sz) take);
        xpar_blake3_update(&h, bytes, (sz) take);
        at += take; left -= take;
      }
    }
    if (xpar_fsync(f) != 0) FATAL_IO("Cannot flush '%s'.", p);
    xpar_xclose(f);
    xpar_blake3_final(&h, got, sizeof got);
    FATAL_UNLESS("Repaired output '%.*s' failed its content hash.",
                 xpar_ct_equal(got, e->content_hash, sizeof got),
                 (int) e->name_len, e->name);
    rp_publish_tree_stage(o, stage, p);
    xpar_free(stage);
    xpar_free(io);
    xpar_free(p);
  }
  for (i = 0; i < m->count; i++) if (m->entry[i].entry_type ==
                                      XPAR_ENTRY_SYMLINK) {
    const xpar_entry * e = &m->entry[i];
    xpar_path_status why;
    char * p = xpar_path_resolve(o->to_dir, e->name, e->name_len, 0, &why);
    char * target;
    FATAL_UNLESS("Refusing repair output '%.*s': %s.", p != NULL,
                 (int) e->name_len, e->name, xpar_path_reason(why));
    target = xpar_strndup((const char *) e->extra, e->extra_len);
    {
      int err = 0;
      char * link_stage = rp_symlink_aside(target, p, &err);
      if (!link_stage)
        FATAL_IO("Cannot stage symbolic link '%s': %s.", p,
                 xpar_strerror(err));
      rp_publish_tree_stage(o, link_stage, p);
      xpar_free(link_stage);
    }
    if (xpar_lstat(p, &st) != 0)
      FATAL_IO("Cannot publish symbolic link '%s': %s.", p,
               xpar_strerror(xpar_errno()));
    xpar_free(target); xpar_free(p);
  }
  xpar_nameidx_build(m, &nix);
  for (i = 0; i < m->count; i++) if (m->entry[i].entry_type ==
                                      XPAR_ENTRY_HARDLINK) {
    const xpar_entry * e = &m->entry[i];
    i64 t = xpar_link_target(m, &nix, i);
    xpar_path_status why, src_why = XPAR_PATH_OK;
    char * p, * src;
    FATAL_UNLESS("Hard-link output '%.*s' has no canonical target.", t >= 0,
                 (int) e->name_len, e->name);
    p = xpar_path_resolve(o->to_dir, e->name, e->name_len, 0, &why);
    FATAL_UNLESS("Refusing repair output '%.*s': %s.", p != NULL,
                 (int) e->name_len, e->name, xpar_path_reason(why));
    src = xpar_path_resolve(o->to_dir, m->entry[t].name,
                            m->entry[t].name_len, 0, &src_why);
    FATAL_UNLESS("Unsafe hard-link target '%.*s': %s.",
                 src != NULL, (int) m->entry[t].name_len, m->entry[t].name,
                 xpar_path_reason(src_why));
    {
      int err = 0;
      char * link_stage = rp_link_aside(src, p, &err);
      if (link_stage) {
        rp_publish_tree_stage(o, link_stage, p);
        xpar_free(link_stage);
        xpar_free(src); xpar_free(p);
        continue;
      }
    }
    {
      char * stage = NULL;
      xpar_file * in = xpar_open(src, XPAR_O_RDONLY);
      int err = xpar_errno();
      xpar_file * out;
      u8 buf[65536];
      u64 at = 0;
      if (!in) FATAL_IO("Cannot read canonical hard-link '%s': %s.", src,
                        xpar_strerror(err));
      out = rp_tree_stage(p, &stage);
      while (at < e->length) {
        sz take = (sz) MIN(e->length - at, (u64) sizeof buf);
        if (xpar_pread(in, buf, take, at) != take)
          FATAL_IO("Reading canonical hard-link '%s' failed.", src);
        xpar_xwrite(out, buf, take); at += take;
      }
      if (xpar_fsync(out) != 0)
        FATAL_IO("Flushing hard-link copy '%s' failed.", p);
      xpar_xclose(in); xpar_xclose(out);
      rp_publish_tree_stage(o, stage, p);
      rp_meta_skip(meta.meta_skip, o, e, RP_META_COPY,
                   "the destination cannot create the hard link");
      xpar_fprintf(xpar_stderr, "xpar: %.*s: materialised-as-copy.\n",
                   (int) e->name_len, e->name);
      xpar_free(stage);
    }
    xpar_free(src); xpar_free(p);
  }
  /*  Apply directory metadata deepest-first using reversed name order.  */
  {
    u32 caps = xpar_fs_caps(o->to_dir);
    for (i = 0; i < m->count; i++) {
      u32 idx = nix.order[m->count - 1 - i], owner = metadata_owner[idx];
      const xpar_entry * e = &m->entry[idx];
      const xpar_posix_rec * pr = NULL;
      xpar_path_status why;
      bool link = e->entry_type == XPAR_ENTRY_SYMLINK;
      char * p = xpar_path_resolve(o->to_dir, e->name, e->name_len,
                                   link ? XPAR_PATH_LEAF_LINK : 0, &why);
      if (!p) continue;
      if (link && !(caps & XPAR_FS_NOFOLLOW)) {
        rp_meta_skip(meta.meta_skip, o, e, RP_META_SYMLINK,
                     "the host has no symlink-safe metadata call");
        xpar_free(p); continue;
      }
      if (e->posix_index != XPAR_ABSENT_U32 &&
          owner < metadata.gen_count &&
          e->posix_index < metadata_posix_count[owner])
        pr = &metadata_posix[owner][e->posix_index];
      rp_apply_meta(&meta, e, pr, p, link, caps);
      xpar_free(p);
    }
  }
  xpar_nameidx_free(&nix);
  for (i = 0; i < RP_META_CLASSES; i++)
    acct->meta_skip[i] += meta.meta_skip[i];
  for (i = 0; i < metadata.gen_count; i++)
    xpar_gchain_posix_free(metadata_posix[i], metadata_posix_count[i]);
  xpar_free(metadata_posix);  xpar_free(metadata_posix_count);
  xpar_free(metadata_owner);
  xpar_manifest_free(&metadata_mf);
  xpar_gchain_free(&metadata);
  if (have_ancestry) {
    for (i = 0; i < ancestry.gen_count; i++)
      if (ancestor_loaded[i]) xpar_manifest_free(&ancestor_mf[i]);
    xpar_free(ancestor_mf);
    xpar_free(ancestor_loaded);
    xpar_gchain_free(&ancestry);
  }
}

static int repair_owned(const xpar_options * o, xpar_vset * s, int checked,
                        owned_acct * acct) {
  const xpar_setd * sd = xpar_vset_setd(s);
  const xpar_erasures * er = xpar_vset_erasures(s);
  const xpar_layt * l = xpar_vset_layt(s);
  u64 rtop = 0, i;
  const u8 ** rec;
  u8 * rpresent;
  owned_vol * rv = NULL;
  u32 rv_count = 0;
  int out;

  if (xpar_vset_authenticated(s) && !xpar_vset_key(s))
    FATAL_CODE(XPAR_EXIT_AUTH,
               "Repairing an authenticated set requires --auth-key=FILE; "
               "keyless access is read-only.");
  if (checked == XPAR_EXIT_UNREPAIRABLE) return checked;
  if (checked == XPAR_EXIT_IO || xpar_vset_io_errors(s)) return XPAR_EXIT_IO;
  /*  A clean owned set needs no codec storage; inner corrections still
      require publication.  */
  if (checked == XPAR_EXIT_OK && o->dest != XPAR_DEST_TO &&
      !(sd->layout == XPAR_LAYOUT_ARMOURED &&
        xpar_vset_inner_corrected(s)))
    return XPAR_EXIT_OK;

  if (l) for (u32 q = 0; q < l->count; q++)
    if (l->vol[q].kind == XPAR_VOL_RECOVERY &&
        l->vol[q].recovery_first + l->vol[q].byte_length > rtop)
      rtop = l->vol[q].recovery_first + l->vol[q].byte_length;
  if (sd->layout == XPAR_LAYOUT_ARMOURED) rtop = xpar_vset_recovery(s);
  if (!rtop && er->bad_count) return XPAR_EXIT_UNREPAIRABLE;
  rec = (const u8 **) xpar_calloc(rtop ? (sz) rtop : 1, sizeof(u8 *));

  if (sd->layout == XPAR_LAYOUT_ARMOURED) {
    for (i = 0; i < rtop; i++) { u64 n; rec[i] = xpar_vset_rcvs(s, i, &n); }
  } else if (l) {
    rv = (owned_vol *) xpar_calloc(l->count ? l->count : 1, sizeof(*rv));
    for (u32 q = 0; q < l->count; q++)
      if (l->vol[q].kind == XPAR_VOL_RECOVERY) {
      char * path = xpar_path_vol(xpar_vset_dir(s), l->vol[q].name);
      if (owned_open(&rv[rv_count], path)) {
        owned_scan_recovery(s, &rv[rv_count], rec, rtop); rv_count++;
      }
      xpar_free(path);
    }
  }

  rpresent = (u8 *) xpar_calloc(rtop ? (sz) rtop : 1, 1);
  for (i = 0; i < rtop; i++) rpresent[i] = rec[i] != NULL;
  out = owned_repair_stream(o, s, checked, rec, rpresent, rtop, acct);
  if (out == XPAR_EXIT_OK && sd->layout == XPAR_LAYOUT_SPLIT &&
      o->dest != XPAR_DEST_TO) {
    const char * why = NULL;
    u64 failed = 0;
    if (!xpar_vset_rewrite_substituted(s, &acct->volumes_rewritten,
                                       &acct->volumes_relengthed, &failed,
                                       &why)) {
      if (!o->quiet)
        xpar_fprintf(xpar_stderr,
                     "xpar: %" PRIu64 " data volume%s could not be "
                     "rewritten: %s\n", failed, PLURAL(failed),
                     why ? why : "unknown error");
      out = XPAR_EXIT_UNREPAIRABLE;
    }
  }
  if (out != XPAR_EXIT_OK && xpar_vset_io_errors(s)) out = XPAR_EXIT_IO;
  for (u32 q = 0; q < rv_count; q++) owned_close(&rv[q]);
  xpar_free(rv); xpar_free(rpresent); xpar_free((void *) rec);
  return out;
}

/*  Regenerate missing recovery slices from repaired data.  */
static u64 repair_regen_recovery(const xpar_options * o, u64 * volumes) {
  const char * why = NULL;
  u64 done;
  if (o->dest == XPAR_DEST_TO) return 0;
  /*  A dry run only counts what a real one would rewrite.  */
  done = xpar_gen_regen_recovery(o, volumes, &why, o->dry_run);
  if (!done && why && !o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: recovery slices could not be regenerated: %s\n", why);
  return done;
}

/*  Drop volume mappings so a regenerated volume can be renamed over them;
    nothing below reads volume-backed memory again.  */
static void rp_release_vols(rp * r) {
  u32 i;
  for (i = 0; i < r->vol_count; i++) xpar_volimg_close(&r->vol[i]);
  xpar_free((void *) r->rec);  r->rec = NULL;
  xpar_free(r->rec_present);   r->rec_present = NULL;
  r->rec_avail = 0;
}

/*  Put a volume found under another name back where the layout says.  */
static u64 repair_restore_names(const xpar_options * o, xpar_vset * s) {
  const char * why = NULL;
  u64 n = 0, failed = 0;
  if (o->dry_run || o->dest == XPAR_DEST_TO) return 0;
  if (!xpar_vset_restore_names(s, &n, &failed, &why) && !o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: %" PRIu64 " volume%s could not be "
                 "restored to the recorded name: %s\n", failed,
                 PLURAL(failed), why ? why : "unknown error");
  if (n && !o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: restored %" PRIu64 " volume%s to the "
                 "name the layout records\n", n, PLURAL(n));
  return n;
}

/*  Cut trailing bytes that are not packets off the set's volumes.  */
static u64 repair_trim_ragged(const xpar_options * o, xpar_vset * s) {
  const char * why = NULL;
  u64 n = 0, failed = 0;
  if (o->dry_run || o->dest == XPAR_DEST_TO) return 0;
  if (!xpar_vset_volumes_ragged(s)) return 0;
  if (!xpar_vset_trim_ragged(s, &n, &failed, &why) && !o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: %" PRIu64 " nonconforming volume%s "
                 "could not be trimmed: %s\n", failed, PLURAL(failed),
                 why ? why : "unknown error");
  if (n && !o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: trimmed %" PRIu64 " nonconforming "
                 "volume%s back to its last packet\n", n, PLURAL(n));
  return n;
}

/*  Recreate index volumes the layout names but disk has lost.  */
static u64 repair_regen_index(const xpar_options * o) {
  const char * why = NULL;
  u64 vols = 0, done;
  if (o->dest == XPAR_DEST_TO) return 0;
  done = xpar_gen_regen_index(o, &vols, &why, o->dry_run);
  if (why && !o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: index volumes could not be recreated: %s\n", why);
  return done;
}

static u64 repair_rewrite_stale(const xpar_options * o) {
  const char * why = NULL;
  u64 vols = 0, n;
  if (o->dest == XPAR_DEST_TO) return 0;
  n = xpar_gen_rewrite_stale(o, &vols, &why, o->dry_run);
  if (why && !o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: could not rewrite all stale volumes: "
                 "%s\n", why);
  return n;
}

/*  Rewrite stale volumes from intact packet replicas.  */
static u64 repair_rewrite_dropped(const xpar_options * o, xpar_vset * s) {
  const char * why = NULL;
  u64 n = 0, failed = 0;
  if (o->dry_run || o->dest == XPAR_DEST_TO) return 0;
  if (!xpar_vset_volumes_dropped(s)) return 0;
  if (!xpar_vset_rewrite_dropped(s, &n, &failed, &why)) {
    /*  Recovery and index regeneration handle lost packets separately.  */
    if (!o->quiet && !xpar_vset_recovery_bad(s) &&
        !xpar_vset_volumes_ragged(s)) {
      xpar_fprintf(xpar_stderr,
                   "xpar: %" PRIu64 " stale volume%s could not be "
                   "rewritten: %s\n", failed, PLURAL(failed),
                   why ? why : "unknown error");
      /*  Inner-coded regions need no replicas.  */
      if (xpar_vset_inner_corrected(s))
        xpar_fprintf(xpar_stderr,
                     "xpar: inner-code corrections remain; run "
                     "`xpar scrub --rewrite` to persist them\n");
    }
  }
  if (n && !o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: rewrote %" PRIu64 " volume%s from "
                 "intact packet replicas\n", n, PLURAL(n));
  return n;
}

static void owned_json_repair(xpar_json * js, u8 layout, bool changed,
                              const owned_acct * a,
                              const xpar_options * o) {
  xpar_json_begin(js, "repair");
  xpar_json_str(js, "layout", xpar_layout_name(layout));
  xpar_json_bool(js, "changed", changed);
  xpar_json_str(js, "destination", o->dest == XPAR_DEST_TO ? "tree" : "set");
  xpar_json_u64(js, "cells_damaged", a->cells_bad);
  xpar_json_u64(js, "slices_rebuilt", a->slices_rebuilt);
  xpar_json_u64(js, "bytes_rebuilt", a->bytes_rebuilt);
  xpar_json_u64(js, "volumes_rebuilt", a->volumes_rebuilt);
  xpar_json_u64(js, "volumes_rewritten", a->volumes_rewritten);
  xpar_json_u64(js, "volumes_relengthed", a->volumes_relengthed);
  xpar_json_u64(js, "volumes_trimmed", a->ragged_trimmed);
  xpar_json_u64(js, "volumes_dropped_rewritten", a->volumes_dropped);
  xpar_json_u64(js, "recovery_regenerated", a->recovery_regen);
  xpar_json_u64(js, "recovery_volumes", a->recovery_volumes);
  xpar_json_u64(js, "index_volumes_recreated", a->index_regen);
  xpar_json_u64(js, "volumes_restored", a->names_restored);
  xpar_json_u64(js, "volumes_stale_rewritten", a->stale_regen);
  xpar_json_bool(js, "inner_corrected", a->inner_corrected);
  xpar_json_end(js);
}

int xpar_op_repair(const xpar_options * o) {
  rp r;
  u32 i, chunk;
  u64 depth;
  u64 pre_restored = 0, pre_trimmed = 0, pre_dropped = 0;
  xpar_progress_t pg;
  xpar_plan pl;
  int rc = XPAR_EXIT_OK;
  bool content_ok;
  xpar_vset * owned = NULL;
  bool walk = o->chain, partial = false;

  /*  Repair unselected ancestry oldest first.  */
  if (!walk && !o->gen_count) {
    owned = xpar_vset_open(o);
    if (xpar_vset_setd(owned)->generation) {
      xpar_vset_close(owned);
      owned = NULL;
      walk = true;
    }
  }

  if (walk) {
    xpar_chain c;
    xpar_options metadata = *o;
    xpar_json chain_js;
    int worst = XPAR_EXIT_OK;
    u32 g, selected, at, walked = 0, q;
    u32 * order;
    char * chain_stage = NULL;
    bool all_owned = true;
    metadata.chain_metadata_only = true;
    xpar_json_init(&chain_js, xpar_stdout, o->json);
    xpar_gchain_load(&metadata, &c);
    selected = xpar_gchain_select(&c,
                                  o->gen_count ? &o->gens[0] : NULL);
    order = (u32 *) xpar_calloc(c.gen_count ? c.gen_count : 1,
                                sizeof(*order));
    for (at = selected; at != XPAR_GEN_NONE && walked++ < c.gen_count;
         at = c.gen[at].parent) order[walked - 1] = at;
    FATAL_UNLESS("The selected generation's ancestry is cyclic.",
                 at == XPAR_GEN_NONE);
    for (q = 0; q < walked; q++)
      if (c.gen[order[q]].sd.layout == XPAR_LAYOUT_SIDECAR)
        all_owned = false;
    if (walked > 1 && all_owned && o->dest == XPAR_DEST_TO && !o->dry_run)
      chain_stage = owned_chain_stage_new(o->to_dir);

    /*  Walk the head-to-root list backwards; discovery order is not
        ancestry order.  */
    for (q = walked; q-- > 0; ) {
      xpar_options one = *o;
      xpar_genref ref;
      char id[XPAR_SET_ID_LEN * 2 + 1];
      char head_id[XPAR_SET_ID_LEN * 2 + 1];
      char * step_dir = NULL;
      int one_rc;
      g = order[q];
      xpar_gchain_genref(&c, g, &ref, id);
      one.chain = false;
      one.chain_member = true;
      one.json = false;
      if (o->json) one.quiet = true;
      one.gens = &ref;
      one.gen_count = 1;
      one.repair_head_set = true;
      {
        xpar_genref ignored;
        xpar_gchain_genref(&c, selected, &ignored, head_id);
      }
      one.repair_head_id = head_id;
      one.repair_chain_stage = chain_stage;
      if (chain_stage && g != selected) {
        step_dir = owned_chain_gen_dir(chain_stage, c.gen[g].set_id);
        if (xpar_mkdir(step_dir, 0700) != 0)
          FATAL_IO("Cannot create chain-repair generation stage '%s': %s.",
                   step_dir, xpar_strerror(xpar_errno()));
        one.to_dir = step_dir;
      }
      if (o->json) {
        xpar_json_begin(&chain_js, "set");
        xpar_json_hex(&chain_js, "set_id", c.gen[g].set_id,
                      XPAR_SET_ID_LEN);
        xpar_json_u64(&chain_js, "generation", c.gen[g].sd.generation);
        xpar_json_u64(&chain_js, "slice_size", c.gen[g].sd.slice_size);
        xpar_json_u64(&chain_js, "slices", c.gen[g].sd.data_slice_count);
        xpar_json_str(&chain_js, "layout",
                      xpar_layout_name(c.gen[g].sd.layout));
        xpar_json_end(&chain_js);
      }
      one_rc = xpar_op_repair(&one);
      if (o->json) {
        xpar_json_begin(&chain_js, "generation_result");
        xpar_json_u64(&chain_js, "generation", c.gen[g].sd.generation);
        xpar_json_i64(&chain_js, "exit", one_rc);
        xpar_json_end(&chain_js);
      }
      if (one_rc > worst) worst = one_rc;
      xpar_free(step_dir);
      /*  A staged chain cannot continue past a failed generation.  */
      if (chain_stage && one_rc >= XPAR_EXIT_UNREPAIRABLE) break;
    }
    if (chain_stage)
      owned_chain_stage_remove(&c, order, walked, selected, chain_stage);
    if (o->json)
      xpar_json_summary(&chain_js, xpar_status_word(worst), worst);
    else if (worst == XPAR_EXIT_OK && !o->quiet)
      xpar_fputs("xpar: no damage found.\n", xpar_stderr);
    xpar_free(chain_stage);
    xpar_free(order);
    xpar_gchain_free(&c);
    return worst;
  }

  {
    const xpar_setd * osd;
    if (!owned) owned = xpar_vset_open(o);
    osd = xpar_vset_setd(owned);
    if (osd->layout != XPAR_LAYOUT_SIDECAR) {
      xpar_json owned_js;
      owned_acct acct;
      u8 owned_layout = osd->layout;
      int before = xpar_vset_check(owned, o, NULL);
      int out;
      bool changed = before != XPAR_EXIT_OK;
      xpar_memset(&acct, 0, sizeof acct);
      xpar_json_init(&owned_js, xpar_stdout, o->json);
      if (o->json) xpar_vset_json_set(owned, &owned_js);
      if (o->dry_run) {
        xpar_vset_report(owned, o, before);
        acct.stale_regen = repair_rewrite_stale(o);
        acct.index_regen = repair_regen_index(o);
        acct.recovery_regen =
          repair_regen_recovery(o, &acct.recovery_volumes);
        if (o->dest != XPAR_DEST_TO)
          acct.ragged_trimmed = xpar_vset_volumes_ragged(owned);
        if (acct.index_regen || acct.recovery_regen || acct.stale_regen ||
            acct.ragged_trimmed)
          changed = true;
        if (!o->quiet && acct.ragged_trimmed)
          xpar_fprintf(xpar_stderr,
                       "xpar: %" PRIu64 " nonconforming volume%s would be "
                       "trimmed back to the last packet\n",
                       acct.ragged_trimmed, PLURAL(acct.ragged_trimmed));
        if (!o->quiet && acct.stale_regen)
          xpar_fprintf(xpar_stderr,
                       "xpar: %" PRIu64 " stale volume%s would be rewritten\n",
                       acct.stale_regen, PLURAL(acct.stale_regen));
        if (!o->quiet && acct.index_regen)
          xpar_fprintf(xpar_stderr,
                       "xpar: %" PRIu64 " index volume%s would be recreated "
                       "from packet replicas\n",
                       acct.index_regen, PLURAL(acct.index_regen));
        if (!o->quiet && acct.recovery_regen)
          xpar_fprintf(xpar_stderr,
                       "xpar: %" PRIu64 " recovery slice%s would be "
                       "regenerated in %" PRIu64 " volume%s\n",
                       acct.recovery_regen, PLURAL(acct.recovery_regen),
                       acct.recovery_volumes, PLURAL(acct.recovery_volumes));
        if (o->json) {
          owned_json_repair(&owned_js, owned_layout, changed, &acct, o);
          rp_meta_report(o, acct.meta_skip, &owned_js);
          xpar_json_summary(&owned_js, xpar_status_word(before), before);
        }
        xpar_vset_close(owned);
        if (before == XPAR_EXIT_OK && changed && o->exit_on_change)
          return XPAR_EXIT_REPAIRABLE;
        return before;
      }
      /*  --to extracts an intact stream without rewriting the source set.  */
      if (o->dest == XPAR_DEST_TO && xpar_vset_stream_intact(owned, before)) {
        xpar_options ex = *o;
        if (!o->quiet && xpar_vset_volumes_to_rewrite(owned))
          xpar_fprintf(xpar_stderr,
                       "xpar: %" PRIu64 " source data volume%s %s rewriting\n",
                       xpar_vset_volumes_to_rewrite(owned),
                       PLURAL(xpar_vset_volumes_to_rewrite(owned)),
                       xpar_vset_volumes_to_rewrite(owned) == 1
                         ? "still needs" : "still need");
        xpar_vset_close(owned);
        ex.verb = XPAR_VERB_EXTRACT;
        return xpar_op_extract(&ex);
      }
      out = repair_owned(o, owned, before, &acct);
      /*  Repair packet-bearing volumes as for a sidecar.  */
      if (out == XPAR_EXIT_OK) {
        acct.names_restored  = repair_restore_names(o, owned);
        acct.ragged_trimmed  = repair_trim_ragged(o, owned);
        acct.volumes_dropped = repair_rewrite_dropped(o, owned);
      }
      xpar_vset_close(owned);
      if (out == XPAR_EXIT_OK && o->dest != XPAR_DEST_TO) {
        acct.stale_regen = repair_rewrite_stale(o);
        acct.index_regen = repair_regen_index(o);
        acct.recovery_regen =
          repair_regen_recovery(o, &acct.recovery_volumes);
        if (acct.recovery_regen || acct.index_regen || acct.names_restored ||
            acct.stale_regen)
          changed = true;
        owned = xpar_vset_open(o);
        out = xpar_vset_check(owned, o, NULL);
        xpar_vset_close(owned);
        if (out != XPAR_EXIT_OK && out != XPAR_EXIT_IO)
          out = XPAR_EXIT_UNREPAIRABLE;
      }
      if (acct.slices_rebuilt || acct.volumes_rebuilt ||
          acct.volumes_rewritten || acct.volumes_relengthed ||
          acct.ragged_trimmed || acct.volumes_dropped ||
          acct.inner_corrected)
        changed = true;
      if (rp_require_lost(o, acct.meta_skip, true) && out == XPAR_EXIT_OK)
        out = XPAR_EXIT_IO;
      if (out == XPAR_EXIT_OK && changed && o->exit_on_change)
        out = XPAR_EXIT_REPAIRABLE;
      if (o->json) {
        owned_json_repair(&owned_js, owned_layout, changed, &acct, o);
        rp_meta_report(o, acct.meta_skip, &owned_js);
        xpar_json_summary(&owned_js,
                          out == XPAR_EXIT_OK ? "clean" :
                          out == XPAR_EXIT_REPAIRABLE ? "changed" :
                          out == XPAR_EXIT_IO ? "io-error" : "unrepairable",
                          out);
      }
      if (!o->quiet && out == XPAR_EXIT_UNREPAIRABLE)
        xpar_fprintf(xpar_stderr, "xpar: owned-layout repair: unrepairable\n");
      else if (!o->quiet && !o->chain_member && !acct.slices_rebuilt &&
               !acct.volumes_rebuilt &&
               !acct.volumes_rewritten && !acct.volumes_relengthed &&
               !acct.recovery_regen && !acct.index_regen &&
               !acct.names_restored && !acct.stale_regen &&
               !acct.ragged_trimmed && !acct.volumes_dropped &&
               !acct.inner_corrected)
        xpar_fprintf(xpar_stderr,
                     "xpar: owned-layout repair: no repair needed\n");
      /*  Index and name work is reported on its own line below.  */
      else if (!o->quiet && (acct.slices_rebuilt || acct.volumes_rebuilt ||
                             acct.volumes_rewritten ||
                             acct.volumes_relengthed || acct.inner_corrected))
        xpar_fprintf(xpar_stderr,
                     "xpar: owned-layout repair: %" PRIu64 " cell%s damaged, %"
                     PRIu64 " slice%s rebuilt, %" PRIu64 " byte%s; %" PRIu64
                     " data volume%s rebuilt, %" PRIu64
                     " rewritten from a substitute, %" PRIu64
                     " restored to its recorded length%s.\n",
                     acct.cells_bad, PLURAL(acct.cells_bad),
                     acct.slices_rebuilt, PLURAL(acct.slices_rebuilt),
                     acct.bytes_rebuilt, PLURAL(acct.bytes_rebuilt),
                     acct.volumes_rebuilt, PLURAL(acct.volumes_rebuilt),
                     acct.volumes_rewritten,
                     acct.volumes_relengthed,
                     acct.inner_corrected
                       ? "; the inner code corrected the archive" : "");
      if (!o->quiet && acct.index_regen)
        xpar_fprintf(xpar_stderr,
                     "xpar: recreated %" PRIu64 " index volume%s from packet "
                     "replicas\n", acct.index_regen, PLURAL(acct.index_regen));
      if (!o->quiet && acct.recovery_regen)
        xpar_fprintf(xpar_stderr,
                     "xpar: %" PRIu64 " recovery slice%s regenerated in %"
                     PRIu64 " volume%s\n",
                     acct.recovery_regen, PLURAL(acct.recovery_regen),
                     acct.recovery_volumes, PLURAL(acct.recovery_volumes));
      if (!o->quiet && acct.stale_regen)
        xpar_fprintf(xpar_stderr,
                     "xpar: rewrote %" PRIu64 " stale volume%s\n",
                     acct.stale_regen, PLURAL(acct.stale_regen));
      if (!o->json) rp_meta_report(o, acct.meta_skip, NULL);
      return out;
    }
    pre_restored = repair_restore_names(o, owned);
    pre_trimmed  = repair_trim_ragged(o, owned);
    pre_dropped  = repair_rewrite_dropped(o, owned);
    if (o->dry_run && o->dest != XPAR_DEST_TO)
      pre_trimmed = xpar_vset_volumes_ragged(owned);
    xpar_vset_close(owned);
  }

  xpar_memset(&r, 0, sizeof r);
  r.names_restored = pre_restored;
  r.ragged_trimmed = pre_trimmed;
  r.vols_dropped   = pre_dropped;
  r.o = o;  r.verbose = o->verbose;  r.quiet = o->quiet;
  xpar_json_init(&r.js, o->json ? xpar_stdout : xpar_stderr, o->json);
  xpar_crc32c_init();
  xpar_critset_init(&r.crit);
  if (o->auth_key) {
    xpar_keyfile_load_or_die(o->auth_key, &r.key, r.master);
    r.key_loaded = true;
  }

  FATAL_UNLESS("Repair needs a set to work on.", o->set_ref.count > 0);
  /*  Resolve entries beside the named set.  */
  r.dir = o->set_ref.dir  ? xpar_strdup(o->set_ref.dir)
        : o->set_ref.home ? xpar_strdup(o->set_ref.home)
                          : xpar_path_dir(o->set_ref.vol[0]);
  for (i = 0; i < o->set_ref.count; i++)
    (void) rp_vol_open(&r, o->set_ref.vol[i]);
  FATAL_UNLESS("Nothing in '%s' could be opened.", r.vol_count > 0, o->set);
  rp_key_preflight(&r);
  for (i = 0; i < r.vol_count; i++)
    rp_collect(&r, r.vol[i].data, r.vol[i].size,
               o->resync == XPAR_RESYNC_ALWAYS);
  if (!rp_have_setd(&r))
    for (i = 0; i < r.vol_count; i++)
      rp_salvage(&r, r.vol[i].data, r.vol[i].size);

  rp_pick_setd(&r);
  rp_authenticate(&r);
  FATAL_UNLESS("This set requires `xpar extract`, not repair.",
               r.sd.layout == XPAR_LAYOUT_SIDECAR);
  if (!xpar_geom_from_setd(&r.sd, &r.geom))
    FATAL_FORMAT("The set descriptor's geometry is inconsistent.");

  /*  CRC32C alone cannot authorise overwriting the evidence.  */
  if (o->dest != XPAR_DEST_TO && o->dest != XPAR_DEST_BACKUP &&
      r.sd.slice_tag_len == 0 && !o->force)
    FATAL("In-place repair without slice tags requires -f; use --to DIR to "
          "preserve the originals.");

  rp_read_manifest(&r);
  rp_read_tags(&r);
  rp_open_recovery(&r);

  rp_entry_state_alloc(&r);

  if (o->dest == XPAR_DEST_BACKUP && !o->force) {
    for (i = 0; i < r.mf.count; i++)
      if (r.mf.entry[i].entry_type == XPAR_ENTRY_HARDLINK)
        FATAL("--backup would break the hard-link group containing '%.*s'; "
              "use --in-place, --to, or -f.",
              (int) r.mf.entry[i].name_len, r.mf.entry[i].name);
  }

  xpar_erasures_init(&r.er, r.geom.slice_count, r.geom.cells_per_slice);
  r.susp = (u8 *) xpar_calloc((sz) MAX(r.geom.slice_count *
                                       r.geom.cells_per_slice, 1), 1);

  rp_resync_tree(&r);
  xpar_progress_init(&pg, xpar_progress_wanted(o),
                     r.sd.stream_length, "Repairing");
  if (o->json) xpar_progress_sink(&pg, xpar_json_progress_sink, &r.js);
  rp_scan_stream(&r, &pg);
  rp_scan_entries(&r, &pg);
  rp_scan_structure(&r);
  rp_classify(&r);
  xpar_progress_end(&pg);
  rp_close_files(&r);

  depth = xpar_erasures_max_depth(&r.er);
  partial = o->dest == XPAR_DEST_TO || o->dest == XPAR_DEST_BACKUP;
  if (depth > r.rec_avail) {
    rp_note(&r, "xpar: the deepest column has %" PRIu64 " erasures against %"
            PRIu64 " "
                "recovery slices; %" PRIu64 " short.\n",
            depth, r.rec_avail,
            (depth - r.rec_avail));
    if (!partial) {
      { int code = rp_code(&r, XPAR_EXIT_UNREPAIRABLE);
        rp_report(&r, code == XPAR_EXIT_IO ? "io-error"
                                           : "unrepairable", code);
        rp_free(&r);
        return code; }
    }
    rp_note(&r, "xpar: writing the entries that can still be "
            "reproduced\n");
  }
  { bool unclean = false;
    for (i = 0; i < r.mf.count; i++) {
      if (!(r.fstate[i] & 2)) continue;
      unclean = true;
      /*  Aliases share the canonical file.  */
      if (!r.alias[i]) r.overlong++;
    }
    /*  A missing hard-link name is damage even if all cells verify.  */
    for (i = 0; i < r.mf.count; i++) {
      xpar_stat_t lst, cst;
      if (r.mf.entry[i].entry_type != XPAR_ENTRY_HARDLINK ||
          r.canon[i] == i) continue;
      if (xpar_lstat(r.path[i], &lst) != 0) { r.links_missing++;  break; }
      /*  A copy standing where the link belongs is damage as well.  */
      if (xpar_lstat(r.path[r.canon[i]], &cst) == 0 && (lst.dev | lst.ino) &&
          (cst.dev | cst.ino) &&
          (lst.dev != cst.dev || lst.ino != cst.ino)) {
        r.links_missing++;  break;
      }
    }
    /*  Hash failures may be repairable even when no cell is marked.  */
    for (i = 0; i < r.mf.count; i++)
      if (r.hash_bad[i] && !r.alias[i]) { unclean = true;  break; }
    if (rp_missing_names(&r)) unclean = true;
    if (!r.cell_count && !unclean && !r.links_missing &&
        !r.structure_bad) {
      bool regen;
      rp_release_vols(&r);
      r.stale_regen = repair_rewrite_stale(o);
      r.index_regen = repair_regen_index(o);
      r.rec_regen = repair_regen_recovery(o, &r.rec_regen_vols);
      regen = r.rec_regen != 0 || r.index_regen != 0 || r.stale_regen != 0 ||
              r.names_restored != 0 || r.ragged_trimmed != 0 ||
              r.vols_dropped != 0;
      if (r.io_errors) {
        rp_report(&r, "io-error", XPAR_EXIT_IO);
        rp_free(&r);
        return XPAR_EXIT_IO;
      }
      rp_report(&r, "clean", XPAR_EXIT_OK);
      rp_free(&r);
      if (regen && o->exit_on_change) return XPAR_EXIT_REPAIRABLE;
      return XPAR_EXIT_OK;
    }
  }

  /*  Decode holds S + R column buffers; refuse budgets below one 64-byte
      quantum.  */
  chunk = (u32) MIN((u64) (r.geom.cell_bytes ? r.geom.cell_bytes
                                             : r.geom.slice_size),
                    (u64) 1 << 20);
  if (xpar_plan_for_repair(&r.sd, r.rec_total, o->memory, o->jobs, &pl) ==
        XPAR_PLAN_OK &&
      pl.column_chunk)
    chunk = (u32) MIN((u64) pl.column_chunk, (u64) chunk);
  { u64 lanes = r.geom.slice_count + r.rec_total;
    if (o->memory && lanes && (u64) chunk * lanes > o->memory)
      chunk = (u32) ((o->memory / lanes) & ~(u64) 63);
    if (chunk < 64 && r.er.bad_count)
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "Decoding needs at least %" PRIu64 " bytes of buffer; raise -m.",
                 (lanes * 64));
    if (chunk < 64) chunk = 64;
  }
  rp_solve_copies(&r);
  if (!rp_solve_decode(&r, chunk, partial)) {
    rp_note(&r, "xpar: decoded data does not match the recorded cell tags\n");
    { int code = rp_code(&r, XPAR_EXIT_UNREPAIRABLE);
      rp_report(&r, code == XPAR_EXIT_IO ? "io-error"
                                         : "unrepairable", code);
      rp_free(&r);
      return code; }
  }
  { bool widened = false;
    bool gated = rp_slice_gate(&r, &widened);
    if (gated && widened) {
      /*  A CRC-invisible forgery only shows up here; decode those cells.  */
      depth = xpar_erasures_max_depth(&r.er);
      if (depth > r.rec_avail) {
        rp_note(&r, "xpar: the deepest column has %" PRIu64 " erasures against %"
                PRIu64 " recovery slices; %" PRIu64 " short.\n",
                depth, r.rec_avail, (depth - r.rec_avail));
        gated = false;
      } else if (!rp_solve_decode(&r, chunk, partial)) {
        rp_note(&r,
                "xpar: decoded data does not match the recorded cell tags\n");
        gated = false;
      } else gated = rp_slice_gate(&r, NULL);
    }
    if (!gated) {
      rp_note(&r, "xpar: a reconstructed slice failed its strong tag\n");
      { int code = rp_code(&r, XPAR_EXIT_UNREPAIRABLE);
        rp_report(&r, code == XPAR_EXIT_IO ? "io-error"
                                           : "unrepairable", code);
        rp_free(&r);
        return code; }
    }
  }
  if (o->paranoid && !rp_paranoid(&r, chunk)) {
    { int code = rp_code(&r, XPAR_EXIT_UNREPAIRABLE);
      rp_report(&r, code == XPAR_EXIT_IO ? "io-error"
                                         : "unrepairable", code);
      rp_free(&r);
      return code; }
  }
  if (o->repair_head_set) rp_select_head_output(&r);

  rp_build_writes(&r);
  /*  --to restores structural damage from the manifest.  */
  if (o->dest != XPAR_DEST_TO) r.opaque += r.structure_bad;
  /*  A separate destination still reproduces everything else.  */
  if (r.opaque && !partial) {
    { int code = rp_code(&r, XPAR_EXIT_UNREPAIRABLE);
      rp_report(&r, code == XPAR_EXIT_IO ? "io-error"
                                         : "unrepairable", code);
      rp_free(&r);
      return code; }
  }
  if (o->dest == XPAR_DEST_TO || o->dest == XPAR_DEST_BACKUP) {
    if (!o->dry_run)
      rp_write_tree(&r, o->dest == XPAR_DEST_TO ? o->to_dir : r.dir,
                    o->dest == XPAR_DEST_BACKUP);
    r.changed = r.writes > 0 || r.names_restored > 0 ||
                r.ragged_trimmed > 0 || r.vols_dropped > 0;
    if (r.unrecovered || r.opaque) {
      u64 lost = r.unrecovered ? r.unrecovered : r.opaque;
      rp_note(&r, "xpar: %" PRIu64 " entr%s unrecoverable; repaired the "
              "rest of the tree\n",
              lost, lost == 1 ? "y is" : "ies are");
      { int code = rp_code(&r, XPAR_EXIT_UNREPAIRABLE);
        rp_report(&r, code == XPAR_EXIT_IO ? "io-error"
                                           : "unrepairable", code);
        rp_free(&r);
        return code; }
    }
    { int code;
      rp_require_lost(o, r.meta_skip, true);
      code = rp_code(&r, XPAR_EXIT_OK);
      rp_report(&r, code == XPAR_EXIT_IO ? "io-error" : "repaired", code);
      rp_free(&r);
      if (code != XPAR_EXIT_OK) return code;
      return o->exit_on_change && r.changed ? XPAR_EXIT_REPAIRABLE
                                            : XPAR_EXIT_OK; }
  }

  if (o->dry_run) {
    u64 total = 0;
    for (i = 0; i < r.wr_count; i++)
      if (!r.wr[i].trunc && !r.wr[i].shadow) total += r.wr[i].len;
    rp_note(&r, "xpar: --dry-run: %" PRIu32 " writes totalling %" PRIu64
            " bytes would "
                "be made.\n", r.wr_count,
            total);
    /*  Include name and volume work in the plan.  */
    r.names_made  = rp_missing_names(&r);
    r.stale_regen = repair_rewrite_stale(o);
    r.index_regen = repair_regen_index(o);
    r.rec_regen   = repair_regen_recovery(o, &r.rec_regen_vols);
    { int code = rp_code(&r, XPAR_EXIT_OK);
      rp_report(&r, code == XPAR_EXIT_IO ? "io-error" : "dry-run", code);
      r.changed = r.wr_count || r.names_made || r.rec_regen ||
                  r.index_regen || r.links_missing || r.stale_regen ||
                  r.names_restored || r.ragged_trimmed || r.vols_dropped;
      rp_free(&r);
      if (code != XPAR_EXIT_OK) return code;
      return o->exit_on_change && r.changed ? XPAR_EXIT_REPAIRABLE
                                            : XPAR_EXIT_OK; }
  }

  /*  Nothing may change protected data before the journal is durable.  */
  r.journal = o->set_ref.base ? xpar_strdup(o->set_ref.base)
                              : xpar_path_join(r.dir, "xpar");
  { char * j;
    if (r.sd.generation)
      xpar_asprintf(&j, "%s.g%03" PRIu32 ".xparundo", r.journal, r.sd.generation);
    else
      xpar_asprintf(&j, "%s.xparundo", r.journal);
    xpar_free(r.journal);
    r.journal = j;
  }
  /*  Only --replace-journal replaces an existing journal path.  */
  { xpar_stat_t st;
    if (!o->no_journal && xpar_lstat(r.journal, &st) == 0) {
      if (xpar_journal_live(r.journal) && !o->replace_journal)
        FATAL("Undo journal '%s' exists; run xpar undo or pass "
              "--replace-journal.", r.journal);
      if (xpar_remove(r.journal) != 0)
        FATAL_IO("Cannot replace undo journal '%s': %s.", r.journal,
                 xpar_strerror(xpar_errno()));
    }
  }
  rp_read_old(&r);
  if (!o->no_journal) rp_journal(&r);
  rp_apply(&r);
  content_ok = rp_reverify(&r);
  if (!content_ok) rc = XPAR_EXIT_UNREPAIRABLE;
  else if (r.names_failed || r.links_failed) rc = XPAR_EXIT_IO;
  rp_require_lost(o, r.meta_skip, true);
  rc = rp_code(&r, rc);
  if ((content_ok || !r.writes) && !o->keep_journal && !o->no_journal &&
      !xpar_journal_drop(r.journal) && rc == XPAR_EXIT_OK)
    rc = XPAR_EXIT_IO;
  if (content_ok) {
    rp_release_vols(&r);
    r.stale_regen = repair_rewrite_stale(o);
    r.index_regen = repair_regen_index(o);
    r.rec_regen = repair_regen_recovery(o, &r.rec_regen_vols);
  }
  r.changed = r.writes > 0 || r.rec_regen > 0 || r.index_regen > 0 ||
              r.stale_regen > 0 || r.names_restored > 0 ||
              r.ragged_trimmed > 0 || r.vols_dropped > 0;

  rp_report(&r, rc == XPAR_EXIT_OK ? "repaired" :
                rc == XPAR_EXIT_IO ? "io-error" : "unrepairable", rc);
  rp_free(&r);
  if (rc == XPAR_EXIT_OK && o->exit_on_change && r.changed)
    return XPAR_EXIT_REPAIRABLE;
  return rc;
}
