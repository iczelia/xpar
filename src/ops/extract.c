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

/*  Extract split or armoured entries under the manifest path and metadata
    rules.  */

#include "ops.h"
#include "auth.h"
#include "chain.h"
#include "vset.h"
#include "volimg.h"

#include "armour.h"
#include "blake3.h"
#include "container.h"
#include "crc32c.h"
#include "json.h"
#include "manifest.h"
#include "pathname.h"
#include "port-fs.h"
#include "slice.h"

enum {
  EX_SK_SETID = 0, EX_SK_OWNER, EX_SK_XATTR, EX_SK_XATTR_NS, EX_SK_MODE,
  EX_SK_TIMES, EX_SK_BTIME, EX_SK_ATIME, EX_SK_CTIME, EX_SK_ATTRS,
  EX_SK_SYMLINK, EX_SK_LINKCOPY, EX_SK_COUNT
};

static const char * const ex_sk_name[EX_SK_COUNT] = {
  "setid", "owner", "xattr", "xattr-namespace", "mode", "times", "btime",
  "atime", "ctime", "attrs", "symlink-unsafe", "materialised-as-copy"
};

/*  Map reported degradations to their --require bits.  */
static const struct { u32 bit;  u8 cls; } ex_require_map[] = {
  { XPAR_PRES_SETID,     EX_SK_SETID    },
  { XPAR_PRES_OWNER,     EX_SK_OWNER    },
  { XPAR_PRES_XATTR,     EX_SK_XATTR    },
  { XPAR_PRES_XATTR_ALL, EX_SK_XATTR_NS },
  { XPAR_PRES_MODE,      EX_SK_MODE     },
  { XPAR_PRES_MTIME,     EX_SK_TIMES    },
  { XPAR_PRES_BTIME,     EX_SK_BTIME    },
  { XPAR_PRES_ATIME,     EX_SK_ATIME    },
  { XPAR_PRES_CTIME,     EX_SK_CTIME    },
  { XPAR_PRES_ATTRS,     EX_SK_ATTRS    },
  { XPAR_PRES_LINKS,     EX_SK_SYMLINK  },
  { XPAR_PRES_LINKS,     EX_SK_LINKCOPY }
};

/*  Armoured volume and its lazily corrected plaintext.  */
typedef struct {
  xpar_armour * a;
  const u8 *    src;           /*  The armoured region, 384 bytes in.  */
  u64           src_len;
  u8 *          plain;
  u64           plain_len;
  u8 *          done;          /*  Decoded frames.  */
  u64           frames;
} ex_arm;

typedef struct {
  const xpar_options * o;
  xpar_json  js;
  bool       quiet;
  xpar_key   key;
  u8         master[XPAR_BLAKE3_KEY_LEN];
  bool       key_loaded, keyed, auth_only;

  xpar_volimg * vol;  u32 vol_count, vol_cap;
  u8 **         plain;  u32 plain_count, plain_cap;

  xpar_critset crit;
  xpar_setd    sd;
  bool         have_setd;
  u8           set_id[XPAR_SET_ID_LEN];
  xpar_manifest mf;
  xpar_nameidx  nix;
  xpar_layt     layt;
  bool          have_layt;
  xpar_posix_rec * posix;
  u32              posix_count;

  /*  A chain has one POSX namespace and one independently encoded stream
      per generation. `owner[i]` selects the metadata namespace for an
      effective-manifest entry; stream_set is searched by the global stream
      ranges committed by SETD.  */
  xpar_chain chain;
  bool       have_chain;
  u32        selected;
  u32 *      owner;
  xpar_posix_rec ** posix_tab;
  u32 *      posix_tab_count;
  xpar_vset ** stream_set;
  u32          stream_count;

  char *  dir;                 /*  Where the set's volumes live.  */
  char *  dest;                /*  Where the tree is written.  */
  u32     caps;                /*  Of the destination filesystem.  */
  u32     path_flags;

  /*  The set stream, either as data volumes or as one plaintext run.  */
  const u8 * strm;             /*  ARMOURED: the STRM payload.  */
  u64        strm_off, strm_len;

  /*  Armoured volumes retained for lazy payload correction.  */
  ex_arm * arm;  u32 arm_count, arm_cap;

  u64 skip[EX_SK_COUNT];
  u64 entries, bytes, links, copies, mismatches, io_failures;
  u64 substituted;             /*  Data volumes read from a spare copy.  */
  u64 vol_damaged;             /*  Data volumes read despite a bad tag.  */
  u64 hash_bad;                /*  Entries whose content hash still fails.  */
} ex;

static void ex_note(ex * x, const char * fmt, ...) XPAR_PRINTF(2, 3);

static void ex_note(ex * x, const char * fmt, ...) {
  va_list ap;
  if (x->quiet) return;
  va_start(ap, fmt);
  /*  stdout carries JSON Lines or, under --stdout, the data itself.  */
  xpar_vfprintf(xpar_stderr, fmt, ap);
  va_end(ap);
}

static void ex_skip(ex * x, const xpar_entry * e, int cls,
                    const char * why) {
  x->skip[cls]++;
  if (!x->o->json) return;
  xpar_json_begin(&x->js, "metadata_skipped");
  xpar_json_name(&x->js, "entry", e->name, e->name_len);
  xpar_json_str (&x->js, "class", ex_sk_name[cls]);
  xpar_json_str (&x->js, "reason", why);
  xpar_json_end(&x->js);
}

static char * ex_find_data(ex * x, const xpar_vol * v, char ** basename,
                           bool * damaged) {
  char * named, * path;
  xpar_dir * d;
  const xpar_dirent * de;
  xpar_stat_t st;
  *basename = NULL;
  *damaged = false;
  named = xpar_path_vol(x->dir, v->name);
  if (xpar_vol_tag_match(named, v)) {
    *basename = xpar_strdup(v->name);
    return named;
  }
  if (v->vol_tag && (d = xpar_opendir(x->dir)) != NULL) {
    while ((de = xpar_readdir(d)) != NULL) {
      if (!de->is_regular || !xpar_strcmp(de->name, v->name)) continue;
      path = xpar_path_join(x->dir, de->name);
      if (xpar_vol_tag_match(path, v)) {
        *basename = xpar_strdup(de->name);
        xpar_closedir(d);
        xpar_free(named);
        return path;
      }
      xpar_free(path);
    }
    xpar_closedir(d);
  }
  /*  A present volume with a bad tag is damaged, not missing.  */
  if (xpar_lstat(named, &st) == 0 && st.is_regular) {
    *basename = xpar_strdup(v->name);
    *damaged = true;
    return named;
  }
  xpar_free(named);
  return NULL;
}

static bool ex_vol_open(ex * x, const char * path) {
  xpar_volimg v;
  int err = 0;
  xpar_volimg_status st = xpar_volimg_read(&v, path, &err);
  if (st == XPAR_VOLIMG_IO)
    FATAL_IO("Cannot read volume '%s': %s.", path,
             xpar_strerror(err ? err : xpar_errno()));
  if (st != XPAR_VOLIMG_OK) return false;
  if (x->vol_count == x->vol_cap) {
    x->vol_cap = x->vol_cap ? x->vol_cap * 2 : 8;
    x->vol = (xpar_volimg *)
               xpar_realloc(x->vol, x->vol_cap * sizeof(xpar_volimg));
  }
  x->vol[x->vol_count++] = v;
  return true;
}

static void ex_keep_plain(ex * x, u8 * p) {
  if (x->plain_count == x->plain_cap) {
    x->plain_cap = x->plain_cap ? x->plain_cap * 2 : 4;
    x->plain = (u8 **) xpar_realloc(x->plain, x->plain_cap * sizeof(u8 *));
  }
  x->plain[x->plain_count++] = p;
}

static void ex_collect(ex * x, const u8 * buf, u64 size);

/*  Plaintext an armoured group gave up: scan it, then keep it, because
    the collector now points into it.  */
static void ex_plain(void * user, u8 * plain, u64 len) {
  ex * x = (ex *) user;
  ex_collect(x, plain, len);
  ex_keep_plain(x, plain);
}

static void ex_collect(ex * x, const u8 * buf, u64 size) {
  xpar_scan sc;
  xpar_pkt h;
  const u8 * body;
  u64 off;
  xpar_scan_init(&sc, buf, size, x->key_loaded ? &x->key : NULL, false);
  sc.accept_unverified_keyed = false;
  while (xpar_scan_next(&sc, &h, &body, &off)) {
    if (xpar_pkt_is(&h, XPAR_T_ARMG)) {
      /*  Skip wrapped recovery slices during metadata extraction.  */
      char wt[4];
      if (xpar_armg_wrapped_type(body, (sz) (h.length - XPAR_PKT_HDR), wt) &&
          !xpar_memcmp(wt, XPAR_T_RCVS, 4)) continue;
      xpar_armg_unwrap(body, h.length - XPAR_PKT_HDR, false, ex_plain, x);
    } else {
      if (xpar_pkt_is(&h, XPAR_T_STRM)) {
        xpar_strm s;
        if (xpar_strm_read(body, (sz) (h.length - XPAR_PKT_HDR),
                           &s) == XPAR_OK) {
          x->strm = s.data;  x->strm_off = s.stream_offset;
          x->strm_len = s.length;
        }
      }
      xpar_critset_add(&x->crit, &h, body);
    }
  }
  xpar_reject_unknown_critical(&sc);
}

static bool ex_have_setd(const ex * x) {
  For(u32, i, x->crit.count,
      if (xpar_pkt_is(&x->crit.pkt[i].hdr, XPAR_T_SETD)) return true)
  return false;
}

static void ex_salvage(ex * x, const u8 * buf, u64 size) {
  xpar_armg_salvage(buf, size, x->key_loaded ? &x->key : NULL, ex_plain, x);
}

static void ex_open_armoured(ex * x, const xpar_volimg * v) {
  xpar_arm_prologue pr;
  xpar_armour_params p;
  xpar_armour * a;
  u8 * plain;
  const u8 * arm;
  u64 plain_len, arm_len;
  FATAL_UNLESS_CODE(XPAR_EXIT_UNREPAIRABLE,
               "No prologue copy in '%s' verifies; try "
               "`xpar recover-prologue`.",
               xpar_garm_prologue(v->data, (sz) v->size, &pr, NULL),
               v->path);
  p.symbol_bits = pr.symbol_bits;
  p.poly = pr.poly; p.n = pr.n; p.k = pr.k;
  p.fcr = pr.fcr; p.prim = pr.prim; p.depth = pr.depth;
  plain_len = pr.plain_length; arm_len = pr.armoured_length;
  FATAL_UNLESS("The armoured prologue names unusable parameters: %s",
               xpar_armour_check(&p) == NULL, xpar_armour_check(&p));
  FATAL_UNLESS("The armoured region is too large for this host.",
               plain_len <= (u64) (sz) -1 / 2 && arm_len <= (u64) (sz) -1);
  /*  Zero-fill a missing tail as erasures.  */
  if (384 + arm_len > v->size) {
    u64 avail = v->size > 384 ? v->size - 384 : 0;
    u8 * pad = (u8 *) xpar_alloc_raw((sz) arm_len);
    xpar_memcpy(pad, v->data + 384, (sz) avail);
    xpar_memset(pad + avail, 0, (sz) (arm_len - avail));
    ex_keep_plain(x, pad);
    arm = pad;
  } else arm = v->data + 384;
  xpar_gf_init();
  a = xpar_armour_new(&p);
  plain = (u8 *) xpar_alloc_raw((sz) plain_len ? (sz) plain_len : 1);
  xpar_armour_extract(a, plain, plain_len, arm);
  ex_collect(x, plain, plain_len);
  ex_keep_plain(x, plain);
  if (x->arm_count == x->arm_cap) {
    x->arm_cap = x->arm_cap ? x->arm_cap * 2 : 2;
    x->arm = (ex_arm *) xpar_realloc(x->arm, x->arm_cap * sizeof *x->arm);
  }
  { ex_arm * m = &x->arm[x->arm_count++];
    u64 fp = xpar_armour_frame_plain(a);
    m->a = a;  m->src = arm;  m->src_len = arm_len;
    m->plain = plain;  m->plain_len = plain_len;
    m->frames = fp ? xpar_ceil_div(plain_len, fp) : 0;
    m->done = (u8 *) xpar_calloc((sz) (m->frames ? m->frames : 1), 1); }
  /*  Decode when packets or the descriptor do not verify.  */
  if (!xpar_verify_packets_ok(plain, plain_len, x->key_loaded ? &x->key : NULL)
      || !ex_have_setd(x)) {
    u8 * region = (u8 *) xpar_alloc_raw((sz) arm_len ? (sz) arm_len : 1);
    u8 * fixed  = (u8 *) xpar_alloc_raw((sz) plain_len ? (sz) plain_len : 1);
    u64 fd = xpar_armour_frame_disk(a);
    xpar_memcpy(region, arm, (sz) arm_len);
    if (fd) xpar_armour_decode_frames(a, region, arm_len / fd, NULL);
    xpar_armour_extract(a, fixed, plain_len, region);
    xpar_free(region);
    ex_collect(x, fixed, plain_len);
    ex_keep_plain(x, fixed);
  }
}

/*  Lazily decode frames backing [lo, hi), once each; report any change.  */
static bool ex_armour_apply(ex * x, u64 lo, u64 hi) {
  bool changed = false;
  u32 i;
  /*  A chain reads its payload through per-generation sets instead.  */
  for (i = 0; i < x->stream_count; i++)
    if (x->stream_set[i] &&
        xpar_vset_armour_correct(x->stream_set[i], lo, hi))
      changed = true;
  for (i = 0; i < x->arm_count; i++) {
    ex_arm * m = &x->arm[i];
    u64 fp = xpar_armour_frame_plain(m->a), fd = xpar_armour_frame_disk(m->a);
    u64 base, first, last, f;
    u8 * enc, * out;
    /*  Correct only the active stream buffer.  */
    if (!fp || !m->frames || !x->strm) continue;
    if (x->strm < m->plain || x->strm >= m->plain + m->plain_len) continue;
    base = (u64) (x->strm - m->plain);
    if (lo < x->strm_off) continue;
    first = (base + (lo - x->strm_off)) / fp;
    last  = (base + (hi - x->strm_off) - (hi > lo ? 1 : 0)) / fp;
    if (last >= m->frames) last = m->frames - 1;
    enc = (u8 *) xpar_alloc_raw((sz) fd ? (sz) fd : 1);
    out = (u8 *) xpar_alloc_raw((sz) fp);
    for (f = first; f <= last; f++) {
      u64 po = f * fp, have = MIN(fp, m->plain_len - po);
      if (m->done[f]) continue;
      if (f * fd > m->src_len || fd > m->src_len - f * fd) break;
      m->done[f] = 1;
      xpar_memcpy(enc, m->src + f * fd, (sz) fd);
      if (xpar_armour_decode_frame(m->a, enc, NULL) == XPAR_ARMOUR_FAILED)
        continue;
      xpar_armour_extract(m->a, out, have, enc);
      if (!xpar_memcmp(out, m->plain + po, (sz) have)) continue;
      /*  Preserve collected pointers.  */
      xpar_memcpy(m->plain + po, out, (sz) have);
      changed = true;
    }
    xpar_free(enc);  xpar_free(out);
  }
  return changed;
}

/*  The set.  */

static void ex_pick_setd(ex * x) {
  u32 i, j;
  const xpar_crit_pkt * want = NULL;
  xpar_setd sd;
  for (i = 0; i < x->crit.count; i++) {
    const xpar_crit_pkt * p = &x->crit.pkt[i];
    bool named = false, head = true;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_SETD)) continue;
    if (xpar_setd_read(p->body, (sz) p->body_len, &sd) != XPAR_OK) {
      xpar_setd_free(&sd);
      continue;
    }
    if (x->o->gen_count) {
      const xpar_genref * g = &x->o->gens[0];
      named = g->by_id
                ? xpar_hex_prefix(p->hdr.set_id, XPAR_SET_ID_LEN,
                                  g->id_prefix)
                : sd.generation == (u32) g->number;
    }
    for (j = 0; j < x->crit.count && head; j++) {
      const xpar_crit_pkt * q = &x->crit.pkt[j];
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
    if (x->o->gen_count ? named : head) { want = p;  break; }
  }
  FATAL_UNLESS_CODE(XPAR_EXIT_UNREPAIRABLE,
                    "No set descriptor survived in '%s'.", want != NULL,
                    x->o->set);
  if (xpar_setd_read(want->body, (sz) want->body_len, &x->sd) != XPAR_OK)
    FATAL_FORMAT("The set descriptor is malformed.");
  xpar_memcpy(x->set_id, want->hdr.set_id, XPAR_SET_ID_LEN);
  x->have_setd = true;
}

static void ex_authenticate(ex * x) {
  u32 i;
  for (i = 0; i < x->crit.count; i++) {
    const xpar_crit_pkt * p = &x->crit.pkt[i];
    xpar_auth a;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_AUTH) ||
        xpar_memcmp(p->hdr.set_id, x->set_id, XPAR_SET_ID_LEN) ||
        xpar_auth_read(p->body, (sz) p->body_len, &a) != XPAR_OK) continue;
    if (!x->key_loaded)
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "Extracting an authenticated set requires --auth-key=FILE; "
                 "keyless access is read-only.");
    if (!xpar_auth_key_ok(&a, x->master))
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "The authentication key is wrong for this set.");
    x->keyed = true;
    x->auth_only = !a.unkeyed_retained;
    return;
  }
}

static void ex_read_manifest(ex * x) {
  u32 i;
  for (i = 0; i < x->sd.file_count; i++) {
    const xpar_crit_pkt * p = xpar_critset_find_file(
                                &x->crit, x->set_id, x->sd.file_id[i]);
    xpar_entry tmp, * e;
    FATAL_UNLESS_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "Manifest entry %" PRIu32 " of %" PRIu32
                 " is missing from every volume.",
                 p != NULL, i + 1, x->sd.file_count);
    if (xpar_entry_read(p->body, (sz) p->body_len, x->sd.posix_record_count,
                        &tmp) != XPAR_OK)
      FATAL_FORMAT("Manifest entry %" PRIu32 " is malformed.", i + 1);
    e  = xpar_manifest_append(&x->mf);
    *e = tmp;
  }
  x->mf.stream_base   = x->sd.stream_base;
  x->mf.stream_length = x->sd.stream_length;
  x->mf.slice_size    = x->sd.slice_size;
  x->mf.align         = x->sd.align;
  xpar_nameidx_build(&x->mf, &x->nix);
  /*  POSX is per generation and is indexed from FILE.posix_index.  */
  if (xpar_posx_collect(&x->crit, x->set_id, x->sd.posix_record_count,
                        &x->posix) != XPAR_OK)
    FATAL_FORMAT("The POSX table has gaps, overlaps, or invalid ranges.");
  x->posix_count = x->sd.posix_record_count;
}

/*  Reading the set stream.  */

static bool ex_read_stream(ex * x, u64 off, u64 len, u8 * dst) {
  if (x->stream_count) {
    u32 i;
    for (i = 0; i < x->stream_count; i++) {
      const xpar_geom * g = xpar_vset_geom(x->stream_set[i]);
      u64 end;
      if (g->stream_length > UINT64_MAX - g->stream_base) continue;
      end = g->stream_base + g->stream_length;
      if (off >= g->stream_base && off <= end && len <= end - off)
        return xpar_vset_read(x->stream_set[i], off, dst, len);
    }
    return false;
  }
  if (x->strm) {
    if (off < x->strm_off || off + len > x->strm_off + x->strm_len)
      return false;
    xpar_memcpy(dst, x->strm + (off - x->strm_off), (sz) len);
    return true;
  }
  while (len) {
    u32 i;
    bool hit = false;
    for (i = 0; i < x->layt.count && !hit; i++) {
      const xpar_vol * v = &x->layt.vol[i];
      u64 lo, hi, take;
      u32 k;
      if (v->kind != XPAR_VOL_DATA) continue;
      lo = v->stream_offset;  hi = lo + v->byte_length;
      if (off < lo || off >= hi) continue;
      take = MIN(len, hi - off);
      for (k = 0; k < x->vol_count; k++) {
        const char * b = x->vol[k].path, * s;
        for (s = b; *s; s++) if (*s == '/') b = s + 1;
        if (xpar_strcmp(b, v->name)) continue;
        if (off - lo + take > x->vol[k].size) return false;
        xpar_memcpy(dst, x->vol[k].data + (off - lo), (sz) take);
        hit = true;
        break;
      }
      if (!hit) return false;
      off += take;  len -= take;  dst += take;
    }
    if (!hit) return false;
  }
  return true;
}

static void ex_apply_meta(ex * x, u32 idx, const char * path) {
  const xpar_entry * e = &x->mf.entry[idx];
  const xpar_options * o = x->o;
  bool link = e->entry_type == XPAR_ENTRY_SYMLINK;
  const xpar_posix_rec * pr = NULL;
  if (x->have_chain && idx < x->mf.count && x->owner &&
      x->owner[idx] < x->chain.gen_count) {
    u32 g = x->owner[idx];
    if (e->posix_index != XPAR_ABSENT_U32 &&
        e->posix_index < x->posix_tab_count[g])
      pr = &x->posix_tab[g][e->posix_index];
  } else if (e->posix_index != XPAR_ABSENT_U32 &&
             e->posix_index < x->posix_count) {
    pr = &x->posix[e->posix_index];
  }

  if (link && !(x->caps & XPAR_FS_NOFOLLOW)) {
    ex_skip(x, e, EX_SK_SYMLINK,
            "the host has no symlink-safe metadata call");
    return;
  }

  /*  Set ownership before mode because chown clears set-ID bits.  */
  if (pr && (pr->uid != XPAR_ID_NONE || pr->gid != XPAR_ID_NONE ||
             pr->owner || pr->group)) {
    if (!(o->preserve & XPAR_PRES_OWNER))
      ex_skip(x, e, EX_SK_OWNER, "--preserve=owner was not given");
    else {
      bool by_name = o->owner_map == XPAR_OWNERMAP_NAME;
      if (xpar_set_owner(path, 1, pr->uid, pr->gid,
                         by_name ? pr->owner : NULL,
                         by_name ? pr->group : NULL) != 0)
        ex_skip(x, e, EX_SK_OWNER, xpar_strerror(xpar_errno()));
    }
  }

  if ((o->preserve & XPAR_PRES_MODE) && !link && e->mode != XPAR_ABSENT_U32) {
    u32 m = e->mode & XPAR_MODE_PERM;
    u32 id = m & (XPAR_MODE_SETUID | XPAR_MODE_SETGID | XPAR_MODE_STICKY);
    if (id && !(o->preserve & XPAR_PRES_SETID)) {
      m &= ~id;
      ex_skip(x, e, EX_SK_SETID, "setuid, setgid and sticky bits cleared");
    }
    if (xpar_set_mode(path, 1, m) != 0)
      ex_skip(x, e, EX_SK_MODE, xpar_strerror(xpar_errno()));
  }

  { i64 at = XPAR_TIME_NONE, mt = XPAR_TIME_NONE, bt = XPAR_TIME_NONE;
    if ((o->preserve & XPAR_PRES_MTIME) && e->mtime_ns != XPAR_ABSENT_TIME)
      mt = e->mtime_ns;
    if ((o->preserve & XPAR_PRES_ATIME) && e->atime_ns != XPAR_ABSENT_TIME)
      at = e->atime_ns;
    else if (e->atime_ns != XPAR_ABSENT_TIME)
      ex_skip(x, e, EX_SK_ATIME, "--preserve=atime was not given");
    if ((o->preserve & XPAR_PRES_BTIME) && e->btime_ns != XPAR_ABSENT_TIME) {
      if (x->caps & XPAR_FS_BTIME) bt = e->btime_ns;
      else ex_skip(x, e, EX_SK_BTIME, "this host cannot set a birth time");
    }
    if (e->ctime_ns != XPAR_ABSENT_TIME)
      ex_skip(x, e, EX_SK_CTIME, "ctime cannot be set on any host");
    if ((at != XPAR_TIME_NONE || mt != XPAR_TIME_NONE ||
         bt != XPAR_TIME_NONE) && xpar_set_times(path, 1, at, mt, bt) != 0)
      ex_skip(x, e, EX_SK_TIMES, xpar_strerror(xpar_errno()));
  }

  if ((o->preserve & XPAR_PRES_ATTRS) && e->attrs) {
    u16 a = (u16) (e->attrs & XPAR_ATTR_SETTABLE);
    if (a && xpar_set_attrs(path, 1, a) != 0)
      ex_skip(x, e, EX_SK_ATTRS, xpar_strerror(xpar_errno()));
  }

  if (pr && pr->xattr_count) {
    u32 k;
    if (!(o->preserve & XPAR_PRES_XATTR))
      ex_skip(x, e, EX_SK_XATTR, "--preserve=xattr was not given");
    else for (k = 0; k < pr->xattr_count; k++) {
      const xpar_xattr * a = &pr->xattrs[k];
      bool user = a->name && !xpar_strncmp(a->name, "user.", 5);
      if (!user && !(o->preserve & XPAR_PRES_XATTR_ALL)) {
        ex_skip(x, e, EX_SK_XATTR_NS,
                "outside the user. namespace and --preserve=xattr-all "
                "was not given");
        continue;
      }
      if (xpar_setxattr(path, 1, a->name, a->value, a->value_len) != 0)
        ex_skip(x, e, EX_SK_XATTR, xpar_strerror(xpar_errno()));
    }
  }
}

/*  Writing entries.  */

static char * ex_resolve(ex * x, const xpar_entry * e) {
  xpar_path_status why;
  char * p = xpar_path_resolve(x->dest, e->name, e->name_len,
                               x->path_flags, &why);
  if (!p) {
    ex_note(x, "xpar: refusing '%.*s': %s.\n", (int) e->name_len, e->name,
            xpar_path_reason(why));
    x->mismatches++;
  }
  return p;
}

static char * ex_resolve_leaf(ex * x, const xpar_entry * e) {
  u32 cut = e->name_len;
  char * parent, * out;
  xpar_path_status why;
  if (xpar_path_check(e->name, e->name_len, x->path_flags) != XPAR_PATH_OK)
    return ex_resolve(x, e);
  while (cut && e->name[cut - 1] != '/') cut--;
  if (!cut) return xpar_path_join_n(x->dest, e->name, e->name_len);
  parent = xpar_path_resolve(x->dest, e->name, cut - 1, x->path_flags,
                             &why);
  if (!parent) {
    ex_note(x, "xpar: refusing '%.*s': %s.\n", (int) e->name_len, e->name,
            xpar_path_reason(why));
    x->mismatches++;
    return NULL;
  }
  out = xpar_path_join_n(parent, e->name + cut, e->name_len - cut);
  xpar_free(parent);
  return out;
}

/*  Trim full-length components to leave room for the staging suffix.  */
static char * ex_stage_name(const char * path) {
  xpar_stat_t st;
  char * stem = xpar_stage_stem(path, sizeof ".xpar-stage-000" - 1);
  u32 i;
  for (i = 0; i < 1000; i++) {
    char * p = NULL;
    xpar_asprintf(&p, "%s.xpar-stage-%03" PRIu32, stem, i);
    if (xpar_lstat(p, &st) != 0) { xpar_free(stem);  return p; }
    xpar_free(p);
  }
  xpar_free(stem);
  return NULL;
}

static void ex_put_back(ex * x, const char * path, const char * backup) {
  if (xpar_put_back(path, backup) == 0) return;
  xpar_fprintf(xpar_stderr, "xpar: cannot restore '%s': %s; old copy at "
               "'%s'.\n", path, xpar_strerror(xpar_errno()), backup);
}

/*  Publish STAGE while keeping any old copy reachable.  */
static bool ex_replace(ex * x, char * stage, const char * path) {
  xpar_stat_t st;
  char * backup = NULL;
  const char * why;
  int err = 0;
  bool had = xpar_lstat(path, &st) == 0;
  u32 i;
  if (had && st.is_dir) { why = "it is a directory";  goto refuse; }
  if (had && !x->o->force) {
    why = "it exists and -f was not given";  goto refuse;
  }
  if (had) {
    for (i = 0; i < 1000; i++) {
      xpar_asprintf(&backup, "%s.xpar-old-%03" PRIu32, path, i);
      if (xpar_lstat(backup, &st) != 0) break;
      xpar_free(backup);  backup = NULL;
    }
    if (!backup) {
      why = "no backup name available";  goto refuse;
    }
    if (xpar_keep_aside(path, backup) != 0) {
      err = xpar_errno();  why = "cannot preserve old copy";
      goto refuse;
    }
  }
  if (xpar_rename(stage, path) != 0) {
    err = xpar_errno();  why = "cannot publish staged copy";
    if (backup) ex_put_back(x, path, backup);
    goto refuse;
  }
  if (xpar_fsync_dir(path) != 0) {
    err = xpar_errno();  why = "cannot sync parent directory";
    (void) xpar_rename(path, stage);
    if (backup) ex_put_back(x, path, backup);
    goto refuse;
  }
  if (backup && xpar_remove(backup) != 0)
    ex_note(x, "xpar: old destination remains at '%s'.\n", backup);
  xpar_free(backup);
  return true;

refuse:
  xpar_fprintf(xpar_stderr, "xpar: cannot write '%s': %s%s%s.\n", path, why,
               err ? ": " : "", err ? xpar_strerror(err) : "");
  xpar_remove(stage);
  xpar_free(backup);
  x->io_failures++;
  return false;
}

static bool ex_write_entry(ex * x, u32 idx, const char * path) {
  const xpar_entry * e = &x->mf.entry[idx];
  xpar_file * f;
  xpar_blake3_t h;
  u8 got[32];
  u8 * buf;
  u64 fo = 0, chunk = 1 << 20;
  u32 k;
  char * stage;
  { char * d = xpar_path_dir(path);
    if (xpar_mkdir_p(d, 0777) != 0) {
      xpar_free(d);  x->io_failures++;
      return false;
    }
    xpar_free(d); }
  stage = ex_stage_name(path);
  if (!stage) { x->io_failures++;  return false; }
  f = xpar_open(stage, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_EXCL |
                XPAR_O_NOFOLLOW);
  if (!f) FATAL_IO("Cannot create '%s': %s.", stage,
                   xpar_strerror(xpar_errno()));
  /*  Narrow modes before writing; privileged bits wait for metadata.  */
  if ((x->o->preserve & XPAR_PRES_MODE) && e->mode != XPAR_ABSENT_U32)
    (void) xpar_set_mode(stage, 1,
                         e->mode & XPAR_MODE_PERM &
                         ~(u32) (XPAR_MODE_SETUID | XPAR_MODE_SETGID |
                                 XPAR_MODE_STICKY));
  buf = (u8 *) xpar_alloc_raw((sz) chunk);
  if (x->auth_only) xpar_blake3_init_keyed(&h, x->key.k_file);
  else              xpar_blake3_init(&h);
  for (k = 0; k < e->extent_count; k++) {
    u64 left = e->extents[k].length, at = e->extents[k].stream_offset;
    while (left) {
      u64 take = MIN(left, chunk);
      if (!ex_read_stream(x, at, take, buf)) {
        /*  Remove the partial stage on read failure.  */
        xpar_xclose(f);
        xpar_remove(stage);
        xpar_free(stage);  xpar_free(buf);
        FATAL_IO("The set stream is missing bytes [%" PRIu64 ", %" PRIu64
                 ") that "
                 "'%.*s' needs; the data volume holding them is not "
                 "here.", at,
                 (at + take), (int) e->name_len,
                 e->name);
      }
      xpar_xwrite(f, buf, (sz) take);
      xpar_blake3_update(&h, buf, (sz) take);
      at += take;  left -= take;  fo += take;
    }
  }
  if (xpar_fsync(f) != 0) FATAL_IO("Cannot flush '%s'.", stage);
  xpar_xclose(f);
  xpar_free(buf);
  xpar_blake3_final(&h, got, 32);
  /*  Retry after lazy inner-code correction.  */
  if (xpar_memcmp(got, e->content_hash, 32)) {
    u64 lo = 0, hi = 0;
    for (k = 0; k < e->extent_count; k++) {
      u64 a = e->extents[k].stream_offset;
      u64 b = a + e->extents[k].length;
      if (!k || a < lo) lo = a;
      if (!k || b > hi) hi = b;
    }
    if (e->extent_count && ex_armour_apply(x, lo, hi)) {
      xpar_remove(stage);
      xpar_free(stage);
      return ex_write_entry(x, idx, path);
    }
  }
  if (xpar_memcmp(got, e->content_hash, 32)) {
    x->mismatches++;  x->hash_bad++;
    xpar_remove(stage);
    ex_note(x, "xpar: '%.*s' does not match its recorded hash; the damaged "
               "copy was not written. Run `xpar repair` first.\n",
            (int) e->name_len, e->name);
    xpar_free(stage);
    return false;
  }
  if (!ex_replace(x, stage, path)) {
    xpar_free(stage);
    return false;
  }
  x->entries++;  x->bytes += fo;
  xpar_free(stage);
  return true;
}

static void ex_link_entry(ex * x, u32 idx, const char * path) {
  const xpar_entry * e = &x->mf.entry[idx];
  i64 t = xpar_link_target(&x->mf, &x->nix, idx);
  char * src;
  if (t < 0) {
    ex_note(x, "xpar: '%.*s' names a hard-link target that is not in the "
               "manifest.\n", (int) e->name_len, e->name);
    x->mismatches++;
    return;
  }
  src = ex_resolve(x, &x->mf.entry[t]);
  if (!src) return;
  { char * d = xpar_path_dir(path);
    xpar_mkdir_p(d, 0777);
    xpar_free(d); }
  { const char * why = "this filesystem has no links";
    if (x->caps & XPAR_FS_HARDLINK) {
      char * stage = ex_stage_name(path);
      if (!stage) why = "no staging name is free";
      else if (xpar_link(src, stage) != 0) why = xpar_strerror(xpar_errno());
      else if (ex_replace(x, stage, path)) {
        x->links++;
        xpar_free(stage);  xpar_free(src);
        return;
      } else why = "the staged link could not be published";
      xpar_free(stage);
    }
    ex_skip(x, e, EX_SK_LINKCOPY, why); }
  ex_note(x, "xpar: %.*s: materialised-as-copy.\n", (int) e->name_len,
          e->name);
  x->copies++;
  xpar_free(src);
  (void) ex_write_entry(x, (u32) t, path);
}

static void ex_free(ex * x) {
  u32 i;
  for (i = 0; i < x->stream_count; i++) xpar_vset_close(x->stream_set[i]);
  xpar_free(x->stream_set);
  if (x->have_chain) {
    for (i = 0; i < x->chain.gen_count; i++)
      if (x->posix_tab && x->posix_tab[i])
        xpar_gchain_posix_free(x->posix_tab[i], x->posix_tab_count[i]);
    xpar_free(x->posix_tab);  xpar_free(x->posix_tab_count);
    xpar_free(x->owner);
    xpar_gchain_free(&x->chain);
  }
  for (i = 0; i < x->arm_count; i++) {
    xpar_armour_free(x->arm[i].a);  xpar_free(x->arm[i].done);
  }
  xpar_free(x->arm);  x->arm = NULL;  x->arm_count = x->arm_cap = 0;
  for (i = 0; i < x->vol_count; i++) xpar_volimg_close(&x->vol[i]);
  xpar_free(x->vol);
  for (i = 0; i < x->plain_count; i++) xpar_free(x->plain[i]);
  xpar_free(x->plain);
  for (i = 0; i < x->posix_count; i++) {
    u32 k;
    for (k = 0; k < x->posix[i].xattr_count; k++) {
      xpar_free(x->posix[i].xattrs[k].name);
      xpar_free(x->posix[i].xattrs[k].value);
    }
    xpar_free(x->posix[i].xattrs);
    xpar_free(x->posix[i].owner);  xpar_free(x->posix[i].group);
  }
  xpar_free(x->posix);
  if (x->have_layt) xpar_layt_free(&x->layt);
  xpar_nameidx_free(&x->nix);
  xpar_manifest_free(&x->mf);
  if (x->have_setd) xpar_setd_free(&x->sd);
  xpar_critset_free(&x->crit);
  xpar_free(x->dir);  xpar_free(x->dest);
  xpar_key_forget(&x->key, x->master);
}

/*  Open one stream reader per generation in the selected lineage.  */
static void ex_open_chain(ex * x) {
  u32 g, walked = 0, i;
  xpar_gchain_load(x->o, &x->chain);
  x->have_chain = true;
  x->selected = xpar_gchain_select(&x->chain,
                                    x->o->gen_count ? &x->o->gens[0] : NULL);
  g = x->selected;
  while (g != XPAR_GEN_NONE) {
    if (x->chain.gen[g].parent_missing)
      FATAL_FORMAT("Generation %" PRIu32
                   "'s parent is missing; extraction needs "
                   "the complete selected lineage.",
                   x->chain.gen[g].sd.generation);
    if (++walked > x->chain.gen_count)
      FATAL_FORMAT("The selected generation's ancestry is cyclic.");
    g = x->chain.gen[g].parent;
  }

  xpar_manifest_free(&x->mf);
  xpar_nameidx_free(&x->nix);
  xpar_gchain_manifest(&x->chain, x->selected, &x->mf, &x->owner);
  xpar_nameidx_build(&x->mf, &x->nix);
  x->posix_tab = (xpar_posix_rec **) xpar_calloc(
    x->chain.gen_count ? x->chain.gen_count : 1, sizeof(void *));
  x->posix_tab_count = (u32 *) xpar_calloc(
    x->chain.gen_count ? x->chain.gen_count : 1, sizeof(u32));
  for (i = 0; i < x->chain.gen_count; i++)
    x->posix_tab_count[i] = xpar_gchain_posix(&x->chain, i,
                                               &x->posix_tab[i]);

  x->stream_set = (xpar_vset **) xpar_calloc(walked, sizeof(void *));
  g = x->selected;
  while (g != XPAR_GEN_NONE) {
    xpar_options ro = *x->o;
    xpar_genref ref;
    char id[XPAR_SET_ID_LEN * 2 + 1];
    xpar_gchain_genref(&x->chain, g, &ref, id);
    ro.gens = &ref;  ro.gen_count = 1;  ro.chain = false;
    x->stream_set[x->stream_count] = xpar_vset_open(&ro);
    if (xpar_vset_setd(x->stream_set[x->stream_count])->layout ==
        XPAR_LAYOUT_SIDECAR)
      FATAL("Generation %" PRIu32
            " uses external files and cannot be extracted.",
            x->chain.gen[g].sd.generation);
    x->stream_count++;
    g = x->chain.gen[g].parent;
  }
}

/*  Validate the manifest for every extraction path.  */
static void ex_validate(ex * x) {
  xpar_mf_limits lim;
  xpar_mf_result res;
  xpar_mf_status s;
  xpar_gen_range * anc = NULL;
  u32 anc_count = 0;
  xpar_memset(&lim, 0, sizeof lim);
  lim.stream_base        = x->sd.stream_base;
  lim.stream_length      = x->sd.stream_length;
  lim.slice_size         = x->sd.slice_size;
  lim.posix_record_count = x->have_chain ? XPAR_ABSENT_U32
                                         : x->sd.posix_record_count;
  lim.path_flags         = x->path_flags;
  lim.align              = x->sd.align;
  if (x->have_chain) {
    u32 g = x->chain.gen[x->selected].parent, n = 0, k;
    while (g != XPAR_GEN_NONE) { n++; g = x->chain.gen[g].parent; }
    anc = (xpar_gen_range *) xpar_calloc(n ? n : 1, sizeof(xpar_gen_range));
    g = x->chain.gen[x->selected].parent;
    while (g != XPAR_GEN_NONE) {
      anc[n - ++anc_count].base = x->chain.gen[g].sd.stream_base;
      anc[n - anc_count].length = x->chain.gen[g].sd.stream_length;
      g = x->chain.gen[g].parent;
    }
    /*  The fill above is oldest first by writing the reverse lineage.  */
    for (k = 1; k < n; k++)
      FATAL_UNLESS("Generation stream ranges overlap.",
                   anc[k - 1].base + anc[k - 1].length <= anc[k].base);
    lim.ancestor = anc;  lim.ancestor_count = n;
  }
  s = xpar_manifest_validate(&x->mf, &lim, &res);
  if (s != XPAR_MF_OK) {
    const xpar_entry * e = &x->mf.entry[res.entry];
    /*  A name this destination cannot hold is data that cannot be
        reproduced here, not a malformed set.  */
    if (s == XPAR_MF_PATH && lim.path_flags)
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "Entry %" PRIu32 " ('%.*s'): %s under the naming rules in "
                 "force for this destination.", res.entry,
                 (int) e->name_len, e->name, xpar_mf_reason(s));
    FATAL_FORMAT("Entry %" PRIu32 " ('%.*s'): %s.", res.entry,
                 (int) e->name_len, e->name, xpar_mf_reason(s));
  }
  xpar_free(anc);
}

int xpar_op_extract(const xpar_options * o) {
  ex x;
  u32 i;
  int rc = XPAR_EXIT_OK;
  xpar_stat_t st;

  xpar_memset(&x, 0, sizeof x);
  x.o = o;  x.quiet = o->quiet;
  xpar_json_init(&x.js, o->json ? xpar_stdout : xpar_stderr, o->json);
  xpar_crc32c_init();
  xpar_critset_init(&x.crit);
  if (o->auth_key) {
    xpar_keyfile_load_or_die(o->auth_key, &x.key, x.master);
    x.key_loaded = true;
  }

  FATAL_UNLESS("Extract needs a set to work on.", o->set_ref.count > 0);
  {
    xpar_vset * guard = xpar_vset_open(o);
    xpar_vset_close(guard);
  }
  /*  Resolve entries beside the named set.  */
  x.dir  = o->set_ref.dir  ? xpar_strdup(o->set_ref.dir)
         : o->set_ref.home ? xpar_strdup(o->set_ref.home)
                           : xpar_path_dir(o->set_ref.vol[0]);
  x.dest = xpar_strdup(o->to_dir ? o->to_dir : ".");

  for (i = 0; i < o->set_ref.count; i++) {
    if (!ex_vol_open(&x, o->set_ref.vol[i])) continue;
    { const xpar_volimg * v = &x.vol[x.vol_count - 1];
      if (xpar_garm_is_archive(v->data, (sz) v->size))
        ex_open_armoured(&x, v);
      else
        ex_collect(&x, v->data, v->size);
    }
  }
  FATAL_UNLESS("Nothing in '%s' could be opened.", x.vol_count > 0, o->set);
  if (!ex_have_setd(&x))
    for (i = 0; i < x.vol_count; i++)
      ex_salvage(&x, x.vol[i].data, x.vol[i].size);

  ex_pick_setd(&x);
  ex_authenticate(&x);
  FATAL_UNLESS("This set uses external files; use verify or repair.",
               x.sd.layout != XPAR_LAYOUT_SIDECAR);

  if (x.sd.generation) ex_open_chain(&x);
  else ex_read_manifest(&x);
  { const xpar_crit_pkt * p = NULL;
    for (i = 0; i < x.crit.count && !p; i++)
      if (xpar_pkt_is(&x.crit.pkt[i].hdr, XPAR_T_LAYT) &&
          !xpar_memcmp(x.crit.pkt[i].hdr.set_id, x.set_id, XPAR_SET_ID_LEN))
        p = &x.crit.pkt[i];
    if (p && xpar_layt_read(p->body, (sz) p->body_len, &x.layt) == XPAR_OK)
      x.have_layt = true;
  }
  if (!x.stream_count && !x.strm) {
    FATAL_UNLESS_CODE(XPAR_EXIT_UNREPAIRABLE,
                      "This set has no volume layout.", x.have_layt);
    if (xpar_layt_tiles(&x.layt, x.sd.stream_length) != XPAR_OK)
      FATAL_FORMAT("The data volumes do not tile the stream.");
    for (i = 0; i < x.layt.count; i++) {
      xpar_vol * v = &x.layt.vol[i];
      char * path, * basename;
      u32 k;
      bool seen = false, damaged = false;
      if (v->kind != XPAR_VOL_DATA || !v->name) continue;
      path = ex_find_data(&x, v, &basename, &damaged);
      FATAL_UNLESS_CODE(XPAR_EXIT_UNREPAIRABLE,
                   "Data volume '%s' is missing; extraction needs the whole "
                   "stream. Run `xpar repair` to rebuild it.",
                   path != NULL, v->name);
      for (k = 0; k < x.vol_count && !seen; k++)
        if (!xpar_strcmp(x.vol[k].path, path)) seen = true;
      if (!seen && !ex_vol_open(&x, path))
        FATAL_IO("Data volume '%s' cannot be read.", path);
      if (damaged) {
        /*  Entry hashes determine whether damaged input survived.  */
        x.vol_damaged++;
        if (!o->quiet)
          xpar_fprintf(xpar_stderr,
                       "xpar: data volume '%s' is damaged; run `xpar repair` "
                       "first for a faithful copy\n", v->name);
      }
      if (xpar_strcmp(v->name, basename)) {
        /*  Report substituted volumes consistently across verbs.  */
        char * named = xpar_path_join(x.dir, v->name);
        xpar_stat_t vst;
        x.substituted++;
        if (!o->quiet)
          xpar_fprintf(xpar_stderr,
                       xpar_lstat(named, &vst) == 0
                         ? "xpar: data volume '%s' is damaged; intact copy "
                           "found as '%s'\n"
                         : "xpar: data volume '%s' is missing; using '%s'\n",
                       v->name, basename);
        xpar_free(named);
        xpar_free(v->name);
        v->name = basename;
      } else {
        xpar_free(basename);
      }
      xpar_free(path);
    }
  }

  if (o->to_stdout) {
    u64 chunk = 1 << 20, wrote = 0;
    u8 * buf;
    u32 regs = 0, only = 0, k, pass;
    const xpar_entry * e;
    xpar_blake3_t h;
    u8 got[32];
    for (i = 0; i < x.mf.count; i++)
      if (x.mf.entry[i].entry_type == XPAR_ENTRY_REGULAR) { regs++;  only = i; }
    FATAL_UNLESS("--stdout requires one entry, but the set has %" PRIu32
                 "; extract to a directory.", regs == 1, regs);
    /*  stdout creates no paths, so destination name rules do not apply.  */
    x.path_flags = 0;
    ex_validate(&x);
    e = &x.mf.entry[only];
    /*  Standard output cannot preserve metadata.  */
    { u32 q;
      for (q = 0; q < ARRAY_LEN(ex_require_map); q++)
        if (o->require & ex_require_map[q].bit)
          ex_skip(&x, e, ex_require_map[q].cls,
                  "standard output carries no metadata"); }
    buf = (u8 *) xpar_alloc_raw((sz) chunk);
    /*  Hash before emitting: a pipe cannot be taken back, so the bytes
        are proved against the manifest first and written second. The
        entry's own extents are followed, which a chained entry needs.  */
    for (pass = 0; pass < 2; pass++) {
      if (x.auth_only) xpar_blake3_init_keyed(&h, x.key.k_file);
      else             xpar_blake3_init(&h);
      for (k = 0; k < e->extent_count; k++) {
        u64 left = e->extents[k].length, at = e->extents[k].stream_offset;
        while (left) {
          u64 take = MIN(left, chunk);
          if (!ex_read_stream(&x, at, take, buf)) {
            xpar_free(buf);
            FATAL_IO("Set stream lacks required bytes [%" PRIu64 ", %" PRIu64
                     ").", at, at + take);
          }
          xpar_blake3_update(&h, buf, (sz) take);
          at += take;  left -= take;
        }
      }
      xpar_blake3_final(&h, got, 32);
      if (!xpar_memcmp(got, e->content_hash, 32)) break;
      /*  Retry after lazy inner-code correction, as file extraction does.  */
      if (pass == 0 && e->extent_count) {
        u64 lo = 0, hi = 0;
        for (k = 0; k < e->extent_count; k++) {
          u64 a = e->extents[k].stream_offset;
          u64 b = a + e->extents[k].length;
          if (!k || a < lo) lo = a;
          if (!k || b > hi) hi = b;
        }
        if (ex_armour_apply(&x, lo, hi)) continue;
      }
      xpar_free(buf);
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "Stream hash mismatch; nothing written. Run `xpar repair`.");
    }
    for (k = 0; k < e->extent_count; k++) {
      u64 left = e->extents[k].length, at = e->extents[k].stream_offset;
      while (left) {
        u64 take = MIN(left, chunk);
        if (!ex_read_stream(&x, at, take, buf)) {
          xpar_free(buf);
          FATAL_IO("Set stream lacks required bytes [%" PRIu64 ", %" PRIu64
                   ").", at, at + take);
        }
        xpar_xwrite(xpar_stdout, buf, (sz) take);
        at += take;  left -= take;  wrote += take;
      }
    }
    xpar_free(buf);
    if (xpar_flush(xpar_stdout) != 0 || xpar_error(xpar_stdout)) {
      xpar_fprintf(xpar_stderr, "xpar: cannot flush standard output: %s\n",
                   xpar_strerror(xpar_errno()));
      x.io_failures++;
    } else {
      x.entries++;  x.bytes += wrote;
    }
    goto summary;
  }

  { int err;
    if (xpar_mkdir_p(x.dest, 0777) != 0 && (err = xpar_errno(),
                                            xpar_lstat(x.dest, &st) != 0))
      FATAL_IO("Cannot create '%s': %s.", x.dest, xpar_strerror(err)); }
  FATAL_UNLESS("Extraction directory '%s' is a symbolic link; use its "
               "target path.",
               xpar_lstat(x.dest, &st) == 0 && !st.is_symlink, x.dest);
  x.caps = xpar_fs_caps(x.dest);
  /*  Windows and DOS always interpret backslashes and reserved names.  */
  x.path_flags = xpar_host_path_flags();
  /*  Apply portable naming rules to non-POSIX destinations.  */
  if (!(x.caps & (XPAR_FS_LINKID | XPAR_FS_HARDLINK | XPAR_FS_OWNER)) ||
      o->strict_names)
    x.path_flags |= XPAR_PATH_WIN | XPAR_PATH_NOCASE;

  ex_validate(&x);

  for (i = 0; i < x.mf.count; i++) {
    const xpar_entry * e = &x.mf.entry[i];
    char * p = ex_resolve_leaf(&x, e);
    if (p) {
      bool exists = xpar_lstat(p, &st) == 0;
      if (e->entry_type == XPAR_ENTRY_DIR) {
        FATAL_UNLESS("Destination '%s' exists and is not a directory.",
                     !exists || st.is_dir, p);
      } else {
        FATAL_UNLESS("Destination '%s' exists; -f overwrites it.",
                     !exists || o->force, p);
        FATAL_UNLESS("Refusing to replace destination directory '%s'.",
                     !exists || !st.is_dir, p);
      }
    }
    xpar_free(p);
  }

  for (i = 0; i < x.mf.count; i++) {
    const xpar_entry * e = &x.mf.entry[i];
    char * p;
    if (e->entry_type != XPAR_ENTRY_DIR) continue;
    p = ex_resolve(&x, e);
    if (!p) continue;
    { int err;
      if (xpar_mkdir_p(p, 0777) != 0 && (err = xpar_errno(),
                                         xpar_lstat(p, &st) != 0))
        FATAL_IO("Cannot create '%s': %s.", p, xpar_strerror(err)); }
    /*  Restrict private directories before populating them.  */
    if ((o->preserve & XPAR_PRES_MODE) && e->mode != XPAR_ABSENT_U32)
      (void) xpar_set_mode(p, 1,
                           (e->mode & XPAR_MODE_PERM &
                            ~(u32) (XPAR_MODE_SETUID | XPAR_MODE_SETGID |
                                    XPAR_MODE_STICKY)) | 0700);
    xpar_free(p);
  }
  for (i = 0; i < x.mf.count; i++) {
    const xpar_entry * e = &x.mf.entry[i];
    char * p;
    if (e->entry_type != XPAR_ENTRY_REGULAR) continue;
    p = ex_resolve_leaf(&x, e);
    if (!p) continue;
    (void) ex_write_entry(&x, i, p);
    xpar_free(p);
  }
  for (i = 0; i < x.mf.count; i++) {
    const xpar_entry * e = &x.mf.entry[i];
    char * p, * target;
    if (e->entry_type != XPAR_ENTRY_SYMLINK) continue;
    p = ex_resolve_leaf(&x, e);
    if (!p) continue;
    target = xpar_strndup((const char *) e->extra, e->extra_len);
    {
      char * stage = ex_stage_name(p);
      const char * why = NULL;
      if (!stage) why = "no staging name is free";
      else if (xpar_symlink(target, stage) != 0) why =
        xpar_strerror(xpar_errno());
      if (why) {
        xpar_fprintf(xpar_stderr,
                     "xpar: cannot create symlink '%.*s': %s.\n",
                     (int) e->name_len, e->name, why);
        x.io_failures++;
      } else (void) ex_replace(&x, stage, p);
      xpar_free(stage);
    }
    xpar_free(target);  xpar_free(p);
  }

  /*  Phase two: links only, no data.  */
  for (i = 0; i < x.mf.count; i++) {
    const xpar_entry * e = &x.mf.entry[i];
    char * p;
    if (e->entry_type != XPAR_ENTRY_HARDLINK) continue;
    p = ex_resolve_leaf(&x, e);
    if (!p) continue;
    ex_link_entry(&x, i, p);
    xpar_free(p);
  }

  /*  Apply metadata deepest-first using reversed name order.  */
  for (i = x.nix.count; i-- > 0;) {
    u32 q = x.nix.order[i];
    char * p = ex_resolve_leaf(&x, &x.mf.entry[q]);
    if (!p) continue;
    if (xpar_lstat(p, &st) == 0) ex_apply_meta(&x, q, p);
    xpar_free(p);
  }

summary:
  { u32 q;
    for (q = 0; q < ARRAY_LEN(ex_require_map); q++) {
      u8 cls = ex_require_map[q].cls;
      /*  --require makes a degradation fatal whether or not --preserve
          asked for the class, since it was not applied either way.  */
      if (!(o->require & ex_require_map[q].bit)) continue;
      if (!x.skip[cls]) continue;
      ex_note(&x, "xpar: --require: %" PRIu64 " entr%s lost %s.\n",
              x.skip[cls], x.skip[cls] == 1 ? "y" : "ies", ex_sk_name[cls]);
      rc = XPAR_EXIT_IO;
    }
  }
  if (x.io_failures) rc = XPAR_EXIT_IO;
  if (x.mismatches && rc == XPAR_EXIT_OK) rc = XPAR_EXIT_REPAIRABLE;
  /*  Extract cannot recover a surviving hash mismatch.  */
  if (x.hash_bad && (rc == XPAR_EXIT_OK || rc == XPAR_EXIT_REPAIRABLE))
    rc = XPAR_EXIT_UNREPAIRABLE;

  if (o->json) {
    xpar_json_begin(&x.js, "extract");
    xpar_json_u64(&x.js, "entries", x.entries);
    xpar_json_u64(&x.js, "bytes", x.bytes);
    xpar_json_u64(&x.js, "links", x.links);
    xpar_json_u64(&x.js, "materialised_as_copy", x.copies);
    xpar_json_u64(&x.js, "mismatches", x.mismatches);
    xpar_json_u64(&x.js, "volumes_substituted", x.substituted);
    xpar_json_u64(&x.js, "volumes_damaged", x.vol_damaged);
    xpar_json_u64(&x.js, "io_failures", x.io_failures);
    xpar_json_end(&x.js);
    xpar_json_summary(&x.js, rc == XPAR_EXIT_OK ? "ok" : "damaged", rc);
  } else
    ex_note(&x, "xpar: %" PRIu64 " %s, %" PRIu64 " bytes, %" PRIu64
            " link%s, %" PRIu64 " cop%s, "
                "%" PRIu64 " hash mismatch%s.\n", x.entries,
            x.entries == 1 ? "entry" : "entries",
            x.bytes,
            x.links, PLURAL(x.links),
            x.copies, x.copies == 1 ? "y" : "ies",
            x.mismatches,
            x.mismatches == 1 ? "" : "es");
  ex_free(&x);
  return rc;
}
