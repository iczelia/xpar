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

/*  Verification and the shared validated volume-set reader.  */

#include "ops.h"
#include "auth.h"
#include "chain.h"
#include "vset.h"

#include "armour.h"
#include "blake3.h"
#include "container.h"
#include "crc32c.h"
#include "json.h"
#include "manifest.h"
#include "pathname.h"
#include "plan.h"
#include "port-fs.h"
#include "port-thread.h"
#include "resync.h"
#include "slice.h"

/*  Monotone count of inner-code syndrome passes.  */

static u64 syndromes;

u64 xpar_verify_syndromes(void) { return syndromes; }

#define VERIFY_IOBUF  ((sz) 1 << 20)
#define VERIFY_BATCH  8u

/*  Feed padding and alignment gaps through the ordinary accumulator.  */
static const u8 zeros[4096];

static bool progress_on(const xpar_options * o) {
  if (o->quiet || o->json) return false;
  if (o->progress == XPAR_PROGRESS_ON) return true;
  if (o->progress == XPAR_PROGRESS_OFF) return false;
  return xpar_is_tty(xpar_stderr);
}

static bool color_on(const xpar_options * o) {
  return !o->json && (o->color == XPAR_COLOR_ALWAYS ||
         (o->color == XPAR_COLOR_AUTO && xpar_is_tty(xpar_stderr)));
}

/*  Map volume images where possible; 32-bit hosts need the read fallback.  */

typedef struct {
  char *    path;
  xpar_mmap map;
  xpar_mmap plain_map;
  xpar_file * plain_file;
  char *    plain_stage;
  u8 *      data;
  u64       size;
  bool      armoured;
  u8 *      plain;
  u64       plain_len, stream_offset, stream_length;
  xpar_armour_params armour_params;
} xpar_vimg;

static void vimg_load(xpar_vimg * v, const char * path) {
  xpar_file * f;
  i64 n;
  xpar_memset(v, 0, sizeof *v);
  v->path = xpar_strdup(path);
  v->map  = xpar_map(path);
  if (v->map.valid) { v->data = v->map.map;  v->size = v->map.size;  return; }
  f = xpar_open(path, XPAR_O_RDONLY);
  if (!f) FATAL_IO("Cannot open '%s': %s.", path,
                   xpar_strerror(xpar_errno()));
  n = xpar_size(f);
  if (n < 0) FATAL_IO("Cannot size '%s'.", path);
  v->size = (u64) n;
  FATAL_UNLESS("'%s' is too large to read without a mapping on this host.",
               v->size <= (u64) (sz) -1, path);
  v->data = (u8 *) xpar_alloc_raw(v->size ? (sz) v->size : 1);
  if (v->size) xpar_xread(f, v->data, (sz) v->size);
  xpar_xclose(f);
}

static void vimg_free(xpar_vimg * v) {
  if (v->plain_map.valid) xpar_unmap(&v->plain_map);
  if (v->plain_file) xpar_close(v->plain_file);
  if (v->plain_stage) xpar_remove(v->plain_stage);
  xpar_free(v->plain_stage);
  if (v->map.valid) xpar_unmap(&v->map);
  else xpar_free(v->data);
  xpar_free(v->path);
  xpar_memset(v, 0, sizeof *v);
}

typedef struct {
  u8  id[XPAR_SET_ID_LEN];
  u8  parent_id[XPAR_SET_ID_LEN];
  u32 generation;
  u32 parent, posix_count;
  u64 base, len;
} xpar_genid;

/*  A split volume whose bytes came from a file the layout does not name.  */
typedef struct {
  char * want;
  char * got;
  u32    vol;
  bool   present;   /*  The named file exists but is damaged.  */
} xpar_subst;

struct xpar_vset {
  xpar_vimg *  img;
  u32          img_count, img_cap;
  u8 **        plain;         /*  ARMG plaintexts; bodies point into them.  */
  u64 *        plain_len;
  bool *       plain_owned;
  u32          plain_count, plain_cap, archive_plain;
  u64          archive_plain_len;
  xpar_critset crit;
  xpar_key     key;
  u8           master[XPAR_BLAKE3_KEY_LEN];
  xpar_auth    auth;
  bool         key_loaded, keyed, have_auth, auth_only;
  u64          memory_budget;

  xpar_setd    setd;
  u8           set_id[XPAR_SET_ID_LEN];
  xpar_genid * gen;
  u32          gen_count, gen_target;

  xpar_manifest mf;
  xpar_occindex occ;
  u32 *         ext_first;    /*  Prefix sum of extent counts per entry.  */
  u8 *          ext_alias;    /*  1 where that extent names shared bytes.  */
  u32           ext_total;
  xpar_geom     geom;         /*  Y as SETD records it.  */
  xpar_geom     eg;           /*  Erasure geometry; Y is 0 when degraded.  */
  xpar_tagset   tagset;
  u32           have;         /*  XPAR_TAGS_* actually complete.  */
  xpar_layt     layt;
  bool          have_layt;
  xpar_subst *  subst;        /*  Substituted data volumes.  */
  u32           subst_count, subst_cap;
  u64           subst_damaged;/*  Named volumes needing rewrite.  */
  char *        dir;
  char * const * source;
  u64           recovery, recovery_gone;

  xpar_erasures er;
  u64 bad_slices, bad_entries, alias_bad, opaque_bad, missing_entries;
  u64 column_groups;          /* Distinct erasure patterns. */
  bool hash_sampled, hash_parallel;
  u8 * superseded;
  u8 * ignored_cell;
  xpar_resync_map * resync;
  u64 superseded_entries;
  u64 bytes_read, depth, armg_failed, armg_corrected;
  bool degraded;              /*  SLCL absent or incomplete.  */
  const u8 * strm;            /*  Armoured-layout STRM body.  */
  u64        strm_len;
  xpar_armour_params archive_ap;
  u32         archive_img;
  bool        have_archive_img;

  /*  Cache the current entry during sequential stream reads.  */
  xpar_file * fh;
  u32         fh_entry;
  bool        fh_open;
};

static void keep_plain(xpar_vset * s, u8 * p, u64 len, bool owned) {
  if (s->plain_count == s->plain_cap) {
    s->plain_cap = s->plain_cap ? s->plain_cap * 2 : 8;
    s->plain = (u8 **) xpar_realloc(s->plain,
                                    (sz) s->plain_cap * sizeof(u8 *));
    s->plain_len = (u64 *) xpar_realloc(s->plain_len,
                                    (sz) s->plain_cap * sizeof(u64));
    s->plain_owned = (bool *) xpar_realloc(s->plain_owned,
                                    (sz) s->plain_cap * sizeof(bool));
  }
  s->plain[s->plain_count] = p;
  s->plain_len[s->plain_count] = len;
  s->plain_owned[s->plain_count++] = owned;
}

static xpar_file * plain_stage_open(const char * archive, char ** path) {
  char * stem = NULL;
  xpar_file * f;
  xpar_asprintf(&stem, "%s.plain-", archive);
  f = xpar_stage_open(stem, XPAR_O_RDWR, 0, path);
  xpar_free(stem);
  if (!f)
    FATAL_IO("Cannot create a secure plaintext stage beside '%s': %s.",
             archive, xpar_strerror(xpar_errno()));
  return f;
}

/*  Demodulate whole-file armour frame by frame into private scratch.  */
static u8 * open_armoured_plain(xpar_vset * s, xpar_vimg * v,
                                xpar_armour * a,
                                const xpar_arm_prologue * pr,
                                xpar_armour_status * result) {
  u64 fp = xpar_armour_frame_plain(a), fd = xpar_armour_frame_disk(a);
  u64 frames = xpar_ceil_div(pr->plain_length, fp), f, at = 0;
  u8 * frame = NULL;
  FATAL_UNLESS("The armoured frame needs %" PRIu64 " bytes but -m permits %"
               PRIu64 "; "
               "raise -m to read this archive.",
               fd <= s->memory_budget && fd <= (u64) (sz) -1,
               fd,
               s->memory_budget);
  FATAL_UNLESS("The armoured frame geometry is invalid.",
               pr->plain_length != 0 &&
               (!frames || fd <= UINT64_MAX / frames) &&
               pr->armoured_length == frames * fd);
  v->plain_file = plain_stage_open(v->path, &v->plain_stage);
  for (f = 0; f < frames; f++) {
    u64 take = MIN(fp, pr->plain_length - at);
    xpar_xwrite(v->plain_file, v->data + 384 + f * fd, (sz) take);
    at += take;
  }
  if (xpar_flush(v->plain_file) != 0 || xpar_fsync(v->plain_file) != 0)
    FATAL_IO("Flushing plaintext stage for '%s' failed.", v->path);
  v->plain_map = xpar_map(v->plain_stage);
  FATAL_UNLESS("The plaintext stage for '%s' cannot be mapped on this host.",
               v->plain_map.valid, v->path);
  if (xpar_verify_packets_ok(v->plain_map.map, pr->plain_length, NULL)) {
    *result = XPAR_ARMOUR_CLEAN;
    return v->plain_map.map;
  }

  xpar_unmap(&v->plain_map);
  FATAL_UNLESS("The plaintext stage for '%s' cannot be truncated.",
               xpar_ftruncate(v->plain_file, 0) == 0, v->path);
  frame = (u8 *) xpar_alloc_raw((sz) fd);
  at = 0;  *result = XPAR_ARMOUR_FAILED;  syndromes++;
  for (f = 0; f < frames; f++) {
    xpar_armour_status st;
    u64 take = MIN(fp, pr->plain_length - at);
    xpar_memcpy(frame, v->data + 384 + f * fd, (sz) fd);
    st = xpar_armour_decode_frame(a, frame, NULL);
    if (st == XPAR_ARMOUR_CORRECTED) *result = XPAR_ARMOUR_CORRECTED;
    if (xpar_pwrite(v->plain_file, frame, (sz) take, at) != (sz) take)
      FATAL_IO("Writing corrected plaintext stage for '%s' failed.", v->path);
    at += take;
  }
  xpar_free(frame);
  if (xpar_fsync(v->plain_file) != 0)
    FATAL_IO("Flushing corrected plaintext stage for '%s' failed.", v->path);
  v->plain_map = xpar_map(v->plain_stage);
  FATAL_UNLESS("The corrected plaintext stage for '%s' cannot be mapped.",
               v->plain_map.valid, v->path);
  if (!xpar_verify_packets_ok(v->plain_map.map, pr->plain_length, NULL))
    *result = XPAR_ARMOUR_FAILED;
  else if (*result == XPAR_ARMOUR_FAILED)
    *result = XPAR_ARMOUR_CLEAN;
  return v->plain_map.map;
}

static void open_armoured_image(xpar_vset * s, xpar_vimg * v) {
  xpar_arm_prologue pr;
  xpar_armour_params ap;
  xpar_armour * a;
  u8 * plain;
  xpar_armour_status ast;
  int copy = -1;
  FATAL_UNLESS("The armoured prologue of '%s' cannot be recovered; run "
               "`xpar recover-prologue`.",
               xpar_garm_prologue(v->data, (sz) v->size, &pr, &copy),
               v->path);
  ap.symbol_bits = pr.symbol_bits;  ap.poly = pr.poly;
  ap.n = pr.n;  ap.k = pr.k;  ap.fcr = pr.fcr;  ap.prim = pr.prim;
  ap.depth = pr.depth;
  FATAL_UNLESS("The armoured prologue of '%s' names unusable parameters.",
               xpar_armour_check(&ap) == NULL, v->path);
  FATAL_UNLESS("The armoured region of '%s' is truncated.",
               v->size >= 384 && pr.armoured_length <= v->size - 384,
               v->path);
  FATAL_UNLESS("The armoured plaintext of '%s' is too large for this host.",
               pr.plain_length <= (u64) (sz) -1, v->path);
  xpar_gf_init();
  a = xpar_armour_new(&ap);
  FATAL_UNLESS("The armoured parameters of '%s' cannot be instantiated.",
               a != NULL, v->path);
  plain = open_armoured_plain(s, v, a, &pr, &ast);
  if (ast == XPAR_ARMOUR_CORRECTED) s->armg_corrected++;
  xpar_armour_free(a);
  FATAL_UNLESS("The armoured plaintext of '%s' is damaged past the inner "
               "code's capacity.", ast != XPAR_ARMOUR_FAILED, v->path);
  FATAL_UNLESS("The STRM range in '%s' is outside its plaintext.",
               pr.stream_offset <= pr.plain_length &&
               pr.stream_length <= pr.plain_length - pr.stream_offset,
               v->path);
  v->armoured = true;
  v->plain = plain;
  v->plain_len = pr.plain_length;
  v->stream_offset = pr.stream_offset;
  v->stream_length = pr.stream_length;
  v->armour_params = ap;
  keep_plain(s, plain, pr.plain_length, false);
  s->archive_plain = s->plain_count;
}

static bool correct_armoured_slice(xpar_vset * s, u64 slice) {
  xpar_armour * a = xpar_armour_new(&s->archive_ap);
  u64 fp, fd, lo, hi, first, last, f;
  bool ok = true;
  xpar_vimg * image = NULL;
  u8 * enc, * plain;
  if (!a || !s->have_archive_img || !s->strm) return false;
  image = &s->img[s->archive_img];
  if (!image) { xpar_armour_free(a); return false; }
  fp = xpar_armour_frame_plain(a);
  fd = xpar_armour_frame_disk(a);
  lo = (u64) (s->strm - image->plain) + slice * s->geom.slice_size;
  hi = MIN(lo + s->geom.slice_size, s->archive_plain_len);
  first = lo / fp;  last = hi ? (hi - 1) / fp : first;
  enc = (u8 *) xpar_alloc_raw((sz) fd);
  plain = (u8 *) xpar_alloc_raw((sz) fp);
  for (f = first; f <= last; f++) {
    u64 po = f * fp, have = MIN(fp, s->archive_plain_len - po);
    xpar_armour_status st;
    if (384 + f * fd > image->size || fd > image->size - (384 + f * fd)) {
      ok = false; break;
    }
    xpar_memcpy(enc, image->data + 384 + f * fd, (sz) fd);
    syndromes++;
    st = xpar_armour_decode_frame(a, enc, NULL);
    if (st == XPAR_ARMOUR_FAILED) { ok = false; break; }
    xpar_armour_extract(a, plain, have, enc);
    if (!image->plain_file ||
        xpar_pwrite(image->plain_file, plain, (sz) have, po) != (sz) have) {
      ok = false; break;
    }
    if (st == XPAR_ARMOUR_CORRECTED) s->armg_corrected++;
  }
  xpar_free(enc);  xpar_free(plain);  xpar_armour_free(a);
  return ok;
}

/*  Parse ARMG plaintext before inner decoding. Corrected bytes are accepted
    only when the same packet-checksum parse succeeds afterwards.  */

typedef struct { const xpar_key * key; } armg_ctx;

bool xpar_verify_packets_ok(const u8 * p, u64 n, const xpar_key * key) {
  u64 pos = 0;
  while (n - pos >= XPAR_PKT_HDR) {
    xpar_pkt h;
    xpar_status st;
    if (xpar_memcmp(p + pos, XPAR_PKT_MAGIC, 8) != 0) break;
    st = xpar_pkt_read(p + pos, n - pos, key, &h);
    if (st != XPAR_OK && st != XPAR_E_NEEDKEY) return false;
    pos += h.length;
  }
  if (!pos) return false;
  /*  Bytes after the last packet are zero frame padding.  */
  for (; pos < n; pos++) if (p[pos]) return false;
  return true;
}

static bool armg_check(void * ctx, const u8 * plain, u64 len) {
  return xpar_verify_packets_ok(plain, len, ((armg_ctx *) ctx)->key);
}

/*  Return owned ARMG plaintext, or NULL beyond correction capacity.  */
static const u8 * armg_plain(xpar_vset * s, const u8 * body, sz n,
                             u64 * out_len) {
  xpar_armg a;
  xpar_armour_params p;
  xpar_armour * ar;
  u8 * plain;
  armg_ctx ctx;
  xpar_armour_status rc;

  if (xpar_armg_read(body, n, &a) != XPAR_OK) return NULL;
  xpar_memset(&p, 0, sizeof p);
  p.symbol_bits = a.symbol_bits;  p.poly = a.poly;
  p.n     = a.n;    p.k    = a.k;
  p.fcr   = a.fcr;  p.prim = a.prim;
  p.depth = a.depth;
  if (xpar_armour_check(&p)) return NULL;
  ar    = xpar_armour_new(&p);
  plain = (u8 *) xpar_alloc_raw(a.plain_length ? (sz) a.plain_length : 1);
  ctx.key = s->keyed ? &s->key : NULL;

  xpar_armour_extract(ar, plain, a.plain_length, a.data);
  rc = xpar_verify_packets_ok(plain, a.plain_length, ctx.key)
         ? XPAR_ARMOUR_CLEAN : XPAR_ARMOUR_FAILED;
  if (rc != XPAR_ARMOUR_CLEAN) {
    u8 * region = (u8 *) xpar_alloc_raw((sz) a.armoured_length);
    xpar_memcpy(region, a.data, (sz) a.armoured_length);
    syndromes++;
    rc = xpar_armour_decode(ar, region, a.armoured_length, plain,
                            a.plain_length, armg_check, &ctx, NULL);
    xpar_free(region);
    if (rc == XPAR_ARMOUR_CORRECTED) s->armg_corrected++;
  }
  xpar_armour_free(ar);
  if (rc == XPAR_ARMOUR_FAILED) {
    s->armg_failed++;
    xpar_free(plain);
    return NULL;
  }
  keep_plain(s, plain, a.plain_length, true);
  *out_len = a.plain_length;
  return plain;
}

/*  ARMG bodies may be damaged, so accept a structurally valid header after
    body-checksum failure and let the inner decoder decide. Other packet
    types remain checksum-gated.  */
bool xpar_verify_next_armg(const u8 * buf, u64 size, const xpar_key * key,
                           u64 * pos, const u8 ** body, u64 * body_len) {
  u64 p = *pos;
  if (size < XPAR_PKT_HDR) return false;
  for (; p <= size - XPAR_PKT_HDR; p += XPAR_PKT_ALIGN) {
    xpar_pkt h;
    xpar_status st;
    if (xpar_memcmp(buf + p, XPAR_PKT_MAGIC, 8) != 0) continue;
    st = xpar_pkt_read(buf + p, size - p, key, &h);
    if (st != XPAR_OK && st != XPAR_E_CHECKSUM && st != XPAR_E_NEEDKEY)
      continue;
    /*  Never skip by a length whose checksum did not validate it.  */
    if (!xpar_pkt_is(&h, XPAR_T_ARMG)) {
      if (st == XPAR_OK) p += h.length - XPAR_PKT_ALIGN;
      continue;
    }
    *pos      = p + (st == XPAR_OK ? h.length : XPAR_PKT_ALIGN);
    *body     = buf + p + XPAR_PKT_HDR;
    *body_len = h.length - XPAR_PKT_HDR;
    return true;
  }
  return false;
}

static void scan_into(xpar_vset * s, const u8 * buf, u64 size,
                      bool resync, bool nested) {
  const xpar_key * key = s->keyed ? &s->key : NULL;
  xpar_scan sc;
  xpar_pkt hdr;
  const u8 * body;
  u64 off;
  xpar_scan_init(&sc, buf, size, key, resync);
  sc.accept_unverified_keyed = false;
  while (xpar_scan_next(&sc, &hdr, &body, &off)) {
    if (s->have_auth && !(hdr.flags & XPAR_PF_KEYED))
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "An authenticated volume contains an unkeyed packet.");
    /*  STRM uses slice checks; ARMG uses the recovery sweep below.  */
    if (xpar_pkt_is(&hdr, XPAR_T_STRM)) continue;
    if (xpar_pkt_is(&hdr, XPAR_T_ARMG)) continue;
    xpar_critset_add(&s->crit, &hdr, body);
  }
  /*  ARMG nesting is exactly one level.  */
  if (nested) return;
  { u64 pos = 0, blen = 0;
    while (xpar_verify_next_armg(buf, size, key, &pos, &body, &blen)) {
      u64 plen = 0;
      const u8 * pl = armg_plain(s, body, (sz) blen, &plen);
      if (pl) scan_into(s, pl, plen, false, true);
    } }
}

static void collect_gens(xpar_vset * s) {
  u32 i, j, n = 0;
  s->gen = (xpar_genid *) xpar_calloc(s->crit.count ? s->crit.count : 1,
                                      sizeof(xpar_genid));
  for (i = 0; i < s->crit.count; i++) {
    const xpar_crit_pkt * p = &s->crit.pkt[i];
    xpar_setd sd;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_SETD)) continue;
    xpar_status st = xpar_setd_read(p->body, (sz) p->body_len, &sd);
    if (st != XPAR_OK && st != XPAR_E_UNSUPPORTED) continue;
    xpar_memcpy(s->gen[n].id, p->hdr.set_id, XPAR_SET_ID_LEN);
    xpar_memcpy(s->gen[n].parent_id, sd.parent_set_id, XPAR_SET_ID_LEN);
    s->gen[n].generation = sd.generation;
    s->gen[n].parent     = XPAR_GEN_NONE;
    s->gen[n].posix_count= sd.posix_record_count;
    s->gen[n].base       = sd.stream_base;
    s->gen[n].len        = sd.stream_length;
    n++;
    xpar_setd_free(&sd);
  }
  s->gen_count = n;
  /*  Newest FILE owner shadows its ancestors.  */
  for (i = 1; i < n; i++) {
    xpar_genid t = s->gen[i];
    u32 k = i;
    while (k && s->gen[k - 1].generation < t.generation) {
      s->gen[k] = s->gen[k - 1];  k--;
    }
    s->gen[k] = t;
  }
  /*  Inheritance follows parent_set_id, not generation-number ordering.  */
  for (i = 0; i < n; i++) {
    xpar_genid * child = &s->gen[i];
    if (!child->generation) continue;
    for (j = 0; j < n; j++)
      if (j != i && !xpar_memcmp(child->parent_id, s->gen[j].id,
                                 XPAR_SET_ID_LEN)) {
        u64 end;
        child->parent = j;
        if (child->generation != s->gen[j].generation + 1 ||
            s->gen[j].len > (u64) -1 - s->gen[j].base) {
          child->parent = XPAR_GEN_NONE;
          break;
        }
        end = s->gen[j].base + s->gen[j].len;
        if (child->base == end) break;
        child->parent = XPAR_GEN_NONE;
        break;
      }
  }
}

static u32 pick_gen(const xpar_vset * s, const xpar_options * o) {
  u32 i, found = XPAR_GEN_NONE, matches = 0;
  if (!s->gen_count) FATAL_FORMAT("No set descriptor found in this set.");
  if (!o->gen_count) {
    /*  Multiple heads require explicit branch selection.  */
    for (i = 0; i < s->gen_count; i++) {
      u32 j;
      bool named = false;
      for (j = 0; j < s->gen_count; j++)
        if (s->gen[j].parent == i) { named = true;  break; }
      if (!named) { found = i;  matches++; }
    }
    if (matches != 1)
      FATAL_FORMAT("This chain is forked or disconnected; select a branch "
                   "with --generation=<set-id-prefix>.");
    return found;
  }
  for (i = 0; i < s->gen_count; i++) {
    const xpar_genref * g = &o->gens[0];
    if (g->by_id ? xpar_hex_prefix(s->gen[i].id, XPAR_SET_ID_LEN, g->id_prefix)
                 : (s->gen[i].generation == g->number)) {
      found = i;  matches++;
    }
  }
  if (matches > 1)
    FATAL_FORMAT("That generation selector is ambiguous; use a longer "
                 "set-id prefix.");
  if (matches == 1) return found;
  FATAL_FORMAT("This set has no such generation.");
  return 0;
}

static const xpar_crit_pkt * find_file(const xpar_vset * s,
                                       const u8 * fid, u32 * owner) {
  u32 g = s->gen_target, walked = 0;
  while (g != XPAR_GEN_NONE && walked++ < s->gen_count) {
    const xpar_crit_pkt * p = xpar_critset_find_file(
                                &s->crit, s->gen[g].id, fid);
    if (p) {
      *owner = g;
      return p;
    }
    g = s->gen[g].parent;
  }
  return NULL;
}

static void build_manifest(xpar_vset * s) {
  xpar_mf_limits lim;
  xpar_mf_result res;
  xpar_gen_range * anc;
  u32 * lineage;
  u32 f, na = 0;

  for (f = 0; f < s->setd.file_count; f++) {
    u32 owner = XPAR_GEN_NONE;
    const xpar_crit_pkt * p = find_file(s, s->setd.file_id[f], &owner);
    xpar_entry * e;
    if (!p)
      FATAL_FORMAT("Manifest entry %" PRIu32 " of %" PRIu32
                   " is missing from the supplied "
                   "volumes.", (f + 1),
                   s->setd.file_count);
    e = xpar_manifest_append(&s->mf);
    if (xpar_entry_read(p->body, (sz) p->body_len,
                        s->gen[owner].posix_count, e) != XPAR_OK)
      FATAL_FORMAT("Manifest entry %" PRIu32 " is malformed.",
                   (f + 1));
  }
  s->mf.stream_base   = s->setd.stream_base;
  s->mf.stream_length = s->setd.stream_length;
  s->mf.align         = s->setd.align;
  s->mf.slice_size    = s->setd.slice_size;

  anc = (xpar_gen_range *) xpar_calloc(s->gen_count ? s->gen_count : 1,
                                       sizeof(xpar_gen_range));
  lineage = (u32 *) xpar_calloc(s->gen_count ? s->gen_count : 1,
                                sizeof(u32));
  f = s->gen[s->gen_target].parent;
  while (f != XPAR_GEN_NONE && na < s->gen_count) {
    lineage[na++] = f;
    if (s->gen[f].generation && s->gen[f].parent == XPAR_GEN_NONE)
      FATAL_FORMAT("Generation %" PRIu32 "'s parent is missing or malformed.",
                   s->gen[f].generation);
    f = s->gen[f].parent;
  }
  if (s->gen[s->gen_target].generation && !na)
    FATAL_FORMAT("The selected generation's parent is missing or malformed.");
  FATAL_UNLESS("The selected generation's ancestry is cyclic.",
               f == XPAR_GEN_NONE);
  for (f = 0; f < na; f++) {
    u32 g = lineage[na - f - 1];
    anc[f].base = s->gen[g].base;
    anc[f].length = s->gen[g].len;
  }
  xpar_free(lineage);
  xpar_memset(&lim, 0, sizeof lim);
  lim.stream_base        = s->setd.stream_base;
  lim.stream_length      = s->setd.stream_length;
  lim.slice_size         = s->setd.slice_size;
  lim.align              = s->setd.align;
  lim.posix_record_count = s->setd.posix_record_count;
  lim.ancestor           = na ? anc : NULL;
  lim.ancestor_count     = na;
  if (xpar_manifest_validate(&s->mf, &lim, &res) != XPAR_MF_OK)
    FATAL_FORMAT("Manifest entry %" PRIu32 " is invalid: %s.",
                 res.entry, xpar_mf_reason(res.status));
  xpar_free(anc);
}

static void validate_identities(xpar_vset * s) {
  const u8 * file_key = s->keyed ? s->key.k_file : NULL;
  const u8 * set_key  = s->keyed ? s->key.k_set : NULL;
  const xpar_crit_pkt * setp;
  xpar_set_id_ctx sh;
  u8 got[XPAR_SET_ID_LEN];
  u32 i;
  if (s->have_auth && !s->keyed) return;  /*  Authenticity is unknown.  */
  for (i = 0; i < s->mf.count; i++) {
    xpar_file_id(&s->mf.entry[i], file_key, got);
    if (!xpar_ct_equal(got, s->mf.entry[i].file_id, sizeof got))
      FATAL_FORMAT("Manifest entry %" PRIu32
                   " has a forged or inconsistent file_id.",
                   i);
  }
  setp = xpar_critset_find(&s->crit, s->set_id, XPAR_T_SETD, 0);
  FATAL_UNLESS("The selected set descriptor disappeared during identity "
               "validation.", setp != NULL);
  xpar_set_id_begin(&sh, set_key, setp->body, (sz) setp->body_len);
  for (i = 0; i < s->setd.file_count; i++) {
    u32 owner = XPAR_GEN_NONE;
    const xpar_crit_pkt * fp = find_file(s, s->setd.file_id[i], &owner);
    FATAL_UNLESS("Manifest entry %" PRIu32 " disappeared during set identity "
                 "validation.", fp != NULL, i);
    xpar_set_id_update(&sh, fp->body, (sz) fp->body_len);
  }
  xpar_set_id_final(&sh, got);
  if (!xpar_ct_equal(got, s->set_id, sizeof got))
    FATAL_CODE(s->keyed ? XPAR_EXIT_AUTH : XPAR_EXIT_NOTFOUND,
               "The set_id does not match its canonical metadata.");
}

static void load_tables(xpar_vset * s) {
  u64 input = 0;
  u32 i, cps;
  for (i = 0; i < s->img_count; i++) input += s->img[i].size;
  cps = s->setd.cell_bytes
          ? (u32) xpar_ceil_div(s->setd.slice_size, s->setd.cell_bytes) : 0;
  if (!xpar_tagset_init(&s->tagset, s->geom.slice_count,
                        s->setd.slice_tag_len, cps, input))
    FATAL_FORMAT("The slice tables this set claims are larger than the "
                 "volumes that carry them.");
  for (i = 0; i < s->crit.count; i++) {
    const xpar_crit_pkt * p = &s->crit.pkt[i];
    if (xpar_memcmp(p->hdr.set_id, s->set_id, XPAR_SET_ID_LEN) != 0)
      continue;
    if (xpar_pkt_is(&p->hdr, XPAR_T_SLCR)) {
      xpar_slcr t;
      if (xpar_slcr_read(p->body, (sz) p->body_len, &t) == XPAR_OK &&
          xpar_tagset_slcr(&s->tagset, &t) != XPAR_OK)
        FATAL_FORMAT("Slice CRC table coverage overlaps or is out of range.");
      xpar_slcr_free(&t);
    } else if (xpar_pkt_is(&p->hdr, XPAR_T_SLTG)) {
      xpar_sltg t;
      if (xpar_sltg_read(p->body, (sz) p->body_len, &t) == XPAR_OK &&
          xpar_tagset_sltg(&s->tagset, &t) != XPAR_OK)
        FATAL_FORMAT("Slice tag table coverage overlaps or is out of range.");
      xpar_sltg_free(&t);
    } else if (xpar_pkt_is(&p->hdr, XPAR_T_SLCL)) {
      xpar_slcl t;
      if (xpar_slcl_read(p->body, (sz) p->body_len, s->setd.slice_size,
                         &t) == XPAR_OK) {
        if (t.cell_bytes != s->setd.cell_bytes ||
            xpar_tagset_slcl(&s->tagset, &t) != XPAR_OK)
          FATAL_FORMAT("Cell table geometry or coverage is malformed.");
      }
      xpar_slcl_free(&t);
    }
  }
  s->have = xpar_tagset_complete(&s->tagset);
  if (!(s->have & XPAR_TAGS_CRC) && s->geom.slice_count && !s->auth_only)
    FATAL_FORMAT("The slice checksum table is incomplete; the set cannot "
                 "be verified without it.");
  if (s->keyed && (!(s->have & XPAR_TAGS_TAG) || s->setd.slice_tag_len != 16))
    FATAL_CODE(XPAR_EXIT_AUTH,
               "An authenticated set must carry a complete 16-byte keyed "
               "slice-tag table.");
}

static void load_layt(xpar_vset * s) {
  u32 i;
  for (i = 0; i < s->crit.count; i++) {
    const xpar_crit_pkt * p = &s->crit.pkt[i];
    if (!xpar_pkt_is(&p->hdr, XPAR_T_LAYT)) continue;
    if (xpar_memcmp(p->hdr.set_id, s->set_id, XPAR_SET_ID_LEN) != 0)
      continue;
    if (xpar_layt_read(p->body, (sz) p->body_len, &s->layt) != XPAR_OK)
      continue;
    s->have_layt = true;
    break;
  }
  if (s->have_layt) {
    u64 axis = xpar_setd_recovery_limit(&s->setd);
    u8 * used = (u8 *) xpar_calloc((sz) axis, 1);
    for (i = 0; i < s->layt.count; i++) {
      const xpar_vol * v = &s->layt.vol[i];
      u64 e;
      if (v->kind != XPAR_VOL_RECOVERY) continue;
      if (!v->byte_length || v->byte_length > axis ||
          v->recovery_first > axis - v->byte_length) {
        xpar_free(used);
        FATAL_FORMAT("The volume layout names recovery outside its axis.");
      }
      for (e = v->recovery_first; e < v->recovery_first + v->byte_length;
           e++) if (used[e]) {
        xpar_free(used);
        FATAL_FORMAT("Recovery ranges overlap in the volume layout.");
      } else used[e] = 1;
    }
    xpar_free(used);
    if (s->setd.layout == XPAR_LAYOUT_ARMOURED) {
      for (i = 0; i < s->crit.count; i++)
        if (xpar_pkt_is(&s->crit.pkt[i].hdr, XPAR_T_RCVS) &&
            xpar_memcmp(s->crit.pkt[i].hdr.set_id, s->set_id,
                        XPAR_SET_ID_LEN) == 0) s->recovery++;
      return;
    }
    for (i = 0; i < s->layt.count; i++) {
      const xpar_vol * v = &s->layt.vol[i];
      char * path;
      xpar_stat_t st;
      u64 minimum;
      if (v->kind != XPAR_VOL_RECOVERY) continue;
      if (v->byte_length > axis || v->recovery_first > axis - v->byte_length ||
          s->recovery > axis - v->byte_length)
        FATAL_FORMAT("The volume layout names recovery exponents outside "
                     "the generation's axis.");
      s->recovery += v->byte_length;
      if (!v->name) continue;
      path = xpar_path_join(s->dir, v->name);
      /*  Missing recovery volumes do not count toward the budget.  */
      minimum = v->byte_length > ((u64) -1) /
                                  (s->setd.slice_size + 64)
                  ? (u64) -1
                  : v->byte_length * (s->setd.slice_size + 64);
      if (xpar_lstat(path, &st) != 0 || !st.is_regular ||
          st.size < minimum) {
        s->recovery -= v->byte_length;
        s->recovery_gone += v->byte_length;
      }
      xpar_free(path);
    }
    return;
  }
  /*  Without LAYT, count recovery packets actually present.  */
  for (i = 0; i < s->crit.count; i++)
    if (xpar_pkt_is(&s->crit.pkt[i].hdr, XPAR_T_RCVS) &&
        xpar_memcmp(s->crit.pkt[i].hdr.set_id, s->set_id,
                    XPAR_SET_ID_LEN) == 0) s->recovery++;
}

static void select_armoured_image(xpar_vset * s) {
  u32 i;
  for (i = 0; i < s->img_count; i++) {
    xpar_vimg * v = &s->img[i];
    xpar_scan sc;
    xpar_pkt h;
    const u8 * body;
    u64 off;
    if (!v->armoured || !v->plain) continue;
    xpar_scan_init(&sc, v->plain, v->plain_len,
                   s->keyed ? &s->key : NULL, false);
    sc.accept_unverified_keyed = !s->keyed;
    while (xpar_scan_next(&sc, &h, &body, &off)) {
      if (!xpar_pkt_is(&h, XPAR_T_SETD) ||
          xpar_memcmp(h.set_id, s->set_id, XPAR_SET_ID_LEN)) continue;
      s->archive_img = i;
      s->have_archive_img = true;
      s->archive_plain_len = v->plain_len;
      s->archive_ap = v->armour_params;
      s->strm = v->plain + v->stream_offset;
      s->strm_len = v->stream_length;
      return;
    }
  }
}

static bool split_image_seen(const xpar_vset * s, const char * path) {
  For(u32, i, s->img_count,
      if (!xpar_strcmp(s->img[i].path, path)) return true)
  return false;
}

static void split_image_add(xpar_vset * s, const char * path) {
  if (split_image_seen(s, path)) return;
  if (s->img_count == s->img_cap) {
    s->img_cap = s->img_cap ? s->img_cap * 2 : 8;
    s->img = (xpar_vimg *) xpar_realloc(
      s->img, (sz) s->img_cap * sizeof(xpar_vimg));
  }
  vimg_load(&s->img[s->img_count++], path);
}

/*  Match a damaged renamed volume by its wholly contained slice
    certificates; exclude evidence supplied by other volumes.  */
static u64 split_volume_score(xpar_vset * s, const xpar_vol * lv,
                              const char * path, u64 * tested) {
  xpar_file * f;
  u8 * buf;
  u64 slice, score = 0, z = s->geom.slice_size;
  *tested = 0;
  if (!(s->have & (XPAR_TAGS_CRC | XPAR_TAGS_TAG))) return 0;
  f = xpar_open(path, XPAR_O_RDONLY);
  if (!f) return 0;
  buf = (u8 *) xpar_alloc_raw((sz) z);
  for (slice = 0; slice < s->geom.slice_count; slice++) {
    u64 begin = slice * z;
    u64 have = MIN(z, s->geom.stream_length - begin);
    bool match = false;
    if (begin < lv->stream_offset ||
        begin - lv->stream_offset > lv->byte_length ||
        have > lv->byte_length - (begin - lv->stream_offset)) continue;
    xpar_memset(buf, 0, (sz) z);
    if (xpar_pread(f, buf, (sz) have,
                   begin - lv->stream_offset) != (sz) have) continue;
    if (s->have & XPAR_TAGS_TAG) {
      u8 got[16];
      if (s->keyed)
        xpar_slice_tag_keyed(&s->setd, slice, buf, s->key.k_slice,
                             got, s->tagset.t.tag_len);
      else
        xpar_slice_tag(&s->setd, slice, buf, got, s->tagset.t.tag_len);
      match = xpar_blake3_tag_equal(
        got, s->tagset.t.slice_tag + slice * s->tagset.t.tag_len,
        s->tagset.t.tag_len);
    } else if (s->have & XPAR_TAGS_CRC) {
      match = xpar_crc32c(0, buf, (sz) z) ==
              s->tagset.t.slice_crc[slice];
    }
    (*tested)++;
    if (match) score++;
  }
  xpar_free(buf);
  xpar_close(f);
  return score;
}

static void split_score_candidate(xpar_vset * s, const xpar_vol * lv,
                                  const char * path, const char * name,
                                  char ** best_path, char ** best_name,
                                  u64 * best, u32 * tied, u32 * candidates) {
  xpar_stat_t st;
  u64 tested, score;
  if (xpar_lstat(path, &st) != 0 || !st.is_regular ||
      st.size != lv->byte_length) return;
  (*candidates)++;
  score = split_volume_score(s, lv, path, &tested);
  if (!tested) score = 0;
  if (!*best_path || score > *best) {
    xpar_free(*best_path);  xpar_free(*best_name);
    *best_path = xpar_strdup(path);  *best_name = xpar_strdup(name);
    *best = score;  *tied = 1;
  } else if (score == *best) {
    (*tied)++;
  }
}

/*  Return the matched full path and in-memory replacement basename. Bound
    directory entries before hashing to preserve the DOS name limit.  */
static char * find_split_volume(xpar_vset * s, const xpar_vol * lv,
                                char ** found_name) {
  char * expected = xpar_path_join(s->dir, lv->name);
  char * best_path = NULL, * best_name = NULL;
  xpar_stat_t st;
  u64 best = 0;
  u32 tied = 0, candidates = 0;
  *found_name = NULL;
  if (xpar_lstat(expected, &st) == 0 && st.is_regular &&
      st.size == lv->byte_length &&
      xpar_vol_tag_match(expected, lv)) {
    *found_name = xpar_strdup(lv->name);
    return expected;
  }
  xpar_free(expected);
  if (lv->vol_tag) {
    xpar_dir * d = xpar_opendir(s->dir);
    const xpar_dirent * de;
    if (!d) return NULL;
    while ((de = xpar_readdir(d)) != NULL) {
      char * candidate;
      if (!de->is_regular || !xpar_strcmp(de->name, lv->name)) continue;
      candidate = xpar_path_join(s->dir, de->name);
      if (xpar_lstat(candidate, &st) == 0 && st.is_regular &&
          xpar_vol_tag_match(candidate, lv)) {
        *found_name = xpar_strdup(de->name);
        xpar_closedir(d);
        return candidate;
      }
      xpar_free(candidate);
    }
    xpar_closedir(d);
  }

  /*  Fall back to slice certificates; the expected basename gets no
      positional preference.  */
  expected = xpar_path_join(s->dir, lv->name);
  split_score_candidate(s, lv, expected, lv->name, &best_path, &best_name,
                        &best, &tied, &candidates);
  xpar_free(expected);
  {
    xpar_dir * d = xpar_opendir(s->dir);
    const xpar_dirent * de;
    if (d) {
      while ((de = xpar_readdir(d)) != NULL) {
        char * candidate;
        if (!de->is_regular || !xpar_strcmp(de->name, lv->name)) continue;
        candidate = xpar_path_join(s->dir, de->name);
        split_score_candidate(s, lv, candidate, de->name,
                              &best_path, &best_name, &best, &tied,
                              &candidates);
        xpar_free(candidate);
      }
      xpar_closedir(d);
    }
  }
  if (best_path && ((best > 0 && tied == 1) || candidates == 1)) {
    *found_name = best_name;
    return best_path;
  }
  xpar_free(best_path);  xpar_free(best_name);
  return NULL;
}

/*  Track renamed substitutes and damaged files they supersede.  */
static void subst_add(xpar_vset * s, const char * want, const char * got,
                      u32 vol) {
  xpar_subst * e;
  char * named = xpar_path_join(s->dir, want);
  xpar_stat_t st;
  if (s->subst_count == s->subst_cap) {
    s->subst_cap = s->subst_cap ? s->subst_cap * 2 : 4;
    s->subst = (xpar_subst *) xpar_realloc(
                 s->subst, (sz) s->subst_cap * sizeof *s->subst);
  }
  e = &s->subst[s->subst_count++];
  e->want    = xpar_strdup(want);
  e->got     = xpar_strdup(got);
  e->vol     = vol;
  e->present = xpar_lstat(named, &st) == 0 && st.is_regular;
  if (e->present) s->subst_damaged++;
  xpar_free(named);
}

static void load_owned_data(xpar_vset * s) {
  u32 i;
  if (s->setd.layout == XPAR_LAYOUT_ARMOURED) {
    FATAL_UNLESS("The armoured archive has no usable STRM body.",
                 s->strm && s->strm_len == s->setd.stream_length);
    return;
  }
  if (s->setd.layout != XPAR_LAYOUT_SPLIT) return;
  FATAL_UNLESS("A split set needs a volume layout.", s->have_layt);
  if (xpar_layt_tiles(&s->layt, s->setd.stream_length) != XPAR_OK)
    FATAL_FORMAT("The split data volumes do not tile the protected stream.");
  for (i = 0; i < s->layt.count; i++) {
    xpar_vol * lv = &s->layt.vol[i];
    char * path, * found_name;
    if (lv->kind != XPAR_VOL_DATA || !lv->name) continue;
    path = find_split_volume(s, lv, &found_name);
    if (path) {
      split_image_add(s, path);
      if (xpar_strcmp(lv->name, found_name)) {
        subst_add(s, lv->name, found_name, i);
        xpar_free(lv->name);
        lv->name = found_name;
      } else {
        xpar_free(found_name);
      }
      xpar_free(path);
    }
  }
}

/*  Rewrite damaged named volumes from their intact substitutes.  */
bool xpar_vset_rewrite_substituted(xpar_vset * s, const char ** reason) {
  u32 i;
  u8 * buf;
  bool ok = true;
  if (reason) *reason = NULL;
  if (!s->subst_damaged || !s->have_layt) return true;
  buf = (u8 *) xpar_alloc_raw(1u << 16);
  for (i = 0; i < s->subst_count && ok; i++) {
    const xpar_vol * v;
    char * path, * stage = NULL;
    xpar_file * f;
    u64 at = 0, left;
    if (!s->subst[i].present) continue;
    if (s->subst[i].vol >= s->layt.count) continue;
    v = &s->layt.vol[s->subst[i].vol];
    path = xpar_path_join(s->dir, s->subst[i].want);
    f = xpar_stage_open(path, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_TRUNC,
                        1, &stage);
    if (!f) {
      if (reason) *reason = "staging failed";
      ok = false;
    }
    left = ok ? v->byte_length : 0;
    while (left) {
      sz take = (sz) MIN(left, (u64) (1u << 16));
      if (!xpar_vset_read(s, v->stream_offset + at, buf, take)) {
        if (reason) *reason = "stream read failed";
        ok = false;  break;
      }
      xpar_xwrite(f, buf, take);
      at += take;  left -= take;
    }
    if (f) {
      if (ok && xpar_fsync(f) != 0) {
        if (reason) *reason = "flush failed";
        ok = false;
      }
      xpar_xclose(f);
      if (ok && (xpar_rename(stage, path) != 0 || xpar_fsync_dir(path) != 0)) {
        if (reason) *reason = "publish failed";
        ok = false;
      }
      if (!ok) xpar_remove(stage);
    }
    xpar_free(stage);  xpar_free(path);
  }
  xpar_free(buf);
  return ok;
}

/*  Extent canonicality is uniform, so its first byte classifies it.  */

static void classify(xpar_vset * s) {
  u32 i, total = 0;
  s->ext_first = (u32 *) xpar_calloc((sz) s->mf.count + 1, sizeof(u32));
  for (i = 0; i < s->mf.count; i++) {
    s->ext_first[i] = total;
    total += s->mf.entry[i].extent_count;
  }
  s->ext_first[s->mf.count] = total;
  s->ext_total = total;
  s->ext_alias = (u8 *) xpar_calloc(total ? total : 1, 1);
  for (i = 0; i < s->occ.count; i++) {
    const xpar_occurrence * o = &s->occ.occ[i];
    xpar_occurrence c;
    u64 run;
    if (!xpar_occindex_canonical(&s->occ, o->stream_offset, &c, &run))
      continue;
    if (c.entry != o->entry || c.file_offset != o->file_offset)
      s->ext_alias[s->ext_first[o->entry] + o->extent] = 1;
  }
}

static bool aliased(const xpar_vset * s, u32 entry, u32 ext) {
  return s->ext_alias[s->ext_first[entry] + ext] != 0;
}

typedef struct {
  xpar_vset * s;
  xpar_file * f;
  const xpar_resync_probe * probe;
  u8 * buf;
} v_confirm;

static bool v_confirm_at(void * user, u32 at, u64 physical) {
  v_confirm * c = (v_confirm *) user;
  const xpar_resync_probe * p = &c->probe[at];
  u8 got[XPAR_BLAKE3_OUT_LEN];
  u64 z = c->s->geom.slice_size;
  if (physical > UINT64_MAX - z ||
      xpar_pread(c->f, c->buf, (sz) z, physical) != (sz) z) return false;
  xpar_slice_tag(&c->s->setd, p->slice, c->buf, got,
                 c->s->tagset.t.tag_len);
  return xpar_blake3_tag_equal(
    got, c->s->tagset.t.slice_tag + p->slice * c->s->tagset.t.tag_len,
    c->s->tagset.t.tag_len);
}

static xpar_resync_probe * v_entry_probes(xpar_vset * s, u32 entry,
                                           u32 * count) {
  const xpar_entry * e = &s->mf.entry[entry];
  xpar_resync_probe * p = NULL;
  u32 cap = 0, n = 0, k;
  u64 file_off = 0, z = s->geom.slice_size;
  u64 gen_begin = s->geom.stream_base;
  u64 gen_end = gen_begin + s->geom.stream_length;
  for (k = 0; k < e->extent_count; k++) {
    const xpar_extent * x = &e->extents[k];
    u64 begin = MAX(x->stream_offset, gen_begin), end;
    if (aliased(s, entry, k) ||
        x->stream_offset > UINT64_MAX - x->length) goto next;
    end = MIN(x->stream_offset + x->length, gen_end);
    if (begin < end) {
      u64 rem = (begin - gen_begin) % z;
      u64 at = rem ? begin + (z - rem) : begin;
      for (; at <= end && end - at >= z; at += z) {
        xpar_occurrence o;
        u64 run, slice = (at - gen_begin) / z;
        if (!xpar_occindex_canonical(&s->occ, at, &o, &run) ||
            o.entry != entry || o.extent != k || run < z) continue;
        if (n == cap) {
          cap = cap ? cap * 2 : 16;
          p = (xpar_resync_probe *)
                xpar_realloc(p, cap * sizeof(xpar_resync_probe));
        }
        p[n].crc = s->tagset.t.slice_crc[slice];
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

static void v_resync_entry(xpar_vset * s, u32 entry,
                           const xpar_options * o, xpar_file * f,
                           u64 file_size, const char * path) {
  const xpar_entry * e = &s->mf.entry[entry];
  xpar_resync_map * map = &s->resync[entry];
  xpar_resync_probe * p;
  xpar_resync_result result;
  v_confirm confirm;
  u64 * located;
  u64 z = s->geom.slice_size, aligned = 0, confirmations = 0;
  u32 n, i, d;
  bool engage;
  if (map->searched) return;
  map->searched = true;
  if (o->resync == XPAR_RESYNC_OFF || !f || s->keyed) return;
  p = v_entry_probes(s, entry, &n);
  if (!n) { xpar_free(p);  return; }
  confirm.s = s;  confirm.f = f;  confirm.probe = p;
  confirm.buf = (u8 *) xpar_alloc_raw((sz) z);
  for (i = 0; i < n; i++)
    if (p[i].expected <= file_size && file_size - p[i].expected >= z &&
        xpar_pread(f, confirm.buf, (sz) z, p[i].expected) == (sz) z &&
        xpar_crc32c(0, confirm.buf, (sz) z) == p[i].crc) aligned++;
  engage = o->resync == XPAR_RESYNC_ALWAYS ||
           (o->resync == XPAR_RESYNC_AUTO &&
            (file_size != e->length || aligned * 2 < n));
  if (!engage || !(s->have & XPAR_TAGS_TAG) ||
      !s->tagset.t.slice_tag) goto done;
  if (!xpar_resync_search(f, file_size, z, p, n, o->resync_step,
                          o->resync_window, &result)) goto done;
  located = (u64 *) xpar_alloc_raw((sz) n * sizeof(u64));
  for (i = 0; i < n; i++) located[i] = UINT64_MAX;
  if (result.dominant && !result.overflow) {
    for (d = 0; d < result.count; d++) {
      if (d && result.delta[d].votes < 2) break;
      for (i = 0; i < n; i++) {
        u64 physical;
        if (located[i] != UINT64_MAX ||
            !xpar_resync_shift(p[i].expected, result.delta[d].delta,
                               &physical) ||
            physical > file_size || file_size - physical < z) continue;
        confirmations++;
        if (v_confirm_at(&confirm, i, physical)) located[i] = physical;
      }
    }
  } else if (o->resync == XPAR_RESYNC_ALWAYS && o->resync_exhaustive) {
    confirmations = xpar_resync_exhaustive(
      f, file_size, z, p, n, o->resync_step, o->resync_window,
      v_confirm_at, &confirm, located);
  } else if (result.candidates && !o->quiet) {
    xpar_fprintf(xpar_stderr,
                 "xpar: %s: misplaced slices have no dominant "
                 "displacement; use --resync=always "
                 "--resync-exhaustive to confirm all %" PRIu64 " candidates.\n",
                 path, result.candidates);
  }
  for (i = 0; i < n; i++)
    if (located[i] != UINT64_MAX)
      xpar_resync_map_add(map, p[i].expected, located[i]);
  if (map->count && !o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: %s: found %" PRIu32 " displaced slices with %" PRIu64 " strong "
                 "confirmations.\n", path, map->count,
                 confirmations);
  xpar_free(located);
done:
  xpar_free(confirm.buf);  xpar_free(p);
}


static void load_key(xpar_vset * s, const char * path) {
  xpar_keyfile_status st = xpar_keyfile_load(path, &s->key, s->master);
  if (st == XPAR_KEYFILE_OPEN) FATAL_IO("Cannot open key file '%s'.", path);
  if (st == XPAR_KEYFILE_EMPTY)
    FATAL_CODE(XPAR_EXIT_AUTH, "The key file is empty.");
  if (st != XPAR_KEYFILE_OK)
    FATAL_CODE(XPAR_EXIT_AUTH, "Reading key file '%s' failed.", path);
  s->key_loaded = true;
}

/*  The first archive pass accepts structurally valid NEEDKEY packets to
    reach AUTH. After accepting the key, repeat packet checks and inner
    decoding. STRM remains governed by slice tags.  */
static void authenticate_armoured_images(xpar_vset * s) {
  u32 i;
  for (i = 0; i < s->img_count; i++) {
    xpar_vimg * v = &s->img[i];
    xpar_armour * a;
    u64 fp, fd, frames, f, at;
    u8 * frame;
    bool failed = false, corrected = false;
    u32 j;
    if (!v->armoured ||
        xpar_verify_packets_ok(v->plain, v->plain_len, &s->key)) continue;

    a = xpar_armour_new(&v->armour_params);
    FATAL_UNLESS("The authenticated armoured parameters of '%s' cannot be "
                 "instantiated.", a != NULL, v->path);
    fp = xpar_armour_frame_plain(a);
    fd = xpar_armour_frame_disk(a);
    frames = xpar_ceil_div(v->plain_len, fp);
    frame = (u8 *) xpar_alloc_raw((sz) fd);
    at = 0;
    for (f = 0; f < frames; f++) {
      xpar_armour_status st;
      u64 take = MIN(fp, v->plain_len - at);
      xpar_memcpy(frame, v->data + 384 + f * fd, (sz) fd);
      syndromes++;
      st = xpar_armour_decode_frame(a, frame, NULL);
      if (st == XPAR_ARMOUR_FAILED) { failed = true; break; }
      if (st == XPAR_ARMOUR_CORRECTED) corrected = true;
      if (xpar_pwrite(v->plain_file, frame, (sz) take, at) != (sz) take)
        FATAL_IO("Writing authenticated plaintext stage for '%s' failed.",
                 v->path);
      at += take;
    }
    xpar_free(frame);
    xpar_armour_free(a);
    FATAL_UNLESS("The authenticated metadata in '%s' is damaged past the "
                 "inner code's capacity.", !failed, v->path);
    if (xpar_fsync(v->plain_file) != 0)
      FATAL_IO("Flushing authenticated plaintext stage for '%s' failed.",
               v->path);

    /*  Refresh both aliases after remapping corrected plaintext.  */
    for (j = 0; j < s->archive_plain; j++)
      if (s->plain[j] == v->plain) break;
    xpar_unmap(&v->plain_map);
    v->plain_map = xpar_map(v->plain_stage);
    FATAL_UNLESS("The authenticated plaintext stage for '%s' cannot be "
                 "remapped.", v->plain_map.valid, v->path);
    v->plain = v->plain_map.map;
    if (j < s->archive_plain) s->plain[j] = v->plain;
    FATAL_UNLESS("Authenticated metadata in '%s' exceeds inner-code "
                 "recovery.",
                 xpar_verify_packets_ok(v->plain, v->plain_len, &s->key),
                 v->path);
    if (corrected) s->armg_corrected++;
  }
}

/*  Preflight only fixed-size AUTH bodies; rescan every packet with MACs
    after accepting the key.  */
static void auth_preflight(xpar_vset * s) {
  u32 i;
  u64 archive_failed = s->armg_failed;
  u64 archive_corrected = s->armg_corrected;
  bool saw_auth = false;
  s->keyed = false;
  for (i = 0; i < s->archive_plain && !s->have_auth; i++) {
    xpar_scan sc;
    xpar_pkt h;
    const u8 * body;
    u64 off;
    /*  The prologue bounds tentative plaintext before authenticated scan.  */
    xpar_scan_init(&sc, s->plain[i], s->plain_len[i], NULL, true);
    sc.accept_unverified_keyed = true;
    while (xpar_scan_next(&sc, &h, &body, &off)) {
      xpar_auth a;
      if (!xpar_pkt_is(&h, XPAR_T_AUTH) || !(h.flags & XPAR_PF_KEYED))
        continue;
      if (xpar_auth_read(body, (sz) (h.length - XPAR_PKT_HDR), &a) != XPAR_OK)
        continue;
      saw_auth = true;
      if (s->key_loaded && xpar_auth_key_ok(&a, s->master)) {
        s->auth = a;  s->have_auth = true;  break;
      }
    }
  }
  for (i = 0; i < s->img_count && !s->have_auth; i++) {
    xpar_scan sc;
    xpar_pkt h;
    const u8 * body;
    u64 off;
    xpar_scan_init(&sc, s->img[i].data, s->img[i].size, NULL, true);
    sc.accept_unverified_keyed = true;
    while (xpar_scan_next(&sc, &h, &body, &off)) {
      xpar_auth a;
      if (!xpar_pkt_is(&h, XPAR_T_AUTH) || !(h.flags & XPAR_PF_KEYED))
        continue;
      if (xpar_auth_read(body, (sz) (h.length - XPAR_PKT_HDR), &a) != XPAR_OK)
        continue;
      saw_auth = true;
      if (s->key_loaded && xpar_auth_key_ok(&a, s->master)) {
        s->auth = a;  s->have_auth = true;  break;
      }
    }
    if (!s->have_auth) {
      u64 pos = 0, blen;
      while (xpar_verify_next_armg(s->img[i].data, s->img[i].size, NULL,
                                   &pos, &body, &blen)) {
        u64 plen = 0;
        const u8 * plain = armg_plain(s, body, (sz) blen, &plen);
        xpar_scan inner;
        if (!plain) continue;
        xpar_scan_init(&inner, plain, plen, NULL, false);
        inner.accept_unverified_keyed = true;
        while (xpar_scan_next(&inner, &h, &body, &off)) {
          xpar_auth a;
          if (!xpar_pkt_is(&h, XPAR_T_AUTH) ||
              !(h.flags & XPAR_PF_KEYED)) continue;
          if (xpar_auth_read(body, (sz) (h.length - XPAR_PKT_HDR), &a) !=
              XPAR_OK) continue;
          saw_auth = true;
          if (s->key_loaded && xpar_auth_key_ok(&a, s->master)) {
            s->auth = a;  s->have_auth = true;  break;
          }
        }
        if (s->have_auth) break;
      }
    }
  }
  /*  Never reuse unauthenticated preflight plaintext.  */
  for (i = s->archive_plain; i < s->plain_count; i++)
    if (s->plain_owned[i]) xpar_free(s->plain[i]);
  s->plain_count = s->archive_plain;
  /*  Discard tentative ARMG corrections, retaining whole-archive ones for
      later publication.  */
  s->armg_failed = archive_failed;
  s->armg_corrected = archive_corrected;
  if (!saw_auth) {
    if (s->key_loaded)
      FATAL_CODE(XPAR_EXIT_AUTH, "This set is not authenticated.");
    s->keyed = false;
    return;
  }
  if (!s->key_loaded) {
    FATAL_CODE(XPAR_EXIT_AUTH,
               "This set is authenticated; supply --auth-key=FILE.");
  }
  if (!s->have_auth)
    FATAL_CODE(XPAR_EXIT_AUTH, "The authentication key is wrong for this set.");
  s->keyed = true;
  s->auth_only = !s->auth.unkeyed_retained;
  authenticate_armoured_images(s);
}

xpar_vset * xpar_vset_open(const xpar_options * o) {
  xpar_vset * s = (xpar_vset *) xpar_calloc(1, sizeof *s);
  const xpar_crit_pkt * p;
  xpar_status st;
  u32 i;

  xpar_crc32c_init();
  xpar_critset_init(&s->crit);
  s->memory_budget = o->memory ? o->memory : xpar_plan_default_memory();
  if (o->auth_key) load_key(s, o->auth_key);

  FATAL_UNLESS("No set volumes to read.", o->set_ref.count > 0);
  s->img_cap = o->set_ref.count + 8;
  s->img = (xpar_vimg *) xpar_calloc(s->img_cap, sizeof(xpar_vimg));
  for (i = 0; i < o->set_ref.count; i++) {
    vimg_load(&s->img[i], o->set_ref.vol[i]);
    if (s->img[i].size >= 8 &&
        !xpar_memcmp(s->img[i].data, "XPAR2ARM", 8))
      open_armoured_image(s, &s->img[i]);
    s->img_count++;
  }
  auth_preflight(s);
  s->dir = o->set_ref.dir ? xpar_strdup(o->set_ref.dir)
                          : xpar_path_dir(o->set_ref.vol[0]);
  for (i = 0; i < s->img_count; i++)
    if (!s->img[i].armoured)
      scan_into(s, s->img[i].data, s->img[i].size,
                o->resync == XPAR_RESYNC_ALWAYS, false);
  for (i = 0; i < s->archive_plain; i++) {
    scan_into(s, s->plain[i], s->plain_len[i], false, false);
  }
  if (s->crit.conflicts)
    FATAL_FORMAT("Replicated packets verify but disagree.");

  collect_gens(s);
  s->gen_target = pick_gen(s, o);
  xpar_memcpy(s->set_id, s->gen[s->gen_target].id, XPAR_SET_ID_LEN);
  p = xpar_critset_find(&s->crit, s->set_id, XPAR_T_SETD, 0);
  if (!p) FATAL_FORMAT("The set descriptor is unreadable.");
  st = xpar_setd_read(p->body, (sz) p->body_len, &s->setd);
  if (st == XPAR_E_UNSUPPORTED)
    FATAL_FORMAT("This set requires a format feature this build does not "
                 "implement.");
  if (st != XPAR_OK)
    FATAL_FORMAT("The set descriptor is unreadable: %s.",
                 xpar_status_str(st));

  if (s->setd.layout == XPAR_LAYOUT_ARMOURED) select_armoured_image(s);

  if (!xpar_geom_from_setd(&s->setd, &s->geom))
    FATAL_FORMAT("The set descriptor's geometry is inconsistent.");

  {
    u32 g = s->gen_target, walked = 0;
    while (g != XPAR_GEN_NONE && walked++ < s->gen_count) {
      xpar_posix_rec * tab = NULL;
      if (xpar_posx_collect(&s->crit, s->gen[g].id,
                            s->gen[g].posix_count, &tab) != XPAR_OK)
        FATAL_FORMAT("Generation %" PRIu32
                     "'s POSX table has gaps, overlaps, or "
                     "invalid ranges.", s->gen[g].generation);
      xpar_posix_records_free(tab, s->gen[g].posix_count);
      g = s->gen[g].parent;
    }
  }

  build_manifest(s);
  validate_identities(s);
  s->resync = (xpar_resync_map *)
                xpar_calloc(s->mf.count ? s->mf.count : 1,
                            sizeof(xpar_resync_map));
  xpar_occindex_build(&s->mf, &s->occ);
  classify(s);
  load_tables(s);
  load_layt(s);
  load_owned_data(s);

  /*  Report SLCL loss because slice-granular fallback reduces tolerance.  */
  s->eg       = s->geom;
  s->degraded = s->geom.cell_bytes != 0 && !(s->have & XPAR_TAGS_CELL);
  if (s->degraded || !s->geom.cell_bytes) {
    s->eg.cell_bytes = 0;  s->eg.cells_per_slice = 1;
  }
  xpar_erasures_init(&s->er, s->geom.slice_count, s->eg.cells_per_slice);
  return s;
}

void xpar_vset_close(xpar_vset * s) {
  if (!s) return;
  if (s->fh_open) xpar_close(s->fh);
  xpar_erasures_free(&s->er);
  if (s->have_layt) xpar_layt_free(&s->layt);
  xpar_tagset_free(&s->tagset);
  xpar_occindex_free(&s->occ);
  For(u32, i, s->mf.count, xpar_resync_map_free(&s->resync[i]))
  xpar_manifest_free(&s->mf);
  xpar_setd_free(&s->setd);
  xpar_critset_free(&s->crit);
  For(u32, i, s->plain_count, if (s->plain_owned[i]) xpar_free(s->plain[i]))
  xpar_free(s->plain);
  xpar_free(s->plain_len);
  xpar_free(s->plain_owned);
  For(u32, i, s->img_count, vimg_free(&s->img[i]))
  xpar_free(s->img);       xpar_free(s->ext_first);
  xpar_free(s->ext_alias); xpar_free(s->gen);
  xpar_free(s->superseded); xpar_free(s->ignored_cell);
  xpar_free(s->resync);
  For(u32, i, s->subst_count,
      xpar_free(s->subst[i].want);  xpar_free(s->subst[i].got))
  xpar_free(s->subst);
  xpar_free(s->dir);
  xpar_key_forget(&s->key, s->master);
  xpar_free(s);
}

const xpar_setd * xpar_vset_setd(const xpar_vset * s) {
  return &s->setd;
}
const xpar_geom * xpar_vset_geom(const xpar_vset * s) {
  return &s->geom;
}
const xpar_geom * xpar_vset_egeom(const xpar_vset * s) {
  return &s->eg;
}
const xpar_manifest * xpar_vset_manifest(const xpar_vset * s) {
  return &s->mf;
}
const xpar_occindex * xpar_vset_occ(const xpar_vset * s) {
  return &s->occ;
}
const xpar_tags * xpar_vset_tags(const xpar_vset * s) {
  return &s->tagset.t;
}
const xpar_layt * xpar_vset_layt(const xpar_vset * s) {
  return s->have_layt ? &s->layt : NULL;
}
const xpar_erasures * xpar_vset_erasures(const xpar_vset * s) {
  return &s->er;
}
const char * xpar_vset_dir(const xpar_vset * s) { return s->dir; }
const u8 * xpar_vset_id(const xpar_vset * s) { return s->set_id; }
const xpar_key * xpar_vset_key(const xpar_vset * s) {
  return s->keyed ? &s->key : NULL;
}
bool xpar_vset_authenticated(const xpar_vset * s) { return s->have_auth; }
u32 xpar_vset_have_tables(const xpar_vset * s) { return s->have; }
u32 xpar_vset_volumes(const xpar_vset * s) { return s->img_count; }
u64 xpar_vset_recovery(const xpar_vset * s) { return s->recovery; }
u64 xpar_vset_recovery_total(const xpar_vset * s) {
  return s->recovery + s->recovery_gone;
}

bool xpar_vset_bind_sources(xpar_vset * s, const xpar_manifest * m) {
  u32 i;
  if (!m || m->count != s->mf.count || !m->source) return false;
  for (i = 0; i < m->count; i++) {
    const xpar_entry * a = &m->entry[i], * b = &s->mf.entry[i];
    if (!m->source[i] || a->name_len != b->name_len ||
        xpar_memcmp(a->name, b->name, a->name_len) ||
        a->entry_type != b->entry_type || a->length != b->length ||
        xpar_memcmp(a->content_hash, b->content_hash,
                    XPAR_BLAKE3_OUT_LEN))
      return false;
  }
  s->source = m->source;
  return true;
}

/*  Writer read-back strictly parses every emitted byte, including recovery
    payloads that normal verify skips. Junk, truncation, duplicate exponents
    and wrong VOLH metadata are write failures.  */
bool xpar_verify_written_volume(const char * path, const xpar_key * key,
                                const u8 * set_id, u32 volume_index,
                                u32 volume_kind, u64 first, u64 count,
                                u64 slice_size) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  u8 head[XPAR_PKT_HDR];
  u64 size, pos = 0, found = 0;
  bool have_volh = false, ok = true;
  i64 signed_size;
  if (!f) return false;
  signed_size = xpar_size(f);
  if (signed_size < 0) { xpar_close(f); return false; }
  size = (u64) signed_size;
  while (pos < size) {
    xpar_pkt h;
    xpar_status st;
    u8 * packet;
    u64 len, body_len;
    if (size - pos < XPAR_PKT_HDR ||
        xpar_xread(f, head, sizeof head) != sizeof head ||
        xpar_memcmp(head, XPAR_PKT_MAGIC, 8) != 0) {
      ok = false; break;
    }
    len = xpar_rd64(head + 8);
    if (len < XPAR_PKT_HDR || len % XPAR_PKT_ALIGN ||
        len > XPAR_PKT_LEN_MAX || len > size - pos ||
        len > (u64) (sz) -1) {
      ok = false; break;
    }
    packet = (u8 *) xpar_alloc_raw((sz) len);
    xpar_memcpy(packet, head, sizeof head);
    body_len = len - XPAR_PKT_HDR;
    if (body_len && xpar_xread(f, packet + XPAR_PKT_HDR,
                               (sz) body_len) != (sz) body_len) {
      xpar_free(packet); ok = false; break;
    }
    st = xpar_pkt_read(packet, len, key, &h);
    if (st != XPAR_OK ||
        xpar_memcmp(h.set_id, set_id, XPAR_SET_ID_LEN) != 0) {
      xpar_free(packet); ok = false; break;
    }
    if (xpar_pkt_is(&h, XPAR_T_VOLH)) {
      xpar_volh vh;
      if (have_volh ||
          xpar_volh_read(packet + XPAR_PKT_HDR, (sz) body_len, &vh) !=
            XPAR_OK ||
          vh.volume_index != volume_index || vh.volume_kind != volume_kind) {
        xpar_free(packet); ok = false; break;
      }
      have_volh = true;
    } else if (xpar_pkt_is(&h, XPAR_T_RCVS)) {
      xpar_rcvs r;
      if (volume_kind != XPAR_VOL_RECOVERY ||
          xpar_rcvs_read(packet + XPAR_PKT_HDR, (sz) body_len, slice_size,
                         &r) != XPAR_OK ||
          found >= count || r.exponent < first ||
          r.exponent - first != found) {
        xpar_free(packet); ok = false; break;
      }
      found++;
    }
    xpar_free(packet);
    pos += len;
  }
  if (!have_volh || found != count) ok = false;
  if (xpar_close(f) != 0) ok = false;
  return ok;
}

typedef struct {
  xpar_vset * s;
  xpar_options * options;
  xpar_json * json;
  char * path;
  const xpar_key * key;
  const u8 * set_id;
  u32 volume_index, volume_kind;
  u64 first, count, slice_size;
  int result;
} writer_check;

static void writer_check_job(sz index, void * arg) {
  writer_check * c = (writer_check *) arg + index;
  if (index == 0) {
    c->result = xpar_vset_check(c->s, c->options, c->json);
    return;
  }
  c->result = xpar_verify_written_volume(
                c->path, c->key, c->set_id, c->volume_index,
                c->volume_kind, c->first, c->count, c->slice_size)
                ? XPAR_EXIT_OK : XPAR_EXIT_REPAIRABLE;
}

/*  Writer verification uses the public resolver and stream checker.  */
static void verify_written_set_at(const xpar_options * o,
                                  const char * index_path,
                                  const xpar_genref * generation,
                                  const xpar_manifest * sources,
                                  bool exact_file) {
  xpar_options ro = *o;
  xpar_vset * s;
  xpar_json js;
  const xpar_layt * layt;
  const xpar_setd * setd;
  writer_check * check = NULL;
  u32 check_count = 1;
  int rc;
  xpar_memset(&ro.set_ref, 0, sizeof ro.set_ref);
  ro.verb = XPAR_VERB_VERIFY;
  ro.set = (char *) index_path;
  ro.chain = false;
  ro.gens = (xpar_genref *) generation;
  ro.gen_count = generation != NULL;
  ro.json = false;  ro.quiet = true;
  ro.fast = false;
  if (exact_file) {
    ro.set_ref.vol = (char **) xpar_calloc(1, sizeof(char *));
    ro.set_ref.vol[0] = xpar_strdup(index_path);
    ro.set_ref.count = 1;
  } else {
    xpar_cli_resolve_set(index_path, &ro.set_ref);
  }
  s = xpar_vset_open(&ro);
  if (sources && !xpar_vset_bind_sources(s, sources))
    FATAL_CODE(XPAR_EXIT_INTERNAL,
               "internal: the source manifest differs from the set just "
               "written.");
  xpar_json_init(&js, xpar_stderr, false);
  layt = xpar_vset_layt(s);
  setd = xpar_vset_setd(s);
  if (setd->layout != XPAR_LAYOUT_ARMOURED) {
    u32 i;
    check_count++;
    for (i = 0; layt && i < layt->count; i++)
      if (layt->vol[i].kind == XPAR_VOL_RECOVERY) check_count++;
  }
  check = (writer_check *) xpar_calloc(check_count, sizeof(*check));
  check[0].s = s;  check[0].options = &ro;  check[0].json = &js;
  if (check_count > 1) {
    u32 at = 1, i;
    int total = o->jobs > 0 ? o->jobs : xpar_cpu_count();
    int outer = MIN(total, (int) check_count);
    xpar_pool * pool;
    check[at].path = xpar_strdup(index_path);
    check[at].key = xpar_vset_key(s);
    check[at].set_id = xpar_vset_id(s);
    check[at].volume_index = XPAR_VOL_STANDALONE;
    check[at].volume_kind = XPAR_VOL_INDEX;
    check[at].slice_size = setd->slice_size;
    at++;
    for (i = 0; layt && i < layt->count; i++) {
      const xpar_vol * v = &layt->vol[i];
      if (v->kind != XPAR_VOL_RECOVERY) continue;
      check[at].path = xpar_path_join(xpar_vset_dir(s), v->name);
      check[at].key = xpar_vset_key(s);
      check[at].set_id = xpar_vset_id(s);
      check[at].volume_index = i;
      check[at].volume_kind = XPAR_VOL_RECOVERY;
      check[at].first = v->recovery_first;
      check[at].count = v->byte_length;
      check[at].slice_size = setd->slice_size;
      at++;
    }
    ro.jobs = MAX(1, total - outer + 1);
    pool = xpar_pool_create(outer);
    xpar_pool_run(pool, check_count, writer_check_job, check);
    xpar_pool_destroy(pool);
  } else {
    writer_check_job(0, check);
  }
  rc = check[0].result;
  for (u32 i = 1; i < check_count; i++)
    if (check[i].result != XPAR_EXIT_OK) rc = check[i].result;
  if (xpar_vset_recovery(s) != xpar_vset_recovery_total(s)) {
    xpar_fprintf(xpar_stderr,
                 "xpar: internal read-back found only %" PRIu64 " of %" PRIu64 " "
                 "recovery slices in '%s'.\n",
                 xpar_vset_recovery(s),
                 xpar_vset_recovery_total(s),
                 index_path);
    rc = XPAR_EXIT_REPAIRABLE;
  }
  for (u32 i = 0; i < check_count; i++) xpar_free(check[i].path);
  xpar_free(check);
  xpar_vset_close(s);
  xpar_setref_free(&ro.set_ref);
  if (rc != XPAR_EXIT_OK)
    FATAL_CODE(XPAR_EXIT_INTERNAL,
               "internal: the set just written does not verify through "
               "the public reader (status %d).", rc);
}

void xpar_verify_written_set_at(const xpar_options * o,
                                const char * index_path,
                                const xpar_genref * generation) {
  verify_written_set_at(o, index_path, generation, NULL, false);
}

void xpar_verify_written_archive_at(const xpar_options * o,
                                    const char * path,
                                    const xpar_genref * generation) {
  verify_written_set_at(o, path, generation, NULL, true);
}

void xpar_verify_written_set(const xpar_options * o, const char * index_path) {
  xpar_verify_written_set_at(o, index_path, NULL);
}

void xpar_verify_written_set_sources(const xpar_options * o,
                                     const char * index_path,
                                     const xpar_manifest * sources) {
  verify_written_set_at(o, index_path, NULL, sources, false);
}
u64 xpar_vset_bad_cells(const xpar_vset * s) {
  return s->er.bad_count;
}
u64 xpar_vset_volumes_to_rewrite(const xpar_vset * s) {
  return s->subst_damaged;
}
u64 xpar_vset_bad_slices(const xpar_vset * s) {
  return s->bad_slices;
}
u64 xpar_vset_bad_entries(const xpar_vset * s) {
  return s->bad_entries;
}
u64 xpar_vset_alias_bad(const xpar_vset * s) { return s->alias_bad; }
u64 xpar_vset_max_depth(const xpar_vset * s) { return s->depth; }
u64 xpar_vset_bytes_read(const xpar_vset * s) {
  return s->bytes_read;
}
u64 xpar_vset_inner_corrected(const xpar_vset * s) {
  return s->armg_corrected;
}

const u8 * xpar_vset_volume(const xpar_vset * s, u32 i, u64 * size) {
  if (i >= s->img_count) return NULL;
  *size = s->img[i].size;
  return s->img[i].data;
}

const char * xpar_vset_volume_path(const xpar_vset * s, u32 i) {
  return i < s->img_count ? s->img[i].path : NULL;
}

const u8 * xpar_vset_rcvs(const xpar_vset * s, u64 exponent, u64 * len) {
  const xpar_crit_pkt * p = xpar_critset_find(&s->crit, s->set_id,
                                               XPAR_T_RCVS, exponent);
  xpar_rcvs r;
  if (len) *len = 0;
  if (!p || xpar_rcvs_read(p->body, (sz) p->body_len,
                            s->setd.slice_size, &r) != XPAR_OK)
    return NULL;
  if (len) *len = r.length;
  return r.data;
}

bool xpar_vset_armoured(const xpar_vset * s, const u8 ** plain,
                        u64 * plain_len, u64 * strm_offset,
                        xpar_armour_params * ap, const char ** path) {
  const xpar_vimg * image;
  if (!s->have_archive_img || !s->strm) return false;
  image = &s->img[s->archive_img];
  if (plain) *plain = image->plain;
  if (plain_len) *plain_len = s->archive_plain_len;
  if (strm_offset) *strm_offset = (u64) (s->strm - image->plain);
  if (ap) *ap = s->archive_ap;
  if (path) *path = image->path;
  return true;
}

/*  Rebuild only uncovered cell runs; overlapping SLCL ranges are invalid.  */
bool xpar_vset_cell_covered(const xpar_vset * s, u64 slice) {
  if (!s->tagset.seen_cell || slice >= s->geom.slice_count) return false;
  return s->tagset.seen_cell[slice] != 0;
}

/*  Read canonical stream bytes; zero-fill gaps and return false.  */

static char * entry_path(const xpar_vset * s, u32 entry) {
  if (s->source) return xpar_strdup(s->source[entry]);
  return xpar_path_join_n(s->dir, s->mf.entry[entry].name,
                          s->mf.entry[entry].name_len);
}

static xpar_file * entry_handle(xpar_vset * s, u32 entry) {
  char * path;
  if (s->fh_open && s->fh_entry == entry) return s->fh;
  if (s->fh_open) { xpar_close(s->fh);  s->fh_open = false; }
  path = entry_path(s, entry);
  s->fh = xpar_open(path, XPAR_O_RDONLY | XPAR_O_NOFOLLOW);
  xpar_free(path);
  if (!s->fh) return NULL;
  s->fh_open = true;  s->fh_entry = entry;
  return s->fh;
}

bool xpar_vset_read(xpar_vset * s, u64 off, u8 * buf, u64 len) {
  u64 end = s->geom.stream_base + s->geom.stream_length;
  bool ok = true;
  if (s->setd.layout == XPAR_LAYOUT_ARMOURED) {
    u64 rel;
    xpar_memset(buf, 0, (sz) len);
    if (off < s->geom.stream_base) return false;
    rel = off - s->geom.stream_base;
    if (rel >= s->strm_len) return off >= end;
    {
      u64 take = MIN(len, s->strm_len - rel);
      xpar_memcpy(buf, s->strm + rel, (sz) take);
      return take == len || off + take >= end;
    }
  }
  if (s->setd.layout == XPAR_LAYOUT_SPLIT) {
    u64 rel;
    xpar_memset(buf, 0, (sz) len);
    if (off < s->geom.stream_base) return false;
    rel = off - s->geom.stream_base;
    while (len && rel < s->geom.stream_length) {
      const xpar_vol * lv = NULL;
      u32 i, j;
      u64 take;
      for (i = 0; i < s->layt.count; i++) {
        const xpar_vol * q = &s->layt.vol[i];
        if (q->kind == XPAR_VOL_DATA && rel >= q->stream_offset &&
            rel - q->stream_offset < q->byte_length) { lv = q; break; }
      }
      if (!lv) { ok = false; break; }
      take = MIN(len, lv->byte_length - (rel - lv->stream_offset));
      for (j = 0; j < s->img_count; j++) {
        const char * base = xpar_path_base(s->img[j].path);
        if (lv->name && !xpar_strcmp(base, lv->name)) break;
      }
      if (j == s->img_count ||
          rel - lv->stream_offset > s->img[j].size ||
          take > s->img[j].size - (rel - lv->stream_offset)) {
        ok = false;
      } else {
        xpar_memcpy(buf, s->img[j].data + rel - lv->stream_offset,
                    (sz) take);
      }
      rel += take;  buf += take;  len -= take;
    }
    if (len) xpar_memset(buf, 0, (sz) len);
    /*  Success is measured against this subrange, not the whole stream.  */
    return ok && (len == 0 || rel >= s->geom.stream_length);
  }
  while (len) {
    xpar_span sp;
    xpar_file * f;
    u64 take, physical;
    if (off >= end) {
      /*  Past L is the zero padding of the final slice.  */
      xpar_memset(buf, 0, (sz) len);
      return ok;
    }
    if (!xpar_stream_locate(&s->occ, off, &sp)) {
      /*  Zero alignment padding only to the next extent.  */
      take = MIN(len, end - off);
      take = MIN(take, xpar_occindex_next(&s->occ, off, off + take) - off);
      if (!take) take = MIN(len, end - off);
      xpar_memset(buf, 0, (sz) take);
      off += take;  buf += take;  len -= take;
      continue;
    }
    take = MIN(len, sp.length);
    if (s->resync[sp.entry].count) {
      u64 left = s->geom.slice_size -
                 sp.file_offset % s->geom.slice_size;
      take = MIN(take, left);
    }
    f = entry_handle(s, sp.entry);
    physical = sp.file_offset;
    xpar_resync_map_shift(&s->resync[sp.entry], sp.file_offset, &physical);
    if (!f || xpar_pread(f, buf, (sz) take, physical) != take) {
      xpar_memset(buf, 0, (sz) take);
      ok = false;
    }
    off += take;  buf += take;  len -= take;
  }
  return ok;
}

/*  Canonical extents follow a validated monotone high-water mark, so one
    running checksum per cell and slice suffices.  */

typedef struct {
  xpar_vset * s;
  u64 pos;
  u32 cell_crc, slice_crc;
  xpar_blake3_t tag;
  u8 * tag_buf;
  u8 * cell_susp;
  bool cell_bad, slice_bad, want_tag, subtree;
  xpar_json * js;
} stream_acc;

static void acc_init(stream_acc * a, xpar_vset * s, bool strong,
                     xpar_json * js) {
  xpar_memset(a, 0, sizeof *a);
  a->s   = s;
  a->js  = js;
  a->pos = s->geom.stream_base;
  /*  Under authentication, CRC is only a candidate filter; MAC decides.  */
  a->want_tag = (s->keyed || strong) && (s->have & XPAR_TAGS_TAG) != 0;
  a->subtree = a->want_tag &&
               (s->setd.required_features & XPAR_FEAT_B3_SUBTREE) != 0;
  if (a->subtree)
    a->tag_buf = (u8 *) xpar_calloc((sz) s->geom.slice_size, 1);
  if (s->keyed)
    a->cell_susp = (u8 *) xpar_calloc(s->eg.cells_per_slice, 1);
  if (s->keyed) xpar_blake3_init_keyed(&a->tag, s->key.k_slice);
  else          xpar_blake3_init(&a->tag);
}

static void note_slice(stream_acc * a, u64 slice, const char * why) {
  if (!a->js) return;
  xpar_json_begin(a->js, "slice");
  xpar_json_u64 (a->js, "index", slice);
  xpar_json_str (a->js, "status", "bad");
  xpar_json_str (a->js, "reason", why);
  xpar_json_end (a->js);
}

static void flush_cell(stream_acc * a, u64 slice, u32 col) {
  xpar_vset * s = a->s;
  const xpar_tags * t = &s->tagset.t;
  bool bad = false;
  if (s->ignored_cell &&
      s->ignored_cell[slice * s->eg.cells_per_slice + col]) {
    /*  Superseded bytes are outside this generation's verdict.  */
  } else if (a->cell_bad) {
    bad = true;
  } else if (s->eg.cell_bytes && (s->have & XPAR_TAGS_CELL) &&
             t->cell_crc[slice * t->cells_per_slice + col] != a->cell_crc) {
    bad = true;
  }
  if (bad && s->keyed) a->cell_susp[col] = 1;
  else if (bad) xpar_cell_mark(&s->er, slice, col);
  if (bad) a->slice_bad = true;
  a->cell_crc = 0;
  a->cell_bad = false;
}

static void flush_slice(stream_acc * a, u64 slice) {
  xpar_vset * s = a->s;
  const xpar_tags * t = &s->tagset.t;
  bool whole_bad = false;
  bool ignored = false;
  u32 col;
  if (s->ignored_cell)
    for (col = 0; col < s->eg.cells_per_slice; col++)
      if (s->ignored_cell[slice * s->eg.cells_per_slice + col]) {
        ignored = true;  break;
      }
  bool crc_bad = !ignored && (s->have & XPAR_TAGS_CRC) &&
                 a->slice_crc != t->slice_crc[slice];
  bool tag_bad = false;
  if (!s->keyed && !a->slice_bad && crc_bad) {
    whole_bad = true;
    note_slice(a, slice, "crc");
  }
  if (!ignored && (s->keyed || (!a->slice_bad && !whole_bad)) && a->want_tag) {
    u8 got[XPAR_BLAKE3_OUT_LEN];
    if (a->subtree && s->keyed)
      xpar_slice_tag_keyed(&s->setd, slice, a->tag_buf, s->key.k_slice,
                           got, t->tag_len);
    else if (a->subtree)
      xpar_slice_tag(&s->setd, slice, a->tag_buf, got, t->tag_len);
    else
      xpar_blake3_final(&a->tag, got, t->tag_len);
    if (!xpar_blake3_tag_equal(got, t->slice_tag + slice * t->tag_len,
                               t->tag_len)) {
      tag_bad = true;
      note_slice(a, slice, "tag");
    }
  }
  if (s->keyed) {
    if (tag_bad) {
      bool local = false;
      for (col = 0; col < s->eg.cells_per_slice; col++)
        if (a->cell_susp[col]) {
          xpar_cell_mark(&s->er, slice, col);
          local = true;
        }
      if (!local) whole_bad = true;
      a->slice_bad = true;
    } else {
      /*  A valid MAC overrides CRC/SLCL filter mismatch.  */
      a->slice_bad = false;
      (void) crc_bad;
    }
    xpar_memset(a->cell_susp, 0, s->eg.cells_per_slice);
  }
  if (whole_bad) {
    /*  When all cells pass but the slice fails, erase the whole slice.  */
    xpar_erasures_mark_slice(&s->er, slice);
    a->slice_bad = true;
  }
  if (a->slice_bad) {
    s->bad_slices++;
    if (!whole_bad) note_slice(a, slice, "cell");
  }
  a->slice_crc = 0;
  a->slice_bad = false;
  if (s->keyed) xpar_blake3_init_keyed(&a->tag, s->key.k_slice);
  else          xpar_blake3_init(&a->tag);
}

/*  NULL input erases its cells without manufacturing a checksum.  */
static void acc_feed(stream_acc * a, const u8 * p, u64 n) {
  const xpar_geom * g = &a->s->eg;
  while (n) {
    u64 in    = (a->pos - g->stream_base) % g->slice_size;
    u64 slice = (a->pos - g->stream_base) / g->slice_size;
    u32 col   = g->cell_bytes ? (u32) (in / g->cell_bytes) : 0;
    u64 base  = g->cell_bytes ? (u64) col * g->cell_bytes : 0;
    u64 size  = xpar_cell_size(g, col);
    u64 take  = MIN(n, base + size - in);
    if (slice >= g->slice_count) return;
    if (p) {
      a->cell_crc  = xpar_crc32c(a->cell_crc, p, (sz) take);
      a->slice_crc = xpar_crc32c(a->slice_crc, p, (sz) take);
      if (a->want_tag) {
        if (a->subtree) xpar_memcpy(a->tag_buf + in, p, (sz) take);
        else            xpar_blake3_update(&a->tag, p, (sz) take);
      }
      p += take;
    } else {
      a->cell_bad = a->slice_bad = true;
    }
    a->pos += take;  n -= take;
    if (in + take == base + size) {
      flush_cell(a, slice, col);
      if (base + size >= g->slice_size) flush_slice(a, slice);
    }
  }
}

static void acc_zero(stream_acc * a, u64 n) {
  while (n) {
    u64 take = MIN(n, (u64) sizeof zeros);
    acc_feed(a, zeros, take);
    n -= take;
  }
}

static void acc_ignore(stream_acc * a, u64 n) {
  const xpar_geom * g = &a->s->eg;
  while (n) {
    u64 in = (a->pos - g->stream_base) % g->slice_size;
    u64 slice = (a->pos - g->stream_base) / g->slice_size;
    u32 col = g->cell_bytes ? (u32) (in / g->cell_bytes) : 0;
    u64 base = g->cell_bytes ? (u64) col * g->cell_bytes : 0;
    u64 size = xpar_cell_size(g, col);
    u64 take = MIN(n, base + size - in);
    if (slice >= g->slice_count) return;
    a->s->ignored_cell[slice * g->cells_per_slice + col] = 1;
    a->pos += take;  n -= take;
    if (in + take == base + size) {
      flush_cell(a, slice, col);
      if (base + size >= g->slice_size) flush_slice(a, slice);
    }
  }
}

static void acc_ignore_at(stream_acc * a, u64 off, u64 n) {
  if (off > a->pos) acc_zero(a, off - a->pos);
  if (off == a->pos) acc_ignore(a, n);
}

/*  Feed zero alignment gaps before the next extent.  */
static void acc_at(stream_acc * a, u64 off, const u8 * p, u64 n) {
  if (off > a->pos) acc_zero(a, off - a->pos);
  if (off != a->pos) return;
  acc_feed(a, p, n);
}

static void acc_done(stream_acc * a) {
  const xpar_geom * g = &a->s->geom;
  u64 end = g->stream_base + g->slice_count * g->slice_size;
  if (a->pos < end) acc_zero(a, end - a->pos);
  xpar_free(a->tag_buf);
  xpar_free(a->cell_susp);
  a->tag_buf = NULL;
}

typedef struct {
  bool exists, wrong_length, hash_bad, prefix_bad, link_broken;
  u64  size;
} entry_result;

static void hash_range(xpar_vset * s, xpar_file * f, u64 off, u64 n,
                       xpar_blake3_t * h, u8 * buf, xpar_pool * pool) {
  while (n) {
    sz take = (sz) MIN(n, (u64) VERIFY_BATCH * VERIFY_IOBUF);
    if (xpar_pread(f, buf, take, off) != take) xpar_memset(buf, 0, take);
    xpar_blake3_update_parallel(h, buf, take, pool);
    s->bytes_read += take;
    off += take;  n -= take;
  }
}

/*  Confirm hard-link aliases by (dev, ino), or hash where unavailable.  */
static void check_link(xpar_vset * s, u32 i, const char * path,
                       const xpar_stat_t * st, const xpar_nameidx * nix,
                       entry_result * r, u8 * buf, xpar_pool * pool) {
  i64 tgt = xpar_link_target(&s->mf, nix, i);
  const xpar_entry * te;
  xpar_file * f;
  xpar_blake3_t h;
  u8 got[XPAR_BLAKE3_OUT_LEN];
  if (tgt < 0) { r->link_broken = true;  return; }
  te = &s->mf.entry[tgt];
  if (!st->is_regular) { r->link_broken = true;  return; }
  if (st->size != te->length) {
    r->wrong_length = true;
    return;
  }
  if (xpar_fs_caps(path) & XPAR_FS_LINKID) {
    char * tpath = entry_path(s, (u32) tgt);
    xpar_stat_t ts;
    int ok = xpar_lstat(tpath, &ts);
    xpar_free(tpath);
    if (ok == 0) {
      if (ts.dev != st->dev || ts.ino != st->ino) r->link_broken = true;
      return;
    }
  }
  f = xpar_open(path, XPAR_O_RDONLY | XPAR_O_NOFOLLOW);
  if (!f) { r->exists = false;  return; }
  if (s->auth_only) xpar_blake3_init_keyed(&h, s->key.k_file);
  else              xpar_blake3_init(&h);
  hash_range(s, f, 0, te->length, &h, buf, pool);
  xpar_close(f);
  xpar_blake3_final(&h, got, XPAR_BLAKE3_OUT_LEN);
  if (xpar_memcmp(got, te->content_hash, XPAR_BLAKE3_OUT_LEN) != 0)
    r->hash_bad = true;
}

static void check_entry(xpar_vset * s, u32 i, stream_acc * acc,
                        const xpar_options * o, const xpar_nameidx * nix,
                        u8 * buf, entry_result * r, xpar_progress_t * pg,
                        xpar_pool * pool) {
  const xpar_entry * e = &s->mf.entry[i];
  char * path = entry_path(s, i);
  xpar_stat_t st;
  xpar_blake3_t h, ph;
  xpar_file * f = NULL;
  bool hashing = !o->fast;
  u64 fo = 0;
  u32 k;

  xpar_memset(r, 0, sizeof *r);
  r->exists = xpar_lstat(path, &st) == 0;
  r->size   = r->exists ? st.size : 0;

  if (e->entry_type == XPAR_ENTRY_DIR) {
    if (!r->exists || !st.is_dir) r->link_broken = true;
    xpar_free(path);
    return;
  }
  if (e->entry_type == XPAR_ENTRY_SYMLINK) {
    u32 n = 0;
    char * tgt = r->exists ? xpar_read_symlink(path, &n) : NULL;
    if (!tgt || n != e->extra_len ||
        xpar_memcmp(tgt, e->extra, e->extra_len) != 0) r->hash_bad = true;
    xpar_free(tgt);
    xpar_free(path);
    return;
  }
  if (e->entry_type == XPAR_ENTRY_HARDLINK) {
    if (r->exists) check_link(s, i, path, &st, nix, r, buf, pool);
    xpar_free(path);
    return;
  }

  if (r->exists && !st.is_regular) {
    r->link_broken = true;
    xpar_free(path);
    return;
  }
  if (r->exists && st.size != e->length) r->wrong_length = true;
  if (r->exists)
    f = xpar_open(path, XPAR_O_RDONLY | XPAR_O_NOFOLLOW);
  if (!f) r->exists = false;
  if (r->exists)
    v_resync_entry(s, i, o, f, r->size, path);
  xpar_free(path);
  if (hashing) {
    if (s->auth_only) xpar_blake3_init_keyed(&h, s->key.k_file);
    else              xpar_blake3_init(&h);
    xpar_blake3_init(&ph);
  }

  for (k = 0; k < e->extent_count; k++) {
    u64 len = e->extents[k].length, off = e->extents[k].stream_offset;
    bool canon = !aliased(s, i, k);
    u64 done = 0;
    /*  --fast reads canonical stream extents only.  */
    if (o->fast && !canon) { fo += len;  continue; }
    while (done < len) {
      xpar_read_req req[VERIFY_BATCH];
      u64 logical[VERIFY_BATCH] = { 0 }, queued = done;
      u64 io_start = 0, io_usec = 0;
      sz b, count = 0;
      bool sample = hashing && !s->hash_sampled && pool &&
                    xpar_pool_threads(pool) > 1;
      while (queued < len && count < VERIFY_BATCH) {
        sz take = (sz) MIN(len - queued, (u64) VERIFY_IOBUF);
        u64 physical = fo + queued;
        if (s->resync[i].count)
          take = (sz) MIN((u64) take, s->geom.slice_size -
                          physical % s->geom.slice_size);
        xpar_resync_map_shift(&s->resync[i], physical, &physical);
        logical[count] = queued;
        req[count].file = f && physical <= r->size &&
                          take <= r->size - physical ? f : NULL;
        req[count].buf = buf + count * VERIFY_IOBUF;
        req[count].length = take;  req[count].offset = physical;
        req[count].result = 0;
        queued += take;  count++;
      }
      if (sample) io_start = xpar_usec_now();
      xpar_pread_batch(req, count);
      if (sample) io_usec = xpar_usec_now() - io_start;
      {
        bool packed = true, all_got = true;
        sz packed_len = 0;
        u64 hash_usec = 0, scan_start;
        for (b = 0; b < count; b++) {
          if (req[b].result != req[b].length) all_got = false;
          if (req[b].buf != buf + packed_len) packed = false;
          packed_len += req[b].length;
        }
        if (hashing && all_got && packed) {
          u64 at = fo + logical[0];
          u64 hash_start = sample ? xpar_usec_now() : 0;
          xpar_blake3_update_parallel(&h, buf, packed_len,
                                      s->hash_parallel ? pool : NULL);
          if (at < 16384)
            xpar_blake3_update(&ph, buf,
              (sz) MIN((u64) packed_len, 16384 - at));
          if (sample) hash_usec = xpar_usec_now() - hash_start;
        }
      scan_start = sample ? xpar_usec_now() : 0;
      for (b = 0; b < count; b++) {
        sz take = req[b].length;
        u8 * piece = buf + b * VERIFY_IOBUF;
        bool got = req[b].result == take;
        if (got) {
          s->bytes_read += take;
          xpar_progress_tick(pg, take);
          if (hashing && (!all_got || !packed)) {
            u64 at = fo + logical[b];
            xpar_blake3_update_parallel(&h, piece, take,
                                        s->hash_parallel ? pool : NULL);
            if (at < 16384)
              xpar_blake3_update(&ph, piece,
                (sz) MIN((u64) take, 16384 - at));
          }
        }
        if (canon)
          acc_at(acc, off + logical[b], got ? piece : NULL, take);
      }
      /*  Enable hash workers only when measured hashing exceeds read and
          slice-scan time. CV reduction remains deterministic.  */
      if (sample && all_got && packed && packed_len >= VERIFY_IOBUF) {
        u64 scan_usec = xpar_usec_now() - scan_start;
        s->hash_parallel = hash_usec > io_usec + scan_usec;
        s->hash_sampled = true;
      }
      }
      done = queued;
    }
    fo += len;
  }

  if (f) xpar_close(f);
  if (!hashing) return;
  { u8 got[XPAR_BLAKE3_OUT_LEN];
    xpar_blake3_final(&h, got, XPAR_BLAKE3_OUT_LEN);
    if (xpar_memcmp(got, e->content_hash, XPAR_BLAKE3_OUT_LEN) != 0)
      r->hash_bad = true;
    xpar_blake3_final(&ph, got, 16);
    if (e->length && xpar_memcmp(got, e->prefix_hash, 16) != 0)
      r->prefix_bad = true; }
}

/* True when another entry defines any of this entry's bytes. */
static bool has_alias(const xpar_vset * s, u32 i) {
  const xpar_entry * e = &s->mf.entry[i];
  u32 k;
  for (k = 0; k < e->extent_count; k++) if (aliased(s, i, k)) return true;
  return false;
}

/*  Clean canonical cells with a bad entry hash identify alias-local damage,
    which consumes no recovery.  */
static bool canon_erased(const xpar_vset * s, u32 i) {
  const xpar_entry * e = &s->mf.entry[i];
  const xpar_geom * g = &s->eg;
  u32 k;
  for (k = 0; k < e->extent_count; k++) {
    u64 p, end;
    if (aliased(s, i, k)) continue;
    p   = e->extents[k].stream_offset;
    end = p + e->extents[k].length;
    while (p < end) {
      u64 slice = (p - g->stream_base) / g->slice_size;
      u64 in    = (p - g->stream_base) % g->slice_size;
      u32 col   = g->cell_bytes ? (u32) (in / g->cell_bytes) : 0;
      if (slice >= g->slice_count) break;
      if (xpar_cell_bad(&s->er, slice, col)) return true;
      p += xpar_cell_size(g, col) -
           (in - (g->cell_bytes ? (u64) col * g->cell_bytes : 0));
    }
  }
  return false;
}

void xpar_vset_mark_superseded(xpar_vset * old, const xpar_vset * head) {
  xpar_nameidx ix;
  u32 i;
  old->superseded = (u8 *) xpar_calloc(old->mf.count ? old->mf.count : 1, 1);
  xpar_nameidx_build(&head->mf, &ix);
  for (i = 0; i < old->mf.count; i++) {
    const xpar_entry * e = &old->mf.entry[i];
    i64 h = xpar_nameidx_find(&head->mf, &ix, e->name, e->name_len);
    if (h < 0 || xpar_memcmp(e->file_id, head->mf.entry[h].file_id,
                             XPAR_SET_ID_LEN))
      old->superseded[i] = 1;
  }
  xpar_nameidx_free(&ix);
}

int xpar_vset_check(xpar_vset * s, const xpar_options * o,
                    xpar_json * js) {
  stream_acc acc;
  xpar_nameidx nix;
  xpar_progress_t pg;
  sz buf_size = VERIFY_BATCH * VERIFY_IOBUF;
  u8 * buf;
  u8 * shape = (u8 *) xpar_calloc(s->mf.count ? s->mf.count : 1, 1);
  xpar_pool * pool = xpar_pool_create(o->jobs);
  u64 total = 0;
  u32 i;
  int rc;

  if (s->setd.layout != XPAR_LAYOUT_SIDECAR) {
    if (s->geom.slice_size > (u64) (sz) -1)
      FATAL_FORMAT("The slice size exceeds this host's address space.");
    buf_size = MAX(buf_size, (sz) s->geom.slice_size);
  }
  buf = (u8 *) xpar_alloc_raw(buf_size);

  if (s->superseded)
    s->ignored_cell = (u8 *) xpar_calloc(
      (sz) MAX(s->eg.slice_count * s->eg.cells_per_slice, 1), 1);

  /*  Authenticated deduplicated aliases require full-tree verification.  */
  if (o->fast && s->keyed && s->setd.dedup_level != XPAR_DEDUP_NONE)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "--fast cannot authenticate a deduplicated set; drop "
               "--fast.");

  xpar_nameidx_build(&s->mf, &nix);
  if (o->fast) total = s->geom.stream_length;
  else for (i = 0; i < s->mf.count; i++) total += s->mf.entry[i].length;
  xpar_progress_init(&pg, progress_on(o), total, "Verifying");
  acc_init(&acc, s, o->strong, js);

  if (s->setd.layout != XPAR_LAYOUT_SIDECAR) {
    u64 slice;
    for (slice = 0; slice < s->geom.slice_count; slice++) {
      u64 z = s->geom.slice_size;
      bool got = xpar_vset_read(s, xpar_slice_begin(&s->geom, slice),
                                buf, z);
      if (got && s->setd.layout == XPAR_LAYOUT_ARMOURED) {
        bool suspect = false;
        const xpar_tags * t = &s->tagset.t;
        if ((s->have & XPAR_TAGS_CRC) &&
            xpar_crc32c(0, buf, (sz) z) != t->slice_crc[slice])
          suspect = true;
        if (s->keyed && (s->have & XPAR_TAGS_TAG)) {
          u8 tag[16];
          xpar_slice_tag_keyed(&s->setd, slice, buf, s->key.k_slice,
                               tag, t->tag_len);
          if (!xpar_blake3_tag_equal(tag,
                t->slice_tag + slice * t->tag_len, t->tag_len)) suspect = true;
        }
        if (suspect && correct_armoured_slice(s, slice))
          got = xpar_vset_read(s, xpar_slice_begin(&s->geom, slice), buf, z);
      }
      acc_feed(&acc, got ? buf : NULL, z);
      s->bytes_read += z;
      xpar_progress_tick(&pg, xpar_slice_bytes(&s->geom, slice));
    }
    acc_done(&acc);
    if (!o->fast) for (i = 0; i < s->mf.count; i++) {
      const xpar_entry * e = &s->mf.entry[i];
      xpar_blake3_t h, ph;
      u8 got_hash[32];
      u64 file_off = 0;
      u32 k;
      bool damaged = false;
      bool local = true;
      if (e->entry_type == XPAR_ENTRY_HARDLINK) continue;
      /*  A selected generation cannot hash inherited extents it does not
          expose; --chain verifies their owning ancestors.  */
      if (e->entry_type == XPAR_ENTRY_REGULAR)
        for (k = 0; k < e->extent_count; k++) {
          u64 off = e->extents[k].stream_offset;
          u64 end = off + e->extents[k].length;
          if (off < s->geom.stream_base || end < off ||
              end > s->geom.stream_base + s->geom.stream_length) {
            local = false;
            break;
          }
        }
      if (!local) continue;
      if (s->auth_only) xpar_blake3_init_keyed(&h, s->key.k_file);
      else              xpar_blake3_init(&h);
      xpar_blake3_init(&ph);
      if (e->entry_type == XPAR_ENTRY_SYMLINK) {
        xpar_blake3_update(&h, e->extra, e->extra_len);
        if (e->extra_len)
          xpar_blake3_update(&ph, e->extra, MIN((u32) 16384, e->extra_len));
      } else for (k = 0; k < e->extent_count; k++) {
        u64 done = 0, n = e->extents[k].length;
        while (done < n) {
          u64 take = MIN(n - done, (u64) VERIFY_BATCH * VERIFY_IOBUF);
          bool read_ok = xpar_vset_read(s, e->extents[k].stream_offset + done,
                                        buf, take);
          xpar_blake3_update_parallel(&h, buf, (sz) take, pool);
          if (file_off + done < 16384)
            xpar_blake3_update(&ph, buf,
              (sz) MIN(take, 16384 - (file_off + done)));
          if (!read_ok) damaged = true;
          s->bytes_read += take;
          done += take;
        }
        file_off += n;
      }
      xpar_blake3_final(&h, got_hash, 32);
      if (!xpar_blake3_tag_equal(got_hash, e->content_hash, 32)) damaged = true;
      xpar_blake3_final(&ph, got_hash, 16);
      if (e->length && !xpar_blake3_tag_equal(got_hash, e->prefix_hash, 16))
        damaged = true;
      if (damaged) s->bad_entries++;
      if (js) {
        xpar_json_begin(js, "file_result");
        xpar_json_u64(js, "index", i);
        xpar_json_strn(js, "name", e->name, e->name_len);
        xpar_json_str(js, "status", damaged ? "damaged" : "ok");
        xpar_json_bool(js, "content_hash_ok", !damaged);
        xpar_json_end(js);
      }
    }
    xpar_progress_end(&pg);
    xpar_nameidx_free(&nix);
    goto scanned;
  }

  for (i = 0; i < s->mf.count; i++) {
    const xpar_entry * e = &s->mf.entry[i];
    entry_result r;
    bool damaged;
    if (s->superseded && s->superseded[i]) {
      u32 k;
      for (k = 0; k < e->extent_count; k++)
        if (!aliased(s, i, k))
          acc_ignore_at(&acc, e->extents[k].stream_offset,
                        e->extents[k].length);
      s->superseded_entries++;
      xpar_progress_tick(&pg, e->length);
      if (js) {
        xpar_json_begin(js, "file_result");
        xpar_json_u64(js, "index", i);
        xpar_json_strn(js, "name", e->name, e->name_len);
        xpar_json_str(js, "status", "superseded");
        xpar_json_end(js);
      }
      if (!o->quiet)
        xpar_fprintf(xpar_stderr, "xpar: %.*s: superseded\n",
                     (int) e->name_len, e->name);
      continue;
    }
    check_entry(s, i, &acc, o, &nix, buf, &r, &pg, pool);
    damaged = !r.exists || r.wrong_length || r.hash_bad || r.prefix_bad ||
              r.link_broken;
    if (!r.exists) s->missing_entries++;
    if (damaged) s->bad_entries++;
    /*  Defer alias-local classification until the erasure table is final.  */
    if (damaged && r.exists && !r.wrong_length && !r.link_broken)
      shape[i] = 1;
    if (js) {
      xpar_json_begin(js, "file_result");
      xpar_json_u64 (js, "index", i);
      xpar_json_strn(js, "name", e->name, e->name_len);
      xpar_json_str (js, "status", damaged ? "damaged" : "ok");
      xpar_json_bool(js, "present", r.exists);
      if (!o->fast) xpar_json_bool(js, "content_hash_ok", !r.hash_bad);
      xpar_json_end (js);
    }
    if (damaged && !o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: %.*s: %s\n", (int) e->name_len,
                   e->name,
                   !r.exists      ? "missing" :
                   r.wrong_length ? "wrong length" :
                   r.link_broken  ? "structure differs" :
                   r.prefix_bad   ? "replaced (prefix hash differs)"
                                  : "content differs");
  }
  acc_done(&acc);
  xpar_progress_end(&pg);
  xpar_nameidx_free(&nix);

scanned:
  /* Clean canonical cells make hash failures alias-local or unlocalisable. */
  for (i = 0; i < s->mf.count; i++) {
    if (!shape[i] || canon_erased(s, i)) continue;
    if (has_alias(s, i)) s->alias_bad++;  else s->opaque_bad++;
  }
  xpar_free(shape);
  xpar_free(buf);
  xpar_pool_destroy(pool);

  s->depth = xpar_erasures_max_depth(&s->er);
  /* Repair uses one decode plan per column pattern. */
  s->column_groups = 0;
  if (s->er.bad_count) {
    xpar_col_groups cg;
    xpar_col_groups_build(&s->er, &cg);
    s->column_groups = cg.group_count;
    xpar_col_groups_free(&cg);
  }
  rc = XPAR_EXIT_OK;
  if (s->er.bad_count || s->bad_entries) rc = XPAR_EXIT_REPAIRABLE;
  /*  A damaged named volume makes the set repairable.  */
  if (s->subst_damaged && rc == XPAR_EXIT_OK) rc = XPAR_EXIT_REPAIRABLE;
  if (s->depth > s->recovery || s->opaque_bad) rc = XPAR_EXIT_UNREPAIRABLE;
  return rc;
}

void xpar_vset_report(const xpar_vset * s, const xpar_options * o,
                      int rc) {
  const bool color = color_on(o);
  u64 aliased_bytes = 0;
  u32 i, k;

  /*  --quiet still reports reduced protection despite a clean-data exit.  */
  if (s->degraded)
    xpar_fprintf(xpar_stderr,
                 "xpar: SETD records a %" PRIu32 "-byte cell but no complete cell "
                 "table survives; erasures fall back to slice granularity "
                 "(`xpar scrub --rebuild-cells` restores it)\n",
                 s->geom.cell_bytes);
  if (s->recovery_gone)
    xpar_fprintf(xpar_stderr,
                 "xpar: warning: %" PRIu64 " of %" PRIu64 " recovery slices named by the "
                 "layout are not on disk; available protection is reduced\n",
                 s->recovery_gone,
                 (s->recovery + s->recovery_gone));
  /*  Always report substituted volumes.  */
  for (i = 0; i < s->subst_count; i++) {
    if (s->subst[i].present)
      xpar_fprintf(xpar_stderr,
                   "xpar: data volume '%s' is damaged; intact copy found "
                   "as '%s'\n",
                   s->subst[i].want, s->subst[i].got);
    else
      xpar_fprintf(xpar_stderr,
                   "xpar: data volume '%s' is missing; using '%s'\n",
                   s->subst[i].want, s->subst[i].got);
  }

  if (o->quiet) return;
  for (i = 0; i < s->mf.count; i++)
    for (k = 0; k < s->mf.entry[i].extent_count; k++)
      if (aliased(s, i, k)) aliased_bytes += s->mf.entry[i].extents[k].length;

  xpar_fprintf(xpar_stderr,
               "xpar: %" PRIu64 " slice%s of %" PRIu64 " bytes, %" PRIu64 " recovery slice%s, "
               "erasure unit ",
               s->geom.slice_count,
               PLURAL(s->geom.slice_count),
               s->geom.slice_size,
               s->recovery, PLURAL(s->recovery));
  if (s->eg.cell_bytes)
    xpar_fprintf(xpar_stderr, "cell of %" PRIu32 " bytes (%" PRIu32
                 " per slice)\n",
                 s->eg.cell_bytes,
                 s->eg.cells_per_slice);
  else
    xpar_fputs("slice\n", xpar_stderr);
  if (s->armg_corrected || s->armg_failed)
    xpar_fprintf(xpar_stderr,
                 "xpar: armoured metadata: %" PRIu64 " region%s corrected, %" PRIu64 " "
                 "past the inner code\n",
                 s->armg_corrected,
                 PLURAL(s->armg_corrected),
                 s->armg_failed);
  if (o->fast)
    xpar_fprintf(xpar_stderr,
                 "xpar: coverage: stream only (%" PRIu32 " %s, %" PRIu64 " bytes of "
                 "aliased occurrences not checked; run without --fast)\n",
                 s->mf.count,
                 s->mf.count == 1 ? "entry" : "entries",
                 aliased_bytes);
  else
    xpar_fprintf(xpar_stderr, "xpar: coverage: tree (%" PRIu32 " %s)\n",
                 s->mf.count,
                 s->mf.count == 1 ? "entry" : "entries");
  if (s->superseded_entries)
    xpar_fprintf(xpar_stderr, "xpar: superseded: %" PRIu64 " %s excluded from "
                 "this generation's verdict\n",
                 s->superseded_entries,
                 s->superseded_entries == 1 ? "entry" : "entries");

  if (rc == XPAR_EXIT_OK) {
    xpar_fprintf(xpar_stderr, "xpar: status: %sclean%s\n",
                 color ? "\033[32m" : "", color ? "\033[0m" : "");
    return;
  }
  xpar_fprintf(xpar_stderr,
               "xpar: damaged: %" PRIu64 " %s (%" PRIu64 " missing), %" PRIu64 " slice%s, "
               "%" PRIu64 " cell%s; deepest column %" PRIu64 "\n",
               s->bad_entries,
               s->bad_entries == 1 ? "entry" : "entries",
               s->missing_entries,
               s->bad_slices, PLURAL(s->bad_slices),
               s->er.bad_count,
               PLURAL(s->er.bad_count),
               s->depth);
  if (s->column_groups > 1)
    xpar_fprintf(xpar_stderr,
                 "xpar: damage has %" PRIu64 " column patterns; repair needs "
                 "that many decode plans\n", s->column_groups);
  if (s->alias_bad)
    xpar_fprintf(xpar_stderr,
                 "xpar: %" PRIu64 " %s damaged only at aliased occurrences; "
                 "repair copies those and spends no recovery\n",
                 s->alias_bad,
                 s->alias_bad == 1 ? "entry is" : "entries are");
  if (s->opaque_bad)
    xpar_fprintf(xpar_stderr,
                 "xpar: %" PRIu64 " %s checksum-invisible damage; "
                 "recovery cannot localise it\n",
                 s->opaque_bad,
                 s->opaque_bad == 1 ? "entry has" : "entries have");
  if (rc == XPAR_EXIT_UNREPAIRABLE && s->depth > s->recovery)
    xpar_fprintf(xpar_stderr,
                 "xpar: status: %sunrepairable%s, short by %" PRIu64 " recovery "
                 "slice%s in the deepest column\n",
                 color ? "\033[31m" : "", color ? "\033[0m" : "",
                 (s->depth - s->recovery),
                 PLURAL(s->depth - s->recovery));
  else if (rc == XPAR_EXIT_UNREPAIRABLE)
    xpar_fprintf(xpar_stderr,
                 "xpar: status: %sunrepairable%s, checksum-invisible damage\n",
                 color ? "\033[31m" : "", color ? "\033[0m" : "");
  else
    xpar_fprintf(xpar_stderr, "xpar: status: %srepairable%s\n",
                 color ? "\033[33m" : "", color ? "\033[0m" : "");
}

void xpar_vset_json_set(const xpar_vset * s, xpar_json * js) {
  xpar_json_begin(js, "set");
  xpar_json_u64(js, "schema", XPAR_JSON_SCHEMA);
  xpar_json_hex(js, "set_id", s->set_id, XPAR_SET_ID_LEN);
  xpar_json_u64(js, "slice_size", s->geom.slice_size);
  xpar_json_u64(js, "slices", s->geom.slice_count);
  xpar_json_u64(js, "recovery", s->recovery);
  xpar_json_u64(js, "field", s->setd.field_log2);
  xpar_json_str(js, "codec",
                s->setd.codec == XPAR_CODEC_FFT_LOW ? "fft-low" :
                s->setd.codec == XPAR_CODEC_FFT     ? "fft" : "matrix");
  xpar_json_str(js, "layout",
                s->setd.layout == XPAR_LAYOUT_SPLIT ? "split" :
                s->setd.layout == XPAR_LAYOUT_ARMOURED ? "armoured" :
                                                        "sidecar");
  xpar_json_u64(js, "files", s->setd.file_count);
  xpar_json_u64(js, "generation", s->setd.generation);
  xpar_json_end(js);
}

void xpar_vset_json_summary(const xpar_vset * s, xpar_json * js,
                            int rc) {
  xpar_json_begin(js, "summary");
  xpar_json_str(js, "status",
                rc == XPAR_EXIT_OK          ? "clean" :
                rc == XPAR_EXIT_REPAIRABLE  ? "repairable"
                                            : "unrepairable");
  xpar_json_u64(js, "exit", (u64) rc);
  xpar_json_u64(js, "slices_checked", s->geom.slice_count);
  xpar_json_u64(js, "slices_bad", s->bad_slices);
  xpar_json_u64(js, "cells_bad", s->er.bad_count);
  xpar_json_u64(js, "column_depth", s->depth);
  xpar_json_u64(js, "column_groups", s->column_groups);
  xpar_json_u64(js, "recovery_available", s->recovery);
  xpar_json_u64(js, "recovery_needed", s->depth);
  xpar_json_u64(js, "entries_damaged", s->bad_entries);
  xpar_json_u64(js, "entries_alias_only", s->alias_bad);
  xpar_json_u64(js, "entries_opaque", s->opaque_bad);
  xpar_json_u64(js, "entries_superseded", s->superseded_entries);
  xpar_json_u64(js, "volumes_substituted", s->subst_count);
  xpar_json_u64(js, "volumes_to_rewrite", s->subst_damaged);
  xpar_json_u64(js, "syndromes", syndromes);
  xpar_json_u64(js, "bytes_read", s->bytes_read);
  xpar_json_u64(js, "bytes_written", 0);
  xpar_json_end(js);
}

static int verify_loaded(xpar_vset * s, const xpar_options * o,
                         xpar_json * shared, bool summary) {
  xpar_json local;
  xpar_json * js = shared ? shared : &local;
  int rc;
  if (!shared) xpar_json_init(js, xpar_stdout, o->json);
  if (o->json) xpar_vset_json_set(s, js);
  rc = xpar_vset_check(s, o, o->json ? js : NULL);
  xpar_vset_report(s, o, rc);
  if (o->json && summary) xpar_vset_json_summary(s, js, rc);
  else if (o->json) {
    xpar_json_begin(js, "generation_result");
    xpar_json_u64(js, "generation", xpar_vset_setd(s)->generation);
    xpar_json_i64(js, "exit", rc);
    xpar_json_str(js, "status",
                  rc == XPAR_EXIT_OK ? "clean" :
                  rc == XPAR_EXIT_REPAIRABLE ? "repairable" :
                                               "unrepairable");
    xpar_json_end(js);
  }
  return rc;
}

int xpar_op_verify(const xpar_options * o) {
  xpar_vset * s;
  int rc;

  FATAL_UNLESS("verify cannot read a pipe; use --spool.", !o->from_stdin);
  FATAL_UNLESS("Options --fast and --strong are mutually exclusive.",
               !(o->fast && o->strong));

  if (o->chain) {
    xpar_chain c;
    xpar_options metadata = *o;
    u32 selected;
    metadata.chain_metadata_only = true;
    xpar_gchain_load(&metadata, &c);
    selected = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
    if (c.gen_count > 1) {
      xpar_options top = *o;
      xpar_genref top_ref;
      char top_id[XPAR_SET_ID_LEN * 2 + 1];
      xpar_vset * head;
      u8 * member = (u8 *) xpar_calloc(c.gen_count, 1);
      u32 at = selected, walked = 0;
      xpar_gchain_genref(&c, selected, &top_ref, top_id);
      top.chain = false;
      top.gens = &top_ref;
      top.gen_count = 1;
      head = xpar_vset_open(&top);
      while (at != XPAR_GEN_NONE && walked++ < c.gen_count) {
        member[at] = 1;
        at = c.gen[at].parent;
      }
      FATAL_UNLESS("The selected generation's ancestry is cyclic.",
                   at == XPAR_GEN_NONE);
      {
        int worst = XPAR_EXIT_OK;
        xpar_json chain_js;
        u32 g;
        xpar_json_init(&chain_js, xpar_stdout, o->json);
        for (g = 0; g < c.gen_count; g++) {
          xpar_options one = *o;
          xpar_genref ref;
          char id[XPAR_SET_ID_LEN * 2 + 1];
          xpar_vset * current;
          if (!member[g]) continue;
          xpar_gchain_genref(&c, g, &ref, id);
          one.chain = false;
          one.gens = &ref;
          one.gen_count = 1;
          current = g == selected ? head : xpar_vset_open(&one);
          if (current != head) xpar_vset_mark_superseded(current, head);
          rc = verify_loaded(current, &one, &chain_js, false);
          if (rc > worst) worst = rc;
          if (current != head) xpar_vset_close(current);
        }
        xpar_vset_close(head);
        if (o->json)
          xpar_json_summary(&chain_js,
                            worst == XPAR_EXIT_OK ? "clean" :
                            worst == XPAR_EXIT_REPAIRABLE ? "repairable" :
                                                           "unrepairable",
                            worst);
        xpar_free(member);
        xpar_gchain_free(&c);
        return worst;
      }
    }
    xpar_gchain_free(&c);
  }
  s = xpar_vset_open(o);
  rc = verify_loaded(s, o, NULL, true);
  xpar_vset_close(s);
  return rc;
}
