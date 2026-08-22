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
#include "port-cpu.h"
#include "port-fs.h"
#include "slice.h"

static xpar_file * gen_hout(const xpar_options * o) {
  return o->json ? xpar_stderr : xpar_stdout;
}

static void gen_json_result(const xpar_options * o, const char * verb,
                            const u8 * set_id, u32 generation,
                            const char * status, int rc) {
  xpar_json js;
  if (!o->json) return;
  xpar_json_init(&js, xpar_stdout, true);
  if (set_id) {
    xpar_json_begin(&js, "set");
    xpar_json_u64(&js, "schema", XPAR_JSON_SCHEMA);
    xpar_json_hex(&js, "set_id", set_id, XPAR_SET_ID_LEN);
    xpar_json_u64(&js, "generation", generation);
    xpar_json_end(&js);
  }
  xpar_json_begin(&js, "operation");
  xpar_json_str(&js, "verb", verb);
  xpar_json_u64(&js, "generation", generation);
  xpar_json_end(&js);
  xpar_json_summary(&js, status, rc);
}

static u8 * gen_read_whole(const char * path, sz * out_len, bool fatal) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  i64 n;  u8 * p;
  *out_len = 0;
  if (!f) {
    if (fatal) FATAL_IO("Cannot open '%s': %s.", path,
                        xpar_strerror(xpar_errno()));
    return NULL;
  }
  n = xpar_size(f);
  if (n < 0 || (u64) n >= (u64) (sz) -1) {
    xpar_close(f);
    if (fatal) FATAL_IO("Cannot size '%s'.", path);
    return NULL;
  }
  p = (u8 *) xpar_alloc_raw((sz) n + 1);
  if (n) xpar_xread(f, p, (sz) n);
  xpar_xclose(f);
  *out_len = (sz) n;
  return p;
}

static bool gen_exists(const char * path) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  if (!f) return false;
  xpar_close(f);
  return true;
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
    xpar_asprintf(&tmp, "%s.xpar-tmp-%03u", path, i);
    f = xpar_open(tmp, access | XPAR_O_CREAT | XPAR_O_EXCL);
    if (f) break;
    xpar_free(tmp);  tmp = NULL;
  }
  if (!f) FATAL_IO("Cannot create a temporary file beside '%s': %s.", path,
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
    FATAL_IO("Cannot flush temporary volume '%s'.", tmp);
  xpar_xclose(f);
  return tmp;
}

static void gen_publish_whole(char * tmp, const char * path, bool replace) {
  if (!replace && gen_exists(path)) {
    xpar_remove(tmp);
    xpar_free(tmp);
    FATAL("'%s' exists; -f overwrites it.", path);
  }
  if (xpar_rename(tmp, path) != 0) {
    int e = xpar_errno();
    xpar_remove(tmp);
    xpar_free(tmp);
    FATAL_IO("Cannot publish '%s': %s.", path, xpar_strerror(e));
  }
  if (xpar_fsync_dir(path) != 0) {
    xpar_free(tmp);
    FATAL_IO("Cannot make the published volume '%s' durable: %s.", path,
             xpar_strerror(xpar_errno()));
  }
  xpar_free(tmp);
}

static void gen_write_whole(const char * path, const void * p, sz n,
                            bool replace) {
  char * tmp;
  if (!replace && gen_exists(path))
    FATAL("'%s' exists; -f overwrites it.", path);
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

static bool gen_chain_sibling(const char * name, const char * stem) {
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  if (!xpar_path_ends_with(name, XPAR_EXT) || n - XPAR_EXT_LEN < p ||
      xpar_strncmp(name, stem, p)) return false;
  n -= XPAR_EXT_LEN;
  if (p == n) return true;
  if (name[p++] != '.' || p == n) return false;
  if (name[p] == 'g') {
    i = p + 1;
    if (!xpar_scan_digits(name, &i, n)) return false;
    if (i == n) return true;
    if (name[i++] != '.' || i == n || name[i] != 'v') return false;
    p = i;
  }
  if (name[p] != 'v') return false;
  i = p + 1;
  if (!xpar_scan_digits(name, &i, n) || i == n || name[i++] != '+')
    return false;
  return xpar_scan_digits(name, &i, n) && i == n;
}

static bool gen_chain_index_sibling(const char * name, const char * stem) {
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  if (!xpar_path_ends_with(name, XPAR_EXT) || n - XPAR_EXT_LEN < p ||
      xpar_strncmp(name, stem, p)) return false;
  n -= XPAR_EXT_LEN;
  if (p == n) return true;
  if (name[p++] != '.' || p == n || name[p] != 'g') return false;
  i = p + 1;
  return xpar_scan_digits(name, &i, n) && i == n;
}

static char * gen_unused_base(const char * base, const char * label) {
  u32 i;
  for (i = 0; i < 1000; i++) {
    char * candidate, * index;
    xpar_asprintf(&candidate, "%s.%s-%03u", base, label, i);
    xpar_asprintf(&index, "%s" XPAR_EXT, candidate);
    if (!gen_exists(index)) { xpar_free(index);  return candidate; }
    xpar_free(index);  xpar_free(candidate);
  }
  FATAL_IO("Cannot choose a staging name beside '%s'.", base);
  return NULL;
}

static char * gen_unused_path(const char * path, const char * label) {
  u32 i;
  for (i = 0; i < 1000; i++) {
    char * candidate;
    xpar_stat_t st;
    xpar_asprintf(&candidate, "%s.%s-%03u", path, label, i);
    if (xpar_lstat(candidate, &st) != 0) return candidate;
    xpar_free(candidate);
  }
  return NULL;
}

/*  The cache is regenerable, so failure to publish it does not invalidate
    an already durable set. It is nevertheless staged before encoding so
    the dedup index does not remain resident alongside the codec plan.  */
static bool gen_publish_cache(char * stage, const char * final) {
  if (xpar_rename(stage, final) != 0) return false;
  return xpar_fsync_dir(final) == 0;
}

static char * gen_name_index(const char * base, u32 g) {
  char * s;
  if (!g) xpar_asprintf(&s, "%s" XPAR_EXT, base);
  else    xpar_asprintf(&s, "%s.g%03u" XPAR_EXT, base, g);
  return s;
}

/*  One width per placeholder, taken from the widest value that will
    appear in it, so every recovery name of the generation lines up. Two
    digits unless the generation needs more.  */
static void gen_recovery_widths(u64 max_first, u64 max_count, int * wf,
                                int * wc) {
  *wf = xpar_digits10(max_first);  if (*wf < 2) *wf = 2;
  *wc = xpar_digits10(max_count);  if (*wc < 2) *wc = 2;
}

static char * gen_name_recovery(const char * base, u32 g, u64 first,
                                u64 count, int wf, int wc) {
  char * s;
  if (!g) xpar_asprintf(&s, "%s.v%0*llu+%0*llu" XPAR_EXT, base, wf,
                        (unsigned long long) first, wc,
                        (unsigned long long) count);
  else    xpar_asprintf(&s, "%s.g%03u.v%0*llu+%0*llu" XPAR_EXT, base, g,
                        wf, (unsigned long long) first, wc,
                        (unsigned long long) count);
  return s;
}

static char * gen_name_data(const char * base, u32 g, u32 index, int width) {
  char * s;
  if (!g) xpar_asprintf(&s, "%s.d%0*u", base, width, index);
  else    xpar_asprintf(&s, "%s.g%03u.d%0*u", base, g, width, index);
  return s;
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
  if (p[8] != XPAR_FORMAT_MAJOR ||
      (p[10] != 8 && p[10] != 16) || p[11])
    return false;
  for (u32 i = 72; i < 88; i++) if (p[i]) return false;
  arm_prologue_decode(p, out);
  arm_params_of(out, &ap);
  if (xpar_armour_check(&ap)) return false;
  if (out->armoured_length != xpar_armg_length(
                                  ap.symbol_bits, ap.n, ap.k,
                                  ap.depth, out->plain_length))
    return false;
  if (out->armoured_length > (u64) len - ARM_HDR_LEN ||
      out->armoured_length != (u64) len - ARM_HDR_LEN)
    return false;
  if (out->stream_offset > out->plain_length ||
      out->stream_length > out->plain_length - out->stream_offset)
    return false;
  return true;
}

bool xpar_garm_prologue(const u8 * file, sz len, xpar_arm_prologue * out,
                        int * which) {
  u8 corrected[3][ARM_PLAIN_LEN], vote[ARM_PLAIN_LEN];
  int j;
  if (len < ARM_HDR_LEN) return false;
  for (j = 0; j < 3; j++)
    if (arm_checksum_ok(file + (sz) j * ARM_COPY_LEN)) {
      if (arm_prologue_valid(file + (sz) j * ARM_COPY_LEN, len, out)) {
        if (which) *which = j;
        return true;
      }
    }

  /*  Reinsert the implicit zero data symbols before RS(255,223) decoding;
      attempt this only after checksum failure.  */
  {
    xpar_armour_params ap;
    xpar_armour * a;
    xpar_gf_init();
    xpar_armour_defaults(&ap, 8);
    ap.n = 255; ap.k = 223; ap.depth = 1;
    a = xpar_armour_new(&ap);
    if (!a) return false;
    for (j = 0; j < 3; j++) {
      u8 frame[255];
      xpar_memset(frame, 0, sizeof frame);
      xpar_memcpy(frame, file + (sz) j * ARM_COPY_LEN, ARM_PLAIN_LEN);
      xpar_memcpy(frame + ap.k,
                  file + (sz) j * ARM_COPY_LEN + ARM_PLAIN_LEN, 32);
      if (xpar_armour_decode_frame(a, frame, NULL) != XPAR_ARMOUR_FAILED)
        xpar_memcpy(corrected[j], frame, ARM_PLAIN_LEN);
      else
        xpar_memcpy(corrected[j], file + (sz) j * ARM_COPY_LEN,
                    ARM_PLAIN_LEN);
      if (arm_checksum_ok(corrected[j])) {
        if (arm_prologue_valid(corrected[j], len, out)) {
          if (which) *which = j;
          xpar_armour_free(a);
          return true;
        }
      }
    }
    xpar_armour_free(a);
  }

  for (j = 0; j < ARM_PLAIN_LEN; j++) {
    u8 a = corrected[0][j], b = corrected[1][j], c = corrected[2][j];
    vote[j] = a == b ? a : (a == c ? a : (b == c ? b : a));
  }
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

static bool chain_arm_check(void * key, const u8 * plain, u64 len) {
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
  plain = (u8 *) xpar_alloc_raw((sz) plain_len + 1);
  xpar_armour_extract(a, plain, plain_len, region);
  if (!xpar_verify_packets_ok(plain, plain_len, key)) {
    u64 encoded = xpar_armour_size(a, plain_len);
    u8 * copy = (u8 *) xpar_alloc_raw((sz) encoded);
    xpar_armour_status st;
    xpar_memcpy(copy, region, (sz) encoded);
    st = xpar_armour_decode(a, copy, encoded, plain, plain_len,
                            chain_arm_check, (void *) key, NULL);
    xpar_free(copy);
    if (st == XPAR_ARMOUR_FAILED) {
      xpar_free(plain);  xpar_armour_free(a);
      return NULL;
    }
  }
  xpar_armour_free(a);
  *out_len = (sz) plain_len;
  return plain;
}

/*  Loading a chain.  */

static void chain_blob(xpar_chain * c, u8 * p) {
  c->blob = (u8 **) xpar_realloc(c->blob, (sz) (c->blob_count + 1) *
                                          sizeof(u8 *));
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
                   "This set is authenticated; supply --auth-key=FILE.");
      continue;
    }
    if (c->key_loaded && !(hdr.flags & XPAR_PF_KEYED))
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "A keyed operation encountered an unkeyed packet.");
    if (!xpar_pkt_is(&hdr, XPAR_T_VOLH) && v->has_volh &&
        xpar_memcmp(hdr.set_id, v->set_id, XPAR_SET_ID_LEN))
      FATAL_FORMAT("A volume contains a packet owned by another generation.");
    if (xpar_pkt_is(&hdr, XPAR_T_ARMG)) continue;
    if (xpar_pkt_is(&hdr, XPAR_T_VOLH)) {
      xpar_volh vh;
      if (xpar_volh_read(body, (sz) blen, &vh) == XPAR_OK) {
        v->has_volh     = true;
        v->volume_index = vh.volume_index;
        v->volume_kind  = vh.volume_kind;
        xpar_memcpy(v->set_id, hdr.set_id, XPAR_SET_ID_LEN);
      }
    }
    if (xpar_pkt_is(&hdr, XPAR_T_RCVS)) {
      u64 e = blen >= 8 ? xpar_rd64(body) : 0;
      if (!v->recovery_count || e < v->recovery_first) v->recovery_first = e;
      v->recovery_count++;
    }
    if (xpar_pkt_is(&hdr, XPAR_T_LAYT)) {
      if (!v->layt_body) { v->layt_body = body;  v->layt_len = (sz) blen; }
      xpar_critset_add(&c->crit, &hdr, body);
      continue;
    }
    if (xpar_pkt_is(&hdr, XPAR_T_VOLH)) continue;
    xpar_critset_add(&c->crit, &hdr, body);
  }
  if (!nested) {
    u64 pos = 0, blen = 0;
    while (xpar_verify_next_armg(buf, len,
                                 c->key_loaded ? &c->key : NULL,
                                 &pos, &body, &blen)) {
      xpar_armg ag;
      xpar_armour_params ap;
      u8 * plain;
      sz plen;
      if (xpar_armg_read(body, (sz) blen, &ag) != XPAR_OK) continue;
      ap.symbol_bits = ag.symbol_bits;  ap.poly = ag.poly;
      ap.n = ag.n;  ap.k = ag.k;  ap.fcr = ag.fcr;  ap.prim = ag.prim;
      ap.depth = ag.depth;
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

  for (i = 0; i < c->vol_count; i++)
    if (!xpar_strcmp(c->vol[i].path, path)) { xpar_free(path);  return; }

  data = gen_read_whole(path, &len, false);
  if (!data) { xpar_free(path);  return; }

  c->vol = (xpar_chain_vol *) xpar_realloc(c->vol,
             (sz) (c->vol_count + 1) * sizeof(xpar_chain_vol));
  v = &c->vol[c->vol_count++];
  xpar_memset(v, 0, sizeof *v);
  v->path = path;  v->data = data;  v->len = len;
  v->gen = XPAR_GEN_NONE;  v->volume_index = XPAR_VOL_STANDALONE;
  
  if (len >= ARM_HDR_LEN) {
    xpar_arm_prologue pr;  xpar_armour_params ap;  u8 * plain;  sz plen;
    int copy = -1;
    if (!xpar_garm_prologue(data, len, &pr, &copy)) goto packets;
    v->armoured_file = true;
    arm_params_of(&pr, &ap);
    plain = arm_extract(&ap, data + ARM_HDR_LEN, (u64) len - ARM_HDR_LEN,
                        pr.plain_length, &plen,
                        c->key_loaded ? &c->key : NULL);
    if (!plain) {
      xpar_fprintf(xpar_stderr, "xpar: '%s': the armoured region is "
                   "shorter than its prologue says.\n", path);
      return;
    }
    chain_blob(c, plain);
    chain_scan(c, v, plain, plen, false);
    return;
  }

packets:
  chain_scan(c, v, data, len, false);
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

static void chain_gather(const xpar_options * o, xpar_chain * c) {
  u32 i;
  for (i = 0; i < o->set_ref.count; i++)
    chain_add_vol(c, xpar_strdup(o->set_ref.vol[i]));

  if (o->set_ref.base) {
    char * dir;  char * stem;  xpar_dir * d;
    gen_split_path(o->set_ref.base, &dir, &stem);
    chain_strip_gen(stem);
    c->base = xpar_path_join(dir, stem);
    c->dir  = xpar_strdup(dir);
    d = xpar_opendir(*dir ? dir : ".");
    if (d) {
      const xpar_dirent * e;
      while ((e = xpar_readdir(d)) != NULL)
        if (!e->is_dir &&
            (o->chain_metadata_only
               ? gen_chain_index_sibling(e->name, stem)
               : gen_chain_sibling(e->name, stem)))
          chain_add_vol(c, xpar_path_join(dir, e->name));
      xpar_closedir(d);
    }
    xpar_free(dir);  xpar_free(stem);
  } else if (o->set_ref.dir) {
    c->dir = xpar_strdup(o->set_ref.dir);
  }

  if (o->scan_dir) {
    xpar_dir * d = xpar_opendir(o->scan_dir);
    if (d) {
      const xpar_dirent * e;
      char * pfx;
      xpar_asprintf(&pfx, "%s/", o->scan_dir);
      while ((e = xpar_readdir(d)) != NULL)
        if (!e->is_dir && xpar_path_ends_with(e->name, XPAR_EXT))
          chain_add_vol(c, xpar_path_join(pfx, e->name));
      xpar_free(pfx);
      xpar_closedir(d);
    }
  }
}

static void chain_link(xpar_chain * c) {
  u32 i, j, n = 0, heads = 0;
  for (i = 0; i < c->crit.count; i++)
    if (xpar_pkt_is(&c->crit.pkt[i].hdr, XPAR_T_SETD)) n++;
  if (!n) FATAL_FORMAT("No set descriptor found; this is not an xpar 2 set.");

  c->gen = (xpar_chain_gen *) xpar_calloc(n, sizeof(xpar_chain_gen));
  for (i = 0; i < c->crit.count; i++) {
    const xpar_crit_pkt * p = &c->crit.pkt[i];
    xpar_chain_gen * g;
    xpar_status st;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_SETD)) continue;
    g = &c->gen[c->gen_count];
    st = xpar_setd_read(p->body, (sz) p->body_len, &g->sd);
    if (st != XPAR_OK && st != XPAR_E_UNSUPPORTED) {
      xpar_fprintf(xpar_stderr,
                   "xpar: a set descriptor is unreadable (%s); ignored.\n",
                   xpar_status_str(st));
      continue;
    }
    if (st == XPAR_E_UNSUPPORTED)
      xpar_fprintf(xpar_stderr, "xpar: generation %u requires features this "
                   "build does not implement (0x%08lx).\n", g->sd.generation,
                   (unsigned long) g->sd.required_features);
    xpar_memcpy(g->set_id, p->hdr.set_id, XPAR_SET_ID_LEN);
    g->parent = XPAR_GEN_NONE;
    c->gen_count++;
  }
  if (!c->gen_count) FATAL_FORMAT("Every set descriptor is malformed.");

  /*  Insertion sort by generation number: a chain is short and the order
      is what every later walk assumes.  */
  for (i = 1; i < c->gen_count; i++) {
    xpar_chain_gen t = c->gen[i];
    j = i;
    while (j && c->gen[j - 1].sd.generation > t.sd.generation) {
      c->gen[j] = c->gen[j - 1];  j--;
    }
    c->gen[j] = t;
  }

  for (i = 0; i < c->gen_count; i++) {
    xpar_chain_gen * g = &c->gen[i];
    if (!g->sd.generation) continue;
    for (j = 0; j < c->gen_count; j++)
      if (j != i && !xpar_memcmp(g->sd.parent_set_id, c->gen[j].set_id,
                                 XPAR_SET_ID_LEN)) { g->parent = j;  break; }
    if (g->parent == XPAR_GEN_NONE) g->parent_missing = true;
    else {
      xpar_status st = xpar_setd_check_parent(&g->sd,
                                              c->gen[g->parent].set_id,
                                              &c->gen[g->parent].sd);
      if (st != XPAR_OK)
        FATAL_FORMAT("Generation %u does not follow generation %u: %s.",
                     g->sd.generation, c->gen[g->parent].sd.generation,
                     xpar_status_str(st));
    }
  }

  c->head = XPAR_GEN_NONE;
  for (i = 0; i < c->gen_count; i++) {
    bool named = false;
    for (j = 0; j < c->gen_count; j++)
      if (c->gen[j].parent == i) {
        if (named) c->forked = true;
        named = true;
      }
    if (!named) { c->head = i;  heads++; }
  }
  if (heads > 1) c->forked = true;
  if (c->head == XPAR_GEN_NONE) c->head = c->gen_count - 1;
}

static void chain_map_volumes(xpar_chain * c) {
  u32 i, j;
  for (i = 0; i < c->vol_count; i++) {
    if (!c->vol[i].has_volh) continue;
    for (j = 0; j < c->gen_count; j++)
      if (!xpar_memcmp(c->vol[i].set_id, c->gen[j].set_id, XPAR_SET_ID_LEN)) {
        c->vol[i].gen = j;
        c->gen[j].vol_count++;
        if (!c->gen[j].layt_body && c->vol[i].layt_body) {
          c->gen[j].layt_body = c->vol[i].layt_body;
          c->gen[j].layt_len  = c->vol[i].layt_len;
        }
        break;
      }
  }
  for (i = 0; i < c->crit.count; i++) {
    const xpar_crit_pkt * p = &c->crit.pkt[i];
    if (!xpar_pkt_is(&p->hdr, XPAR_T_RCVS) || p->body_len < 8) continue;
    for (j = 0; j < c->gen_count; j++)
      if (!xpar_memcmp(p->hdr.set_id, c->gen[j].set_id, XPAR_SET_ID_LEN)) {
        u64 e = xpar_rd64(p->body);
        c->gen[j].recovery_count++;
        if (e + 1 > c->gen[j].recovery_top) c->gen[j].recovery_top = e + 1;
        break;
      }
  }
}

void xpar_gchain_load(const xpar_options * o, xpar_chain * c) {
  xpar_memset(c, 0, sizeof *c);
  xpar_critset_init(&c->crit);
  if (o->auth_key) {
    xpar_keyfile_status ks = xpar_keyfile_load(o->auth_key, &c->key,
                                               c->master);
    if (ks == XPAR_KEYFILE_OPEN) FATAL_PERROR(o->auth_key);
    if (ks == XPAR_KEYFILE_EMPTY)
      FATAL_CODE(XPAR_EXIT_AUTH, "The key file is empty.");
    if (ks != XPAR_KEYFILE_OK)
      FATAL_CODE(XPAR_EXIT_AUTH, "Reading key file '%s' failed.", o->auth_key);
    c->key_loaded = true;
  }
  chain_gather(o, c);
  if (!c->vol_count) FATAL_FORMAT("No readable volume of this set.");
  chain_link(c);
  chain_map_volumes(c);
  /*  Derive a writable base from the oldest index when input was a
      directory.  */
  if (!c->base) {
    u32 j;
    for (j = 0; j < c->vol_count; j++) {
      char * stem;  char * dir;
      if (c->vol[j].volume_kind != XPAR_VOL_INDEX) continue;
      if (!xpar_path_ends_with(c->vol[j].path, XPAR_EXT)) continue;
      gen_split_path(c->vol[j].path, &dir, &stem);
      stem[xpar_strlen(stem) - XPAR_EXT_LEN] = 0;
      chain_strip_gen(stem);
      c->base = xpar_path_join(dir, stem);
      if (!c->dir) c->dir = xpar_strdup(dir);
      xpar_free(dir);  xpar_free(stem);
      break;
    }
  }
  if (c->crit.conflicts)
    FATAL_FORMAT("Replicated packets verify but disagree.");
  for (u32 j = 0; j < c->crit.count; j++) {
    const xpar_crit_pkt * p = &c->crit.pkt[j];
    xpar_auth a;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_AUTH) ||
        xpar_auth_read(p->body, (sz) p->body_len, &a) != XPAR_OK) continue;
    if (c->key_loaded && !xpar_auth_key_ok(&a, c->master))
      FATAL_CODE(XPAR_EXIT_AUTH,
                 "The authentication key is wrong for this set.");
    c->authenticated = true;
    c->auth_only = !a.unkeyed_retained;
  }
  if (c->key_loaded && !c->authenticated)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "An authentication key was supplied, but this set has no "
               "AUTH descriptor.");
}

void xpar_gchain_free(xpar_chain * c) {
  u32 i;
  for (i = 0; i < c->vol_count; i++) {
    xpar_free(c->vol[i].path);  xpar_free(c->vol[i].data);
  }
  for (i = 0; i < c->blob_count; i++) xpar_free(c->blob[i]);
  for (i = 0; i < c->gen_count; i++) xpar_setd_free(&c->gen[i].sd);
  xpar_free(c->vol);  xpar_free(c->blob);  xpar_free(c->gen);
  xpar_free(c->base);  xpar_free(c->dir);
  xpar_critset_free(&c->crit);
  xpar_key_forget(&c->key, c->master);
  xpar_memset(c, 0, sizeof *c);
}

static const xpar_key * gen_chain_key(const xpar_chain * c) {
  return c->key_loaded ? &c->key : NULL;
}

static void gen_require_write_key(const xpar_chain * c, const char * verb) {
  if (c->authenticated && !c->key_loaded)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "%s on an authenticated set requires --auth-key=FILE; "
               "keyless access is read-only.", verb);
}

u32 xpar_gchain_select(const xpar_chain * c, const xpar_genref * g) {
  u32 i, found = XPAR_GEN_NONE, matches = 0;
  if (!g) {
    if (c->forked)
      FATAL("This chain has forked: two generations name the same parent. "
            "Name one with --generation; xpar will not guess.");
    return c->head;
  }
  if (g->by_id) {
    for (i = 0; i < c->gen_count; i++)
      if (xpar_hex_prefix(c->gen[i].set_id, XPAR_SET_ID_LEN, g->id_prefix)) {
        found = i;  matches++;
      }
    if (matches > 1)
      FATAL("Set-id prefix '%s' is ambiguous; provide more hexadecimal "
            "digits.", g->id_prefix);
    if (matches == 1) return found;
    FATAL("No generation of this set has a set_id beginning '%s'.",
          g->id_prefix);
  }
  for (i = 0; i < c->gen_count; i++)
    if (c->gen[i].sd.generation == (u32) g->number) {
      found = i;  matches++;
    }
  if (matches > 1)
    FATAL("Generation %llu is ambiguous across fork branches; select it "
          "by set-id prefix.", (unsigned long long) g->number);
  if (matches == 1) return found;
  FATAL("This set has no generation %llu.", (unsigned long long) g->number);
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
  u32 h = g, i;
  for (;;) {
    const xpar_crit_pkt * p = xpar_critset_find_file(
                                &c->crit, c->gen[h].set_id, file_id);
    if (p && p->body_len >= XPAR_SET_ID_LEN &&
        !xpar_memcmp(p->body, file_id, XPAR_SET_ID_LEN)) {
      *owner = h;  return p;
    }
    for (i = 0; i < c->crit.count; i++) {
      const xpar_crit_pkt * q = &c->crit.pkt[i];
      if (!xpar_pkt_is(&q->hdr, XPAR_T_FILE)) continue;
      if (q->body_len < XPAR_SET_ID_LEN) continue;
      if (xpar_memcmp(q->hdr.set_id, c->gen[h].set_id, XPAR_SET_ID_LEN))
        continue;
      if (!xpar_memcmp(q->body, file_id, XPAR_SET_ID_LEN)) {
        *owner = h;  return q;
      }
    }
    if (c->gen[h].parent == XPAR_GEN_NONE) return NULL;
    h = c->gen[h].parent;
  }
}

void xpar_gchain_manifest(const xpar_chain * c, u32 g, xpar_manifest * m,
                          u32 ** owner) {
  const xpar_setd * sd = &c->gen[g].sd;
  u32 i, * own;

  xpar_memset(m, 0, sizeof *m);
  own = (u32 *) xpar_calloc(sd->file_count ? sd->file_count : 1, sizeof(u32));
  for (i = 0; i < sd->file_count; i++) {
    u32 h = XPAR_GEN_NONE;
    const xpar_crit_pkt * p = chain_file_pkt(c, g, sd->file_id[i], &h);
    xpar_entry * e;
    xpar_status st;
    if (!p)
      FATAL_FORMAT("Generation %u names a manifest entry no generation "
                   "owns; the chain is incomplete.", sd->generation);
    e  = xpar_manifest_append(m);
    st = xpar_entry_read(p->body, (sz) p->body_len,
                         c->gen[h].sd.posix_record_count, e);
    if (st != XPAR_OK)
      FATAL_FORMAT("A manifest entry of generation %u is unreadable (%s).",
                   c->gen[h].sd.generation, xpar_status_str(st));
    if (e->posix_index != XPAR_ABSENT_U32 &&
        e->posix_index >= c->gen[h].sd.posix_record_count)
      FATAL_FORMAT("Manifest entry %u names a POSX record outside generation "
                   "%u's table.", i, c->gen[h].sd.generation);
    own[i] = h;
  }
  m->stream_base   = sd->stream_base;
  m->stream_length = sd->stream_length;
  m->dedup_level   = sd->dedup_level;
  m->align         = sd->align;
  m->slice_size    = sd->slice_size;
  {
    xpar_mf_limits lim;
    xpar_mf_result res;
    xpar_gen_range * anc;
    u32 * lineage;
    u32 na = 0, h = c->gen[g].parent;
    anc = (xpar_gen_range *) xpar_calloc(c->gen_count ? c->gen_count : 1,
                                         sizeof(xpar_gen_range));
    lineage = (u32 *) xpar_calloc(c->gen_count ? c->gen_count : 1,
                                  sizeof(u32));
    while (h != XPAR_GEN_NONE && na < c->gen_count) {
      lineage[na++] = h;
      h = c->gen[h].parent;
    }
    FATAL_UNLESS("The selected generation's ancestry is cyclic.",
                 h == XPAR_GEN_NONE);
    for (i = 0; i < na; i++) {
      u32 a = lineage[na - i - 1];
      anc[i].base = c->gen[a].sd.stream_base;
      anc[i].length = c->gen[a].sd.stream_length;
    }
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
      FATAL_FORMAT("Manifest entry %u is invalid: %s.", res.entry,
                   xpar_mf_reason(res.status));
    if (res.link_meta_mismatch)
      xpar_fprintf(xpar_stderr, "xpar: %u hard-link aliases disagree with "
                   "their canonical metadata; canonical values will be "
                   "used.\n", res.link_meta_mismatch);
    xpar_free(anc);
  }
  {
    xpar_nameidx ix;
    xpar_posix_rec ** tab;
    u32 * tabn, posix_mismatch = 0;
    xpar_nameidx_build(m, &ix);
    tab = (xpar_posix_rec **) xpar_calloc(c->gen_count, sizeof *tab);
    tabn = (u32 *) xpar_calloc(c->gen_count, sizeof *tabn);
    for (i = 0; i < c->gen_count; i++)
      tabn[i] = xpar_gchain_posix(c, i, &tab[i]);
    for (i = 0; i < m->count; i++) {
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
      own[i] = own[target];
    }
    if (posix_mismatch)
      xpar_fprintf(xpar_stderr, "xpar: %u hard-link aliases disagree with "
                   "their canonical POSX metadata; canonical values will "
                   "be used.\n", posix_mismatch);
    for (i = 0; i < c->gen_count; i++)
      xpar_gchain_posix_free(tab[i], tabn[i]);
    xpar_free(tab);  xpar_free(tabn);
    xpar_nameidx_free(&ix);
  }
  *owner = own;
}

u32 xpar_gchain_posix(const xpar_chain * c, u32 g, xpar_posix_rec ** out) {
  u32 count = c->gen[g].sd.posix_record_count;
  *out = NULL;
  if (xpar_posx_collect(&c->crit, c->gen[g].set_id, count, out) != XPAR_OK)
    FATAL_FORMAT("Generation %u's POSX table has gaps, overlaps, or invalid "
                 "ranges.", c->gen[g].sd.generation);
  return count;
}

void xpar_gchain_posix_free(xpar_posix_rec * rec, u32 count) {
  xpar_posix_records_free(rec, count);
}

i64 xpar_gchain_gen_of(const xpar_chain * c, u64 off, u64 len) {
  u32 i;
  for (i = 0; i < c->gen_count; i++) {
    u64 lo = c->gen[i].sd.stream_base;
    u64 hi = lo + c->gen[i].sd.stream_length;
    if (off >= lo && off < hi && len <= hi - off) return (i64) i;
  }
  return -1;
}

void xpar_gchain_deps(const xpar_chain * c, const xpar_manifest * m,
                      const u32 * owner, u64 * by_extent, u64 * by_packet) {
  u32 i, j, k;
  for (i = 0; i < c->gen_count; i++) { by_extent[i] = 0;  by_packet[i] = 0; }
  for (i = 0; i < m->count; i++) {
    const xpar_entry * e = &m->entry[i];
    by_packet[owner[i]]++;
    for (j = 0; j < c->gen_count; j++) {
      bool hit = false;
      for (k = 0; k < e->extent_count && !hit; k++) {
        i64 g = xpar_gchain_gen_of(c, e->extents[k].stream_offset,
                                   e->extents[k].length);
        if (g == (i64) j) hit = true;
      }
      if (hit) by_extent[j]++;
    }
  }
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
    if (off >= s->limit) {
      xpar_memset(out, 0, (sz) len);
      return;
    }
    take = MIN(len, s->limit - off);
    if (take && !s->read(s->read_ctx, off, out, take))
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "The selected generation's stored stream became "
                 "unreadable. Nothing was written.");
    if (take < len) xpar_memset(out + take, 0, (sz) (len - take));
    return;
  }
  while (len) {
    xpar_occurrence occ;  u64 run = 0, take, at;
    if (off >= s->limit || !xpar_occindex_canonical(&s->ix, off, &occ, &run)) {
      /*  Past L: the zero padding of 4.1, which is never stored and is
          regenerated here for the coder and the tags alike.  */
      xpar_memset(out, 0, (sz) len);
      return;
    }
    take = MIN(run, len);
    if (occ.entry != s->open_entry) {
      const char * path = s->m->source ? s->m->source[occ.entry] : NULL;
      if (s->open_file) xpar_close(s->open_file);
      s->open_file  = NULL;
      s->open_entry = occ.entry;
      if (!path)
        FATAL("Entry %lu has no readable source; the stream cannot be "
              "rebuilt.", (unsigned long) occ.entry);
      s->open_file = xpar_open(path, XPAR_O_RDONLY);
      if (!s->open_file)
        FATAL_IO("Cannot open '%s': %s.", path, xpar_strerror(xpar_errno()));
    }
    /*  Offset the read within the occurrence, not merely its entry.  */
    at = occ.file_offset + (off - occ.stream_offset);
    if (xpar_pread(s->open_file, out, (sz) take, at) != (sz) take)
      FATAL_IO("Short read from '%s' at %llu.",
               s->m->source[occ.entry], (unsigned long long) at);
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

static const char * gen_codec_name(u8 c) {
  return c == XPAR_CODEC_FFT_LOW ? "fft-low"
       : c == XPAR_CODEC_FFT ? "fft" : "matrix";
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
    rq.cell_bytes    = 0;
    rq.field_log2    = p->field_log2;
    st = xpar_geom_choose(&rq, &p->geom);
    if (st == XPAR_GEOM_FIELD && o->field == XPAR_CLI_AUTO &&
        p->field_log2 == 8) {
      p->field_log2 = 16;
      st = xpar_geom_choose(&rq, &p->geom);
    }
    if (st != XPAR_GEOM_OK)
      FATAL("Cannot choose a geometry: %s.", xpar_geom_reason(st));
    if (o->align == XPAR_ALIGN_1K &&
        (p->geom.slice_size < XPAR_BLAKE3_CHUNK_LEN ||
         (p->geom.slice_size & (p->geom.slice_size - 1)) != 0)) {
      FATAL_UNLESS("--align=1k needs a power-of-two slice size of at least "
                   "1 KiB; the explicit geometry does not provide one.",
                   !o->slice_size && !o->slices);
      rq.slice_size = xpar_next_pow2(MAX(p->geom.slice_size,
                                         (u64) XPAR_BLAKE3_CHUNK_LEN));
      rq.slice_count = 0;
      st = xpar_geom_choose(&rq, &p->geom);
      if (st != XPAR_GEOM_OK)
        FATAL("Cannot choose a geometry: %s.", xpar_geom_reason(st));
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
                 "The %s codec cannot express S=%llu, R=%llu over GF(2^%u). "
                 "The FFT code needs the rounded smaller axis plus the "
                 "larger axis to fit in 2^field; --codec=matrix has no "
                 "power-of-two rounding constraint.",
                 gen_codec_name(p->codec),
                 (unsigned long long) p->geom.slice_count,
                 (unsigned long long) r, p->field_log2);
  }

  /*  Matrix uses the whole field axis; FFT records its power-of-two
      recovery bracket.  */
  if (p->codec == XPAR_CODEC_MATRIX) {
    p->axis = p->field_log2;
  } else if (p->codec == XPAR_CODEC_FFT_LOW) {
    p->axis = (u8) xpar_log2_floor(
                         xpar_next_pow2(p->geom.slice_count));
  } else {
    u64 m = xpar_next_pow2(r ? r : 1);
    maxr = gen_resolve_r(&o->max_recovery, p->geom.slice_count,
                         p->geom.slice_size);
    if (maxr > m) {
      u64 wide = xpar_next_pow2(maxr);
      if (!xpar_codec_supports(p->codec, p->field_log2, p->geom.slice_count,
                               wide))
        FATAL_CODE(XPAR_EXIT_NOPLAN,
                   "--max-recovery=%llu needs a recovery axis of %llu, "
                   "which this field and S cannot express.",
                   (unsigned long long) maxr, (unsigned long long) wide);
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
  u64   rec_count;
} gen_tables;

static void gen_tables_free(gen_tables * t) {
  xpar_free(t->slice_crc);  xpar_free(t->slice_tag);
  xpar_free(t->cell_crc);   xpar_free(t->rec);
  if (t->rec_spill) xpar_close(t->rec_spill);
  if (t->rec_path) xpar_remove(t->rec_path);
  xpar_free(t->rec_path);
  xpar_memset(t, 0, sizeof *t);
}

static u64 gen_default_budget(void) {
  u64 phys = xpar_physical_memory();
  u64 cap = sizeof(void *) >= 8 ? ((u64) 1 << 30) : ((u64) 512 << 20);
  u64 want = phys ? phys / 4 : cap;
  if (want > cap) want = cap;
  if (want < ((u64) 1 << 20)) want = (u64) 1 << 20;
  return want;
}

static void gen_rec_spill_open(gen_tables * t, const char * base) {
  u32 i;
  for (i = 0; i < 1000; i++) {
    xpar_asprintf(&t->rec_path, "%s.xpar-encode-tmp-%03u", base, i);
    t->rec_spill = xpar_open(t->rec_path, XPAR_O_RDWR | XPAR_O_CREAT |
                                          XPAR_O_EXCL);
    if (t->rec_spill) return;
    xpar_free(t->rec_path);  t->rec_path = NULL;
  }
  FATAL_IO("Cannot create an encoding scratch file beside '%s': %s.", base,
           xpar_strerror(xpar_errno()));
}

static void gen_rec_put(gen_tables * t, u64 e, u64 off,
                        const u8 * p, u64 n) {
  if (t->rec) {
    xpar_memcpy(t->rec + e * t->rec_z + off, p, (sz) n);
    return;
  }
  if (xpar_pwrite(t->rec_spill, p, (sz) n, e * t->rec_z + off) != (sz) n)
    FATAL_IO("Cannot write encoding scratch '%s'.", t->rec_path);
}

static const u8 * gen_rec_get(gen_tables * t, u64 e, u8 * scratch) {
  if (t->rec) return t->rec + e * t->rec_z;
  if (xpar_pread(t->rec_spill, scratch, (sz) t->rec_z,
                 e * t->rec_z) != (sz) t->rec_z)
    FATAL_IO("Cannot read encoding scratch '%s'.", t->rec_path);
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
  u64 c, i, j, chunk, budget = memory ? memory : gen_default_budget();
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
  t->rec_count = R;
  if (!S) return;

  if (Y && K && S > (u64) -1 / K)
    FATAL_CODE(XPAR_EXIT_NOPLAN, "The cell checksum table is too large.");
  if (Y) cells = S * K;
  if (S > ((u64) -1) / (4 + tag_len) ||
      cells > (((u64) -1) - S * (4 + tag_len)) / 4)
    FATAL_CODE(XPAR_EXIT_NOPLAN, "The checksum tables are too large.");
  meta = S * (4 + tag_len) + cells * 4;

  if (S > ((u64) (sz) -1) / 4 ||
      (tag_len && S > ((u64) (sz) -1) / tag_len) ||
      cells > ((u64) (sz) -1) / 4 || meta > budget || Z > budget - meta)
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "The checksum tables plus one %llu-byte slice need more than "
               "-m %llu; raise -m or choose a smaller slice size.",
               (unsigned long long) Z, (unsigned long long) budget);

  t->slice_crc = (u32 *) xpar_calloc((sz) S, 4);
  if (tag_len)
    t->slice_tag = (u8 *) xpar_calloc((sz) S, tag_len);
  if (Y) t->cell_crc = (u32 *) xpar_calloc((sz) cells, 4);

  data = (u8 *) xpar_alloc_raw((sz) Z);
  gen_src_init(&src, m, m->stream_base + m->stream_length);
  gen_src_use_reader(&src, read, read_ctx);
  for (i = 0; i < S; i++) {
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
    if (prog) xpar_progress_tick(prog, Z);
  }
  gen_src_free(&src);
  if (!R) { xpar_free(data);  return; }

  codec = xpar_codec_new_axis(p->codec, p->field_log2, S, R, p->axis);
  if (meta + xpar_codec_encode_footprint_axis(
               p->codec, p->field_log2, S, R, p->axis, (sz) Z) <= budget &&
      R <= ((u64) (sz) -1) / Z) {
    t->rec = (u8 *) xpar_alloc_raw((sz) (R * Z));
  } else {
    gen_rec_spill_open(t, scratch_base);
  }

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
                 "The matrix encoder needs one data slice and one recovery "
                 "accumulator (%llu bytes each), which do not fit -m %llu.",
                 (unsigned long long) Z, (unsigned long long) budget);
    pool = t->rec ? NULL : (u8 *) xpar_alloc_aligned((sz) (batch * Z), 64);
    rptr = (u8 **) xpar_alloc_raw((sz) batch * sizeof(u8 *));
    while (first < R) {
      u64 nr = MIN(batch, R - first);
      for (j = 0; j < nr; j++)
        rptr[j] = t->rec ? t->rec + (first + j) * Z : pool + j * Z;
      gen_src_init(&src, m, m->stream_base + m->stream_length);
      gen_src_use_reader(&src, read, read_ctx);
      for (i = 0; i < S; i++) {
        gen_src_read(&src, m->stream_base + i * Z, Z, data);
        if (xpar_codec_matrix_accumulate(codec, i, data, first, rptr, nr,
                                         (sz) Z, i == 0) != XPAR_CODEC_OK)
          FATAL_CODE(XPAR_EXIT_INTERNAL, "internal: matrix streaming encode "
                     "refused a supported range.");
      }
      gen_src_free(&src);
      if (!t->rec)
        for (j = 0; j < nr; j++) gen_rec_put(t, first + j, 0, rptr[j], Z);
      first += nr;
    }
    xpar_free(rptr);
    if (pool) xpar_free_aligned(pool);
  } else {
    const u8 ** dptr;
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
                 "The FFT encoder's minimum 64-byte column does not fit "
                 "-m %llu.", (unsigned long long) budget);
    xpar_free(data);  data = NULL;
    pool = (u8 *) xpar_alloc_aligned(
             (sz) ((S + (t->rec ? 0 : R)) * chunk), 64);
    dptr = (const u8 **) xpar_alloc_raw((sz) S * sizeof(u8 *));
    rptr = (u8 **) xpar_alloc_raw((sz) R * sizeof(u8 *));
    for (i = 0; i < S; i++) dptr[i] = pool + i * chunk;
    if (!t->rec)
      for (j = 0; j < R; j++) rptr[j] = pool + (S + j) * chunk;
    gen_src_init(&src, m, m->stream_base + m->stream_length);
    gen_src_use_reader(&src, read, read_ctx);
    for (c = 0; c < Z; c += chunk) {
      u64 len = MIN(chunk, Z - c);
      for (i = 0; i < S; i++)
        gen_src_read(&src, m->stream_base + i * Z + c, len, (u8 *) dptr[i]);
      if (t->rec)
        for (j = 0; j < R; j++) rptr[j] = t->rec + j * Z + c;
      if (xpar_codec_encode(codec, dptr, rptr, (sz) len) != XPAR_CODEC_OK)
        FATAL_CODE(XPAR_EXIT_INTERNAL, "internal: FFT encode refused a "
                   "supported geometry.");
      if (!t->rec)
        for (j = 0; j < R; j++) gen_rec_put(t, j, c, rptr[j], len);
    }
    gen_src_free(&src);
    xpar_free(dptr);  xpar_free(rptr);  xpar_free_aligned(pool);
  }
  xpar_codec_free(codec);
  xpar_free(data);
}

/*  Critical metadata uses GF(2^8), t = 16 for correctable fraction.  */
static void gen_armour_params(const xpar_options * o,
                              xpar_armour_params * p) {
  u32 t = 16;
  xpar_armour_defaults(p, o->armour_field == 16 ? 16 : 8);
  if (o->armour_t) t = o->armour_t;
  else if (o->armour_pct > 0.0)
    t = (u32) (o->armour_pct / 100.0 * (f64) p->n / 2.0 + 0.5);
  if (!t) t = 1;
  if (t > (p->n - 1) / 2) t = (p->n - 1) / 2;
  p->k = p->n - 2 * t;
  if (o->depth) p->depth = o->depth;
  else if (o->burst) {
    p->depth = xpar_ceil_div(o->burst + 1, (u64) t * (p->symbol_bits / 8));
    if (!p->depth) p->depth = 1;
  }
}

static void gen_armour_pack(xpar_buf * out, const xpar_options * o,
                            const u8 * plain, sz plain_len,
                            const u8 * set_id, const xpar_key * key) {
  xpar_armour_params ap;  xpar_armour * a;  xpar_armg ag;  u8 * arm;
  const char * why;
  gen_armour_params(o, &ap);
  why = xpar_armour_check(&ap);
  if (why) FATAL("Armour parameters are not servable: %s", why);
  a = xpar_armour_new(&ap);
  xpar_memset(&ag, 0, sizeof ag);
  ag.symbol_bits     = (u8) ap.symbol_bits;
  ag.poly            = ap.poly;   ag.n = ap.n;  ag.k = ap.k;
  ag.fcr             = ap.fcr;    ag.prim = ap.prim;
  ag.depth           = ap.depth;
  ag.plain_length    = plain_len;
  ag.armoured_length = xpar_armour_size(a, plain_len);
  arm = (u8 *) xpar_calloc((sz) ag.armoured_length, 1);
  xpar_armour_encode(a, arm, plain, plain_len);
  xpar_armg_write(out, &ag, arm, set_id, key);
  xpar_free(arm);
  xpar_armour_free(a);
}

typedef struct {
  const xpar_armour * a;
  xpar_file * f;
  u8 * frame;
  u64 cap, fill;
} gen_armsink;

static void gen_as_init(gen_armsink * s, const xpar_armour * a,
                        xpar_file * f) {
  s->a = a;  s->f = f;
  s->cap = xpar_armour_frame_plain(a);
  s->frame = (u8 *) xpar_calloc((sz) xpar_armour_frame_disk(a), 1);
  s->fill = 0;
}

static void gen_as_flush(gen_armsink * s) {
  if (!s->fill) return;
  xpar_memset(s->frame + s->fill, 0, (sz) (s->cap - s->fill));
  xpar_armour_encode_frame(s->a, s->frame);
  xpar_xwrite(s->f, s->frame, (sz) xpar_armour_frame_disk(s->a));
  s->fill = 0;
}

static void gen_as_put(gen_armsink * s, const void * data, u64 len) {
  const u8 * p = (const u8 *) data;
  while (len) {
    u64 take = MIN(len, s->cap - s->fill);
    xpar_memcpy(s->frame + s->fill, p, (sz) take);
    s->fill += take;  p += take;  len -= take;
    if (s->fill == s->cap) gen_as_flush(s);
  }
}

static void gen_as_free(gen_armsink * s) {
  xpar_free(s->frame);
  xpar_memset(s, 0, sizeof *s);
}

static void gen_strm_header(xpar_buf * out, u64 stream_len,
                            const u8 * set_id, const xpar_key * key) {
  const sz fixed = XPAR_PKT_HDR + 16;
  xpar_blake3_t h;
  u8 * p;
  xpar_strm_write(out, 0, NULL, 0, set_id, key);
  p = out->data + out->len - fixed;
  xpar_wr64(p + 8, xpar_align_up(fixed + stream_len, XPAR_PKT_ALIGN));
  if (key) xpar_blake3_init_keyed(&h, key->k_pkt);
  else     xpar_blake3_init(&h);
  xpar_blake3_update(&h, p, 40);
  xpar_blake3_final(&h, p + 40, 8);
}

static void gen_write_arm_prologue(xpar_file * f,
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
  for (i = 0; i < 3; i++) xpar_xwrite(f, copy, sizeof copy);
}

/*  Publish rebuilt armour with regenerated prologue lengths and copies.  */
void xpar_garm_write_plain(const char * path, const xpar_armour_params * ap,
                           const u8 * plain, u64 plain_len,
                           u64 stream_offset, u64 stream_len) {
  xpar_garm_write_patched(path, ap, plain, plain_len, stream_offset,
                          stream_len, NULL, NULL, 0, 0);
}

/*  Re-armour mapped plaintext with sparse repaired-slice replacements.  */
void xpar_garm_write_patched(const char * path,
                             const xpar_armour_params * ap,
                             const u8 * plain, u64 plain_len,
                             u64 stream_offset, u64 stream_len,
                             xpar_file * staged, const u64 * slot,
                             u64 slice_count, u64 slice_size) {
  xpar_armour * a = xpar_armour_new(ap);
  gen_armsink sink;
  xpar_file * f;
  char * tmp;
  u8 * io = NULL;
  u64 at = 0, stream_end;
  FATAL_UNLESS("The armoured maintenance parameters are invalid.", a != NULL);
  FATAL_UNLESS("The armoured protected stream lies outside its plaintext.",
               stream_offset <= plain_len && stream_len <= plain_len -
                 stream_offset);
  stream_end = stream_offset + stream_len;
  if (staged) io = (u8 *) xpar_alloc_raw(1u << 16);
  f = gen_stage_open(path, &tmp);
  gen_write_arm_prologue(f, ap, plain_len, xpar_armour_size(a, plain_len),
                         stream_offset, stream_len);
  gen_as_init(&sink, a, f);
  while (at < plain_len) {
    u64 take;
    if (!staged || !slot || at < stream_offset || at >= stream_end) {
      u64 boundary = at < stream_offset ? stream_offset : plain_len;
      if (at >= stream_end) boundary = plain_len;
      take = MIN(boundary - at, (u64) 1 << 20);
      gen_as_put(&sink, plain + at, take);
    } else {
      u64 rel = at - stream_offset;
      u64 slice = slice_size ? rel / slice_size : 0;
      u64 in = slice_size ? rel % slice_size : 0;
      FATAL_UNLESS("A staged armoured slice lies outside the stream.",
                   slice < slice_count && slice_size != 0);
      take = MIN(MIN(stream_end - at, slice_size - in),
                 (u64) 1 << 16);
      if (slot[slice] == UINT64_MAX) {
        gen_as_put(&sink, plain + at, take);
      } else {
        if (xpar_pread(staged, io, (sz) take,
                       slot[slice] * slice_size + in) != (sz) take)
          FATAL_IO("Reading staged armoured repair bytes failed.");
        gen_as_put(&sink, io, take);
      }
    }
    at += take;
  }
  gen_as_flush(&sink);
  gen_as_free(&sink);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("Flushing rebuilt armoured archive '%s' failed.", tmp);
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
  gen_armsink sink;
  xpar_file * f;
  char * tmp;
  FATAL_UNLESS("The armoured maintenance parameters are invalid.", a != NULL);
  FATAL_UNLESS("The armoured metadata insertion lies outside its plaintext.",
               insert <= plain_len &&
               stream_len <= UINT64_MAX - stream_offset &&
               (insert <= stream_offset ||
                insert >= stream_offset + stream_len) &&
               extra_len <= UINT64_MAX - plain_len);
  f = gen_stage_open(path, &tmp);
  gen_write_arm_prologue(f, ap, plain_len + extra_len,
                         xpar_armour_size(a, plain_len + extra_len),
                         stream_offset + (insert <= stream_offset
                                            ? extra_len : 0),
                         stream_len);
  gen_as_init(&sink, a, f);
  gen_as_put(&sink, plain, insert);
  gen_as_put(&sink, extra, extra_len);
  gen_as_put(&sink, plain + insert, plain_len - insert);
  gen_as_flush(&sink);
  gen_as_free(&sink);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("Flushing rebuilt armoured archive '%s' failed.", tmp);
  xpar_xclose(f);
  gen_publish_whole(tmp, path, true);
  xpar_armour_free(a);
}

static u64 gen_stream_tag(const xpar_manifest * m, u64 local_offset,
                          u64 length) {
  gen_src src;
  xpar_blake3_t h;
  u8 * buf = (u8 *) xpar_alloc_raw(1u << 16);
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

static void gen_write_data_range(const xpar_manifest * m,
                                 const char * path, u64 local_offset,
                                 u64 length, bool replace) {
  gen_src src;
  char * tmp;
  xpar_file * f;
  u8 * buf = (u8 *) xpar_alloc_raw(1u << 16);
  u64 at, left = length;
  if (local_offset > m->stream_length ||
      length > m->stream_length - local_offset)
    FATAL_CODE(XPAR_EXIT_INTERNAL,
               "internal: a data-volume range is outside its generation.");
  at = m->stream_base + local_offset;
  if (!replace && gen_exists(path))
    FATAL("'%s' exists; -f overwrites it.", path);
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
    FATAL_IO("Cannot flush temporary data volume '%s'.", tmp);
  xpar_xclose(f);
  gen_publish_whole(tmp, path, replace);
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
  gen_armsink sink;
  gen_src src;
  xpar_buf tail, crtr;
  xpar_file * f;
  char * tmp;
  u8 * buf;
  u64 stream_packet, stream_at, plain_len, at, left, e;
  gen_strm_header(head, m->stream_length, set_id, key);
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
  gen_write_arm_prologue(f, ap, plain_len, xpar_armour_size(a, plain_len),
                         stream_at, m->stream_length);
  gen_as_init(&sink, a, f);
  gen_as_put(&sink, head->data, head->len);
  buf = (u8 *) xpar_alloc_raw(1u << 16);
  gen_src_init(&src, m, m->stream_base + m->stream_length);
  gen_src_use_reader(&src, read, read_ctx);
  at = m->stream_base;  left = m->stream_length;
  while (left) {
    u64 take = MIN(left, (u64) 1 << 16);
    gen_src_read(&src, at, take, buf);
    gen_as_put(&sink, buf, take);
    at += take;  left -= take;
  }
  gen_src_free(&src);
  {
    u8 zero[XPAR_PKT_ALIGN] = { 0 };
    u64 pad = stream_packet - (XPAR_PKT_HDR + 16 + m->stream_length);
    if (pad) gen_as_put(&sink, zero, pad);
  }
  gen_as_put(&sink, tail.data, tail.len);
  for (e = 0; e < plan->recovery; e++) {
    xpar_buf pkt;
    const u8 * rec = gen_rec_get(tables, e, rec_scratch);
    xpar_buf_init(&pkt);
    xpar_rcvs_write(&pkt, e, rec, (sz) plan->geom.slice_size, set_id, key);
    gen_as_put(&sink, pkt.data, pkt.len);
    xpar_buf_free(&pkt);
  }
  gen_as_put(&sink, crtr.data, crtr.len);
  gen_as_flush(&sink);
  gen_as_free(&sink);
  xpar_free(buf);
  if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
    FATAL_IO("Cannot flush temporary armoured archive '%s'.", tmp);
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

  v = (gen_vol *) xpar_calloc(1, sizeof(gen_vol));
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
      v = (gen_vol *) xpar_realloc(v, (sz) (n + 1) * sizeof(gen_vol));
      xpar_memset(&v[n], 0, sizeof(gen_vol));
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

  {
    u64 max_first = 0, max_count = 1;
    for (i = 1; i < n; i++) {
      if (v[i].first > max_first) max_first = v[i].first;
      if (v[i].count > max_count) max_count = v[i].count;
    }
    gen_recovery_widths(max_first, max_count, &wf, &wc);
  }
  v[0].name = gen_name_index(base, gen);
  for (i = 1; i < n; i++)
    v[i].name = gen_name_recovery(base, gen, v[i].first, v[i].count, wf, wc);
  *count = n;
  return v;
}

static void gen_volumes_free(gen_vol * v, u32 n) {
  u32 i;
  for (i = 0; i < n; i++) xpar_free(v[i].name);
  xpar_free(v);
}

static bool gen_chain_names(const xpar_chain * c, const char * path) {
  u32 i;
  for (i = 0; i < c->vol_count; i++)
    if (!xpar_strcmp(c->vol[i].path, path)) return true;
  return false;
}

static bool gen_path_equal(const char * a, const char * b) {
  while (a[0] == '.' && (a[1] == '/' || a[1] == '\\')) a += 2;
  while (b[0] == '.' && (b[1] == '/' || b[1] == '\\')) b += 2;
  return !xpar_strcmp(a, b);
}

static bool gen_chain_data_names(const xpar_chain * c, const char * path) {
  u32 g;
  for (g = 0; g < c->gen_count; g++) {
    xpar_layt l;
    u32 i;
    if (!c->gen[g].layt_body ||
        xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, &l) !=
          XPAR_OK) continue;
    for (i = 0; i < l.count; i++) if (l.vol[i].kind == XPAR_VOL_DATA) {
      char * p = xpar_path_join(c->dir, l.vol[i].name);
      bool same = gen_path_equal(p, path);
      xpar_free(p);
      if (same) { xpar_layt_free(&l); return true; }
    }
    xpar_layt_free(&l);
  }
  return false;
}

/*  Retain rollback names until consolidation is fully published.  */
static void gen_commit_consolidation(const xpar_chain * c,
                                     const xpar_options * o,
                                     const char * stage_base,
                                     const char * final_base,
                                     const gen_plan * p) {
  gen_vol * stage, * final;
  char ** backup, ** stage_data = NULL, ** final_data = NULL;
  char ** data_backup = NULL;
  char ** stage_label = NULL, ** final_label = NULL;
  bool * published, * data_published = NULL, * data_moved = NULL;
  bool * label_published = NULL;
  u32 ns, nf, data_n = 0, i, moved = 0;
  int saved = 0;

  stage = gen_volumes(o, p->recovery, stage_base, 0, &ns);
  final = gen_volumes(o, p->recovery, final_base, 0, &nf);
  xpar_assert(ns == nf);
  backup = (char **) xpar_calloc(c->vol_count ? c->vol_count : 1,
                                 sizeof(char *));
  published = (bool *) xpar_calloc(nf ? nf : 1, sizeof(bool));
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    int width;
    data_n = o->volumes == XPAR_VOLS_FIXED ? o->volume_count : 1;
    if (!p->geom.slice_count) data_n = 1;
    else if (data_n > p->geom.slice_count) data_n = (u32) p->geom.slice_count;
    width = xpar_digits10(data_n ? data_n - 1 : 0);
    if (width < 2) width = 2;
    stage_data = (char **) xpar_calloc(data_n, sizeof(char *));
    final_data = (char **) xpar_calloc(data_n, sizeof(char *));
    data_backup = (char **) xpar_calloc(data_n, sizeof(char *));
    data_published = (bool *) xpar_calloc(data_n, sizeof(bool));
    data_moved = (bool *) xpar_calloc(data_n, sizeof(bool));
    if (o->labels) {
      stage_label = (char **) xpar_calloc(data_n, sizeof(char *));
      final_label = (char **) xpar_calloc(data_n, sizeof(char *));
      label_published = (bool *) xpar_calloc(data_n, sizeof(bool));
    }
    for (i = 0; i < data_n; i++) {
      stage_data[i] = gen_name_data(stage_base, 0, i, width);
      final_data[i] = gen_name_data(final_base, 0, i, width);
      if (o->labels) {
        xpar_asprintf(&stage_label[i], "%s" XPAR_EXT, stage_data[i]);
        xpar_asprintf(&final_label[i], "%s" XPAR_EXT, final_data[i]);
      }
    }
  }

  for (i = 0; i < nf; i++)
    if (gen_exists(final[i].name) && !gen_chain_names(c, final[i].name) &&
        !o->force) {
      u32 k;
      for (k = 0; k < ns; k++) xpar_remove(stage[k].name);
      FATAL("'%s' exists and is not a volume of the chain being replaced; "
            "-f overwrites it.", final[i].name);
    }
  for (i = 0; i < data_n; i++)
    if (gen_exists(final_data[i]) &&
        !gen_chain_data_names(c, final_data[i]) && !o->force) {
      u32 k;
      for (k = 0; k < ns; k++) xpar_remove(stage[k].name);
      for (k = 0; k < data_n; k++) xpar_remove(stage_data[k]);
      FATAL("'%s' exists and is not a data volume of the chain being "
            "replaced; -f overwrites it.", final_data[i]);
    }
  for (i = 0; i < data_n && o->labels; i++)
    if (gen_exists(final_label[i]) &&
        !gen_chain_names(c, final_label[i]) && !o->force) {
      u32 k;
      for (k = 0; k < ns; k++) xpar_remove(stage[k].name);
      for (k = 0; k < data_n; k++) {
        xpar_remove(stage_data[k]); xpar_remove(stage_label[k]);
      }
      FATAL("'%s' exists and is not a label of the chain being replaced; "
            "-f overwrites it.", final_label[i]);
    }

  for (i = 0; i < c->vol_count; i++) {
    u32 suffix;
    for (suffix = 0; suffix < 1000; suffix++) {
      xpar_asprintf(&backup[i], "%s.xpar-old-%03u", c->vol[i].path, suffix);
      if (!gen_exists(backup[i])) break;
      xpar_free(backup[i]);  backup[i] = NULL;
    }
    if (!backup[i]) {
      u32 k;
      for (k = 0; k < ns; k++) xpar_remove(stage[k].name);
      FATAL("Cannot choose a rollback name for '%s'.", c->vol[i].path);
    }
  }
  for (i = 0; i < data_n; i++) if (gen_exists(final_data[i])) {
    u32 suffix;
    for (suffix = 0; suffix < 1000; suffix++) {
      xpar_asprintf(&data_backup[i], "%s.xpar-old-%03u", final_data[i],
                    suffix);
      if (!gen_exists(data_backup[i])) break;
      xpar_free(data_backup[i]); data_backup[i] = NULL;
    }
    if (!data_backup[i])
      FATAL("Cannot choose a rollback name for '%s'.", final_data[i]);
  }

  for (i = 0; i < c->vol_count; i++) {
    if (xpar_rename(c->vol[i].path, backup[i]) != 0) {
      saved = xpar_errno();  break;
    }
    moved++;
  }
  if (moved != c->vol_count) goto rollback;
  for (i = 0; i < data_n; i++) if (data_backup[i]) {
    if (xpar_rename(final_data[i], data_backup[i]) != 0) {
      saved = xpar_errno(); goto rollback;
    }
    data_moved[i] = true;
  }
  if (xpar_fsync_dir(final_base) != 0) {
    saved = xpar_errno();  goto rollback;
  }

  /*  As elsewhere, make the index visible last.  */
  for (i = 0; i < nf; i++) {
    u32 k = i + 1 < nf ? i + 1 : 0;
    if (k == 0) {
      u32 d;
      for (d = 0; d < data_n; d++) {
        if (gen_exists(final_data[d]) && xpar_remove(final_data[d]) != 0) {
          saved = xpar_errno(); goto rollback;
        }
        if (xpar_rename(stage_data[d], final_data[d]) != 0) {
          saved = xpar_errno(); goto rollback;
        }
        data_published[d] = true;
        if (o->labels) {
          if (gen_exists(final_label[d]) && xpar_remove(final_label[d]) != 0) {
            saved = xpar_errno(); goto rollback;
          }
          if (xpar_rename(stage_label[d], final_label[d]) != 0) {
            saved = xpar_errno(); goto rollback;
          }
          label_published[d] = true;
        }
      }
    }
    if (gen_exists(final[k].name) && xpar_remove(final[k].name) != 0) {
      saved = xpar_errno();  goto rollback;
    }
    if (xpar_rename(stage[k].name, final[k].name) != 0) {
      saved = xpar_errno();  goto rollback;
    }
    published[k] = true;
  }
  if (xpar_fsync_dir(final_base) != 0) {
    saved = xpar_errno();  goto rollback;
  }

  for (i = 0; i < c->vol_count; i++) {
    if (xpar_remove(backup[i]) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot remove rollback volume '%s': "
                   "%s\n", backup[i], xpar_strerror(xpar_errno()));
  }
  for (i = 0; i < data_n; i++) if (data_backup[i]) {
    if (xpar_remove(data_backup[i]) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot remove rollback volume '%s': "
                   "%s\n", data_backup[i], xpar_strerror(xpar_errno()));
  }
  /*  Generation data volumes are bare and therefore absent from c->vol.
      Once the new generation-0 index is durable, remove every old bare
      volume that is not one of the newly published names.  */
  for (i = 0; i < c->gen_count; i++) {
    xpar_layt l;
    u32 k;
    if (!c->gen[i].layt_body ||
        xpar_layt_read(c->gen[i].layt_body, c->gen[i].layt_len, &l) !=
          XPAR_OK) continue;
    for (k = 0; k < l.count; k++) if (l.vol[k].kind == XPAR_VOL_DATA) {
      char * old = xpar_path_join(c->dir, l.vol[k].name);
      u32 d;
      for (d = 0; d < data_n; d++)
        if (!xpar_strcmp(old, final_data[d])) break;
      if (d == data_n && gen_exists(old)) xpar_remove(old);
      xpar_free(old);
    }
    xpar_layt_free(&l);
  }
  if (xpar_fsync_dir(final_base) != 0 && o->verbose)
    xpar_fprintf(xpar_stderr, "xpar: cannot sync the directory after "
                 "removing rollback volumes: %s\n",
                 xpar_strerror(xpar_errno()));
  goto done;

rollback:
  for (i = 0; i < nf; i++)
    if (published[i]) xpar_remove(final[i].name);
  for (i = 0; i < ns; i++) xpar_remove(stage[i].name);
  for (i = 0; i < data_n; i++) {
    if (data_published[i]) xpar_remove(final_data[i]);
    else xpar_remove(stage_data[i]);
    if (o->labels) {
      if (label_published[i]) xpar_remove(final_label[i]);
      else xpar_remove(stage_label[i]);
    }
  }
  for (i = data_n; i-- > 0;)
    if (data_moved[i] &&
        xpar_rename(data_backup[i], final_data[i]) != 0)
      xpar_fprintf(xpar_stderr, "xpar: rollback volume remains at '%s'.\n",
                   data_backup[i]);
  while (moved) {
    moved--;
    if (xpar_rename(backup[moved], c->vol[moved].path) != 0)
      xpar_fprintf(xpar_stderr, "xpar: rollback volume remains at '%s'.\n",
                   backup[moved]);
  }
  xpar_fsync_dir(final_base);
  FATAL_IO("Cannot publish the consolidated set: %s.",
           xpar_strerror(saved));

done:
  for (i = 0; i < c->vol_count; i++) xpar_free(backup[i]);
  for (i = 0; i < data_n; i++) {
    xpar_free(stage_data[i]); xpar_free(final_data[i]);
    xpar_free(data_backup[i]);
    if (o->labels) {
      xpar_free(stage_label[i]); xpar_free(final_label[i]);
    }
  }
  xpar_free(backup);  xpar_free(published);
  xpar_free(stage_data); xpar_free(final_data); xpar_free(data_backup);
  xpar_free(data_published); xpar_free(data_moved);
  xpar_free(stage_label); xpar_free(final_label); xpar_free(label_published);
  gen_volumes_free(stage, ns);  gen_volumes_free(final, nf);
}

static void gen_layt_build(xpar_layt * l, const gen_vol * v, u32 n) {
  u32 i;
  l->this_volume = XPAR_VOL_STANDALONE;
  l->count = n;
  l->vol = (xpar_vol *) xpar_calloc(n, sizeof(xpar_vol));
  for (i = 0; i < n; i++) {
    char * dir;  char * name;
    l->vol[i].kind = v[i].is_index ? (u8) XPAR_VOL_INDEX
                                   : (u8) XPAR_VOL_RECOVERY;
    l->vol[i].recovery_first = (u32) v[i].first;
    l->vol[i].byte_length    = v[i].count;
    gen_split_path(v[i].name, &dir, &name);
    l->vol[i].name = name;
    xpar_free(dir);
  }
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
  for (i = 0; i < m->count; i++) {
    xpar_entry * e = &m->entry[i];
    xpar_blake3_t h;
    if (!owned[i] || e->entry_type == XPAR_ENTRY_HARDLINK) continue;
    xpar_blake3_init_keyed(&h, key->k_file);
    if (e->entry_type == XPAR_ENTRY_SYMLINK) {
      xpar_blake3_update(&h, e->extra, e->extra_len);
    } else if (e->entry_type == XPAR_ENTRY_REGULAR) {
      xpar_file * f = xpar_open(m->source[i], XPAR_O_RDONLY);
      u8 buf[16384];
      if (!f) FATAL_PERROR(m->source[i]);
      for (;;) {
        sz n = xpar_read(f, buf, sizeof buf);
        if (n) xpar_blake3_update(&h, buf, n);
        if (n < sizeof buf) {
          if (xpar_error(f)) FATAL_IO("Reading '%s' failed.", m->source[i]);
          if (xpar_eof(f) || !n) break;
        }
      }
      xpar_xclose(f);
      xpar_secure_zero(buf, sizeof buf);
    }
    xpar_blake3_final(&h, e->content_hash, 32);
    xpar_secure_zero(&h, sizeof h);
  }
  for (i = 0; i < m->count; i++) if (owned[i]) {
    xpar_entry * e = &m->entry[i];
    if (e->entry_type == XPAR_ENTRY_HARDLINK) {
      i64 t = xpar_link_target(m, &ix, i);
      FATAL_UNLESS("Hard-link entry '%.*s' has no canonical target.",
                   t >= 0, (int) e->name_len, e->name);
      xpar_memcpy(e->content_hash, m->entry[t].content_hash, 32);
    }
  }
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
  for (i = 0; i < m->count; i++) {
    if (owned[i]) {
      xpar_buf e;
      xpar_buf_init(&e);
      xpar_entry_write(&e, &m->entry[i], zero, NULL, w);
      xpar_set_id_update(&ctx, e.data + XPAR_PKT_HDR, e.len - XPAR_PKT_HDR);
      xpar_buf_free(&e);
    } else {
      xpar_set_id_update(&ctx, inh_body[i], inh_len[i]);
    }
  }
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
  for (i = 0; i < m->count; i++)
    if (owned[i]) xpar_entry_write(out, &m->entry[i], set_id, key, w);
  if (m->posix_count && !w->reproducible)
    xpar_posx_write_all(out, m->posix, m->posix_count, set_id, key);
  if (!auth_only && sd->data_slice_count && t->slice_crc)
    xpar_slcr_write_all(out, t->slice_crc, sd->data_slice_count, set_id,
                        key);
  if (auth) xpar_auth_write(out, auth, set_id, key);
  xpar_layt_write(out, &l, set_id, key);
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
  u32 nvol, data_n = 0, i, j;
  u64 crit_bytes;
  u8 * rec_scratch = NULL;
  xpar_key key;
  u8 master[XPAR_BLAKE3_KEY_LEN];
  xpar_auth auth;
  bool keyed = false;
  const xpar_key * kp = NULL;
  u8 tag_len = (u8) (o->slice_tag < 0 ? 8 : o->slice_tag);

  xpar_memset(&key, 0, sizeof key);
  xpar_memset(master, 0, sizeof master);
  xpar_memset(&auth, 0, sizeof auth);
  if (o->auth_key) {
    xpar_keyfile_status ks = xpar_keyfile_load(o->auth_key, &key, master);
    if (ks == XPAR_KEYFILE_OPEN) FATAL_PERROR(o->auth_key);
    if (ks == XPAR_KEYFILE_EMPTY)
      FATAL_CODE(XPAR_EXIT_AUTH, "The key file is empty.");
    if (ks != XPAR_KEYFILE_OK)
      FATAL_CODE(XPAR_EXIT_AUTH, "Reading key file '%s' failed.",
                 o->auth_key);
    keyed = true; kp = &key; tag_len = 16;
    auth.kdf_id = 0; auth.slice_tag_keyed = 1;
    auth.packet_tag_keyed = 1;
    auth.unkeyed_retained = !rq->auth_only;
    xpar_key_check(auth.key_check, master);
    if (rq->auth_only) gen_auth_only_hashes(m, rq->owned, kp);
    for (i = 0; i < m->count; i++) if (rq->owned[i])
      xpar_file_id(&m->entry[i], key.k_file, m->entry[i].file_id);
  }
  FATAL_UNLESS("--align=1k needs slice tags; choose --slice-tag=8 or 16.",
               o->align != XPAR_ALIGN_1K || tag_len != 0);

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
    data_name = (char **) xpar_calloc(data_n, sizeof(char *));
    layout_data_name = (char **) xpar_calloc(data_n, sizeof(char *));
    label_name = (char **) xpar_calloc(data_n, sizeof(char *));
    for (i = 0; i < data_n; i++) {
      data_name[i] = gen_name_data(rq->base, rq->generation, i, width);
      layout_data_name[i] = gen_name_data(
        rq->layout_base ? rq->layout_base : rq->base,
        rq->generation, i, width);
      if (o->labels)
        xpar_asprintf(&label_name[i], "%s" XPAR_EXT, data_name[i]);
    }
  }
  if (!o->force)
    for (i = 0; i < nvol; i++)
      if (gen_exists(vol[i].name))
        FATAL("'%s' exists; -f overwrites it. Nothing was written.",
              vol[i].name);
  if (!o->force)
    for (i = 0; i < data_n; i++) {
      if (gen_exists(data_name[i]))
        FATAL("'%s' exists; -f overwrites it. Nothing was written.",
              data_name[i]);
      if (label_name[i] && gen_exists(label_name[i]))
        FATAL("'%s' exists; -f overwrites it. Nothing was written.",
              label_name[i]);
    }

  xpar_progress_init(&prog, o->progress != XPAR_PROGRESS_OFF && !o->quiet,
                     rq->plan.geom.slice_count * rq->plan.geom.slice_size,
                     "Encoding");
  gen_encode(m, &rq->plan, tag_len, o->memory, rq->base, kp, NULL, NULL,
             &t, &prog);
  if (t.rec_spill) rec_scratch = (u8 *) xpar_alloc_raw((sz) t.rec_z);
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
  for (i = 0; i < m->count; i++)
    xpar_memcpy(sd.file_id[i], m->entry[i].file_id, XPAR_SET_ID_LEN);

  gen_set_id(&sd, m, rq->owned, rq->inh_body, rq->inh_len, &w, kp,
             rq->set_id);

  if (rq->layout_base) {
    u32 layout_n;
    layout_vol = gen_volumes(o, rq->plan.recovery, rq->layout_base,
                             rq->generation, &layout_n);
    xpar_assert(layout_n == nvol);
    gen_layt_build(&layt, layout_vol, layout_n);
  } else {
    gen_layt_build(&layt, vol, nvol);
  }
  for (i = 0; i < layt.count; i++)
    layt.vol[i].vflags = o->armour != XPAR_ARMOUR_NONE ||
                         o->layout == XPAR_LAYOUT_ARMOURED;
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    u64 per = rq->plan.geom.slice_count
                ? xpar_ceil_div(rq->plan.geom.slice_count, data_n) : 0;
    u64 slice = 0;
    layt.vol = (xpar_vol *) xpar_realloc(layt.vol,
                   (sz) (layt.count + data_n) * sizeof(xpar_vol));
    for (i = 0; i < data_n; i++) {
      char * dir, * name;
      u64 count = MIN(per, rq->plan.geom.slice_count - slice);
      u64 off = slice * rq->plan.geom.slice_size;
      u64 len = MIN(count * rq->plan.geom.slice_size,
                    m->stream_length - off);
      xpar_memset(&layt.vol[layt.count], 0, sizeof(xpar_vol));
      layt.vol[layt.count].kind = XPAR_VOL_DATA;
      /*  LAYT tiles this generation's local [0,L) stream; SETD.stream_base
         places that stream in the chain-wide address space.  */
      layt.vol[layt.count].stream_offset = off;
      layt.vol[layt.count].byte_length = len;
      layt.vol[layt.count].vol_tag = gen_stream_tag(m, off, len);
      gen_split_path(layout_data_name[i], &dir, &name);
      layt.vol[layt.count].name = name;
      xpar_free(dir);
      layt.count++; slice += count;
    }
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

  if (o->layout == XPAR_LAYOUT_ARMOURED) {
    xpar_armour_params ap;
    xpar_armour * a;
    gen_armsink sink;
    xpar_buf head, tail, crtr;
    xpar_volh vh;
    gen_src src;
    u8 * buf;
    char * tmp;
    xpar_file * f;
    u64 stream_packet, stream_at, plain_len, at, left, e;
    const char * why;

    gen_armour_params(o, &ap);
    why = xpar_armour_check(&ap);
    if (why) FATAL("Armour parameters are not servable: %s", why);
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
    gen_strm_header(&head, m->stream_length, rq->set_id, kp);
    stream_at = head.len;

    xpar_buf_init(&tail);
    if (t.slice_tag)
      xpar_sltg_write_all(&tail, t.slice_tag, rq->plan.geom.slice_count,
                          tag_len, rq->set_id, kp);
    if (t.cell_crc)
      xpar_slcl_write_all(&tail, t.cell_crc, rq->plan.geom.slice_count,
                          rq->plan.geom.cell_bytes,
                          rq->plan.geom.cells_per_slice, rq->set_id, kp);
    xpar_buf_init(&crtr);
    xpar_crtr_write(&crtr, "xpar " PACKAGE_VERSION, rq->set_id, kp, &w);
    stream_packet = xpar_align_up(XPAR_PKT_HDR + 16 + m->stream_length,
                                  XPAR_PKT_ALIGN);
    plain_len = head.len - (XPAR_PKT_HDR + 16) + stream_packet + tail.len +
                rq->plan.recovery *
                  (XPAR_PKT_HDR + 16 + rq->plan.geom.slice_size) + crtr.len;

    f = gen_stage_open(vol[0].name, &tmp);
    gen_write_arm_prologue(f, &ap, plain_len,
                           xpar_armour_size(a, plain_len), stream_at,
                           m->stream_length);
    gen_as_init(&sink, a, f);
    gen_as_put(&sink, head.data, head.len);
    buf = (u8 *) xpar_alloc_raw(1u << 16);
    gen_src_init(&src, m, m->stream_base + m->stream_length);
    at = m->stream_base;  left = m->stream_length;
    while (left) {
      u64 take = MIN(left, (u64) 1 << 16);
      gen_src_read(&src, at, take, buf);
      gen_as_put(&sink, buf, take);
      at += take;  left -= take;
    }
    gen_src_free(&src);
    {
      u8 zero[XPAR_PKT_ALIGN] = { 0 };
      u64 pad = stream_packet - (XPAR_PKT_HDR + 16 + m->stream_length);
      if (pad) gen_as_put(&sink, zero, pad);
    }
    gen_as_put(&sink, tail.data, tail.len);
    for (e = 0; e < rq->plan.recovery; e++) {
      xpar_buf pkt;
      const u8 * rec = gen_rec_get(&t, e, rec_scratch);
      xpar_buf_init(&pkt);
      xpar_rcvs_write(&pkt, e, rec, (sz) rq->plan.geom.slice_size,
                      rq->set_id, kp);
      gen_as_put(&sink, pkt.data, pkt.len);
      xpar_buf_free(&pkt);
    }
    gen_as_put(&sink, crtr.data, crtr.len);
    gen_as_flush(&sink);
    gen_as_free(&sink);
    xpar_free(buf);
    if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
      FATAL_IO("Cannot flush temporary armoured archive '%s'.", tmp);
    xpar_xclose(f);
    gen_publish_whole(tmp, vol[0].name, o->force);
    xpar_buf_free(&head);  xpar_buf_free(&tail);  xpar_buf_free(&crtr);
    xpar_armour_free(a);
  } else {
  /*  Recovery volumes are published before the index. A reader discovers a
      generation through its index, so interruption leaves only unreferenced
      recovery volumes rather than a visible index naming absent volumes.  */
  for (j = 0; j < nvol; j++) {
    xpar_buf out, group;
    xpar_volh vh;
    i = j + 1 < nvol ? j + 1 : 0;
    u64 payload = vol[i].count * rq->plan.geom.slice_size;
    bool carry = vol[i].is_index ||
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

    if (vol[i].is_index || i == 1) {
      if (t.slice_tag)
        xpar_sltg_write_all(&out, t.slice_tag, rq->plan.geom.slice_count,
                            tag_len, rq->set_id, kp);
      if (t.cell_crc)
        xpar_slcl_write_all(&out, t.cell_crc, rq->plan.geom.slice_count,
                            rq->plan.geom.cell_bytes,
                            rq->plan.geom.cells_per_slice, rq->set_id, kp);
    }
    {
      u32 k;
      if (vol[i].count) {
        char * tmp;
        xpar_file * f = gen_stage_open(vol[i].name, &tmp);
        xpar_xwrite(f, out.data, out.len);
        xpar_buf_free(&out);
        for (k = 0; k < vol[i].count; k++) {
          xpar_buf pkt;
          u64 e = vol[i].first + k;
          const u8 * rec = gen_rec_get(&t, e, rec_scratch);
          xpar_buf_init(&pkt);
          xpar_rcvs_write(&pkt, e, rec, (sz) rq->plan.geom.slice_size,
                          rq->set_id, kp);
          xpar_xwrite(f, pkt.data, pkt.len);
          xpar_buf_free(&pkt);
        }
        {
          xpar_buf tail;
          xpar_buf_init(&tail);
          xpar_crtr_write(&tail, "xpar " PACKAGE_VERSION, rq->set_id, kp,
                          &w);
          xpar_xwrite(f, tail.data, tail.len);
          xpar_buf_free(&tail);
        }
        if (xpar_flush(f) != 0 || xpar_fsync(f) != 0)
          FATAL_IO("Cannot flush temporary volume '%s'.", tmp);
        xpar_xclose(f);
        gen_publish_whole(tmp, vol[i].name, o->force);
      } else {
        xpar_crtr_write(&out, "xpar " PACKAGE_VERSION, rq->set_id, kp, &w);
        gen_write_whole(vol[i].name, out.data, out.len, o->force);
        xpar_buf_free(&out);
      }
    }
    if (!rq->quiet && o->verbose)
      xpar_fprintf(xpar_stderr, "xpar: wrote %s\n", vol[i].name);
  }
  if (o->layout == XPAR_LAYOUT_SPLIT) {
    u32 data_first = layt.count - data_n;
    for (i = 0; i < data_n; i++) {
      const xpar_vol * dv = &layt.vol[data_first + i];
      gen_write_data_range(m, data_name[i], dv->stream_offset,
                           dv->byte_length, o->force);
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
      gen_write_whole(label_name[i], out.data, out.len, o->force);
      xpar_buf_free(&out);
    }
    }
  }
  }
  rq->volumes = nvol;
  rq->index_path = xpar_strdup(vol[0].name);

  xpar_layt_free(&layt);
  if (layout_vol) gen_volumes_free(layout_vol, nvol);
  gen_volumes_free(vol, nvol);
  gen_tables_free(&t);
  for (i = 0; i < data_n; i++) {
    xpar_free(data_name[i]); xpar_free(layout_data_name[i]);
    xpar_free(label_name[i]);
  }
  xpar_free(data_name); xpar_free(layout_data_name); xpar_free(label_name);
  xpar_free(rec_scratch);
  xpar_free(sd.file_id);
  xpar_key_forget(&key, master);
}

/*  Comparing an entry against the disk.  */

static void gen_entry_copy(xpar_entry * d, const xpar_entry * s) {
  *d = *s;
  d->name    = s->name_len ? (char *) xpar_malloc(s->name_len) : NULL;
  if (s->name_len) xpar_memcpy(d->name, s->name, s->name_len);
  d->extra   = s->extra_len ? (u8 *) xpar_malloc(s->extra_len) : NULL;
  if (s->extra_len) xpar_memcpy(d->extra, s->extra, s->extra_len);
  d->extents = s->extent_count
                 ? (xpar_extent *) xpar_malloc((sz) s->extent_count *
                                               sizeof(xpar_extent)) : NULL;
  if (s->extent_count)
    xpar_memcpy(d->extents, s->extents,
                (sz) s->extent_count * sizeof(xpar_extent));
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
  if (o->base_dir) {
    p = xpar_path_join(o->base_dir, name);
    xpar_free(name);
    return p;
  }
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
    if (st.mode & 0111u) e->attrs |= XPAR_ATTR_EXEC;
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
    buf = (u8 *) xpar_alloc_raw(1u << 16);
    if (auth_only) xpar_blake3_init_keyed(&h, key->k_file);
    else           xpar_blake3_init(&h);
    xpar_blake3_init(&prefix);
    got = xpar_xread(f, buf, 16384);
    if (got) {
      xpar_blake3_update(&h, buf, got);
      xpar_blake3_update(&prefix, buf, got);
    }
    total = got;
    xpar_blake3_final(&prefix, e->prefix_hash, 16);
    while ((got = xpar_xread(f, buf, 1u << 16)) > 0) {
      xpar_blake3_update(&h, buf, got);  total += got;
    }
    xpar_blake3_final(&h, e->content_hash, 32);
    xpar_free(buf);
    xpar_xclose(f);
    e->entry_type = XPAR_ENTRY_REGULAR;
    e->length     = total;
  } else {
    return false;
  }
  xpar_file_id(e, key ? key->k_file : NULL, e->file_id);
  return true;
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
  gen_chunk_pack * c = (gen_chunk_pack *) user;
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
    if (!xpar_chunk_index_put(c->ix, hash, len, off)) {
      c->full = true;
      return false;
    }
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
                 "The chunk fingerprint index exceeded --dedup-memory; "
                 "raise it or --dedup-chunk.");
    FATAL("Cannot read '%s' while chunking it.",
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
    g->owned = (bool *) xpar_realloc(g->owned, cap * sizeof(bool));
    g->reuse = (bool *) xpar_realloc(g->reuse, cap * sizeof(bool));
    g->body  = (const u8 **) xpar_realloc(g->body, cap * sizeof(const u8 *));
    g->blen  = (sz *) xpar_realloc(g->blen, cap * sizeof(sz));
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

/*  Preserve inherited extents; append new bytes at the monotone high-water
    mark in manifest order.  */
static void gen_repack(gen_merge * g, const xpar_options * o,
                       const char * cache_path, const u8 * ancestor_id,
                       u64 base, xpar_chunk_index * cache_out) {
  u64 H = base;
  u32 i, j;
  xpar_chunk_index chunks;
  xpar_vset * ancestor = NULL;
  bool have_chunks = false;
  g->m.stream_base   = base;
  g->m.entry_bytes   = 0;
  g->m.shared_bytes  = 0;
  g->m.alias_extents = 0;
  xpar_memset(&chunks, 0, sizeof chunks);
  if (o->dedup == XPAR_DEDUP_CHUNK) {
    u64 budget = o->dedup_memory ? o->dedup_memory :
                 (o->memory ? o->memory : gen_default_budget()) / 4;
    if (!xpar_chunk_index_init(&chunks, budget))
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "--dedup-memory=%llu is too small for a chunk index.",
                 (unsigned long long) budget);
    have_chunks = true;
    if (o->dedup_scope == XPAR_SCOPE_CHAIN) {
      u64 average = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
      bool available = cache_path && ancestor_id &&
        xpar_chunk_cache_load(cache_path, ancestor_id, average, &chunks);
      if (available)
        for (i = 0; i < chunks.capacity; i++) {
          const xpar_chunk_slot * s = &chunks.slot[i];
          if (s->length &&
              (s->stream_offset >= base || s->length > base -
                                                    s->stream_offset)) {
            available = false;
            break;
          }
        }
      if (!available) {
        xpar_chunk_index_free(&chunks);
        if (!o->quiet)
          xpar_fprintf(xpar_stderr, "xpar: chain dedup is unavailable "
                       "because its verified cache is absent or stale; "
                       "using generation scope.\n");
        xpar_assert(xpar_chunk_index_init(&chunks, budget));
      } else ancestor = xpar_vset_open(o);
    }
  }
  for (i = 0; i < g->m.count; i++) {
    xpar_entry * e = &g->m.entry[i];
    xpar_extent x;
    if (e->entry_type == XPAR_ENTRY_REGULAR) g->m.entry_bytes += e->length;
    if (!g->owned[i] || g->reuse[i]) continue;
    if (e->entry_type != XPAR_ENTRY_REGULAR || !e->length) {
      xpar_free(e->extents);  e->extents = NULL;  e->extent_count = 0;
      continue;
    }
    if (o->dedup != XPAR_DEDUP_NONE) {
      for (j = 0; j < i; j++) {
        const xpar_entry * c = &g->m.entry[j];
        if (c->entry_type != XPAR_ENTRY_REGULAR || !c->extent_count) continue;
        if (c->length != e->length) continue;
        if (xpar_memcmp(c->content_hash, e->content_hash, 32)) continue;
        xpar_free(e->extents);
        e->extent_count = c->extent_count;
        e->extents = (xpar_extent *) xpar_malloc((sz) c->extent_count *
                                                 sizeof(xpar_extent));
        xpar_memcpy(e->extents, c->extents,
                    (sz) c->extent_count * sizeof(xpar_extent));
        g->m.shared_bytes  += e->length;
        g->m.alias_extents += c->extent_count;
        break;
      }
      if (j < i) continue;
    }
    if (have_chunks && o->dedup == XPAR_DEDUP_CHUNK) {
      gen_chunk_entry(g, o, &chunks, i, &H, ancestor);
      continue;
    }
    {
      u64 q = o->align == XPAR_ALIGN_SLICE ? g->m.slice_size
            : o->align == XPAR_ALIGN_1K ? XPAR_BLAKE3_CHUNK_LEN : 0;
      u64 pad = q ? (H - base) % q : 0;
      if (pad) H += q - pad;
    }
    x.stream_offset = H;  x.length = e->length;
    xpar_free(e->extents);
    e->extents = (xpar_extent *) xpar_malloc(sizeof(xpar_extent));
    e->extents[0] = x;  e->extent_count = 1;
    H += e->length;
  }
  g->m.stream_length = H - base;
  g->m.dedup_level   = g->m.alias_extents ? (u8) o->dedup : XPAR_DEDUP_NONE;
  if (have_chunks && cache_out) {
    *cache_out = chunks;
    xpar_memset(&chunks, 0, sizeof chunks);
  }
  xpar_vset_close(ancestor);
  if (have_chunks) xpar_chunk_index_free(&chunks);
}

/*  Whole-entry references copy ancestor extents at every deduplication
    scope; the dependency already existed under the prior manifest.  */
static const xpar_entry * gen_find_content(const xpar_manifest * anc,
                                           const xpar_entry * e) {
  u32 i;
  if (e->entry_type != XPAR_ENTRY_REGULAR || !e->length) return NULL;
  for (i = 0; i < anc->count; i++) {
    const xpar_entry * a = &anc->entry[i];
    if (a->entry_type != XPAR_ENTRY_REGULAR || !a->extent_count) continue;
    if (a->length != e->length) continue;
    if (!xpar_memcmp(a->content_hash, e->content_hash, 32)) return a;
  }
  return NULL;
}

static void gen_take_extents(xpar_entry * d, const xpar_entry * s) {
  xpar_free(d->extents);
  d->extent_count = s->extent_count;
  d->extents = s->extent_count
                 ? (xpar_extent *) xpar_malloc((sz) s->extent_count *
                                               sizeof(xpar_extent)) : NULL;
  if (s->extent_count)
    xpar_memcpy(d->extents, s->extents,
                (sz) s->extent_count * sizeof(xpar_extent));
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
      if (xpar_armg_read(body, (sz) blen, &ag) != XPAR_OK) continue;
      ap.symbol_bits = ag.symbol_bits;  ap.poly = ag.poly;
      ap.n = ag.n;  ap.k = ag.k;  ap.fcr = ag.fcr;  ap.prim = ag.prim;
      ap.depth = ag.depth;
      plain = arm_extract(&ap, ag.data, ag.armoured_length, ag.plain_length,
                          &plen, rw->key);
      if (!plain) FATAL_FORMAT("An armoured critical group will not extract.");
      xpar_buf_init(&inner);
      gen_rebuild(&inner, o, plain, plen, rw, true);
      gen_armour_pack(out, o, inner.data, inner.len, rw->set_id, rw->key);
      xpar_buf_free(&inner);
      xpar_free(plain);
      continue;
    }
    if (rw->group && gen_is_critical(&hdr)) {
      if (!group_done) {
        xpar_buf_put(out, rw->group, rw->group_len);
        group_done = true;
      }
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
  for (;;) {
    const xpar_crit_pkt * best = NULL;
    u64 best_key = 0;
    u32 i;
    for (i = 0; i < c->crit.count; i++) {
      const xpar_crit_pkt * p = &c->crit.pkt[i];
      u64 key;
      if (!xpar_pkt_is(&p->hdr, type)) continue;
      if (xpar_memcmp(p->hdr.set_id, c->gen[g].set_id, XPAR_SET_ID_LEN))
        continue;
      key = p->body_len >= 8 ? xpar_rd64(p->body) : 0;
      if (!xpar_memcmp(type, XPAR_T_POSX, 4))
        key = p->body_len >= 4 ? xpar_rd32(p->body) : 0;
      if (key < want) continue;
      if (!best || key < best_key) { best = p;  best_key = key; }
    }
    if (!best) return;
    xpar_pkt_write(out, best->hdr.type, best->hdr.flags, set_id, best->body,
                   (sz) best->body_len, gen_chain_key(c));
    want = best_key + 1;
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
  for (i = 0; i < m->count; i++)
    m->source[i] = gen_entry_path(o, &m->entry[i]);
}

static bool gen_read_vset(void * ctx, u64 off, u8 * out, u64 len) {
  return xpar_vset_read((xpar_vset *) ctx, off, out, len);
}

static void gen_require_source_tables(const xpar_vset * set,
                                      const gen_tables * made,
                                      const gen_plan * plan) {
  const xpar_tags * stored = xpar_vset_tags(set);
  u32 have = xpar_vset_have_tables(set);
  u64 i;
  if ((have & XPAR_TAGS_CRC) && stored->slice_crc)
    for (i = 0; i < plan->geom.slice_count; i++)
      if (stored->slice_crc[i] != made->slice_crc[i])
        FATAL_CODE(XPAR_EXIT_REPAIRABLE,
                   "The selected generation changed while its recovery "
                   "data was being prepared (slice %llu CRC differs). "
                   "Nothing was written.", (unsigned long long) i);
  if ((have & XPAR_TAGS_TAG) && stored->slice_tag && made->slice_tag) {
    if (stored->tag_len != made->tag_len)
      FATAL_FORMAT("The selected generation's slice-tag table has the "
                   "wrong tag length.");
    for (i = 0; i < plan->geom.slice_count; i++)
      if (!xpar_blake3_tag_equal(stored->slice_tag + i * stored->tag_len,
                                made->slice_tag + i * made->tag_len,
                                made->tag_len))
        FATAL_CODE(XPAR_EXIT_REPAIRABLE,
                   "The selected generation changed while its recovery "
                   "data was being prepared (slice %llu tag differs). "
                   "Nothing was written.", (unsigned long long) i);
  }
}

typedef struct {
  char * stage;
  const char * final;
  bool replace;
} gen_addrec_file;

static void gen_addrec_discard(gen_addrec_file * files, u32 count) {
  u32 i;
  for (i = 0; i < count; i++) {
    if (files[i].stage) xpar_remove(files[i].stage);
    xpar_free(files[i].stage);
  }
  xpar_free(files);
}

static void gen_addrec_publish(gen_addrec_file * files, u32 count) {
  u32 i;
  for (i = 0; i < count; i++) {
    gen_publish_whole(files[i].stage, files[i].final, files[i].replace);
    files[i].stage = NULL;
  }
  xpar_free(files);
}

int xpar_op_addrecovery(const xpar_options * o) {
  xpar_chain c;
  xpar_manifest m;
  gen_plan p;
  gen_tables t;
  u32 * owner = NULL;
  u32 g, i, j, nvol, base_vol;
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

  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "addrecovery");
  g = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  xpar_gchain_genref(&c, g, &verify_ref, verify_id);
  have = c.gen[g].recovery_top;
  axis = xpar_setd_recovery_limit(&c.gen[g].sd);

  if (!c.gen[g].sd.data_slice_count)
    FATAL("Generation %u is stream-empty, so it has nothing to protect.",
          c.gen[g].sd.generation);
  if (!c.gen[g].layt_body)
    FATAL_FORMAT("Generation %u carries no volume layout.",
                 c.gen[g].sd.generation);
  if (xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &old) != XPAR_OK)
    FATAL_FORMAT("Generation %u's volume layout is malformed.",
                 c.gen[g].sd.generation);

  want = gen_resolve_r(&o->recovery, c.gen[g].sd.data_slice_count,
                       c.gen[g].sd.slice_size);
  if (!want)
    FATAL("addrecovery needs --recovery=SPEC, which names the total this "
          "generation should end up with; it has %llu now.",
          (unsigned long long) have);
  if (want <= have) {
    xpar_fprintf(xpar_stderr, "xpar: generation %u already has %llu "
                 "recovery slice%s; nothing to do.\n",
                 c.gen[g].sd.generation, (unsigned long long) have,
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
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "Generation %u was encoded with the FFT codec on a recovery "
                 "axis of %llu slices, and %llu is past it. Growing the axis "
                 "across a power-of-two boundary changes every recovery slice "
                 "already on disk, so the old and the new ones would no "
                 "longer decode together; xpar will not write them. Re-encode "
                 "with `xpar consolidate --max-recovery=%llu`, or create the "
                 "set with --codec=matrix, which can be topped up without "
                 "limit.",
                 c.gen[g].sd.generation, (unsigned long long) axis,
                 (unsigned long long) want, (unsigned long long) want);
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "Generation %u's recovery axis holds %llu slices and %llu was "
               "asked for; exponents must stay inside the axis.",
               c.gen[g].sd.generation, (unsigned long long) axis,
               (unsigned long long) want);
  }
  if (!xpar_codec_supports_axis(c.gen[g].sd.codec,
                                c.gen[g].sd.field_log2,
                                c.gen[g].sd.data_slice_count, want,
                                c.gen[g].sd.recovery_axis_log2))
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "The %s codec cannot express S=%llu with R=%llu over "
               "GF(2^%u).", gen_codec_name(c.gen[g].sd.codec),
               (unsigned long long) c.gen[g].sd.data_slice_count,
               (unsigned long long) want, c.gen[g].sd.field_log2);

  /*  Strongly verify the stored generation stream before encoding new
      recovery.  */
  source_set = xpar_vset_open(o);
  source_rc = xpar_vset_check(source_set, o, NULL);
  if (source_rc != XPAR_EXIT_OK)
    FATAL_CODE(source_rc,
               "Generation %u's protected stream is not clean; refusing to "
               "derive new recovery data from it. Nothing was written.",
               c.gen[g].sd.generation);

  gen_manifest_on_disk(&c, g, o, &m, &owner);
  xpar_memset(&p, 0, sizeof p);
  if (!xpar_geom_from_setd(&c.gen[g].sd, &p.geom))
    FATAL_FORMAT("Generation %u's geometry is malformed.",
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
  if (t.rec_spill) rec_scratch = (u8 *) xpar_alloc_raw((sz) t.rec_z);

  /*  Re-encoding must reproduce every existing recovery exponent exactly.  */
  for (i = 0; i < c.crit.count; i++) {
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
                 "internal: re-encoding at R=%llu changed the bytes of "
                 "recovery slice %llu, which the recovery axis promises it "
                 "cannot. Nothing was written.",
                 (unsigned long long) want, (unsigned long long) r.exponent);
  }

  if (c.gen[g].sd.layout == XPAR_LAYOUT_ARMOURED) {
    const xpar_chain_vol * source = NULL;
    xpar_arm_prologue pr;
    xpar_armour_params ap;
    xpar_buf head, group;
    xpar_volh vh;
    xpar_wropt w;
    char * arm_stage;
    for (i = 0; i < c.vol_count; i++)
      if (c.vol[i].gen == g && c.vol[i].armoured_file) {
        source = &c.vol[i];
        break;
      }
    if (!source || !xpar_garm_prologue(source->data, source->len, &pr, NULL))
      FATAL_FORMAT("Generation %u's armoured archive is unavailable.",
                   c.gen[g].sd.generation);
    arm_params_of(&pr, &ap);
    xpar_buf_init(&head);
    xpar_memset(&vh, 0, sizeof vh);
    vh.volume_index = XPAR_VOL_STANDALONE;
    vh.volume_kind = XPAR_VOL_INDEX;
    xpar_volh_write(&head, &vh, c.gen[g].set_id, gen_chain_key(&c));
    xpar_buf_init(&group);
    xpar_setd_write(&group, &c.gen[g].sd, c.gen[g].set_id,
                    gen_chain_key(&c));
    for (j = 0; j < c.gen[g].sd.file_count; j++) {
      const xpar_crit_pkt * q = gen_owned_file(&c, g,
                                               c.gen[g].sd.file_id[j]);
      if (q) xpar_pkt_write(&group, XPAR_T_FILE, q->hdr.flags,
                            c.gen[g].set_id, q->body, (sz) q->body_len,
                            gen_chain_key(&c));
    }
    gen_emit_stored(&group, &c, g, XPAR_T_POSX, c.gen[g].set_id);
    gen_emit_stored(&group, &c, g, XPAR_T_SLCR, c.gen[g].set_id);
    gen_emit_stored(&group, &c, g, XPAR_T_AUTH, c.gen[g].set_id);
    old.this_volume = XPAR_VOL_STANDALONE;
    xpar_layt_write(&group, &old, c.gen[g].set_id, gen_chain_key(&c));
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
                   "xpar: generation %u now carries %llu recovery slice%s "
                   "inside its armoured archive (%llu added).\n",
                   c.gen[g].sd.generation, (unsigned long long) want,
                   PLURAL(want), (unsigned long long) (want - have));
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
  {
    u64 left = want - have, first = have, step = 1;
    u32 n = old.count;
    layt.this_volume = XPAR_VOL_STANDALONE;
    layt.count = n;
    layt.vol = (xpar_vol *) xpar_calloc(n + 64, sizeof(xpar_vol));
    for (i = 0; i < n; i++) {
      layt.vol[i] = old.vol[i];
      layt.vol[i].name = xpar_strdup(old.vol[i].name);
    }
    base_vol = n;
    nvol = 0;
    vol = NULL;
    while (left) {
      u64 take = MIN(step, left);
      vol = (gen_vol *) xpar_realloc(vol, (sz) (nvol + 1) * sizeof(gen_vol));
      xpar_memset(&vol[nvol], 0, sizeof(gen_vol));
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
    {
      u64 max_first = 0, max_count = 1;
      int wf, wc;
      for (i = 0; i < layt.count; i++) {
        if (layt.vol[i].kind != XPAR_VOL_RECOVERY) continue;
        if (layt.vol[i].recovery_first > max_first)
          max_first = layt.vol[i].recovery_first;
        if (layt.vol[i].byte_length > max_count)
          max_count = layt.vol[i].byte_length;
      }
      gen_recovery_widths(max_first, max_count, &wf, &wc);
      for (i = 0; i < nvol; i++) {
        char * nd, * nn;
        vol[i].name = gen_name_recovery(c.base ? c.base : o->set,
                                        c.gen[g].sd.generation, vol[i].first,
                                        vol[i].count, wf, wc);
        gen_split_path(vol[i].name, &nd, &nn);
        layt.vol[base_vol + i].name = nn;
        xpar_free(nd);
      }
    }
  }

  if (!o->force)
    for (i = 0; i < nvol; i++)
      if (gen_exists(vol[i].name))
        FATAL("'%s' exists; -f overwrites it. Nothing was written.",
              vol[i].name);

  /*  Strictly parse all staged files, then publish new volumes before their
      referring index.  */
  staged_cap = nvol + c.vol_count;
  staged = (gen_addrec_file *) xpar_calloc(staged_cap ? staged_cap : 1,
                                           sizeof *staged);
  for (i = 0; i < nvol; i++) {
    xpar_buf out;
    xpar_volh vh;
    xpar_buf_init(&out);
    xpar_memset(&vh, 0, sizeof vh);
    vh.volume_index = base_vol + i;
    vh.volume_kind  = XPAR_VOL_RECOVERY;
    xpar_volh_write(&out, &vh, c.gen[g].set_id, gen_chain_key(&c));
    {
      xpar_buf group;
      xpar_buf_init(&group);
      xpar_setd_write(&group, &c.gen[g].sd, c.gen[g].set_id,
                      gen_chain_key(&c));
      for (j = 0; j < c.gen[g].sd.file_count; j++) {
        const xpar_crit_pkt * q = gen_owned_file(&c, g,
                                                 c.gen[g].sd.file_id[j]);
        if (q) xpar_pkt_write(&group, XPAR_T_FILE, q->hdr.flags,
                              c.gen[g].set_id, q->body, (sz) q->body_len,
                              gen_chain_key(&c));
      }
      gen_emit_stored(&group, &c, g, XPAR_T_POSX, c.gen[g].set_id);
      gen_emit_stored(&group, &c, g, XPAR_T_SLCR, c.gen[g].set_id);
      gen_emit_stored(&group, &c, g, XPAR_T_AUTH, c.gen[g].set_id);
      {
        xpar_layt l = layt;
        l.this_volume = base_vol + i;
        xpar_layt_write(&group, &l, c.gen[g].set_id, gen_chain_key(&c));
      }
      if (o->armour != XPAR_ARMOUR_NONE)
        gen_armour_pack(&out, o, group.data, group.len, c.gen[g].set_id,
                        gen_chain_key(&c));
      else
        xpar_buf_put(&out, group.data, group.len);
      xpar_buf_free(&group);
    }
    for (e = vol[i].first; e < vol[i].first + vol[i].count; e++) {
      const u8 * rec = gen_rec_get(&t, e, rec_scratch);
      xpar_rcvs_write(&out, e, rec, (sz) p.geom.slice_size,
                      c.gen[g].set_id, gen_chain_key(&c));
    }
    xpar_crtr_write(&out, "xpar " PACKAGE_VERSION, c.gen[g].set_id,
                    gen_chain_key(&c), NULL);
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
                 "internal: a staged addrecovery volume did not pass its "
                 "strict packet read-back. Nothing was published.");
    }
    staged_count++;
    xpar_buf_free(&out);
  }

  /*  Now every existing volume of this generation learns the new layout.
      Its recovery slices are copied byte for byte, which is what makes
      this cheap and what 9.7 promises.  */
  for (i = 0; i < c.vol_count; i++) {
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
                 "internal: a staged layout update did not pass its strict "
                 "packet read-back. Nothing was published.");
    }
    staged_count++;
    xpar_buf_free(&out);
  }

  FATAL_UNLESS("Generation %u has no index volume to verify after writing.",
               verify_path != NULL, c.gen[g].sd.generation);
  gen_addrec_publish(staged, staged_count);
  xpar_verify_written_set_at(o, verify_path, &verify_ref);

  if (!o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: generation %u now carries %llu recovery slice%s "
                 "(%llu added in %u volume%s); every existing slice is "
                 "unchanged.\n", c.gen[g].sd.generation,
                 (unsigned long long) want, PLURAL(want),
                 (unsigned long long) (want - have), nvol, PLURAL(nvol));
  gen_json_result(o, "addrecovery", c.gen[g].set_id,
                  c.gen[g].sd.generation, "ok", XPAR_EXIT_OK);

  gen_volumes_free(vol, nvol);
  xpar_layt_free(&layt);
  xpar_layt_free(&old);
  gen_tables_free(&t);
  xpar_free(rec_scratch);
  xpar_free(owner);
  xpar_manifest_free(&m);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

int xpar_op_add(const xpar_options * o) {
  xpar_chain c;
  xpar_manifest inh, fresh;
  gen_merge g;
  gen_write_req rq;
  xpar_posix_rec ** tab;
  u32 * tabn;
  u32 * owner = NULL;
  u32 head, i, ia = 0, ib = 0, caps;
  u32 added = 0, changed = 0, kept = 0, dropped = 0;
  bool warn_posix = false;
  char idbuf[XPAR_SET_ID_LEN * 2 + 1];
  char * input_cache = NULL, * output_cache = NULL, * stage_cache = NULL;
  char * stdin_stage = NULL, * stdin_final = NULL;
  xpar_chunk_index chunk_cache;

  xpar_memset(&fresh, 0, sizeof fresh);
  xpar_memset(&g, 0, sizeof g);
  xpar_memset(&chunk_cache, 0, sizeof chunk_cache);
  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "add");
  if (c.authenticated && o->auth_only && !c.auth_only)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "A generation must keep its chain's authentication mode; "
               "this chain retains public verification hashes.");
  head = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);

  for (i = head; i != XPAR_GEN_NONE; i = c.gen[i].parent)
    if (c.gen[i].parent_missing) {
      xpar_hex(idbuf, c.gen[i].sd.parent_set_id, XPAR_SET_ID_LEN);
      FATAL_FORMAT("Generation %u names parent %s, which is not here; an "
                   "incomplete chain cannot be extended.",
                   c.gen[i].sd.generation, idbuf);
    }
  if (o->align == XPAR_ALIGN_SLICE && !o->slice_size)
    FATAL("--align=slice on an existing set needs an explicit -s: the "
          "padding is inserted while the stream is built, which is before "
          "the planner would otherwise pick a slice size.");
  /*  Whole-file chain deduplication searches the effective ancestor
      manifest; only chunk deduplication needs the disk index.  */
  if (o->dedup == XPAR_DEDUP_NONE && o->verbose)
    xpar_fprintf(xpar_stderr,
                 "xpar: --dedup=none: a renamed or chmod-ed file is "
                 "re-appended and re-encoded rather than reusing the extents "
                 "the chain already holds.\n");

  xpar_gchain_manifest(&c, head, &inh, &owner);
  tab  = (xpar_posix_rec **) xpar_calloc(c.gen_count, sizeof(void *));
  tabn = (u32 *) xpar_calloc(c.gen_count, sizeof(u32));
  for (i = 0; i < c.gen_count; i++)
    tabn[i] = xpar_gchain_posix(&c, i, &tab[i]);

  if (o->path_count) {
    xpar_walk_opts wo;
    xpar_progress_t prog;
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
    wo.dedup_memory    = o->dedup_memory ? o->dedup_memory :
                         (o->memory ? o->memory : gen_default_budget()) / 4;
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
    /*  A secure spool has a random private basename. Selection is defined
        over --stdin-name, not that implementation detail, and is therefore
        applied immediately after the manifest entry is renamed.  */
    if (stdin_stage) {
      wo.exclude = NULL; wo.exclude_count = 0;
      wo.include = NULL; wo.include_count = 0;
    }
    xpar_manifest_walk(&fresh, walk_paths, walk_count, &wo);
    if (stdin_stage) {
      FATAL_UNLESS("The staged pipe did not produce one regular entry.",
                   fresh.count == 1 &&
                   fresh.entry[0].entry_type == XPAR_ENTRY_REGULAR);
      xpar_free(fresh.entry[0].name);
      fresh.entry[0].name = xpar_strdup(o->stdin_name);
      fresh.entry[0].name_len = (u32) xpar_strlen(o->stdin_name);
      {
        xpar_walk_opts select = wo;
        select.exclude = o->exclude; select.exclude_count = o->exclude_count;
        select.include = o->include; select.include_count = o->include_count;
        if (!xpar_manifest_name_selected(&select, o->stdin_name)) {
          xpar_manifest_free(&fresh);
          xpar_memset(&fresh, 0, sizeof fresh);
        }
      }
    }
    xpar_progress_init(&prog, o->progress != XPAR_PROGRESS_OFF && !o->quiet,
                       0, "Hashing");
    xpar_manifest_pack(&fresh, &wo, &prog);
    xpar_progress_end(&prog);
    if (c.auth_only) {
      bool * all = (bool *) xpar_calloc(fresh.count ? fresh.count : 1,
                                        sizeof(bool));
      for (i = 0; i < fresh.count; i++) all[i] = true;
      gen_auth_only_hashes(&fresh, all, &c.key);
      xpar_free(all);
    }
    if (c.authenticated)
      for (i = 0; i < fresh.count; i++)
        xpar_file_id(&fresh.entry[i], c.key.k_file,
                     fresh.entry[i].file_id);
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
      anc = o->dedup != XPAR_DEDUP_NONE ? gen_find_content(&inh, e) : NULL;
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
        const xpar_entry * anc;
        e = merge_append(&g);
        gen_entry_copy(e, &fresh.entry[ib]);
        if (fresh.source && fresh.source[ib])
          g.m.source[g.m.count - 1] = xpar_strdup(fresh.source[ib]);
        anc = o->dedup != XPAR_DEDUP_NONE ? gen_find_content(&inh, e) : NULL;
        if (anc) { gen_take_extents(e, anc);  g.reuse[g.m.count - 1] = true; }
        changed++;
      }
      ia++;  ib++;
      continue;
    }

    /*  Inherited and not named by <paths...>: --rescan decides whether
        the copy on disk is still the one the chain describes.  */
    {
      const xpar_entry * old = &inh.entry[ia];
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
      char * path = gen_entry_path(o, old);
      xpar_stat_t st;
      bool gone = xpar_lstat(path, &st) != 0;
      bool stale = false;

      if (gone) {
        if (!o->allow_missing)
          FATAL("'%.*s' is in the set but not on disk. Pass "
                "--allow-missing to record the deletion.",
                (int) old->name_len, old->name);
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
        if (gen_refresh(&probe, path, o, caps, &warn_posix,
                        gen_chain_key(&c), c.auth_only) &&
            xpar_memcmp(probe.content_hash, old->content_hash, 32))
          stale = true;
        xpar_entry_free(&probe);
      }

      e = merge_append(&g);
      gen_entry_copy(e, old);
      if (stale) {
        const xpar_entry * anc;
        if (!gen_refresh(e, path, o, caps, &warn_posix,
                         gen_chain_key(&c), c.auth_only))
          FATAL_IO("Cannot re-read '%s'.", path);
        g.m.source[g.m.count - 1] = xpar_strdup(path);
        anc = o->dedup != XPAR_DEDUP_NONE ? gen_find_content(&inh, e) : NULL;
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

  for (i = 0; i < g.m.count; i++) {
    xpar_entry * e = &g.m.entry[i];
    u32 j;
    if (e->entry_type != XPAR_ENTRY_HARDLINK || g.owned[i]) continue;
    for (j = 0; j < g.m.count; j++) {
      const xpar_entry * t = &g.m.entry[j];
      if (j == i || t->name_len != e->extra_len) continue;
      if (xpar_memcmp(t->name, e->extra, e->extra_len)) continue;
      if (!g.owned[j]) break;
      xpar_memcpy(e->content_hash, t->content_hash, 32);
      xpar_memcpy(e->prefix_hash, t->prefix_hash, 16);
      e->length = t->length;
      xpar_file_id(e, c.authenticated ? c.key.k_file : NULL, e->file_id);
      g.owned[i] = true;
      break;
    }
  }

  if (warn_posix)
    xpar_fprintf(xpar_stderr,
                 "xpar: an inherited entry was rescanned without being named "
                 "on the command line, so its POSX record was dropped; name "
                 "its path to keep ownership and extended attributes.\n");

  if (fresh.posix_count) {
    g.m.posix       = fresh.posix;
    g.m.posix_count = fresh.posix_count;
    g.m.posix_cap   = fresh.posix_cap;
    fresh.posix     = NULL;
    fresh.posix_count = fresh.posix_cap = 0;
  }
  for (i = 0; i < g.m.count; i++)
    if (!g.m.source[i]) g.m.source[i] = gen_entry_path(o, &g.m.entry[i]);
  if (c.base) xpar_asprintf(&input_cache, "%s.xparidx", c.base);
  gen_repack(&g, o, input_cache, c.gen[head].set_id,
             c.gen[head].sd.stream_base + c.gen[head].sd.stream_length,
             o->dedup == XPAR_DEDUP_CHUNK &&
             o->dedup_scope == XPAR_SCOPE_CHAIN ? &chunk_cache : NULL);

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
  if (!rq.base) FATAL("This set has no base name; pass --output.");
  if (chunk_cache.slot) {
    u64 average = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
    xpar_asprintf(&output_cache, "%s.xparidx", rq.base);
    stage_cache = gen_unused_path(output_cache, "xpar-cache");
    if (!stage_cache ||
        !xpar_chunk_cache_write(stage_cache, c.gen[head].set_id, average,
                                &chunk_cache)) {
      if (o->verbose)
        xpar_fprintf(xpar_stderr, "xpar: could not stage chunk cache '%s'.\n",
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
      xpar_fprintf(xpar_stderr, "xpar: could not update chunk cache '%s'.\n",
                   output_cache);
    xpar_remove(stage_cache);
  }

  xpar_hex(idbuf, rq.set_id, XPAR_SET_ID_LEN);
  if (!o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: generation %u, set %s: %u %s "
                 "(%u added, %u changed, %u inherited, %u dropped), "
                 "%llu new stream bytes, %llu recovery slice%s in %u "
                 "volume%s.\n", rq.generation, idbuf, g.m.count,
                 g.m.count == 1 ? "entry" : "entries", added,
                 changed, kept, dropped,
                 (unsigned long long) g.m.stream_length,
                 (unsigned long long) rq.plan.recovery,
                 PLURAL(rq.plan.recovery), rq.volumes - 1,
                 PLURAL(rq.volumes - 1));
  gen_json_result(o, "add", rq.set_id, rq.generation, "ok", XPAR_EXIT_OK);

  for (i = 0; i < c.gen_count; i++)
    if (tab[i]) xpar_gchain_posix_free(tab[i], tabn[i]);
  xpar_free(tab);  xpar_free(tabn);  xpar_free(owner);
  xpar_free(input_cache);  xpar_free(output_cache);  xpar_free(stage_cache);
  if (stdin_stage && xpar_remove(stdin_stage) != 0 && o->verbose)
    xpar_fprintf(xpar_stderr, "xpar: warning: could not remove spool '%s'.\n",
                 stdin_stage);
  xpar_free(stdin_stage);
  xpar_free(stdin_final);
  xpar_free(rq.index_path);
  xpar_chunk_index_free(&chunk_cache);
  merge_free(&g);
  xpar_manifest_free(&fresh);
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
  for (k = 0; k < e->extent_count; k++) {
    i64 h = xpar_gchain_gen_of(c, e->extents[k].stream_offset,
                               e->extents[k].length);
    if (h >= 0 && removed[h]) return true;
  }
  return false;
}

static u64 gen_volume_bytes(const xpar_chain * c, u32 g) {
  u64 n = 0;  u32 i;
  for (i = 0; i < c->vol_count; i++)
    if (c->vol[i].gen == g) n += c->vol[i].len;
  if (c->gen[g].layt_body) {
    xpar_layt l;
    if (xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, &l) ==
        XPAR_OK) {
      for (i = 0; i < l.count; i++) if (l.vol[i].kind == XPAR_VOL_DATA) {
        char * p = xpar_path_join(c->dir, l.vol[i].name);
        xpar_stat_t st;
        if (xpar_lstat(p, &st) == 0 && !st.is_dir) n += st.size;
        xpar_free(p);
      }
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
  for (i = 0; i < m->count; i++) if (keep[i])
    for (k = 0; k < m->entry[i].extent_count; k++) {
      xpar_extent * x = &m->entry[i].extents[k];
      i64 h = xpar_gchain_gen_of(c, x->stream_offset, x->length);
      FATAL_UNLESS("internal: a surviving extent does not belong to a "
                   "surviving generation.", h >= 0 && !removed[h]);
      x->stream_offset = base[h] +
                         (x->stream_offset - c->gen[h].sd.stream_base);
    }
}

static void gen_prune_name(xpar_vol * v, const xpar_chain * c, u32 generation,
                           u32 data_index, u32 data_count, int wf, int wc) {
  char * full = NULL, * dir, * name;
  int width;
  xpar_free(v->name);  v->name = NULL;
  if (v->kind == XPAR_VOL_INDEX) {
    full = gen_name_index(c->base, generation);
  } else if (v->kind == XPAR_VOL_RECOVERY) {
    full = gen_name_recovery(c->base, generation, v->recovery_first,
                             v->byte_length, wf, wc);
  } else {
    width = xpar_digits10(data_count ? data_count - 1 : 0);
    if (width < 2) width = 2;
    full = gen_name_data(c->base, generation, data_index, width);
  }
  gen_split_path(full, &dir, &name);
  v->name = name;
  xpar_free(dir);  xpar_free(full);
}

static bool gen_prune_layout(const xpar_chain * c, u32 g, u32 generation,
                             xpar_layt * old, xpar_layt * now) {
  u32 i, di = 0, dn = 0;
  u64 max_first = 0, max_count = 1;
  int wf, wc;
  if (!c->gen[g].layt_body ||
      xpar_layt_read(c->gen[g].layt_body, c->gen[g].layt_len, old) !=
        XPAR_OK) return false;
  xpar_memset(now, 0, sizeof *now);
  now->this_volume = old->this_volume;
  now->count = old->count;
  now->vol = (xpar_vol *) xpar_calloc(now->count, sizeof(xpar_vol));
  for (i = 0; i < old->count; i++) {
    if (old->vol[i].kind == XPAR_VOL_DATA) dn++;
    if (old->vol[i].kind != XPAR_VOL_RECOVERY) continue;
    if (old->vol[i].recovery_first > max_first)
      max_first = old->vol[i].recovery_first;
    if (old->vol[i].byte_length > max_count)
      max_count = old->vol[i].byte_length;
  }
  gen_recovery_widths(max_first, max_count, &wf, &wc);
  for (i = 0; i < old->count; i++) {
    now->vol[i] = old->vol[i];
    now->vol[i].name = NULL;
    gen_prune_name(&now->vol[i], c, generation, di, dn, wf, wc);
    if (old->vol[i].kind == XPAR_VOL_DATA) di++;
  }
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
  for (i = 0; i < m->count; i++) if (keep[i]) {
    xpar_buf e;
    xpar_buf_init(&e);
    xpar_entry_write(&e, &m->entry[i], zero, NULL, &w);
    xpar_set_id_update(&ctx, e.data + XPAR_PKT_HDR, e.len - XPAR_PKT_HDR);
    xpar_buf_free(&e);
  }
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
  for (i = 0; i < m->count; i++)
    if (keep[i] && owner[i] == g)
      xpar_entry_write(group, &m->entry[i], id, gen_chain_key(c), &w);
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
  int which;
  FATAL_UNLESS("internal: an armoured generation has no recoverable "
               "prologue.",
               xpar_garm_prologue(v->data, v->len, &pr, &which));
  arm_params_of(&pr, &ap);
  plain = arm_extract(&ap, v->data + ARM_HDR_LEN,
                      (u64) v->len - ARM_HDR_LEN, pr.plain_length, &plen,
                      gen_chain_key(c));
  FATAL_UNLESS("internal: an armoured generation has no recoverable "
               "packet stream.", plain != NULL);
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
  FATAL_UNLESS("internal: a pruned armoured generation no longer carries "
               "its stream.", stream_at != 0);
  xpar_gf_init();
  a = xpar_armour_new(&ap);
  region = (u8 *) xpar_alloc_raw((sz) xpar_armour_size(a, plain_out.len));
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
  for (u32 i = 0; i < 3; i++) xpar_buf_put(out, copy, sizeof copy);
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
  for (i = 0; i < t->count; i++)
    if (gen_path_equal(t->f[i].old_path, path)) return i;
  return -1;
}

static gen_prune_file * gen_prune_add(gen_prune_tx * t,
                                      const char * old_path) {
  gen_prune_file * f;
  i64 found = gen_prune_find(t, old_path);
  if (found >= 0) return &t->f[found];
  if (t->count == t->cap) {
    t->cap = t->cap ? t->cap * 2 : 16;
    t->f = (gen_prune_file *) xpar_realloc(
      t->f, (sz) t->cap * sizeof(gen_prune_file));
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
  FATAL_UNLESS("internal: a pruned volume was given two replacements.",
               f->new_path == NULL);
  f->new_path = xpar_strdup(new_path);
  f->stage = stage;  f->move = move;  f->index = index;  f->order = order;
}

static void gen_prune_tx_free(gen_prune_tx * t) {
  u32 i;
  for (i = 0; i < t->count; i++) {
    xpar_free(t->f[i].old_path);  xpar_free(t->f[i].new_path);
    xpar_free(t->f[i].stage);     xpar_free(t->f[i].backup);
  }
  xpar_free(t->f);  xpar_memset(t, 0, sizeof *t);
}

static void gen_prune_discard_stages(gen_prune_tx * t) {
  u32 i;
  for (i = 0; i < t->count; i++)
    if (t->f[i].stage) xpar_remove(t->f[i].stage);
}

static void gen_prune_commit(gen_prune_tx * t, const char * sync_path) {
  u32 i, j;
  int saved = 0;
  /*  If a crash lands between two index renames, every visible child must
      already have a visible parent. Keep index publication oldest first
      regardless of readdir order in the chain collector.  */
  for (i = 1; i < t->count; i++) {
    gen_prune_file f = t->f[i];
    u32 key = f.index ? f.order + 1 : 0;
    j = i;
    while (j && (t->f[j - 1].index ? t->f[j - 1].order + 1 : 0) > key) {
      t->f[j] = t->f[j - 1];  j--;
    }
    t->f[j] = f;
  }
  /*  A canonical target may currently be one of the old chain's names. Any
     other occupant is unrelated and is never overwritten by rotation.  */
  for (i = 0; i < t->count; i++) if (t->f[i].new_path) {
    for (j = i + 1; j < t->count; j++)
      FATAL_UNLESS("internal: two surviving volumes share a final name.",
                   !t->f[j].new_path ||
                   !gen_path_equal(t->f[i].new_path, t->f[j].new_path));
    if (gen_exists(t->f[i].new_path) &&
        gen_prune_find(t, t->f[i].new_path) < 0) {
      gen_prune_discard_stages(t);
      FATAL("'%s' is not part of this chain; prune will not overwrite it.",
            t->f[i].new_path);
    }
  }
  for (i = 0; i < t->count; i++) if (gen_exists(t->f[i].old_path)) {
    t->f[i].backup = gen_unused_path(t->f[i].old_path, "xpar-prune-old");
    if (!t->f[i].backup) {
      gen_prune_discard_stages(t);
      FATAL("Cannot choose a rollback name for '%s'.", t->f[i].old_path);
    }
  }
  for (i = 0; i < t->count; i++) if (t->f[i].backup)
    if (xpar_rename(t->f[i].old_path, t->f[i].backup) != 0) {
      saved = xpar_errno();  goto rollback;
    }
  if (xpar_fsync_dir(sync_path) != 0) {
    saved = xpar_errno();  goto rollback;
  }
  /*  Bare data and packet-bearing non-index volumes first, index files last.
     The transaction's rollback names keep every original byte reachable.  */
  for (j = 0; j < 2; j++)
    for (i = 0; i < t->count; i++) if (t->f[i].new_path) {
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
    }
  if (xpar_fsync_dir(sync_path) != 0) {
    saved = xpar_errno();  goto rollback;
  }
  for (i = 0; i < t->count; i++) if (t->f[i].backup && !t->f[i].move)
    if (xpar_remove(t->f[i].backup) != 0)
      xpar_fprintf(xpar_stderr, "xpar: cannot remove rollback volume '%s': "
                   "%s\n", t->f[i].backup, xpar_strerror(xpar_errno()));
  xpar_fsync_dir(sync_path);
  return;

rollback:
  for (i = 0; i < t->count; i++)
    if (t->f[i].published && !t->f[i].move) xpar_remove(t->f[i].new_path);
  for (i = 0; i < t->count; i++)
    if (t->f[i].published && t->f[i].move)
      xpar_rename(t->f[i].new_path, t->f[i].backup);
  for (i = 0; i < t->count; i++) if (t->f[i].backup)
    xpar_rename(t->f[i].backup, t->f[i].old_path);
  for (i = 0; i < t->count; i++)
    if (t->f[i].stage && !t->f[i].published) xpar_remove(t->f[i].stage);
  xpar_fsync_dir(sync_path);
  FATAL_IO("Cannot publish the pruned chain: %s.", xpar_strerror(saved));
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
  head = xpar_gchain_select(&c, NULL);
  removed   = (bool *) xpar_calloc(c.gen_count, sizeof(bool));
  new_id    = (u8 (*)[XPAR_SET_ID_LEN]) xpar_calloc(c.gen_count,
                                                    XPAR_SET_ID_LEN);
  new_generation = (u32 *) xpar_calloc(c.gen_count, sizeof(u32));
  new_base = (u64 *) xpar_calloc(c.gen_count, sizeof(u64));
  xpar_memset(&tx, 0, sizeof tx);

  if (o->have_before) {
    u32 lim = (u32) o->before.number;
    if (o->before.by_id) lim = c.gen[xpar_gchain_select(&c, &o->before)].
                                 sd.generation;
    for (g = 0; g < c.gen_count; g++)
      if (c.gen[g].sd.generation < lim) removed[g] = true;
  }
  for (i = 0; i < o->gen_count; i++) removed[xpar_gchain_select(&c,
                                                &o->gens[i])] = true;
  for (g = 0; g < c.gen_count; g++) if (!removed[g]) survivors++;
  if (survivors == c.gen_count)
    FATAL("prune needs --before=G or --generation=G to say what to remove.");
  if (!survivors)
    FATAL("That would remove every generation, leaving nothing behind; "
          "delete the volumes yourself if that is what you want.");
  if (removed[head])
    FATAL("Generation %u is the newest one in the chain, and every other "
          "generation is an older snapshot of it; prune drops older "
          "generations and cannot drop the newest.",
          c.gen[head].sd.generation);

  xpar_gchain_manifest(&c, head, &m, &owner);

  for (g = 0; g < c.gen_count; g++) {
    u64 dep = 0;
    for (i = 0; i < m.count; i++) {
      bool hit = owner[i] == g;
      for (k = 0; k < m.entry[i].extent_count && !hit; k++) {
        i64 h = xpar_gchain_gen_of(&c, m.entry[i].extents[k].stream_offset,
                                   m.entry[i].extents[k].length);
        if (h == (i64) g) hit = true;
      }
      if (hit) dep++;
    }
    if (!removed[g]) continue;
    reclaim += gen_volume_bytes(&c, g);
    xpar_fprintf(gen_hout(o),
                 "  gen %-3u: %llu bytes of stream, %llu bytes of volumes, "
                 "%u entries owned\n", c.gen[g].sd.generation,
                 (unsigned long long) c.gen[g].sd.stream_length,
                 (unsigned long long) gen_volume_bytes(&c, g),
                 c.gen[g].sd.file_count);
    xpar_fprintf(gen_hout(o),
                 "           %llu of generation %u's %u entries still depend "
                 "on it\n", (unsigned long long) dep,
                 c.gen[head].sd.generation, m.count);
  }

  for (i = 0; i < m.count; i++)
    if (gen_orphaned(&c, &m.entry[i], owner[i], removed)) orphans++;

  if (orphans && !o->force) {
    xpar_fprintf(gen_hout(o),
                 "refusing: %llu of generation %u's %u entries would become "
                 "unrecoverable.\n", (unsigned long long) orphans,
                 c.gen[head].sd.generation, m.count);
    xpar_fprintf(gen_hout(o),
                 "run `xpar consolidate` first: it collapses the chain so "
                 "that no earlier generation is still depended on. Pass "
                 "--force to prune anyway and accept the loss.\n");
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
    for (i = 0; i < m.count; i++)
      if (gen_orphaned(&c, &m.entry[i], owner[i], removed))
        xpar_fprintf(gen_hout(o), "  %.*s\n", (int) m.entry[i].name_len,
                     m.entry[i].name);
  }
  if (o->dry_run) {
    xpar_fprintf(gen_hout(o), "would reclaim %llu bytes of volumes.\n",
                 (unsigned long long) reclaim);
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
                   "xpar: prune cancelled; nothing was written.\n");
      xpar_free(owner);  xpar_manifest_free(&m);
      xpar_free(removed);  xpar_free(new_id);
      xpar_free(new_generation);  xpar_free(new_base);
      xpar_gchain_free(&c);
      return XPAR_EXIT_USAGE;
    }
  }
  gen_require_write_key(&c, "prune");

  {
    u32 rank = 0;
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
  for (i = 0; i < c.vol_count; i++) gen_prune_add(&tx, c.vol[i].path);
  for (g = 0; g < c.gen_count; g++) if (c.gen[g].layt_body) {
    xpar_layt l;
    if (xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &l) != XPAR_OK)
      FATAL_FORMAT("Generation %u has a malformed volume layout.",
                   c.gen[g].sd.generation);
    for (i = 0; i < l.count; i++) if (l.vol[i].kind == XPAR_VOL_DATA) {
      char * data = xpar_path_join(c.dir, l.vol[i].name);
      char * label;
      gen_prune_add(&tx, data);
      xpar_asprintf(&label, "%s" XPAR_EXT, data);
      if (gen_exists(label)) gen_prune_add(&tx, label);
      xpar_free(label);  xpar_free(data);
    }
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
    keep = (bool *) xpar_calloc(gm.count ? gm.count : 1, sizeof(bool));
    for (i = 0; i < gm.count; i++) {
      keep[i] = !gen_orphaned(&c, &gm.entry[i], gown[i], removed);
      if (keep[i]) kept++;
    }
    if (!kept)
      FATAL("Generation %u would be left with no entries at all; that is a "
            "chain with nothing in it, so nothing was written.",
            c.gen[g].sd.generation);

    gen_prune_rebase(&c, &gm, keep, removed, new_base);
    FATAL_UNLESS("internal: a surviving generation has no volume layout.",
                 gen_prune_layout(&c, g, new_generation[g],
                                  &old_layt, &layt));

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
      FATAL_UNLESS("internal: a pruned generation above zero has no "
                   "surviving parent.",
                   parent < g && !removed[parent]);
      xpar_memcpy(sd.parent_set_id, new_id[parent], XPAR_SET_ID_LEN);
    }
    gen_prune_id(&sd, &gm, keep, gen_chain_key(&c), new_id[g]);

    for (i = 0; i < c.vol_count; i++) {
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
        target = gen_name_index(c.base, new_generation[g]);
      stage = gen_stage_whole(target, out.data, out.len);
      gen_prune_output(&tx, c.vol[i].path, target, stage, false,
                       c.vol[i].volume_kind == XPAR_VOL_INDEX ||
                       c.vol[i].armoured_file, new_generation[g]);
      xpar_free(target);
      xpar_buf_free(&out);
      xpar_buf_free(&group);
    }

    /*  Bare data is already byte-correct and is moved without copying. An
       optional label is packet-bearing and therefore gets the same new
       critical group and set_id as the other volume copies.  */
    {
      u32 di;
      for (di = 0; di < old_layt.count; di++)
        if (old_layt.vol[di].kind == XPAR_VOL_DATA) {
          char * old_data = xpar_path_join(c.dir, old_layt.vol[di].name);
          char * new_data = xpar_path_join(c.dir, layt.vol[di].name);
          char * old_label, * new_label;
          if (gen_exists(old_data))
            gen_prune_output(&tx, old_data, new_data, NULL, true, false, 0);
          xpar_asprintf(&old_label, "%s" XPAR_EXT, old_data);
          xpar_asprintf(&new_label, "%s" XPAR_EXT, new_data);
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
               "pruned %u generation%s, reclaimed %llu bytes; %llu %s "
               "dropped from the surviving manifests.\n",
               c.gen_count - survivors, PLURAL(c.gen_count - survivors),
               (unsigned long long) reclaim, (unsigned long long) orphans,
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

int xpar_op_consolidate(const xpar_options * o) {
  xpar_chain c;
  xpar_manifest m;
  xpar_posix_rec ** tab;
  u32 * tabn;
  u32 * owner = NULL;
  gen_write_req rq;
  bool * owned;
  u32 head, i, caps, bad = 0;
  u64 live = 0, total = 0;
  bool warn_posix = false;
  const char * base;
  char * stage_base = NULL, * cache_path = NULL, * stage_cache = NULL;
  xpar_chunk_index chunk_cache;

  xpar_memset(&chunk_cache, 0, sizeof chunk_cache);
  xpar_memset(&rq, 0, sizeof rq);
  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "consolidate");
  if (c.authenticated && o->auth_only && !c.auth_only)
    FATAL_CODE(XPAR_EXIT_AUTH,
               "A consolidated set must keep its chain's authentication "
               "mode; this chain retains public verification hashes.");
  head = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  base = o->output ? o->output : c.base;
  if (!base) FATAL("This set has no base name; pass --output.");
  if (!o->output && !o->replace)
    FATAL("consolidate writes a new generation-0 set: give --output=BASE, "
          "or --replace to overwrite this chain in place.");

  xpar_gchain_manifest(&c, head, &m, &owner);
  tab  = (xpar_posix_rec **) xpar_calloc(c.gen_count, sizeof(void *));
  tabn = (u32 *) xpar_calloc(c.gen_count, sizeof(u32));
  for (i = 0; i < c.gen_count; i++) tabn[i] = xpar_gchain_posix(&c, i,
                                                                &tab[i]);
  for (i = 0; i < c.gen_count; i++) total += c.gen[i].sd.stream_length;
  for (i = 0; i < m.count; i++) {
    u32 k;
    for (k = 0; k < m.entry[i].extent_count; k++)
      live += m.entry[i].extents[k].length;
  }

  caps = xpar_fs_caps(o->base_dir ? o->base_dir : ".");
  owned = (bool *) xpar_calloc(m.count ? m.count : 1, sizeof(bool));
  for (i = 0; i < m.count; i++) {
    char * path = gen_entry_path(o, &m.entry[i]);
    u8 want[32];
    u32 pi = m.entry[i].posix_index;
    u32 og = owner[i];
    owned[i] = true;
    xpar_memcpy(want, m.entry[i].content_hash, 32);
    if (m.entry[i].entry_type == XPAR_ENTRY_HARDLINK) {
      xpar_free(path);  continue;
    }
    if (!gen_refresh(&m.entry[i], path, o, caps, &warn_posix,
                     gen_chain_key(&c), c.auth_only)) {
      xpar_fprintf(xpar_stderr, "xpar: cannot read '%s'.\n", path);
      bad++;  xpar_free(path);  continue;
    }
    if (xpar_memcmp(want, m.entry[i].content_hash, 32)) {
      xpar_fprintf(xpar_stderr,
                   "xpar: '%.*s' does not match the content the chain "
                   "records for it.\n", (int) m.entry[i].name_len,
                   m.entry[i].name);
      bad++;
    }
    if (pi != XPAR_ABSENT_U32 && pi < tabn[og])
      m.entry[i].posix_index = xpar_posix_intern(&m, &tab[og][pi]);
    m.source[i] = path;
  }

  if (o->dry_run) {
    xpar_fprintf(gen_hout(o),
                 "  chain      : %u generations, %u entries\n"
                 "  stream     : %llu bytes across the chain, %llu still "
                 "referenced (%.1f%%)\n"
                 "  reclaim    : %llu bytes of stream\n"
                 "  cost       : read %llu bytes, one full encode\n",
                 c.gen_count, m.count, (unsigned long long) total,
                 (unsigned long long) live,
                 total ? 100.0 * (f64) live / (f64) total : 100.0,
                 (unsigned long long) (total - live),
                 (unsigned long long) live);
    goto done;
  }
  if (bad && !o->force)
    FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
               "%u entries do not match the chain; consolidating would "
               "record the damage as the new truth. Repair first, or pass "
               "--force.", bad);

  {
    gen_merge g;
    xpar_memset(&g, 0, sizeof g);
    g.m         = m;
    g.owned     = owned;
    g.reuse     = (bool *) xpar_calloc(m.count ? m.count : 1, sizeof(bool));
    g.m.slice_size = o->slice_size;
    gen_repack(&g, o, NULL, NULL, 0,
               o->dedup == XPAR_DEDUP_CHUNK ? &chunk_cache : NULL);
    m = g.m;
    xpar_free(g.reuse);
  }

  xpar_memset(&rq, 0, sizeof rq);
  rq.o = o;  rq.m = &m;  rq.owned = owned;
  rq.generation = 0;  rq.stream_base = 0;  rq.parent_set_id = NULL;
  if (o->replace) {
    stage_base = gen_unused_base(base, "xpar-consolidate");
    rq.base = stage_base;
    rq.layout_base = base;
  } else {
    rq.base = base;
  }
  rq.quiet = o->quiet;
  rq.auth_only = c.authenticated ? c.auth_only : o->auth_only;
  if (chunk_cache.slot) {
    static const u8 unbound[XPAR_SET_ID_LEN];
    u64 average = o->dedup_chunk ? o->dedup_chunk : (u64) 1 << 20;
    xpar_asprintf(&cache_path, "%s.xparidx", base);
    stage_cache = gen_unused_path(cache_path, "xpar-cache");
    if (!stage_cache ||
        !xpar_chunk_cache_write(stage_cache, unbound, average,
                                &chunk_cache)) {
      if (o->verbose)
        xpar_fprintf(xpar_stderr, "xpar: could not stage chunk cache '%s'.\n",
                     cache_path);
      xpar_free(stage_cache);  stage_cache = NULL;
    }
    xpar_chunk_index_free(&chunk_cache);
  }
  gen_write_set(&rq);
  if (o->replace)
    gen_commit_consolidation(&c, o, stage_base, base, &rq.plan);
  if (!o->no_verify_after) {
    if (o->replace) {
      gen_vol * final;
      u32 final_n;
      final = gen_volumes(o, rq.plan.recovery, base, 0, &final_n);
      xpar_verify_written_set(o, final[0].name);
      gen_volumes_free(final, final_n);
    } else {
      xpar_verify_written_set(o, rq.index_path);
    }
  }
  if (stage_cache &&
      (!xpar_chunk_cache_rebind(stage_cache, rq.set_id) ||
       !gen_publish_cache(stage_cache, cache_path))) {
    if (o->verbose)
      xpar_fprintf(xpar_stderr, "xpar: could not publish chunk cache '%s'.\n",
                   cache_path);
    xpar_remove(stage_cache);
  }
  if (!o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: collapsed %u generations into one: %u %s, "
                 "%llu stream bytes, %llu recovery slice%s.\n", c.gen_count,
                 m.count, m.count == 1 ? "entry" : "entries",
                 (unsigned long long) m.stream_length,
                 (unsigned long long) rq.plan.recovery,
                 PLURAL(rq.plan.recovery));

done:
  gen_json_result(o, "consolidate",
                  rq.index_path ? rq.set_id : c.gen[head].set_id,
                  rq.index_path ? 0 : c.gen[head].sd.generation,
                  o->dry_run ? "dry-run" : "ok", XPAR_EXIT_OK);
  for (i = 0; i < c.gen_count; i++)
    if (tab[i]) xpar_gchain_posix_free(tab[i], tabn[i]);
  xpar_free(tab);  xpar_free(tabn);  xpar_free(owned);  xpar_free(owner);
  xpar_free(stage_base);  xpar_free(cache_path);  xpar_free(stage_cache);
  xpar_free(rq.index_path);
  xpar_chunk_index_free(&chunk_cache);
  xpar_manifest_free(&m);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

/*  recover.  */

/*  Reproduce a critical group from stored bodies without changing set_id
    inputs.  */
static void gen_group_stored(xpar_buf * out, const xpar_chain * c, u32 g,
                             const xpar_layt * layt, u32 this_vol,
                             const u8 * set_id) {
  u32 j;
  xpar_setd_write(out, &c->gen[g].sd, set_id, gen_chain_key(c));
  for (j = 0; j < c->gen[g].sd.file_count; j++) {
    const xpar_crit_pkt * q = gen_owned_file(c, g, c->gen[g].sd.file_id[j]);
    if (q) xpar_pkt_write(out, XPAR_T_FILE, q->hdr.flags, set_id, q->body,
                          (sz) q->body_len, gen_chain_key(c));
  }
  gen_emit_stored(out, c, g, XPAR_T_POSX, set_id);
  gen_emit_stored(out, c, g, XPAR_T_SLCR, set_id);
  gen_emit_stored(out, c, g, XPAR_T_AUTH, set_id);
  if (layt) {
    xpar_layt l = *layt;
    l.this_volume = this_vol;
    xpar_layt_write(out, &l, set_id, gen_chain_key(c));
  }
}

int xpar_op_recover(const xpar_options * o) {
  xpar_chain c;
  xpar_manifest m;
  xpar_layt layt;
  gen_plan p;
  gen_tables t;
  u32 * owner = NULL;
  u32 g, i, target = 0xFFFFFFFFu;
  u64 r_total = 0, e;
  xpar_buf out;
  xpar_volh vh;
  char * path;
  u8 * rec_scratch = NULL;
  xpar_vset * source_set;
  int source_rc;

  xpar_gchain_load(o, &c);
  gen_require_write_key(&c, "recover");
  g = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  if (!c.gen[g].layt_body)
    FATAL_FORMAT("Generation %u carries no volume layout, so there is "
                 "nothing to say what the lost volume held.",
                 c.gen[g].sd.generation);
  if (xpar_layt_read(c.gen[g].layt_body, c.gen[g].layt_len, &layt) != XPAR_OK)
    FATAL_FORMAT("Generation %u's volume layout is malformed.",
                 c.gen[g].sd.generation);

  for (i = 0; i < layt.count; i++) {
    if (layt.vol[i].kind == XPAR_VOL_RECOVERY)
      r_total = MAX(r_total, layt.vol[i].recovery_first +
                             layt.vol[i].byte_length);
    if (o->volume_name) {
      if (layt.vol[i].name && !xpar_strcmp(layt.vol[i].name, o->volume_name))
        target = i;
    } else if (i == (u32) o->volume_index) {
      target = i;
    }
  }
  if (target == 0xFFFFFFFFu) {
    u32 h;
    if (o->volume_name)
      for (h = 0; h < c.gen_count; h++) {
        xpar_layt other;
        if (h == g || !c.gen[h].layt_body) continue;
        if (xpar_layt_read(c.gen[h].layt_body, c.gen[h].layt_len, &other) !=
            XPAR_OK) continue;
        for (i = 0; i < other.count; i++)
          if (other.vol[i].name &&
              !xpar_strcmp(other.vol[i].name, o->volume_name)) {
            u32 num = c.gen[h].sd.generation;
            xpar_layt_free(&other);
            FATAL("'%s' belongs to generation %u, not to generation %u; "
                  "pass --generation=%u.", o->volume_name, num,
                  c.gen[g].sd.generation, num);
          }
        xpar_layt_free(&other);
      }
    if (o->volume_name)
      FATAL("Generation %u's layout names no volume '%s'.",
            c.gen[g].sd.generation, o->volume_name);
    FATAL("Generation %u's layout has %u volumes, so there is no volume "
          "%llu.", c.gen[g].sd.generation, layt.count,
          (unsigned long long) o->volume_index);
  }
  if (c.gen[g].sd.layout == XPAR_LAYOUT_ARMOURED) {
    const xpar_chain_vol * source = NULL;
    for (i = 0; i < c.vol_count; i++)
      if (c.vol[i].gen == g && c.vol[i].armoured_file) {
        source = &c.vol[i];
        break;
      }
    if (!source)
      FATAL_FORMAT("Generation %u's armoured archive is unavailable.",
                   c.gen[g].sd.generation);
    if (o->to_dir && xpar_strlen(o->to_dir))
      xpar_asprintf(&path, "%s/%s", o->to_dir, layt.vol[target].name);
    else
      path = xpar_path_join(c.dir, layt.vol[target].name);
    gen_write_whole(path, source->data, source->len, o->force);
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: regenerated %s (%llu armoured "
                   "bytes).\n", path, (unsigned long long) source->len);
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
      FATAL("'%s' exists; -f overwrites it.", path);
    set = xpar_vset_open(o);
    dst = gen_stage_open_rw(path, &tmp);
    if (!xpar_vset_recover_data(set, layt.vol[target].stream_offset,
                                layt.vol[target].byte_length, o->memory,
                                dst, &why)) {
      xpar_xclose(dst); xpar_remove(tmp); xpar_free(tmp);
      xpar_vset_close(set);
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "Volume '%s' cannot be reconstructed from the surviving "
                 "data and recovery slices: %s.", layt.vol[target].name,
                 why ? why : "unknown decoder failure");
    }
    if (xpar_flush(dst) != 0 || xpar_fsync(dst) != 0) {
      xpar_xclose(dst); xpar_remove(tmp); xpar_free(tmp);
      xpar_vset_close(set);
      FATAL_IO("Cannot flush reconstructed volume '%s'.", path);
    }
    xpar_xclose(dst);
    xpar_vset_close(set);
    gen_publish_whole(tmp, path, o->force);
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: recovered %s from survivor and "
                   "parity slices (%llu bare stream bytes).\n", path,
                   (unsigned long long) layt.vol[target].byte_length);
    xpar_free(path);
    xpar_layt_free(&layt);
    xpar_gchain_free(&c);
    return XPAR_EXIT_OK;
  }
  gen_manifest_on_disk(&c, g, o, &m, &owner);
  xpar_memset(&p, 0, sizeof p);
  if (!xpar_geom_from_setd(&c.gen[g].sd, &p.geom))
    FATAL_FORMAT("Generation %u's geometry is malformed.",
                 c.gen[g].sd.generation);
  p.recovery   = r_total;
  p.encode_r   = r_total;
  p.field_log2 = c.gen[g].sd.field_log2;
  p.codec      = c.gen[g].sd.codec;
  p.axis       = c.gen[g].sd.recovery_axis_log2;
  /*  Recover one volume in one pass over surviving data.  */
  source_set = xpar_vset_open(o);
  source_rc = xpar_vset_check(source_set, o, NULL);
  if (source_rc != XPAR_EXIT_OK)
    FATAL_CODE(source_rc,
               "Generation %u's protected stream is not clean; refusing to "
               "derive a replacement recovery volume from it.",
               c.gen[g].sd.generation);
  gen_encode(&m, &p, c.gen[g].sd.slice_tag_len, o->memory,
             c.base ? c.base : o->set, gen_chain_key(&c), gen_read_vset,
             source_set, &t, NULL);
  gen_require_source_tables(source_set, &t, &p);
  xpar_vset_close(source_set);
  if (t.rec_spill) rec_scratch = (u8 *) xpar_alloc_raw((sz) t.rec_z);

  xpar_buf_init(&out);
  xpar_memset(&vh, 0, sizeof vh);
  vh.volume_index = layt.vol[target].kind == XPAR_VOL_INDEX
                      ? XPAR_VOL_STANDALONE : target;
  vh.volume_kind  = layt.vol[target].kind;
  xpar_volh_write(&out, &vh, c.gen[g].set_id, gen_chain_key(&c));
  {
    xpar_buf group;
    u64 payload = layt.vol[target].byte_length * p.geom.slice_size;
    bool carry;
    xpar_buf_init(&group);
    gen_group_stored(&group, &c, g, &layt,
                     layt.vol[target].kind == XPAR_VOL_INDEX
                       ? XPAR_VOL_STANDALONE : target, c.gen[g].set_id);
    carry = layt.vol[target].kind == XPAR_VOL_INDEX ||
            xpar_replicate_here(group.len, payload, target - 1,
                                layt.count - 1);
    if (carry) {
      if (o->armour != XPAR_ARMOUR_NONE)
        gen_armour_pack(&out, o, group.data, group.len, c.gen[g].set_id,
                        gen_chain_key(&c));
      else
        xpar_buf_put(&out, group.data, group.len);
    }
    xpar_buf_free(&group);
  }
  if (layt.vol[target].kind == XPAR_VOL_INDEX || target == 1) {
    if (t.slice_tag)
      xpar_sltg_write_all(&out, t.slice_tag, p.geom.slice_count, t.tag_len,
                          c.gen[g].set_id, gen_chain_key(&c));
    if (t.cell_crc)
      xpar_slcl_write_all(&out, t.cell_crc, p.geom.slice_count,
                          p.geom.cell_bytes, p.geom.cells_per_slice,
                          c.gen[g].set_id, gen_chain_key(&c));
  }
  for (e = layt.vol[target].recovery_first;
       e < layt.vol[target].recovery_first + layt.vol[target].byte_length;
       e++) {
    const u8 * rec = gen_rec_get(&t, e, rec_scratch);
    xpar_rcvs_write(&out, e, rec, (sz) p.geom.slice_size,
                    c.gen[g].set_id, gen_chain_key(&c));
  }
  xpar_crtr_write(&out, "xpar " PACKAGE_VERSION, c.gen[g].set_id,
                  gen_chain_key(&c), NULL);

  if (o->to_dir && xpar_strlen(o->to_dir))
    xpar_asprintf(&path, "%s/%s", o->to_dir, layt.vol[target].name);
  else
    path = xpar_path_join(c.dir, layt.vol[target].name);
  gen_write_whole(path, out.data, out.len, o->force);
  if (!o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: regenerated %s (%llu bytes, %llu "
                 "recovery slices).\n", path, (unsigned long long) out.len,
                 (unsigned long long) layt.vol[target].byte_length);
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

/*  Replay complete, checked repair journals. Missing footers prove pass 4
    never began; invalid records are never applied.  */

#define UNDO_HDR   64u
#define UNDO_REC   40u
#define UNDO_FOOT  24u
#define UNDO_CREATED 1u          /*  rflags bit 0: the file was created.  */

static bool gen_undo_path_allowed(const xpar_chain * c,
                                  const xpar_manifest * m,
                                  const char * path, u32 plen) {
  u32 i;
  for (i = 0; i < m->count; i++) {
    const xpar_entry * e = &m->entry[i];
    const char * dir = c->dir && *c->dir ? c->dir : ".";
    sz dn = xpar_strlen(dir);
    bool sep = dn && dir[dn - 1] != '/' && dir[dn - 1] != '\\';
    char * allowed;
    bool same;
    if (e->entry_type != XPAR_ENTRY_REGULAR) continue;
    allowed = (char *) xpar_malloc(dn + sep + e->name_len + 1);
    xpar_memcpy(allowed, dir, dn);
    if (sep) allowed[dn++] = '/';
    xpar_memcpy(allowed + dn, e->name, e->name_len);
    allowed[dn + e->name_len] = '\0';
    same = xpar_strlen(allowed) == plen && !xpar_memcmp(allowed, path, plen);
    xpar_free(allowed);
    if (same) return true;
  }
  return false;
}

static bool gen_has_nul(const u8 * p, u32 n) {
  u32 i;
  for (i = 0; i < n; i++) if (!p[i]) return true;
  return false;
}

int xpar_op_undo(const xpar_options * o) {
  u8 * j;  sz n = 0;  u64 at, count, i;
  char * path = NULL;
  u32 applied = 0, skipped = 0, removed = 0;
  xpar_chain chain;
  xpar_manifest manifest;
  u32 * owner = NULL;
  u32 generation;

  if (o->set && xpar_path_ends_with(o->set, ".xparundo"))
    FATAL("undo takes the protected set, not a journal pathname; it needs "
          "the set to bind the journal to the right generation and "
          "manifest.");
  xpar_gchain_load(o, &chain);
  generation = xpar_gchain_select(&chain,
                         o->gen_count ? &o->gens[0] : NULL);
  if (o->set_ref.base) {
    if (chain.gen[generation].sd.generation)
      xpar_asprintf(&path, "%s.g%03u.xparundo", o->set_ref.base,
                    chain.gen[generation].sd.generation);
    else
      xpar_asprintf(&path, "%s.xparundo", o->set_ref.base);
  } else {
    FATAL("undo needs a set with a resolvable base name.");
  }

  j = gen_read_whole(path, &n, false);
  if (!j) {
    xpar_fprintf(xpar_stderr, "xpar: no journal at '%s'; nothing to undo.\n",
                 path);
    gen_json_result(o, "undo", chain.gen[generation].set_id,
                    chain.gen[generation].sd.generation, "not-found",
                    XPAR_EXIT_NOTFOUND);
    xpar_free(path);
    xpar_gchain_free(&chain);
    return XPAR_EXIT_NOTFOUND;
  }
  if (n < UNDO_HDR + UNDO_FOOT || xpar_memcmp(j, "XPARUNDO", 8))
    FATAL_FORMAT("'%s' is not an xpar repair journal.", path);
  if (xpar_rd32(j + 8) != 1)
    FATAL_FORMAT("'%s' is a version %lu journal; this build reads 1.", path,
                 (unsigned long) xpar_rd32(j + 8));
  if (xpar_crc32c(0, j, 60) != xpar_rd32(j + 60))
    FATAL_FORMAT("The header of '%s' does not verify.", path);
  if (xpar_rd32(j + 12) || xpar_rd32(j + 56))
    FATAL_FORMAT("The header of '%s' has non-zero reserved fields.", path);

  count = xpar_rd64(j + 32);
  {
    const u8 * foot = j + n - UNDO_FOOT;
    bool complete = !xpar_memcmp(foot, "XPARUNDN", 8) &&
                    xpar_rd64(foot + 8) == count &&
                    xpar_crc32c(0, j, (sz) (n - UNDO_FOOT)) ==
                      xpar_rd32(foot + 16);
    if (!complete) {
      /*  An incomplete journal predates all protected-data writes.  */
      xpar_fprintf(xpar_stderr,
                   "xpar: the journal '%s' is incomplete, so the repair that "
                   "owns it never wrote anything and there is nothing to "
                   "undo. %s\n", path,
                   o->keep_journal ? "Kept." : "Removing it.");
      if (!o->keep_journal && xpar_remove(path) != 0)
        xpar_fprintf(xpar_stderr, "xpar: cannot remove '%s'.\n", path);
      gen_json_result(o, "undo", chain.gen[generation].set_id,
                      chain.gen[generation].sd.generation, "ok",
                      XPAR_EXIT_OK);
      xpar_free(j);  xpar_free(path);
      xpar_gchain_free(&chain);
      return XPAR_EXIT_OK;
    }
    if (xpar_rd32(foot + 20))
      FATAL_FORMAT("The footer of '%s' has a non-zero reserved field.", path);
  }

  if (xpar_memcmp(j + 16, chain.gen[generation].set_id, XPAR_SET_ID_LEN))
    FATAL_FORMAT("The journal '%s' belongs to a different set or generation.",
                 path);
  xpar_gchain_manifest(&chain, generation, &manifest, &owner);

  /*  Validate every record before replaying any of them. The whole-file CRC
      detects a torn write, but it does not make attacker-controlled lengths
      or paths safe to use.  */
  {
    u64 payload = 0;
    at = UNDO_HDR;
    for (i = 0; i < count; i++) {
      const u8 * rec;
      u32 plen, rflags;
      u64 off, len, raw, step, remain;
      const u8 * old;
      u64 k;
      if (at > (u64) n - UNDO_FOOT ||
          (u64) n - UNDO_FOOT - at < UNDO_REC)
        FATAL_FORMAT("Journal '%s' ends before record %llu.", path,
                     (unsigned long long) i);
      rec = j + at;
      plen = xpar_rd32(rec);
      rflags = xpar_rd32(rec + 4);
      off = xpar_rd64(rec + 8);
      len = xpar_rd64(rec + 16);
      remain = (u64) n - UNDO_FOOT - at;
      if ((rflags & ~UNDO_CREATED) || !plen || off + len < off ||
          (u64) UNDO_REC + plen > remain ||
          len > remain - UNDO_REC - plen)
        FATAL_FORMAT("Journal '%s' has invalid framing in record %llu.", path,
                     (unsigned long long) i);
      raw = (u64) UNDO_REC + plen + len;
      if (raw > (u64) -1 - 7)
        FATAL_FORMAT("Journal '%s' overflows record %llu's length.", path,
                     (unsigned long long) i);
      step = xpar_align_up(raw, 8);
      if (step > remain)
        FATAL_FORMAT("Journal '%s' truncates record %llu.", path,
                     (unsigned long long) i);
      if (payload + len < payload)
        FATAL_FORMAT("Journal '%s' overflows its payload count.", path);
      payload += len;
      if (gen_has_nul(rec + UNDO_REC, plen) ||
          !gen_undo_path_allowed(&chain, &manifest,
                                 (const char *) rec + UNDO_REC, plen))
        FATAL_FORMAT("Journal record %llu names a path outside the selected "
                     "set.", (unsigned long long) i);
      old = rec + UNDO_REC + plen;
      if (xpar_crc32c(0, rec, 36) != xpar_rd32(rec + 36) ||
          xpar_crc32c(0, old, (sz) len) != xpar_rd32(rec + 32))
        FATAL_FORMAT("Journal record %llu does not verify.",
                     (unsigned long long) i);
      for (k = raw; k < step; k++)
        if (rec[k]) FATAL_FORMAT("Journal record %llu has non-zero padding.",
                                 (unsigned long long) i);
      at += step;
    }
    if (at != (u64) n - UNDO_FOOT || payload != xpar_rd64(j + 40))
      FATAL_FORMAT("Journal '%s' has inconsistent record or payload counts.",
                   path);
  }

  at = UNDO_HDR;
  for (i = 0; i < count; i++) {
    const u8 * rec = j + at;
    u32 plen, rflags;
    u64 off, len, orig, step;
    const char * rp;  const u8 * old;
    char * full;
    xpar_file * f;

    if (at + UNDO_REC > n - UNDO_FOOT) break;
    plen   = xpar_rd32(rec);
    rflags = xpar_rd32(rec + 4);
    off    = xpar_rd64(rec + 8);
    len    = xpar_rd64(rec + 16);
    orig   = xpar_rd64(rec + 24);
    step   = xpar_align_up((u64) UNDO_REC + plen + len, 8);
    if (at + step > n - UNDO_FOOT) break;
    rp  = (const char *) rec + UNDO_REC;
    old = rec + UNDO_REC + plen;
    if (xpar_crc32c(0, rec, 36) != xpar_rd32(rec + 36) ||
        xpar_crc32c(0, old, (sz) len) != xpar_rd32(rec + 32)) {
      xpar_fprintf(xpar_stderr,
                   "xpar: journal record %llu does not verify; it and "
                   "everything after it are a torn tail and are not "
                   "replayed.\n", (unsigned long long) i);
      skipped += (u32) (count - i);
      break;
    }
    at += step;

    full = (char *) xpar_malloc(plen + 1);
    xpar_memcpy(full, rp, plen);  full[plen] = 0;
    if (rflags & UNDO_CREATED) {
      /*  The file did not exist before the repair, so putting it back
          means removing it rather than truncating it to zero.  */
      if (xpar_remove(full) != 0)
        xpar_fprintf(xpar_stderr, "xpar: cannot remove '%s': %s\n", full,
                     xpar_strerror(xpar_errno()));
      else { removed++;  applied++; }
      xpar_free(full);
      continue;
    }
    f = xpar_open(full, XPAR_O_RDWR);
    if (!f) {
      xpar_fprintf(xpar_stderr, "xpar: cannot open '%s': %s\n", full,
                   xpar_strerror(xpar_errno()));
      skipped++;  xpar_free(full);  continue;
    }
    if (len && xpar_pwrite(f, old, (sz) len, off) != (sz) len) {
      xpar_fprintf(xpar_stderr, "xpar: short write to '%s'.\n", full);
      skipped++;
    } else {
      /*  A repair that lengthened or truncated a file is undone only
          when the length goes back too, which is what orig_size is for.  */
      if (xpar_size(f) != (i64) orig && xpar_ftruncate(f, orig) != 0)
        xpar_fprintf(xpar_stderr, "xpar: cannot restore the length of "
                     "'%s'.\n", full);
      applied++;
    }
    xpar_fsync(f);
    xpar_xclose(f);
    xpar_free(full);
  }

  xpar_fprintf(xpar_stderr,
               "xpar: replayed %lu of %llu journal records%s%s.\n",
               (unsigned long) applied, (unsigned long long) count,
               removed ? ", removing files the repair had created" : "",
               skipped ? "; the rest could not be applied" : "");
  if (!skipped && !o->keep_journal && xpar_remove(path) != 0)
    xpar_fprintf(xpar_stderr, "xpar: cannot remove '%s'.\n", path);
  gen_json_result(o, "undo", chain.gen[generation].set_id,
                  chain.gen[generation].sd.generation,
                  skipped ? "unrepairable" : "ok",
                  skipped ? XPAR_EXIT_UNREPAIRABLE : XPAR_EXIT_OK);
  xpar_free(j);  xpar_free(path);
  xpar_free(owner);  xpar_manifest_free(&manifest);
  xpar_gchain_free(&chain);
  return skipped ? XPAR_EXIT_UNREPAIRABLE : XPAR_EXIT_OK;
}

/*  Recover armour parameters by requiring all-zero frame syndromes. Field
    choice fixes n and the polynomial, bounding the search.  */

int xpar_op_recover_prologue(const xpar_options * o) {
  u8 * f;  sz n = 0;  u64 region;
  xpar_arm_prologue pr;
  xpar_key key;
  u8 master[XPAR_BLAKE3_KEY_LEN];
  bool key_loaded = false;
  int bits, bit_order[2] = { 8, 16 }, bi, found = 0;
  u32 t;
  u64 d;
  u8 * frame = NULL;

  xpar_memset(&key, 0, sizeof key);
  xpar_memset(master, 0, sizeof master);
  if (o->auth_key) {
    xpar_keyfile_status ks = xpar_keyfile_load(o->auth_key, &key, master);
    if (ks == XPAR_KEYFILE_OPEN) FATAL_PERROR(o->auth_key);
    if (ks == XPAR_KEYFILE_EMPTY)
      FATAL_CODE(XPAR_EXIT_AUTH, "The key file is empty.");
    if (ks != XPAR_KEYFILE_OK)
      FATAL_CODE(XPAR_EXIT_AUTH, "Reading key file '%s' failed.",
                 o->auth_key);
    key_loaded = true;
  }

  f = gen_read_whole(o->set, &n, true);
  if (n <= ARM_HDR_LEN)
    FATAL_FORMAT("'%s' is too short to be an armoured archive.", o->set);
  region = (u64) n - ARM_HDR_LEN;
  xpar_memset(&pr, 0, sizeof pr);

  /*  Try symbol widths whose frame size divides the input first; syndromes
      remain the authority.  */
  if (region >= 2ull * 65535 && region % (2ull * 65535) == 0) {
    bit_order[0] = 16; bit_order[1] = 8;
  }

  /*  Try preferred power-of-two depths first, then every depth below 64.  */
  /*  Search t descending: every true codeword also satisfies all smaller-t
      generators, so an ascending search would always stop at one.  */
  for (bi = 0; bi < 2 && !found; bi++) {
    u32 di;
    bits = bit_order[bi];
    for (di = 0; di < 82 && !found; di++) {
      u32 nmax, ncand;
      u64 symbols;
      d = di < 64 ? (u64) di + 1 : ((u64) 1 << (di - 64 + 7));
      if (d > XPAR_ARMG_DEPTH_MAX) break;
      if (region % (d * (u64) (bits / 8))) continue;
      symbols = region / (d * (u64) (bits / 8));
      nmax = bits == 8 ? 255u : 65535u;
      if (symbols < nmax) nmax = (u32) symbols;
      /*  Shortened n must divide the region into integral D*n*W frames.  */
      for (ncand = nmax; ncand >= 3 && !found; ncand--) {
        xpar_armour_params ap;
        xpar_armour * a;
        u64 fx;
        xpar_armour_defaults(&ap, (u32) bits);
        if (symbols % ncand) continue;
        ap.n = ncand; ap.depth = d;
        for (t = MIN(128u, (ncand - 1) / 2); t >= 1 && !found; t--) {
          ap.k = ap.n - 2 * t;
          if (xpar_armour_check(&ap)) continue;
          fx = (u64) d * ap.n * (ap.symbol_bits / 8);
          a = xpar_armour_new(&ap);
          frame = (u8 *) xpar_realloc(frame, (sz) fx);
          xpar_memcpy(frame, f + ARM_HDR_LEN, (sz) fx);
          if (xpar_armour_decode_frame(a, frame, NULL) ==
              XPAR_ARMOUR_CLEAN) {
            u64 frames = region / fx;
            u64 q;
            bool all_clean = true;
            /*  A short false framing can occasionally satisfy one syndrome
               check because the actual region is itself coded. Require the
               candidate to frame every byte before accepting it.  */
            for (q = 1; q < frames && all_clean; q++) {
              xpar_memcpy(frame, f + ARM_HDR_LEN + q * fx, (sz) fx);
              if (xpar_armour_decode_frame(a, frame, NULL) !=
                  XPAR_ARMOUR_CLEAN) all_clean = false;
            }
            if (all_clean) {
              u64 maxplain = frames * d * ap.k * (ap.symbol_bits / 8);
              u8 * probe = (u8 *) xpar_alloc_raw((sz) maxplain);
              xpar_scan ps;
              xpar_pkt ph;
              const u8 * pb;
              u64 po;
              bool have_setd = false, have_strm = false;
              xpar_armour_extract(a, probe, maxplain, f + ARM_HDR_LEN);
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
               "No (symbol width, t, depth) triple gives clean syndromes on "
               "the first frame of '%s'. Either the region is damaged at its "
               "head, or the archive was written with a non-default field "
               "polynomial, first root or primitive step, which the search "
               "cannot cover.", o->set);

  /*  The parameters demodulate the region; the stream range comes out of
      the packets inside it, which is where SETD says how long the
      protected stream is.  */
  {
    xpar_armour_params ap;
    u8 * plain;  sz plen;
    xpar_scan sc;  xpar_pkt hdr;  const u8 * body;
    u64 off, last = 0, declared_stream = 0;
    bool authenticated = false;
    arm_params_of(&pr, &ap);
    plain = arm_extract(&ap, f + ARM_HDR_LEN, region, pr.plain_length, &plen,
                        NULL);
    if (!plain) FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                           "The recovered parameters do not frame the "
                           "region.");
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
                   "Recovering this authenticated archive's prologue "
                   "requires --auth-key=FILE.");
      }
      if (!xpar_auth_key_ok(&auth, master)) {
        xpar_free(plain);
        FATAL_CODE(XPAR_EXIT_AUTH,
                   "The authentication key is wrong for this archive.");
      }
      break;
    }
    xpar_scan_init(&sc, plain, plen,
                   authenticated && key_loaded ? &key : NULL, true);
    while (xpar_scan_next(&sc, &hdr, &body, &off)) {
      if (off + hdr.length > last) last = off + hdr.length;
      if (xpar_pkt_is(&hdr, XPAR_T_STRM)) {
        pr.stream_offset = off + XPAR_PKT_HDR + 16;
        pr.stream_length = hdr.length - XPAR_PKT_HDR - 16;
      }
      if (xpar_pkt_is(&hdr, XPAR_T_SETD)) {
        xpar_setd sd;
        if (xpar_setd_read(body, (sz) (hdr.length - XPAR_PKT_HDR), &sd) ==
            XPAR_OK) {
          if (o->verbose > 1)
            xpar_fprintf(xpar_stderr,
                         "xpar: recovered SETD stream length %llu.\n",
                         (unsigned long long) sd.stream_length);
          declared_stream = sd.stream_length;
          xpar_setd_free(&sd);
        }
      }
    }
    if (declared_stream &&
        (!pr.stream_length || declared_stream <= pr.stream_length))
      pr.stream_length = declared_stream;
    if (!last || !pr.stream_length) {
      xpar_free(plain);
      FATAL_CODE(XPAR_EXIT_UNREPAIRABLE,
                 "The recovered framing does not contain a validating "
                 "protected stream.");
    }
    pr.plain_length = last;
    xpar_free(plain);
  }

  xpar_fprintf(gen_hout(o),
               "recovered prologue for %s:\n"
               "  symbol_bits     %u\n  poly            0x%lX\n"
               "  n               %lu\n  k               %lu   (t = %lu)\n"
               "  fcr             %lu\n  prim            %lu\n"
               "  depth D         %llu\n  plain_length    %llu\n"
               "  armoured_length %llu\n  stream_offset   %llu\n"
               "  stream_length   %llu\n",
               o->set, pr.symbol_bits, (unsigned long) pr.poly,
               (unsigned long) pr.n, (unsigned long) pr.k,
               (unsigned long) ((pr.n - pr.k) / 2), (unsigned long) pr.fcr,
               (unsigned long) pr.prim, (unsigned long long) pr.depth,
               (unsigned long long) pr.plain_length,
               (unsigned long long) pr.armoured_length,
               (unsigned long long) pr.stream_offset,
               (unsigned long long) pr.stream_length);

  if (!o->dry_run) {
    xpar_armour_params ap;
    xpar_armour * a;
    u8 copy[ARM_COPY_LEN];
    xpar_file * out;
    int i;
    /*  Each 96-byte prologue is zero-extended to RS(255,223) data and
        followed by its 32 parity bytes.  */
    xpar_armour_defaults(&ap, 8);
    a = xpar_armour_new(&ap);
    xpar_memset(copy, 0, sizeof copy);
    arm_prologue_encode(copy, &pr);
    {
      u8 * fr = (u8 *) xpar_calloc((sz) ap.n, 1);
      xpar_memcpy(fr, copy, ARM_PLAIN_LEN);
      xpar_armour_encode_frame(a, fr);
      xpar_memcpy(copy + ARM_PLAIN_LEN, fr + ap.k, 32);
      xpar_free(fr);
    }
    xpar_armour_free(a);
    out = xpar_open(o->set, XPAR_O_RDWR);
    if (!out) FATAL_PERROR(o->set);
    for (i = 0; i < 3; i++)
      if (xpar_pwrite(out, copy, sizeof copy, (u64) i * ARM_COPY_LEN) !=
          sizeof copy)
        FATAL_IO("Short write to '%s'.", o->set);
    xpar_fsync(out);
    xpar_xclose(out);
    xpar_fprintf(xpar_stderr, "xpar: wrote three repaired prologue copies "
                 "to %s.\n", o->set);
  }
  xpar_free(f);
  gen_json_result(o, "recover-prologue", NULL, 0, "ok", XPAR_EXIT_OK);
  xpar_key_forget(&key, master);
  return XPAR_EXIT_OK;
}

/*  Differential installed selftest: every runtime-selected kernel tier
    must match scalar output on fixed-seed inputs.  */

typedef struct { u64 s; } st_rng;

static u32 st_next(st_rng * r) {
  r->s ^= r->s << 13;  r->s ^= r->s >> 7;  r->s ^= r->s << 17;
  return (u32) (r->s >> 32);
}

static void st_fill(st_rng * r, u8 * p, sz n) {
  sz i;
  for (i = 0; i < n; i++) p[i] = (u8) st_next(r);
}

static u32 st_cmp(const char * tier, const char * what, const u8 * a,
                  const u8 * b, sz n) {
  sz i;
  for (i = 0; i < n; i++)
    if (a[i] != b[i]) {
      xpar_fprintf(xpar_stderr,
                   "xpar: selftest: %s disagrees with scalar in %s at byte "
                   "%lu of %lu (%02X against %02X).\n", tier, what,
                   (unsigned long) i, (unsigned long) n, a[i], b[i]);
      return 1;
    }
  return 0;
}

/*  Exercise each GF region entry point across vector-width boundaries.  */
static u32 st_check_gf(const xpar_gf_kernels * k, const char * tier) {
  static const sz len[] = { 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
                            65, 127, 128, 255, 256, 1023, 4096 };
  const sz big = 4096;
  u8 * src = (u8 *) xpar_alloc_raw(big);
  u8 * d0  = (u8 *) xpar_alloc_raw(big);
  u8 * a   = (u8 *) xpar_alloc_raw(big);
  u8 * b   = (u8 *) xpar_alloc_raw(big);
  u8 * a2  = (u8 *) xpar_alloc_raw(big);
  u8 * b2  = (u8 *) xpar_alloc_raw(big);
  st_rng r;
  u32 bad = 0, i;

  r.s = 0x9E3779B97F4A7C15ull;
  st_fill(&r, src, big);
  st_fill(&r, d0, big);

  for (i = 0; i < ARRAY_LEN(len) && !bad; i++) {
    sz n = len[i], n2 = n & ~(sz) 1;
    u8  c8  = (u8)  (1 + (st_next(&r) & 0xFE));
    u16 c16 = (u16) (1 + (st_next(&r) & 0xFFFE));
    xpar_gf8_coef  m8;
    xpar_gf16_coef m16;
    xpar_gf8_prepare(&m8, c8);
    xpar_gf16_prepare(&m16, c16);

    xpar_memcpy(a, d0, n);  k->mac8(a, src, n, &m8);
    xpar_memcpy(b, d0, n);  xpar_gf8_mac_ref(b, src, n, c8);
    bad += st_cmp(tier, "mac8", a, b, n);

    k->mul8(a, src, n, &m8);
    xpar_gf8_mul_ref(b, src, n, c8);
    bad += st_cmp(tier, "mul8", a, b, n);

    xpar_memcpy(a, d0, n);  k->xor2(a, src, n);
    xpar_memcpy(b, d0, n);  xpar_xor2_ref(b, src, n);
    bad += st_cmp(tier, "xor2", a, b, n);

    k->xor3(a, d0, src, n);
    xpar_xor3_ref(b, d0, src, n);
    bad += st_cmp(tier, "xor3", a, b, n);

    xpar_memcpy(a, d0, n);  xpar_memcpy(a2, src, n);
    k->fft8(a, a2, n, &m8);
    xpar_memcpy(b, d0, n);  xpar_memcpy(b2, src, n);
    xpar_gf8_fft2_ref(b, b2, n, c8);
    bad += st_cmp(tier, "fft8 x", a, b, n);
    bad += st_cmp(tier, "fft8 y", a2, b2, n);

    xpar_memcpy(a, d0, n);  xpar_memcpy(a2, src, n);
    k->ifft8(a, a2, n, &m8);
    xpar_memcpy(b, d0, n);  xpar_memcpy(b2, src, n);
    xpar_gf8_ifft2_ref(b, b2, n, c8);
    bad += st_cmp(tier, "ifft8 x", a, b, n);
    bad += st_cmp(tier, "ifft8 y", a2, b2, n);

    if (!n2) continue;
    xpar_memcpy(a, d0, n2);  k->mac16(a, src, n2, &m16);
    xpar_memcpy(b, d0, n2);  xpar_gf16_mac_ref(b, src, n2, c16);
    bad += st_cmp(tier, "mac16", a, b, n2);

    k->mul16(a, src, n2, &m16);
    xpar_gf16_mul_ref(b, src, n2, c16);
    bad += st_cmp(tier, "mul16", a, b, n2);

    xpar_memcpy(a, d0, n2);  xpar_memcpy(a2, src, n2);
    k->fft16(a, a2, n2, &m16);
    xpar_memcpy(b, d0, n2);  xpar_memcpy(b2, src, n2);
    xpar_gf16_fft2_ref(b, b2, n2, c16);
    bad += st_cmp(tier, "fft16 x", a, b, n2);
    bad += st_cmp(tier, "fft16 y", a2, b2, n2);

    xpar_memcpy(a, d0, n2);  xpar_memcpy(a2, src, n2);
    k->ifft16(a, a2, n2, &m16);
    xpar_memcpy(b, d0, n2);  xpar_memcpy(b2, src, n2);
    xpar_gf16_ifft2_ref(b, b2, n2, c16);
    bad += st_cmp(tier, "ifft16 x", a, b, n2);
    bad += st_cmp(tier, "ifft16 y", a2, b2, n2);
  }
  xpar_free(src);  xpar_free(d0);  xpar_free(a);  xpar_free(b);
  xpar_free(a2);   xpar_free(b2);
  return bad;
}

/*  Exercise armour tiers through encode, t-symbol damage and decode.  */
static u32 st_armour_frame(const xpar_armour_params * p, u8 * frame,
                           st_rng * r) {
  xpar_armour * a = xpar_armour_new(p);
  u64 fd = xpar_armour_frame_plain(a), fx = xpar_armour_frame_disk(a);
  st_fill(r, frame, (sz) fx);
  xpar_memset(frame + fd, 0, (sz) (fx - fd));
  xpar_armour_encode_frame(a, frame);
  xpar_armour_free(a);
  return (u32) fx;
}

static u32 st_check_armour(const char * tier, const xpar_armour_params * p,
                           const u8 * ref, u64 fx, const char * what) {
  u8 * frame = (u8 *) xpar_alloc_raw((sz) fx);
  xpar_armour * a = xpar_armour_new(p);
  st_rng r;
  u32 bad = 0, i, t = (p->n - p->k) / 2, w = p->symbol_bits / 8;

  r.s = 0x243F6A8885A308D3ull;
  xpar_memcpy(frame, ref, (sz) fx);
  xpar_armour_encode_frame(a, frame);
  bad += st_cmp(tier, what, frame, ref, (sz) fx);

  /*  t errors in every codeword is the capacity, so a tier that decodes
      one symbol differently shows up as a whole frame that will not come
      back.  */
  for (i = 0; i < t; i++) {
    u32 s = st_next(&r) % p->n;
    u64 at = ((u64) s * p->depth) * w;
    frame[at] ^= 0xA5;
    if (w == 2) frame[at + 1] ^= 0x5A;
  }
  if (xpar_armour_decode_frame(a, frame, NULL) == XPAR_ARMOUR_FAILED) {
    xpar_fprintf(xpar_stderr, "xpar: selftest: %s failed to decode a frame "
                 "at capacity (%s).\n", tier, what);
    bad++;
  } else {
    bad += st_cmp(tier, what, frame, ref, (sz) fx);
  }
  xpar_armour_free(a);
  xpar_free(frame);
  return bad;
}

static u32 st_check_crc32c(void) {
  static const sz len[] = { 1, 7, 8, 63, 64, 255, 256, 1024, 8192, 24577,
                            65536 };
  u32 bad = 0, i;
  u8 * buf;
  st_rng r;
#if defined(HAVE_SSE42) || defined(HAVE_ARM_CRC32)
  u32 feat = xpar_cpu_features();
#endif
  r.s = 0xB5026F5AA96619Eull;
  buf = (u8 *) xpar_alloc_raw(65536);
  st_fill(&r, buf, 65536);
  for (i = 0; i < ARRAY_LEN(len); i++) {
    u32 want = xpar_crc32c_scalar(0x1234u, buf, len[i]);
    u32 got  = want;
    const char * name = NULL;
#ifdef HAVE_SSE42
    if (feat & XPAR_CPU_SSE42) {
      got = xpar_crc32c_sse42(0x1234u, buf, len[i]);  name = "sse42";
    }
#endif
#ifdef HAVE_ARM_CRC32
    if (feat & XPAR_CPU_ARMCRC) {
      got = xpar_crc32c_arm(0x1234u, buf, len[i]);  name = "armcrc";
    }
#endif
    if (name && got != want) {
      xpar_fprintf(xpar_stderr, "xpar: selftest: crc32c %s gives %08lX at "
                   "%lu bytes, scalar gives %08lX.\n", name,
                   (unsigned long) got, (unsigned long) len[i],
                   (unsigned long) want);
      bad++;
    }
  }
  xpar_free(buf);
  return bad;
}

static u32 st_check_blake3(void) {
  enum { LANES = XPAR_BLAKE3_MAX_DEGREE, BLOCKS = 3 };
  u8 * in = (u8 *) xpar_alloc_raw(LANES * BLOCKS * XPAR_BLAKE3_BLOCK_LEN);
  const u8 * ptr[LANES];
  u8 want[LANES * XPAR_BLAKE3_OUT_LEN], got[LANES * XPAR_BLAKE3_OUT_LEN];
  st_rng r;
  u32 bad = 0, i;
  const char * name = NULL;
#if defined(HAVE_AVX2) || defined(HAVE_NEON)
  u32 feat = xpar_cpu_features();
#endif

  r.s = 0x452821E638D01377ull;
  st_fill(&r, in, LANES * BLOCKS * XPAR_BLAKE3_BLOCK_LEN);
  for (i = 0; i < LANES; i++)
    ptr[i] = in + (sz) i * BLOCKS * XPAR_BLAKE3_BLOCK_LEN;
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
  if (name) bad += st_cmp(name, "blake3 hash_many", got, want, sizeof got);
  xpar_free(in);
  return bad;
}

static u8 st_hexdigit(char c) {
  if (c >= '0' && c <= '9') return (u8) (c - '0');
  if (c >= 'a' && c <= 'f') return (u8) (c - 'a' + 10);
  return (u8) (c - 'A' + 10);
}

static u32 st_kat_hex(const char * what, const u8 * got, sz n,
                      const char * hex) {
  sz i;
  for (i = 0; i < n; i++) {
    u8 want = (u8) ((st_hexdigit(hex[2 * i]) << 4) |
                    st_hexdigit(hex[2 * i + 1]));
    if (got[i] != want) {
      xpar_fprintf(xpar_stderr,
                   "xpar: selftest: conformance KAT %s differs at byte %lu "
                   "(%02X against %02X).\n", what, (unsigned long) i,
                   got[i], want);
      return 1;
    }
  }
  return 0;
}

/*  Frozen installed KATs pin the published hash, CRC and generation bytes.  */
static u32 st_check_kats(void) {
  static const u32 roll_want[] = {
    0xcf762298u, 0x96fce802u, 0x35c5ff48u, 0x73771252u,
    0xdff7f330u, 0x6f56d7b1u, 0xa1d86dc4u, 0x4ca80c42u,
    0xa2d3bb0cu, 0xd14a6a4cu, 0x2ee14d0bu, 0x8b5fd88eu,
    0xef1bba18u, 0x6b86a270u, 0x3ebe57efu, 0x0363e9c7u,
    0xbf835fabu
  };
  u8 data[20000], setd_body[96], file_body[160];
  u8 content[32], prefix[16], file_id[16], set_id[16], master[32], check[16];
  xpar_entry entry;
  xpar_set_id_ctx set_hash;
  xpar_crc32c_roll roll;
  xpar_armour_params ap;
  xpar_armour * armour;
  u8 frame[255];
  u32 generator[3], crc, bad = 0, i;

  for (i = 0; i < sizeof data; i++) data[i] = (u8) (i * 29 + 7);
  for (i = 0; i < sizeof setd_body; i++)
    setd_body[i] = (u8) (i * 3 + 1);
  for (i = 0; i < sizeof file_body; i++)
    file_body[i] = (u8) (i * 5 + 9);
  xpar_blake3_hash(data, sizeof data, content, sizeof content);
  xpar_blake3_hash(data, 16384, prefix, sizeof prefix);
  bad += st_kat_hex("V-HASH content_hash", content, sizeof content,
                    "f9d161476303e9b8a45d8a4403d6bd5b"
                    "6649ae5a333b1d1787334fcf603f0011");
  bad += st_kat_hex("V-HASH prefix_hash", prefix, sizeof prefix,
                    "a24032354ddaf4559e32caf4f14ba510");
  xpar_memset(&entry, 0, sizeof entry);
  entry.name = (char *) "tree/fixed.bin";
  entry.name_len = 14;
  entry.length = sizeof data;
  xpar_memcpy(entry.prefix_hash, prefix, sizeof prefix);
  xpar_file_id(&entry, NULL, file_id);
  bad += st_kat_hex("V-HASH file_id", file_id, sizeof file_id,
                    "0144119834d4eefb811fb9935c3f7523");
  xpar_set_id_begin(&set_hash, NULL, setd_body, sizeof setd_body);
  xpar_set_id_update(&set_hash, file_body, sizeof file_body);
  xpar_set_id_final(&set_hash, set_id);
  bad += st_kat_hex("V-HASH set_id", set_id, sizeof set_id,
                    "cf2b9c0a22b17377f7873c716ad20c97");
  xpar_key_master(master, "xpar2 conformance key\n", 22);
  xpar_key_check(check, master);
  bad += st_kat_hex("V-HASH key_check", check, sizeof check,
                    "485ae68f1442ed7c0aead7358b86a037");
  xpar_memset(master, 0, sizeof master);

  crc = xpar_crc32c(0, data, 4096);
  if (crc != 0x752b349cu) {
    xpar_fprintf(xpar_stderr,
                 "xpar: selftest: conformance KAT V-CRC slice is %08lX, "
                 "expected 752B349C.\n", (unsigned long) crc);
    bad++;
  }
  xpar_crc32c_roll_init(&roll, 64);
  crc = xpar_crc32c(0, data, 64);
  for (i = 0; i < ARRAY_LEN(roll_want); i++) {
    if (i) crc = xpar_crc32c_roll_step(&roll, crc, data[i - 1],
                                       data[i + 63]);
    if (crc != roll_want[i]) {
      xpar_fprintf(xpar_stderr,
                   "xpar: selftest: conformance KAT V-CRC rolling state "
                   "%lu is %08lX, expected %08lX.\n", (unsigned long) i,
                   (unsigned long) crc, (unsigned long) roll_want[i]);
      bad++;
      break;
    }
  }

  xpar_armour_defaults(&ap, 8);
  ap.n = 255; ap.k = 253; ap.depth = 1;
  armour = xpar_armour_new(&ap);
  for (i = 0; i < 253; i++) frame[i] = (u8) (11 + i * 37);
  frame[253] = frame[254] = 0;
  xpar_armour_generator(armour, generator);
  if (generator[0] != 0x96 || generator[1] != 0x70 || generator[2] != 1) {
    xpar_fprintf(xpar_stderr,
                 "xpar: selftest: conformance KAT V-GEN generator differs.\n");
    bad++;
  }
  xpar_armour_encode_frame(armour, frame);
  bad += st_kat_hex("V-GEN codeword parity", frame + 253, 2, "03fc");
  xpar_armour_free(armour);
  return bad;
}

static void st_rate(const char * tier, const char * operation,
                    u64 bytes, u64 usec) {
  f64 mib_s;
  if (!usec) usec = 1;
  mib_s = ((f64) bytes * 1000000.0) / ((f64) usec * 1048576.0);
  xpar_fprintf(xpar_stderr,
               "xpar: selftest: %-12s %-12s %10llu bytes %8llu us "
               "%9.2f MiB/s\n", tier, operation,
               (unsigned long long) bytes, (unsigned long long) usec, mib_s);
}

static void st_measure_gf(const xpar_gf_kernels * k, const char * tier) {
  const sz n = (sz) 1 << 20;
  const u32 repeat = 8;
  u8 * src = (u8 *) xpar_alloc_aligned(n, 64);
  u8 * dst = (u8 *) xpar_alloc_aligned(n, 64);
  xpar_gf8_coef coef;
  st_rng rng;
  u64 begin, elapsed;
  u32 i;
  rng.s = 0x6A09E667F3BCC909ull;
  st_fill(&rng, src, n);
  xpar_memset(dst, 0, n);
  xpar_gf8_prepare(&coef, 173);
  begin = xpar_usec_now();
  for (i = 0; i < repeat; i++) k->mac8(dst, src, n, &coef);
  elapsed = xpar_usec_now() - begin;
  st_rate(tier, "gf8-mac", (u64) n * repeat, elapsed);
  xpar_free_aligned(src); xpar_free_aligned(dst);
}

static void st_measure_armour(const xpar_armour_params * p,
                              const char * tier, const char * operation) {
  const u32 repeat = 32;
  xpar_armour * a = xpar_armour_new(p);
  u64 plain = xpar_armour_frame_plain(a), disk = xpar_armour_frame_disk(a);
  u8 * frame = (u8 *) xpar_alloc_raw((sz) disk);
  st_rng rng;
  u64 begin, elapsed;
  u32 i;
  rng.s = 0xBB67AE8584CAA73Bull;
  st_fill(&rng, frame, (sz) plain);
  begin = xpar_usec_now();
  for (i = 0; i < repeat; i++) xpar_armour_encode_frame(a, frame);
  elapsed = xpar_usec_now() - begin;
  st_rate(tier, operation, plain * repeat, elapsed);
  xpar_free(frame); xpar_armour_free(a);
}

int xpar_op_selftest(const xpar_options * o) {
  u32 bad = 0, tiers = 0;
  int n, i, saved_gf, saved_arm;
  xpar_armour_params p8, p16;
  u8 * ref8;  u8 * ref16;
  u64 fx8, fx16;
  st_rng r;

  xpar_gf_init();
  xpar_crc32c_init();
  saved_gf  = xpar_gf_tier();
  saved_arm = xpar_armour_tier();

  bad += st_check_kats();
  if (!o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: selftest: V-HASH, V-CRC and V-GEN KATs %s\n",
                 bad ? "failed" : "ok");
  if (o->selftest_tiers && !o->quiet)
    xpar_fprintf(xpar_stderr,
                 "xpar: selftest: tier         operation         bytes"
                 "     time       rate\n");

  n = xpar_gf_tier_count();
  for (i = 0; i < n; i++) {
    if (!o->selftest_tiers && i != saved_gf) continue;
    if (!xpar_gf_tier_usable(i)) {
      if (o->verbose)
        xpar_fprintf(xpar_stderr, "xpar: selftest: gf tier %s is compiled "
                     "and not runnable here; skipped.\n",
                     xpar_gf_tier_name(i));
      continue;
    }
    if (!xpar_gf_use_tier(i)) continue;
    bad += st_check_gf(xpar_gf_active(), xpar_gf_tier_name(i));
    if (o->selftest_tiers && !o->quiet)
      st_measure_gf(xpar_gf_active(), xpar_gf_tier_name(i));
    tiers++;
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: selftest: gf tier %-8s ok\n",
                   xpar_gf_tier_name(i));
  }
  xpar_gf_use_tier(saved_gf);

  /*  A shortened GF(2^16) frame keeps the differential test bounded.  */
  xpar_armour_defaults(&p8, 8);
  p8.k = p8.n - 32;  p8.depth = 8;
  xpar_armour_defaults(&p16, 16);
  p16.n = 4096;  p16.k = 4096 - 16;  p16.depth = 3;
  r.s = 0xC0FFEE123456789ull;
  {
    xpar_armour * a = xpar_armour_new(&p8);
    fx8 = xpar_armour_frame_disk(a);
    xpar_armour_free(a);
    a = xpar_armour_new(&p16);
    fx16 = xpar_armour_frame_disk(a);
    xpar_armour_free(a);
  }
  ref8  = (u8 *) xpar_alloc_raw((sz) fx8);
  ref16 = (u8 *) xpar_alloc_raw((sz) fx16);
  xpar_armour_use_tier(xpar_armour_tier_count() - 1);   /*  Scalar.  */
  st_armour_frame(&p8, ref8, &r);
  st_armour_frame(&p16, ref16, &r);

  n = xpar_armour_tier_count();
  for (i = 0; i < n; i++) {
    if (!o->selftest_tiers && i != saved_arm) continue;
    if (!xpar_armour_tier_usable(i)) continue;
    if (!xpar_armour_use_tier(i)) continue;
    bad += st_check_armour(xpar_armour_tier_name(i), &p8, ref8, fx8,
                           "armour GF(2^8)");
    bad += st_check_armour(xpar_armour_tier_name(i), &p16, ref16, fx16,
                           "armour GF(2^16)");
    if (o->selftest_tiers && !o->quiet) {
      st_measure_armour(&p8, xpar_armour_tier_name(i), "armour-gf8");
      st_measure_armour(&p16, xpar_armour_tier_name(i), "armour-gf16");
    }
    tiers++;
    if (!o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: selftest: armour tier %-8s ok\n",
                   xpar_armour_tier_name(i));
  }
  xpar_armour_use_tier(saved_arm);
  xpar_free(ref8);  xpar_free(ref16);

  bad += st_check_crc32c();
  bad += st_check_blake3();
  if (!o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: selftest: crc32c %s, blake3 %s\n",
                 xpar_crc32c_variant(), xpar_blake3_variant());

  if (bad) {
    xpar_fprintf(xpar_stderr, "xpar: selftest: %lu differences across %lu "
                 "tiers. This build's kernels do not agree on this "
                 "machine.\n", (unsigned long) bad, (unsigned long) tiers);
    gen_json_result(o, "selftest", NULL, 0, "failed", XPAR_EXIT_INTERNAL);
    return XPAR_EXIT_INTERNAL;
  }
  if (!o->quiet)
    xpar_fprintf(xpar_stderr, "xpar: selftest: %lu tiers checked, every "
                 "kernel byte-identical to scalar.\n", (unsigned long) tiers);
  gen_json_result(o, "selftest", NULL, 0, "ok", XPAR_EXIT_OK);
  return XPAR_EXIT_OK;
}
