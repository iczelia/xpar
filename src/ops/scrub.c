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

/*  Scrub checks inner parity and recovery data that verify need not read.  */

#include "ops.h"
#include "chain.h"
#include "volimg.h"
#include "vset.h"

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
#include "slice.h"

#include "platform/port-thread.h"

/*  Scrub state.  */

/*  Counts above this limit remain in totals but not the distribution.  */
#define SCRUB_HIST_MAX  4096u

typedef struct {
  const u8 * data;              /*  Points into a volume image.  */
  u64        exponent;
  bool       seen;
} scrub_rcvs;

typedef struct {
  xpar_vset * s;
  const xpar_options * o;

  /*  Volume images opened here: the recovery volumes LAYT names. Index
      volumes are already mapped by the set reader.  */
  xpar_volimg * rvol;
  u32           rcount;

  scrub_rcvs * rcvs;            /*  Indexed by exponent.  */
  u64          rcvs_count, rcvs_present;
  u64 pkt_bad, pkt_short;       /*  Packets a volume scan rejected.  */

  /*  The inner code, over every codeword and not only the failing ones.  */
  u64   frames, codewords, clean, corrected, failed, symbols;
  u32   worst;
  u64 * hist;
  u32   hist_len;
  u64   regions, regions_rewritten;
  /*  Statistics for wrapped recovery regions.  */
  u64   rec_regions, rec_codewords, rec_corrected, rec_failed, rec_symbols;

  /*  --deep and --rebuild-cells.  */
  u64  rcvs_wrong, rcvs_unchecked;
  u64  cells_rebuilt, cells_unseeded;
  u64  link_drift;
  u64  append_skipped;          /*  Index volumes that would not open.  */
  bool write_failed;
} scrub;

static void take_rcvs(scrub * c, const xpar_pkt * hdr, const u8 * body) {
  const xpar_setd * sd = xpar_vset_setd(c->s);
  xpar_rcvs r;
  if (xpar_memcmp(hdr->set_id, xpar_vset_id(c->s), XPAR_SET_ID_LEN) != 0)
    return;
  if (xpar_rcvs_read(body, (sz) (hdr->length - XPAR_PKT_HDR),
                     sd->slice_size, &r) != XPAR_OK) return;
  if (r.exponent >= c->rcvs_count) return;
  if (c->rcvs[r.exponent].seen) return;
  c->rcvs[r.exponent].seen     = true;
  c->rcvs[r.exponent].data     = r.data;
  c->rcvs[r.exponent].exponent = r.exponent;
  c->rcvs_present++;
}

/*  Collect RCVS packets from decoded plaintext.  */
static void take_rcvs_decoded(scrub * c) {
  const xpar_setd * sd = xpar_vset_setd(c->s);
  u64 i;
  for (i = 0; i < c->rcvs_count; i++) {
    u64 n = 0;
    const u8 * p;
    if (c->rcvs[i].seen) continue;
    p = xpar_vset_rcvs(c->s, i, &n);
    if (!p || n != sd->slice_size) continue;
    c->rcvs[i].seen = true;
    c->rcvs[i].data = p;
    c->rcvs[i].exponent = i;
    c->rcvs_present++;
  }
}

static void scan_image(scrub * c, const u8 * buf, u64 size) {
  xpar_scan sc;
  xpar_pkt hdr;
  const u8 * body;
  u64 off;
  xpar_scan_init(&sc, buf, size, xpar_vset_key(c->s), false);
  sc.accept_unverified_keyed = xpar_vset_key(c->s) == NULL;
  while (xpar_scan_next(&sc, &hdr, &body, &off))
    if (xpar_pkt_is(&hdr, XPAR_T_RCVS)) take_rcvs(c, &hdr, body);
  xpar_reject_unknown_critical(&sc);
  c->pkt_bad   += sc.skip_checksum;
  c->pkt_short += sc.skip_length;
}

static void load_recovery(scrub * c) {
  const xpar_layt * l = xpar_vset_layt(c->s);
  const xpar_setd * sd = xpar_vset_setd(c->s);
  const char * dir = xpar_vset_dir(c->s);
  u32 i;

  c->rcvs_count = xpar_vset_recovery_total(c->s);
  c->rcvs = (scrub_rcvs *) xpar_calloc(c->rcvs_count ? (sz) c->rcvs_count
                                                     : 1,
                                       sizeof(scrub_rcvs));
  /* Scan decoded plaintext, not armoured codewords, for RCVS packets. */
  if (sd->layout == XPAR_LAYOUT_ARMOURED) { take_rcvs_decoded(c);  return; }
  /*  The index volumes may carry recovery slices themselves on a small
      set, so they are scanned before anything is opened.  */
  for (i = 0; i < xpar_vset_volumes(c->s); i++) {
    u64 n = 0;
    const u8 * p = xpar_vset_volume(c->s, i, &n);
    if (p) scan_image(c, p, n);
  }
  if (!l) return;

  c->rvol = (xpar_volimg *) xpar_calloc(l->count ? l->count : 1,
                                        sizeof(xpar_volimg));
  for (i = 0; i < l->count; i++) {
    const xpar_vol * v = &l->vol[i];
    xpar_volimg_status vs;
    char * path;
    int err = 0;
    if (v->kind != XPAR_VOL_RECOVERY || !v->name) continue;
    path = xpar_path_vol(dir, v->name);
    vs = xpar_volimg_read(&c->rvol[c->rcount], path, &err);
    if (vs == XPAR_VOLIMG_IO)
      FATAL_IO("Cannot read recovery volume '%s': %s.", path,
               xpar_strerror(err ? err : xpar_errno()));
    if (vs != XPAR_VOLIMG_OK) {
      xpar_fprintf(xpar_stderr, "xpar: recovery volume '%s' is missing\n",
                   v->name);
      xpar_free(path);
      continue;
    }
    xpar_free(path);
    scan_image(c, c->rvol[c->rcount].data, c->rvol[c->rcount].size);
    c->rcount++;
  }
  take_rcvs_decoded(c);
}

/*  The inner code, over everything.  */

static void hist_add(scrub * c, const xpar_armour_stat * st) {
  c->frames    += st->frames;     c->codewords += st->codewords;
  c->clean     += st->clean;      c->corrected += st->corrected;
  c->failed    += st->failed;     c->symbols   += st->symbols;
  if (st->worst > c->worst) c->worst = st->worst;
  For(u32, i, MIN(st->hist_len, c->hist_len), c->hist[i] += st->hist[i])
}

static void stat_merge(xpar_armour_stat * d, const xpar_armour_stat * s) {
  d->frames    += s->frames;     d->codewords += s->codewords;
  d->clean     += s->clean;      d->corrected += s->corrected;
  d->failed    += s->failed;     d->symbols   += s->symbols;
  if (s->worst > d->worst) d->worst = s->worst;
  if (d->hist && s->hist)
    For(u32, i, MIN(d->hist_len, s->hist_len), d->hist[i] += s->hist[i])
}

/*  Decode independent frame ranges with per-worker codecs.  */

typedef struct {
  const xpar_armour_params * p;
  u8 * region;
  u64  frames, fx;
  sz   chunks;
  xpar_armour_stat * st;
} scrub_frames_job;

static void scrub_frames_run(sz index, void * ctx) {
  scrub_frames_job * j = (scrub_frames_job *) ctx;
  u64 lo = j->frames * (u64) index / (u64) j->chunks;
  u64 hi = j->frames * (u64) (index + 1) / (u64) j->chunks;
  xpar_armour * a;
  if (hi <= lo) return;
  a = xpar_armour_new(j->p);
  xpar_armour_decode_frames(a, j->region + lo * j->fx, hi - lo, &j->st[index]);
  xpar_armour_free(a);
}

static void scrub_frames(scrub * c, const xpar_armour * ar, u8 * region,
                         u64 frames, xpar_armour_stat * st) {
  scrub_frames_job j;
  xpar_pool * pool;
  u64 batch = xpar_armour_batch(ar), fit;
  sz i;
  if (!frames) return;
  fit = frames / (batch ? batch : 1);
  pool = xpar_pool_create(c->o->jobs);
  j.chunks = (sz) xpar_pool_threads(pool);
  /*  Give each worker at least one full batch. */
  if ((u64) j.chunks > fit) j.chunks = (sz) fit;
  if (j.chunks <= 1) {
    xpar_pool_destroy(pool);
    xpar_armour_decode_frames(ar, region, frames, st);
    return;
  }
  j.p = xpar_armour_params_of(ar);
  j.region = region;  j.frames = frames;
  j.fx = xpar_armour_frame_disk(ar);
  j.st = (xpar_armour_stat *) xpar_calloc(j.chunks,
                                          sizeof(xpar_armour_stat));
  for (i = 0; i < j.chunks; i++) {
    j.st[i].hist_len = st->hist_len;
    j.st[i].hist = st->hist_len
                     ? (u64 *) xpar_calloc(st->hist_len, sizeof(u64)) : NULL;
  }
  xpar_pool_run(pool, j.chunks, scrub_frames_run, &j);
  for (i = 0; i < j.chunks; i++) {
    stat_merge(st, &j.st[i]);
    xpar_free(j.st[i].hist);
  }
  xpar_free(j.st);
  xpar_pool_destroy(pool);
}

/*  Lazily opened rewrite handle, flushed once per volume.  */
typedef struct { const char * path;  xpar_file * f;  bool failed; } rw_sink;

static void sink_init(rw_sink * k, const char * path) {
  k->path = path;  k->f = NULL;  k->failed = false;
}

static void sink_write(scrub * c, rw_sink * k, const u8 * p, u64 n,
                       u64 off) {
  if (k->failed || !k->path) return;
  if (!k->f) {
    k->f = xpar_open(k->path, XPAR_O_RDWR);
    if (!k->f) {
      xpar_fprintf(xpar_stderr, "xpar: --rewrite: cannot open '%s': %s\n",
                   k->path, xpar_strerror(xpar_errno()));
      k->failed = true;  c->write_failed = true;
      return;
    }
  }
  if (xpar_pwrite(k->f, p, (sz) n, off) != (sz) n) {
    xpar_fprintf(xpar_stderr, "xpar: --rewrite: short write to '%s' at %"
                 PRIu64 "\n", k->path, off);
    k->failed = true;  c->write_failed = true;
    return;
  }
  c->regions_rewritten++;
}

static void sink_close(scrub * c, rw_sink * k) {
  if (!k->f) return;
  if (xpar_fsync(k->f) != 0) {
    xpar_fprintf(xpar_stderr, "xpar: --rewrite: cannot flush '%s': %s\n",
                 k->path, xpar_strerror(xpar_errno()));
    c->write_failed = true;
  }
  xpar_close(k->f);  k->f = NULL;
}

/*  One armoured packet group: every frame, then the plaintext parse that
    decides whether a rewrite is authorised.  */
static void scrub_armg(scrub * c, const u8 * body, sz n, const u8 * base,
                       rw_sink * k) {
  xpar_armg a;
  xpar_armour_params p;
  xpar_armour * ar;
  xpar_armour_stat st;
  u8 * region, * plain;
  u64 fdisk, nframes;
  bool ok;

  if (xpar_armg_read(body, n, &a) != XPAR_OK) return;
  xpar_memset(&p, 0, sizeof p);
  p.symbol_bits = a.symbol_bits;  p.poly = a.poly;
  p.n     = a.n;    p.k    = a.k;
  p.fcr   = a.fcr;  p.prim = a.prim;
  p.depth = a.depth;
  if (xpar_armour_check(&p)) return;
  ar = xpar_armour_new(&p);
  c->regions++;

  region = (u8 *) xpar_alloc_raw((sz) a.armoured_length);
  xpar_memcpy(region, a.data, (sz) a.armoured_length);
  fdisk   = xpar_armour_frame_disk(ar);
  nframes = fdisk ? a.armoured_length / fdisk : 0;

  /*  A private histogram per region, folded in afterwards, so a region
      whose parameters give a different `t` cannot overrun a shared one.  */
  { u32 t   = (a.n - a.k) / 2;
    u32 len = t + 1 < SCRUB_HIST_MAX ? t + 1 : SCRUB_HIST_MAX;
    u64 * h = (u64 *) xpar_calloc(len, sizeof(u64));
    xpar_memset(&st, 0, sizeof st);
    st.hist = h;  st.hist_len = len;
    scrub_frames(c, ar, region, nframes, &st);
    hist_add(c, &st);
    xpar_free(h); }

  plain = (u8 *) xpar_alloc_raw(a.plain_length ? (sz) a.plain_length : 1);
  xpar_armour_extract(ar, plain, a.plain_length, region);
  ok = xpar_verify_packets_ok(plain, a.plain_length, xpar_vset_key(c->s));
  /*  Track wrapped RCVS damage separately from metadata.  */
  if (a.plain_length >= XPAR_PKT_HDR &&
      !xpar_memcmp(plain, XPAR_PKT_MAGIC, 8) &&
      xpar_rd64(plain + 8) == a.plain_length &&
      !xpar_memcmp(plain + 32, XPAR_T_RCVS, 4)) {
    c->rec_regions++;
    c->rec_codewords += st.codewords;
    c->rec_corrected += st.corrected;
    c->rec_failed    += st.failed;
    c->rec_symbols   += st.symbols;
  }
  xpar_free(plain);

  if (c->o->rewrite && ok && st.symbols)
    sink_write(c, k, region, a.armoured_length, (u64) (a.data - base));
  xpar_free(region);
  xpar_armour_free(ar);
}

/*  The sweep, and not the ordinary scan: an ARMG whose packet checksum
    fails is precisely the case scrub exists to measure, so letting the
    checksum drop it would hide the only damage worth reporting.  */
static void scrub_image(scrub * c, const u8 * buf, u64 size,
                        const char * path) {
  const u8 * body;
  u64 pos = 0, blen = 0;
  rw_sink k;
  sink_init(&k, path);
  while (xpar_verify_next_armg(buf, size, xpar_vset_key(c->s),
                               &pos, &body, &blen))
    scrub_armg(c, body, (sz) blen, buf, &k);
  sink_close(c, &k);
}

/*  Whole-file armour is not an ARMG packet. The 384-byte prologue gives
    the framing, and every frame after it is part of one protected region.
    Scrub deliberately takes all syndromes here, including on a clean
    archive, and authorises --rewrite only after the corrected plaintext
    parses with all packet checks intact.  */
static void scrub_archive(scrub * c, const u8 * buf, u64 size,
                          const char * path) {
  xpar_arm_prologue pr;
  xpar_armour_params p;
  xpar_armour * ar;
  xpar_armour_stat st;
  u8 * region, * plain;
  u64 fd, frames, avail;
  bool ok;
  if (!xpar_garm_is_archive(buf, (sz) size) ||
      !xpar_garm_prologue(buf, (sz) size, &pr, NULL)) return;
  avail = size > 384 ? size - 384 : 0;
  if (avail > pr.armoured_length) avail = pr.armoured_length;
  p.symbol_bits = pr.symbol_bits; p.poly = pr.poly;
  p.n = pr.n; p.k = pr.k; p.fcr = pr.fcr; p.prim = pr.prim;
  p.depth = pr.depth;
  if (xpar_armour_check(&p)) { c->failed++; return; }
  ar = xpar_armour_new(&p);
  fd = xpar_armour_frame_disk(ar);
  frames = fd ? pr.armoured_length / fd : 0;
  region = (u8 *) xpar_alloc_raw((sz) pr.armoured_length);
  plain = (u8 *) xpar_alloc_raw(pr.plain_length ? (sz) pr.plain_length : 1);
  /*  A short file is read as erasures of its missing tail.  */
  xpar_memcpy(region, buf + 384, (sz) avail);
  if (avail < pr.armoured_length)
    xpar_memset(region + avail, 0, (sz) (pr.armoured_length - avail));
  c->regions++;
  {
    u32 t = (p.n - p.k) / 2;
    u32 len = t + 1 < SCRUB_HIST_MAX ? t + 1 : SCRUB_HIST_MAX;
    u64 * h = (u64 *) xpar_calloc(len, sizeof(u64));
    xpar_memset(&st, 0, sizeof st);
    st.hist = h; st.hist_len = len;
    scrub_frames(c, ar, region, frames, &st);
    hist_add(c, &st);
    xpar_free(h);
  }
  xpar_armour_extract(ar, plain, pr.plain_length, region);
  ok = xpar_verify_packets_ok(plain, pr.plain_length,
                              xpar_vset_key(c->s));
  if (c->o->rewrite && ok && st.symbols) {
    rw_sink k;
    sink_init(&k, path);
    sink_write(c, &k, region, pr.armoured_length, 384);
    sink_close(c, &k);
  }
  xpar_free(plain); xpar_free(region); xpar_armour_free(ar);
}

static void scrub_armour(scrub * c) {
  u32 i;
  c->hist_len = SCRUB_HIST_MAX;
  c->hist = (u64 *) xpar_calloc(c->hist_len, sizeof(u64));
  for (i = 0; i < xpar_vset_volumes(c->s); i++) {
    u64 n = 0;
    const u8 * p = xpar_vset_volume(c->s, i, &n);
    if (!p) continue;
    if (xpar_garm_is_archive(p, (sz) n))
      scrub_archive(c, p, n, xpar_vset_volume_path(c->s, i));
    else
      scrub_image(c, p, n, xpar_vset_volume_path(c->s, i));
  }
  for (i = 0; i < c->rcount; i++)
    scrub_image(c, c->rvol[i].data, c->rvol[i].size, c->rvol[i].path);
}

/*  Hard-link structure.  */

typedef struct { u64 dev, ino;  u32 entry;  bool have; } linkid;

static bool link_less(const linkid * a, const linkid * b) {
  if (a->dev != b->dev) return a->dev < b->dev;
  if (a->ino != b->ino) return a->ino < b->ino;
  return a->entry < b->entry;
}

/*  id[] remains entry-sorted until the inode sort.  */
static bool link_find(const linkid * id, u32 n, u32 entry, linkid * out) {
  u32 lo = 0, hi = n;
  while (lo < hi) {
    u32 mid = lo + (hi - lo) / 2;
    if (id[mid].entry == entry) { *out = id[mid];  return true; }
    if (id[mid].entry < entry) lo = mid + 1;  else hi = mid;
  }
  return false;
}

static void link_sift(linkid * a, u32 root, u32 n) {
  while (1) {
    u32 ch = 2 * root + 1, big;
    if (ch >= n) return;
    big = ch;
    if (ch + 1 < n && link_less(&a[ch], &a[ch + 1])) big = ch + 1;
    if (!link_less(&a[root], &a[big])) return;
    { linkid t = a[root];  a[root] = a[big];  a[big] = t; }
    root = big;
  }
}

static void check_links(scrub * c) {
  const xpar_manifest * m = xpar_vset_manifest(c->s);
  const char * dir = xpar_vset_dir(c->s);
  xpar_nameidx nix;
  linkid * id;
  u32 i, n = 0;

  if (!m->count) return;
  id = (linkid *) xpar_calloc(m->count, sizeof(linkid));
  xpar_nameidx_build(m, &nix);
  for (i = 0; i < m->count; i++) {
    const xpar_entry * e = &m->entry[i];
    char * path;
    xpar_stat_t st;
    if (e->entry_type != XPAR_ENTRY_REGULAR &&
        e->entry_type != XPAR_ENTRY_HARDLINK) continue;
    path = xpar_path_join(dir, e->name);
    if (!(xpar_fs_caps(path) & XPAR_FS_LINKID)) { xpar_free(path);  continue; }
    if (xpar_lstat(path, &st) == 0) {
      id[n].dev = st.dev;  id[n].ino = st.ino;
      id[n].entry = i;     id[n].have = true;
      n++;
    }
    xpar_free(path);
  }
  /*  Direction one: the manifest says these names share an inode.  */
  for (i = 0; i < m->count; i++) {
    i64 tgt;
    bool ha, hb;
    linkid ia, ib;
    if (m->entry[i].entry_type != XPAR_ENTRY_HARDLINK) continue;
    tgt = xpar_link_target(m, &nix, i);
    if (tgt < 0) continue;
    xpar_memset(&ia, 0, sizeof ia);  xpar_memset(&ib, 0, sizeof ib);
    ha = link_find(id, n, i, &ia);
    hb = link_find(id, n, (u32) tgt, &ib);
    if (ha && hb && (ia.dev != ib.dev || ia.ino != ib.ino)) {
      c->link_drift++;
      if (!c->o->quiet)
        xpar_fprintf(xpar_stderr,
                     "xpar: link-structure-drift: '%s' is no longer the "
                     "same inode as '%s'\n", m->entry[i].name,
                     m->entry[tgt].name);
    }
  }
  /*  Direction two: names the manifest calls independent that now share
      one. Sorted rather than compared pairwise, since a large tree makes
      the quadratic form the dominant cost of the whole operation.  */
  if (n > 1) {
    for (i = n / 2; i-- > 0;) link_sift(id, i, n);
    for (i = n; i-- > 1;) {
      linkid t = id[0];  id[0] = id[i];  id[i] = t;
      link_sift(id, 0, i);
    }
    for (i = 1; i < n; i++) {
      const xpar_entry * a = &m->entry[id[i - 1].entry];
      const xpar_entry * b = &m->entry[id[i].entry];
      if (id[i].dev != id[i - 1].dev || id[i].ino != id[i - 1].ino) continue;
      if (a->entry_type == XPAR_ENTRY_HARDLINK ||
          b->entry_type == XPAR_ENTRY_HARDLINK) continue;
      c->link_drift++;
      if (!c->o->quiet)
        xpar_fprintf(xpar_stderr,
                     "xpar: link-structure-drift: '%s' and '%s' now share "
                     "one inode\n", a->name, b->name);
    }
  }
  xpar_nameidx_free(&nix);
  xpar_free(id);
}

static void deep(scrub * c) {
  const xpar_setd * sd = xpar_vset_setd(c->s);
  const xpar_geom * g  = xpar_vset_geom(c->s);
  u64 s_count = g->slice_count, r_count = c->rcvs_count;
  u64 budget = c->o->memory ? c->o->memory : xpar_plan_default_memory();
  u64 chunk, off, i, j;
  u8 ** data, ** rec;
  const u8 ** din;
  xpar_codec * codec;

  if (!s_count || !r_count) {
    xpar_fputs("xpar: --deep: nothing to re-encode\n", xpar_stderr);
    return;
  }
  for (i = 0; i < r_count; i++)
    if (!c->rcvs[i].seen) c->rcvs_unchecked++;
  if (!xpar_codec_supports_axis(sd->codec, sd->field_log2, s_count, r_count,
                                sd->recovery_axis_log2))
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "--deep: this build's codec cannot express S = %" PRIu64 " with "
               "R = %" PRIu64 ".", s_count,
               r_count);

  chunk = budget / (s_count + r_count);
  chunk &= ~(u64) 63;
  if (chunk < 64)
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "--deep needs at least %" PRIu64
               " bytes of slice buffers; raise -m.",
               (s_count + r_count) * 64);
  if (chunk > g->slice_size) chunk = g->slice_size;

  data = (u8 **) xpar_calloc((sz) s_count, sizeof(u8 *));
  rec  = (u8 **) xpar_calloc((sz) r_count, sizeof(u8 *));
  din  = (const u8 **) xpar_calloc((sz) s_count, sizeof(u8 *));
  for (i = 0; i < s_count; i++)
    { data[i] = (u8 *) xpar_alloc_raw((sz) chunk);  din[i] = data[i]; }
  for (i = 0; i < r_count; i++)
    rec[i] = (u8 *) xpar_alloc_raw((sz) chunk);
  codec = xpar_codec_new_axis(sd->codec, sd->field_log2, s_count, r_count,
                              sd->recovery_axis_log2);

  { u8 * wrong = (u8 *) xpar_calloc((sz) r_count, 1);
    bool data_lost = false;
    for (off = 0; off < g->slice_size; off += chunk) {
      u64 take = MIN(chunk, g->slice_size - off);
      for (i = 0; i < s_count; i++)
        if (!xpar_vset_read(c->s, g->stream_base + i * g->slice_size + off,
                            data[i], take)) {
          /* Missing data invalidates every parity comparison. */
          xpar_memset(data[i], 0, (sz) take);
          data_lost = true;
        }
      if (data_lost) break;
      xpar_codec_encode(codec, (const u8 * const *) din,
                        (u8 * const *) rec, (sz) take);
      for (j = 0; j < r_count; j++) {
        if (!c->rcvs[j].seen || wrong[j]) continue;
        if (xpar_memcmp(rec[j], c->rcvs[j].data + off, (sz) take) != 0) {
          wrong[j] = 1;
          c->rcvs_wrong++;
          if (!c->o->quiet)
            xpar_fprintf(xpar_stderr,
                         "xpar: recovery slice %" PRIu64 " does not recompute "
                         "from the data\n", j);
        }
      }
    }
    if (data_lost && !c->o->quiet)
      xpar_fprintf(xpar_stderr, "xpar: --deep: protected data is incomplete; "
                   "repair it before checking recovery slices.\n");
    xpar_free(wrong); }

  xpar_codec_free(codec);
  for (i = 0; i < s_count; i++) xpar_free(data[i]);
  for (i = 0; i < r_count; i++) xpar_free(rec[i]);
  xpar_free(data);  xpar_free(rec);  xpar_free((void *) din);
}

/* Buffer one SLCL packet's cell checksums. */

typedef struct {
  xpar_buf out;
  u32 *    crc;
  u64      cap, first, run;
} cellrun;

static void cellrun_flush(cellrun * cr, scrub * c) {
  const xpar_setd * sd = xpar_vset_setd(c->s);
  if (!cr->run) return;
  xpar_slcl_write(&cr->out, cr->first, cr->run, sd->cell_bytes,
                  xpar_vset_geom(c->s)->cells_per_slice, cr->crc,
                  xpar_vset_id(c->s), xpar_vset_key(c->s));
  cr->run = 0;
}

static void rebuild_cells(scrub * c) {
  const xpar_setd * sd = xpar_vset_setd(c->s);
  const xpar_geom * g  = xpar_vset_geom(c->s);
  const xpar_tags * t  = xpar_vset_tags(c->s);
  u32 k = g->cells_per_slice;
  u64 i, need = 0;
  u8 * slice;
  cellrun cr;

  if (!sd->cell_bytes) {
    xpar_fputs("xpar: --rebuild-cells: this set has no cell table\n",
               xpar_stderr);
    return;
  }
  for (i = 0; i < g->slice_count; i++)
    if (!xpar_vset_cell_covered(c->s, i)) need++;
  if (!need) {
    xpar_fputs("xpar: --rebuild-cells: the cell table is already "
               "complete\n", xpar_stderr);
    return;
  }

  cr.cap   = k ? XPAR_TABLE_SPLIT / k : XPAR_TABLE_SPLIT;
  if (!cr.cap) cr.cap = 1;
  cr.first = cr.run = 0;
  cr.crc   = (u32 *) xpar_calloc((sz) (cr.cap * k), 4);
  xpar_buf_init(&cr.out);
  slice = (u8 *) xpar_alloc_raw((sz) g->slice_size);

  /* Flush at coverage gaps, unverified slices and packet limits. */
  for (i = 0; i < g->slice_count; i++) {
    bool ok;
    u32 col;
    if (xpar_vset_cell_covered(c->s, i)) { cellrun_flush(&cr, c);  continue; }
    if (cr.run == cr.cap) cellrun_flush(&cr, c);
    if (!cr.run) cr.first = i;
    xpar_vset_read(c->s, g->stream_base + i * g->slice_size, slice,
                   g->slice_size);
    if (t->tag_len && (xpar_vset_have_tables(c->s) & XPAR_TAGS_TAG)) {
      u8 got[XPAR_BLAKE3_OUT_LEN];
      if (xpar_vset_key(c->s))
        xpar_slice_tag_keyed(sd, i, slice,
                             xpar_vset_key(c->s)->k_slice,
                             got, t->tag_len);
      else
        xpar_slice_tag(sd, i, slice, got, t->tag_len);
      ok = xpar_blake3_tag_equal(got, t->slice_tag + i * t->tag_len,
                                 t->tag_len);
    } else {
      ok = xpar_crc32c(0, slice, (sz) g->slice_size) == t->slice_crc[i];
    }
    if (!ok) {
      c->cells_unseeded++;
      if (!c->o->quiet)
        xpar_fprintf(xpar_stderr,
                     "xpar: --rebuild-cells: slice %" PRIu64 " does not verify "
                     "and cannot seed a cell table\n",
                     i);
      cellrun_flush(&cr, c);
      continue;
    }
    for (col = 0; col < k; col++)
      cr.crc[cr.run * k + col] = xpar_crc32c(0, slice +
                                                (u64) col * sd->cell_bytes,
                                             (sz) xpar_cell_size(g, col));
    cr.run++;
    c->cells_rebuilt += k;
  }
  cellrun_flush(&cr, c);

  if (c->cells_unseeded) {
    xpar_fprintf(xpar_stderr,
                 "xpar: --rebuild-cells: %" PRIu64 " slices could not be seeded\n",
                 c->cells_unseeded);
  } else {
    { u32 v, wrote = 0;
      if (sd->layout == XPAR_LAYOUT_ARMOURED) {
        const u8 * plain;
        u64 plain_len, stream_at, insert;
        xpar_armour_params ap;
        const char * path;
        xpar_scan sc;
        xpar_pkt h;
        const u8 * body;
        u64 off;
        if (!xpar_vset_armoured(c->s, &plain, &plain_len, &stream_at,
                                &ap, &path) ||
            cr.out.len > UINT64_MAX - plain_len) {
          c->write_failed = true;
        } else {
          insert = plain_len;
          xpar_scan_init(&sc, plain, plain_len, xpar_vset_key(c->s), false);
          sc.accept_unverified_keyed = false;
          while (xpar_scan_next(&sc, &h, &body, &off))
            if (xpar_pkt_is(&h, XPAR_T_RCVS) ||
                xpar_pkt_is(&h, XPAR_T_CRTR)) { insert = off; break; }
          xpar_garm_write_inserted(path, &ap, plain, plain_len, insert,
                                   cr.out.data, cr.out.len, stream_at,
                                   g->stream_length);
          wrote = 1;
          xpar_fprintf(xpar_stderr,
                       "xpar: --rebuild-cells: wrote %" PRIu64 " cell checksums "
                       "inside '%s'\n",
                       c->cells_rebuilt, path);
        }
      } else for (v = 0; v < xpar_vset_volumes(c->s) && wrote < 2; v++) {
        const char * path = xpar_vset_volume_path(c->s, v);
        const xpar_layt * layt = xpar_vset_layt(c->s);
        const char * base;
        xpar_file * f;
        bool is_data = false;
        u32 q;
        if (!path) continue;
        for (base = path + xpar_strlen(path); base > path &&
             base[-1] != '/' && base[-1] != '\\'; base--) {}
        if (layt) for (q = 0; q < layt->count; q++)
          if (layt->vol[q].kind == XPAR_VOL_DATA && layt->vol[q].name &&
              xpar_path_same(base, layt->vol[q].name)) {
            is_data = true;
            break;
          }
        if (is_data) continue;
        f = xpar_open(path, XPAR_O_WRONLY | XPAR_O_APPEND);
        if (!f) {
          c->append_skipped++;
          xpar_fprintf(xpar_stderr, "xpar: --rebuild-cells: cannot append "
                       "to '%s': %s\n", path,
                       xpar_strerror(xpar_errno()));
          continue;
        }
        if (xpar_write(f, cr.out.data, cr.out.len) != cr.out.len ||
            xpar_fsync(f) != 0) {
          c->write_failed = true;
          xpar_close(f);
          continue;
        }
        xpar_xclose(f);
        wrote++;
        xpar_fprintf(xpar_stderr,
                     "xpar: --rebuild-cells: wrote %" PRIu64 " cell checksums to "
                     "'%s'\n", c->cells_rebuilt, path);
      }
      if (!wrote) {
        c->write_failed = true;
        xpar_fputs("xpar: --rebuild-cells: no index volume could be "
                   "opened for writing\n", xpar_stderr);
      } }
  }
  xpar_buf_free(&cr.out);
  xpar_free(slice);  xpar_free(cr.crc);
}

/*  Reporting.  */

static void report(const scrub * c, int rc) {
  u32 i;
  if (c->o->quiet) return;
  xpar_fprintf(xpar_stderr,
               "xpar: recovery: %" PRIu64 " slices named, %" PRIu64 " present, %" PRIu64 " "
               "packets failed their checksum\n",
               c->rcvs_count,
               c->rcvs_present,
               c->pkt_bad);
  if (c->pkt_short)
    xpar_fprintf(xpar_stderr,
                 "xpar: recovery: %" PRIu64 " packets had invalid lengths\n",
                 c->pkt_short);
  if (c->regions) {
    xpar_fprintf(xpar_stderr,
                 "xpar: inner code: %" PRIu64 " regions, %" PRIu64 " codewords, %" PRIu64 " "
                 "clean, %" PRIu64 " corrected, %" PRIu64 " past capacity\n",
                 c->regions,
                 c->codewords,
                 c->clean,
                 c->corrected,
                 c->failed);
    xpar_fprintf(xpar_stderr,
                 "xpar: corrected symbols: %" PRIu64 " total, worst codeword %" PRIu32 "\n",
                 c->symbols,
                 c->worst);
    if (c->rec_regions)
      xpar_fprintf(xpar_stderr,
                   "xpar: recovery regions: %" PRIu64 " regions, %" PRIu64
                   " codewords, %" PRIu64 " corrected, %" PRIu64
                   " symbols, %" PRIu64 " past capacity\n",
                   c->rec_regions,
                   c->rec_codewords,
                   c->rec_corrected,
                   c->rec_symbols,
                   c->rec_failed);
    for (i = 1; i < c->hist_len; i++)
      if (c->hist[i])
        xpar_fprintf(xpar_stderr,
                     "xpar:   codewords corrected at %" PRIu32 " symbols: %" PRIu64 "\n",
                     i, c->hist[i]);
  }
  if (c->regions_rewritten)
    xpar_fprintf(xpar_stderr, "xpar: --rewrite: refreshed %" PRIu64
                 " regions\n",
                 c->regions_rewritten);
  /*  --rewrite does not re-encode missing recovery slices.  */
  if (c->o->rewrite && (c->pkt_bad || c->rcvs_present < c->rcvs_count))
    xpar_fprintf(xpar_stderr,
                 "xpar: --rewrite does not rebuild recovery slices; run "
                 "`xpar repair` to regenerate them from the data\n");
  if (c->rcvs_wrong)
    xpar_fprintf(xpar_stderr,
                 "xpar: --deep: %" PRIu64 " recovery slices do not recompute "
                 "from the data\n", c->rcvs_wrong);
  if (c->rcvs_unchecked)
    xpar_fprintf(xpar_stderr,
                 "xpar: --deep: %" PRIu64 " recovery slices were not present and "
                 "could not be compared\n",
                 c->rcvs_unchecked);
  if (c->link_drift)
    xpar_fprintf(xpar_stderr, "xpar: %" PRIu64
                 " link-structure-drift reports\n",
                 c->link_drift);
  if (c->append_skipped)
    xpar_fprintf(xpar_stderr, "xpar: --rebuild-cells: %" PRIu64 " index "
                 "volume%s could not be opened for writing\n",
                 c->append_skipped, PLURAL(c->append_skipped));
  xpar_fprintf(xpar_stderr, "xpar: scrub: exit %d\n", rc);
}

static int scrub_one(const xpar_options * o, xpar_vset * opened,
                     xpar_json * shared, bool summary) {
  scrub c;
  xpar_json local;
  xpar_json * js = shared ? shared : &local;
  int rc;

  FATAL_UNLESS("scrub cannot read a pipe; use --spool.", !o->from_stdin);

  xpar_memset(&c, 0, sizeof c);
  c.o = o;
  c.s = opened ? opened : xpar_vset_open(o);
  if (!shared) xpar_json_init(js, xpar_stdout, o->json);
  if (o->json) xpar_vset_json_set(c.s, js);
  if ((o->rewrite || o->rebuild_cells) &&
      xpar_vset_authenticated(c.s) && !xpar_vset_key(c.s))
    FATAL_CODE(XPAR_EXIT_AUTH,
               "Writing this set requires --auth-key=FILE.");

  if ((o->rewrite || o->rebuild_cells) && !o->force &&
      xpar_vset_setd(c.s)->slice_tag_len == 0)
    FATAL("Writing a set without slice tags requires -f.");

  rc = xpar_vset_check(c.s, o, o->json ? js : NULL);
  xpar_vset_report(c.s, o, rc);

  load_recovery(&c);
  scrub_armour(&c);
  check_links(&c);
  if (o->deep) deep(&c);
  if (o->rebuild_cells) rebuild_cells(&c);

  /*  Parity-side rot and a recovery slice that was computed wrong are
      both repairable conditions and neither is data loss, so they raise a
      clean verdict to "there is work to do" and never past it.  */
  if (c.write_failed || xpar_vset_io_errors(c.s)) rc = XPAR_EXIT_IO;
  else if (rc == XPAR_EXIT_OK &&
      (c.pkt_bad || c.rcvs_wrong || c.failed ||
       c.rcvs_present < c.rcvs_count || c.cells_unseeded))
    rc = XPAR_EXIT_REPAIRABLE;
  report(&c, rc);

  if (o->json) {
    xpar_json_begin(js, "scrub");
    xpar_json_u64(js, "generation", xpar_vset_setd(c.s)->generation);
    xpar_json_u64(js, "recovery_named", c.rcvs_count);
    xpar_json_u64(js, "recovery_present", c.rcvs_present);
    xpar_json_u64(js, "packets_bad", c.pkt_bad);
    xpar_json_u64(js, "packets_short", c.pkt_short);
    xpar_json_u64(js, "codewords", c.codewords);
    xpar_json_u64(js, "codewords_clean", c.clean);
    xpar_json_u64(js, "codewords_corrected", c.corrected);
    xpar_json_u64(js, "codewords_failed", c.failed);
    xpar_json_u64(js, "symbols_corrected", c.symbols);
    xpar_json_u64(js, "recovery_regions", c.rec_regions);
    xpar_json_u64(js, "recovery_codewords", c.rec_codewords);
    xpar_json_u64(js, "recovery_corrected", c.rec_corrected);
    xpar_json_u64(js, "recovery_symbols_corrected", c.rec_symbols);
    xpar_json_u64(js, "recovery_codewords_failed", c.rec_failed);
    xpar_json_u64(js, "worst_codeword", c.worst);
    xpar_json_u64(js, "recovery_wrong", c.rcvs_wrong);
    xpar_json_u64(js, "link_drift", c.link_drift);
    xpar_json_u64(js, "index_volumes_unwritable", c.append_skipped);
    xpar_json_u64(js, "cells_rebuilt", c.cells_rebuilt);
    xpar_json_u64(js, "syndromes", xpar_verify_syndromes());
    xpar_json_end(js);
    if (summary) xpar_vset_json_summary(c.s, js, rc);
    else {
      xpar_json_begin(js, "generation_result");
      xpar_json_u64(js, "generation", xpar_vset_setd(c.s)->generation);
      xpar_json_i64(js, "exit", rc);
      xpar_json_end(js);
    }
  }

  { u32 i;
    for (i = 0; i < c.rcount; i++) xpar_volimg_close(&c.rvol[i]); }
  xpar_free(c.rvol);
  xpar_free(c.rcvs);  xpar_free(c.hist);
  /*  Chain walks close shared heads after all generations.  */
  if (!opened) xpar_vset_close(c.s);
  return rc;
}

int xpar_op_scrub(const xpar_options * o) {
  int rc;
  xpar_vset * only = NULL;
  bool walk = o->chain;

  /*  Scrub unselected ancestry.  */
  if (!walk && !o->gen_count && !o->from_stdin) {
    only = xpar_vset_open(o);
    if (xpar_vset_setd(only)->generation) {
      xpar_vset_close(only);
      only = NULL;
      walk = true;
    }
  }
  if (!walk) {
    rc = scrub_one(o, only, NULL, true);
    if (only) xpar_vset_close(only);
    return rc;
  }

  {
    /*  The public reader does not expose the generation table, so use the
        chain reader for the ordered generation numbers and the set reader
        for each actual scrub.  */
    xpar_chain c;
    xpar_vset * head;
    int worst = XPAR_EXIT_OK;
    u32 g, selected, at, walked = 0;
    u8 * member;
    xpar_options top = *o;
    xpar_options metadata = *o;
    xpar_json js;
    xpar_genref top_ref;
    char top_id[XPAR_SET_ID_LEN * 2 + 1];
    metadata.chain_metadata_only = true;
    xpar_json_init(&js, xpar_stdout, o->json);
    xpar_gchain_load(&metadata, &c);
    selected = xpar_gchain_select(&c,
                                  o->gen_count ? &o->gens[0] : NULL);
    member = (u8 *) xpar_calloc(c.gen_count, 1);
    for (at = selected; at != XPAR_GEN_NONE && walked++ < c.gen_count;
         at = c.gen[at].parent) member[at] = 1;
    FATAL_UNLESS("The selected generation's ancestry is cyclic.",
                 at == XPAR_GEN_NONE);
    xpar_gchain_genref(&c, selected, &top_ref, top_id);
    top.chain = false;
    top.gens = &top_ref;
    top.gen_count = 1;
    head = xpar_vset_open(&top);
    for (g = 0; g < c.gen_count; g++) {
      xpar_options one = *o;
      xpar_genref ref;
      char id[XPAR_SET_ID_LEN * 2 + 1];
      xpar_vset * current;
      if (!member[g]) continue;
      xpar_gchain_genref(&c, g, &ref, id);
      one.chain = false;
      one.chain_member = true;
      one.gens = &ref;
      one.gen_count = 1;
      current = g == selected ? head : xpar_vset_open(&one);
      if (current != head) xpar_vset_mark_superseded(current, head);
      rc = scrub_one(&one, current, o->json ? &js : NULL, false);
      if (rc > worst) worst = rc;
      if (current != head) xpar_vset_close(current);
    }
    xpar_vset_close(head);
    xpar_free(member);
    xpar_gchain_free(&c);
    if (o->json)
      xpar_json_summary(&js, xpar_status_word(worst), worst);
    return worst;
  }
}
