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

/*  The list, info, and explain reporting verbs.  */

#include "ops.h"
#include "chain.h"

#include "armour.h"
#include "blake3.h"
#include "codec.h"
#include "container.h"
#include "json.h"
#include "manifest.h"
#include "pathname.h"
#include "plan.h"
#include "port-fs.h"
#include "slice.h"

#define ARM_HDR_EXPLAIN 384

/*  The packet volumes in xpar_chain are found by their set identity. Bare
    split volumes have no packet to scan, so report them by the content
    identity LAYT gives them, including a renamed match on the same shelf.  */
static char * li_data_present(const xpar_chain * c, const xpar_vol * v) {
  char * path;
  xpar_dir * d;
  const xpar_dirent * de;
  if (!v->name) return NULL;
  path = xpar_path_join(c->dir, v->name);
  if (xpar_vol_tag_match(path, v)) return path;
  xpar_free(path);
  if (!v->vol_tag || !(d = xpar_opendir(c->dir))) return NULL;
  while ((de = xpar_readdir(d)) != NULL) {
    if (!de->is_regular || !xpar_strcmp(de->name, v->name)) continue;
    path = xpar_path_join(c->dir, de->name);
    if (xpar_vol_tag_match(path, v)) { xpar_closedir(d); return path; }
    xpar_free(path);
  }
  xpar_closedir(d);
  return NULL;
}

/*  Binary prefixes, because every size here is a count of bytes chosen
    as a power of two and a decimal prefix would make Z = 2.5 MiB print
    as 2.6 MB.  */
static const char * li_size(char * buf, sz cap, u64 v) {
  static const char * unit[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };
  int u = 0;
  f64 x = (f64) v;
  while (x >= 1024.0 && u < 5) { x /= 1024.0;  u++; }
  if (!u) xpar_snprintf(buf, cap, "%" PRIu64 " B", v);
  else    xpar_snprintf(buf, cap, "%.1f %s", x, unit[u]);
  return buf;
}

/*  Convert days to a civil date without host time functions.  */
static void li_time(char * buf, sz cap, i64 ns) {
  i64 s, days, rem;
  i64 era, doe, yoe, y, doy, mp, d, mo;
  if (ns == XPAR_ABSENT_TIME) {
    xpar_snprintf(buf, cap, "%-20s", "-");  return;
  }
  s = ns / 1000000000;
  rem = ns % 1000000000;
  if (rem < 0) s--;
  days = s / 86400;
  rem  = s % 86400;
  if (rem < 0) { rem += 86400;  days--; }
  days += 719468;
  era = (days >= 0 ? days : days - 146096) / 146097;
  doe = days - era * 146097;
  yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y   = yoe + era * 400;
  doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  mp  = (5 * doy + 2) / 153;
  d   = doy - (153 * mp + 2) / 5 + 1;
  mo  = mp + (mp < 10 ? 3 : -9);
  if (mo <= 2) y++;
  xpar_snprintf(buf, cap, "%04" PRId64 "-%02" PRId64 "-%02" PRId64 "T%02"
                PRId64 ":%02" PRId64 ":%02" PRId64 "Z",
                y, mo, d,
                (rem / 3600), ((rem / 60) % 60),
                (rem % 60));
}

static char li_type(const xpar_entry * e) {
  switch (e->entry_type) {
    case XPAR_ENTRY_DIR:      return 'd';
    case XPAR_ENTRY_SYMLINK:  return 'l';
    case XPAR_ENTRY_HARDLINK: return 'h';
    default: break;
  }
  return 'f';
}

static const char * li_codec(u8 c) {
  return c == XPAR_CODEC_FFT_LOW ? "fft-low"
       : c == XPAR_CODEC_FFT ? "fft" : "matrix";
}

static const char * li_layout(u8 l) {
  switch (l) {
    case XPAR_LAYOUT_SPLIT:    return "split";
    case XPAR_LAYOUT_ARMOURED: return "armoured";
    default: break;
  }
  return "sidecar";
}

static u64 li_extent_refs(const xpar_manifest * m, const xpar_extent * want) {
  u64 refs = 0;
  u32 i, k;
  for (i = 0; i < m->count; i++)
    for (k = 0; k < m->entry[i].extent_count; k++) {
      const xpar_extent * e = &m->entry[i].extents[k];
      if (e->stream_offset == want->stream_offset &&
          e->length == want->length) refs++;
    }
  return refs;
}

int xpar_op_list(const xpar_options * o) {
  xpar_chain c;
  xpar_json js;
  u32 sel, g, i;
  u8 * member;
  char idbuf[XPAR_SET_ID_LEN * 2 + 1];

  xpar_gchain_load(o, &c);
  sel = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  xpar_json_init(&js, xpar_stdout, o->json);
  member = (u8 *) xpar_calloc(c.gen_count, 1);
  if (o->chain) {
    u32 at = sel, walked = 0;
    while (at != XPAR_GEN_NONE && walked++ < c.gen_count) {
      member[at] = 1;
      at = c.gen[at].parent;
    }
    FATAL_UNLESS("The selected generation's ancestry is cyclic.",
                 at == XPAR_GEN_NONE);
  } else member[sel] = 1;

  for (g = 0; g < c.gen_count; g++) {
    xpar_manifest m;
    u32 * owner = NULL;
    xpar_posix_rec * posix = NULL;
    u32 pcount = 0;
    if (!member[g]) continue;
    xpar_gchain_manifest(&c, g, &m, &owner);
    if (o->verbose) pcount = xpar_gchain_posix(&c, g, &posix);

    xpar_hex(idbuf, c.gen[g].set_id, XPAR_SET_ID_LEN);
    if (o->json) {
      xpar_json_begin(&js, "set");
      xpar_json_u64(&js, "schema", XPAR_JSON_SCHEMA);
      xpar_json_str(&js, "set_id", idbuf);
      xpar_json_u64(&js, "generation", c.gen[g].sd.generation);
      xpar_json_u64(&js, "files", m.count);
      xpar_json_end(&js);
    } else {
      xpar_fprintf(xpar_stdout, "generation %" PRIu32 "  set %s  %" PRIu32
                   " entries\n",
                   c.gen[g].sd.generation, idbuf, m.count);
      xpar_fprintf(xpar_stdout,
                   "  t %12s  gen  mode   %-20s  name\n", "size", "mtime");
    }

    for (i = 0; i < m.count; i++) {
      const xpar_entry * e = &m.entry[i];
      char tbuf[40], mbuf[8];
      if (o->list_links && !o->json &&
          e->entry_type == XPAR_ENTRY_HARDLINK) continue;
      li_time(tbuf, sizeof tbuf, e->mtime_ns);
      if (e->mode == XPAR_ABSENT_U32) xpar_snprintf(mbuf, sizeof mbuf, "-");
      else xpar_snprintf(mbuf, sizeof mbuf, "%04o",
                         (unsigned) (e->mode & 07777));
      if (o->json) {
        xpar_json_begin(&js, "file");
        xpar_json_u64(&js, "index", i);
        xpar_json_strn(&js, "name", e->name, e->name_len);
        xpar_json_u64(&js, "length", e->length);
        xpar_json_u64(&js, "entry_type", e->entry_type);
        xpar_json_u64(&js, "owner_generation",
                      c.gen[owner[i]].sd.generation);
        xpar_json_hex(&js, "content_hash", e->content_hash, 32);
        xpar_json_hex(&js, "file_id", e->file_id, XPAR_SET_ID_LEN);
        xpar_json_u64(&js, "extents", e->extent_count);
        if (e->entry_type == XPAR_ENTRY_HARDLINK ||
            e->entry_type == XPAR_ENTRY_SYMLINK)
          xpar_json_strn(&js, "target", (const char *) e->extra,
                         e->extra_len);
        xpar_json_end(&js);
        continue;
      }
      xpar_fprintf(xpar_stdout, "  %c %12" PRIu64 "  %3" PRIu32
                   "  %-5s  %-20s  %.*s",
                   li_type(e), e->length,
                   c.gen[owner[i]].sd.generation, mbuf, tbuf,
                   (int) e->name_len, e->name);
      if (e->extra_len && (e->entry_type == XPAR_ENTRY_HARDLINK ||
                           e->entry_type == XPAR_ENTRY_SYMLINK))
        xpar_fprintf(xpar_stdout, " -> %.*s", (int) e->extra_len,
                     (const char *) e->extra);
      xpar_fputs("\n", xpar_stdout);
      if (o->verbose || o->list_dedup) {
        u32 k;
        char hb[65];
        if (o->verbose) {
          xpar_hex(hb, e->content_hash, 32);
          xpar_fprintf(xpar_stdout, "      hash %s\n", hb);
        }
        for (k = 0; k < e->extent_count; k++) {
          i64 h = xpar_gchain_gen_of(&c, e->extents[k].stream_offset,
                                     e->extents[k].length);
          char gb[32];
          /*  An extent may belong to a different generation than its entry.  */
          if (h < 0) xpar_snprintf(gb, sizeof gb, "outside the chain");
          else xpar_snprintf(gb, sizeof gb, "generation %" PRIu32,
                             c.gen[h].sd.generation);
          if (o->list_dedup)
            xpar_fprintf(xpar_stdout,
                         "      extent %" PRIu64 " + %" PRIu64 "  in %s  refs=%" PRIu64 "\n",
                         e->extents[k].stream_offset,
                         e->extents[k].length, gb,

                           li_extent_refs(&m, &e->extents[k]));
          else
            xpar_fprintf(xpar_stdout, "      extent %" PRIu64 " + %" PRIu64
                         "  in %s\n",
                         e->extents[k].stream_offset,
                         e->extents[k].length, gb);
        }
        if (e->posix_index != XPAR_ABSENT_U32 && e->posix_index < pcount) {
          const xpar_posix_rec * r = &posix[e->posix_index];
          xpar_fprintf(xpar_stdout,
                       "      owner %s:%s (%" PRIu32 ":%" PRIu32 "), %" PRIu32 " xattrs\n",
                       r->owner ? r->owner : "-", r->group ? r->group : "-",
                       r->uid, r->gid,
                       r->xattr_count);
        }
      }
      if (o->list_links && e->entry_type != XPAR_ENTRY_HARDLINK) {
        u32 a;
        for (a = 0; a < m.count; a++) {
          const xpar_entry * al = &m.entry[a];
          if (al->entry_type != XPAR_ENTRY_HARDLINK ||
              al->extra_len != e->name_len ||
              xpar_memcmp(al->extra, e->name, e->name_len)) continue;
          xpar_fprintf(xpar_stdout, "      hard-link %.*s -> %.*s\n",
                       (int) al->name_len, al->name,
                       (int) e->name_len, e->name);
        }
      }
    }
    if (posix) xpar_gchain_posix_free(posix, pcount);
    xpar_free(owner);
    xpar_manifest_free(&m);
  }
  if (o->json) xpar_json_summary(&js, "ok", XPAR_EXIT_OK);
  xpar_free(member);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

/*  Read the stored armour parameters from ARMG or the archive prologue.  */
static bool li_armour_of(const xpar_chain * c, u32 g, xpar_armour_params * p,
                         bool * whole_file) {
  u32 i;
  *whole_file = false;
  for (i = 0; i < c->vol_count; i++) {
    xpar_scan sc;  xpar_pkt hdr;  const u8 * body;  u64 off;
    xpar_arm_prologue pr;
    if (c->vol[i].gen != g) continue;
    if (c->vol[i].armoured_file &&
        xpar_garm_prologue(c->vol[i].data, c->vol[i].len, &pr, NULL)) {
      p->symbol_bits = pr.symbol_bits;  p->poly = pr.poly;
      p->n = pr.n;  p->k = pr.k;  p->fcr = pr.fcr;  p->prim = pr.prim;
      p->depth = pr.depth;
      *whole_file = true;
      return true;
    }
    xpar_scan_init(&sc, c->vol[i].data, c->vol[i].len, NULL, false);
    while (xpar_scan_next(&sc, &hdr, &body, &off)) {
      xpar_armg ag;
      if (!xpar_pkt_is(&hdr, XPAR_T_ARMG)) continue;
      if (xpar_armg_read(body, (sz) (hdr.length - XPAR_PKT_HDR), &ag) !=
          XPAR_OK) continue;
      p->symbol_bits = ag.symbol_bits;  p->poly = ag.poly;
      p->n = ag.n;  p->k = ag.k;  p->fcr = ag.fcr;  p->prim = ag.prim;
      p->depth = ag.depth;
      return true;
    }
  }
  return false;
}

static void li_chain_table(const xpar_chain * c, u32 sel) {
  u32 g;
  char idbuf[XPAR_SET_ID_LEN * 2 + 1], sbuf[32];
  xpar_fprintf(xpar_stdout, "  chain      : %" PRIu32 " generation%s\n", c->gen_count,
               c->gen_count == 1 ? "" : "s");
  for (g = 0; g < c->gen_count; g++) {
    u64 s = c->gen[g].sd.data_slice_count;
    xpar_hex(idbuf, c->gen[g].set_id, 4);
    xpar_fprintf(xpar_stdout,
                 "    gen %-3" PRIu32 "  set %s...  %-10s  S=%" PRIu64 " R=%" PRIu64 " (%.1f%%)  "
                 "volumes %" PRIu32 "%s\n", c->gen[g].sd.generation, idbuf,
                 li_size(sbuf, sizeof sbuf, c->gen[g].sd.stream_length),
                 s,
                 c->gen[g].recovery_count,
                 s ? 100.0 * (f64) c->gen[g].recovery_count / (f64) s : 0.0,
                 c->gen[g].vol_count, g == sel ? "  <- selected" : "");
    if (c->gen[g].parent_missing)
      xpar_fprintf(xpar_stdout, "             parent is not present; this "
                   "chain is truncated below here\n");
    if (!c->gen[g].recovery_count && c->gen[g].sd.stream_length)
      xpar_fprintf(xpar_stdout,
                   "             no recovery data for this generation is "
                   "present; no other\n             generation's recovery "
                   "volumes can substitute\n");
  }
  if (c->gen_count > 1)
    xpar_fprintf(xpar_stdout,
                 "  note       : redundancy is per generation. One "
                 "generation's recovery\n"
                 "               volumes cannot repair another's stream.\n"
                 "               `xpar consolidate` collapses the chain "
                 "into one generation.\n");
  if (c->forked)
    xpar_fprintf(xpar_stdout,
                 "  warning    : two generations name the same parent, so "
                 "this chain has\n               forked; --generation "
                 "selects between them.\n");
}

static void li_deps(const xpar_chain * c, u32 sel) {
  xpar_manifest m;
  u32 * owner = NULL;
  u64 * ext;  u64 * pkt;
  u32 g, i, k;

  xpar_gchain_manifest(c, sel, &m, &owner);
  ext = (u64 *) xpar_calloc(c->gen_count, sizeof(u64));
  pkt = (u64 *) xpar_calloc(c->gen_count, sizeof(u64));
  xpar_gchain_deps(c, &m, owner, ext, pkt);

  xpar_fprintf(xpar_stdout,
               "  deps       : what generation %" PRIu32 "'s manifest would lose if "
               "a generation\n               were removed from the chain\n"
               "    gen  stream bytes  owns packets  entries using  "
               "would be lost\n", c->gen[sel].sd.generation);
  for (g = 0; g < c->gen_count; g++) {
    u64 lost = 0;
    for (i = 0; i < m.count; i++) {
      bool hit = owner[i] == g;
      for (k = 0; k < m.entry[i].extent_count && !hit; k++) {
        i64 h = xpar_gchain_gen_of(c, m.entry[i].extents[k].stream_offset,
                                   m.entry[i].extents[k].length);
        if (h == (i64) g) hit = true;
      }
      if (hit) lost++;
    }
    xpar_fprintf(xpar_stdout,
                 "    %-3" PRIu32 "  %12" PRIu64 "  %12" PRIu64 "  %13" PRIu64 "  %13" PRIu64 "\n",
                 c->gen[g].sd.generation,
                 c->gen[g].sd.stream_length,
                 pkt[g], ext[g],
                 lost);
  }
  xpar_fprintf(xpar_stdout,
               "               `prune` refuses to remove a generation whose "
               "last column\n               is non-zero. That count comes "
               "from the extents recorded in\n               the manifest, "
               "not from --dedup-scope.\n");
  xpar_free(ext);  xpar_free(pkt);  xpar_free(owner);
  xpar_manifest_free(&m);
}

int xpar_op_info(const xpar_options * o) {
  xpar_chain c;
  xpar_json js;
  u32 sel, i;
  char idbuf[XPAR_SET_ID_LEN * 2 + 1], sbuf[32], sbuf2[32];
  const xpar_setd * sd;
  xpar_armour_params ap;
  bool whole_file = false, have_armour;
  xpar_layt layt;
  bool have_layt = false;
  u64 crit_bytes = 0;

  xpar_gchain_load(o, &c);
  sel = xpar_gchain_select(&c, o->gen_count ? &o->gens[0] : NULL);
  if (o->chain && c.gen_count > 1) {
    u32 * number = (u32 *) xpar_calloc(c.gen_count, sizeof(u32));
    u8 * member = (u8 *) xpar_calloc(c.gen_count, 1);
    u32 count = 0, g, at = sel, walked = 0;
    while (at != XPAR_GEN_NONE && walked++ < c.gen_count) {
      member[at] = 1;
      at = c.gen[at].parent;
    }
    FATAL_UNLESS("The selected generation's ancestry is cyclic.",
                 at == XPAR_GEN_NONE);
    if (o->json) {
      xpar_json_init(&js, xpar_stdout, true);
      for (g = 0; g < c.gen_count; g++) if (member[g]) {
        const xpar_setd * gs = &c.gen[g].sd;
        xpar_hex(idbuf, c.gen[g].set_id, XPAR_SET_ID_LEN);
        xpar_json_begin(&js, "set");
        xpar_json_u64(&js, "schema", XPAR_JSON_SCHEMA);
        xpar_json_str(&js, "set_id", idbuf);
        xpar_json_u64(&js, "generation", gs->generation);
        xpar_json_u64(&js, "generations", count ? count : walked);
        xpar_json_u64(&js, "slice_size", gs->slice_size);
        xpar_json_u64(&js, "slices", gs->data_slice_count);
        xpar_json_u64(&js, "stream_base", gs->stream_base);
        xpar_json_u64(&js, "stream_length", gs->stream_length);
        xpar_json_u64(&js, "recovery", c.gen[g].recovery_count);
        xpar_json_u64(&js, "field", gs->field_log2);
        xpar_json_str(&js, "codec", li_codec(gs->codec));
        xpar_json_str(&js, "layout", li_layout(gs->layout));
        xpar_json_u64(&js, "files", gs->file_count);
        xpar_json_end(&js);
      }
      xpar_json_summary(&js, "ok", XPAR_EXIT_OK);
      xpar_free(number);  xpar_free(member);
      xpar_gchain_free(&c);
      return XPAR_EXIT_OK;
    }
    for (g = 0; g < c.gen_count; g++)
      if (member[g]) number[count++] = c.gen[g].sd.generation;
    xpar_free(member);
    xpar_gchain_free(&c);
    for (g = 0; g < count; g++) {
      xpar_options one = *o;
      xpar_genref ref;
      xpar_memset(&ref, 0, sizeof ref);
      ref.number = number[g];
      one.chain = false;
      one.gens = &ref;
      one.gen_count = 1;
      xpar_op_info(&one);
    }
    xpar_free(number);
    return XPAR_EXIT_OK;
  }
  sd  = &c.gen[sel].sd;
  xpar_hex(idbuf, c.gen[sel].set_id, XPAR_SET_ID_LEN);
  have_armour = li_armour_of(&c, sel, &ap, &whole_file);
  if (c.gen[sel].layt_body &&
      xpar_layt_read(c.gen[sel].layt_body, c.gen[sel].layt_len, &layt) ==
        XPAR_OK) have_layt = true;

  if (o->json) {
    xpar_json_init(&js, xpar_stdout, true);
    xpar_json_begin(&js, "set");
    xpar_json_u64(&js, "schema", XPAR_JSON_SCHEMA);
    xpar_json_str(&js, "set_id", idbuf);
    xpar_json_u64(&js, "generation", sd->generation);
    xpar_json_u64(&js, "generations", c.gen_count);
    xpar_json_u64(&js, "slice_size", sd->slice_size);
    xpar_json_u64(&js, "slices", sd->data_slice_count);
    xpar_json_u64(&js, "stream_base", sd->stream_base);
    xpar_json_u64(&js, "stream_length", sd->stream_length);
    xpar_json_u64(&js, "recovery", c.gen[sel].recovery_count);
    xpar_json_u64(&js, "recovery_axis",
                  (u64) 1 << sd->recovery_axis_log2);
    xpar_json_u64(&js, "recovery_limit", xpar_setd_recovery_limit(sd));
    xpar_json_u64(&js, "field", sd->field_log2);
    xpar_json_str(&js, "codec", li_codec(sd->codec));
    xpar_json_str(&js, "layout", li_layout(sd->layout));
    xpar_json_u64(&js, "cell_bytes", sd->cell_bytes);
    xpar_json_u64(&js, "slice_tag_len", sd->slice_tag_len);
    xpar_json_u64(&js, "files", sd->file_count);
    xpar_json_end(&js);
    xpar_json_summary(&js, "ok", XPAR_EXIT_OK);
    if (have_layt) xpar_layt_free(&layt);
    xpar_gchain_free(&c);
    return XPAR_EXIT_OK;
  }

  xpar_fprintf(xpar_stdout,
               "  set        : %s\n"
               "  format     : %d.%d, layout %s%s\n"
               "  generation : %" PRIu32 " of %" PRIu32 "%s\n",
               idbuf, XPAR_FORMAT_MAJOR, XPAR_FORMAT_MINOR,
               li_layout(sd->layout),
               sd->required_features ? ", with unimplemented required "
                                       "features" : "",
               sd->generation, c.gen_count,
               sel == c.head ? " (the newest)" : "");
  xpar_fprintf(xpar_stdout,
               "  geometry   : Z = %" PRIu64 " (%s), S = %" PRIu64 ", L = %" PRIu64 " (%s)\n"
               "               stream base %" PRIu64 ", %" PRIu32 " entries\n",
               sd->slice_size,
               li_size(sbuf, sizeof sbuf, sd->slice_size),
               sd->data_slice_count,
               sd->stream_length,
               li_size(sbuf2, sizeof sbuf2, sd->stream_length),
               sd->stream_base, sd->file_count);
  if (sd->cell_bytes)
    xpar_fprintf(xpar_stdout,
                 "  cells      : Y = %" PRIu32 " bytes, K = %" PRIu64 " per slice; the "
                 "erasure unit is\n               (slice, column), not a "
                 "whole slice\n", sd->cell_bytes,
                 xpar_ceil_div(sd->slice_size,
                                                    sd->cell_bytes));
  else
    xpar_fprintf(xpar_stdout,
                 "  cells      : none; erasures are whole slices\n");
  if (sd->codec == XPAR_CODEC_FFT_LOW)
    xpar_fprintf(xpar_stdout,
                 "  codec      : %s over GF(2^%" PRIu8 "), data axis 2^%" PRIu8 "; up to "
                 "%" PRIu64 " recovery slices\n", li_codec(sd->codec),
                 sd->field_log2, sd->recovery_axis_log2,
                 xpar_setd_recovery_limit(sd));
  else
    xpar_fprintf(xpar_stdout,
                 "  codec      : %s over GF(2^%" PRIu8 "), recovery axis 2^%" PRIu8 " = "
                 "%" PRIu64 " slices\n", li_codec(sd->codec), sd->field_log2,
                 sd->recovery_axis_log2,
                 xpar_setd_recovery_limit(sd));
  xpar_fprintf(xpar_stdout,
               "  redundancy : R = %" PRIu64 " (%.1f%% of S), %" PRIu64 " recovery "
               "slices present\n",
               c.gen[sel].recovery_top,
               sd->data_slice_count
                 ? 100.0 * (f64) c.gen[sel].recovery_top /
                   (f64) sd->data_slice_count : 0.0,
               c.gen[sel].recovery_count);
  if (c.gen[sel].recovery_count < c.gen[sel].recovery_top)
    xpar_fprintf(xpar_stdout,
                 "               %" PRIu64 " recovery slice%s missing, so this "
                 "generation\n               tolerates that many fewer "
                 "erasures\n",
                 (c.gen[sel].recovery_top -
                                       c.gen[sel].recovery_count),
                 c.gen[sel].recovery_top - c.gen[sel].recovery_count == 1
                   ? " is" : "s are");
  xpar_fprintf(xpar_stdout,
               "  tags       : CRC32C per slice%s%s\n",
               sd->slice_tag_len ? ", BLAKE3 strong tag of " : "",
               sd->slice_tag_len == 16 ? "16 bytes"
                                       : (sd->slice_tag_len ? "8 bytes"
                                                            : ""));
  xpar_fprintf(xpar_stdout, "  dedup      : level %" PRIu8 " (%s)\n",
               sd->dedup_level,
               sd->dedup_level == XPAR_DEDUP_CHUNK ? "chunk" :
               (sd->dedup_level == XPAR_DEDUP_FILE ? "whole entry" : "none"));

  if (have_armour) {
    xpar_armour * a = xpar_armour_new(&ap);
    const char * why = xpar_armour_check(&ap);
    if (why || !a) {
      xpar_fprintf(xpar_stdout, "  armour     : unusable parameters (%s)\n",
                   why ? why : "?");
    } else {
      u32 t = (ap.n - ap.k) / 2;
      xpar_fprintf(xpar_stdout,
                   "  armour     : GF(2^%" PRIu32 ") RS(%" PRIu32 ", %" PRIu32 "), "
                   "t = %" PRIu32 ", D = %" PRIu64
                   "\n               %s; frame %" PRIu64 " bytes on disk carrying "
                   "%" PRIu64 " of plaintext\n"
                   "               correctable burst %" PRIu64 " bytes anywhere in "
                   "a frame\n"
                   "               overhead %.3f%%\n",
                   ap.symbol_bits, ap.n,
                   ap.k, t,
                   ap.depth,
                   whole_file ? "the whole archive is armoured"
                              : "the critical packet group is armoured",
                   xpar_armour_frame_disk(a),
                   xpar_armour_frame_plain(a),
                   xpar_armour_burst(a),
                   100.0 * (f64) (ap.n - ap.k) / (f64) ap.k);
    }
    if (a) xpar_armour_free(a);
  } else {
    xpar_fprintf(xpar_stdout, "  armour     : none\n");
  }

  if (have_layt) {
    xpar_fprintf(xpar_stdout, "  volumes    : %" PRIu32 "\n", layt.count);
    for (i = 0; i < layt.count; i++) {
      const char * kind = layt.vol[i].kind == XPAR_VOL_INDEX ? "index" :
                          (layt.vol[i].kind == XPAR_VOL_DATA ? "data"
                                                             : "recovery");
      char * data_path = NULL;
      u32 v;
      bool present = false;
      for (v = 0; v < c.vol_count; v++)
        if (c.vol[v].gen == sel && c.vol[v].path && layt.vol[i].name) {
          sz pl = xpar_strlen(c.vol[v].path);
          sz nl = xpar_strlen(layt.vol[i].name);
          if (pl >= nl && !xpar_strcmp(c.vol[v].path + pl - nl,
                                       layt.vol[i].name)) present = true;
        }
      if (layt.vol[i].kind == XPAR_VOL_DATA) {
        data_path = li_data_present(&c, &layt.vol[i]);
        present = data_path != NULL;
      }
      if (layt.vol[i].kind == XPAR_VOL_RECOVERY)
        xpar_fprintf(xpar_stdout,
                     "    %-8s %-32s exponents %" PRIu32 "..%" PRIu64 "  %s\n", kind,
                     layt.vol[i].name ? layt.vol[i].name : "?",
                     layt.vol[i].recovery_first,
                     (layt.vol[i].recovery_first +
                                           layt.vol[i].byte_length - 1),
                     present ? "present" : "MISSING");
      else {
        const char * actual = data_path;
        if (actual) {
          const char * p;
          for (p = actual; *p; p++)
            if (*p == '/' || *p == '\\') actual = p + 1;
        }
        if (actual && layt.vol[i].name &&
            xpar_strcmp(actual, layt.vol[i].name))
          xpar_fprintf(xpar_stdout,
                       "    %-8s %-32s present as %s\n", kind,
                       layt.vol[i].name, actual);
        else
          xpar_fprintf(xpar_stdout, "    %-8s %-32s %s\n", kind,
                       layt.vol[i].name ? layt.vol[i].name : "?",
                       present ? "present" : "MISSING");
      }
      xpar_free(data_path);
    }
  }

  for (i = 0; i < c.vol_count; i++)
    if (c.vol[i].gen == sel && c.vol[i].volume_kind == XPAR_VOL_INDEX)
      crit_bytes = c.vol[i].len;
  if (have_layt && crit_bytes) {
    xpar_fprintf(xpar_stdout,
                 "  replication: the critical group is about %s per copy. "
                 "xpar puts a copy\n               in the index volume, in "
                 "volume 0, in every power-of-two\n               volume and "
                 "in the last one; when the group is at most\n"
                 "               max(1 MiB, payload / 20) it goes in every "
                 "volume\n",
                 li_size(sbuf, sizeof sbuf, crit_bytes));
  }

  li_chain_table(&c, sel);

  {
    xpar_plan pl;
    xpar_plan_status st = xpar_plan_for_repair(
      sd, c.gen[sel].recovery_top, o->memory, o->jobs, &pl);
    if (st == XPAR_PLAN_OK) {
      xpar_fprintf(xpar_stdout, "  plan       : to repair this generation\n");
      xpar_plan_print(&pl, xpar_stdout, o->verbose > 0);
    } else {
      xpar_fprintf(xpar_stdout, "  plan       : no feasible repair plan "
                   "under this budget: %s\n", xpar_plan_reason(st));
    }
  }

  if (o->deps) li_deps(&c, sel);

  if (have_layt) xpar_layt_free(&layt);
  xpar_gchain_free(&c);
  return XPAR_EXIT_OK;
}

static void li_recipe(const char * file, u64 hdr, u64 w, u64 n, u64 k,
                      u64 d, u64 frames, u64 stream_off, u64 stream_len,
                      const char * what) {
  u64 fd = d * k * w, fx = d * n * w;
  xpar_fprintf(xpar_stdout,
    "# xpar hand-recovery recipe for %s\n"
    "# %s\n"
    "# Frame f holds its plaintext at the front: the D*k*W plaintext bytes\n"
    "# come first and the D*2t*W parity bytes follow, so the data is never\n"
    "# moved, never reordered and never transformed. That is the whole\n"
    "# guarantee, and it holds at every interleave depth and in both\n"
    "# fields.\n"
    "set -e\n"
    "in=%s\n"
    "out=recovered.bin\n"
    "W=%" PRIu64 "; n=%" PRIu64 "; k=%" PRIu64 "; D=%" PRIu64 "; hdr=%" PRIu64 "\n"
    "Fd=$((D*k*W))          # plaintext bytes per frame = %" PRIu64 "\n"
    "Fx=$((D*n*W))          # disk bytes per frame      = %" PRIu64 "\n"
    "frames=%" PRIu64 "\n"
    "off=%" PRIu64 "               # stream_offset from the prologue\n"
    "len=%" PRIu64 "               # stream_length from the prologue\n"
    "\n"
    "# 1. drop the prologue in one read, so no later step needs to skip it\n"
    "dd if=\"$in\" of=region.bin bs=$hdr skip=1 status=none\n"
    "\n"
    "# 2. take the first Fd bytes of every Fx-byte frame.\n"
    "f=0\n"
    "while [ $f -lt $frames ]; do\n"
    "  dd if=region.bin bs=$Fx skip=$f count=1 status=none | \\\n"
    "    dd bs=$Fd count=1 status=none\n"
    "  f=$((f+1))\n"
    "done > plain.bin\n"
    "\n"
    "# 3. the protected stream is len bytes at off inside that plaintext\n"
    "if [ $off -gt 0 ]; then\n"
    "  dd if=plain.bin bs=$off skip=1 status=none | head -c $len > \"$out\"\n"
    "else\n"
    "  head -c $len plain.bin > \"$out\"\n"
    "fi\n"
    "# end of recipe\n",
    file, what, file,
    w, n, k,
    d, hdr,
    fd, fx,
    frames, stream_off,
    stream_len);
}

int xpar_op_explain(const xpar_options * o) {
  u8 * data;  sz len = 0;
  xpar_arm_prologue pr;
  int which = 0;
  xpar_file * f;
  i64 fsize;
  xpar_json js;

  xpar_json_init(&js, xpar_stdout, o->json);

  f = xpar_open(o->set, XPAR_O_RDONLY);
  if (!f) FATAL_IO("Cannot open '%s': %s.", o->set,
                   xpar_strerror(xpar_errno()));
  fsize = xpar_size(f);
  if (fsize < 0 || (u64) fsize >= (u64) (sz) -1)
    FATAL_IO("Cannot size '%s'.", o->set);
  data = (u8 *) xpar_alloc_raw((sz) fsize + 1);
  if (fsize) xpar_xread(f, data, (sz) fsize);
  xpar_xclose(f);
  len = (sz) fsize;

  if (xpar_garm_prologue(data, len, &pr, &which)) {
    u64 w = pr.symbol_bits / 8;
    u64 fx = pr.depth * pr.n * w;
    u64 frames = fx ? xpar_ceil_div(pr.armoured_length, fx) : 0;
    if (!o->quiet && !o->json) {
      xpar_fprintf(xpar_stdout,
                   "%s is an armoured xpar archive. Prologue copy %d of 3 "
                   "verifies.\n"
                   "It carries every parameter needed to decode the "
                   "region and to find\n"
                   "the protected stream inside it, which is why the recipe "
                   "below needs\n"
                   "nothing but dd:\n\n"
                   "  symbol width W   %" PRIu64 " byte%s (GF(2^%" PRIu8 "))\n"
                   "  code             RS(%" PRIu32 ", %" PRIu32 "), t = %" PRIu32 "\n"
                   "  interleave D     %" PRIu64 "\n"
                   "  frame            %" PRIu64 " bytes on disk, %" PRIu64 " of plaintext\n"
                   "  frames           %" PRIu64 "\n"
                   "  plaintext        %" PRIu64 " bytes\n"
                   "  armoured region  %" PRIu64 " bytes at offset %d\n"
                   "  protected stream %" PRIu64 " bytes at plaintext offset %" PRIu64 "\n\n",
                   o->set, which + 1, w,
                   w == 1 ? "" : "s", pr.symbol_bits,
                   pr.n, pr.k,
                   ((pr.n - pr.k) / 2),
                   pr.depth,
                   fx,
                   (pr.depth * pr.k * w),
                   frames,
                   pr.plain_length,
                   pr.armoured_length,
                   ARM_HDR_EXPLAIN,
                   pr.stream_length,
                   pr.stream_offset);
    }
    if (o->json) {
      xpar_json_begin(&js, "set");
      xpar_json_u64(&js, "schema", XPAR_JSON_SCHEMA);
      xpar_json_str(&js, "layout", "armoured");
      xpar_json_end(&js);
      xpar_json_begin(&js, "recipe");
      xpar_json_str(&js, "source", o->set);
      xpar_json_str(&js, "kind", "protected-stream");
      xpar_json_u64(&js, "header_bytes", ARM_HDR_EXPLAIN);
      xpar_json_u64(&js, "symbol_bytes", w);
      xpar_json_u64(&js, "n", pr.n);  xpar_json_u64(&js, "k", pr.k);
      xpar_json_u64(&js, "depth", pr.depth);
      xpar_json_u64(&js, "frames", frames);
      xpar_json_u64(&js, "stream_offset", pr.stream_offset);
      xpar_json_u64(&js, "stream_length", pr.stream_length);
      xpar_json_end(&js);
      xpar_json_summary(&js, "ok", XPAR_EXIT_OK);
    } else {
      li_recipe(o->set, ARM_HDR_EXPLAIN, w, pr.n, pr.k, pr.depth, frames,
                pr.stream_offset, pr.stream_length,
                "the whole archive is armoured; this extracts the protected "
                "stream");
    }
    xpar_free(data);
    return XPAR_EXIT_OK;
  }

  {
    xpar_scan sc;  xpar_pkt hdr;  const u8 * body;  u64 off;
    bool found = false, packet = false;
    xpar_scan_init(&sc, data, len, NULL, false);
    /*  Framing can be described without authenticating packet content.  */
    sc.accept_unverified_keyed = true;
    while (xpar_scan_next(&sc, &hdr, &body, &off)) {
      xpar_armg ag;
      u64 fx, frames;
      packet = true;
      if (!xpar_pkt_is(&hdr, XPAR_T_ARMG)) continue;
      if (xpar_armg_read(body, (sz) (hdr.length - XPAR_PKT_HDR), &ag) !=
          XPAR_OK) continue;
      fx = ag.depth * ag.n * (ag.symbol_bits / 8);
      frames = fx ? xpar_ceil_div(ag.armoured_length, fx) : 0;
      if (!o->quiet && !o->json)
        xpar_fprintf(xpar_stdout,
                     "%s is a packet-bearing xpar volume.\n\n"
                     "The protected data is not in here: in the sidecar and "
                     "split layouts the\n"
                     "original files are the data, and they are never "
                     "rewritten or armoured.\n"
                     "What is armoured is the critical metadata group, one "
                     "ARMG packet at file\n"
                     "offset %" PRIu64 " whose payload begins at %" PRIu64 ". The recipe "
                     "below recovers that\n"
                     "group as a plain packet stream, which begins with "
                     "\"XPAR2PKT\" and holds\n"
                     "the set descriptor, the manifest and the slice "
                     "checksums.\n\n"
                     "  code             RS(%" PRIu32 ", %" PRIu32 "), t = %" PRIu32 " over GF(2^%" PRIu8 ")"
                     "\n  interleave D     %" PRIu64 "\n"
                     "  frame            %" PRIu64 " bytes on disk, %" PRIu64 " of "
                     "plaintext\n  frames           %" PRIu64 "\n\n",
                     o->set, off,
                     (off + XPAR_PKT_HDR + 48),
                     ag.n, ag.k,
                     ((ag.n - ag.k) / 2), ag.symbol_bits,
                     ag.depth,
                     fx,
                     (ag.depth * ag.k *
                                           (ag.symbol_bits / 8)),
                     frames);
      if (o->json) {
        xpar_json_begin(&js, "set");
        xpar_json_u64(&js, "schema", XPAR_JSON_SCHEMA);
        xpar_json_str(&js, "layout", "packet-bearing");
        xpar_json_end(&js);
        xpar_json_begin(&js, "recipe");
        xpar_json_str(&js, "source", o->set);
        xpar_json_str(&js, "kind", "critical-metadata");
        xpar_json_u64(&js, "header_bytes", off + XPAR_PKT_HDR + 48);
        xpar_json_u64(&js, "symbol_bytes", ag.symbol_bits / 8);
        xpar_json_u64(&js, "n", ag.n);  xpar_json_u64(&js, "k", ag.k);
        xpar_json_u64(&js, "depth", ag.depth);
        xpar_json_u64(&js, "frames", frames);
        xpar_json_u64(&js, "stream_offset", 0);
        xpar_json_u64(&js, "stream_length", ag.plain_length);
        xpar_json_end(&js);
      } else {
        li_recipe(o->set, off + XPAR_PKT_HDR + 48, ag.symbol_bits / 8, ag.n,
                  ag.k, ag.depth, frames, 0, ag.plain_length,
                  "this extracts the armoured critical metadata group");
      }
      found = true;
      break;
    }
    if (!found) {
      if (!packet)
        FATAL_FORMAT("'%s' contains no valid xpar 2.0 packet or armoured "
                     "prologue.", o->set);
      if (o->json) {
        xpar_json_begin(&js, "set");
        xpar_json_u64(&js, "schema", XPAR_JSON_SCHEMA);
        xpar_json_str(&js, "layout", "packet-bearing");
        xpar_json_end(&js);
        xpar_json_begin(&js, "recipe");
        xpar_json_str(&js, "source", o->set);
        xpar_json_str(&js, "kind", "packet-walk");
        xpar_json_u64(&js, "packet_header_bytes", XPAR_PKT_HDR);
        xpar_json_u64(&js, "packet_alignment", XPAR_PKT_ALIGN);
        xpar_json_end(&js);
      } else xpar_fprintf(xpar_stdout,
                   "%s is a packet-bearing xpar volume with no armoured "
                   "region.\n\n"
                   "Every packet in it is framed the same way: the eight "
                   "bytes \"XPAR2PKT\",\n"
                   "a little-endian 64-bit total length at offset 8, the "
                   "set identity at 16,\n"
                   "the four-character type at 32, flags at 36, an 8-byte "
                   "BLAKE3 tag at 40 and\n"
                   "the body from 48. Packets are 8-byte aligned, so the "
                   "whole file can be\n"
                   "walked with nothing but that rule, and any packet can "
                   "be found by\n"
                   "searching for the magic.\n\n"
                   "The protected data is the original files themselves; "
                   "this volume holds\n"
                   "their checksums, their manifest and the recovery "
                   "slices.\n", o->set);
    }
    if (o->json) xpar_json_summary(&js, "ok", XPAR_EXIT_OK);
  }
  xpar_free(data);
  return XPAR_EXIT_OK;
}
