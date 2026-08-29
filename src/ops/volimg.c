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

/*  Whole volume images, and the armoured frames inside them.  */

#include "volimg.h"

#include "gf.h"

bool xpar_volimg_open(xpar_volimg * v, const char * path) {
  xpar_memset(v, 0, sizeof *v);
  v->map = xpar_map(path);
  if (v->map.valid) { v->data = v->map.map;  v->size = v->map.size; }
  else {
    xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
    i64 n;
    if (!f) return false;
    n = xpar_size(f);
    /*  Half the address space, so that a later copy of the image still
        has somewhere to live on a 32-bit host.  */
    if (n < 0 || (u64) n > (u64) (sz) -1 / 2) { xpar_close(f);  return false; }
    v->heap = (u8 *) xpar_alloc_raw((sz) n ? (sz) n : 1);
    if (xpar_read(f, v->heap, (sz) n) != (sz) n) {
      xpar_close(f);  xpar_free(v->heap);  v->heap = NULL;  return false;
    }
    xpar_close(f);
    v->data = v->heap;  v->size = (u64) n;
  }
  v->path = xpar_strdup(path);
  return true;
}

void xpar_volimg_close(xpar_volimg * v) {
  if (v->map.valid) xpar_unmap(&v->map);
  xpar_free(v->heap);
  xpar_free(v->path);
  xpar_memset(v, 0, sizeof *v);
}

void xpar_armg_unwrap(const u8 * body, u64 length, bool damaged,
                      xpar_armg_plain_fn fn, void * user) {
  xpar_armg g;
  xpar_armour_params p;
  xpar_armour * a;
  u8 * plain;
  if (xpar_armg_read(body, (sz) length, &g) != XPAR_OK) return;
  if (g.plain_length > (u64) (sz) -1 / 2) return;
  p.symbol_bits = g.symbol_bits;  p.poly = g.poly;
  p.n = g.n;  p.k = g.k;  p.fcr = g.fcr;  p.prim = g.prim;
  p.depth = g.depth;
  if (xpar_armour_check(&p)) return;
  /*  Empty field tables make every syndrome zero and silently accept
      damage; initialise them before constructing the decoder.  */
  xpar_gf_init();
  a = xpar_armour_new(&p);
  if (!a) return;
  plain = (u8 *) xpar_alloc_raw((sz) g.plain_length ? (sz) g.plain_length : 1);
  xpar_armour_extract(a, plain, g.plain_length, g.data);
  fn(user, plain, g.plain_length);
  if (damaged) {
    u8 * region = (u8 *) xpar_alloc_raw((sz) g.armoured_length ?
                                        (sz) g.armoured_length : 1);
    u8 * fixed  = (u8 *) xpar_alloc_raw((sz) g.plain_length ?
                                        (sz) g.plain_length : 1);
    u64 fd = xpar_armour_frame_disk(a);
    xpar_memcpy(region, g.data, (sz) g.armoured_length);
    if (fd)
      xpar_armour_decode_frames(a, region, g.armoured_length / fd, NULL);
    xpar_armour_extract(a, fixed, g.plain_length, region);
    xpar_free(region);
    fn(user, fixed, g.plain_length);
  }
  xpar_armour_free(a);
}

void xpar_armg_salvage(const u8 * buf, u64 size, const xpar_key * key,
                       xpar_armg_plain_fn fn, void * user) {
  u64 at;
  for (at = 0; at + XPAR_PKT_HDR <= size; at += XPAR_PKT_ALIGN) {
    xpar_pkt h;
    xpar_status st;
    if (xpar_memcmp(buf + at, XPAR_PKT_MAGIC, 8)) continue;
    st = xpar_pkt_read(buf + at, size - at, key, &h);
    if (st != XPAR_E_CHECKSUM && st != XPAR_E_NEEDKEY) continue;
    if (!xpar_pkt_is(&h, XPAR_T_ARMG)) continue;
    xpar_armg_unwrap(buf + at + XPAR_PKT_HDR, h.length - XPAR_PKT_HDR, true,
                     fn, user);
  }
}

static void wrap_solve(const xpar_options * o, u64 object_bytes, u32 sym,
                       xpar_armour_params * p) {
  u32 w, t2, n;
  u64 d = 1;
  xpar_armour_defaults(p, sym);
  w  = sym / 8;
  n  = p->n;
  t2 = p->n - p->k;
  if (o->armour_t) t2 = 2 * o->armour_t;
  else if (o->armour_pct > 0.0)
    t2 = (u32) (o->armour_pct * (f64) n / (100.0 + o->armour_pct) + 0.5);
  /*  Shorten the codeword to the packet.  */
  { int it;
    for (it = 0; it < 8; it++) {
      u64 need = xpar_ceil_div(object_bytes, w) + t2;
      u32 n2 = need < (u64) n ? (u32) need : n;
      u32 t3 = t2;
      if (!o->armour_t && o->armour_pct > 0.0)
        t3 = (u32) (o->armour_pct * (f64) n2 / (100.0 + o->armour_pct) + 0.5);
      if (t3 < 2) t3 = 2;
      t3 &= ~1u;
      if (t3 >= n2) t3 = (n2 - 1) & ~1u;
      if (n2 == n && t3 == t2) break;
      n = n2;  t2 = t3;
    } }
  if (t2 < 2) t2 = 2;
  if (n < t2 + 1) n = t2 + 1;
  if (n < 16) n = 16;
  if (t2 >= n) t2 = (n - 1) & ~1u;
  p->n = n;  p->k = n - t2;
  if (o->depth) d = o->depth;
  else if (o->burst) {
    u64 t = t2 / 2, sym_burst = o->burst / w;
    d = sym_burst == (u64) -1 ? (u64) -1
                              : xpar_ceil_div(sym_burst + 1, t ? t : 1);
  }
  if (!d) d = 1;
  if (d > XPAR_ARMG_DEPTH_MAX) d = XPAR_ARMG_DEPTH_MAX;
  /*  Limit frame padding to the packet size.  */
  while (d > 1 && d * (u64) p->k * w > object_bytes) d /= 2;
  p->depth = d;
}

void xpar_armour_wrap_params(const xpar_options * o, u64 object_bytes,
                             xpar_armour_params * p) {
  xpar_armour_params wide, full;
  u64 wide_bytes;
  if (o->armour_field == 8 || o->armour_field == 16) {
    wrap_solve(o, object_bytes, (u32) o->armour_field, p);
    return;
  }
  /*  Use GF(2^16) only when its padding is negligible.  */
  wrap_solve(o, object_bytes, 8, p);
  wrap_solve(o, object_bytes, 16, &wide);
  xpar_armour_defaults(&full, 16);
  wide_bytes = xpar_armg_length((u8) wide.symbol_bits, wide.n, wide.k,
                                wide.depth, object_bytes);
  if (object_bytes >= (u64) full.k * 2 && wide_bytes &&
      wide_bytes - object_bytes <= object_bytes / 50) *p = wide;
}

void xpar_armg_wrap_with(xpar_buf * out, const xpar_armour * a,
                         const void * plain, sz plain_len,
                         const u8 * set_id, const xpar_key * key) {
  const xpar_armour_params * ap = xpar_armour_params_of(a);
  xpar_armg g;
  u8 * enc;
  xpar_memset(&g, 0, sizeof g);
  g.symbol_bits     = (u8) ap->symbol_bits;
  g.poly            = ap->poly;
  g.n               = ap->n;
  g.k               = ap->k;
  g.fcr             = ap->fcr;
  g.prim            = ap->prim;
  g.depth           = ap->depth;
  g.plain_length    = plain_len;
  g.armoured_length = xpar_armour_size(a, plain_len);
  enc = (u8 *) xpar_calloc((sz) g.armoured_length ? (sz) g.armoured_length
                                                  : 1, 1);
  xpar_armour_encode(a, enc, (const u8 *) plain, plain_len);
  xpar_armg_write(out, &g, enc, set_id, key);
  xpar_free(enc);
}

void xpar_armg_wrap(xpar_buf * out, const xpar_options * o,
                    const void * plain, sz plain_len,
                    const u8 * set_id, const xpar_key * key) {
  xpar_armour_params ap;
  xpar_armour * a;
  const char * why;
  xpar_armour_wrap_params(o, plain_len, &ap);
  why = xpar_armour_check(&ap);
  if (why) FATAL("Invalid armour parameters: %s", why);
  xpar_gf_init();
  a = xpar_armour_new(&ap);
  xpar_armg_wrap_with(out, a, plain, plain_len, set_id, key);
  xpar_armour_free(a);
}

void xpar_armg_wrap_each(xpar_buf * out, const xpar_options * o,
                         const u8 * pkts, sz len, const u8 * set_id,
                         const xpar_key * key) {
  sz at = 0;
  while (at + XPAR_PKT_HDR <= len) {
    u64 n = xpar_rd64(pkts + at + 8);
    FATAL_UNLESS("internal: a packet buffer to be armoured is malformed.",
                 n >= XPAR_PKT_HDR && !(n % XPAR_PKT_ALIGN) &&
                 n <= (u64) (len - at));
    xpar_armg_wrap(out, o, pkts + at, (sz) n, set_id, key);
    at += (sz) n;
  }
  FATAL_UNLESS("internal: a packet buffer to be armoured is malformed.",
               at == len);
}

void xpar_armsink_init(xpar_armsink * s, const xpar_armour * a,
                       xpar_file * f) {
  s->armour = a;  s->file = f;
  s->cap = xpar_armour_frame_plain(a);
  s->frame = (u8 *) xpar_calloc((sz) xpar_armour_frame_disk(a), 1);
  s->fill = 0;
}

void xpar_armsink_flush(xpar_armsink * s) {
  if (!s->fill) return;
  xpar_memset(s->frame + s->fill, 0, (sz) (s->cap - s->fill));
  xpar_armour_encode_frame(s->armour, s->frame);
  xpar_xwrite(s->file, s->frame, (sz) xpar_armour_frame_disk(s->armour));
  s->fill = 0;
}

void xpar_armsink_put(xpar_armsink * s, const void * data, u64 length) {
  const u8 * p = (const u8 *) data;
  while (length) {
    u64 take = MIN(length, s->cap - s->fill);
    xpar_memcpy(s->frame + s->fill, p, (sz) take);
    s->fill += take;  p += take;  length -= take;
    if (s->fill == s->cap) xpar_armsink_flush(s);
  }
}

void xpar_armsink_free(xpar_armsink * s) {
  xpar_free(s->frame);
  xpar_memset(s, 0, sizeof *s);
}
