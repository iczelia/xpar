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
