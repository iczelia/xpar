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

/*  xpar: whole volume images, and the armoured frames inside them.  */
#ifndef XPAR_OPS_VOLIMG_H
#define XPAR_OPS_VOLIMG_H

#include "ops.h"

#include "armour.h"
#include "container.h"

/*  A volume read whole: mapped where the host can map it, on the heap
    where it cannot. `data` and `size` are the image either way.  */
typedef struct {
  xpar_mmap  map;
  u8 *       heap;            /*  Set when the host declined to map.  */
  const u8 * data;
  u64        size;
  char *     path;
} xpar_volimg;

bool xpar_volimg_open (xpar_volimg *, const char * path);
void xpar_volimg_close(xpar_volimg *);

typedef void (* xpar_armg_plain_fn)(void * user, u8 * plain, u64 length);
void xpar_armg_unwrap(const u8 * body, u64 length, bool damaged,
                      xpar_armg_plain_fn, void * user);
void xpar_armg_salvage(const u8 * buf, u64 size, const xpar_key *,
                       xpar_armg_plain_fn, void * user);

/*  Choose inner-code parameters for one wrapped packet.  */
void xpar_armour_wrap_params(const xpar_options *, u64 object_bytes,
                             xpar_armour_params *);

/*  Wrap one packet; `_with` reuses the caller's codec.  */
void xpar_armg_wrap_with(xpar_buf * out, const xpar_armour *,
                         const void * plain, sz plain_len,
                         const u8 * set_id, const xpar_key *);
void xpar_armg_wrap(xpar_buf * out, const xpar_options *,
                    const void * plain, sz plain_len,
                    const u8 * set_id, const xpar_key *);

/*  Wrap every packet in a buffer separately.  */
void xpar_armg_wrap_each(xpar_buf * out, const xpar_options *,
                         const u8 * pkts, sz len,
                         const u8 * set_id, const xpar_key *);

typedef struct {
  const xpar_armour * armour;
  xpar_file * file;
  u8 * frame;
  u64 cap, fill;
} xpar_armsink;

void xpar_armsink_init (xpar_armsink *, const xpar_armour *, xpar_file *);
void xpar_armsink_put  (xpar_armsink *, const void *, u64);
void xpar_armsink_flush(xpar_armsink *);
void xpar_armsink_free (xpar_armsink *);

#endif
