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

/*  Stream, slice, and cell geometry interface.

    The stream is cut uniformly at Z independently of entry boundaries.
    Cells are Y-byte columns and record erasure granularity; codec tiling C
    is not part of the format.  Columns with equal erasure patterns share
    decode plans.  */

#ifndef XPAR_SLICE_H
#define XPAR_SLICE_H

#include "manifest.h"
#include "xpar2.h"

/*  Planner floor for Y; smaller cells burden small sets with large tables.  */
#define XPAR_CELL_DEFAULT  65536u

#define XPAR_SLICE_TARGET  4000u

#define XPAR_SLICE_FLOOR   4096u
#define XPAR_SLICE_CEIL    ((u64) 4 << 20)
#define XPAR_SLICE_REFUSE  ((u64) 1 << 30)

typedef struct {
  u64 slice_size;       /*  Z; multiple of 64.  */
  u64 slice_count;      /*  S = ceil(L / Z); 0 in a stream-empty gen.  */
  u64 stream_length;    /*  L, this generation's own bytes.  */
  u64 stream_base;      /*  Its origin in the chain space.  */
  u32 cell_bytes;       /*  Y; 0 means no SLCL and slice erasures.  */
  u32 cells_per_slice;  /*  K = ceil(Z / Y); 1 when Y is 0.  */
} xpar_geom;

typedef struct {
  u64 stream_length;    /*  L.  */
  u64 slice_size;       /*  -s; 0 when unset. Excludes slice_count.  */
  u64 slice_count;      /*  -b; 0 when unset.  */
  u64 recovery;         /*  R where known; 0 when the caller cannot say
                            yet, which is the -r 10% case, and then 1 is
                            assumed so the field bound stays honest.  */
  u64 stream_base;
  u32 cell_bytes;       /*  Y; 0 asks for the planner's default.  */
  u32 armour_frame;     /*  Frame bytes when the sliced stream is itself
                            armoured; 0 otherwise.  */
  u8  field_log2;       /*  8 or 16.  */
} xpar_geom_req;

typedef enum {
  XPAR_GEOM_OK = 0,
  XPAR_GEOM_EXCLUSIVE,  /*  -s and -b together.  */
  XPAR_GEOM_QUANTUM,    /*  Z not a multiple of 64, or below 64.  */
  XPAR_GEOM_HUGE,       /*  Z past refusal point or cap.  */
  XPAR_GEOM_FIELD,      /*  S + R does not fit the field.  */
  XPAR_GEOM_UNREACHABLE /*  -b asks for more slices than there are bytes.  */
} xpar_geom_status;

xpar_geom_status xpar_geom_choose(const xpar_geom_req * req,
                                  xpar_geom * out);
const char * xpar_geom_reason(xpar_geom_status s);

u32 xpar_cell_choose(u64 slice_size, u32 want, u32 armour_frame);

/*  Fill in slice_count, cell_bytes and cells_per_slice from a Z, an L
    and a Y that came off the wire rather than from the planner. Returns
    false when SETD's own constraints do not hold, which a reader treats
    as a malformed packet.  */
bool xpar_geom_from_setd(const xpar_setd * sd, xpar_geom * out);

u64 xpar_slice_begin(const xpar_geom * g, u64 slice);
u64 xpar_slice_of   (const xpar_geom * g, u64 stream_off);

u64 xpar_slice_bytes(const xpar_geom * g, u64 slice);

/*  Compute the SLTG value selected by SETD.required_features. The ordinary
    form is a standard BLAKE3 digest; the aligned form is the chaining value
    of the complete slice subtree at its absolute stream chunk position.  */
void xpar_slice_tag(const xpar_setd * sd, u64 slice, const u8 * bytes,
                    u8 * out, sz n);
void xpar_slice_tag_keyed(const xpar_setd * sd, u64 slice, const u8 * bytes,
                          const u8 * key, u8 * out, sz n);

u32 xpar_cell_of    (const xpar_geom * g, u64 stream_off);
u64 xpar_cell_begin (const xpar_geom * g, u64 slice, u32 col);

/*  Y, or the short remainder in the last column. A function of the
    column alone: a slice is Z bytes for every coding purpose, padding
    included, so only the final column of a slice is short and it is
    short by the same amount in every slice.  */
u64 xpar_cell_size  (const xpar_geom * g, u32 col);

/*  Of those, how many are real content rather than padding.  */
u64 xpar_cell_bytes (const xpar_geom * g, u64 slice, u32 col);

/*  Erasures over cells.  */

void xpar_erasures_init (xpar_erasures * e, u64 slices, u32 cells);
void xpar_erasures_free (xpar_erasures * e);
void xpar_erasures_clear(xpar_erasures * e);

/*  Mark every cell that overlaps a stream byte range. This is how a
    failed entry hash, a failed cell CRC or a short file becomes an
    erasure set.  */
void xpar_erasures_mark_range(xpar_erasures * e, const xpar_geom * g,
                              u64 off, u64 len);
void xpar_erasures_mark_slice(xpar_erasures * e, u64 slice);

u64 xpar_erasures_max_depth(const xpar_erasures * e);

typedef struct {
  u8 *  present;
  u32 * column;
  u32   column_count;
  u64   erased;
} xpar_col_group;

typedef struct {
  xpar_col_group * group;
  u32 group_count;
  u64 slice_count;
  u32 cells_per_slice;
} xpar_col_groups;

void xpar_col_groups_build(const xpar_erasures * e, xpar_col_groups * g);
void xpar_col_groups_free (xpar_col_groups * g);

typedef struct {
  u64 stream_offset;
  u64 file_offset;
  u64 length;
  u32 entry;
} xpar_span;

/*  The canonical occurrence covering `off`, which is where the bytes of
    that stream offset are read from and written to. False past L.  */
bool xpar_stream_locate(const xpar_occindex * ix, u64 off,
                        xpar_span * out);

u32 xpar_slice_spans(const xpar_geom * g, const xpar_occindex * ix,
                     u64 slice, xpar_span * out, u32 max);

#endif
