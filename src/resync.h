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

/*  xpar: sliding-window misplaced-data search interface.  */

#ifndef XPAR_RESYNC_H
#define XPAR_RESYNC_H

#include "common.h"
#include "port.h"

/*  One full slice whose canonical occurrence lies wholly in an entry.
    `expected` is its byte offset in that entry, not in the set stream.  */
typedef struct {
  u32 crc;
  u64 expected;
  u64 slice;
} xpar_resync_probe;

#define XPAR_RESYNC_DELTAS 8

typedef struct {
  i64 delta;
  u64 votes;
} xpar_resync_delta;

typedef struct {
  xpar_resync_delta delta[XPAR_RESYNC_DELTAS];
  u32 count;
  u64 candidates;
  bool dominant;
  bool overflow;
} xpar_resync_result;

/*  The rolling and consensus passes.  */
bool xpar_resync_search(xpar_file * f, u64 file_size, u64 window,
                        const xpar_resync_probe * probe, u32 probe_count,
                        u32 step, u64 max_delta, xpar_resync_result * out);

typedef struct { u64 expected, physical; } xpar_resync_loc;

typedef struct {
  xpar_resync_loc * loc;
  u32 count, cap;
  bool searched;
} xpar_resync_map;

void xpar_resync_map_add (xpar_resync_map *, u64 expected, u64 physical);
void xpar_resync_map_free(xpar_resync_map *);

bool xpar_resync_map_shift(const xpar_resync_map *, u64 off, u64 * physical);

/*  `expected` plus a signed displacement, refusing rather than wrapping.  */
bool xpar_resync_shift(u64 expected, i64 delta, u64 * physical);

typedef bool (*xpar_resync_confirm_fn)(void * user, u32 probe,
                                       u64 physical);

u64 xpar_resync_exhaustive(xpar_file * f, u64 file_size, u64 window,
                           const xpar_resync_probe * probe, u32 probe_count,
                           u32 step, u64 max_delta,
                           xpar_resync_confirm_fn confirm, void * user,
                           u64 * located);

/*  Per-entry resync policy and outcome.  */

typedef struct {
  u32  mode;
  u32  step;
  u64  window;
  bool exhaustive;
  bool have_tags;
} xpar_resync_opts;

typedef struct {
  bool engaged;
  bool searched;
  bool need_tags;
  u64  confirmations;
  u64  candidates;
} xpar_resync_outcome;

/*  Fill located with physical offsets or UINT64_MAX; scratch is one slice.  */
void xpar_resync_entry(xpar_file * f, u64 file_size, u64 slice_size,
                       u64 entry_length,
                       const xpar_resync_probe * probe, u32 probe_count,
                       const xpar_resync_opts * o,
                       xpar_resync_confirm_fn confirm, void * user,
                       u8 * scratch, u64 * located,
                       xpar_resync_outcome * out);

#endif
