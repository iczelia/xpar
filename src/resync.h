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

typedef bool (*xpar_resync_confirm_fn)(void * user, u32 probe,
                                       u64 physical);

/*  Explicit expensive fallback. Every rolling-CRC candidate is offered
    to the strong-tag callback until that probe has been located. The
    caller supplies `located`, initialised here to UINT64_MAX.  */
u64 xpar_resync_exhaustive(xpar_file * f, u64 file_size, u64 window,
                           const xpar_resync_probe * probe, u32 probe_count,
                           u32 step, u64 max_delta,
                           xpar_resync_confirm_fn confirm, void * user,
                           u64 * located);

#endif
