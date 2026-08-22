/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Outer erasure-code interface for additive FFT and Cauchy matrix codes.  */

#ifndef XPAR_CODEC_H
#define XPAR_CODEC_H

#include "common.h"
#include "xpar2.h"

typedef struct xpar_codec      xpar_codec;
typedef struct xpar_codec_plan xpar_codec_plan;

/*  Status returned by the operations that a caller must distinguish.
    Anything not listed is a programming error and is fatal at the call
    site rather than returned.  */
typedef enum {
  XPAR_CODEC_OK = 0,
  XPAR_CODEC_TOO_MANY_LOST,   /*  More erasures than recovery slices.  */
  XPAR_CODEC_UNSUPPORTED      /*  Parameters this codec cannot express.  */
} xpar_codec_status;

/*  FFT additionally requires R <= S and S + NextPow2(R) <= 2^w.  */
bool xpar_codec_supports(u8 codec, u8 field_log2, u64 s, u64 r);
bool xpar_codec_supports_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                              u8 axis_log2);

/*  Fatal for unsupported parameters; call xpar_codec_supports first.  */
xpar_codec * xpar_codec_new(u8 codec, u8 field_log2, u64 s, u64 r);
xpar_codec * xpar_codec_new_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                 u8 axis_log2);
void         xpar_codec_free(xpar_codec *);

/*  Encode `bytes` bytes of every slice. `data` has S entries and `recovery`
    has R, each pointing at `bytes` bytes. Slices are processed
    independently along the byte axis, so a caller may encode a column at a
    time; `bytes` is the column width, not the slice size.  */
xpar_codec_status xpar_codec_encode(xpar_codec *,
                                    const u8 * const * data,
                                    u8 * const * recovery, sz bytes);

/*  Matrix-only streaming entry point. Accumulate one data slice into a
    contiguous range of recovery rows. `clear` zeroes those accumulators
    first.  */
xpar_codec_status xpar_codec_matrix_accumulate(xpar_codec *, u64 data_index,
                                               const u8 * data,
                                               u64 recovery_first,
                                               u8 * const * recovery,
                                               u64 recovery_count, sz bytes,
                                               bool clear);
xpar_codec_status xpar_codec_matrix_accumulate_many(
  xpar_codec *, u64 data_first, const u8 * const * data, u64 data_count,
  u64 recovery_first, u8 * const * recovery, u64 recovery_count,
  sz bytes, bool clear);

/*  Build a decode plan for one erasure pattern. `data_present[i]` is zero
    when data slice `i` is erased, and likewise `recovery_present[j]`.
    Returns NULL and sets `*status` when the pattern is unrecoverable, which
    a caller must report rather than treat as fatal: it is the ordinary
    outcome of damage past the redundancy.

    A plan is immutable and may be applied to any number of columns
    concurrently, which is what lets one setup serve every column sharing a
    pattern.  */
xpar_codec_plan * xpar_codec_plan_new(xpar_codec *,
                                      const u8 * data_present,
                                      const u8 * recovery_present,
                                      xpar_codec_status * status);
void xpar_codec_plan_free(xpar_codec_plan *);

/*  Reconstruct the erased data slices in place. Entries of `data` and
    `recovery` marked present must hold their bytes; entries marked erased
    are written. Erased *recovery* slices are not reconstructed here; use
    xpar_codec_encode against the repaired data for that, which is what
    `recover` does.  */
xpar_codec_status xpar_codec_plan_apply(const xpar_codec_plan *,
                                        u8 * const * data,
                                        u8 * const * recovery, sz bytes);

/*  Working-set size in bytes for encoding or decoding `bytes` per slice, so
    that plan.c can choose a column width that fits `-m` before committing
    to a codec (DESIGN 12.3). Must not allocate.  */
u64 xpar_codec_encode_footprint(u8 codec, u8 field_log2, u64 s, u64 r,
                                sz bytes);
u64 xpar_codec_decode_footprint(u8 codec, u8 field_log2, u64 s, u64 r,
                                sz bytes);
u64 xpar_codec_encode_footprint_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                     u8 axis_log2, sz bytes);
u64 xpar_codec_decode_footprint_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                     u8 axis_log2, sz bytes);

/*  Arithmetic performed per column, in units of region multiply-accumulate
    bytes, for the plan explanation in `create -v`. Structural: it counts
    butterflies or muladds and multiplies by `bytes`, and does not measure
    anything.  */
u64 xpar_codec_encode_work(u8 codec, u64 s, u64 r, sz bytes);

#endif
