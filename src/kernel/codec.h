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

/*  Outer erasure-code interface for additive FFT and Cauchy matrix codes.  */

#ifndef XPAR_CODEC_H
#define XPAR_CODEC_H

#include "common.h"
#include "xpar2.h"

typedef struct xpar_codec      xpar_codec;
typedef struct xpar_codec_plan xpar_codec_plan;

/*  Recoverable codec outcomes; programming errors are fatal.  */
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

/*  Encode `bytes` bytes from each of S data slices into R recovery slices.  */
xpar_codec_status xpar_codec_encode(xpar_codec *,
                                    const u8 * const * data,
                                    u8 * const * recovery, sz bytes);

/*  Matrix streaming over a contiguous recovery range.  */
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

/*  Build a plan for one erasure pattern; zero marks an erasure.  */
xpar_codec_plan * xpar_codec_plan_new(xpar_codec *,
                                      const u8 * data_present,
                                      const u8 * recovery_present,
                                      xpar_codec_status * status);
void xpar_codec_plan_free(xpar_codec_plan *);

/*  Reconstruct erased data slices in place. Recovery slices are inputs only.  */
xpar_codec_status xpar_codec_plan_apply(const xpar_codec_plan *,
                                        u8 * const * data,
                                        const u8 * const * recovery,
                                        sz bytes);

/*  Allocation-free working-set estimates for `bytes` per slice.  */
u64 xpar_codec_encode_footprint(u8 codec, u8 field_log2, u64 s, u64 r,
                                sz bytes);
u64 xpar_codec_decode_footprint(u8 codec, u8 field_log2, u64 s, u64 r,
                                sz bytes);
u64 xpar_codec_encode_footprint_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                     u8 axis_log2, sz bytes);
u64 xpar_codec_decode_footprint_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                     u8 axis_log2, sz bytes);

/*  Estimated multiply-accumulate bytes per column for plan reporting.  */
u64 xpar_codec_encode_work(u8 codec, u64 s, u64 r, sz bytes);

#endif
