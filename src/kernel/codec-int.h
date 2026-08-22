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

/*  codec-fft.c's entry points, shared with the dispatcher in
    codec-matrix.c. Not part of codec.h, because nothing outside the two
    codec files may reach a codec by name: which one a set uses is read
    from SETD and never chosen at a call site.

    They deal in void * because codec-matrix.c owns the public handle
    types and codec-fft.c does not see them.

    This header exists because the declarations previously sat inside
    codec-matrix.c, so codec-fft.c compiled its definitions with no
    prototype in scope: ten -Wmissing-prototypes warnings, and no compiler
    check that the two files agreed on a single signature.  */

#ifndef XPAR_CODEC_INT_H
#define XPAR_CODEC_INT_H

#include "codec.h"

bool  xpar_fft_supports(u8 kind, u8 field_log2, u64 s, u64 r);
bool  xpar_fft_supports_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                             u8 axis_log2);
void * xpar_fft_new    (u8 kind, u8 field_log2, u64 s, u64 r);
void * xpar_fft_new_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                         u8 axis_log2);
void  xpar_fft_free    (void * self);

xpar_codec_status xpar_fft_encode(void * self, const u8 * const * data,
                                  u8 * const * rec, sz bytes);

void * xpar_fft_plan_new (void * self, const u8 * dpres, const u8 * rpres,
                          xpar_codec_status * status);
void   xpar_fft_plan_free(void * self);

xpar_codec_status xpar_fft_plan_apply(const void * self, u8 * const * data,
                                      u8 * const * rec, sz bytes);

u64 xpar_fft_encode_footprint(u8 kind, u8 field_log2, u64 s, u64 r,
                              sz bytes);
u64 xpar_fft_decode_footprint(u8 kind, u8 field_log2, u64 s, u64 r,
                              sz bytes);
u64 xpar_fft_encode_footprint_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                                   u8 axis_log2, sz bytes);
u64 xpar_fft_decode_footprint_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                                   u8 axis_log2, sz bytes);
u64 xpar_fft_encode_work(u8 kind, u64 s, u64 r, sz bytes);

#endif
