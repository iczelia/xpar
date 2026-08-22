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

/*  Internal entry points shared by the two RISC-V vector tiers.  */

#ifndef XPAR_GF_RVV_INT_H
#define XPAR_GF_RVV_INT_H

#include "gf.h"

void xpar_rvv_mac8(u8 *, const u8 *, sz, const xpar_gf8_coef *);
void xpar_rvv_mac8x2(u8 * const [2], const u8 *, sz,
                     const xpar_gf8_coef [2]);
void xpar_rvv_mul8(u8 *, const u8 *, sz, const xpar_gf8_coef *);
void xpar_rvv_xor2(u8 *, const u8 *, sz);
void xpar_rvv_xor3(u8 *, const u8 *, const u8 *, sz);
void xpar_rvv_fft8(u8 *, u8 *, sz, const xpar_gf8_coef *);
void xpar_rvv_ifft8(u8 *, u8 *, sz, const xpar_gf8_coef *);

#endif
