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
