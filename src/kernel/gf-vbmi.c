/*  xpar: AVX-512 VBMI 6+6+4 GF(2^16) kernel.

    Copyright (C) 2022-2026 Kamila Szewczyk.  GPLv3-only (see COPYING).  */

#define XPAR_GF_VARIANT_VBMI512
#define XPAR_GF_HAVE_SIMD
#include "gf-impl.h"
