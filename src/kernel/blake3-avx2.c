/*  xpar: AVX2 BLAKE3 compression variant, eight chunks at a time.

    Copyright (C) 2022-2026 Kamila Szewczyk.  GPLv3-only (see COPYING).  */

#define XPAR_BLAKE3_VARIANT_AVX2
#define XPAR_BLAKE3_HAVE_SIMD
#include "blake3-impl.h"
