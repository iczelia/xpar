/*  xpar: SSE4.2 CRC-32C variant, three chains and a PCLMULQDQ recombine.

    Copyright (C) 2022-2026 Kamila Szewczyk.  GPLv3-only (see COPYING).  */

#define XPAR_CRC32C_VARIANT_SSE42
#define XPAR_CRC32C_HAVE_HW
#include "crc32c-impl.h"
