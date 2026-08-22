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

#ifndef XPAR_V1DETECT_H
#define XPAR_V1DETECT_H

#include "common.h"

typedef enum {
  XPAR_V1_NONE = 0,   /*  Not a v1 container.  */
  XPAR_V1_JOINT,      /*  'X' 'P' major minor ifactor  */
  XPAR_V1_SHARD_VAN,  /*  "XPAS", Vandermonde sharded  */
  XPAR_V1_SHARD_LEO   /*  "XPAL", Leopard sharded  */
} xpar_v1_kind;

typedef struct {
  xpar_v1_kind kind;
  u8 major, minor;    /*  Meaningful for XPAR_V1_JOINT only.  */
  u8 ifactor;         /*  Interlacing digit '1'..'4'; JOINT only.  */
} xpar_v1_info;

/*  Classify the first `n` bytes of a file. `n` below 8 always yields
    XPAR_V1_NONE, since no v1 header is shorter.  */
xpar_v1_kind xpar_v1_detect(const u8 * head, sz n, xpar_v1_info * out);

/*  Print the refusal of to stderr.  */
void xpar_v1_report(const char * path, const xpar_v1_info * info);

/*  Open `path`, classify it, and if it is v1 report and exit non-zero.  */
void xpar_v1_refuse_if_v1(const char * path);

#endif
