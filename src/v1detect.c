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

#include "v1detect.h"

#define V1_MAJOR_MAX 1
#define V1_MINOR_MAX 9

xpar_v1_kind xpar_v1_detect(const u8 * head, sz n, xpar_v1_info * out) {
  xpar_v1_info info;
  info.kind = XPAR_V1_NONE;  info.major = 0;
  info.minor = 0;            info.ifactor = 0;

  if (n < 8) { if (out) *out = info;  return XPAR_V1_NONE; }

  if (head[0] == 'X' && head[1] == 'P' && head[2] == 'A') {
    if (head[3] == 'S') info.kind = XPAR_V1_SHARD_VAN;
    if (head[3] == 'L') info.kind = XPAR_V1_SHARD_LEO;
  }

  /*  '4' is systematic mode, which v1 encoded in the same digit as the
      interlacing factors '1' to '3'.  */
  if (info.kind == XPAR_V1_NONE && head[0] == 'X' && head[1] == 'P' &&
      head[2] <= V1_MAJOR_MAX && head[3] <= V1_MINOR_MAX &&
      head[4] >= '1' && head[4] <= '4') {
    info.kind    = XPAR_V1_JOINT;
    info.major   = head[2];
    info.minor   = head[3];
    info.ifactor = head[4];
  }

  if (out) *out = info;
  return info.kind;
}

void xpar_v1_report(const char * path, const xpar_v1_info * info) {
  const char * what = "archive";
  switch (info->kind) {
    case XPAR_V1_JOINT:     what = "joint-mode archive";           break;
    case XPAR_V1_SHARD_VAN: what = "Vandermonde-sharded parity file";  break;
    case XPAR_V1_SHARD_LEO: what = "Leopard-sharded parity file";  break;
    default:                return;
  }

  if (info->kind == XPAR_V1_JOINT)
    xpar_fprintf(xpar_stderr, "xpar: '%s' is an xpar %" PRIu8 ".%" PRIu8
                 " %s\n",
                 path, info->major, info->minor, what);
  else
    xpar_fprintf(xpar_stderr, "xpar: '%s' is an xpar 1.x %s\n", path, what);

  xpar_fprintf(xpar_stderr,
    "xpar: decode it with xpar 1.x, then re-protect it\n");
}

void xpar_v1_refuse_if_v1(const char * path) {
  u8 head[8];
  xpar_v1_info info;
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  sz n;

  if (!f) return;
  n = xpar_read(f, head, sizeof head);
  xpar_close(f);

  if (xpar_v1_detect(head, n, &info) == XPAR_V1_NONE) return;
  xpar_v1_report(path, &info);
  xpar_exit(XPAR_EXIT_NOTFOUND);
}
