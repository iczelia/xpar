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

/*  The part of the host interface that is the same on every host: it is
    written against the rest of port.h and never against a system header,
    so a new port implements the primitives and inherits all of this.  */

#include "common.h"

char * xpar_strdup(const char * s) {
  sz n = xpar_strlen(s) + 1;
  char * c = (char *) xpar_alloc_raw(n);
  xpar_memcpy(c, s, n);
  return c;
}

char * xpar_strndup(const char * s, sz n) {
  sz len = 0;
  char * c;
  while (len < n && s[len]) len++;
  c = (char *) xpar_alloc_raw(len + 1);
  xpar_memcpy(c, s, len);
  c[len] = '\0';
  return c;
}

int xpar_parse_u64(const char * s, u64 * out) {
  u64 v = 0;
  if (!s || !*s) return -1;
  if (*s == '+') s++;
  if (!*s) return -1;
  while (*s) {
    u64 nv;
    if (*s < '0' || *s > '9') return -1;
    nv = v * 10 + (u64) (*s - '0');
    if (nv < v) return -1;
    v = nv;  s++;
  }
  *out = v;
  return 0;
}

bool xpar_pread_serial(xpar_read_req * r, sz count) {
  sz i;
  for (i = 0; i < count; i++)
    r[i].result = r[i].file
                    ? xpar_pread(r[i].file, r[i].buf, r[i].length,
                                 r[i].offset) : 0;
  return false;
}

int xpar_snprintf(char * buf, sz cap, const char * fmt, ...) {
  va_list ap;  int r;
  va_start(ap, fmt);
  r = xpar_vsnprintf(buf, cap, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_asprintf(char ** out, const char * fmt, ...) {
  va_list ap, ap2;  int n;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  n = xpar_vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0) { va_end(ap2);  *out = NULL;  return -1; }
  *out = (char *) xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(*out, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  return n;
}

int xpar_fprintf(xpar_file * f, const char * fmt, ...) {
  va_list ap;  int r;
  va_start(ap, fmt);
  r = xpar_vfprintf(f, fmt, ap);
  va_end(ap);
  return r;
}
