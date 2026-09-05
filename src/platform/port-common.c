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

#include <string.h>
#include "common.h"

/*  Shared C-library wrappers.  */
void * xpar_memcpy (void * d, const void * s, sz n) { return memcpy(d, s, n); }
void * xpar_memmove(void * d, const void * s, sz n) { return memmove(d, s, n); }
void * xpar_memset (void * d, int c, sz n)          { return memset(d, c, n); }
int    xpar_memcmp (const void * a, const void * b, sz n) {
  return memcmp(a, b, n);
}
sz     xpar_strlen (const char * s)                 { return strlen(s); }
int    xpar_strcmp (const char * a, const char * b) { return strcmp(a, b); }
int    xpar_strncmp(const char * a, const char * b, sz n) {
  return strncmp(a, b, n);
}

/*  Format on the stack when possible, then use the backend text sink.  */
int xpar_vfprintf(xpar_file * f, const char * fmt, va_list ap) {
  char stack[1024];
  char * big;
  va_list ap2;
  int n;
  va_copy(ap2, ap);
  n = xpar_vsnprintf(stack, sizeof stack, fmt, ap);
  if (n < 0) { va_end(ap2);  return -1; }
  if ((sz) n < sizeof stack) { va_end(ap2);  xpar_port_write_text(f, stack, (sz) n);  return n; }
  big = xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(big, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  xpar_port_write_text(f, big, (sz) n);
  xpar_free(big);
  return n;
}

int xpar_fputs(const char * s, xpar_file * f) {
  sz n = xpar_strlen(s);
  xpar_port_write_text(f, s, n);
  return (int) n;
}

char * xpar_strdup(const char * s) {
  sz n = xpar_strlen(s) + 1;
  char * c = xpar_alloc_raw(n);
  xpar_memcpy(c, s, n);
  return c;
}

char * xpar_strndup(const char * s, sz n) {
  sz len = 0;
  char * c;
  while (len < n && s[len]) len++;
  c = xpar_alloc_raw(len + 1);
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
  Fi(count,
    r[i].result = r[i].file
    ? xpar_pread(r[i].file, r[i].buf, r[i].length,
                 r[i].offset) : 0);
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
  *out = xpar_alloc_raw((sz) n + 1);
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
