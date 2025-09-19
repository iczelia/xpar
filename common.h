/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

#ifndef _COMMON_H_
#define _COMMON_H_

#include "platform.h"

#define FATAL(fmt, ...) do { \
  xpar_fprintf(xpar_stderr, fmt "\n", ##__VA_ARGS__); \
  xpar_exit(1); \
} while (0)

#define FATAL_UNLESS(fmt, cond, ...) do { \
  if (!(cond)) { \
    xpar_fprintf(xpar_stderr, fmt "\n", ##__VA_ARGS__); \
    xpar_exit(1); \
  } \
} while (0)

#define FATAL_PERROR(who) do { \
  xpar_fprintf(xpar_stderr, "%s: %s\n", (who), xpar_strerror(xpar_errno())); \
  xpar_exit(1); \
} while (0)

#define xpar_assert(x) do { \
  if (!(x)) FATAL("assertion failed: %s", #x); \
} while (0)

#define Fi(n, ...) for (int i = 0; i < (n); i++) { __VA_ARGS__; }
#define Fj(n, ...) for (int j = 0; j < (n); j++) { __VA_ARGS__; }
#define Fk(n, ...) for (int k = 0; k < (n); k++) { __VA_ARGS__; }
#define Fi0(n, s, ...) for (int i = s; i < (n); i++) { __VA_ARGS__; }
#define Fj0(n, s, ...) for (int j = s; j < (n); j++) { __VA_ARGS__; }
#define Fk0(n, s, ...) for (int k = s; k < (n); k++) { __VA_ARGS__; }

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*  Integrity algorithm identifiers stored verbatim in joint-mode headers.  */
#define INTEGRITY_CRC32C  0
#define INTEGRITY_BLAKE2B 1

/*  Derive output name: POSIX appends suffix; DOS (8.3) replaces extension.  */
static inline char * xpar_derive_name(const char * input, const char * suffix) {
  sz ilen = xpar_strlen(input), slen = xpar_strlen(suffix);
#if defined(XPAR_DOS)
  sz cut = ilen;
  for (sz i = ilen; i > 0; i--) {
    char c = input[i-1];
    if (c == '/' || c == '\\' || c == ':') break;
    if (c == '.') { cut = i - 1; break; }
  }
#else
  sz cut = ilen;
#endif
  char * r = xpar_alloc_raw(cut + slen + 1);
  xpar_memcpy(r, input, cut);
  xpar_memcpy(r + cut, suffix, slen);
  r[cut + slen] = '\0';
  return r;
}

#endif
