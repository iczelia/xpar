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

#if !defined(XPAR_WIN_LEGACY)
#error "port-win95-crt.c requires XPAR_WIN_LEGACY"
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "common.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
  #define XPAR_KEEP __attribute__((used))
#else
  #define XPAR_KEEP
#endif

int __cdecl __mingw_vsnprintf(char *, size_t, const char *, va_list);
void __cdecl xpar_entry(void) XPAR_NORETURN;

XPAR_KEEP void * malloc(size_t n) {
  return HeapAlloc(GetProcessHeap(), 0, n ? n : 1);
}

XPAR_KEEP void * calloc(size_t n, size_t size) {
  if (n && size > (size_t) -1 / n) return NULL;
  return HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                   n && size ? n * size : 1);
}

XPAR_KEEP void * realloc(void * p, size_t n) {
  if (!p) return malloc(n);
  return HeapReAlloc(GetProcessHeap(), 0, p, n ? n : 1);
}

XPAR_KEEP void free(void * p) {
  if (p) HeapFree(GetProcessHeap(), 0, p);
}

#if defined(__i386__) && (defined(__GNUC__) || defined(__clang__))

XPAR_KEEP void * memcpy(void * d, const void * s, size_t n) {
  void * out = d;
  __asm__ volatile ("cld\n\trep movsb"
                    : "+D" (d), "+S" (s), "+c" (n)
                    :
                    : "memory");
  return out;
}

XPAR_KEEP void * memmove(void * d, const void * s, size_t n) {
  void * out = d;
  if (!n) return out;
  if ((const unsigned char *) d < (const unsigned char *) s ||
      (const unsigned char *) d >= (const unsigned char *) s + n) {
    __asm__ volatile ("cld\n\trep movsb"
                      : "+D" (d), "+S" (s), "+c" (n)
                      :
                      : "memory");
  } else {
    d = (unsigned char *) d + n - 1;
    s = (const unsigned char *) s + n - 1;
    __asm__ volatile ("std\n\trep movsb\n\tcld"
                      : "+D" (d), "+S" (s), "+c" (n)
                      :
                      : "memory");
  }
  return out;
}

XPAR_KEEP void * memset(void * d, int c, size_t n) {
  void * out = d;
  __asm__ volatile ("cld\n\trep stosb"
                    : "+D" (d), "+c" (n)
                    : "a" ((unsigned char) c)
                    : "memory");
  return out;
}

XPAR_KEEP int memcmp(const void * a, const void * b, size_t n) {
  const unsigned char * ap = a;
  const unsigned char * bp = b;
  if (!n) return 0;
  __asm__ volatile ("cld\n\trepe cmpsb"
                    : "+S" (ap), "+D" (bp), "+c" (n)
                    :
                    : "memory", "cc");
  return (int) ap[-1] - (int) bp[-1];
}

XPAR_KEEP size_t strlen(const char * s) {
  const char * p = s;
  size_t n = (size_t) -1;
  __asm__ volatile ("cld\n\trepne scasb"
                    : "+D" (p), "+c" (n)
                    : "a" (0)
                    : "memory", "cc");
  return ~n - 1;
}

#else

XPAR_KEEP void * memcpy(void * d, const void * s, size_t n) {
  unsigned char * dp = d;
  const unsigned char * sp = s;
  while (n--) *dp++ = *sp++;
  return d;
}

XPAR_KEEP void * memmove(void * d, const void * s, size_t n) {
  unsigned char * dp = d;
  const unsigned char * sp = s;
  if (dp < sp || dp >= sp + n) {
    while (n--) *dp++ = *sp++;
  } else {
    dp += n;  sp += n;
    while (n--) *--dp = *--sp;
  }
  return d;
}

XPAR_KEEP void * memset(void * d, int c, size_t n) {
  unsigned char * p = d;
  while (n--) *p++ = (unsigned char) c;
  return d;
}

XPAR_KEEP int memcmp(const void * a, const void * b, size_t n) {
  const unsigned char * ap = a;
  const unsigned char * bp = b;
  while (n--) {
    if (*ap != *bp) return (int) *ap - (int) *bp;
    ap++;  bp++;
  }
  return 0;
}

XPAR_KEEP size_t strlen(const char * s) {
  const char * p = s;
  while (*p) p++;
  return (size_t) (p - s);
}

#endif

XPAR_KEEP int strcmp(const char * a, const char * b) {
  while (*a && *a == *b) { a++;  b++; }
  return (int) (unsigned char) *a - (int) (unsigned char) *b;
}

XPAR_KEEP int strncmp(const char * a, const char * b, size_t n) {
  while (n && *a && *a == *b) { a++;  b++;  n--; }
  if (!n) return 0;
  return (int) (unsigned char) *a - (int) (unsigned char) *b;
}

XPAR_KEEP char * strcpy(char * d, const char * s) {
  char * out = d;
  memcpy(d, s, strlen(s) + 1);
  return out;
}

XPAR_KEEP char * strchr(const char * s, int c) {
#if defined(__i386__) && (defined(__GNUC__) || defined(__clang__))
  const char * p = s;
  size_t n = strlen(s) + 1;
  char ch = (char) c;
  __asm__ volatile ("cld\n\trepne scasb"
                    : "+D" (p), "+c" (n)
                    : "a" ((unsigned char) ch)
                    : "memory", "cc");
  return p[-1] == ch ? (char *) p - 1 : NULL;
#else
  char ch = (char) c;
  do {
    if (*s == ch) return (char *) s;
  } while (*s++);
  return NULL;
#endif
}

XPAR_KEEP int __cdecl __mingw_vsnprintf(char * s, size_t n,
                                        const char * fmt, va_list ap) {
  return xpar_vsnprintf(s, n, fmt, ap);
}


#define XPAR_ARGV_CH char
#define XPAR_ARGV_L(c) c
#define XPAR_ARGV_FN split
#include "port-win-argv.h"
#undef XPAR_ARGV_CH
#undef XPAR_ARGV_L
#undef XPAR_ARGV_FN

XPAR_KEEP XPAR_NORETURN void __cdecl xpar_entry(void) {
  char ** argv;
  int argc;
  int rc;
  xpar_host_init();
  xpar_crash_install();
  argc = split(GetCommandLineA(), &argv);
  if (argc < 0) ExitProcess(XPAR_EXIT_NOPLAN);
  rc = xpar_main(argc, argv);
  ExitProcess((UINT) rc);
}
