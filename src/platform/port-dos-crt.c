/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.  */

#if !defined(__DJGPP__)
#error "port-dos-crt.c compiled for a non-DJGPP target"
#endif

#include <stddef.h>
#include <string.h>

static void dos_copy_forward(void * dst, const void * src, size_t n) {
  unsigned char * d = (unsigned char *) dst;
  const unsigned char * s = (const unsigned char *) src;
  __asm__ volatile ("cld\n\trep movsb"
                    : "+D" (d), "+S" (s), "+c" (n)
                    :
                    : "memory");
}

void * memcpy(void * dst, const void * src, size_t n) {
  dos_copy_forward(dst, src, n);
  return dst;
}

void * memmove(void * dst, const void * src, size_t n) {
  unsigned char * d = (unsigned char *) dst;
  const unsigned char * s = (const unsigned char *) src;
  if (!n || d == s) return dst;
  if ((unsigned long) d < (unsigned long) s ||
      (unsigned long) d >= (unsigned long) s + n) {
    dos_copy_forward(d, s, n);
  } else {
    d += n - 1;
    s += n - 1;
    __asm__ volatile ("std\n\trep movsb\n\tcld"
                      : "+D" (d), "+S" (s), "+c" (n)
                      :
                      : "memory");
  }
  return dst;
}

void * memset(void * dst, int c, size_t n) {
  unsigned char * d = (unsigned char *) dst;
  __asm__ volatile ("cld\n\trep stosb"
                    : "+D" (d), "+c" (n)
                    : "a" ((unsigned char) c)
                    : "memory");
  return dst;
}

int memcmp(const void * a, const void * b, size_t n) {
  const unsigned char * ap = (const unsigned char *) a;
  const unsigned char * bp = (const unsigned char *) b;
  if (!n) return 0;
  __asm__ volatile ("cld\n\trepe cmpsb"
                    : "+S" (ap), "+D" (bp), "+c" (n)
                    :
                    : "memory", "cc");
  return (int) ap[-1] - (int) bp[-1];
}

size_t strlen(const char * s) {
  const char * p = s;
  size_t n = (size_t) -1;
  __asm__ volatile ("cld\n\trepne scasb"
                    : "+D" (p), "+c" (n)
                    : "a" (0)
                    : "memory", "cc");
  return ~n - 1;
}

int strcmp(const char * a, const char * b) {
  while (*a && *a == *b) { a++;  b++; }
  return (int) (unsigned char) *a - (int) (unsigned char) *b;
}

int strncmp(const char * a, const char * b, size_t n) {
  while (n && *a && *a == *b) { a++;  b++;  n--; }
  if (!n) return 0;
  return (int) (unsigned char) *a - (int) (unsigned char) *b;
}

char * strcpy(char * dst, const char * src) {
  memcpy(dst, src, strlen(src) + 1);
  return dst;
}

char * strchr(const char * s, int c) {
  const char * p = s;
  size_t n = strlen(s) + 1;
  unsigned char ch = (unsigned char) c;
  __asm__ volatile ("cld\n\trepne scasb"
                    : "+D" (p), "+c" (n)
                    : "a" (ch)
                    : "memory", "cc");
  return p[-1] == (char) ch ? (char *) p - 1 : NULL;
}
